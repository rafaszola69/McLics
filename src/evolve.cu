// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#include <iostream>
#include <math.h>
#include <omp.h>
#include <string>
#include <cstring>
#include <vector>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "../include/define.h"
#include "../include/evolve.cuh"
#include "../include/geometry.h"
#include "../include/anchoring.h"
#include "../include/io.h"
#include "../include/parameters.h"
#include "../include/parameter_order.h"
#include "../include/potential.cuh"
#include "../include/stringGPU.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "../include/reduce.cuh"
namespace cg = cooperative_groups;
#include <curand_kernel.h>
using std::string;

// GPU Monte Carlo step and energy (see include/evolve.cuh for the data layout).
// Device functions cannot be addressed from the host, so the potential, boundary, geometry and anchoring functions
// are chosen on the device: setGPUup and setSurfaces (one-thread kernels) store pointers to them in the __device__
// variables below, once, in the first call of Monte_Carlo_Step_GPU.

// One cuRAND state per 2x2x2 box (launched with a grid of tick.x * tick.y * tick.z blocks of one thread); the
// state of box b is used by the thread that updates a site of box b in every checkerboard pass.
__global__ void initRNG(curandState *const rngStates, const unsigned int seed, dim3 tick)
{
  // Determine thread ID
  unsigned int tid = blockIdx.x+tick.x*(blockIdx.y+tick.y*blockIdx.z);
  // Initialise the RNG
  curand_init(seed, tid, 0, &rngStates[tid]);
}

// Pointers set on the device by setGPUup / setSurfaces / passGPUns:
//   d_ns       per-site vector passed to the anchoring kernel: the outward normal, or the easy axis for rp and
//              rp_tangent (see Monte_Carlo_Step_GPU); d_W, d_easy: strength and lab easy axis (rp_ghrl) per surface.
__device__ float (*bulk_potentialGPU)(float ni[3], float nj[3], Parameters *params, float rij[3], int nk);
__device__ int (*xBoundary_GPU)(int &ii, int NN);
__device__ int (*yBoundary_GPU)(int &ii, int NN);
__device__ int (*zBoundary_GPU)(int &ii, int NN);
__device__ float *d_ns, *d_phi_s, *d_theta_s, *d_W;
__device__ float *d_easy;
__device__ float (*site_energy_GPU)(const nni fullni[20], Parameters *params);
__device__ p_Surface_Potential *d_Surface_Potential_GPU; 
__device__ void setNni(uint pos, int conditional , nni *nLocal, float *ni, int *pt) {
  if(conditional==1) {
    nLocal->x =ni[pos*3+0];
    nLocal->y =ni[pos*3+1];
    nLocal->z =ni[pos*3+2];
    nLocal->pt=pt[pos];
  }else{
    nLocal->pt=0;
  }
}
// Energy of every site (0 on empty sites) into h_eLat; summed and divided by the number of non-empty sites in
// energy_calculator_GPU, as on the CPU.
__global__ void get_Energy (float *ni, 
                            int *pt, 
                            Parameters *params, 
                            float *h_eLat) {
  int Nx=params->Nx;
  int Ny=params->Ny;
  int Nz=params->Nz;
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if (idx>Nx*Ny*Nz-1) return;
  int i=idx%Nx;
  int j=(idx/Nx)%Ny;
  int k=idx/(Nx*Ny);
  nni nLocal[8];
  
  int im=(i-1); int ip=(i+1);
  int jm=(j-1); int jp=(j+1);
  int km=(k-1); int kp=(k+1);
  int neighbour[6]={
                      xBoundary_GPU(ip,Nx), xBoundary_GPU(im,Nx),
                      yBoundary_GPU(jp,Ny), yBoundary_GPU(jm,Ny),
                      zBoundary_GPU(kp,Nz), zBoundary_GPU(km,Nz)
                    };
  if(pti(i,j,k)==0) {
    h_eLat[idx]=0;
    return;
  }

  setNni(i +Nx*(j +Ny*k ),1,&nLocal[0],ni,pt);
  setNni(ip+Nx*(j +Ny*k ),neighbour[0],&nLocal[1],ni,pt);
  setNni(im+Nx*(j +Ny*k ),neighbour[1],&nLocal[2],ni,pt);
  setNni(i +Nx*(jp+Ny*k ),neighbour[2],&nLocal[3],ni,pt);
  setNni(i +Nx*(jm+Ny*k ),neighbour[3],&nLocal[4],ni,pt);
  setNni(i +Nx*(j +Ny*kp),neighbour[4],&nLocal[5],ni,pt);
  setNni(i +Nx*(j +Ny*km),neighbour[5],&nLocal[6],ni,pt);
  if (nLocal[0].pt>1)
  {
    nLocal[7].x=d_ns[(i+Nx*(j+Ny*k))*3+0];
    nLocal[7].y=d_ns[(i+Nx*(j+Ny*k))*3+1];
    nLocal[7].z=d_ns[(i+Nx*(j+Ny*k))*3+2];  
    nLocal[7].pt=pt[(i+Nx*(j+Ny*k))];
  }
  h_eLat[idx]=site_energy_GPU(nLocal, params);
}
__global__ void callMCGPU(EvolveNGPU * evolve, 
                          dim3 tick, 
                          int di, 
                          int dj, 
                          int dk, 
                          float *d_ni, 
                          int *d_pt, 
                          float *d_T, 
                          float ang_var, 
                          curandState *const d_rngStates, 
                          unsigned int *d_acc, 
                          Parameters *d_params) {
  evolve->MC_GPU(tick, di, dj, dk, d_ni, d_pt, d_T, ang_var, d_rngStates, d_acc, d_params);
}

// Trial move of one site of the sublattice (di, dj, dk): the site (di + 2 iBox, dj + 2 jBox, dk + 2 kBox) of the box
// handled by this thread. Same move and acceptance rule as EvolveN::Monte_Carlo_Step (evolve.cpp); acceptance[]
// receives 1 for an accepted rotation (flips are not counted).
__device__ void EvolveNGPU::MC_GPU (dim3 tick, 
                                    int di, 
                                    int dj, 
                                    int dk, 
                                    float *ni, 
                                    int *pt, 
                                    float *d_T, 
                                    float ang_var, 
                                    curandState *const rngStates, 
                                    unsigned int *acceptance, 
                                    Parameters *params) {
  int Nx=params->Nx;
  int Ny=params->Ny;
  int Nz=params->Nz;
  float fPi=M_PI;
  int Idx=blockIdx.x*blockDim.x+threadIdx.x;
  int iBox=Idx%tick.x;
  int jBox=((Idx)/tick.x)%tick.y;
  int kBox=Idx/(tick.x*tick.y);
  curandState *localState = &rngStates[Idx];
  float T=*d_T;
  nni nLocal[8];
  
    float E_new=0, E_old=0, va, ranVal;
    int i,j,k;
    float nNew[3]; 
    float rotation_type;
    i=di+iBox*2;     if (i>=Nx) return;
    j=dj+jBox*2;     if (j>=Ny) return;
    k=dk+kBox*2;     if (k>=Nz) return;
    int ii=i+Nx*(j+Ny*k);
      acceptance[ii]=0;
    if (pti(i,j,k)==0) {
      return;
    }
    va= (2*curand_uniform(localState)-1)*ang_var; 
    //Create a rotation candidate
    rotation_type=curand_uniform(localState);
    if (rotation_type < 0.333) {
      nNew[0]=nix(i,j,k);
      nNew[1]=niy(i,j,k)*cosf(fPi*va)+niz(i,j,k)*sinf(fPi*va);
      nNew[2]=niz(i,j,k)*cosf(fPi*va)-niy(i,j,k)*sinf(fPi*va);
    }
    else if (rotation_type < 0.666) {
      nNew[0]=nix(i,j,k)*cosf(fPi*va)-niz(i,j,k)*sinf(fPi*va);
      nNew[1]=niy(i,j,k);
      nNew[2]=niz(i,j,k)*cosf(fPi*va)+nix(i,j,k)*sinf(fPi*va);
    }else  {
      nNew[0]=nix(i,j,k)*cosf(fPi*va)+niy(i,j,k)*sinf(fPi*va);
      nNew[1]=niy(i,j,k)*cosf(fPi*va)-nix(i,j,k)*sinf(fPi*va);
      nNew[2]=niz(i,j,k);
    }
    float norm = sqrtf(nNew[0] * nNew[0] + nNew[1] * nNew[1] + nNew[2] * nNew[2]);
    nNew[0] /= norm;
    nNew[1] /= norm;
    nNew[2] /= norm;
    int im=(i-1); int ip=(i+1);
    int jm=(j-1); int jp=(j+1);
    int km=(k-1); int kp=(k+1);
    int neighbour[6]={
                        xBoundary_GPU(ip,Nx), xBoundary_GPU(im,Nx),
                        yBoundary_GPU(jp,Ny), yBoundary_GPU(jm,Ny),
                        zBoundary_GPU(kp,Nz), zBoundary_GPU(km,Nz)
                      };
    setNni(i +Nx*(j +Ny*k ),1,&nLocal[0],ni,pt);
    setNni(ip+Nx*(j +Ny*k ),neighbour[0],&nLocal[1],ni,pt);
    setNni(im+Nx*(j +Ny*k ),neighbour[1],&nLocal[2],ni,pt);
    setNni(i +Nx*(jp+Ny*k ),neighbour[2],&nLocal[3],ni,pt);
    setNni(i +Nx*(jm+Ny*k ),neighbour[3],&nLocal[4],ni,pt);
    setNni(i +Nx*(j +Ny*kp),neighbour[4],&nLocal[5],ni,pt);
    setNni(i +Nx*(j +Ny*km),neighbour[5],&nLocal[6],ni,pt);
    if (nLocal[0].pt > 1)
    {
      nLocal[7].x=d_ns[(i+Nx*(j+Ny*k))*3+0];
      nLocal[7].y=d_ns[(i+Nx*(j+Ny*k))*3+1];
      nLocal[7].z=d_ns[(i+Nx*(j+Ny*k))*3+2];  
      nLocal[7].pt=pt[(i+Nx*(j+Ny*k))];
    }
    E_old=site_energy_GPU(nLocal, params);
    nLocal[0].x = nNew[0];
    nLocal[0].y = nNew[1];
    nLocal[0].z = nNew[2];
    E_new=site_energy_GPU(nLocal, params);
  
    //test new config
    ranVal=curand_uniform(localState);
    float E_cur=E_old;
    // Metropolis; T = 0 is its T -> 0+ limit (accept if the energy does not increase)
    if (T > 0 ? ranVal<=expf(-(E_new-E_old)/T) : E_new<=E_old)
    {
      nix(i,j,k) = nNew[0];
      niy(i,j,k) = nNew[1];
      niz(i,j,k) = nNew[2];
      E_cur=E_new;
      acceptance[ii]=1;
    }else{
      acceptance[ii]=0;
    }
    // Separate n -> -n trial; kept out of `acceptance` so it does not bias the ang_var adaptation.
    if (params->flipMove)
    {
      nLocal[0].x = -nix(i,j,k);
      nLocal[0].y = -niy(i,j,k);
      nLocal[0].z = -niz(i,j,k);
      float E_flip=site_energy_GPU(nLocal, params);
      float uf=curand_uniform(localState);
      if (T > 0 ? uf<=expf(-(E_flip-E_cur)/T) : E_flip<=E_cur)
      {
        nix(i,j,k) = nLocal[0].x;
        niy(i,j,k) = nLocal[0].y;
        niz(i,j,k) = nLocal[0].z;
      }
    }
}

// Selects the bulk potential, boundary functions and site-energy function from the names in params (the host
// has already validated them).
__global__ void setGPUup(Parameters *params, int n_surf) {
    if ( d_strcmp(params->potential,"ll")*d_strcmp(params->potential,"lebwohl-lasher")==0 ) {
     bulk_potentialGPU=&Bulk_Energy_Lebwohl_Lasher_GPU;
   }else if (d_strcmp(params->potential,"ghrl")*d_strcmp(params->potential,"gruhn-hess")==0 ) {
     bulk_potentialGPU=&Bulk_Energy_GHRL_GPU;
   }else if ( d_strcmp(params->potential,"pear")==0 ) {
     bulk_potentialGPU=&Bulk_Energy_Selinger_Pear_GPU;
   }else{
     printf("GPU: potential %s not implemented (options: ll, ghrl, pear)\n",params->potential);
   }
   if(d_strcmp(params->XBoundtype,"free")==0) xBoundary_GPU = &Free_Boundary_GPU;
   else if (d_strcmp(params->XBoundtype,"periodic")==0) xBoundary_GPU = &Periodic_Boundary_GPU;

   if(d_strcmp(params->YBoundtype,"free")==0) yBoundary_GPU = &Free_Boundary_GPU;
   else if (d_strcmp(params->YBoundtype,"periodic")==0) yBoundary_GPU = &Periodic_Boundary_GPU;

   if(d_strcmp(params->ZBoundtype,"free")==0) zBoundary_GPU = &Free_Boundary_GPU;
   else if (d_strcmp(params->ZBoundtype,"periodic")==0) zBoundary_GPU = &Periodic_Boundary_GPU;
   
   if ( d_strcmp(params->geometry,"bulk")==0 )
     site_energy_GPU =  Bulk_site_energy_GPU;
   else if ( d_strcmp(params->geometry,"slab")==0 )
     site_energy_GPU =  Slab_site_energy_GPU;
   else if ( d_strcmp(params->geometry,"sphere")==0 )
     site_energy_GPU =  Sphere_site_energy_GPU;
   else if ( d_strcmp(params->geometry,"custom")==0 )
     site_energy_GPU = Custom_site_energy_GPU;
}
__global__ void passGPUns(float *host, float *h_W, float *h_easy) {
  d_ns=host;
  d_W=h_W;
  d_easy=h_easy;
}
// Kernel for surface ii, selected by the Anchoring name (getName(), include/anchoring.h). LL-form types reduce to
// -(n.v)^2 (RP kernel) or +(n.v)^2 (FG kernel) with v = d_ns of the site: the normal for homeotropic, the easy axis
// for rp and rp_tangent (written into d_ns by Monte_Carlo_Step_GPU). The kernels return the energy per unit W.
__global__ void setSurfaces(p_Surface_Potential *host, int nSurfaces, char *surface_names, int ii) {
  d_Surface_Potential_GPU=host;
       if ( d_strcmp(surface_names,"Homeotropic Anchoring")==0)
          d_Surface_Potential_GPU[ii]=RP_Surface_Potential_GPU;
       else if ( d_strcmp(surface_names,"Fournier-Galatola Anchoring")==0)
          d_Surface_Potential_GPU[ii]=FG_Surface_Potential_GPU;
       else if ( d_strcmp(surface_names,"Rapini-Papoular Anchoring")==0)
          d_Surface_Potential_GPU[ii]=RP_Surface_Potential_GPU;
       else if ( d_strcmp(surface_names,"Rapini-Papoular Tangent Anchoring")==0)
          d_Surface_Potential_GPU[ii]=RP_Surface_Potential_GPU;
       else if ( d_strcmp(surface_names,"Homeotropic Anchoring GHRL")==0)
          d_Surface_Potential_GPU[ii]=Homeotropic_Surface_Potential_GHRL_GPU;
       else if ( d_strcmp(surface_names,"Fournier-Galatola Anchoring GHRL")==0)
          d_Surface_Potential_GPU[ii]=FG_Surface_Potential_GHRL_GPU;
       else if ( d_strcmp(surface_names,"Rapini-Papoular Anchoring GHRL")==0)
          d_Surface_Potential_GPU[ii]=RP_Surface_Potential_GHRL_GPU;
    
}
// Sums n device values; the total ends up at index 0 of the returned buffer
// (either d_data or d_tmp). Each pass reads one buffer and writes the other,
// so blocks never overwrite values another block has yet to read.
// d_tmp must hold at least ceil(n/maxThreads) elements; d_data is clobbered.
template <typename T>
static T *reduce_to_one(T *d_data, T *d_tmp, int n, int maxThreads) {
  T *src=d_data, *dst=d_tmp;
  while (n>1) {
    int threads = n<maxThreads ? n : maxThreads;
    int blocks = (n+threads-1)/threads;
    reduce_sum<<<blocks,threads,threads*sizeof(T)>>>(src,dst,n);
    n=blocks;
    T *swap=src; src=dst; dst=swap;
  }
  return src;
}

// One sweep: 8 kernel launches (one per sublattice), then the accepted rotations are summed on the device and
// ang_var is adapted as on the CPU. The first call also sets up the device functions and surface data.
void EvolveNGPU::Monte_Carlo_Step_GPU(float &ang_var) {
  // The static variables below are set up on the first call and reused: McLics runs one simulation per process.
  float *ns;
  static unsigned int nThread=8;
  static int maxNThread, nValid=0;
  static unsigned int *d_accTmp;
  static int Nt=params->Nx*params->Ny*params->Nz;
  static bool setupGPU=true;
  if (cudaGetLastError()!=cudaSuccess) {
    printf("error before Setup Step\n"); exit(20);
  }
  p_Surface_Potential *Surface_Potential;
  if (setupGPU)
  {
    int nSurfaces=geometry->nSurfaces;
    for (int ii=0; ii<Nt; ii++)
      nValid+=pt[ii]?1:0;
    cudaDeviceGetAttribute(&maxNThread,cudaDevAttrMaxThreadsPerBlock,0);
    for (nThread=128; nThread>1; nThread/=2)
    {
      if(maxNThread > nThread && tick.x*tick.y*tick.z >= nThread) 
        break;
    }
    setGPUup<<<1,1>>>(d_params,nSurfaces);
    
    char *d_surfaceNames;
    cudaMalloc(&Surface_Potential, nSurfaces*sizeof(p_Surface_Potential));
    char *surfaceNames=(char*)malloc(50*sizeof(char));//[nSurfaces][50];
    cudaMalloc(&d_surfaceNames, 50*sizeof(char));
    float *local_W=(float*) malloc(nSurfaces*sizeof(float));
    float *local_easy=(float*) calloc(3*(nSurfaces>0?nSurfaces:1), sizeof(float));
    // Per-site vector sent to the GPU: the surface normal by default, replaced
    // by the easy axis for anchorings whose GPU kernel expects it.
    float *ns_gpu=(float*) malloc(3*Nt*sizeof(float));
    memcpy(ns_gpu, geometry->ns, 3*Nt*sizeof(float));
    for (int ii=0; ii<nSurfaces; ii++)
    {
      local_W[ii]=params->W[ii];
      sprintf(surfaceNames,"%s",geometry->surfaces[ii]->getName());
      cudaMemcpy(d_surfaceNames,surfaceNames, 50*sizeof(char),cudaMemcpyHostToDevice);
      setSurfaces<<<1,1>>>(Surface_Potential,nSurfaces,d_surfaceNames,ii);
      if (params->theta_s.count(ii) && params->phi_s.count(ii))
        Anchoring::lab_axis(params->theta_s[ii], params->phi_s[ii], &local_easy[3*ii]);
      // LL-form rp kernel reads the easy axis from the per-site vector; rp_ghrl keeps the normal there.
      if (strcasecmp(surfaceNames,"Rapini-Papoular Anchoring")==0) 
      {
        for (int i=0; i<params->Nx; i++)
        {
          for (int j=0; j<params->Ny; j++)
          {
            for (int k=0; k<params->Nz; k++)
            {
              if(pti(i,j,k)==ii+2)
              {
                ns_gpu[(i+Nx*(j+Ny*k))*3+0]=sinf(params->theta_s[ii]*M_PI/180)*cosf(params->phi_s[ii]*M_PI/180);
                ns_gpu[(i+Nx*(j+Ny*k))*3+1]=sinf(params->theta_s[ii]*M_PI/180)*sinf(params->phi_s[ii]*M_PI/180);
                ns_gpu[(i+Nx*(j+Ny*k))*3+2]=cosf(params->theta_s[ii]*M_PI/180);
              }
            }
          }
        }
      }
      RP_Tangent_Anchoring *rpt=dynamic_cast<RP_Tangent_Anchoring*>(geometry->surfaces[ii]);
      if (rpt)
      {
        int nSingular=0;
        for (int site=0; site<Nt; site++)
        {
          if (pt[site]!=ii+2) continue;
          if (!rpt->easy_axis(&geometry->ns[site*3], &ns_gpu[site*3])) nSingular++;
        }
        if (nSingular>0)
          printf("rp_tangent surface %d: %d site(s) with normal parallel to the reference axis (no anchoring there).\n", ii, nSingular);
      }
    }
    float *h_W, *h_easy;
    cudaFree(d_surfaceNames);
    cudaMalloc(&h_easy, 3*(nSurfaces>0?nSurfaces:1)*sizeof(float));
    cudaMemcpy(h_easy, local_easy, 3*(nSurfaces>0?nSurfaces:1)*sizeof(float), cudaMemcpyHostToDevice);
    free(local_easy);
    cudaMalloc(&ns, 3*Nt*sizeof(float));
    cudaMalloc(&h_W, nSurfaces*sizeof(float));
      cudaMemcpy(ns,ns_gpu, 3*Nt*sizeof(float),cudaMemcpyHostToDevice);
    free(ns_gpu);
      cudaMemcpy(h_W,local_W, nSurfaces*sizeof(float),cudaMemcpyHostToDevice);
      passGPUns<<<1,1>>>(ns,h_W,h_easy);
    free(local_W); 
    cudaMalloc(&d_accTmp, ((Nt+maxNThread-1)/maxNThread)*sizeof(unsigned int));
    setupGPU=false;
  }
  
  unsigned int acceptance;
  static int thr = nThread;
  static int blk = (tick.x*tick.y*tick.z)/thr+((tick.x*tick.y*tick.z)%thr?1:0);

  if (cudaGetLastError()!=cudaSuccess) {
    printf("error in Setup Step\n"); exit(20);
  }
  acceptance=0;
    for(int di=0; di<(2); di++) {
      for(int dj=0; dj<(2); dj++) {
        for(int dk=0; dk<(2); dk++) {
          callMCGPU<<<blk,thr>>>(this, tick, di, dj, dk, d_ni, d_pt, d_T, ang_var, d_rngStates, d_acc, d_params);
      }
    }
  }
  
  if (cudaGetLastError()!=cudaSuccess) {
    printf("error in MC Step\n"); exit(20);
  }
    unsigned int *d_accSum=reduce_to_one(d_acc, d_accTmp, Nt, maxNThread);
    cudaDeviceSynchronize();
    cudaMemcpy( &acceptance, d_accSum, sizeof(unsigned int), cudaMemcpyDeviceToHost);
   if (1.0*acceptance/nValid<0.5) {
      ang_var*=0.99;
      if(ang_var<0.01) ang_var=0.01;
   }else if (1.0*acceptance/nValid>0.5) {
      ang_var/=0.99;
      if(ang_var > 1.0) {ang_var-=0.5;}
   }
   if (cudaGetLastError()!=cudaSuccess) {
     printf("error in cuda Step\n"); exit(20);
   }
}

float EvolveNGPU::energy_calculator_GPU() {
  // The static variables below are set up on the first call and reused: McLics runs one simulation per process.
  static int nBlocks, nValid=0;
  static float *d_eLat, *d_eTmp;
  static int Nt=params->Nx*params->Ny*params->Nz;
  static bool setUp=true;
  static int maxNThread;
  if(setUp) {
    //Memory Allocation
    cudaMallocManaged(&d_eLat,Nt*sizeof(float));
    for (int ii=0; ii<Nt; ii++)
      nValid+=pt[ii]?1:0;
    cudaDeviceGetAttribute(&maxNThread,cudaDevAttrMaxThreadsPerBlock,0);
    cudaMallocManaged(&d_eTmp,((Nt+maxNThread-1)/maxNThread)*sizeof(float));
    nBlocks=Nt/32+(Nt%32?1:0);
    setUp=false;
  }
   cudaDeviceSynchronize();
   if (cudaGetLastError()!=cudaSuccess) {
     printf("error in cuda, setup\n"); exit(20);
   }
  
   get_Energy<<<nBlocks,32>>>(d_ni, d_pt, d_params, d_eLat);
   cudaDeviceSynchronize();
   
   cudaError_t status=cudaGetLastError();
   if (status!=cudaSuccess) {
     printf("error in cuda, energy %s\n",cudaGetErrorName(status)); exit(20);
   }
   cudaDeviceSynchronize();
   float *d_eSum=reduce_to_one(d_eLat, d_eTmp, Nt, maxNThread);
   if (cudaGetLastError()!=cudaSuccess) {
     printf("error in cuda, reduce\n"); exit(20);
   }
   cudaDeviceSynchronize();
   return d_eSum[0]/nValid;
}
