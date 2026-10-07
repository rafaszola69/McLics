// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#include <iostream>
#include <string>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <math.h>
#include <vector>
#include "../include/anchoring.h"
#include "../include/geometry.h"
#include "../include/parameters.h"
#include "../include/potential.cuh"

#include <map>

// GPU site energies (bonds + anchoring + field) for each geometry, and the anchoring kernels; same definitions as
// the CPU code (geometry*.cpp, anchoring*.cpp). The surface kernel of surface n = pt - 2 returns the energy per unit
// W and is multiplied by d_W[n] here.
__device__ extern float (*bulk_potentialGPU)(float ni[3], float nj[3], Parameters *params, float rij[3]);
__device__ extern p_Surface_Potential *d_Surface_Potential_GPU; 
__device__ extern float *d_W; 
__device__ extern float *d_easy;  // per-surface easy axis (used by rp_ghrl)

__device__ inline float bond_energy_GPU(const nni fullni[], Parameters *params){
  float rij[3];
  double E=0;
  float ni[3]={fullni[0].x,fullni[0].y,fullni[0].z};
  
  if (fullni[1].pt){
    float nj[3]={fullni[1].x,fullni[1].y,fullni[1].z};
    rij[0]=1; rij[1]=0; rij[2]=0;
    E+= bulk_potentialGPU(ni,nj,params,rij);
  }
  if (fullni[2].pt){
    float nj[3]={fullni[2].x,fullni[2].y,fullni[2].z};
    rij[0]=-1; rij[1]=0; rij[2]=0;
    E+= bulk_potentialGPU(ni,nj,params,rij);
  }
  if (fullni[3].pt){
    float nj[3]={fullni[3].x,fullni[3].y,fullni[3].z};
    rij[0]=0; rij[1]=1; rij[2]=0;
    E+= bulk_potentialGPU(ni,nj,params,rij);
  }
  if (fullni[4].pt){
    float nj[3]={fullni[4].x,fullni[4].y,fullni[4].z};
    rij[0]=0; rij[1]=-1; rij[2]=0;
    E+= bulk_potentialGPU(ni,nj,params,rij);
  }
  if (fullni[5].pt){
    float nj[3]={fullni[5].x,fullni[5].y,fullni[5].z};
    rij[0]=0; rij[1]=0; rij[2]=1;
    E+= bulk_potentialGPU(ni,nj,params,rij);
  }
  if (fullni[6].pt){
    float nj[3]={fullni[6].x,fullni[6].y,fullni[6].z};
    rij[0]=0; rij[1]=0; rij[2]=-1;
    E+= bulk_potentialGPU(ni,nj,params,rij);
  }

  return E;
}
__device__ float Bulk_site_energy_GPU(const nni fullni[], Parameters *params){
  float E=0;
  
  float ni[3]={fullni[0].x,fullni[0].y,fullni[0].z};
  
  E=bond_energy_GPU(fullni,params);
  if (params->elecA!=0 || params->elecP!=0)  {
    E+=Electric_Potential_GPU(ni,params)  ;
  }
  return E;
}

__device__ float Slab_site_energy_GPU(const nni fullni[], Parameters *params){
  float E=0;
  float ni[3]={fullni[0].x,fullni[0].y,fullni[0].z};
  
  E=bond_energy_GPU(fullni,params);
  
  float s[3]={fullni[7].x,fullni[7].y,fullni[7].z};
  if (fullni[0].pt>1) {
    E+= d_W[fullni[0].pt-2]*d_Surface_Potential_GPU[fullni[0].pt-2](ni,s,params,&d_easy[3*(fullni[0].pt-2)]);}
  if (params->elecA!=0 || params->elecP!=0)  {
    E+=Electric_Potential_GPU(ni,params)  ;
  }
  
  return E;
}

__device__ float Custom_site_energy_GPU(const nni fullni[], Parameters *params){
  float E=0;
  float ni[3]={fullni[0].x,fullni[0].y,fullni[0].z};
  
  E=bond_energy_GPU(fullni,params);
  float s[3]={fullni[7].x,fullni[7].y,fullni[7].z};

  if (fullni[0].pt>1) 
    E+= d_W[fullni[0].pt-2]*d_Surface_Potential_GPU[fullni[0].pt-2](ni,s,params,&d_easy[3*(fullni[0].pt-2)]);

  if (params->elecA!=0 || params->elecP!=0) 
    E+=Electric_Potential_GPU(ni,params);
      
  return E;
}
 
__device__ float Sphere_site_energy_GPU(const nni fullni[], Parameters *params){
  float E=0;
  float ni[3]={fullni[0].x,fullni[0].y,fullni[0].z};
  
  E=bond_energy_GPU(fullni,params);
  float s[3]={fullni[7].x,fullni[7].y,fullni[7].z};
  if (fullni[0].pt>1) 
    E+= d_W[fullni[0].pt-2]*d_Surface_Potential_GPU[fullni[0].pt-2](ni,s,params,&d_easy[3*(fullni[0].pt-2)]);

  if (params->elecA!=0 || params->elecP!=0) 
    E+=Electric_Potential_GPU(ni,params);
  
  return E;
}

// -(n.v)^2: homeotropic (v = normal), rp and rp_tangent (v = easy axis, stored in d_ns)
__device__ float RP_Surface_Potential_GPU(float ni[3], float s[3], Parameters *params, float rij[3]){
  float nij=ni[0]*s[0]+ni[1]*s[1]+ni[2]*s[2];
  return -nij*nij;
}

// +(n.s)^2: fg
__device__ float FG_Surface_Potential_GPU(float ni[3], float s[3], Parameters *params, float rij[3]){
  
  float nij=ni[0]*s[0]+ni[1]*s[1]+ni[2]*s[2];
  return +nij*nij;
}

// GHRL anchorings: the surface acts as a phantom neighbour located along the outward normal s.
// Same definition as the CPU code. The 4th argument is the surface easy axis (only used by rp_ghrl).
__device__ float RP_Surface_Potential_GHRL_GPU(float ni[3], float s[3], Parameters *params, float e[3]){
  return Bulk_Energy_GHRL_GPU(ni, e, params, s);
}
__device__ float Homeotropic_Surface_Potential_GHRL_GPU(float ni[3], float s[3], Parameters *params, float e[3]){
  return Bulk_Energy_GHRL_GPU(ni, s, params, s);
}
__device__ float FG_Surface_Potential_GHRL_GPU(float ni[3], float s[3], Parameters *params, float e[3]){
  float ns = ni[0]*s[0]+ni[1]*s[1]+ni[2]*s[2];
  float nj[3] = {ni[0]-ns*s[0], ni[1]-ns*s[1], ni[2]-ns*s[2]};
  float mod = sqrtf(nj[0]*nj[0]+nj[1]*nj[1]+nj[2]*nj[2]);
  if (mod < 1e-6f) return 0;
  nj[0] /= mod; nj[1] /= mod; nj[2] /= mod;
  return Bulk_Energy_GHRL_GPU(ni, nj, params, s);
}
