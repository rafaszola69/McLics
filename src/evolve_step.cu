// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#include <iostream>
#include <string>
#include <math.h>
#include <strings.h>
#include <gsl/gsl_rng.h>
#include <omp.h>
#include "../include/define.h"
#include "../include/evolve.h"
#include "../include/evolve.cuh"
#include "../include/geometry.h"
#include "../include/io.h"
#include "../include/parameters.h"
#include "../include/parameter_order.h"
#include "../include/potential.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
using std::string;

stepEvolveNGPU::stepEvolveNGPU(float *ni, int *ppt, Parameters *params)
: EvolveNGPU(ni,ppt,params){
  printf("  Step evolution (GPU)\n");
  print_field("First file:", params->first_file);
  print_field("Last file:", params->first_file + params->fn);
  printf("\n");
}; 

// Time evolution at the fixed temperature Ti: fn blocks numbered first_file, first_file+1, ...; each block has MCT
// sweeps without measurement and MCS sweeps with one measurement each, then one po.dat line (first column = block
// number) and the snapshot director_field_<block>.csv.
// GPU version of src/evolve_step.cpp (same loop; the sweeps and energies run on the device).
int stepEvolveNGPU::run(){
  
  float Nt=Nx*Ny*Nz;
  double S1, S2;
  float sTemp;
  float tempE;
  double E2, E;
  ExtraObservables extra(params);
  float vec_n[3] ;
  float mat_n[9] ;
  float ang_var=0.5;
  char fname[1000];
  
  tick={(unsigned int)Nx/2+Nx%2,(unsigned int) Ny/2+Ny%2,(unsigned int) Nz/2+Nz%2};

  cudaError_t cudaResult = cudaSuccess;
  
  cudaResult = cudaMalloc(&d_rngStates, tick.x*tick.y*tick.z * sizeof(curandState));
  if (cudaResult != cudaSuccess) {
    string msg("Could not allocate memory on device for RNG states: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }
    
  cudaMalloc((void **)&d_ni, 3*Nt*sizeof(float));
  cudaMalloc((void **)&d_pt, Nt*sizeof(int));
  cudaMalloc((void **)&d_T, sizeof(float));
  cudaMalloc((void **)&d_params, sizeof(Parameters));
  cudaMalloc((void **)&d_acc, Nt*sizeof(unsigned int));  
  
  cudaMemcpy(d_ni, ni, 3*Nt*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pt, pt, Nt*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_params, params, sizeof(Parameters), cudaMemcpyHostToDevice);
  initRNG<<<tick, 1>>>(d_rngStates, params->seed, tick);   // was hard-coded 1: all GPU runs used the same stream
  cudaMemcpy(d_T, &params->T, sizeof(float), cudaMemcpyHostToDevice);
    
  
  sprintf(fname,"po.dat");
  FILE *po_file=fopen(fname,"a");
  fprintf(po_file, "ii S varS E varE");
  extra.write_header(po_file);
  fflush(po_file);
  params->T=params->Ti; cudaMemcpy(d_T, &params->T, sizeof(float), cudaMemcpyHostToDevice);
  fflush(stdout);
  for (int ii=params->first_file; ii< params->fn+params->first_file; ii++){
    
    for(int step=0; step<params->MCT; step++){
      Monte_Carlo_Step_GPU(ang_var);
    }
    S1=0;S2=0;E=0;E2=0;extra.reset();
    for(int step=0; step<params->MCS; step++){
      Monte_Carlo_Step_GPU(ang_var);
      tempE=energy_calculator_GPU();
      E2+=tempE*tempE;
      E +=tempE;
      cudaMemcpy( ni, d_ni, 3*Nt*sizeof(float), cudaMemcpyDeviceToHost);
      Q_tensor(ni,mat_n,pt,*params);
      sTemp=Eigen_value_evaluation(mat_n,vec_n);
      S1+=sTemp;
      S2+=sTemp*sTemp;
      extra.accumulate(ni, pt, vec_n);
    }
    E/=params->MCS;
    E2/=params->MCS;
    S1/=params->MCS;
    S2/=params->MCS;
    cudaMemcpy( ni, d_ni, 3*Nt*sizeof(float), cudaMemcpyDeviceToHost);
    sprintf(fname,"director_field_%d.csv",ii);
    print_n(fname,ni,*params,pt);
    fprintf(po_file, "%d %g %g %g %g", ii,S1, S2-S1*S1, E, (E2-E*E));
    extra.write_values(po_file, params->MCS);
    fflush(po_file);
  }
  fclose(po_file);
  return 0;
}
