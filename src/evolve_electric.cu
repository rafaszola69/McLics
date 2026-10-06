// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#include <iostream>
#include <string>
#include <math.h>
#include <strings.h>
#include <gsl/gsl_rng.h>
#include <omp.h>
#include "../include/sweep.h"
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

electricEvolveNGPU::electricEvolveNGPU(float *ni, int *ppt, Parameters *params)
: EvolveNGPU(ni,ppt,params){
  printf("  Electric evolution (GPU)\n");
  print_field("Ei:", params->elecEi);
  print_field("Ef:", params->elecEf);
  print_field("dE:", params->elecdE);
  printf("\n");

}; 

// Field sweep E = Ei, Ei + dE, ..., Ef at the fixed temperature Ti (field direction elecX, elecY, elecZ; couplings
// elecA and elecP). At each E: MCT sweeps of thermalization, then MCS sweeps with one measurement each; one po.dat
// line (first column `field`) and the snapshot director_field_<int(100 E)>.csv (finer steps: see sweep.h).
// GPU version of src/evolve_electric.cpp (same loop; the sweeps and energies run on the device).
int electricEvolveNGPU::run(){
  float Nt=Nx*Ny*Nz;
  double S1, S2;
  float sTemp;
  float vec_n[3] ;
  float mat_n[9] ;
  float ang_var=0.5;
  char fname[1000];
  float tempE;
  double E2, E;
  ExtraObservables extra(params);
  tick={(unsigned int)Nx/2+Nx%2,(unsigned int) Ny/2+Ny%2,(unsigned int) Nz/2+Nz%2};
  cudaError_t cudaResult = cudaSuccess;
  cudaResult = cudaMalloc(&d_rngStates, tick.x*tick.y*tick.z * sizeof(curandState));
  if (cudaResult != cudaSuccess) {
    string msg("Could not allocate memory on device for RNG states: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }
  params->T = params->Ti;
    
  cudaMalloc((void **)&d_ni, 3*Nt*sizeof(float));
  cudaMalloc((void **)&d_pt, Nt*sizeof(int));
  cudaMalloc((void **)&d_T, sizeof(float));
  cudaMalloc((void **)&d_params, sizeof(Parameters));
  cudaMalloc((void **)&d_acc, Nt *sizeof(unsigned int));  
  
  cudaMemcpy(d_ni, ni, 3*Nt*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pt, pt, Nt*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_params, params, sizeof(Parameters), cudaMemcpyHostToDevice);
  initRNG<<<tick, 1>>>(d_rngStates, params->seed, tick);   // was hard-coded 1: all GPU runs used the same stream
      
  
  sprintf(fname,"po.dat");
  FILE *po_file=fopen(fname,"a");
  fprintf(po_file, "field S varS E varE");
  extra.write_header(po_file);
  fflush(po_file);
  fflush(stdout);
  cudaMemcpy(d_T, &params->T, sizeof(float), cudaMemcpyHostToDevice);
    
  const int snapS = snapshot_scale(params->elecdE);
  printf("  Snapshot names: director_field_<int(%d E)>.csv\n", snapS);
  const int nE = sweep_points(params->elecEi, params->elecEf, params->elecdE);
  for (int ie = 0; ie < nE; ie++) {
    params->elecE = sweep_value(params->elecEi, params->elecEf, params->elecdE, ie, nE);
    cudaMemcpy(d_params, params, sizeof(Parameters), cudaMemcpyHostToDevice);
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
    cudaMemcpy( ni, d_ni, 3*Nt*sizeof(float), cudaMemcpyDeviceToHost);
    S1/=params->MCS;
    S2/=params->MCS;
    sprintf(fname, "director_field_%ld.csv", snapshot_index(params->elecE, snapS));
    print_n(fname,ni,*params,pt);
    fprintf(po_file, "%g %g %g %g %g", params->elecE,S1, S2-S1*S1, E, (E2-E*E));
    extra.write_values(po_file, params->MCS);
    fflush(po_file);
  }
  
  fclose(po_file);
    
  return 0;
}
