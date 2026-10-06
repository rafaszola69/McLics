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

thermalEvolveNGPU::thermalEvolveNGPU(float *ni, int *ppt, Parameters *params)
: EvolveNGPU(ni,ppt,params){
  printf("  Thermal evolution (GPU)\n");
  print_field("Ti:", params->Ti);
  print_field("Tf:", params->Tf);
  print_field("dT:", params->dT);
  printf("\n");

}; 

// Temperature sweep T = Ti, Ti + dT, ..., Tf (include/sweep.h). At each T: MCT sweeps of thermalization, then MCS
// sweeps with one measurement each; writes one po.dat line (averages and variances over the MCS samples) and the
// snapshot director_field_<int(100 T)>.csv (finer steps: see sweep.h). The final state of each T starts the next.
// GPU version of src/evolve_thermal.cpp (same loop; the sweeps and energies run on the device).
int thermalEvolveNGPU::run(){
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
  fprintf(po_file, "T S varS E varE");
  extra.write_header(po_file);
  fflush(po_file);
  fflush(stdout);
  const int snapS = snapshot_scale(params->dT);
  if (snapS != 100) printf("  Snapshot names: director_field_<int(%d T)>.csv (|dT| < 0.01)\n", snapS);
  const int nT = sweep_points(params->Ti, params->Tf, params->dT);
  for (int it = 0; it < nT; it++) {
    params->T = sweep_value(params->Ti, params->Tf, params->dT, it, nT);
    cudaMemcpy(d_T, &params->T, sizeof(float), cudaMemcpyHostToDevice);
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
    sprintf(fname, "director_field_%ld.csv", snapshot_index(params->T, snapS));
    print_n(fname,ni,*params,pt);
    fprintf(po_file, "%g %g %g %g %g", params->T,S1, S2-S1*S1, E, (E2-E*E));
    extra.write_values(po_file, params->MCS);
    fflush(po_file);
  }
  
  fclose(po_file);
    
  return 0;
}
