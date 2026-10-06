// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef EVOL_CUH_
#define EVOL_CUH_
#include "../include/potential.h"
#include "../include/geometry.h"
#include <iostream>
#include "../include/parameters.h"
#include "../include/define.h"
#include "../include/evolve.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

// GPU evolution modes (thermalGPU, stepGPU, quenchGPU, electricGPU). The run() loops (evolve_*.cu) mirror the CPU
// ones; the Monte Carlo sweep and the energy are computed on the device (evolve.cu, geometry.cu, potential.cu).
// Device data: d_ni (directors), d_pt (point types), d_params (copy of Parameters), d_T (current temperature),
// d_acc (per-site acceptance flags of the last sweep), d_rngStates (one cuRAND state per 2x2x2 box).
// tick = number of 2x2x2 boxes along x, y, z: in each of the 8 checkerboard passes, one GPU thread updates the
// site with parities (di, dj, dk) of one box. Order parameters and snapshots are computed on the host after
// copying d_ni back.
class EvolveNGPU: public Evolve{
public:

EvolveNGPU(float *ni, int *pt, Parameters *params):ni(ni),pt(pt),params(params),Nx(params->Nx),Ny(params->Ny),Nz(params->Nz){};
virtual int run(){return 0;};
float energy_calculator_GPU();
void Monte_Carlo_Step_GPU(float &ang_var);  // random numbers come from the cuRAND states
__device__ void MC_GPU(dim3 tick, int di, int dj, int dk, float *ni, int *pt, float *d_T, float ang_var, curandState *const rngStates, unsigned int *acceptance, Parameters *params);
protected:
dim3 tick;
curandState *d_rngStates = 0;
int *d_pt=0;
float *d_T;
unsigned int *d_acc=0;
Parameters *d_params=0;
int Nx, Ny, Nz;
float *ni, *d_ni;
int *pt;
Parameters *params;

};
class thermalEvolveNGPU: public EvolveNGPU{
  public:
    thermalEvolveNGPU(float *ni, int *pt, Parameters *params);
    int run();
    ~thermalEvolveNGPU();
};

class stepEvolveNGPU: public EvolveNGPU{
public:
  stepEvolveNGPU(float *ni, int *pt, Parameters *params);
  int run();
  ~stepEvolveNGPU();
};
class quenchEvolveNGPU: public EvolveNGPU{
  public:
    quenchEvolveNGPU(float *ni, int *pt, Parameters *params);
    int run();
    ~quenchEvolveNGPU();
};
class electricEvolveNGPU: public EvolveNGPU{
  public:
    electricEvolveNGPU(float *ni, int *pt, Parameters *params);
    int run();
    ~electricEvolveNGPU();
};
__global__ void initRNG(curandState *const rngStates, const unsigned int seed, dim3 tick);


#endif
