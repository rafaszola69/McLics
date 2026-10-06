// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef POTENTIAL_CUH_
#define POTENTIAL_CUH_
#include "../include/potential.h"
#include <iostream>
#include "../include/parameters.h"
#include "../include/define.h"

__device__ float Bulk_Energy_Selinger_Pear_GPU(float ni[3], float nj[3], Parameters *params, float rij[3], int nk =1);
__device__ float Bulk_Energy_Lebwohl_Lasher_GPU(float ni[3], float nj[3], Parameters *params, float rij[3], int nk =1);
__device__ float Bulk_Energy_GHRL_GPU(float ni[3], float nj[3], Parameters *params, float rij[3], int nk =1);
__device__ int Periodic_Boundary_GPU(int &ii, int NN);
__device__ int Free_Boundary_GPU(int &ii, int NN);

// Energy of one site in each geometry (geometry.cu): bonds with its neighbours + anchoring + field.
__device__ float Slab_site_energy_GPU  (const nni fullni[8], Parameters *params);
__device__ float Bulk_site_energy_GPU  (const nni fullni[8], Parameters *params);
__device__ float Custom_site_energy_GPU(const nni fullni[8], Parameters *params);
__device__ float Sphere_site_energy_GPU(const nni fullni[8], Parameters *params);

// Anchoring kernels (geometry.cu), per unit W. See setSurfaces in evolve.cu for which kernel each type uses.
__device__ float FG_Surface_Potential_GPU(float ni[3], float s[3], Parameters *params, float rij[3]);
__device__ float RP_Surface_Potential_GPU(float ni[3], float s[3], Parameters *params, float rij[3]);

__device__ float FG_Surface_Potential_GHRL_GPU(float ni[3], float s[3], Parameters *params, float rij[3]);
__device__ float RP_Surface_Potential_GHRL_GPU(float ni[3], float s[3], Parameters *params, float rij[3]);
__device__ float Homeotropic_Surface_Potential_GHRL_GPU(float ni[3], float s[3], Parameters *params, float rij[3]);
__global__ void  setGPUup(Parameters *params, int n_surf);

__device__ float Electric_Potential_GPU(float ni[3], Parameters *params);
#endif
