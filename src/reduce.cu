// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
// Parallel sums on the GPU (energy and acceptance).
#include<iostream>

// Every thread must reach every __syncthreads(): no early returns. Threads past
// `max` contribute 0; the block is first folded onto the largest power of two
// <= blockDim.x, then reduced as a tree.
template <typename T>
__device__ void block_reduce_sum(T *g_idata, T *g_odata, int max, T *sdata) {
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  sdata[tid] = (i < (unsigned)max) ? g_idata[i] : 0;
  __syncthreads();

  int nOff, nInt;
  for (unsigned int s=1024; s>0; s>>=1){
    if (s<=blockDim.x){ nOff=blockDim.x-s; nInt=s; break; }
  }
  if (tid < nOff) sdata[tid] += sdata[tid+nInt];
  __syncthreads();

  for (unsigned int s=nInt/2; s>0; s>>=1) {
    if (tid < s) sdata[tid] += sdata[tid+s];
    __syncthreads();
  }
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce_sum(uint *g_idata, uint *g_odata, int max) {
  extern __shared__ uint sdata[];
  block_reduce_sum(g_idata, g_odata, max, sdata);
}
__global__ void reduce_sum(float *g_idata, float *g_odata, int max) {
  extern __shared__ float fdata[];
  block_reduce_sum(g_idata, g_odata, max, fdata);
}
