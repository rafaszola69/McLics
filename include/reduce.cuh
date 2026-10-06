// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
// Block-wise sums used by reduce_to_one (evolve.cu): each block writes the sum of its elements to g_odata[block].
__global__ void reduce_sum(uint *g_idata, uint *g_odata, int max);
__global__ void reduce_sum(float *g_idata, float *g_odata, int max);
