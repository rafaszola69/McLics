// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef STRING_CUH_
#define STRING_CUH_
// Case-insensitive string comparison on the device (like strcasecmp: 0 when equal), used to select device functions.
__device__ int d_tolower(int ch);
__device__ int d_strcmp(const char *str_a, const char *str_b, unsigned len = 256);
#endif
