// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef _IC_H_
#define _IC_H_
#include <gsl/gsl_eigen.h>

#include <iostream>

#include "../include/define.h"
#include "../include/parameter_order.h"
#include "../include/parameters.h"
void random_ic(float *ni, int *pt, Parameters params);
void homogeneous_ic(float *ni, int *pt, Parameters params);
void cholesteric_ic(float *ni, int *pt, Parameters params);
void read_ic_file(float *ni, int *pt, Parameters params);
void apply_Initial_Conditions(float *ni, int *pt, Parameters params);

#endif
