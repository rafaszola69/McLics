// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#include <gsl/gsl_rng.h>

#include <iostream>

#include "../include/define.h"
#include "../include/evolve.h"
#include "../include/io.h"
#include "../include/parameter_order.h"
#include "../include/parameters.h"
#include "../include/potential.h"

class simulator {
 public:
  simulator(Parameters *params);
  Parameters *params;
  void Setup_simulation(Parameters &params);
  int print_n(char *fname, Parameters *params);
  int Nx, Ny, Nz;
  Evolve *evolve;
  int *pt;
  ~simulator();

 private:
  float *ni;
};
