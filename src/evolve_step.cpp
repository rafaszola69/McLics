// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#include <gsl/gsl_rng.h>
#include <math.h>
#include <strings.h>
#include <omp.h>

#include <iostream>

#include "../include/define.h"
#include "../include/evolve.h"
#include "../include/geometry.h"
#include "../include/io.h"
#include "../include/parameter_order.h"
#include "../include/parameters.h"
#include "../include/potential.h"

stepEvolveN::stepEvolveN(float *ni, int *ppt, Parameters *params)
    : EvolveN(ni, ppt, params) {
  printf("  Step evolution\n");
  print_field("First file:", params->first_file);
  print_field("Last file:", params->first_file + params->fn);
  printf("\n");
};

// Time evolution at the fixed temperature Ti: fn blocks numbered first_file, first_file+1, ...; each block has MCT
// sweeps without measurement and MCS sweeps with one measurement each, then one po.dat line (first column = block
// number) and the snapshot director_field_<block>.csv.
int stepEvolveN::run() {
  double S1, S2;
  float sTemp;
  float tempE;
  double E2, E;
  ExtraObservables extra(params);
  float vec_n[3];
  float mat_n[9];
  float ang_var = 0.5;
  char fname[1000];

  int num_threads = omp_get_max_threads();
  gsl_rng **rng = (gsl_rng **)calloc(num_threads, sizeof(gsl_rng *));
  gsl_rng_env_setup();
  // One generator per OpenMP thread. Which thread updates which site depends on the number of threads, so CPU
  // trajectories (not their averages) change with OMP_NUM_THREADS; for identical runs, fix OMP_NUM_THREADS.
  for (int i = 0; i < num_threads; i++) {
    rng[i] = gsl_rng_alloc(gsl_rng_ranlxs0);
    gsl_rng_set(rng[i], params->seed * 104729 + i);
  }

  sprintf(fname, "po.dat");
  FILE *po_file = fopen(fname, "a");
  fprintf(po_file, "ii S varS E varE");
  extra.write_header(po_file);
  fflush(po_file);
  params->T = params->Ti;
  print_field("Threads:", num_threads);
  printf("\n");
  fflush(stdout);
  for (int ii = params->first_file; ii < params->fn + params->first_file; ii++) {
    for (int step = 0; step < params->MCT; step++) {
      Monte_Carlo_Step(ang_var, rng);
    }
    S1 = 0;
    extra.reset();
    S2 = 0;
    E = 0;
    E2 = 0;
    for (int step = 0; step < params->MCS; step++) {
      Monte_Carlo_Step(ang_var, rng);
      tempE = energy_calculator();
      E2 += tempE * tempE;
      E += tempE;
      Q_tensor(ni, mat_n, pt, *params);
      sTemp = Eigen_value_evaluation(mat_n, vec_n);
      S1 += sTemp;
      S2 += sTemp * sTemp;
      extra.accumulate(ni, pt, vec_n);
    }
    E /= params->MCS;
    E2 /= params->MCS;
    S1 /= params->MCS;
    S2 /= params->MCS;
    sprintf(fname, "director_field_%d.csv", ii);
    print_n(fname, ni, *params, pt);
    fprintf(po_file, "%d %g %g %g %g", ii, S1, S2 - S1 * S1, E, (E2 - E * E));
    extra.write_values(po_file, params->MCS);
    fflush(po_file);
  }

  for (int i = 0; i < num_threads; i++) gsl_rng_free(rng[i]);
  return 0;
}
