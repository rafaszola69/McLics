// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#include <gsl/gsl_rng.h>
#include <math.h>
#include <strings.h>
#include <omp.h>

#include <iostream>

#include "../include/sweep.h"
#include "../include/define.h"
#include "../include/evolve.h"
#include "../include/geometry.h"
#include "../include/io.h"
#include "../include/parameter_order.h"
#include "../include/parameters.h"
#include "../include/potential.h"

electricEvolveN::electricEvolveN(float *ni, int *ppt, Parameters *params)
    : EvolveN(ni, ppt, params) {
  printf("  Electric evolution\n");
  print_field("Ei:", params->elecEi);
  print_field("Ef:", params->elecEf);
  print_field("dE:", params->elecdE);
  printf("\n");
};

// Field sweep E = Ei, Ei + dE, ..., Ef at the fixed temperature Ti (field direction elecX, elecY, elecZ; couplings
// elecA and elecP). At each E: MCT sweeps of thermalization, then MCS sweeps with one measurement each; one po.dat
// line (first column `field`) and the snapshot director_field_<int(100 E)>.csv (finer steps: see sweep.h).
int electricEvolveN::run() {
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
  fprintf(po_file, "field S varS E varE");
  extra.write_header(po_file);
  fflush(po_file);
  params->T = params->Ti;
  print_field("Threads:", num_threads);
  printf("\n");
  fflush(stdout);
  
  const int snapS = snapshot_scale(params->elecdE);
  printf("  Snapshot names: director_field_<int(%d E)>.csv\n", snapS);
  const int nE = sweep_points(params->elecEi, params->elecEf, params->elecdE);
  for (int ie = 0; ie < nE; ie++) {
    params->elecE = sweep_value(params->elecEi, params->elecEf, params->elecdE, ie, nE);
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
    sprintf(fname, "director_field_%ld.csv", snapshot_index(params->elecE, snapS));
    print_n(fname, ni, *params, pt);
    fprintf(po_file, "%g %g %g %g %g", params->elecE, S1, S2 - S1 * S1, E, (E2 - E * E));
    extra.write_values(po_file, params->MCS);
    fflush(po_file);
  }

  for (int i = 0; i < num_threads; i++)
    gsl_rng_free(rng[i]);

  return 0;
}
