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

thermalEvolveN::thermalEvolveN(float *ni, int *ppt, Parameters *params)
    : EvolveN(ni, ppt, params) {
  printf("  Thermal evolution\n");
  print_field("Ti:", params->Ti);
  print_field("Tf:", params->Tf);
  print_field("dT:", params->dT);
  printf("\n");
};

// Temperature sweep T = Ti, Ti + dT, ..., Tf (include/sweep.h). At each T: MCT sweeps of thermalization, then MCS
// sweeps with one measurement each; writes one po.dat line (averages and variances over the MCS samples) and the
// snapshot director_field_<int(100 T)>.csv (finer steps: see sweep.h). The final state of each T starts the next.
int thermalEvolveN::run() {
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
  fprintf(po_file, "T S varS E varE");
  extra.write_header(po_file);
  fflush(po_file);
  print_field("Threads:", num_threads);
  printf("\n");
  fflush(stdout);
  const int snapS = snapshot_scale(params->dT);
  if (snapS != 100) printf("  Snapshot names: director_field_<int(%d T)>.csv (|dT| < 0.01)\n", snapS);
  const int nT = sweep_points(params->Ti, params->Tf, params->dT);
  for (int it = 0; it < nT; it++) {
    params->T = sweep_value(params->Ti, params->Tf, params->dT, it, nT);
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
    sprintf(fname, "director_field_%ld.csv", snapshot_index(params->T, snapS));
    print_n(fname, ni, *params, pt);
    fprintf(po_file, "%g %g %g %g %g", params->T, S1, S2 - S1 * S1, E, (E2 - E * E));
    extra.write_values(po_file, params->MCS);
    fflush(po_file);
  }

  for (int i = 0; i < num_threads; i++)
    gsl_rng_free(rng[i]);

  return 0;
}
