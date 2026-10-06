// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General
// Public License as published by the Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
// more details.
//
// You should have received a copy of the GNU General Public License along with this program (file LICENSE). If
// not, see <https://www.gnu.org/licenses/>.
#include <dirent.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <math.h>
#include <string.h>

#include <iostream>

#include "../include/define.h"
#include "../include/evolve.h"
#include "../include/io.h"
#include "../include/parameter_order.h"
#include "../include/parameters.h"
#include "../include/potential.h"
#include "../include/simulator.h"

// McLics: Monte Carlo simulation of nematic and polar liquid crystals on a cubic lattice (CPU with OpenMP, or GPU
// with CUDA). Usage: mclics input_file. Outputs in the working directory: po.dat (order parameters and energy at
// each temperature, field or block), ic.csv (initial condition) and director_field_*.csv (snapshots).

#define MCLICS_VERSION "0.1"

// ─── Helpers ────────────────────────────────────────────────────────────────

static void print_banner() {
  print_header("McLics - Monte Carlo Liquid Crystal Simulator (v" MCLICS_VERSION ")", 60, "=");
  printf("  Copyright (C) 2026 The McLics authors. Free software under the GNU GPL v3\n"
         "  or later, WITHOUT ANY WARRANTY; see the file LICENSE.\n\n");
}

// ─── cleanPreviousOutput ────────────────────────────────────────────────────
// Removes leftover output files (ic.csv, po.dat, director_field_*.csv) from a
// previous run so results never get appended to old data. It is called only after
// the whole setup, so an initial-condition file that is a snapshot of the previous
// run has already been read when it is deleted.

static bool has_suffix(const char *name, const char *suffix) {
  size_t nameLen = strlen(name);
  size_t sufLen = strlen(suffix);
  if (sufLen > nameLen) return false;
  return strcmp(name + (nameLen - sufLen), suffix) == 0;
}

static bool has_prefix(const char *name, const char *prefix) {
  return strncmp(name, prefix, strlen(prefix)) == 0;
}

static void cleanPreviousOutput() {
  printf("  Removing output files from previous run... ");
  fflush(stdout);
  int removed = 0;
  DIR *dir = opendir(".");
  if (dir) {
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
      const char *name = entry->d_name;
      if (strcmp(name, "ic.csv") == 0 || strcmp(name, "po.dat") == 0 ||
          (has_prefix(name, "director_field_") && has_suffix(name, ".csv"))) {
        remove(name);
        removed++;
      }
    }
    closedir(dir);
  }
  printf("%d file(s) removed.\n\n", removed);
}

int main(int argc, char **argv) {
  print_banner();

  Parameters params = read_input_file(argc > 1 ? argv[1] : NULL);
  // Polar models need n -> -n moves: small rotations cannot cross the nematic barrier.
  params.flipMove = (strcasecmp(params.potential, "pear") == 0 || params.elecP != 0) ? 1 : 0;
  if (params.MCQ < 0) params.MCQ = params.MCT;  // quench: by default, thermalize at Tf as long as at Ti
  if (params.radial[0] == '\0')
    strcpy(params.radial, strcasecmp(params.geometry, "sphere") == 0 ? "sphere" : "none");
  if (strcasecmp(params.radial, "sphere") && strcasecmp(params.radial, "cylinder") &&
      strcasecmp(params.radial, "both") && strcasecmp(params.radial, "none")) {
    fprintf(stderr, "  [ERROR] radial must be sphere, cylinder, both or none (got '%s')\n", params.radial);
    exit(1);
  }
  if (params.elecA != 0 || params.elecP != 0) {
    float norm = sqrtf(params.elecX * params.elecX + params.elecY * params.elecY + params.elecZ * params.elecZ);
    if (norm == 0) {
      fprintf(stderr, "  [ERROR] elecA or elecP is set but the field direction (elecX, elecY, elecZ) is zero\n");
      exit(1);
    }
    params.elecX /= norm;
    params.elecY /= norm;
    params.elecZ /= norm;
  }
  validate_parameters(params);
  print_parameters(params);

  char fname[1000];
  simulator *sim = new simulator(&params);
  sim->Setup_simulation(params);

  // Old outputs are removed only now, after the whole setup succeeded, so an input error never deletes results.
  cleanPreviousOutput();

  sprintf(fname, "ic.csv");
  sim->print_n(fname, &params);
  sim->evolve->run();

  print_header("SIMULATION COMPLETE", 60, "=");
  printf("\n");

  return 0;
}
