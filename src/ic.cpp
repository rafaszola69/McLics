// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
// Initial conditions of the director field.
#include "../include/ic.h"

#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <math.h>
#include <string.h>

#include <fstream>
#include <vector>
#include <string>

#include "../include/define.h"
#include "../include/io.h"
#include "../include/parameters.h"

// Initial director field, selected by `ic`: random (isotropic, from `seed`), homogeneous (direction theta_0, phi_0
// in degrees), ic_file (read from a file) or cholesteric (helix along z with pitch p0_i, or p0 if p0_i = 0, starting
// at azimuth phi_0 with polar angle theta_0). Empty sites get n = (0, 0, 1).
void apply_Initial_Conditions(float *ni, int *pt, Parameters params) {
  if (strcasecmp(params.ic, "random") == 0) {
    random_ic(ni, pt, params);
  } else if (strcasecmp(params.ic, "homogeneous") == 0) {
    homogeneous_ic(ni, pt, params);
  } else if (strcasecmp(params.ic, "ic_file") == 0) {
    read_ic_file(ni, pt, params);
  } else if (strcasecmp(params.ic, "cholesteric") == 0) {
    cholesteric_ic(ni, pt, params);
  } else {
    fprintf(stderr, "  [ERROR] Initial condition '%s' not implemented. Options: random | homogeneous | ic_file | cholesteric\n", params.ic);
    exit(2);
  }
}

void random_ic(float *ni, int *pt, Parameters params) {
  int Nx = params.Nx;
  int Ny = params.Ny;
  int Nz = params.Nz;
  double nx, ny, nz;
  gsl_rng *rng;
  gsl_rng_env_setup();
  rng = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(rng, params.seed);
  printf("  Initial condition: random (seed %d)\n\n", params.seed);
  for (int i = 0; i < params.Nx; i++) {
    for (int j = 0; j < params.Ny; j++) {
      for (int k = 0; k < params.Nz; k++) {
        if (pti(i, j, k)) {
          gsl_ran_dir_3d(rng, &nx, &ny, &nz);
          nix(i, j, k) = nx;
          niy(i, j, k) = ny;
          niz(i, j, k) = nz;
        } else {
          nix(i, j, k) = 0;
          niy(i, j, k) = 0;
          niz(i, j, k) = 1;
        }
      }
    }
  }
  gsl_rng_free(rng);
}

void homogeneous_ic(float *ni, int *pt, Parameters params) {
  int Nx = params.Nx;
  int Ny = params.Ny;
  int Nz = params.Nz;

  float theta = params.theta_0;
  float phi = params.phi_0;
  printf("  Initial condition: homogeneous\n");
  printf("  %-10s %g\n", "phi0:", phi);
  printf("  %-10s %g\n\n", "theta0:", theta);
  for (int i = 0; i < Nx; i++) {
    for (int j = 0; j < Ny; j++) {
      for (int k = 0; k < Nz; k++) {
        if (pti(i, j, k)) {
          nix(i, j, k) = sin(theta * M_PI / 180) * cos(phi * M_PI / 180);
          niy(i, j, k) = sin(theta * M_PI / 180) * sin(phi * M_PI / 180);
          niz(i, j, k) = cos(theta * M_PI / 180);
        } else {
          nix(i, j, k) = 0;
          niy(i, j, k) = 0;
          niz(i, j, k) = 1;
        }
      }
    }
  }
}

// Initial-condition file: comma-separated, one header line naming the columns, then one line per grid point. The
// columns x, y, z (integer grid indices) and nx, ny, nz (director) are required, in any order; other columns are
// ignored, so a director_field snapshot can be used directly. Every non-empty site must be listed. Directors whose
// length differs from 1 by more than 1e-3 are normalized (files written with rounding are used as they are).
static void ic_error(const char *file, int line, const char *msg) {
  fprintf(stderr, "  [ERROR] %s, line %d: %s\n", file, line, msg);
  exit(3);
}

void read_ic_file(float *ni, int *pt, Parameters params) {
  int Nx = params.Nx;
  int Ny = params.Ny;
  int Nz = params.Nz;
  const char *fname = params.ic_file;
  std::ifstream in(fname);
  if (!in) {
    perror(fname);
    exit(2);
  }
  printf("  Initial condition: file (%s)\n", fname);
  fflush(stdout);

  std::string line;
  if (!std::getline(in, line)) ic_error(fname, 1, "empty file (a header line is expected).");
  std::vector<std::string> header = split_csv(line);
  const char *names[6] = {"x", "y", "z", "nx", "ny", "nz"};
  int col[6];
  std::string missing;
  for (int c = 0; c < 6; c++) {
    col[c] = -1;
    for (size_t h = 0; h < header.size(); h++)
      if (strcasecmp(header[h].c_str(), names[c]) == 0) col[c] = (int)h;
    if (col[c] < 0) missing += std::string(missing.empty() ? "" : ", ") + names[c];
  }
  if (!missing.empty()) {
    std::string msg = "missing column(s) in the header: " + missing + " (required: x,y,z,nx,ny,nz).";
    ic_error(fname, 1, msg.c_str());
  }

  const int Nt = Nx * Ny * Nz;
  std::vector<char> seen(Nt, 0);
  int line_number = 1, normalized = 0;
  char msg[300];
  while (std::getline(in, line)) {
    line_number++;
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    std::vector<std::string> f = split_csv(line);
    if (f.size() != header.size()) {
      snprintf(msg, sizeof(msg), "%zu fields, but the header has %zu.", f.size(), header.size());
      ic_error(fname, line_number, msg);
    }
    double v[6];
    for (int c = 0; c < 6; c++) {
      char *end;
      v[c] = strtod(f[col[c]].c_str(), &end);
      if (f[col[c]].empty() || *end != '\0') {
        snprintf(msg, sizeof(msg), "column %s is not a number: '%s'.", names[c], f[col[c]].c_str());
        ic_error(fname, line_number, msg);
      }
    }
    int i = (int)v[0], j = (int)v[1], k = (int)v[2];
    if (v[0] != i || v[1] != j || v[2] != k) ic_error(fname, line_number, "x, y and z must be integers.");
    if (i < 0 || i >= Nx || j < 0 || j >= Ny || k < 0 || k >= Nz) {
      snprintf(msg, sizeof(msg), "point (%d, %d, %d) outside the %d x %d x %d grid.", i, j, k, Nx, Ny, Nz);
      ic_error(fname, line_number, msg);
    }
    int idx = i + Nx * (j + Ny * k);
    if (seen[idx]) {
      snprintf(msg, sizeof(msg), "point (%d, %d, %d) is listed twice.", i, j, k);
      ic_error(fname, line_number, msg);
    }
    seen[idx] = 1;
    double nx = v[3], ny = v[4], nz = v[5], norm = sqrt(nx * nx + ny * ny + nz * nz);
    if (pti(i, j, k)) {
      if (norm < 1e-6) ic_error(fname, line_number, "zero director on a non-empty site.");
      if (fabs(norm - 1) > 1e-3) {
        normalized++;
        nx /= norm;
        ny /= norm;
        nz /= norm;
      }
    } else {  // empty site: the director is not used
      nx = 0;
      ny = 0;
      nz = 1;
    }
    nix(i, j, k) = nx;
    niy(i, j, k) = ny;
    niz(i, j, k) = nz;
  }
  int unlisted = 0;
  for (int idx = 0; idx < Nt; idx++) {
    if (seen[idx]) continue;
    if (pt[idx]) unlisted++;
    ni[3 * idx + 0] = 0;
    ni[3 * idx + 1] = 0;
    ni[3 * idx + 2] = 1;
  }
  if (unlisted > 0) {
    fprintf(stderr, "  [ERROR] %s: %d non-empty grid point(s) are not listed (the grid is %d x %d x %d).\n", fname,
            unlisted, Nx, Ny, Nz);
    exit(3);
  }
  if (normalized > 0)
    fprintf(stderr, "  [WARNING] %d director(s) in %s did not have unit length and were normalized.\n", normalized,
            fname);
}

void cholesteric_ic(float *ni, int *pt, Parameters params) {
  int Nx = params.Nx;
  int Ny = params.Ny;
  int Nz = params.Nz;

  float theta = params.theta_0;
  float phi_0 = params.phi_0 * M_PI / 180;
  float p0 = params.p0_i ? params.p0_i : params.p0;
  float phi;
  printf("  Initial condition: cholesteric\n");
  printf("  %-10s %g\n", "p0:", p0);
  printf("  %-10s %g\n", "phi0:", params.phi_0);
  printf("  %-10s %g\n\n", "theta0:", theta);
  for (int i = 0; i < Nx; i++) {
    for (int j = 0; j < Ny; j++) {
      for (int k = 0; k < Nz; k++) {
        if (pti(i, j, k)) {
          phi = phi_0 - 2 * M_PI * k / p0;

          nix(i, j, k) = sin(theta * M_PI / 180) * cos(phi);
          niy(i, j, k) = sin(theta * M_PI / 180) * sin(phi);
          niz(i, j, k) = cos(theta * M_PI / 180);
        } else {
          nix(i, j, k) = 0;
          niy(i, j, k) = 0;
          niz(i, j, k) = 1;
        }
      }
    }
  }
}
