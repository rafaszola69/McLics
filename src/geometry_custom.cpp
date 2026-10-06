// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
// Custom geometry: point types and normals read from a file (format below); boundaries from the input.
#include <gsl/gsl_rng.h>
#include <math.h>
#include <string.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../include/anchoring.h"
#include "../include/geometry.h"
#include "../include/io.h"
#include "../include/parameters.h"
#include "../include/potential.h"
#define MAX(a, b) (a > b ? a : b)
Custom_Geometry::Custom_Geometry(int *pt, Parameters *params) : Geometry(params) {
  print_field("Geometry:", "Custom");

  ns = (float *)calloc(Nx * Ny * Nz * 3, sizeof(float));
  pt = set_point_type_normals(pt, params);
  surfaces = std::vector<class Anchoring *>(nSurfaces);

  if (strcasecmp(params->XBoundtype, "free") == 0)
    params->XBound = &Free_Boundary;
  else if (strcasecmp(params->XBoundtype, "periodic") == 0)
    params->XBound = &Periodic_Boundary;
  else {
    fprintf(stderr, "  [ERROR] X boundary '%s' not implemented. Options: free | periodic\n", params->XBoundtype);
    exit(2);
  }

  if (strcasecmp(params->YBoundtype, "free") == 0)
    params->YBound = &Free_Boundary;
  else if (strcasecmp(params->YBoundtype, "periodic") == 0)
    params->YBound = &Periodic_Boundary;
  else {
    fprintf(stderr, "  [ERROR] Y boundary '%s' not implemented. Options: free | periodic\n", params->YBoundtype);
    exit(2);
  }

  if (strcasecmp(params->ZBoundtype, "free") == 0)
    params->ZBound = &Free_Boundary;
  else if (strcasecmp(params->ZBoundtype, "periodic") == 0)
    params->ZBound = &Periodic_Boundary;
  else {
    fprintf(stderr, "  [ERROR] Z boundary '%s' not implemented. Options: free | periodic\n", params->ZBoundtype);
    exit(2);
  }

  print_field("X boundary:", params->XBoundtype);
  print_field("Y boundary:", params->YBoundtype);
  print_field("Z boundary:", params->ZBoundtype);
  printf("\n");
}

// Geometry file: comma-separated, one header line naming the columns, then one line per grid point. The columns
// x, y, z (integer grid indices), nx, ny, nz (outward surface normal) and pt (0 empty, 1 bulk, n >= 2 surface n-2)
// are required, in any order; other columns are ignored. Points that are not listed are bulk (pt = 1).
static void geometry_error(const char *file, int line, const char *msg) {
  fprintf(stderr, "  [ERROR] %s, line %d: %s\n", file, line, msg);
  exit(10);
}

int *Custom_Geometry::set_point_type_normals(int *pt, Parameters *params) {
  const char *fname = params->bound_file_name;
  std::ifstream in(fname);
  if (!in) {
    perror(fname);
    exit(2);
  }
  printf("  Boundary file: %s\n", fname);

  // header: locate the required columns by name
  std::string line;
  if (!std::getline(in, line)) geometry_error(fname, 1, "empty file (a header line is expected).");
  std::vector<std::string> header = split_csv(line);
  const char *names[7] = {"x", "y", "z", "nx", "ny", "nz", "pt"};
  int col[7];
  std::string missing;
  for (int c = 0; c < 7; c++) {
    col[c] = -1;
    for (size_t h = 0; h < header.size(); h++)
      if (strcasecmp(header[h].c_str(), names[c]) == 0) col[c] = (int)h;
    if (col[c] < 0) missing += std::string(missing.empty() ? "" : ", ") + names[c];
  }
  if (!missing.empty()) {
    std::string msg = "missing column(s) in the header: " + missing + " (required: x,y,z,nx,ny,nz,pt).";
    geometry_error(fname, 1, msg.c_str());
  }
  printf("  Column positions: x=%d y=%d z=%d nx=%d ny=%d nz=%d pt=%d\n", col[0], col[1], col[2], col[3], col[4],
         col[5], col[6]);

  const int Nt = Nx * Ny * Nz;
  std::vector<char> seen(Nt, 0);
  for (int k = 0; k < Nt; k++) pt[k] = 1;  // points not listed are bulk
  int max_point_kind = 0, listed = 0, renormalized = 0, line_number = 1;
  char msg[300];

  while (std::getline(in, line)) {
    line_number++;
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;  // blank line
    std::vector<std::string> f = split_csv(line);
    if (f.size() != header.size()) {
      snprintf(msg, sizeof(msg), "%zu fields, but the header has %zu.", f.size(), header.size());
      geometry_error(fname, line_number, msg);
    }
    double v[7];
    for (int c = 0; c < 7; c++) {
      char *end;
      v[c] = strtod(f[col[c]].c_str(), &end);
      if (f[col[c]].empty() || *end != '\0') {
        snprintf(msg, sizeof(msg), "column %s is not a number: '%s'.", names[c], f[col[c]].c_str());
        geometry_error(fname, line_number, msg);
      }
    }
    int ii = (int)v[0], jj = (int)v[1], kk = (int)v[2], type = (int)v[6];
    if (v[0] != ii || v[1] != jj || v[2] != kk || v[6] != type)
      geometry_error(fname, line_number, "x, y, z and pt must be integers.");
    if (ii < 0 || ii >= Nx || jj < 0 || jj >= Ny || kk < 0 || kk >= Nz) {
      snprintf(msg, sizeof(msg), "point (%d, %d, %d) outside the %d x %d x %d grid.", ii, jj, kk, Nx, Ny, Nz);
      geometry_error(fname, line_number, msg);
    }
    if (type < 0) geometry_error(fname, line_number, "negative point type (use 0 empty, 1 bulk, 2+ surface).");
    int idx = (kk * Ny + jj) * Nx + ii;
    if (seen[idx]) {
      snprintf(msg, sizeof(msg), "point (%d, %d, %d) is listed twice.", ii, jj, kk);
      geometry_error(fname, line_number, msg);
    }
    seen[idx] = 1;
    listed++;
    double nx = v[3], ny = v[4], nz = v[5], norm = sqrt(nx * nx + ny * ny + nz * nz);
    if (type >= 2) {  // surface points: the normal is used by the anchoring
      if (norm < 1e-6) geometry_error(fname, line_number, "surface point with a zero normal vector.");
      if (fabs(norm - 1) > 1e-3) {  // only clearly non-unit normals are rescaled (files written with rounding are kept)
        renormalized++;
        nx /= norm;
        ny /= norm;
        nz /= norm;
      }
    }
    ns[idx * 3 + 0] = nx;
    ns[idx * 3 + 1] = ny;
    ns[idx * 3 + 2] = nz;
    pt[idx] = type;
    max_point_kind = MAX(max_point_kind, type);
  }
  if (listed < Nt)
    fprintf(stderr, "  [WARNING] %d of the %d grid points are not listed in %s; they were set to bulk (pt = 1).\n",
            Nt - listed, Nt, fname);
  if (renormalized > 0)
    fprintf(stderr, "  [WARNING] %d surface normal(s) in %s did not have unit length and were normalized.\n",
            renormalized, fname);
  nSurfaces = max_point_kind > 1 ? max_point_kind - 1 : 0;
  printf("  %d boundary surface(s) found.\n\n", nSurfaces);
  fflush(stdout);
  return pt;
}

float Custom_Geometry::site_energy(const nni fullni[8]) {
  float E = 0;
  float ni[3] = {fullni[0].x, fullni[0].y, fullni[0].z};
  
  E = Geometry::bond_energy(fullni);

  if (fullni[0].pt > 1) {
    float s[3] = {fullni[7].x, fullni[7].y, fullni[7].z};
    E += surfaces[fullni[0].pt - 2]->surface_potential(ni, s);
  }
  if (params->elecA != 0 || params->elecP != 0)
    E += Electric_Potential(ni, params);

  return E;
}
