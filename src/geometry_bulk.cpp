// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
// Bulk geometry: every site is bulk; boundaries from the input (xbound, ybound, zbound).
#include <gsl/gsl_rng.h>
#include <math.h>
#include <string.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../include/anchoring.h"
#include "../include/geometry.h"
#include "../include/io.h"
#include "../include/parameters.h"
#include "../include/potential.h"

Bulk_Geometry::Bulk_Geometry(int *pt, Parameters *params) : Geometry(params) {
  print_field("Geometry:", "Bulk");
  ns = (float *)calloc(Nx * Ny * Nz * 3, sizeof(float));

  nSurfaces = 0;
  surfaces = std::vector<class Anchoring *>(nSurfaces);
  pt = set_point_type_normals(pt, params);

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

int *Bulk_Geometry::set_point_type_normals(int *pt, Parameters *params) {
  for (int ii = 0; ii < Nx * Ny * Nz; ii++) pt[ii] = 1;

  return pt;
}

float Bulk_Geometry::site_energy(const nni fullni[8]) {
  float E = 0;
  float ni[3] = {fullni[0].x, fullni[0].y, fullni[0].z};
  E = Geometry::bond_energy(fullni);
  if (params->elecA != 0 || params->elecP != 0)
    E += Electric_Potential(ni, params);

  return E;
}
