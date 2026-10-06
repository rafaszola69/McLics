// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
// Sphere geometry: droplet centred at (Nx/2, Ny/2, Nz/2) (integer division) with one surface layer of radial normals.
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

Sphere_Geometry::Sphere_Geometry(int *pt, Parameters *params) : Geometry(params) {
  print_field("Geometry:", "Sphere");
  nSurfaces = 1;
  sprintf(params->XBoundtype, "free");
  sprintf(params->YBoundtype, "free");
  sprintf(params->ZBoundtype, "free");
  ns = (float *)calloc(Nx * Ny * Nz * 3, sizeof(float));
  surfaces = std::vector<class Anchoring *>(nSurfaces);
  pt = set_point_type_normals(pt, params);
  printf("\n");
}

int *Sphere_Geometry::set_point_type_normals(int *pt, Parameters *params) {
  int ii, jj, kk;
  float Rx, Ry, Rz;
  float Hx = Nx / 2;
  float Hy = Ny / 2;
  float Hz = Nz / 2;
  for (ii = 0; ii < Nx; ii++) {
    for (jj = 0; jj < Ny; jj++) {
      for (kk = 0; kk < Nz; kk++) {
        Rx = ii - Hx;
        Ry = jj - Hy;
        Rz = kk - Hz;
        float radius = sqrt(Rx * Rx + Ry * Ry + Rz * Rz);
        if (radius < Hz - 1)
          pt[ii + Nx * (jj + Ny * kk)] = 1;
        else if (radius < Hz + 1) {
          pt[ii + Nx * (jj + Ny * kk)] = 2;
          ns[(ii + Nx * (jj + Ny * kk)) * 3 + 0] = Rx / radius;
          ns[(ii + Nx * (jj + Ny * kk)) * 3 + 1] = Ry / radius;
          ns[(ii + Nx * (jj + Ny * kk)) * 3 + 2] = Rz / radius;
        } else
          pt[ii + Nx * (jj + Ny * kk)] = 0;
      }
    }
  }
  return pt;
}

float Sphere_Geometry::site_energy(const nni fullni[8]) {
  float E = 0;
  float ni[3] = {fullni[0].x, fullni[0].y, fullni[0].z};
  float s[3] = {fullni[7].x, fullni[7].y, fullni[7].z};
  E = Geometry::bond_energy(fullni);

  if (fullni[0].pt > 1)
    E += surfaces[fullni[0].pt - 2]->surface_potential(ni, s);
  if (params->elecA != 0 || params->elecP != 0)
    E += Electric_Potential(ni, params);

  return E;
}
