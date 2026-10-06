// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
// Code shared by all geometries: bond energy of a site and creation of the anchorings.
#include "../include/geometry.h"

#include <gsl/gsl_rng.h>
#include <math.h>
#include <string.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../include/anchoring.h"
#include "../include/parameters.h"
#include "../include/potential.h"

Geometry::Geometry(Parameters *params) : Nx(params->Nx), Ny(params->Ny), Nz(params->Nz) {
  this->params = params;
};
float Geometry::bond_energy(const nni fullni[]) {
  float rij[3];
  double E = 0;
  float ni[3] = {fullni[0].x, fullni[0].y, fullni[0].z};

  if (fullni[1].pt) {
    float nj[3] = {fullni[1].x, fullni[1].y, fullni[1].z};
    rij[0] = 1;
    rij[1] = 0;
    rij[2] = 0;
    E += bulk_potential(ni, nj, params, rij, 1);
  }

  if (fullni[2].pt) {
    float nj[3] = {fullni[2].x, fullni[2].y, fullni[2].z};
    rij[0] = -1;
    rij[1] = 0;
    rij[2] = 0;
    E += bulk_potential(ni, nj, params, rij, 1);
  }

  if (fullni[3].pt) {
    float nj[3] = {fullni[3].x, fullni[3].y, fullni[3].z};
    rij[0] = 0;
    rij[1] = 1;
    rij[2] = 0;
    E += bulk_potential(ni, nj, params, rij, 1);
  }

  if (fullni[4].pt) {
    float nj[3] = {fullni[4].x, fullni[4].y, fullni[4].z};
    rij[0] = 0;
    rij[1] = -1;
    rij[2] = 0;
    E += bulk_potential(ni, nj, params, rij, 1);
  }

  if (fullni[5].pt) {
    float nj[3] = {fullni[5].x, fullni[5].y, fullni[5].z};
    rij[0] = 0;
    rij[1] = 0;
    rij[2] = 1;
    E += bulk_potential(ni, nj, params, rij, 1);
  }

  if (fullni[6].pt) {
    float nj[3] = {fullni[6].x, fullni[6].y, fullni[6].z};
    rij[0] = 0;
    rij[1] = 0;
    rij[2] = -1;
    E += bulk_potential(ni, nj, params, rij, 1);
  }
  return E;
}
// Creates the Anchoring object of each surface from `anchoring_type n` (n = 0 .. nSurfaces-1).
void Geometry::Boundary_Init(Parameters *params) {
  std::string anchoring;
  for (int ii = 0; ii < nSurfaces; ii++) {
    try {
      anchoring = params->anchoring_type.at(ii);
    } catch (std::out_of_range dummy_var) {
      fprintf(stderr, "  [ERROR] The geometry has %d surface(s), but anchoring_type %d is not defined in the input file.\n",
              nSurfaces, ii);
      exit(2);
    }
    if (strcasecmp(anchoring.c_str(), "rp") == 0)
      surfaces.at(ii) = new RP_Anchoring(params, ii);
    else if (strcasecmp(anchoring.c_str(), "rp_tangent") == 0)
      surfaces.at(ii) = new RP_Tangent_Anchoring(params, ii);
    else if (strcasecmp(anchoring.c_str(), "fg") == 0)
      surfaces.at(ii) = new FG_Anchoring(params, ii);
    else if (strcasecmp(anchoring.c_str(), "homeotropic") == 0)
      surfaces.at(ii) = new Homeotropic_Anchoring(params, ii);
    else if (strcasecmp(anchoring.c_str(), "rp_ghrl") == 0)
      surfaces.at(ii) = new RP_Anchoring_GHRL(params, ii);
    else if (strcasecmp(anchoring.c_str(), "fg_ghrl") == 0)
      surfaces.at(ii) = new FG_Anchoring_GHRL(params, ii);
    else if (strcasecmp(anchoring.c_str(), "homeotropic_ghrl") == 0)
      surfaces.at(ii) = new Homeotropic_Anchoring_GHRL(params, ii);
    else {
      fprintf(stderr, "  [ERROR] Anchoring type '%s' not implemented. Options: homeotropic | rp | rp_tangent | fg |"
              " homeotropic_ghrl | rp_ghrl | fg_ghrl\n", anchoring.c_str());
      exit(2);
    }
  }
};
