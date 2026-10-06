// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
// fg and fg_ghrl anchorings: degenerate planar (definitions in include/anchoring.h).
#include <gsl/gsl_rng.h>
#include <math.h>
#include <string.h>

#include <iostream>
#include <map>
#include <vector>

#include "../include/anchoring.h"
#include "../include/io.h"
#include "../include/parameters.h"
#include "../include/potential.h"
FG_Anchoring::FG_Anchoring(Parameters *params, int id) {
  this->id = id;
  char label[24];
  snprintf(label, sizeof(label), "Surface %d:", id);
  print_field(label, name);
  this->params = params;

  try {
    W = params->W.at(id);
    print_field("W:", W);
  } catch (std::out_of_range dummy_var) {
    check_parameter(false, "W");
  }
  printf("\n");
}

float FG_Anchoring::surface_potential(float ni[3], float s[3]) {
  float nij = ni[0] * s[0] + ni[1] * s[1] + ni[2] * s[2];
  return +W * nij * nij;
}

FG_Anchoring_GHRL::FG_Anchoring_GHRL(Parameters *params, int id) {
  this->id = id;
  this->params = params;
  char label[24];
  snprintf(label, sizeof(label), "Surface %d:", id);
  print_field(label, name);
  try {
    W = params->W.at(id);
    print_field("W:", W);
  } catch (std::out_of_range dummy_var) {
    check_parameter(false, "W");
  }
  printf("\n");
}

float FG_Anchoring_GHRL::surface_potential(float ni[3], float s[3]) {
  // Phantom neighbour along the surface normal s, oriented along the projection of ni on the tangent plane.
  float ns = ni[0] * s[0] + ni[1] * s[1] + ni[2] * s[2];
  float nj[3] = {ni[0] - ns * s[0], ni[1] - ns * s[1], ni[2] - ns * s[2]};
  float mod = sqrtf(nj[0] * nj[0] + nj[1] * nj[1] + nj[2] * nj[2]);
  if (mod < 1e-6f) return 0;
  nj[0] /= mod;
  nj[1] /= mod;
  nj[2] /= mod;
  return W * Bulk_Energy_GHRL(ni, nj, params, s);
}
