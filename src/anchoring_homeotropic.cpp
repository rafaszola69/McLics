// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
// homeotropic and homeotropic_ghrl anchorings (definitions in include/anchoring.h).
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

Homeotropic_Anchoring::Homeotropic_Anchoring(Parameters *params, int id) {
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

float Homeotropic_Anchoring::surface_potential(float ni[3], float s[3]) {
  float nij = ni[0] * s[0] + ni[1] * s[1] + ni[2] * s[2];
  return -W * nij * nij;
}

Homeotropic_Anchoring_GHRL::Homeotropic_Anchoring_GHRL(Parameters *params, int id) {
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

float Homeotropic_Anchoring_GHRL::surface_potential(float ni[3], float s[3]) {
  // Phantom neighbour along the surface normal s, oriented along s.
  return W * Bulk_Energy_GHRL(ni, s, params, s);
}
