// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
// rp and rp_ghrl anchorings: fixed easy axis (theta_s, phi_s) in the lab frame (definitions in include/anchoring.h).
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

RP_Anchoring::RP_Anchoring(Parameters *params, int id) {
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
  try {
    phi_s = params->phi_s.at(id);
    print_field("phi_s:", phi_s);
  } catch (std::out_of_range dummy_var) {
    check_parameter(false, "phi_s");
  }
  try {
    theta_s = params->theta_s.at(id);
    print_field("theta_s:", theta_s);
  } catch (std::out_of_range dummy_var) {
    check_parameter(false, "theta_s");
  }
  lab_axis(theta_s, phi_s, n_s);
  printf("\n");
}

float RP_Anchoring::surface_potential(float ni[3], float s[3]) {
  float nij = ni[0] * n_s[0] + ni[1] * n_s[1] + ni[2] * n_s[2];
  return -W * nij * nij;
}

RP_Anchoring_GHRL::RP_Anchoring_GHRL(Parameters *params, int id) {
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
  try {
    phi_s = params->phi_s.at(id);
    print_field("phi_s:", phi_s);
  } catch (std::out_of_range dummy_var) {
    check_parameter(false, "phi_s");
  }
  try {
    theta_s = params->theta_s.at(id);
    print_field("theta_s:", theta_s);
  } catch (std::out_of_range dummy_var) {
    check_parameter(false, "theta_s");
  }
  lab_axis(theta_s, phi_s, n_s);
  printf("\n");
}

float RP_Anchoring_GHRL::surface_potential(float ni[3], float s[3]) {
  // Phantom neighbour with the easy axis, located along the surface normal s.
  return W * Bulk_Energy_GHRL(ni, n_s, params, s);
}
