// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#include <math.h>
#include <string.h>

#include <iostream>
#include <vector>

#include "../include/anchoring.h"
#include "../include/io.h"
#include "../include/parameters.h"

// Planar, non-degenerate Rapini-Papoular anchoring valid on curved surfaces.
// The easy axis at each surface site is the lab direction d(theta_s, phi_s)
// projected onto the local tangent plane: e = (d - (d.s)s) / |d - (d.s)s|.
// On a flat surface with d in-plane this reduces exactly to "rp".
// Where s is parallel to d the projection vanishes (unavoidable on closed
// surfaces, hairy-ball theorem) and the anchoring energy there is zero.

RP_Tangent_Anchoring::RP_Tangent_Anchoring(Parameters *params, int id) {
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
  lab_axis(theta_s, phi_s, d);
  printf("\n");
}

bool RP_Tangent_Anchoring::easy_axis(const float s[3], float e[3]) const {
  float ds = d[0] * s[0] + d[1] * s[1] + d[2] * s[2];
  e[0] = d[0] - ds * s[0];
  e[1] = d[1] - ds * s[1];
  e[2] = d[2] - ds * s[2];
  float mod = sqrtf(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
  if (mod < singular_tol) {
    e[0] = e[1] = e[2] = 0;
    return false;
  }
  e[0] /= mod;
  e[1] /= mod;
  e[2] /= mod;
  return true;
}

float RP_Tangent_Anchoring::surface_potential(float ni[3], float s[3]) {
  float e[3];
  if (!easy_axis(s, e)) return 0;
  float nij = ni[0] * e[0] + ni[1] * e[1] + ni[2] * e[2];
  return -W * nij * nij;
}
