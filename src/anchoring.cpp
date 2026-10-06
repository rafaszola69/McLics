// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
// Helpers shared by all anchoring types.
#include "../include/anchoring.h"

#include <gsl/gsl_rng.h>
#include <math.h>
#include <string.h>

#include <iostream>
#include <map>
#include <vector>

#include "../include/parameters.h"
#include "../include/potential.h"

// Unit vector with polar angle theta and azimuth phi (degrees) in the lab frame: theta = 0 is the z axis.
void Anchoring::lab_axis(float theta_deg, float phi_deg, float out[3]) {
  const float toRad = M_PI / 180;
  out[0] = sinf(toRad * theta_deg) * cosf(toRad * phi_deg);
  out[1] = sinf(toRad * theta_deg) * sinf(toRad * phi_deg);
  out[2] = cosf(toRad * theta_deg);
}

void Anchoring::check_parameter(bool std_val, std::string parameter_name) {
  if (std_val)
    std::cout << "Parameter " << parameter_name << " is not set. Using standard value.\n";

  else {
    std::cout << "Parameter " << parameter_name << " not defined for the boundary condition "
              << "#" << id << ".\n";
    std::cout << "The boundary condition " << this->name << " needs the aforementioned parameter defined.\n";
    std::cout << "Please, set it in your input file, or check if it is misspelled.\n";
    std::cout << "Aborting the program.\n";
    exit(1);
  }
}
