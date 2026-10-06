// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef ANCHORING_H_
#define ANCHORING_H_

#include <gsl/gsl_rng.h>
#include <iostream>
#include <vector>
#include "../include/define.h"
#include "../include/io.h"
#include "../include/parameter_order.h"
#include "../include/parameters.h"
#include "../include/potential.h"

// Surface anchoring. Each surface (pt = n + 2 in the lattice) gets one Anchoring object, chosen by
// `anchoring_type n` in the input (Geometry::Boundary_Init). surface_potential(n_i, s) is the anchoring energy of a
// surface site with director n_i and outward unit normal s; it is added once per surface site to its energy.
// LL-form types (energy -W (n.e)^2 or +W (n.s)^2):
//   homeotropic  -W (n.s)^2, easy axis = local normal
//   rp           -W (n.e)^2, e = fixed lab axis (theta_s, phi_s)
//   rp_tangent   -W (n.e)^2, e = lab axis projected on the local tangent plane (zero where it vanishes)
//   fg           +W (n.s)^2, degenerate planar (Fournier-Galatola-like): any n in the tangent plane
// GHRL-form types (for the ghrl potential): the surface acts as a phantom neighbour at distance 1 along s, with
// director e, energy W U_GHRL(n, e, r = s): e = s (homeotropic_ghrl), the lab axis (rp_ghrl) or the projection of n
// on the tangent plane (fg_ghrl).
// The GPU has its own kernels for these types (geometry.cu), selected by the name string returned by getName()
// (setSurfaces in evolve.cu): changing a name requires changing it there too.
class Anchoring {
 public:
  char name[50];
  int id;
  Parameters* params;
  virtual char* getName() { return name; };
  virtual float surface_potential(float ni[3], float s[3]) { return 0; };
  void check_parameter(bool std_val, std::string parameter_name);
  static void lab_axis(float theta_deg, float phi_deg, float out[3]);
};

class RP_Anchoring : public Anchoring {
 public:
  char name[50] = "Rapini-Papoular Anchoring";
  RP_Anchoring(Parameters* params, int id);
  float W, theta_s, phi_s;
  float n_s[3];
  char* getName() { return name; };
  float surface_potential(float ni[3], float s[3]);
};

class RP_Tangent_Anchoring : public Anchoring {
 public:
  char name[50] = "Rapini-Papoular Tangent Anchoring";
  RP_Tangent_Anchoring(Parameters* params, int id);
  float W, theta_s, phi_s;
  float d[3];
  static constexpr float singular_tol = 1e-3f;
  char* getName() { return name; };
  bool easy_axis(const float s[3], float e[3]) const;
  float surface_potential(float ni[3], float s[3]);
};

class FG_Anchoring : public Anchoring {
 public:
  char name[50] = "Fournier-Galatola Anchoring";
  FG_Anchoring(Parameters* params, int id);
  float W, theta_s, phi_s;
  char* getName() { return name; };
  float surface_potential(float ni[3], float s[3]);
};
class Homeotropic_Anchoring : public Anchoring {
 public:
  char name[50] = "Homeotropic Anchoring";
  Homeotropic_Anchoring(Parameters* params, int id);
  float W, theta_s, phi_s;
  char* getName() { return name; };
  float surface_potential(float ni[3], float s[3]);
};

class RP_Anchoring_GHRL : public Anchoring {
 public:
  char name[50] = "Rapini-Papoular Anchoring GHRL";
  RP_Anchoring_GHRL(Parameters* params, int id);
  float W, theta_s, phi_s;
  float n_s[3];
  char* getName() { return name; };
  float surface_potential(float ni[3], float s[3]);
};

class FG_Anchoring_GHRL : public Anchoring {
 public:
  char name[50] = "Fournier-Galatola Anchoring GHRL";
  FG_Anchoring_GHRL(Parameters* params, int id);
  float W, theta_s, phi_s;
  char* getName() { return name; };
  float surface_potential(float ni[3], float s[3]);
};
class Homeotropic_Anchoring_GHRL : public Anchoring {
 public:
  char name[50] = "Homeotropic Anchoring GHRL";
  Homeotropic_Anchoring_GHRL(Parameters* params, int id);
  float W, theta_s, phi_s;
  char* getName() { return name; };
  float surface_potential(float ni[3], float s[3]);
};
#endif
