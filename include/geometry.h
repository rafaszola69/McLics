// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef GEOMETRY_H_
#define GEOMETRY_H_
#include <gsl/gsl_rng.h>

#include <iostream>
#include <vector>

#include "../include/define.h"
#include "../include/io.h"
#include "../include/parameter_order.h"
#include "../include/parameters.h"
#include "../include/potential.h"

// A geometry fills the point types pt of the lattice and the outward normals ns of the surface sites (layout in
// define.h), creates one Anchoring per surface and evaluates the energy of a site:
//   site_energy(fullni) = sum over the existing neighbours of bulk_potential + anchoring (surface sites)
//                              + field coupling.
// This site energy is what the Metropolis step compares. The energy E in po.dat is its average over the
// non-empty sites, so each bond is counted twice there while anchoring and field are counted once.
//   bulk    all sites bulk; x, y, z boundaries from the input
//   slab    planes z = 0 (surface 0, normal -z) and z = Nz-1 (surface 1, normal +z); z free, x, y from the input
//   sphere  centre (Nx/2, Ny/2, Nz/2): bulk for r < Nz/2 - 1, surface 0 for r < Nz/2 + 1 (radial normal), empty
//           outside; all boundaries free
//   custom  read from the geometry file (format in geometry_custom.cpp); boundaries from the input
class Geometry {
 public:
  Geometry(Parameters *params);
  int Nx, Ny, Nz;
  std::vector<class Anchoring *> surfaces;
  int nSurfaces;
  // bond energy of fullni[0] with its six neighbours (fullni layout in define.h)
  float bond_energy(const nni fullni[]);
  void Boundary_Init(Parameters *params);
  virtual int *set_point_type_normals(int *pt, Parameters *params) = 0;
  virtual float site_energy(const nni ni[8]) = 0;
  // pair potential selected by `potential` (set in simulator.cpp)
  float (*bulk_potential)(float ni[3], float nj[3], Parameters *params, float rij[3]);
  float *ns;  // outward unit normals of the surface sites (3 per site, zero elsewhere)
  Parameters *params;
};

class Bulk_Geometry : public Geometry {
 public:
  Bulk_Geometry(int *pt, Parameters *params);
  float site_energy(const nni ni[8]);

 private:
  int *set_point_type_normals(int *pt, Parameters *params);
};

class Slab_Geometry : public Geometry {
 public:
  Slab_Geometry(int *pt, Parameters *params);
  float site_energy(const nni ni[8]);

 private:
  int *set_point_type_normals(int *pt, Parameters *params);
};

class Sphere_Geometry : public Geometry {
 public:
  Sphere_Geometry(int *pt, Parameters *params);
  float site_energy(const nni ni[8]);

 private:
  int *set_point_type_normals(int *pt, Parameters *params);
};

class Custom_Geometry : public Geometry {
 public:
  Custom_Geometry(int *pt, Parameters *params);
  float site_energy(const nni ni[8]);

 private:
  int *set_point_type_normals(int *pt, Parameters *params);
};

#endif
