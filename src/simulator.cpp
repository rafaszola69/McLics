// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
// Builds the simulation from the parameters (evolution mode, geometry, anchorings, initial condition, potential).
#include <gsl/gsl_rng.h>
#include <string.h>

#include <iostream>

#include "../include/define.h"
#include "../include/evolve.h"
#ifdef CUDA__
#include "../include/evolve.cuh"
#endif
#include "../include/ic.h"
#include "../include/io.h"
#include "../include/parameter_order.h"
#include "../include/parameters.h"
#include "../include/potential.h"
#include "../include/simulator.h"

simulator::simulator(Parameters *params) : Nx(params->Nx), Ny(params->Ny), Nz(params->Nz) {
  this->params = params;
}

simulator::~simulator() {
}

// Setup, in this order: evolution mode, geometry (point types, normals, surfaces), boundary functions,
// checkerboard check, anchorings, initial condition and bulk potential. Every input error aborts here, before
// main removes the output of a previous run.
void simulator::Setup_simulation(Parameters &params) {
  int Nn = 3 * params.Nx * params.Ny * params.Nz;
  ni = (float *)calloc(Nn, sizeof(float));
  pt = (int *)calloc(Nn, sizeof(int));

  if (strcasecmp(params.evol, "thermal") == 0) {
    evolve = new thermalEvolveN(ni, pt, &params);
  } else if (strcasecmp(params.evol, "step") == 0) {
    evolve = new stepEvolveN(ni, pt, &params);
  } else if (strcasecmp(params.evol, "quench") == 0) {
    evolve = new quenchEvolveN(ni, pt, &params);
  } else if (strcasecmp(params.evol, "electric") == 0) {
    evolve = new electricEvolveN(ni, pt, &params);
#ifdef CUDA__
  } else if (strcasecmp(params.evol, "thermalGPU") == 0) {
    evolve = new thermalEvolveNGPU(ni, pt, &params);
  } else if (strcasecmp(params.evol, "stepGPU") == 0) {
    evolve = new stepEvolveNGPU(ni, pt, &params);
  } else if (strcasecmp(params.evol, "quenchGPU") == 0) {
    evolve = new quenchEvolveNGPU(ni, pt, &params);
  } else if (strcasecmp(params.evol, "electricGPU") == 0) {
    evolve = new electricEvolveNGPU(ni, pt, &params);
  } else {
    fprintf(stderr, "  [ERROR] Evol mode '%s' not implemented. Options: thermal | step | quench | electric, or the same"
            " with the GPU suffix (thermalGPU, ...)\n", params.evol);
#else
  } else {
    fprintf(stderr, "  [ERROR] Evol mode '%s' not implemented. Options: thermal | step | quench | electric (this is the"
            " CPU build: compile with nvcc for the GPU modes)\n", params.evol);
#endif
    exit(2);
  }

  if (strcasecmp(params.geometry, "bulk") == 0) {
    evolve->geometry = new Bulk_Geometry(pt, &params);
  } else if (strcasecmp(params.geometry, "slab") == 0) {
    evolve->geometry = new Slab_Geometry(pt, &params);
  } else if (strcasecmp(params.geometry, "sphere") == 0) {
    evolve->geometry = new Sphere_Geometry(pt, &params);
  } else if (strcasecmp(params.geometry, "custom") == 0) {
    evolve->geometry = new Custom_Geometry(pt, &params);
  } else {
    fprintf(stderr, "  [ERROR] Geometry '%s' not implemented. Options: bulk | slab | sphere | custom\n", params.geometry);
    exit(2);
  }

  if (strcasecmp(params.XBoundtype, "free") == 0)
    params.XBound = &Free_Boundary;
  else if (strcasecmp(params.XBoundtype, "periodic") == 0)
    params.XBound = &Periodic_Boundary;
  else {
    fprintf(stderr, "  [ERROR] X boundary '%s' not implemented. Options: free | periodic\n", params.XBoundtype);
    exit(2);
  }

  if (strcasecmp(params.YBoundtype, "free") == 0)
    params.YBound = &Free_Boundary;
  else if (strcasecmp(params.YBoundtype, "periodic") == 0)
    params.YBound = &Periodic_Boundary;
  else {
    fprintf(stderr, "  [ERROR] Y boundary '%s' not implemented. Options: free | periodic\n", params.YBoundtype);
    exit(2);
  }

  if (strcasecmp(params.ZBoundtype, "free") == 0)
    params.ZBound = &Free_Boundary;
  else if (strcasecmp(params.ZBoundtype, "periodic") == 0)
    params.ZBound = &Periodic_Boundary;
  else {
    fprintf(stderr, "  [ERROR] Z boundary '%s' not implemented. Options: free | periodic\n", params.ZBoundtype);
    exit(2);
  }

  check_checkerboard(params);
  evolve->geometry->Boundary_Init(&params);
  apply_Initial_Conditions(ni, pt, params);

  if (strcasecmp(params.potential, "ll") * strcasecmp(params.potential, "lebwohl-lasher") == 0) {
    evolve->geometry->bulk_potential = &Bulk_Energy_Lebwohl_Lasher;
    print_field("Potential:", "Lebwohl-Lasher");
    printf("\n");
  } else if (strcasecmp(params.potential, "ghrl") * strcasecmp(params.potential, "gruhn-hess") == 0) {
    evolve->geometry->bulk_potential = &Bulk_Energy_GHRL;
    setGHRL(params);
    print_field("Potential:", "Gruhn-Hess (GHRL)");
    printf("\n");
  } else if (strcasecmp(params.potential, "pear") == 0) {
    evolve->geometry->bulk_potential = &Bulk_Energy_Selinger_Pear;
    print_field("Potential:", "Pear (Dhakal-Selinger)");
    printf("\n");
  } else {
    fprintf(stderr, "  [ERROR] Potential '%s' not implemented. Options: ll | ghrl | pear\n", params.potential);
    exit(2);
  }
}

// ic.csv has the same format as the snapshots (local order parameter S, NaN on empty sites).
int simulator::print_n(char *fname, Parameters *params) {
  return ::print_n(fname, ni, *params, pt);
}
