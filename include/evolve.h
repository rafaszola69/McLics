// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef EVOL_H_
#define EVOL_H_
#include <iostream>

#include "../include/define.h"
#include "../include/geometry.h"
#include "../include/parameters.h"
#include "../include/potential.h"
// Evolution modes. An Evolve object owns the simulation loop (run): it sweeps the temperature (thermal), the field
// (electric), or jumps from Ti to Tf (quench), or runs at fixed Ti writing fn snapshots (step), and writes po.dat
// and the director_field_*.csv snapshots. EvolveN and its subclasses run on the CPU (evolve*.cpp); EvolveNGPU and
// its subclasses (evolve.cuh, evolve*.cu) run on the GPU.
class Evolve {
 public:
  virtual int run() { return 0; };
  Geometry *geometry;
};
class EvolveN : public Evolve {
 public:
  EvolveN(float *ni, int *pt, Parameters *params) : ni(ni), pt(pt), params(params), Nx(params->Nx), Ny(params->Ny), Nz(params->Nz){};
  virtual int run() { return 0; };
  // One Monte Carlo sweep (one trial move per non-empty site) at T = params->T; adapts ang_var. r: one GSL
  // generator per OpenMP thread.
  void Monte_Carlo_Step(float &ang_var, gsl_rng **r);
  // Mean site energy over the non-empty sites (see geometry.h for what it counts).
  float energy_calculator();

 protected:
  int Nx, Ny, Nz;
  float *ni;
  int *pt;
  Parameters *params;
};

class thermalEvolveN : public EvolveN {
 public:
  thermalEvolveN(float *ni, int *pt, Parameters *params);
  int run();
  ~thermalEvolveN();
};

class stepEvolveN : public EvolveN {
 public:
  stepEvolveN(float *ni, int *pt, Parameters *params);
  int run();
  ~stepEvolveN();
};

class quenchEvolveN : public EvolveN {
 public:
  quenchEvolveN(float *ni, int *pt, Parameters *params);
  int run();
  ~quenchEvolveN();
};
class electricEvolveN : public EvolveN {
 public:
  electricEvolveN(float *ni, int *pt, Parameters *params);
  int run();
  ~electricEvolveN();
};

#endif
