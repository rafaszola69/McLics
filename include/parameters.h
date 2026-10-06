// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef PARAMETERS_H_
#define PARAMETERS_H_
#include <gsl/gsl_rng.h>

#include <map>
#include <string>

enum parameter_status { set,
                        unset };

// All run parameters, with their defaults. read_input_file (io.cpp) fills them from the input file (the input names
// are listed there and in the README); main.cpp derives flipMove, MCQ and radial, normalizes the field direction
// and calls validate_parameters. Angles are in degrees, lengths in lattice spacings, energies and temperatures in
// units of the pair coupling.
struct Parameters {
  //lattice Variables
  char geometry[20] = "bulk";
  int Nx = 16;
  int Ny = 16;
  int Nz = 16;
  char XBoundtype[20] = "free";
  char YBoundtype[20] = "free";
  char ZBoundtype[20] = "free";
  // boundary functions (Periodic_Boundary / Free_Boundary), set from the types above during the setup
  int (*XBound)(int&, int);
  int (*YBound)(int&, int);
  int (*ZBound)(int&, int);

  //Surface Variables: geometry file (custom) and, per surface number, anchoring type, strength and easy axis
  char bound_file_name[1024];
  std::map<int, float> theta_s;
  std::map<int, float> phi_s;
  std::map<int, float> W;
  std::map<int, std::string> anchoring_type;

  //potential Variables
  char potential[20] = "ll";
  float A = 1.0;
  float B1 = 0.04;
  float C = 0.3;
  float Ti = 1.2;
  float Tf = 0.0;
  float dT = -0.02;
  float T;  // current temperature (set by the evolution loop)
  // GHRL coefficients, computed from k11, k22, k33 and p0 by setGHRL (io.cpp)
  float ghrl_rho = 0;
  float ghrl_lambda = 0;
  float ghrl_mu = 0;
  float ghrl_nu = 1;
  float ghrl_sigma = 0;

  //MC Variables
  char evol[20] = "thermal";
  int MCS = 1e2;  // measurement sweeps per point
  int MCT = 1e2;  // thermalization sweeps per point
  // quench mode: steps at Tf before measuring; -1 = not given, replaced by MCT in main
  int MCQ = -1;
  int fn = 1;          // step/quench: number of blocks (one po.dat line and snapshot each)
  int first_file = 0;  // step/quench: number of the first block
  // RNG seed. Independent replicas (same params, different seed) sample
  // different equally-valid defect textures in degenerate geometries
  // (e.g. droplets with homeotropic anchoring) — average several to get
  // a stable order-parameter estimate instead of relying on a single run.
  int seed = 1;

  //IC variables
  char ic[50] = "random";
  char ic_file[1024];
  float phi_0 = 0;
  float theta_0 = 0;
  float p0_i = 0;  // pitch of the cholesteric initial condition (0: use p0)
  // GHRL elastic constants and natural pitch p0 (lattice spacings; 0 = no chirality)
  float k11 = 1;
  float k22 = 1;
  float k33 = 1;
  float p0 = 0;

  //Neighbourhood: only nearest neighbours (nk = 1) are supported
  int neighbourKind = 1;

  //Electric Field Variables
  float elecX  = 0;
  float elecY  = 0;
  float elecZ  = 0;
  float elecA  = 0;  
  float elecP  = 0;
  // n -> -n Metropolis move; set in main for polar models (pear or elecP != 0)
  int flipMove = 0;
  // Radial observables in po.dat: sphere | cylinder | both | none ("" = default, resolved in main)
  char radial[20] = "";
  // electric mode: sweep Ei -> Ef with step dE; elecE is the current magnitude (1 in the other modes)
  float elecEi = 0;
  float elecEf = 0;
  float elecdE = 0;
  float elecE  = 1;

};

#endif
