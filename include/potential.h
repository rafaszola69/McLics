// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef POTENTIAL_H_
#define POTENTIAL_H_
#include <iostream>

#include "../include/define.h"
#include "../include/parameters.h"

// Boundary conditions along one axis: given a neighbour index ii (which may be -1 or NN), return 1 if that
// neighbour exists (wrapping ii into [0, NN) for periodic boundaries) and 0 if it does not (free boundaries).
int Periodic_Boundary(int &ii, int NN);
int Free_Boundary(int &ii, int NN);

// Pair (bulk) potentials U(n_i, n_j, r_ij) between nearest neighbours, in units of the coupling (energy scale of
// the temperature). rij is the unit vector from site i to site j. The same formulas are in potential.cu (GPU).
float Bulk_Energy_Selinger_Pear(float ni[3], float nj[3], Parameters *params, float rij[3]);
float Bulk_Energy_Lebwohl_Lasher(float ni[3], float nj[3], Parameters *params, float rij[3]);
float Bulk_Energy_GHRL(float ni[3], float nj[3], Parameters *params, float rij[3]);
float Electric_Potential(float ni[3], Parameters *params);
#endif
