// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#include "../include/potential.h"

#include <math.h>

#include <iostream>

#include "../include/define.h"
#include "../include/parameters.h"

// Bulk pair potentials, field coupling and boundary functions (CPU). src/potential.cu has the GPU copies, which
// must be kept identical.

// Pear-shaped molecules, Dhakal & Selinger, PRE 81, 031704 (2010):
//   U = -A (n_i.n_j)^2 - B1 (n_i.n_j) - C ((1 + n_i.n_j)/2)^2 r_ij.(n_j - n_i)
// n is a polar vector here (n and -n are different states); the C term couples polarity to splay.
float Bulk_Energy_Selinger_Pear(float ni[3], float nj[3], Parameters *params, float rij[3]) {
  float nij, polar_splay;
  nij = ni[0] * nj[0] + ni[1] * nj[1] + ni[2] * nj[2];
  polar_splay = +(1 + 2 * nij + nij * nij) * ((nj[0] - ni[0]) * rij[0] + (nj[1] - ni[1]) * rij[1] + (nj[2] - ni[2]) * rij[2]);
  return (-params->A * nij * nij - params->B1 * nij - (params->C * 0.25) * polar_splay);
}

// Lebwohl-Lasher: U = -A P2(n_i.n_j) up to a constant, i.e. -1.5 A (n_i.n_j)^2. With A = 1 the reduced temperature
// is the usual one (T_NI = 1.1232); note that the pear potential uses -A (n_i.n_j)^2 without the factor 1.5.
float Bulk_Energy_Lebwohl_Lasher(float ni[3], float nj[3], Parameters *params, float rij[3]) {
  float nij = ni[0] * nj[0] + ni[1] * nj[1] + ni[2] * nj[2];
  return -params->A * 1.5 * nij * nij;
}

// Gruhn-Hess / Romano (GHRL) potential, whose continuum limit has the Frank constants K1, K2, K3 and pitch p0:
//   U = lambda Y + mu (a_i a_j b - 1/9) + nu P2(b) + rho Y P2(b) + sigma sgn(b) c,
// with a_i = n_i.r, a_j = n_j.r, b = n_i.n_j, Y = P2(a_i) + P2(a_j) and c = n_i.(n_j x r) for r taken from j to i
// (the article's r_jk); here rij points from i to j, so `cross` = r_ij.(n_j x n_i) is the same c.
// The coefficients are computed from k11, k22, k33 and p0 in setGHRL (io.cpp), with |nu| = 1 as the
// energy scale (nu = -1 whenever k11 < 3 k22 + k33, as for real nematics).
// sgn(b) makes the chiral term invariant under n -> -n.
float Bulk_Energy_GHRL(float ni[3], float nj[3], Parameters *params, float rij[3]) {
  const float el = params->ghrl_lambda;
  const float em = params->ghrl_mu;
  const float en = params->ghrl_nu;
  const float er = params->ghrl_rho;
  const float es = params->ghrl_sigma;
  float ai = ni[0] * rij[0] + ni[1] * rij[1] + ni[2] * rij[2];
  float aj = nj[0] * rij[0] + nj[1] * rij[1] + nj[2] * rij[2];
  float nij = ni[0] * nj[0] + ni[1] * nj[1] + ni[2] * nj[2];
  float pij = ((float)1.5) * nij * nij - ((float)0.5);
  float cross = (ni[2] * nj[1] - ni[1] * nj[2]) * rij[0] 
              + (ni[0] * nj[2] - ni[2] * nj[0]) * rij[1] 
              + (ni[1] * nj[0] - ni[0] * nj[1]) * rij[2];

  float E1 = ((1.5 * ai * ai) + (1.5 * aj * aj) - 1);
  return ((E1 * (er * pij + el) + em * (ai * aj * nij - 1.0 / 9)) + en * pij + es * (nij > 0 ? 1 : -1) * cross);
}
// Coupling of one site to the applied field E = elecE (elecX, elecY, elecZ), with a unit direction (normalized in
// main) and elecE = 1 outside the electric mode: -elecP (n.E) (dipolar, polar) - elecA (n.E)^2 (dielectric).
float Electric_Potential(float ni[3], Parameters *params){
  float nDotE = params->elecE*(ni[0]*params->elecX+ni[1]*params->elecY+ni[2]*params->elecZ);

  return -params->elecP*nDotE - params->elecA*nDotE*nDotE;
}
int Periodic_Boundary(int &ii, int NN) {
  ii = ((ii + NN) % NN);
  return 1;
}

int Free_Boundary(int &ii, int NN) {
  if (ii == -1 || ii == NN)
    return 0;
  else
    return 1;
}
