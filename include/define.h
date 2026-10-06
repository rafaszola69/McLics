// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef DEFINE_H_
#define DEFINE_H_

#include "../include/parameters.h"

// Lattice storage. Site (i, j, k), 0 <= i < Nx etc., has the linear index idx = i + Nx * (j + Ny * k) (x runs
// fastest). The director field ni holds 3 floats per site (ni[3 idx + 0..2]); pt holds the point type of each site:
//   pt = 0  empty (outside the sample: no director, no energy, skipped everywhere),
//   pt = 1  bulk,
//   pt = n  (n >= 2) site of surface n - 2, whose anchoring is `anchoring_type n-2` in the input file.
// Surface sites also have an outward unit normal in Geometry::ns (same layout as ni).
// The macros below need ni, pt, Nx and Ny in scope.
#define pti(i, j, k) pt[(i + Nx * (j + Ny * k))]
#define nix(i, j, k) ni[(i + Nx * (j + Ny * k)) * 3 + 0]
#define niy(i, j, k) ni[(i + Nx * (j + Ny * k)) * 3 + 1]
#define niz(i, j, k) ni[(i + Nx * (j + Ny * k)) * 3 + 2]

// GPU surface kernel: energy per unit anchoring strength W (the caller multiplies by W).
typedef float (*p_Surface_Potential)(float ni[3], float s[3], Parameters *params, float rij[3]);

// Local environment of a site, as passed to the energy routines (nni fullni[8]):
//   [0] the site itself, [1] +x, [2] -x, [3] +y, [4] -y, [5] +z, [6] -z neighbours, [7] outward surface normal
//   (only filled when fullni[0].pt >= 2). A neighbour with pt = 0 (empty site, or missing across a free
//   boundary) does not interact.
struct nni {
  float x;
  float y;
  float z;
  int pt;
};

#endif
