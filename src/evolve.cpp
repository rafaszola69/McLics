// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#include "../include/evolve.h"

#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <math.h>
#include <omp.h>

#include <iostream>

#include "../include/define.h"
#include "../include/geometry.h"
#include "../include/io.h"
#include "../include/parameter_order.h"
#include "../include/parameters.h"
#include "../include/potential.h"

// CPU Monte Carlo step and energy. The GPU versions are in evolve.cu.

// Copies site pos into nLocal, or marks it as absent (pt = 0) when the neighbour does not exist (free boundary).
void inline setNni(uint pos, int conditional, nni *nLocal, float *ni, int *pt) {
  if (conditional == 1) {
    nLocal->x = ni[pos * 3 + 0];
    nLocal->y = ni[pos * 3 + 1];
    nLocal->z = ni[pos * 3 + 2];
    nLocal->pt = pt[pos];
  } else {
    nLocal->pt = 0;
  }
}
float EvolveN::energy_calculator() {
  // The static variables below are set up on the first call and reused: McLics runs one simulation per process.
  static const int Nt = params->Nx * params->Ny * params->Nz;
  static int nValid = -1;
  if (nValid < 0) {
    nValid = 0;
    for (int idx = 0; idx < Nt; idx++) nValid += pt[idx] ? 1 : 0;
  }

  double Etot = 0;
#pragma omp parallel for simd reduction(+ : Etot) schedule(simd : static) num_threads(omp_get_max_threads())
  for (int idx = 0; idx < Nt; idx++) {
    if (pt[idx] == 0) continue;
    nni nLocal[8];

    int i = idx % Nx;
    int j = (idx / Nx) % Ny;
    int k = idx / (Nx * Ny);

    int im = (i - 1), ip = (i + 1);
    int jm = (j - 1), jp = (j + 1);
    int km = (k - 1), kp = (k + 1);
    int neighbour[6] = {
        params->XBound(ip, Nx), params->XBound(im, Nx),
        params->YBound(jp, Ny), params->YBound(jm, Ny),
        params->ZBound(kp, Nz), params->ZBound(km, Nz)};
    setNni(i + Nx * (j + Ny * k ), 1, &nLocal[0], ni, pt);
    setNni(ip+ Nx * (j + Ny * k ), neighbour[0], &nLocal[1], ni, pt);
    setNni(im+ Nx * (j + Ny * k ), neighbour[1], &nLocal[2], ni, pt);
    setNni(i + Nx * (jp+ Ny * k ), neighbour[2], &nLocal[3], ni, pt);
    setNni(i + Nx * (jm+ Ny * k ), neighbour[3], &nLocal[4], ni, pt);
    setNni(i + Nx * (j + Ny * kp), neighbour[4], &nLocal[5], ni, pt);
    setNni(i + Nx * (j + Ny * km), neighbour[5], &nLocal[6], ni, pt);
    if (nLocal[0].pt > 1) {
      nLocal[7].x = geometry->ns[(i + Nx * (j + Ny * k)) * 3 + 0];
      nLocal[7].y = geometry->ns[(i + Nx * (j + Ny * k)) * 3 + 1];
      nLocal[7].z = geometry->ns[(i + Nx * (j + Ny * k)) * 3 + 2];
      nLocal[7].pt = 1;
    }
    Etot += geometry->site_energy(nLocal);
  }
  return Etot / nValid;
}

// Metropolis sweep with a checkerboard update. The lattice is divided into 8 sublattices by the parity of
// (i, j, k); sites of one sublattice have no common bonds, so they are updated in parallel, and the 8 sublattices
// are visited in turn (tick = 0..7). For each site:
//   1. trial rotation of n by the angle pi*va, va uniform in (-ang_var, ang_var), about x, y or z (chosen at random);
//   2. accept with probability min(1, exp(-dE/T)), dE from the site energy (geometry->site_energy);
//   3. for polar models (flipMove), a second trial n -> -n with the same rule.
// After the sweep, ang_var is multiplied (divided) by 0.99 when the acceptance of step 1 is below (above) 50%,
// in all phases of the run, including the measurement steps.
void EvolveN::Monte_Carlo_Step(float &ang_var, gsl_rng **r) {
  // The static variables below are set up on the first call and reused: McLics runs one simulation per process.
  float T = params->T;
  static int Nt = params->Nx * params->Ny * params->Nz;
  static int nValid = 0;
  int acceptance;
  acceptance = 0;
  int iBoxSize = 2;
  int jBoxSize = 2;
  int kBoxSize = 2;
  int nBoxX = Nx / 2 + Nx % 2;
  int nBoxY = Ny / 2 + Ny % 2;
  int nBoxZ = Nz / 2 + Nz % 2;
  int Ntt = nBoxX * nBoxY * nBoxZ;
  static bool setUp = true;
  if (setUp) {
    for (int ii = 0; ii < Nt; ii++)
      if (pt[ii])
        nValid++;
    setUp = false;
  }
#pragma omp parallel num_threads(omp_get_max_threads())
  for (int tick = 0; tick < 8; tick++) {
#pragma omp for simd reduction(+ : acceptance) schedule(simd : static)
    for (int nt = 0; nt < Ntt; nt++) {
      float E_new, E_old, rotation_type, va;
      int i, j, k;
      float nNew[3];
      int thread = omp_get_thread_num();
      int iBox = nt % nBoxX;
      int jBox = (nt / nBoxX) % nBoxY;
      int kBox = nt / nBoxX / nBoxY;
      int di = tick % 2;
      int dj = (tick / 2) % 2;
      int dk = tick / 4;
      nni nLocal[8];

      i = di + iBox * iBoxSize;
      if (i >= Nx)
        continue;
      j = dj + jBox * jBoxSize;
      if (j >= Ny)
        continue;
      k = dk + kBox * kBoxSize;
      if (k >= Nz)
        continue;
      if (pti(i, j, k) == 0)
        continue;
      int im = (i - 1), ip = (i + 1);
      int jm = (j - 1), jp = (j + 1);
      int km = (k - 1), kp = (k + 1);
      int neighbour[6] = {
          params->XBound(ip, Nx), params->XBound(im, Nx),
          params->YBound(jp, Ny), params->YBound(jm, Ny),
          params->ZBound(kp, Nz), params->ZBound(km, Nz)};
      setNni(i + Nx * (j + Ny * k ), 1, &nLocal[0], ni, pt);
      setNni(ip+ Nx * (j + Ny * k ), neighbour[0], &nLocal[1], ni, pt);
      setNni(im+ Nx * (j + Ny * k ), neighbour[1], &nLocal[2], ni, pt);
      setNni(i + Nx * (jp+ Ny * k ), neighbour[2], &nLocal[3], ni, pt);
      setNni(i + Nx * (jm+ Ny * k ), neighbour[3], &nLocal[4], ni, pt);
      setNni(i + Nx * (j + Ny * kp), neighbour[4], &nLocal[5], ni, pt);
      setNni(i + Nx * (j + Ny * km), neighbour[5], &nLocal[6], ni, pt);
      if (nLocal[0].pt > 1) {
        nLocal[7].x = geometry->ns[(i + Nx * (j + Ny * k)) * 3 + 0];
        nLocal[7].y = geometry->ns[(i + Nx * (j + Ny * k)) * 3 + 1];
        nLocal[7].z = geometry->ns[(i + Nx * (j + Ny * k)) * 3 + 2];
        nLocal[7].pt = 1;
      }
      va = (2 * gsl_rng_uniform(r[thread]) - 1) * ang_var;
      rotation_type = gsl_rng_uniform(r[thread]);
      {
        if (rotation_type < 0.333) {
          nNew[0] = nix(i, j, k);
          nNew[1] = niy(i, j, k) * cosf(M_PI * va) + niz(i, j, k) * sinf(M_PI * va);
          nNew[2] = niz(i, j, k) * cosf(M_PI * va) - niy(i, j, k) * sinf(M_PI * va);
        } else if (rotation_type < 0.666) {
          nNew[0] = nix(i, j, k) * cosf(M_PI * va) - niz(i, j, k) * sinf(M_PI * va);
          nNew[1] = niy(i, j, k);
          nNew[2] = niz(i, j, k) * cosf(M_PI * va) + nix(i, j, k) * sinf(M_PI * va);
        } else {
          nNew[0] = nix(i, j, k) * cosf(M_PI * va) + niy(i, j, k) * sinf(M_PI * va);
          nNew[1] = niy(i, j, k) * cosf(M_PI * va) - nix(i, j, k) * sinf(M_PI * va);
          nNew[2] = niz(i, j, k);
        }
      }
      // Renormalize so that rounding errors of successive rotations do not accumulate (as in the GPU kernel).
      {
        float norm = sqrtf(nNew[0] * nNew[0] + nNew[1] * nNew[1] + nNew[2] * nNew[2]);
        nNew[0] /= norm;
        nNew[1] /= norm;
        nNew[2] /= norm;
      }

      E_old = geometry->site_energy(nLocal);
      nLocal[0].x = nNew[0];
      nLocal[0].y = nNew[1];
      nLocal[0].z = nNew[2];
      E_new = geometry->site_energy(nLocal);

      float E_cur = E_old;
      // Metropolis; T = 0 is its T -> 0+ limit (accept if the energy does not increase). The random number is
      // always drawn, so the random sequence for T > 0 does not depend on this branch.
      float u = gsl_rng_uniform(r[thread]);
      if (T > 0 ? u < exp(-(E_new - E_old) / T) : E_new <= E_old) {
        nix(i, j, k) = nNew[0];
        niy(i, j, k) = nNew[1];
        niz(i, j, k) = nNew[2];
        E_cur = E_new;
        acceptance++;
      }
      // Separate n -> -n trial; kept out of `acceptance` so it does not bias the ang_var adaptation.
      if (params->flipMove) {
        nLocal[0].x = -nix(i, j, k);
        nLocal[0].y = -niy(i, j, k);
        nLocal[0].z = -niz(i, j, k);
        float E_flip = geometry->site_energy(nLocal);
        float uf = gsl_rng_uniform(r[thread]);
        if (T > 0 ? uf < exp(-(E_flip - E_cur) / T) : E_flip <= E_cur) {
          nix(i, j, k) = nLocal[0].x;
          niy(i, j, k) = nLocal[0].y;
          niz(i, j, k) = nLocal[0].z;
        }
      }
    }
  }

  if (1.0 * acceptance / nValid < 0.5) {
    ang_var *= 0.99;
    if (ang_var < 0.01)
      ang_var = 0.01;
  } else if (1.0 * acceptance / nValid > 0.5) {
    ang_var /= 0.99;
    if (ang_var > 1.0) {
      ang_var -= 0.5;
    }
  }
}
