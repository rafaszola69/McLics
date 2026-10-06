// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#include "../include/parameter_order.h"

#include <gsl/gsl_eigen.h>
#include <math.h>
#include <strings.h>

#include <iostream>

#include "../include/define.h"
#include "../include/parameters.h"

// Order parameters written to po.dat and to the snapshots.

// Largest eigenvalue of the symmetric 3x3 matrix mat (row-major); vec receives its eigenvector. Applied to Q, the
// eigenvalue is the nematic order parameter S and the eigenvector the director.
float Eigen_value_evaluation(float *mat, float *vec) {
  // The static variables below are set up on the first call and reused: McLics runs one simulation per process.
  static gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(3);
  static gsl_vector *eval = gsl_vector_alloc(3);
  static gsl_matrix *evec = gsl_matrix_alloc(3, 3);
  static gsl_matrix *m = gsl_matrix_alloc(3, 3);
  float temp;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      temp = mat[3 * i + j];
      gsl_matrix_set(m, j, i, temp);
    }
  }

  gsl_eigen_symmv(m, eval, evec, w);
  gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_VAL_DESC);
  for (int i = 0; i < 3; i++) vec[i] = gsl_matrix_get(evec, i, 0);
  return (gsl_vector_get(eval, 0));
}

// Q tensor averaged over the non-empty sites: Q = < (3 n n - I) / 2 > (row-major, traceless).
void Q_tensor(float *ni, float *Q, int *pt, Parameters params) {
  // The static variables below are set up on the first call and reused: McLics runs one simulation per process.
  static const int Nx = params.Nx;
  static const int Ny = params.Ny;
  static const int Nz = params.Nz;
  int points = 0;
  for (int i = 0; i < 9; i++) Q[i] = 0;
  for (int i = 0; i < Nx; i++) {
    for (int j = 0; j < Ny; j++) {
      for (int k = 0; k < Nz; k++) {
        if (pti(i, j, k)) {
          Q[0] += 3.0 * (nix(i, j, k) * nix(i, j, k)) - 1;
          Q[1] += 3.0 * (nix(i, j, k) * niy(i, j, k));
          Q[2] += 3.0 * (nix(i, j, k) * niz(i, j, k));
          Q[4] += 3.0 * (niy(i, j, k) * niy(i, j, k)) - 1;
          Q[5] += 3.0 * (niy(i, j, k) * niz(i, j, k));
          points++;
        }
      }
    }
  }

  Q[0] /= 2.0 * points;
  Q[1] /= 2.0 * points;
  Q[2] /= 2.0 * points;
  Q[4] /= 2.0 * points;
  Q[5] /= 2.0 * points;
  Q[3] = Q[1];
  Q[6] = Q[2];
  Q[7] = Q[5];
  Q[8] = -Q[0] - Q[4];
}

// Local order parameter S of site (i, j, k), written in the snapshots: largest eigenvalue of Q averaged over the site
// and its existing non-empty nearest neighbours (up to 7 sites).
float lattice_order_parameter(float *ni, int *pt, int i, int j, int k, Parameters params) {
  // The static variables below are set up on the first call and reused: McLics runs one simulation per process.
  static int Nx = params.Nx;
  static int Ny = params.Ny;
  static int Nz = params.Nz;
  int points = 0;
  float Q[9];
  for (int i = 0; i < 9; i++) Q[i] = 0;

  for (int di = -1; di < 2; di++) {
    for (int dj = -1; dj < 2; dj++) {
      for (int dk = -1; dk < 2; dk++) {
        if (abs(di) + abs(dj) + abs(dk) > 1) continue;
        int ii = i + di;
        int jj = j + dj;
        int kk = k + dk;
        if ((params.XBound(ii, Nx)) && (params.YBound(jj, Ny)) && (params.ZBound(kk, Nz)) && pti(ii, jj, kk) != 0) {
          for (int l = 0; l < 3; l++) {
            for (int m = 0; m <= l; m++) {
              Q[3 * m + l] += 3.0 * (ni[(ii + Nx * (jj + Ny * kk)) * 3 + l] * ni[(ii + Nx * (jj + Ny * kk)) * 3 + m]);
            }
            Q[4 * l] -= 1;
          }
          points++;
        }
      }
    }
  }

  Q[0] /= 2.0 * points;
  Q[1] /= 2.0 * points;
  Q[2] /= 2.0 * points;
  Q[4] /= 2.0 * points;
  Q[5] /= 2.0 * points;
  Q[8] /= 2.0 * points;
  Q[3] = Q[1];
  Q[6] = Q[2];
  Q[7] = Q[5];

  static gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(3);
  static gsl_vector *eval = gsl_vector_alloc(3);
  static gsl_matrix *evec = gsl_matrix_alloc(3, 3);
  static gsl_matrix *m = gsl_matrix_alloc(3, 3);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      gsl_matrix_set(m, j, i, Q[3 * i + j]);
    }
  }

  gsl_eigen_symmv(m, eval, evec, w);
  gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_VAL_DESC);
  return (gsl_vector_get(eval, 0));
}

// Dhakal & Selinger, PRE 81, 031704, Eq. (8): P1 = <d.n_i> over valid sites,
// d = director (leading eigenvector of Q). d is only defined up to sign.
float Polarization(float *ni, int *pt, float *director, Parameters params) {
  const int Nt = params.Nx * params.Ny * params.Nz;
  double sum = 0;
  int valid = 0;
  for (int idx = 0; idx < Nt; idx++) {
    if (!pt[idx]) continue;
    sum += director[0] * ni[3 * idx] + director[1] * ni[3 * idx + 1] + director[2] * ni[3 * idx + 2];
    valid++;
  }
  return valid ? fabs(sum) / valid : 0;
}

// <Delta theta>: magnitude of the lattice splay vector n(div n), Eq. (3), averaged
// as a vector over bonds. The paper averages over the four bonds perpendicular to
// the director (z there); bonds along the director carry no splay, so
// 1.5 x (mean over all six bond directions) reproduces it for any orientation.
float Splay_magnitude(float *ni, int *pt, Parameters params) {
  const int Nx = params.Nx, Ny = params.Ny, Nz = params.Nz;
  const int dirs[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  double s[3] = {0, 0, 0};
  long nBonds = 0;
  for (int k = 0; k < Nz; k++) {
    for (int j = 0; j < Ny; j++) {
      for (int i = 0; i < Nx; i++) {
        const int idx = i + Nx * (j + Ny * k);
        if (!pt[idx]) continue;
        const float *a = &ni[3 * idx];
        for (int d = 0; d < 6; d++) {
          int ii = i + dirs[d][0], jj = j + dirs[d][1], kk = k + dirs[d][2];
          if (!(params.XBound(ii, Nx) && params.YBound(jj, Ny) && params.ZBound(kk, Nz))) continue;
          const int jdx = ii + Nx * (jj + Ny * kk);
          if (!pt[jdx]) continue;
          const float *b = &ni[3 * jdx];
          const float ra = a[0] * dirs[d][0] + a[1] * dirs[d][1] + a[2] * dirs[d][2];
          const float rb = b[0] * dirs[d][0] + b[1] * dirs[d][1] + b[2] * dirs[d][2];
          const float ab = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
          for (int c = 0; c < 3; c++)
            s[c] += 0.5 * (b[c] * rb - a[c] * ra + a[c] * ab * rb - b[c] * ab * ra);
          nBonds++;
        }
      }
    }
  }
  if (nBonds == 0) return 0;
  return 1.5 * sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]) / nBonds;
}

// Signed mean splay <div n> over valid sites, in units of 1/(lattice spacing).
// Per axis: central difference in the interior, one-sided where a neighbour is
// missing (surface, pt == 0); an axis with no valid neighbour contributes nothing.
// Reference: radial outward field in a sphere of radius R gives ~ +3/R, inward ~ -3/R.
// In a fully periodic box it telescopes to exactly 0.
float Splay_divergence(float *ni, int *pt, const Parameters &params) {
  const int Nx = params.Nx, Ny = params.Ny, Nz = params.Nz;
  double sum = 0;
  long valid = 0;
  for (int k = 0; k < Nz; k++) {
    for (int j = 0; j < Ny; j++) {
      for (int i = 0; i < Nx; i++) {
        const int idx = i + Nx * (j + Ny * k);
        if (!pt[idx]) continue;
        double div = 0;
        for (int ax = 0; ax < 3; ax++) {
          double d = 0;
          int nb = 0;
          for (int s = -1; s <= 1; s += 2) {
            int ii = i, jj = j, kk = k;
            if (ax == 0) ii += s;
            else if (ax == 1) jj += s;
            else kk += s;
            if (!(params.XBound(ii, Nx) && params.YBound(jj, Ny) && params.ZBound(kk, Nz))) continue;
            const int jdx = ii + Nx * (jj + Ny * kk);
            if (!pt[jdx]) continue;
            d += s * (ni[3 * jdx + ax] - ni[3 * idx + ax]);
            nb++;
          }
          if (nb) div += d / nb;
        }
        sum += div;
        valid++;
      }
    }
  }
  return valid ? sum / valid : 0;
}

// Radial order about the grid centre (Nx/2, Ny/2, Nz/2), the same centre as the
// sphere geometry. With cylinder = true, r_hat is the in-plane direction from the
// z axis through that centre. Sites within half a spacing of the centre/axis are
// skipped. c = n.r_hat:
//   Sr = <(3c^2 - 1)/2>  apolar: 1 radial, 0 isotropic, -1/2 tangential
//   Pr = <c>             signed polar: +1 outward, -1 inward
void Radial_order(float *ni, int *pt, const Parameters &params, bool cylinder, double &Sr, double &Pr) {
  const int Nx = params.Nx, Ny = params.Ny, Nz = params.Nz;
  const float cx = Nx / 2, cy = Ny / 2, cz = Nz / 2;
  double s = 0, p = 0;
  long valid = 0;
  for (int k = 0; k < Nz; k++) {
    for (int j = 0; j < Ny; j++) {
      for (int i = 0; i < Nx; i++) {
        const int idx = i + Nx * (j + Ny * k);
        if (!pt[idx]) continue;
        const float rx = i - cx, ry = j - cy, rz = cylinder ? 0 : k - cz;
        const float r = sqrtf(rx * rx + ry * ry + rz * rz);
        if (r < 0.5f) continue;
        const double c = (ni[3 * idx] * rx + ni[3 * idx + 1] * ry + ni[3 * idx + 2] * rz) / r;
        s += 1.5 * c * c - 0.5;
        p += c;
        valid++;
      }
    }
  }
  Sr = valid ? s / valid : 0;
  Pr = valid ? p / valid : 0;
}

// Optional po.dat columns (see parameter_order.h): accumulate() adds one sample per measurement sweep and
// write_values() writes the averages.
ExtraObservables::ExtraObservables(const Parameters *params) : params(params) {
  pear = strcasecmp(params->potential, "pear") == 0;
  sph = strcasecmp(params->radial, "sphere") == 0 || strcasecmp(params->radial, "both") == 0;
  cyl = strcasecmp(params->radial, "cylinder") == 0 || strcasecmp(params->radial, "both") == 0;
  reset();
}

void ExtraObservables::reset() { P = dTheta = divN = SrSph = PrSph = SrCyl = PrCyl = 0; }

void ExtraObservables::write_header(FILE *f) const {
  if (pear) fprintf(f, " P dtheta divn");
  if (sph) fprintf(f, pear ? " Sr_sph Pr_sph" : " Sr_sph");
  if (cyl) fprintf(f, pear ? " Sr_cyl Pr_cyl" : " Sr_cyl");
  fprintf(f, "\n");
}

void ExtraObservables::accumulate(float *ni, int *pt, float *director) {
  double s, p;
  if (pear) {
    P += Polarization(ni, pt, director, *params);
    dTheta += Splay_magnitude(ni, pt, *params);
    divN += Splay_divergence(ni, pt, *params);
  }
  if (sph) {
    Radial_order(ni, pt, *params, false, s, p);
    SrSph += s;
    PrSph += p;
  }
  if (cyl) {
    Radial_order(ni, pt, *params, true, s, p);
    SrCyl += s;
    PrCyl += p;
  }
}

void ExtraObservables::write_values(FILE *f, int n) const {
  if (pear) fprintf(f, " %g %g %g", P / n, dTheta / n, divN / n);
  if (sph) {
    fprintf(f, " %g", SrSph / n);
    if (pear) fprintf(f, " %g", PrSph / n);
  }
  if (cyl) {
    fprintf(f, " %g", SrCyl / n);
    if (pear) fprintf(f, " %g", PrCyl / n);
  }
  fprintf(f, "\n");
}
