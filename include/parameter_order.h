// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef PARAMETER_ORDER_H_
#define PARAMETER_ORDER_H_
#include <gsl/gsl_eigen.h>

#include <iostream>

#include "../include/define.h"
#include "../include/parameter_order.h"
#include "../include/parameters.h"
float Eigen_value_evaluation(float *mat, float *vec);
void Q_tensor(float *ni, float *Q, int *pt, Parameters params);
float Polarization(float *ni, int *pt, float *director, Parameters params);
float Splay_magnitude(float *ni, int *pt, Parameters params);
float Splay_divergence(float *ni, int *pt, const Parameters &params);
void Radial_order(float *ni, int *pt, const Parameters &params, bool cylinder, double &Sr, double &Pr);

// Optional po.dat columns, appended after "T S varS E varE" in this order:
//   pear only           : P dtheta divn
//   radial sphere/both  : Sr_sph [Pr_sph if pear]
//   radial cylinder/both: Sr_cyl [Pr_cyl if pear]
// Meaning of each column: README.md, section po.dat.
class ExtraObservables {
 public:
  explicit ExtraObservables(const Parameters *params);
  void write_header(FILE *f) const;
  void reset();
  void accumulate(float *ni, int *pt, float *director);
  void write_values(FILE *f, int nSamples) const;

 private:
  const Parameters *params;
  bool pear, sph, cyl;
  double P, dTheta, divN, SrSph, PrSph, SrCyl, PrCyl;
};
float lattice_order_parameter(float *ni, int *pt, int i, int j, int k, Parameters params);

#endif
