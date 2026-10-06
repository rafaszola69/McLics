// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef SWEEP_H_
#define SWEEP_H_
#include <math.h>

// Points of a sweep from a to b with step d (temperature in thermal mode, field in electric mode), computed by
// counting instead of accumulating a += d, so rounding errors neither skip the last point nor shift the values.
// The last point equals b exactly when b lies on the grid a + n d, within 1e-3 of a step (the input values are
// floats, so (b - a) / d is only accurate to ~1e-5 steps for fine steps).

inline int sweep_points(double a, double b, double d) {
  return (int)floor((b - a) / d + 1e-3) + 1;
}

inline double sweep_value(double a, double b, double d, int n, int npoints) {
  if (n == npoints - 1 && fabs((b - a) / d - (npoints - 1)) < 1e-3) return b;
  return a + n * d;
}

// Snapshot names in thermal and electric modes: director_field_<int(S x)>.csv, where x is the temperature or the
// field. S = 100 when |step| >= 0.01 (e.g. director_field_10.csv for T = 0.1) and 1000, 10000, ... for finer steps,
// so that consecutive points never share a name. The margin 1e-3/S absorbs rounding (float) errors of x.

inline int snapshot_scale(double step) {
  int S = 100;
  while (S * fabs(step) < 1 - 1e-9 && S < 100000000) S *= 10;
  return S;
}

inline long snapshot_index(double x, int S) {
  return (long)(S * (x + 1e-3 / S));
}

#endif
