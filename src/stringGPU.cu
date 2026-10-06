// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
__device__ int d_tolower(int ch){
  if ( ch > 64 && ch < 91 )
    return ch+32;
  else
    return ch;
}
__device__ int d_strcmp(const char *str_a, const char *str_b, unsigned len = 256){
  int match = 0;
  unsigned i = 0;
  unsigned done = 0;
  while ((i < len) && (match == 0) && !done){
    if (str_a[i] == 0) if (str_b[i] == 0) done = 1; else return 1;
    else if ( d_tolower(str_a[i]) != d_tolower(str_b[i])){
      match = i+1;
      if (((int)str_a[i] - (int)str_b[i]) < 0) 
        match = 0 - (i + 1);}
    i++;}
  return match;
}
