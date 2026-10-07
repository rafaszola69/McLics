// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
// Input file reading and validation, GHRL coefficients, snapshot output and terminal formatting.
#include "../include/io.h"

#include <math.h>
#include <string.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../include/define.h"
#include "../include/evolve.h"
#include "../include/ic.h"
#include "../include/parameters.h"
#include "../include/potential.h"

// ─── Helpers ────────────────────────────────────────────────────────────────

void print_separator(const char *seg, int count) {
  for (int i = 0; i < count; i++) fputs(seg, stdout);
  putchar('\n');
}

void print_header(const char *title, int count, const char *seg) {
  print_separator(seg, count);
  int titleLen = (int)strlen(title);
  int pad = (count - titleLen - 2) / 2;
  if (pad < 0) pad = 0;
  printf("%*s %s %*s\n", pad, "", title, pad, "");
  print_separator(seg, count);
}

// Standard "  Label: value" row. All setup/summary lines (geometry, anchoring,
// potential, evolve init) go through this so their columns line up regardless
// of which file printed them.
static const int kFieldWidth = 13;

void print_field(const char *label, const char *value) {
  printf("  %-*s%s\n", kFieldWidth, label, value);
}

void print_field(const char *label, double value) {
  printf("  %-*s%g\n", kFieldWidth, label, value);
}

void print_field(const char *label, int value) {
  printf("  %-*s%d\n", kFieldWidth, label, value);
}

// ─── print_n ────────────────────────────────────────────────────────────────

// Snapshot (and ic.csv) format: header x,y,z,nx,ny,nz,S,pt, then one line per grid point in index order (x fastest),
// with S = local order parameter (lattice_order_parameter) and NaN on empty sites. This file can be read back
// with ic ic_file.
int print_n(char *fname, float *ni, Parameters params, int *pt) {
  int Nx = params.Nx;
  int Ny = params.Ny;
  int Nz = params.Nz;
  float S;
  FILE *output = fopen(fname, "w");
  if (output == 0) {
    perror(fname);
    return 1;
  }
  fprintf(output, "x,y,z,nx,ny,nz,S,pt\n");

  for (int k = 0; k < params.Nz; k++) {
    for (int j = 0; j < params.Ny; j++) {
      for (int i = 0; i < params.Nx; i++) {
        // pt == 0 means the site is outside the simulated geometry (no
        // material there) - there is no order parameter to report. NAN
        // is used instead of a numeric placeholder like 0 or 1, since
        // both of those are valid physical S values and would silently
        // masquerade as real data in downstream analysis.
        S = pti(i, j, k) ? lattice_order_parameter(ni, pt, i, j, k, params) : NAN;
        fprintf(output, "%d,%d,%d,%g,%g,%g,%g,%d\n", i, j, k,
                nix(i, j, k), niy(i, j, k), niz(i, j, k), S, pti(i, j, k));
      }
    }
  }
  printf("  Snapshot -> %s\n", fname);
  fflush(stdout);
  fclose(output);
  return 0;
}
// ─── split_csv ──────────────────────────────────────────────────────────────
// Splits one line of a comma-separated file (geometry and initial-condition files) into trimmed fields.
std::vector<std::string> split_csv(const std::string &line) {
  std::vector<std::string> out;
  std::string field;
  for (size_t k = 0; k <= line.size(); k++) {
    if (k == line.size() || line[k] == ',') {
      size_t a = field.find_first_not_of(" \t\r"), b = field.find_last_not_of(" \t\r");
      out.push_back(a == std::string::npos ? "" : field.substr(a, b - a + 1));
      field.clear();
    } else {
      field += line[k];
    }
  }
  return out;
}

// ─── input helpers ──────────────────────────────────────────────────────────
// read_word: reads one whitespace-delimited value into a fixed-size char field, aborting if it does not fit.
// finish_line: discards the rest of the line (any length); text other than a '#' comment is reported, because it
// usually means a typo such as a missing value or a value with spaces.

static void read_word(std::ifstream &in, char *dest, size_t size, const char *parser) {
  std::string value;
  if (!(in >> value)) return;  // the stream error is reported by check_error_bits
  if (value.size() >= size) {
    fprintf(stderr, "  [ERROR] Value of '%s' is too long (%zu characters, maximum %zu): %.60s...\n", parser,
            value.size(), size - 1, value.c_str());
    exit(1);
  }
  strcpy(dest, value.c_str());
}

static void finish_line(std::ifstream &in, const char *parser) {
  std::string rest;
  if (in.fail()) return;  // reported by check_error_bits
  std::getline(in, rest);
  size_t start = rest.find_first_not_of(" \t\r");
  if (start != std::string::npos && rest[start] != '#')
    fprintf(stderr, "  [WARNING] Ignoring extra text after parameter '%s': %s\n", parser, rest.c_str() + start);
}

// Reads the input file: one "name value" pair per line (names are case-insensitive, several have aliases; per-surface
// parameters take the surface number first, e.g. "W 0 1.5"); '#' starts a comment. Parameters not given keep the
// defaults of include/parameters.h. Unknown names abort the run.
Parameters read_input_file(char *fname) {
  Parameters input_params;
  std::ifstream input_file;
  int nn;
  if (fname == NULL) {
    printf("  [!] No parameter file specified - using default parameter values.\n\n");
    return input_params;
  } else {
    input_file.open(fname);
    if (!input_file) {
      perror(fname);
      exit(5);
    }
    printf("  Parameter file : %s\n\n", fname);
  }
  char parser[200];
  std::string token;

  while (input_file >> token) {
    if (token.size() >= sizeof(parser)) {
      fprintf(stderr, "  [ERROR] Parameter name too long: %.60s...\n", token.c_str());
      exit(1);
    }
    strcpy(parser, token.c_str());
    if (strcasecmp(parser, "nx") == 0) {
      input_file >> input_params.Nx;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "ny") == 0) {
      input_file >> input_params.Ny;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "nz") == 0) {
      input_file >> input_params.Nz;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "A") == 0) {
      input_file >> input_params.A;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "B1") == 0) {
      input_file >> input_params.B1;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "C") == 0) {
      input_file >> input_params.C;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "Ti") == 0) {
      input_file >> input_params.Ti;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "Tf") == 0) {
      input_file >> input_params.Tf;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "dT") == 0) {
      input_file >> input_params.dT;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "MCS") == 0) {
      input_file >> input_params.MCS;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "MCT") == 0) {
      input_file >> input_params.MCT;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "MCQ") * strcasecmp(parser, "quench_steps") == 0) {
      input_file >> input_params.MCQ;
      finish_line(input_file, parser);
      if (!input_file.fail() && input_params.MCQ < 0) {
        fprintf(stderr, "  [ERROR] MCQ (quench_steps) must be >= 0 (got %d)\n", input_params.MCQ);
        exit(1);
      }
    } else if (strcasecmp(parser, "seed") == 0) {
      input_file >> input_params.seed;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "ic") == 0) {
      read_word(input_file, input_params.ic, sizeof(input_params.ic), parser);
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "ic_file") == 0) {
      read_word(input_file, input_params.ic_file, sizeof(input_params.ic_file), parser);
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "xbound") == 0) {
      read_word(input_file, input_params.XBoundtype, sizeof(input_params.XBoundtype), parser);
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "ybound") == 0) {
      read_word(input_file, input_params.YBoundtype, sizeof(input_params.YBoundtype), parser);
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "zbound") == 0) {
      read_word(input_file, input_params.ZBoundtype, sizeof(input_params.ZBoundtype), parser);
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "potential") == 0) {
      read_word(input_file, input_params.potential, sizeof(input_params.potential), parser);
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "evol")*strcasecmp(parser, "mode") == 0) {
      read_word(input_file, input_params.evol, sizeof(input_params.evol), parser);
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "k11") == 0) {
      input_file >> input_params.k11;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "k22") == 0) {
      input_file >> input_params.k22;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "k33") == 0) {
      input_file >> input_params.k33;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "p0") == 0) {
      input_file >> input_params.p0;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "phi_0") == 0) {
      input_file >> input_params.phi_0;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "theta_0") == 0) {
      input_file >> input_params.theta_0;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "p0_i") == 0) {
      input_file >> input_params.p0_i;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "fn") == 0) {
      input_file >> input_params.fn;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "first_file_number") * strcasecmp(parser, "initial_output") == 0) {
      input_file >> input_params.first_file;
      finish_line(input_file, parser);
    } else if (parser[0] == '#') {
      std::string comment;
      std::getline(input_file, comment);
    } else if (strcasecmp(parser, "geometry") == 0) {
      read_word(input_file, input_params.geometry, sizeof(input_params.geometry), parser);
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "boundary_file") == 0) {
      read_word(input_file, input_params.bound_file_name, sizeof(input_params.bound_file_name), parser);
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "anchoring_type") == 0) {
      input_file >> nn;
      input_file >> input_params.anchoring_type[nn];
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "W") == 0) {
      input_file >> nn;
      input_file >> input_params.W[nn];
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "phi_s") == 0) {
      input_file >> nn;
      input_file >> input_params.phi_s[nn];
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "theta_s") == 0) {
      input_file >> nn;
      input_file >> input_params.theta_s[nn];
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "elecX")*strcasecmp(parser, "Electric_field_x")*strcasecmp(parser, "EF_x") == 0) {
      input_file >> input_params.elecX;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "elecY")*strcasecmp(parser, "Electric_field_y")*strcasecmp(parser, "EF_y")  == 0) {
      input_file >> input_params.elecY;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "elecZ")*strcasecmp(parser, "Electric_field_z")*strcasecmp(parser, "EF_z")  == 0) {
      input_file >> input_params.elecZ;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "Ei")*strcasecmp(parser, "initial_E")  == 0) {
      input_file >> input_params.elecEi;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "Ef")*strcasecmp(parser, "final_E")  == 0) {
      input_file >> input_params.elecEf;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "dE")*strcasecmp(parser, "E_variation")  == 0) {
      input_file >> input_params.elecdE;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "elecA")*strcasecmp(parser, "dielectric_anisotropy")*strcasecmp(parser, "Aniso_E")  == 0) {
      input_file >> input_params.elecA;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "elecP")*strcasecmp(parser, "dipolar_coupling")*strcasecmp(parser, "Polar_E")  == 0) {
      input_file >> input_params.elecP;
      finish_line(input_file, parser);
    } else if (strcasecmp(parser, "radial") == 0) {
      read_word(input_file, input_params.radial, sizeof(input_params.radial), parser);
      finish_line(input_file, parser);
    } else {
      fprintf(stderr, "  [ERROR] Unknown parameter: '%s'\n", parser);
      exit(3);
    }
    check_error_bits(&input_file, parser);
  }

  return input_params;
}
// ─── print_parameters ───────────────────────────────────────────────────────

void print_parameters(Parameters params) {
  print_header("SIMULATION PARAMETERS");
  printf("  %-20s %dx%dx%d\n",  "Grid:",      params.Nx, params.Ny, params.Nz);
  printf("  %-20s %s\n",        "Geometry:",  params.geometry);
  printf("  %-20s %s\n",        "Potential:", params.potential);
  printf("  %-20s %s\n",        "Mode:",      params.evol);
  printf("  %-20s %d steps\n",  "MCS:",       params.MCS);
  printf("  %-20s %d steps\n",  "MCT:",       params.MCT);
  if (strncasecmp(params.evol, "quench", 6) == 0)
    printf("  %-20s %d steps\n",  "MCQ:",       params.MCQ);
  printf("  %-20s %d\n",        "Seed:",      params.seed);
  if (strcasecmp(params.ic, "ic_file") == 0)
    printf("  %-20s %s\n",      "IC file:",   params.ic_file);
  if (params.flipMove)
    printf("  %-20s %s\n",      "Flip move:", "on (n -> -n, polar model)");
  if (strcasecmp(params.radial, "none") != 0)
    printf("  %-20s %s\n",      "Radial obs.:", params.radial);
  if (params.elecA || params.elecP) {
    printf("\n  Electric Field\n");
    printf("  %-20s (%g, %g, %g)\n", "Direction:",  params.elecX, params.elecY, params.elecZ);
    printf("  %-20s %g\n",           "Anisotropy:", params.elecA);
    printf("  %-20s %g\n",           "Dipolar coupling:", params.elecP);
  }
  print_separator();
  printf("\n");
}
// ─── setGHRL ────────────────────────────────────────────────────────────────

// GHRL coefficients from the Frank constants (article, Eq. of the GHRL potential and the following definitions):
//   lambda = (L/9)(2 K1/K2 - 3 + K3/K2),  mu = L (1 - K1/K2),  nu = (L/9)(K1/K2 - 3 - K3/K2),
//   rho = (L/9)(K1/K2 - K3/K2),  sigma = -q0 L,  with L = Lambda* = Scale k22 and q0 = 2 pi / p0.
// Scale is chosen so that |nu| = 1, which fixes the energy (and temperature) unit as in the LL model; only the
// ratios of the k's matter. nu = -1 needs k11 < 3 k22 + k33 (checked by validate_parameters).
void setGHRL(Parameters &params) {
  float k11 = params.k11;
  float k22 = params.k22;
  float k33 = params.k33;
  float p0  = params.p0;
  float Scale = fabs((9.0) / (k11 - 3 * k22 - k33));

  params.ghrl_sigma  = p0 ? -Scale * k22 * (2 * M_PI / p0) : 0;
  params.ghrl_lambda = (Scale / 9) * (2 * k11 - 3 * k22 + k33);
  params.ghrl_mu     = Scale * (k22 - k11);
  params.ghrl_rho    = (Scale / 9) * (k11 - k33);
  params.ghrl_nu     = (Scale / 9) * (k11 - 3 * k22 - k33);

  print_header("ELASTIC PARAMETERS");
  printf("  %-12s %g\n", "k11:",    k11);
  printf("  %-12s %g\n", "k22:",    k22);
  printf("  %-12s %g\n", "k33:",    k33);
  printf("  %-12s %g\n", "lambda:", params.ghrl_lambda);
  printf("  %-12s %g\n", "mu:",     params.ghrl_mu);
  printf("  %-12s %g\n", "rho:",    params.ghrl_rho);
  printf("  %-12s %g\n", "nu:",     params.ghrl_nu);
  printf("  %-12s %g\n", "sigma:",  params.ghrl_sigma);
  printf("  %-12s %g\n", "scale:",  Scale);
  print_separator();
  printf("\n");
}

// ─── validate_parameters ────────────────────────────────────────────────────

static int n_input_errors = 0;
static void input_error(const char *msg) {
  fprintf(stderr, "  [ERROR] %s\n", msg);
  n_input_errors++;
}

void validate_parameters(const Parameters &p) {
  char msg[400];
  // evolution mode without the GPU suffix
  std::string mode(p.evol);
  if (mode.size() > 3 && strcasecmp(mode.c_str() + mode.size() - 3, "GPU") == 0) mode.resize(mode.size() - 3);
  const char *m = mode.c_str();

  if (p.Nx < 1 || p.Ny < 1 || p.Nz < 1) {
    snprintf(msg, sizeof(msg), "Grid sizes must be >= 1 (got %d x %d x %d).", p.Nx, p.Ny, p.Nz);
    input_error(msg);
  }
  if (p.MCS < 1) {
    snprintf(msg, sizeof(msg), "MCS must be >= 1 (got %d): averages are taken over the MCS steps.", p.MCS);
    input_error(msg);
  }
  if (p.MCT < 0) {
    snprintf(msg, sizeof(msg), "MCT must be >= 0 (got %d).", p.MCT);
    input_error(msg);
  }
  if ((strcasecmp(m, "step") == 0 || strcasecmp(m, "quench") == 0) && p.fn < 1) {
    snprintf(msg, sizeof(msg), "fn must be >= 1 in %s mode (got %d).", m, p.fn);
    input_error(msg);
  }
  if (p.Ti < 0) {
    snprintf(msg, sizeof(msg), "Ti must be >= 0 (got %g).", p.Ti);
    input_error(msg);
  }
  if ((strcasecmp(m, "thermal") == 0 || strcasecmp(m, "quench") == 0) && p.Tf < 0) {
    snprintf(msg, sizeof(msg), "Tf must be >= 0 (got %g).", p.Tf);
    input_error(msg);
  }
  if (strcasecmp(m, "thermal") == 0) {
    if (p.dT == 0)
      input_error("dT must be nonzero in thermal mode.");
    else if ((p.Tf - p.Ti) * p.dT < 0) {
      snprintf(msg, sizeof(msg),
               "dT = %g does not lead from Ti = %g to Tf = %g (use dT < 0 to cool, dT > 0 to heat).", p.dT, p.Ti,
               p.Tf);
      input_error(msg);
    }
  }
  if (strcasecmp(m, "electric") == 0) {
    if (p.elecdE == 0)
      input_error("dE must be nonzero in electric mode.");
    else if ((p.elecEf - p.elecEi) * p.elecdE < 0) {
      snprintf(msg, sizeof(msg), "dE = %g does not lead from Ei = %g to Ef = %g.", p.elecdE, p.elecEi, p.elecEf);
      input_error(msg);
    }
  }
  if (strcasecmp(p.potential, "ghrl") == 0 || strcasecmp(p.potential, "gruhn-hess") == 0) {
    if (p.k11 <= 0 || p.k22 <= 0 || p.k33 <= 0) {
      snprintf(msg, sizeof(msg), "k11, k22 and k33 must be > 0 (got %g, %g, %g).", p.k11, p.k22, p.k33);
      input_error(msg);
    } else if (p.k11 - 3 * p.k22 - p.k33 >= -1e-6 * p.k22) {
      // nu = (Lambda*/9)(k11/k22 - 3 - k33/k22) must be negative with Lambda* > 0 (nematic energy scale)
      snprintf(msg, sizeof(msg),
               "The GHRL potential needs k11 < 3 k22 + k33 (got k11 = %g, k22 = %g, k33 = %g): otherwise its energy "
               "scale nu is not negative and the pair potential is not nematic.", p.k11, p.k22, p.k33);
      input_error(msg);
    }
    if (p.p0 != 0 && fabs(p.p0) < 2 * M_PI)
      fprintf(stderr,
              "  [WARNING] |p0| = %g < 2 pi: the twist per lattice bond exceeds 90 degrees and the lattice has no\n"
              "            equilibrium helix (sin(phi) = 2 pi / p0 > 1).\n",
              fabs(p.p0));
  }
  if (strcasecmp(p.ic, "cholesteric") == 0 && p.p0_i == 0 && p.p0 == 0)
    input_error("ic cholesteric needs a nonzero pitch: set p0_i (or p0).");

  if (n_input_errors > 0) {
    fprintf(stderr, "  %d error(s) in the input file. Nothing was run and no file was changed.\n", n_input_errors);
    exit(1);
  }
}

// ─── check_checkerboard ─────────────────────────────────────────────────────
// The Monte Carlo sweep updates the lattice in 8 sublattices given by the parity of (i, j, k), and all sites of
// a sublattice are updated in parallel. Along a periodic direction with an odd number of sites, the sites 0 and
// N-1 are neighbours and have the same parity, so they would be updated simultaneously. Such lattices are rejected.
// Must be called after the geometry has set the boundary types (sphere and slab force free boundaries).

void check_checkerboard(const Parameters &params) {
  const char *axis[3] = {"x", "y", "z"};
  const char *type[3] = {params.XBoundtype, params.YBoundtype, params.ZBoundtype};
  const int n[3] = {params.Nx, params.Ny, params.Nz};
  for (int a = 0; a < 3; a++) {
    if (strcasecmp(type[a], "periodic") == 0 && n[a] % 2 != 0) {
      fprintf(stderr,
              "  [ERROR] Periodic boundary along %s requires an even number of sites (N%s = %d).\n"
              "          The checkerboard update needs it; use an even size or a free boundary.\n",
              axis[a], axis[a], n[a]);
      exit(2);
    }
  }
}

// ─── check_error_bits ───────────────────────────────────────────────────────

void check_error_bits(std::ifstream *f, char *parser) {
  if (f->eof()) {
    // EOF after getline() is normal, not an error condition: in case there is
    // data between the last delimiter and EOF, getline() extracts it and
    // sets the eofbit.
    return;
  }
  if (f->fail()) {
    std::cerr << "  [ERROR] Invalid value for parameter: " << parser << std::endl;
    exit(1);
  }
  if (f->bad()) {
    std::cerr << "  [ERROR] Stream error while reading parameter: " << parser << std::endl;
    exit(1);
  }
}
