// McLics - Monte Carlo Liquid Crystal Simulator
// Copyright (C) 2026 The McLics authors (see AUTHORS)
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef IO_H_
#define IO_H_
#include "../include/define.h"
#include "../include/parameters.h"

#include <string>
#include <vector>

int print_n(char *fname, float *ni, Parameters params, int *pt);

void setGHRL(Parameters &params);
Parameters read_input_file(char *fname);
void print_parameters(Parameters params);
void check_error_bits(std::ifstream *f, char *parser);
// Splits one line of a comma-separated file into trimmed fields.
std::vector<std::string> split_csv(const std::string &line);
// Aborts if a periodic direction has an odd number of sites (see io.cpp).
// validate_parameters: aborts on impossible values (sizes, steps, temperatures, sweep directions, elastic
// constants, cholesteric pitch) and warns on suspicious ones; called right after reading the input.
void validate_parameters(const Parameters &params);
void check_checkerboard(const Parameters &params);

// ─── Terminal output helpers ────────────────────────────────────────────────
// Shared by main.cpp and every setup routine (geometry/anchoring/potential/
// evolve init) so the whole CLI output follows one consistent style instead
// of each file rolling its own separator/alignment.
void print_separator(const char *seg = "-", int count = 60);
void print_header(const char *title, int count = 60, const char *seg = "-");
// Aligned "  Label: value" line — the standard row style for setup summaries.
void print_field(const char *label, const char *value);
void print_field(const char *label, double value);
void print_field(const char *label, int value);

#endif
