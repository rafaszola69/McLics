# McLics - Monte Carlo Liquid Crystal Simulator
# Copyright (C) 2026 The McLics authors (see AUTHORS)
# SPDX-License-Identifier: GPL-3.0-or-later
# ===========================================================================
#  McLics - Makefile
#  Targets:
#    make            -> GPU binary (mclics), requires nvcc
#    make CPU        -> CPU-only binary (mclics)
#    make clean      -> remove build artefacts and the binary
#    make test       -> quick checks of ./mclics against known answers (CPU modes); make test-gpu: GPU modes
#    make clean-data -> remove simulation output (ic.csv, po.dat, director_field_*.csv,
#                        mclics_*.png written by analyze.py)
# ===========================================================================
 PROGRAM := mclics
########### Gnu:
 COMPILER := g++ 
 GPUCOMP  := nvcc
 FLAGS    :=  -O3  -fopenmp -static
 GPUFLAGS    :=  -O3
 #-Wno-unused-result -Wno-format
 LIB := -lm  -lgsl -lgslcblas  -lgomp 
# LIB := -lm  -lgslcblas  -lgsl -lgomp 
# without nvcc, plain `make` builds the CPU version
ifeq (,$(shell which nvcc))
	.DEFAULT_GOAL=CPU
endif
############ Intel 2 :
#COMPILER = icpc
#FLAGS= -ipo -O3  -no-prec-div -xAVX -simd -qopenmp -fp-model fast=2 -static
#LIB = -mkl -lgsl 


############ Intel Debug :

#COMPILER = icpc
#FLAGS= -O0 -g -static
#LIB = -mkl -lgsl 



 
CPPS := $(wildcard src/*.cpp)
CUDA := $(wildcard src/*.cu)
# every source depends on every header (.h and .cuh), so changing a header rebuilds what uses it
HEADER := $(wildcard include/*.h) $(wildcard include/*.cuh)
# CPU and GPU objects live in separate folders: `make CPU` never reuses objects compiled with -DCUDA__
GPUOBJS  := $(patsubst src/%.cpp,build/gpu/%.o,${CPPS})
CBJS     := $(patsubst src/%.cu,build/gpu/%.cuda.o,${CUDA})
CPUOBJS  := $(patsubst src/%.cpp,build/cpu/%.o,${CPPS})
DBGOBJS  := $(patsubst src/%.cpp,build/dbg/%.o,${CPPS})
DBGCBJS  := $(patsubst src/%.cu,build/dbg/%.cuda.o,${CUDA})

# Each version is linked in its own folder and then copied to ./${PROGRAM}, so switching between `make` and
# `make CPU` always leaves the requested version in place.
all: build/gpu/${PROGRAM}
	@cp build/gpu/${PROGRAM} ${PROGRAM} && echo "GPU version -> ./${PROGRAM}"
CPU: build/cpu/${PROGRAM}
	@cp build/cpu/${PROGRAM} ${PROGRAM} && echo "CPU version -> ./${PROGRAM}"

build/gpu/${PROGRAM}: ${GPUOBJS} ${CBJS}
	@${GPUCOMP} ${GPUFLAGS} ${GPUOBJS} ${CBJS} ${LIB} -o $@
build/cpu/${PROGRAM}: ${CPUOBJS}
	@${COMPILER} ${FLAGS} ${CPUOBJS} ${LIB} -o $@
${GPUOBJS}: build/gpu/%.o: src/%.cpp ${HEADER} | build/gpu
	${COMPILER} ${FLAGS} -c $< -o $@ -D CUDA__="CUDA"
${CBJS}: build/gpu/%.cuda.o: src/%.cu ${HEADER} | build/gpu
	${GPUCOMP} ${GPUFLAGS} -dc -c $< -o $@ -D CUDA__="CUDA"
${CPUOBJS}: build/cpu/%.o: src/%.cpp ${HEADER} | build/cpu
	${COMPILER} ${FLAGS} -c $< -o $@

build/gpu build/cpu build/dbg:
	@mkdir -p $@
debug: ${DBGOBJS} ${DBGCBJS}
	${GPUCOMP} -O0 -g -Xcompiler -fopenmp -lineinfo $(filter-out -O% -fast -static, ${GPUFLAGS}) ${DBGOBJS} ${DBGCBJS} ${LIB} -o ${PROGRAM}_debug
${DBGOBJS}: build/dbg/%.o: src/%.cpp ${HEADER} | build/dbg
	${COMPILER} -O0 -g $(filter-out -O% -fast, ${FLAGS}) -c $< -o $@ -D CUDA__="CUDA"
${DBGCBJS}: build/dbg/%.cuda.o: src/%.cu ${HEADER} | build/dbg
	${GPUCOMP} -O0 -g $(filter-out -O% -fast, ${GPUFLAGS}) -lineinfo -dc -c $< -o $@ -D CUDA__="CUDA"

renew: clean all

clean:
	@rm -f ${PROGRAM}
	@rm -fr build
	@rm -f *.o
	@echo "Clean done."

clean-data:
	@rm -f ic.csv po.dat
	@find . -maxdepth 1 -type f -name "director_field_*.csv" -delete
	@find . -maxdepth 1 -type f -name "mclics_*.png" -delete
	@echo "Simulation output removed (ic.csv, po.dat, director_field_*.csv, mclics_*.png)."

# quick checks against known answers (tests/run_checks.py), with the program built last (./mclics)
test:
	@python3 tests/run_checks.py ./${PROGRAM}

test-gpu:
	@python3 tests/run_checks.py ./${PROGRAM} --gpu

.PHONY: all CPU debug clean clean-data renew test test-gpu
