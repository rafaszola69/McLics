# Performance of the Monte Carlo step (appendix of the paper)

Time per site update of the Monte Carlo step on one CPU thread, on all cores with OpenMP and on the GPU, for the
Lebwohl–Lasher potential in periodic boxes of 32³, 64³ and 128³ sites and for the GHRL potential (E7 constants)
in 64³ sites.

```bash
cd paper/benchmark
./run.sh                 # 1 thread, 8 and 16 threads, GPU; about 15-30 minutes on a desktop
python3 make_table.py    # -> table.tex
```

Each case runs with N Monte Carlo steps and with 0 steps; the difference is the cost of the steps alone, without
the setup and the output files. The single-thread runs do about 10⁸ site updates; the multithread and GPU runs do
4 and 200 times more, so that the steps dominate their run time too. Close other programs before running: the
multithread runs use all cores. `machine.txt` records the processor, the GPU, the compilers and the McLics version.

Variables: `THREADS` (default `"8 16"`), `SKIP_GPU=1`, `REPEAT` (the table uses the fastest repetition),
`SCALE` (fraction of the default number of steps), `MT_FACTOR`, `GPU_FACTOR`.
