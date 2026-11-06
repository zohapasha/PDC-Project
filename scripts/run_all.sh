#!/usr/bin/env bash
# Reproduces every results/*.csv. Untimed experiments run seeds in parallel; timed ones
# (e2, e3, e4) run serially on an otherwise idle machine.
set -e
PY=${PY:-.venv/Scripts/python.exe}
[ -x "$PY" ] || PY=.venv/bin/python
$PY scripts/run_experiment.py configs/e1_accuracy.yaml configs/e1_accuracy_hard.yaml \
    configs/e5_bytes.yaml configs/e6_noise.yaml configs/e7_cross.yaml --seeds 0-4 --jobs 5 --fresh
$PY scripts/run_experiment.py configs/e3_corrections.yaml configs/e2_incremental.yaml \
    configs/e4_weak.yaml configs/e4_strong.yaml --seeds 0-4 --fresh
$PY scripts/make_figures.py
