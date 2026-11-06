"""Runs one experiment config over seeds and appends rows to results/<name>.csv.

Timed experiments (e2, e3, e4) always run seeds serially. Accuracy/byte experiments may
run seeds in parallel worker processes (--jobs), which does not affect their metrics.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

import pandas as pd
import yaml

from .experiments import EXPERIMENTS
from .sysinfo import cfg_hash, sysinfo

TIMED = {"e2", "e3", "e4"}


def _one(args):
    exp, params, seed = args
    t0 = time.time()
    rows = EXPERIMENTS[exp](params, seed)
    for r in rows:
        r["seed"] = seed
        r["run_wall_s"] = time.time() - t0
    return rows


def run(config_path: str, seeds, jobs: int = 1, out_dir: str = "results"):
    with open(config_path) as f:
        params = yaml.safe_load(f)
    exp = params["experiment"]
    name = params.get("name", os.path.splitext(os.path.basename(config_path))[0])
    params.setdefault("tag", name)
    info = sysinfo()
    h = cfg_hash(params)
    tasks = [(exp, params, s) for s in seeds]
    if exp in TIMED or jobs <= 1:
        results = []
        for t in tasks:
            results.append(_one(t))
            print(f"  [{name}] seed {t[2]} done", flush=True)
    else:
        with mp.get_context("spawn").Pool(min(jobs, len(tasks))) as pool:
            results = pool.map(_one, tasks)
    rows = [r for rs in results for r in rs]
    for r in rows:
        r.update({"experiment": exp, "config": name, "config_hash": h, **info})
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.csv")
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"wrote {len(rows)} rows -> {path}")
    return df
