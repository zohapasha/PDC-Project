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
from .sysinfo import config_hash, system_info

TIMED_EXPERIMENTS = {"e2", "e3", "e4"}


def _run_one_seed(task):
    experiment_id, params, seed = task
    start = time.time()
    rows = EXPERIMENTS[experiment_id](params, seed)
    for row in rows:
        row["seed"] = seed
        row["run_wall_s"] = time.time() - start
    return rows


def run(config_path: str, seeds, jobs: int = 1, out_dir: str = "results"):
    with open(config_path) as file:
        params = yaml.safe_load(file)
    experiment_id = params["experiment"]
    name = params.get("name", os.path.splitext(os.path.basename(config_path))[0])
    params.setdefault("tag", name)
    provenance = system_info()
    params_hash = config_hash(params)
    tasks = [(experiment_id, params, seed) for seed in seeds]
    if experiment_id in TIMED_EXPERIMENTS or jobs <= 1:
        results = []
        for task in tasks:
            results.append(_run_one_seed(task))
            print(f"  [{name}] seed {task[2]} done", flush=True)
    else:
        with mp.get_context("spawn").Pool(min(jobs, len(tasks))) as pool:
            results = pool.map(_run_one_seed, tasks)
    rows = [row for seed_rows in results for row in seed_rows]
    for row in rows:
        row.update({"experiment": experiment_id, "config": name, "config_hash": params_hash,
                    **provenance})
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.csv")
    table = pd.DataFrame(rows)
    if os.path.exists(path):
        table = pd.concat([pd.read_csv(path), table], ignore_index=True)
    table.to_csv(path, index=False)
    print(f"wrote {len(rows)} rows -> {path}")
    return table
