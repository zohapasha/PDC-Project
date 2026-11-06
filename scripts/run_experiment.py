"""Usage:  python scripts/run_experiment.py configs/e1_accuracy.yaml --seeds 0-4 [--jobs 5] [--fresh]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argus.eval.harness import run  # noqa: E402


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--fresh", action="store_true", help="delete existing results CSV first")
    a = ap.parse_args()
    for c in a.configs:
        if a.fresh:
            name = os.path.splitext(os.path.basename(c))[0]
            p = os.path.join("results", f"{name}.csv")
            if os.path.exists(p):
                os.remove(p)
        print(f"== {c}", flush=True)
        run(c, parse_seeds(a.seeds), jobs=a.jobs)
