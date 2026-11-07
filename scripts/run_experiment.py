"""Usage:  python scripts/run_experiment.py configs/e1_accuracy.yaml --seeds 0-4 [--jobs 5] [--fresh]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argus.eval.harness import run  # noqa: E402


def parse_seeds(spec):
    """'0-2,5' -> [0, 1, 2, 5]."""
    seeds = []
    for part in spec.split(","):
        if "-" in part:
            first, last = part.split("-")
            seeds += list(range(int(first), int(last) + 1))
        else:
            seeds.append(int(part))
    return seeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--fresh", action="store_true", help="delete existing results CSV first")
    args = parser.parse_args()
    for config_path in args.configs:
        if args.fresh:
            name = os.path.splitext(os.path.basename(config_path))[0]
            results_path = os.path.join("results", f"{name}.csv")
            if os.path.exists(results_path):
                os.remove(results_path)
        print(f"== {config_path}", flush=True)
        run(config_path, parse_seeds(args.seeds), jobs=args.jobs)
