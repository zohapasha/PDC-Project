"""Similarity-histogram calibration figure (Section 7.3 of the plan).

  python scripts/calibrate.py                                  # synthetic only
  python scripts/calibrate.py --real data/real/wildtrack_train.parquet
Plots cosine-similarity densities of true-match vs false-match tracklet pairs for the
synthetic generator (default sigma) and, if given, the real dataset. Tune sigma /
sigma_cam / lookalike_spread until the synthetic curves roughly match the real ones.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from argus.data.features import similarity_histograms  # noqa: E402
from argus.eval.experiments import generator_config  # noqa: E402
from argus.gen.campus import Stream, load_or_generate  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", default=None)
    parser.add_argument("--sigma", type=float, default=None)
    args = parser.parse_args()
    params = {"preset": "medium"}
    if args.sigma is not None:
        params["gen"] = {"sigma": args.sigma}
    sources = [("Synthetic", load_or_generate(generator_config(params, 0)))]
    if args.real:
        sources.append(("Real", Stream.load(args.real)))
    fig, axes = plt.subplots(1, len(sources), figsize=(3.5 * len(sources), 2.2), squeeze=False)
    bins = np.linspace(-0.4, 1.0, 71)
    for ax, (name, stream) in zip(axes[0], sources):
        same_person, different_person = similarity_histograms(stream)
        ax.hist(different_person, bins=bins, density=True, histtype="step", lw=1.4,
                color="#eb6834", label="different person")
        ax.hist(same_person, bins=bins, density=True, histtype="step", lw=1.4,
                color="#2a78d6", label="same person")
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("Cosine similarity")
        ax.legend(fontsize=7, frameon=False)
        print(f"{name}: same mean {same_person.mean():.3f} sd {same_person.std():.3f} | "
              f"diff mean {different_person.mean():.3f} sd {different_person.std():.3f} | "
              f"P(diff >= 0.6) {np.mean(different_person >= 0.6):.4f} | "
              f"P(same < 0.6) {np.mean(same_person < 0.6):.3f}")
    axes[0][0].set_ylabel("Density")
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig("results/figures/fig_calibration.pdf", bbox_inches="tight")
    fig.savefig("results/figures/fig_calibration.png", dpi=200, bbox_inches="tight")
