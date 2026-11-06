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
from argus.eval.experiments import gen_cfg  # noqa: E402
from argus.gen.campus import Stream, load_or_generate  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None)
    ap.add_argument("--sigma", type=float, default=None)
    a = ap.parse_args()
    p = {"preset": "medium"}
    if a.sigma is not None:
        p["gen"] = {"sigma": a.sigma}
    srcs = [("Synthetic", load_or_generate(gen_cfg(p, 0)))]
    if a.real:
        srcs.append(("Real", Stream.load(a.real)))
    fig, axs = plt.subplots(1, len(srcs), figsize=(3.5 * len(srcs), 2.2), squeeze=False)
    bins = np.linspace(-0.4, 1.0, 71)
    for ax, (name, s) in zip(axs[0], srcs):
        pos, neg = similarity_histograms(s)
        ax.hist(neg, bins=bins, density=True, histtype="step", lw=1.4, color="#eb6834", label="different person")
        ax.hist(pos, bins=bins, density=True, histtype="step", lw=1.4, color="#2a78d6", label="same person")
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("Cosine similarity")
        ax.legend(fontsize=7, frameon=False)
        print(f"{name}: same mean {pos.mean():.3f} sd {pos.std():.3f} | diff mean {neg.mean():.3f} "
              f"sd {neg.std():.3f} | P(diff >= 0.6) {np.mean(neg >= 0.6):.4f} | P(same < 0.6) {np.mean(pos < 0.6):.3f}")
    axs[0][0].set_ylabel("Density")
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig("results/figures/fig_calibration.pdf", bbox_inches="tight")
    fig.savefig("results/figures/fig_calibration.png", dpi=200, bbox_inches="tight")
