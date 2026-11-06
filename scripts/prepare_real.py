"""Build ARGUS streams from a real dataset (after features were extracted on Colab).

WILDTRACK:
  python scripts/prepare_real.py wildtrack --root /data/Wildtrack \
      --vec wildtrack_vec.npy --prov wildtrack_prov.npy --out data/real/wildtrack
MTMMC (one recording; camera->site groups by floor/area, e.g. "0,1,2,3;4,5,6,7;..."):
  python scripts/prepare_real.py mtmmc --root /data/MTMMC/campus/s01 --sites "0,1,2,3;4,5,6,7" \
      --vec s01_vec.npy --prov s01_prov.npy --out data/real/mtmmc_s01

Writes <out>_train.parquet and <out>_test.parquet (+ topology/meta JSON) and prints the
tau_accept chosen on the training split. Then run E1 with a config containing
  experiment: e1
  stream: data/real/wildtrack_test.parquet
  engine: {tau_accept: <chosen>}
The tracklet table must be built with the same arguments as used for feature extraction
(the .npy rows are aligned with it).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from argus.data import tracklets as T  # noqa: E402
from argus.eval.experiments import DEFAULT_ENGINE  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["wildtrack", "mtmmc"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--vec", required=True)
    ap.add_argument("--prov", default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--sites", default=None)
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--p_frag", type=float, default=0.0)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--no_center", action="store_true",
                    help="skip mean-centering (OSNet features are non-negative after ReLU, "
                         "which inflates every cosine; the mean is estimated on train only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.dataset == "wildtrack":
        from argus.data import wildtrack as D
        boxes = D.load_boxes(a.root)
        fps = a.fps or D.ANN_FPS
    else:
        from argus.data import mtmmc as D
        boxes = D.load_boxes(a.root)
        fps = a.fps or 23.0
    tr = T.boxes_to_tracklets(boxes, fps=fps, gap_s=a.gap, p_frag=a.p_frag)
    vec = np.load(a.vec).astype(np.float32)
    prov = np.load(a.prov).astype(np.float32) if a.prov else None
    assert len(vec) == len(tr), "feature rows must align with the tracklet table"
    n_cams = int(boxes.cam.max()) + 1
    if a.sites:
        groups = [[int(c) for c in g.split(",")] for g in a.sites.split(";")]
        site = [0] * n_cams
        for i, g in enumerate(groups):
            for c in g:
                site[c] = i
    else:
        site = [0] * n_cams
    t_cut = tr.t_start.min() + a.train_frac * (tr.t_end.max() - tr.t_start.min())
    tr_idx = np.nonzero(tr.t_end.to_numpy() <= t_cut)[0]
    te_idx = np.nonzero(tr.t_start.to_numpy() > t_cut)[0]
    topo = T.estimate_topology(tr.iloc[tr_idx], n_cams, site)
    if not a.no_center:
        mu = vec[tr_idx].mean(0)
        unit = lambda x: (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
        vec = unit(vec - mu)
        if prov is not None:
            prov = unit(prov - mu)
    for name, idx in (("train", tr_idx), ("test", te_idx)):
        sub = tr.iloc[idx].reset_index(drop=True)
        s = T.to_stream(sub, vec[idx], topo, fps, None if prov is None else prov[idx])
        path = f"{a.out}_{name}.parquet"
        s.save(path)
        print(f"{name}: {len(sub)} tracklets, {sub.gt_pid.nunique()} people -> {path}")
        if name == "train":
            tau, f1 = T.choose_thresholds(s, DEFAULT_ENGINE)
            print(f"chosen tau_accept on train = {tau} (pairwise F1 {f1:.3f}); freeze it for test")
