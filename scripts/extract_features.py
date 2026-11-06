"""Extract OSNet appearance vectors for real-data tracklets (run in the feature venv).

    uv venv .venv-feat --python 3.12
    uv pip install --python .venv-feat/Scripts/python.exe --index-url \
        https://download.pytorch.org/whl/cpu torch torchvision
    uv pip install --python .venv-feat/Scripts/python.exe torchreid tensorboard gdown \
        opencv-python-headless pandas pyarrow scipy msgpack pyyaml
    gdown 1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal -O data/models/osnet_ain_x1_0_msmt17_combineall.pth
    .venv-feat/Scripts/python.exe scripts/extract_features.py wildtrack \
        --root data/raw/Wildtrack --out data/real/wildtrack

Writes <out>_vec.npy (final vectors) and <out>_prov.npy (provisional vectors from the
first half of each tracklet), row-aligned with the tracklet table that
scripts/prepare_real.py builds with the same --gap/--p_frag/--fps arguments.
Weights: OSNet-AIN x1.0 trained on all of MSMT17 (torchreid model zoo, cross-domain).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from argus.data import features as F  # noqa: E402
from argus.data import tracklets as T  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["wildtrack"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="data/models/osnet_ain_x1_0_msmt17_combineall.pth")
    ap.add_argument("--model", default="osnet_ain_x1_0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--p_frag", type=float, default=0.0)
    ap.add_argument("--K", type=int, default=8)
    a = ap.parse_args()

    from argus.data import wildtrack as D
    boxes = D.load_boxes(a.root)
    fps = a.fps or D.ANN_FPS
    tr = T.boxes_to_tracklets(boxes, fps=fps, gap_s=a.gap, p_frag=a.p_frag)
    print(f"{len(boxes)} boxes, {len(tr)} tracklets, {tr.gt_pid.nunique()} people")
    ext = F.make_extractor(a.model, a.weights, a.device)
    t0 = time.time()
    vec, prov = F.embed_tracklets(
        tr, boxes, frame_loader=lambda cam, f: cv2.imread(D.image_path(a.root, cam, f)),
        K=a.K, extractor=ext)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.save(f"{a.out}_vec.npy", vec)
    np.save(f"{a.out}_prov.npy", prov)
    print(f"embedded in {time.time() - t0:.0f} s -> {a.out}_vec.npy, {a.out}_prov.npy")
