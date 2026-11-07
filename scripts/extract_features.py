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

from argus.data import features  # noqa: E402
from argus.data import tracklets  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["wildtrack"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True, dest="output_prefix")
    parser.add_argument("--weights", default="data/models/osnet_ain_x1_0_msmt17_combineall.pth")
    parser.add_argument("--model", default="osnet_ain_x1_0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--gap", type=float, default=1.0, dest="gap_s")
    parser.add_argument("--p_frag", type=float, default=0.0)
    parser.add_argument("--K", type=int, default=8, dest="frames_per_tracklet",
                        help="frames sampled per tracklet")
    args = parser.parse_args()

    from argus.data import wildtrack
    boxes = wildtrack.load_boxes(args.root)
    fps = args.fps or wildtrack.ANNOTATION_FPS
    tracklet_table = tracklets.boxes_to_tracklets(boxes, fps=fps, gap_s=args.gap_s,
                                                  p_frag=args.p_frag)
    print(f"{len(boxes)} boxes, {len(tracklet_table)} tracklets, "
          f"{tracklet_table.gt_person_id.nunique()} people")
    extractor = features.make_extractor(args.model, args.weights, args.device)
    start = time.time()
    vectors, provisional_vectors = features.embed_tracklets(
        tracklet_table, boxes,
        frame_loader=lambda camera_id, frame: cv2.imread(wildtrack.image_path(args.root, camera_id, frame)),
        frames_per_tracklet=args.frames_per_tracklet, extractor=extractor)
    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)
    np.save(f"{args.output_prefix}_vec.npy", vectors)
    np.save(f"{args.output_prefix}_prov.npy", provisional_vectors)
    print(f"embedded in {time.time() - start:.0f} s -> {args.output_prefix}_vec.npy, "
          f"{args.output_prefix}_prov.npy")
