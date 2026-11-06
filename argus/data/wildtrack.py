"""WILDTRACK: 7 synchronized cameras over one plaza (single site, one overlap area).

Expected layout (official release):
  Wildtrack/annotations_positions/00000000.json, 00000005.json, ...
  Wildtrack/Image_subsets/C1..C7/00000000.png, ...
Each JSON is a list of {"personID", "positionID", "views": [{"viewNum": 0..6, "xmin",
"ymin", "xmax", "ymax"}, ...]} with -1 coordinates when not visible. Annotation frames
are sampled at 2 fps (file index step 5 at 10 fps).
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

ANN_FPS = 10.0   # file index units (frame k corresponds to k / 10 s)


def load_boxes(root: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "annotations_positions", "*.json"))):
        frame = int(os.path.splitext(os.path.basename(f))[0])
        with open(f) as fh:
            for p in json.load(fh):
                for v in p["views"]:
                    if v["xmax"] < 0 or v["xmin"] < 0:
                        continue
                    rows.append({"cam": int(v["viewNum"]), "frame": frame, "pid": int(p["personID"]),
                                 "x1": v["xmin"], "y1": v["ymin"], "x2": v["xmax"], "y2": v["ymax"]})
    return pd.DataFrame(rows)


def image_path(root: str, cam: int, frame: int) -> str:
    return os.path.join(root, "Image_subsets", f"C{cam + 1}", f"{frame:08d}.png")


def split(tr: pd.DataFrame, train_frac: float = 0.5):
    """Temporal split: first part for walking-time/threshold estimation, rest for test."""
    t_cut = tr.t_start.min() + train_frac * (tr.t_end.max() - tr.t_start.min())
    return tr[tr.t_end <= t_cut].reset_index(drop=True), tr[tr.t_start > t_cut].reset_index(drop=True)
