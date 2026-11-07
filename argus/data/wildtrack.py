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

ANNOTATION_FPS = 10.0   # file index units (frame k corresponds to k / 10 s)


def load_boxes(root: str) -> pd.DataFrame:
    """Columns: camera_id, frame, person_id, x1, y1, x2, y2."""
    rows = []
    for annotation_path in sorted(glob.glob(os.path.join(root, "annotations_positions", "*.json"))):
        frame = int(os.path.splitext(os.path.basename(annotation_path))[0])
        with open(annotation_path) as file:
            for person in json.load(file):
                for view in person["views"]:
                    if view["xmax"] < 0 or view["xmin"] < 0:
                        continue
                    rows.append({"camera_id": int(view["viewNum"]), "frame": frame,
                                 "person_id": int(person["personID"]),
                                 "x1": view["xmin"], "y1": view["ymin"],
                                 "x2": view["xmax"], "y2": view["ymax"]})
    return pd.DataFrame(rows)


def image_path(root: str, camera_id: int, frame: int) -> str:
    return os.path.join(root, "Image_subsets", f"C{camera_id + 1}", f"{frame:08d}.png")


def split(tracklet_table: pd.DataFrame, train_frac: float = 0.5):
    """Temporal split: first part for walking-time/threshold estimation, rest for test."""
    cut_time = (tracklet_table.start_time.min()
                + train_frac * (tracklet_table.end_time.max() - tracklet_table.start_time.min()))
    return (tracklet_table[tracklet_table.end_time <= cut_time].reset_index(drop=True),
            tracklet_table[tracklet_table.start_time > cut_time].reset_index(drop=True))
