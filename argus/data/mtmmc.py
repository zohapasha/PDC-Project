"""MTMMC (multi-target multi-camera, campus + factory, 16 cameras).

ASSUMED layout -- verify against the release you download and adapt `load_boxes`:
  <root>/<scenario>/<camera_dir>/gt/gt.txt     MOT format: frame,id,x,y,w,h,conf,...
  <root>/<scenario>/<camera_dir>/<video>.mp4  (or extracted frames)
Global person IDs are assumed consistent across cameras within a recording (required for
multi-camera ground truth). Sites are assigned with a user-supplied camera -> site map
(e.g. by floor / area from the dataset layout figure).
"""
from __future__ import annotations

import glob
import os

import pandas as pd


def load_boxes(scenario_dir: str, camera_dirs=None) -> pd.DataFrame:
    """Columns: camera_id, frame, person_id, x1, y1, x2, y2."""
    camera_dirs = camera_dirs or sorted(name for name in os.listdir(scenario_dir)
                                        if os.path.isdir(os.path.join(scenario_dir, name)))
    tables = []
    for camera_id, camera_dir in enumerate(camera_dirs):
        gt_files = (glob.glob(os.path.join(scenario_dir, camera_dir, "gt", "gt.txt"))
                    or glob.glob(os.path.join(scenario_dir, camera_dir, "*.txt")))
        if not gt_files:
            continue
        gt = pd.read_csv(gt_files[0], header=None)
        gt = gt.iloc[:, :6]
        gt.columns = ["frame", "person_id", "x", "y", "width", "height"]
        gt["camera_id"] = camera_id
        gt["x1"], gt["y1"] = gt.x, gt.y
        gt["x2"], gt["y2"] = gt.x + gt.width, gt.y + gt.height
        tables.append(gt[["camera_id", "frame", "person_id", "x1", "y1", "x2", "y2"]])
    return pd.concat(tables, ignore_index=True)


def site_map_from_groups(camera_groups: list[list[int]], n_cameras: int):
    """camera_groups = [[cameras of site 0], [cameras of site 1], ...] -> site per camera."""
    site_of_camera = [0] * n_cameras
    for site_id, group in enumerate(camera_groups):
        for camera_id in group:
            site_of_camera[camera_id] = site_id
    return site_of_camera
