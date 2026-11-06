"""MTMMC (multi-target multi-camera, campus + factory, 16 cameras).

ASSUMED layout -- verify against the release you download and adapt `load_boxes`:
  <root>/<scenario>/<cam_dir>/gt/gt.txt     MOT format: frame,id,x,y,w,h,conf,...
  <root>/<scenario>/<cam_dir>/<video>.mp4  (or extracted frames)
Global person IDs are assumed consistent across cameras within a recording (required for
multi-camera ground truth). Sites are assigned with a user-supplied camera -> site map
(e.g. by floor / area from the dataset layout figure).
"""
from __future__ import annotations

import glob
import os

import pandas as pd


def load_boxes(scenario_dir: str, cam_dirs=None) -> pd.DataFrame:
    cam_dirs = cam_dirs or sorted(d for d in os.listdir(scenario_dir)
                                  if os.path.isdir(os.path.join(scenario_dir, d)))
    out = []
    for ci, cd in enumerate(cam_dirs):
        gts = glob.glob(os.path.join(scenario_dir, cd, "gt", "gt.txt")) or \
              glob.glob(os.path.join(scenario_dir, cd, "*.txt"))
        if not gts:
            continue
        g = pd.read_csv(gts[0], header=None)
        g = g.iloc[:, :6]
        g.columns = ["frame", "pid", "x", "y", "w", "h"]
        g["cam"] = ci
        g["x1"], g["y1"] = g.x, g.y
        g["x2"], g["y2"] = g.x + g.w, g.y + g.h
        out.append(g[["cam", "frame", "pid", "x1", "y1", "x2", "y2"]])
    return pd.concat(out, ignore_index=True)


def site_map_from_groups(groups: list[list[int]], n_cams: int):
    """groups = [[cams of site 0], [cams of site 1], ...] -> site per camera."""
    s = [0] * n_cams
    for i, g in enumerate(groups):
        for c in g:
            s[c] = i
    return s
