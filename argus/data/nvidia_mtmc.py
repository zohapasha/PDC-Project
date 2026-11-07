"""NVIDIA PhysicalAI-SmartSpaces, MTMC_Tracking_2024 (AI City Challenge 2024 Track 1).

Hugging Face dataset `nvidia/PhysicalAI-SmartSpaces` (CC-BY-4.0, not gated). 90 synthetic
indoor scenes (warehouses, hospitals, retail), 7-16 cameras each, 1080p video. Ground
truth for train, val AND test is public. Layout per scene:

  MTMC_Tracking_2024/<split>/<scene>/ground_truth.txt     space separated, one box per line:
      camera_id obj_id frame_id xmin ymin width height xworld yworld
  MTMC_Tracking_2024/<split>/<scene>/camera_XXXX/video.mp4
  MTMC_Tracking_2024/<split>/<scene>/camera_XXXX/calibration.json

splits: train = scene_001..040, val = scene_041..060, test = scene_061..090 (071+ have 16 cams).
camera_id is global across the dataset (camera_0124 -> 124); obj_id is a person ID that
is consistent across the cameras of one scene (multi-camera GT). (xworld, yworld) is the
person's floor position in metres.

Frame indexing: frame_id is used as the 0-based video frame index (clamped to the video
length). At 30 fps an off-by-one shifts a crop by 33 ms, which does not matter for
appearance features.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

REPO = "nvidia/PhysicalAI-SmartSpaces"
PREFIX = "MTMC_Tracking_2024"
FPS = 30.0
GT_COLS = ["cam_id", "pid", "frame", "x", "y", "w", "h", "X", "Y"]


def download_scene(split: str, scene: str, local_dir: str, videos: bool = True) -> str:
    """Download one scene from the Hugging Face hub; returns the scene directory."""
    from huggingface_hub import snapshot_download
    base = f"{PREFIX}/{split}/{scene}"
    pats = [f"{base}/ground_truth.txt", f"{base}/camera_*/calibration.json"]
    if videos:
        pats.append(f"{base}/camera_*/video.mp4")
    snapshot_download(REPO, repo_type="dataset", allow_patterns=pats, local_dir=local_dir,
                      max_workers=8)
    return os.path.join(local_dir, PREFIX, split, scene)


def camera_ids(scene_dir: str) -> list[int]:
    ids = []
    for d in glob.glob(os.path.join(scene_dir, "camera_*")):
        m = re.search(r"camera_(\d+)$", d)
        if m and os.path.isdir(d):
            ids.append(int(m.group(1)))
    return sorted(ids)


def video_path(scene_dir: str, cam_id: int) -> str:
    return os.path.join(scene_dir, f"camera_{cam_id:04d}", "video.mp4")


def load_boxes(scene_dir: str, frame_step: int = 1) -> tuple[pd.DataFrame, list[int]]:
    """Returns (boxes, cam_ids). boxes has the shared real-data columns
    cam (0..n_cams-1), frame, pid, x1, y1, x2, y2, plus X, Y (floor position, metres).
    cam_ids[cam] is the dataset's global camera id. frame_step > 1 keeps every k-th frame."""
    g = pd.read_csv(os.path.join(scene_dir, "ground_truth.txt"), sep=r"\s+", header=None,
                    names=GT_COLS, usecols=range(len(GT_COLS)))
    if frame_step > 1:
        g = g[g.frame % frame_step == 0]
    ids = camera_ids(scene_dir) or sorted(int(c) for c in g.cam_id.unique())
    local = {c: i for i, c in enumerate(ids)}
    g = g[g.cam_id.isin(local)]
    out = pd.DataFrame({
        "cam": g.cam_id.map(local).astype(np.int64).to_numpy(),
        "frame": g.frame.astype(np.int64).to_numpy(),
        "pid": g.pid.astype(np.int64).to_numpy(),
        "x1": g.x.to_numpy(float), "y1": g.y.to_numpy(float),
        "x2": (g.x + g.w).to_numpy(float), "y2": (g.y + g.h).to_numpy(float),
        "X": g.X.to_numpy(float), "Y": g.Y.to_numpy(float),
    })
    return out, ids


def site_map_by_fov(boxes: pd.DataFrame, n_cams: int, n_sites: int) -> list[int]:
    """Split the cameras into n_sites spatially contiguous groups ("wings" of the building).

    Each camera's field of view is summarised by the mean floor position of the people it
    sees. Cameras are ordered along the first principal axis of those centroids and cut
    into n_sites groups of (nearly) equal size. This needs only the ground truth, not the
    camera extrinsics (some 2024 calibration fields are missing)."""
    if n_sites <= 1:
        return [0] * n_cams
    c = boxes.groupby("cam")[["X", "Y"]].mean().reindex(range(n_cams))
    c = c.fillna(c.mean())
    xy = c.to_numpy(float)
    xy = xy - xy.mean(0)
    axis = np.linalg.svd(xy, full_matrices=False)[2][0] if len(xy) > 1 else np.array([1.0, 0.0])
    order = np.argsort(xy @ axis, kind="stable")
    site = [0] * n_cams
    for s, chunk in enumerate(np.array_split(order, n_sites)):
        for cam in chunk:
            site[int(cam)] = s
    return site


def profile(boxes: pd.DataFrame, fps: float) -> dict:
    """Small summary printed in the Kaggle log (the data is never profiled locally)."""
    per_pid_cams = boxes.groupby("pid").cam.nunique()
    return {
        "n_boxes": int(len(boxes)), "n_cams": int(boxes.cam.nunique()),
        "n_people": int(boxes.pid.nunique()),
        "frame_min": int(boxes.frame.min()), "frame_max": int(boxes.frame.max()),
        "duration_s": float((boxes.frame.max() - boxes.frame.min() + 1) / fps),
        "cams_per_person_mean": float(per_pid_cams.mean()),
        "cams_per_person_max": int(per_pid_cams.max()),
    }
