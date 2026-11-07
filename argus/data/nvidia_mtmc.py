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

REPO_ID = "nvidia/PhysicalAI-SmartSpaces"
DATASET_PREFIX = "MTMC_Tracking_2024"
FPS = 30.0
GT_COLUMNS = ["dataset_camera_id", "person_id", "frame", "x", "y", "width", "height",
              "world_x", "world_y"]


def download_scene(split: str, scene: str, local_dir: str, videos: bool = True) -> str:
    """Download one scene from the Hugging Face hub; returns the scene directory."""
    from huggingface_hub import snapshot_download
    scene_prefix = f"{DATASET_PREFIX}/{split}/{scene}"
    patterns = [f"{scene_prefix}/ground_truth.txt", f"{scene_prefix}/camera_*/calibration.json"]
    if videos:
        patterns.append(f"{scene_prefix}/camera_*/video.mp4")
    snapshot_download(REPO_ID, repo_type="dataset", allow_patterns=patterns, local_dir=local_dir,
                      max_workers=8)
    return os.path.join(local_dir, DATASET_PREFIX, split, scene)


def list_camera_ids(scene_dir: str) -> list[int]:
    """Dataset (global) camera ids of the camera_XXXX directories of a scene."""
    camera_ids = []
    for camera_dir in glob.glob(os.path.join(scene_dir, "camera_*")):
        match = re.search(r"camera_(\d+)$", camera_dir)
        if match and os.path.isdir(camera_dir):
            camera_ids.append(int(match.group(1)))
    return sorted(camera_ids)


def video_path(scene_dir: str, dataset_camera_id: int) -> str:
    return os.path.join(scene_dir, f"camera_{dataset_camera_id:04d}", "video.mp4")


def load_boxes(scene_dir: str, frame_step: int = 1) -> tuple[pd.DataFrame, list[int]]:
    """Returns (boxes, dataset_camera_ids). boxes has the shared real-data columns
    camera_id (0..n_cameras-1), frame, person_id, x1, y1, x2, y2, plus world_x, world_y
    (floor position, metres). dataset_camera_ids[camera_id] is the dataset's global camera
    id. frame_step > 1 keeps every k-th frame."""
    gt = pd.read_csv(os.path.join(scene_dir, "ground_truth.txt"), sep=r"\s+", header=None,
                     names=GT_COLUMNS, usecols=range(len(GT_COLUMNS)))
    if frame_step > 1:
        gt = gt[gt.frame % frame_step == 0]
    dataset_camera_ids = (list_camera_ids(scene_dir)
                          or sorted(int(camera) for camera in gt.dataset_camera_id.unique()))
    local_camera_id = {dataset_id: local_id for local_id, dataset_id in enumerate(dataset_camera_ids)}
    gt = gt[gt.dataset_camera_id.isin(local_camera_id)]
    boxes = pd.DataFrame({
        "camera_id": gt.dataset_camera_id.map(local_camera_id).astype(np.int64).to_numpy(),
        "frame": gt.frame.astype(np.int64).to_numpy(),
        "person_id": gt.person_id.astype(np.int64).to_numpy(),
        "x1": gt.x.to_numpy(float), "y1": gt.y.to_numpy(float),
        "x2": (gt.x + gt.width).to_numpy(float), "y2": (gt.y + gt.height).to_numpy(float),
        "world_x": gt.world_x.to_numpy(float), "world_y": gt.world_y.to_numpy(float),
    })
    return boxes, dataset_camera_ids


def site_map_by_fov(boxes: pd.DataFrame, n_cameras: int, n_sites: int) -> list[int]:
    """Split the cameras into n_sites spatially contiguous groups ("wings" of the building).

    Each camera's field of view is summarised by the mean floor position of the people it
    sees. Cameras are ordered along the first principal axis of those centroids and cut
    into n_sites groups of (nearly) equal size. This needs only the ground truth, not the
    camera extrinsics (some 2024 calibration fields are missing)."""
    if n_sites <= 1:
        return [0] * n_cameras
    centroids = boxes.groupby("camera_id")[["world_x", "world_y"]].mean().reindex(range(n_cameras))
    centroids = centroids.fillna(centroids.mean())
    floor_xy = centroids.to_numpy(float)
    floor_xy = floor_xy - floor_xy.mean(0)
    principal_axis = (np.linalg.svd(floor_xy, full_matrices=False)[2][0] if len(floor_xy) > 1
                      else np.array([1.0, 0.0]))
    camera_order = np.argsort(floor_xy @ principal_axis, kind="stable")
    site_of_camera = [0] * n_cameras
    for site_id, chunk in enumerate(np.array_split(camera_order, n_sites)):
        for camera_id in chunk:
            site_of_camera[int(camera_id)] = site_id
    return site_of_camera


def summarize_boxes(boxes: pd.DataFrame, fps: float) -> dict:
    """Small summary printed in the Kaggle log (the data is never profiled locally).
    The keys are stored in results/real_nvidia_profile.json."""
    cameras_per_person = boxes.groupby("person_id").camera_id.nunique()
    return {
        "n_boxes": int(len(boxes)), "n_cams": int(boxes.camera_id.nunique()),
        "n_people": int(boxes.person_id.nunique()),
        "frame_min": int(boxes.frame.min()), "frame_max": int(boxes.frame.max()),
        "duration_s": float((boxes.frame.max() - boxes.frame.min() + 1) / fps),
        "cams_per_person_mean": float(cameras_per_person.mean()),
        "cams_per_person_max": int(cameras_per_person.max()),
    }
