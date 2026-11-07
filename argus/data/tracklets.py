"""Shared real-data utilities: GT boxes -> tracklets, data-driven topology estimation,
and conversion to the same Stream format the synthetic generator produces.

Box table columns (pandas): camera_id, frame, person_id, x1, y1, x2, y2
Tracklet table columns:     camera_id, gt_person_id, start_frame, end_frame, frames,
                            box_rows, start_time, end_time
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.topology import Topology, overlap_groups
from ..common.types import EVENT_REVISION, EVENT_TRACKLET
from ..gen.campus import Stream

_FAR_WALK_TIME = 1e4         # walking time assigned to camera pairs never seen connected


def boxes_to_tracklets(boxes: pd.DataFrame, fps: float, gap_s: float = 1.0,
                       p_frag: float = 0.0, max_len_s: float = 120.0, seed: int = 0) -> pd.DataFrame:
    """Group GT boxes per (camera, person) in time order; start a new tracklet when the
    gap exceeds gap_s or the tracklet exceeds max_len_s. Optionally fragment further with
    probability p_frag (simulated single-camera tracker breaks).
    Returns one row per tracklet (see the module docstring for the columns)."""
    rng = np.random.default_rng(seed)
    max_gap_frames = gap_s * fps
    rows = []
    for (camera_id, person_id), person_boxes in boxes.sort_values("frame").groupby(
            ["camera_id", "person_id"], sort=False):
        frames = person_boxes.frame.to_numpy()
        segment_starts = np.nonzero(np.diff(frames) > max_gap_frames)[0] + 1
        for segment in np.split(np.arange(len(frames)), segment_starts):
            # length cap
            piece_start = 0
            while piece_start < len(segment):
                piece_end = piece_start
                while (piece_end + 1 < len(segment) and frames[segment[piece_end + 1]]
                       - frames[segment[piece_start]] <= max_len_s * fps):
                    piece_end += 1
                piece = segment[piece_start:piece_end + 1]
                if len(piece) >= 4 and rng.random() < p_frag:
                    split_at = int(rng.integers(2, len(piece) - 1))
                    parts = [piece[:split_at], piece[split_at:]]
                else:
                    parts = [piece]
                for part in parts:
                    rows.append({"camera_id": int(camera_id), "gt_person_id": int(person_id),
                                 "start_frame": int(frames[part[0]]),
                                 "end_frame": int(frames[part[-1]]),
                                 "frames": frames[part].tolist(),
                                 "box_rows": person_boxes.index[part].tolist()})
                piece_start = piece_end + 1
    tracklet_table = pd.DataFrame(rows)
    tracklet_table["start_time"] = tracklet_table.start_frame / fps
    tracklet_table["end_time"] = (tracklet_table.end_frame + 1) / fps
    return tracklet_table.sort_values(["end_time", "start_time", "camera_id"]).reset_index(drop=True)


def estimate_topology(train_tracklets: pd.DataFrame, n_cameras: int, site_of_camera,
                      margin: float = 0.1, default_far: float = _FAR_WALK_TIME,
                      is_entrance=None) -> Topology:
    """Walking times from the TRAINING split: for each ordered camera pair the minimum
    observed transition time of the same GT person, minus a margin; overlap groups = pairs
    where one person is seen simultaneously; then metric closure (shortest paths)."""
    observed_walk_time = np.full((n_cameras, n_cameras), np.inf)
    np.fill_diagonal(observed_walk_time, 0.0)
    overlap_pairs = set()
    for _, person_tracklets in train_tracklets.sort_values("start_time").groupby("gt_person_id"):
        records = person_tracklets[["start_time", "end_time", "camera_id"]].to_numpy()
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                start_a, end_a, camera_a = records[i]
                start_b, end_b, camera_b = records[j]
                camera_a, camera_b = int(camera_a), int(camera_b)
                if camera_a == camera_b:
                    continue
                if start_b < end_a:                                  # simultaneous
                    overlap_pairs.add((min(camera_a, camera_b), max(camera_a, camera_b)))
                else:
                    transition_time = (start_b - end_a) * (1 - margin)
                    if transition_time < observed_walk_time[camera_a, camera_b]:
                        observed_walk_time[camera_a, camera_b] = transition_time
                        observed_walk_time[camera_b, camera_a] = transition_time
                if start_b - end_a > 600:
                    break
    for camera_a, camera_b in overlap_pairs:
        observed_walk_time[camera_a, camera_b] = observed_walk_time[camera_b, camera_a] = 0.0
    walk_time = observed_walk_time.copy()
    walk_time[np.isinf(walk_time)] = default_far
    for via_camera in range(n_cameras):
        walk_time = np.minimum(walk_time, walk_time[:, via_camera:via_camera + 1]
                               + walk_time[via_camera:via_camera + 1, :])
    site_of_camera = np.asarray(site_of_camera)
    corridors = [(camera_a, camera_b, float(walk_time[camera_a, camera_b]))
                 for camera_a in range(n_cameras) for camera_b in range(camera_a + 1, n_cameras)
                 if walk_time[camera_a, camera_b] < default_far]
    # one "building" per camera keeps the generic site mapping (site_of_building = site)
    return Topology(n_cameras, walk_time, overlap_groups(n_cameras, sorted(overlap_pairs)),
                    np.arange(n_cameras),
                    np.ones(n_cameras, bool) if is_entrance is None else np.asarray(is_entrance),
                    site_of_camera, corridors)


def to_stream(tracklet_table: pd.DataFrame, vectors: np.ndarray, topology: Topology, fps: float,
              provisional_vectors: np.ndarray | None = None, revise_delay: float = 10.0) -> Stream:
    """tracklet_table sorted by end_time (tracklet_id = row index). provisional_vectors
    (optional): vectors sent with the tracklet (e.g. from its first half); the final vector
    arrives revise_delay s later as a revision event."""
    n_tracklets = len(tracklet_table)
    end_times = tracklet_table.end_time.to_numpy(float)
    event_times = [end_times]
    event_types = [np.full(n_tracklets, EVENT_TRACKLET, np.int8)]
    event_tracklet_ids = [np.arange(n_tracklets)]
    if provisional_vectors is not None:
        event_vectors = [provisional_vectors, vectors]
        event_times.append(end_times + revise_delay)
        event_types.append(np.full(n_tracklets, EVENT_REVISION, np.int8))
        event_tracklet_ids.append(np.arange(n_tracklets))
    else:
        event_vectors = [vectors]
    event_times = np.concatenate(event_times)
    event_types = np.concatenate(event_types)
    event_tracklet_ids = np.concatenate(event_tracklet_ids)
    event_vectors = np.concatenate(event_vectors).astype(np.float32)
    order = np.lexsort((event_tracklet_ids, event_types, event_times))
    return Stream(topology, event_times[order], event_types[order], event_tracklet_ids[order],
                  event_vectors[order], tracklet_table.camera_id.to_numpy(np.int64),
                  tracklet_table.start_time.to_numpy(float), end_times,
                  tracklet_table.gt_person_id.to_numpy(np.int64),
                  {"gen": {"fps": fps}, "n_tracklets": n_tracklets, "source": "real"})


def choose_thresholds(train_stream: Stream, base_engine_config: dict,
                      tau_grid=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75)):
    """Pick tau_accept on the training split by pairwise F1; freeze it for test.
    Returns (tau_accept, pairwise_f1)."""
    from ..eval.metrics import evaluate
    from ..sync.sim import run_single
    best = None
    for tau_accept in tau_grid:
        engine_config = dict(base_engine_config, tau_accept=tau_accept,
                             tau_keep=min(base_engine_config.get("tau_keep", 0.5), tau_accept))
        node, _ = run_single(train_stream, engine_config)
        pairwise_f1 = evaluate(node.engine.partition(), train_stream)["pw_f1"]
        if best is None or pairwise_f1 > best[1]:
            best = (tau_accept, pairwise_f1)
    return best


def build_split_streams(tracklet_table: pd.DataFrame, vectors: np.ndarray, provisional_vectors,
                        fps: float, n_cameras: int, site_of_camera, output_prefix: str,
                        train_frac: float = 0.5, center: bool = True,
                        engine_config: dict | None = None, log=print) -> dict:
    """Temporal train/test split -> <output_prefix>_train.parquet, <output_prefix>_test.parquet.

    Walking times and overlap groups are estimated on the training split only, features
    are centred with the training mean (OSNet outputs are non-negative after ReLU, which
    inflates every cosine), and tau_accept is chosen on train by pairwise F1 when
    engine_config is given. Returns paths, the chosen tau, and topology statistics (the
    keys are stored in the profile JSON)."""
    assert len(vectors) == len(tracklet_table), "feature rows must align with the tracklet table"
    vectors = np.asarray(vectors, np.float32)
    provisional_vectors = (None if provisional_vectors is None
                           else np.asarray(provisional_vectors, np.float32))
    first_start = tracklet_table.start_time.min()
    cut_time = first_start + train_frac * (tracklet_table.end_time.max() - first_start)
    train_rows = np.nonzero(tracklet_table.end_time.to_numpy() <= cut_time)[0]
    test_rows = np.nonzero(tracklet_table.start_time.to_numpy() > cut_time)[0]
    topology = estimate_topology(tracklet_table.iloc[train_rows], n_cameras, site_of_camera)
    if center:
        train_mean = vectors[train_rows].mean(0)

        def normalize(rows):
            return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)

        vectors = normalize(vectors - train_mean)
        if provisional_vectors is not None:
            provisional_vectors = normalize(provisional_vectors - train_mean)
    walk_time = topology.walk_time
    is_finite_walk = (walk_time > 0) & (walk_time < _FAR_WALK_TIME)
    summary = {"n_overlap_groups": int(topology.overlap_group_of_camera.max()) + 1,
               "n_sites": int(topology.n_sites),
               "n_border_cams": int(topology.is_border_camera.sum()),
               "D_finite_median_s": float(np.median(walk_time[is_finite_walk]))
               if is_finite_walk.any() else 0.0}
    for split_name, rows in (("train", train_rows), ("test", test_rows)):
        split_table = tracklet_table.iloc[rows].reset_index(drop=True)
        stream = to_stream(split_table, vectors[rows], topology, fps,
                           None if provisional_vectors is None else provisional_vectors[rows])
        path = f"{output_prefix}_{split_name}.parquet"
        stream.save(path)
        n_people = split_table.gt_person_id.nunique()
        summary[f"{split_name}_path"] = path
        summary[f"n_{split_name}_tracklets"] = int(len(split_table))
        summary[f"n_{split_name}_people"] = int(n_people)
        log(f"{split_name}: {len(split_table)} tracklets, {n_people} people -> {path}")
        if split_name == "train" and engine_config is not None:
            tau_accept, pairwise_f1 = choose_thresholds(stream, engine_config)
            summary["tau_accept"], summary["train_pw_f1"] = float(tau_accept), float(pairwise_f1)
            log(f"chosen tau_accept on train = {tau_accept} (pairwise F1 {pairwise_f1:.3f}); "
                "freeze it for test")
    return summary
