"""Appearance features for real tracklets (run on a GPU, e.g. Kaggle).

For each tracklet: sample K evenly spaced frames, crop the box (5% padding), embed with
OSNet (torchreid, MSMT17/Market weights), average, L2-normalise -> float32 vector.
Optionally also returns a *provisional* vector from the first half of the frames, used to
create realistic score revisions (evidence-driven deletions).

    pip install torch torchvision opencv-python gdown
    pip install git+https://github.com/KaiyangZhou/deep-person-reid.git

Usage (WILDTRACK):
    from argus.data import wildtrack, tracklets, features
    boxes = wildtrack.load_boxes(ROOT)
    tracklet_table = tracklets.boxes_to_tracklets(boxes, fps=wildtrack.ANNOTATION_FPS, gap_s=1.0)
    vectors, provisional = features.embed_tracklets(
        tracklet_table, boxes,
        frame_loader=lambda camera_id, frame: cv2.imread(wildtrack.image_path(ROOT, camera_id, frame)))
    np.save("wildtrack_vec.npy", vectors); np.save("wildtrack_prov.npy", provisional)
For videos (one file per camera), use embed_tracklets_video(tracklet_table, boxes, video_paths).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

FEATURE_DIM = 512


class _CropJob(NamedTuple):
    tracklet_index: int
    in_first_half: bool       # frame belongs to the first half (provisional vector)
    camera_id: int
    frame: int
    box: tuple                # (x1, y1, x2, y2)


def make_extractor(model_name: str = "osnet_ain_x1_0", weights: str | None = None,
                   device: str = "cuda"):
    try:
        from torchreid.utils import FeatureExtractor          # GitHub deep-person-reid
    except ImportError:
        from torchreid.reid.utils import FeatureExtractor     # PyPI torchreid 0.2.x
    return FeatureExtractor(model_name=model_name, model_path=weights or "", device=device)


def _crop(image, x1, y1, x2, y2, pad=0.05):
    image_height, image_width = image.shape[:2]
    box_width, box_height = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - pad * box_width))
    x2 = int(min(image_width, x2 + pad * box_width))
    y1 = int(max(0, y1 - pad * box_height))
    y2 = int(min(image_height, y2 + pad * box_height))
    return image[y1:y2, x1:x2, ::-1]          # BGR -> RGB


def _make_crop_jobs(tracklet_table, boxes, frames_per_tracklet):
    """One job per sampled frame: frames_per_tracklet evenly spaced frames per tracklet,
    sorted by (camera, frame) so every camera is read front to back once."""
    jobs = []
    x1, y1, x2, y2 = (boxes[column].to_numpy() for column in ("x1", "y1", "x2", "y2"))
    camera_ids, frames = boxes["camera_id"].to_numpy(), boxes["frame"].to_numpy()
    position_of_row = {row_label: position for position, row_label in enumerate(boxes.index)}
    for tracklet_index, box_rows in enumerate(tracklet_table["box_rows"]):
        sampled = np.linspace(0, len(box_rows) - 1,
                              min(frames_per_tracklet, len(box_rows))).round().astype(int)
        for sample_number, row_offset in enumerate(sampled):
            position = position_of_row[box_rows[row_offset]]
            jobs.append(_CropJob(tracklet_index, sample_number < max(1, len(sampled) // 2),
                                 int(camera_ids[position]), int(frames[position]),
                                 (x1[position], y1[position], x2[position], y2[position])))
    jobs.sort(key=lambda job: (job.camera_id, job.frame))
    return jobs


def _aggregate(jobs, job_features, n_tracklets):
    final_vectors = np.zeros((n_tracklets, job_features.shape[1]), np.float32)
    provisional_vectors = np.zeros_like(final_vectors)
    for job, feature in zip(jobs, job_features):
        final_vectors[job.tracklet_index] += feature
        if job.in_first_half:
            provisional_vectors[job.tracklet_index] += feature

    def normalize(vectors):
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    return normalize(final_vectors), normalize(provisional_vectors)


def embed_tracklets(tracklet_table, boxes, frame_loader, frames_per_tracklet: int = 8,
                    extractor=None, batch_size: int = 128):
    """Returns (final_vectors, provisional_vectors), both (n_tracklets, 512) float32.
    tracklet_table must have a default RangeIndex (as boxes_to_tracklets returns)."""
    extractor = extractor or make_extractor()
    jobs = _make_crop_jobs(tracklet_table, boxes, frames_per_tracklet)
    job_features = np.zeros((len(jobs), FEATURE_DIM), np.float32)
    crops, crop_job_indices, frame_cache = [], [], {}
    for job_index, job in enumerate(jobs):
        key = (job.camera_id, job.frame)
        if key not in frame_cache:
            frame_cache.clear()
            frame_cache[key] = frame_loader(job.camera_id, job.frame)
        crops.append(_crop(frame_cache[key], *job.box))
        crop_job_indices.append(job_index)
        if len(crops) == batch_size or job_index == len(jobs) - 1:
            batch_features = extractor(crops).cpu().numpy()
            job_features[crop_job_indices] = batch_features[:, :FEATURE_DIM]
            crops, crop_job_indices = [], []
    return _aggregate(jobs, job_features, len(tracklet_table))


def _read_camera_crops(video_path, camera_jobs, seek_gap=300, size=(128, 256)):
    """Decode one video front to back and return [(job index, RGB crop)] for its jobs.
    Frames are grabbed sequentially (seeking only across long gaps), crops are resized
    to the re-ID input size right away to bound memory."""
    import cv2
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1 << 30
    crops = []
    held_frame_index, held_image = -1, None      # the frame currently held in held_image
    for job_index, job in camera_jobs:
        frame_index = min(max(job.frame, 0), n_frames - 1)
        if frame_index != held_frame_index:
            if frame_index < held_frame_index or frame_index - held_frame_index > seek_gap:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                held_frame_index = frame_index - 1
            while held_frame_index < frame_index:
                if not capture.grab():
                    break
                held_frame_index += 1
            ok, decoded = capture.retrieve()
            if ok:
                held_image = decoded
        if held_image is None:
            continue
        crop = _crop(held_image, *job.box)
        if crop.size == 0:
            continue
        crops.append((job_index, cv2.resize(np.ascontiguousarray(crop), size)))
    capture.release()
    return crops


def embed_tracklets_video(tracklet_table, boxes, video_paths, frames_per_tracklet: int = 8,
                          extractor=None, batch_size: int = 256, workers: int = 4, log=print):
    """Like embed_tracklets, for one video per camera (video_paths[camera_id]). Cameras are
    decoded in parallel threads (OpenCV releases the GIL); the GPU embeds each camera's
    crops as soon as that camera is done. Jobs whose frame cannot be read keep a zero
    feature (their tracklet is averaged over the remaining frames)."""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    extractor = extractor or make_extractor()
    jobs = _make_crop_jobs(tracklet_table, boxes, frames_per_tracklet)
    jobs_by_camera = {}
    for job_index, job in enumerate(jobs):
        jobs_by_camera.setdefault(job.camera_id, []).append((job_index, job))
    job_features = np.zeros((len(jobs), FEATURE_DIM), np.float32)
    start, n_missing = time.time(), 0
    with ThreadPoolExecutor(workers) as pool:
        camera_of_future = {pool.submit(_read_camera_crops, video_paths[camera_id], camera_jobs): camera_id
                            for camera_id, camera_jobs in jobs_by_camera.items()}
        for future in as_completed(camera_of_future):
            camera_id = camera_of_future[future]
            crops = future.result()
            n_missing += len(jobs_by_camera[camera_id]) - len(crops)
            for batch_start in range(0, len(crops), batch_size):
                batch = crops[batch_start:batch_start + batch_size]
                batch_features = extractor([crop for _, crop in batch]).cpu().numpy()
                job_features[[job_index for job_index, _ in batch]] = batch_features[:, :FEATURE_DIM]
            log(f"    cam {camera_id}: {len(crops)} crops ({time.time() - start:.0f} s)")
    if n_missing:
        log(f"    {n_missing} of {len(jobs)} crops unreadable (zero features, ignored in the mean)")
    return _aggregate(jobs, job_features, len(tracklet_table))


def similarity_histograms(stream, max_pairs: int = 200000, seed: int = 0):
    """Cosine similarity of true-match vs false-match tracklet pairs (calibration figure).
    Uses the final vector of each tracklet."""
    rng = np.random.default_rng(seed)
    final_vector_of = {}
    for i in range(len(stream.event_times)):
        final_vector_of[int(stream.event_tracklet_ids[i])] = stream.event_vectors[i]
    tracklet_ids = np.array(sorted(final_vector_of))
    vectors = np.stack([final_vector_of[tracklet_id] for tracklet_id in tracklet_ids])
    gt_person_ids = stream.gt_person_ids[tracklet_ids]
    index_a = rng.integers(0, len(tracklet_ids), max_pairs)
    index_b = rng.integers(0, len(tracklet_ids), max_pairs)
    keep = index_a != index_b
    index_a, index_b = index_a[keep], index_b[keep]
    pair_similarity = np.einsum("ij,ij->i", vectors[index_a], vectors[index_b])
    same_person = gt_person_ids[index_a] == gt_person_ids[index_b]
    # oversample positives so both histograms are well populated
    indices_of_person = {}
    for index, person_id in enumerate(gt_person_ids):
        indices_of_person.setdefault(person_id, []).append(index)
    positive_similarity = []
    for indices in indices_of_person.values():
        if len(indices) > 1:
            for _ in range(min(50, len(indices))):
                i, j = rng.choice(indices, 2, replace=False)
                positive_similarity.append(float(vectors[i] @ vectors[j]))
    return np.array(positive_similarity), pair_similarity[~same_person]
