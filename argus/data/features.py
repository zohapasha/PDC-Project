"""Appearance features for real tracklets (run on Colab GPU).

For each tracklet: sample K evenly spaced frames, crop the box (5% padding), embed with
OSNet (torchreid, MSMT17/Market weights), average, L2-normalise -> float32 vector.
Optionally also returns a *provisional* vector from the first half of the frames, used to
create realistic score revisions (evidence-driven deletions).

    pip install torch torchvision opencv-python gdown
    pip install git+https://github.com/KaiyangZhou/deep-person-reid.git

Usage (WILDTRACK):
    from argus.data import wildtrack, tracklets, features
    boxes = wildtrack.load_boxes(ROOT)
    tr = tracklets.boxes_to_tracklets(boxes, fps=wildtrack.ANN_FPS, gap_s=1.0)
    vec, prov = features.embed_tracklets(tr, boxes,
                    frame_loader=lambda cam, f: cv2.imread(wildtrack.image_path(ROOT, cam, f)))
    np.save("wildtrack_vec.npy", vec); np.save("wildtrack_prov.npy", prov)
For videos (one file per camera), use embed_tracklets_video(tr, boxes, video_paths).
"""
from __future__ import annotations

import numpy as np


def make_extractor(model_name: str = "osnet_ain_x1_0", weights: str | None = None, device: str = "cuda"):
    try:
        from torchreid.utils import FeatureExtractor          # GitHub deep-person-reid
    except ImportError:
        from torchreid.reid.utils import FeatureExtractor     # PyPI torchreid 0.2.x
    return FeatureExtractor(model_name=model_name, model_path=weights or "", device=device)


def _crop(img, x1, y1, x2, y2, pad=0.05):
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - pad * bw)); x2 = int(min(w, x2 + pad * bw))
    y1 = int(max(0, y1 - pad * bh)); y2 = int(min(h, y2 + pad * bh))
    return img[y1:y2, x1:x2, ::-1]          # BGR -> RGB


def _make_jobs(tr, boxes, K):
    """(tracklet idx, first-half flag, cam, frame, box) for K evenly spaced frames per
    tracklet, sorted by (cam, frame) so every camera is read front to back once."""
    jobs = []
    x1, y1, x2, y2 = (boxes[c].to_numpy() for c in ("x1", "y1", "x2", "y2"))
    cam, frame = boxes["cam"].to_numpy(), boxes["frame"].to_numpy()
    pos = {ix: n for n, ix in enumerate(boxes.index)}
    for i, rows in enumerate(tr["rows"]):
        idx = np.linspace(0, len(rows) - 1, min(K, len(rows))).round().astype(int)
        for j, k in enumerate(idx):
            b = pos[rows[k]]
            jobs.append((i, j < max(1, len(idx) // 2), int(cam[b]), int(frame[b]),
                         (x1[b], y1[b], x2[b], y2[b])))
    jobs.sort(key=lambda x: (x[2], x[3]))
    return jobs


def _aggregate(jobs, feats, n_tr):
    final = np.zeros((n_tr, feats.shape[1]), np.float32)
    prov = np.zeros_like(final)
    for (i, first_half, *_), f in zip(jobs, feats):
        final[i] += f
        if first_half:
            prov[i] += f
    unit = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return unit(final), unit(prov)


def embed_tracklets(tr, boxes, frame_loader, K: int = 8, extractor=None, batch: int = 128):
    """Returns (final_vectors, provisional_vectors), both (n_tracklets, 512) float32.
    tr must have a default RangeIndex (as boxes_to_tracklets returns)."""
    extractor = extractor or make_extractor()
    jobs = _make_jobs(tr, boxes, K)
    feats = np.zeros((len(jobs), 512), np.float32)
    crops, pos, cache = [], [], {}
    for n, (_, _, cam, frame, bb) in enumerate(jobs):
        key = (cam, frame)
        if key not in cache:
            cache.clear()
            cache[key] = frame_loader(cam, frame)
        crops.append(_crop(cache[key], *bb))
        pos.append(n)
        if len(crops) == batch or n == len(jobs) - 1:
            f = extractor(crops).cpu().numpy()
            feats[pos] = f[:, :512]
            crops, pos = [], []
    return _aggregate(jobs, feats, len(tr))


def _read_camera_crops(path, cam_jobs, seek_gap=300, size=(128, 256)):
    """Decode one video front to back and return [(job index, RGB crop)] for its jobs.
    Frames are grabbed sequentially (seeking only across long gaps), crops are resized
    to the re-ID input size right away to bound memory."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1 << 30
    out, pos, img = [], -1, None              # pos = index of the frame held in img
    for n, (_, _, _, frame, bb) in cam_jobs:
        f = min(max(frame, 0), n_frames - 1)
        if f != pos:
            if f < pos or f - pos > seek_gap:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                pos = f - 1
            while pos < f:
                if not cap.grab():
                    break
                pos += 1
            ok, frm = cap.retrieve()
            if ok:
                img = frm
        if img is None:
            continue
        c = _crop(img, *bb)
        if c.size == 0:
            continue
        out.append((n, cv2.resize(np.ascontiguousarray(c), size)))
    cap.release()
    return out


def embed_tracklets_video(tr, boxes, video_paths, K: int = 8, extractor=None,
                          batch: int = 256, workers: int = 4, log=print):
    """Like embed_tracklets, for one video per camera (video_paths[cam]). Cameras are
    decoded in parallel threads (OpenCV releases the GIL); the GPU embeds each camera's
    crops as soon as that camera is done. Jobs whose frame cannot be read keep a zero
    feature (their tracklet is averaged over the remaining frames)."""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    extractor = extractor or make_extractor()
    jobs = _make_jobs(tr, boxes, K)
    by_cam = {}
    for n, j in enumerate(jobs):
        by_cam.setdefault(j[2], []).append((n, j))
    feats = np.zeros((len(jobs), 512), np.float32)
    t0, missing = time.time(), 0
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(_read_camera_crops, video_paths[c], cj): c for c, cj in by_cam.items()}
        for fut in as_completed(futs):
            cam = futs[fut]
            got = fut.result()
            missing += len(by_cam[cam]) - len(got)
            for s in range(0, len(got), batch):
                part = got[s:s + batch]
                f = extractor([c for _, c in part]).cpu().numpy()
                feats[[n for n, _ in part]] = f[:, :512]
            log(f"    cam {cam}: {len(got)} crops ({time.time() - t0:.0f} s)")
    if missing:
        log(f"    {missing} of {len(jobs)} crops unreadable (zero features, ignored in the mean)")
    return _aggregate(jobs, feats, len(tr))


def similarity_histograms(stream, max_pairs: int = 200000, seed: int = 0):
    """Cosine similarity of true-match vs false-match tracklet pairs (calibration figure).
    Uses the final vector of each tracklet."""
    rng = np.random.default_rng(seed)
    final = {}
    for i in range(len(stream.time)):
        final[int(stream.tid[i])] = stream.vec[i]
    tids = np.array(sorted(final))
    V = np.stack([final[t] for t in tids])
    gt = stream.gt[tids]
    a = rng.integers(0, len(tids), max_pairs)
    b = rng.integers(0, len(tids), max_pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    s = np.einsum("ij,ij->i", V[a], V[b])
    same = gt[a] == gt[b]
    # oversample positives so both histograms are well populated
    by_pid = {}
    for k, g in enumerate(gt):
        by_pid.setdefault(g, []).append(k)
    pos = []
    for idx in by_pid.values():
        if len(idx) > 1:
            for _ in range(min(50, len(idx))):
                i, j = rng.choice(idx, 2, replace=False)
                pos.append(float(V[i] @ V[j]))
    return np.array(pos), s[~same]
