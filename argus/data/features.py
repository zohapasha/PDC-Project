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
For videos, pass a frame_loader that seeks into the per-camera video (cache sequential reads).
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


def embed_tracklets(tr, boxes, frame_loader, K: int = 8, extractor=None, batch: int = 128):
    """Returns (final_vectors, provisional_vectors), both (n_tracklets, 512) float32."""
    extractor = extractor or make_extractor()
    jobs = []                                           # (tracklet idx, half flag, cam, frame, box)
    for i, r in tr.iterrows():
        rows = r["rows"]
        idx = np.linspace(0, len(rows) - 1, min(K, len(rows))).round().astype(int)
        for j, k in enumerate(idx):
            b = boxes.loc[rows[k]]
            jobs.append((i, j < max(1, len(idx) // 2), int(b.cam), int(b.frame),
                         (b.x1, b.y1, b.x2, b.y2)))
    jobs.sort(key=lambda x: (x[2], x[3]))               # sequential reads per camera
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
    n_tr = len(tr)
    final = np.zeros((n_tr, feats.shape[1]), np.float32)
    prov = np.zeros_like(final)
    cnt = np.zeros(n_tr)
    cnt_p = np.zeros(n_tr)
    for (i, first_half, *_), f in zip(jobs, feats):
        final[i] += f
        cnt[i] += 1
        if first_half:
            prov[i] += f
            cnt_p[i] += 1
    unit = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return unit(final), unit(prov)


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
