"""Shared real-data utilities: GT boxes -> tracklets, data-driven topology estimation,
and conversion to the same Stream format the synthetic generator produces.

Box table columns (pandas): cam, frame, pid, x1, y1, x2, y2
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.topology import Topology, overlap_groups
from ..gen.campus import Stream


def boxes_to_tracklets(boxes: pd.DataFrame, fps: float, gap_s: float = 1.0,
                       p_frag: float = 0.0, max_len_s: float = 120.0, seed: int = 0) -> pd.DataFrame:
    """Group GT boxes per (cam, pid) in time order; start a new tracklet when the gap
    exceeds gap_s or the tracklet exceeds max_len_s. Optionally fragment further with
    probability p_frag (simulated single-camera tracker breaks).
    Returns one row per tracklet: cam, gt_pid, f_start, f_end, t_start, t_end, frames(list)."""
    rng = np.random.default_rng(seed)
    gap = gap_s * fps
    out = []
    for (cam, pid), g in boxes.sort_values("frame").groupby(["cam", "pid"], sort=False):
        fr = g.frame.to_numpy()
        cut = np.nonzero(np.diff(fr) > gap)[0] + 1
        for seg in np.split(np.arange(len(fr)), cut):
            # length cap
            start = 0
            while start < len(seg):
                end = start
                while end + 1 < len(seg) and fr[seg[end + 1]] - fr[seg[start]] <= max_len_s * fps:
                    end += 1
                piece = seg[start:end + 1]
                if len(piece) >= 4 and rng.random() < p_frag:
                    k = int(rng.integers(2, len(piece) - 1))
                    parts = [piece[:k], piece[k:]]
                else:
                    parts = [piece]
                for pp in parts:
                    out.append({"cam": int(cam), "gt_pid": int(pid), "f_start": int(fr[pp[0]]),
                                "f_end": int(fr[pp[-1]]), "frames": fr[pp].tolist(),
                                "rows": g.index[pp].tolist()})
                start = end + 1
    t = pd.DataFrame(out)
    t["t_start"] = t.f_start / fps
    t["t_end"] = (t.f_end + 1) / fps
    return t.sort_values(["t_end", "t_start", "cam"]).reset_index(drop=True)


def estimate_topology(train: pd.DataFrame, n_cams: int, site_of_cam, margin: float = 0.1,
                      default_far: float = 1e4, entrance=None) -> Topology:
    """Walking times from the TRAINING split: for each ordered camera pair the minimum
    observed transition time of the same GT person, minus a margin; overlap groups = pairs
    where one person is seen simultaneously; then metric closure (shortest paths)."""
    W = np.full((n_cams, n_cams), np.inf)
    np.fill_diagonal(W, 0.0)
    overlaps = set()
    for _, g in train.sort_values("t_start").groupby("gt_pid"):
        rec = g[["t_start", "t_end", "cam"]].to_numpy()
        for i in range(len(rec)):
            for j in range(i + 1, len(rec)):
                s1, e1, c1 = rec[i]
                s2, e2, c2 = rec[j]
                c1, c2 = int(c1), int(c2)
                if c1 == c2:
                    continue
                if s2 < e1:                                  # simultaneous
                    overlaps.add((min(c1, c2), max(c1, c2)))
                else:
                    gap = (s2 - e1) * (1 - margin)
                    if gap < W[c1, c2]:
                        W[c1, c2] = W[c2, c1] = gap
                if s2 - e1 > 600:
                    break
    for a, b in overlaps:
        W[a, b] = W[b, a] = 0.0
    D = W.copy()
    D[np.isinf(D)] = default_far
    for k in range(n_cams):
        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
    site = np.asarray(site_of_cam)
    edges = [(a, b, float(D[a, b])) for a in range(n_cams) for b in range(a + 1, n_cams)
             if D[a, b] < default_far]
    # one "building" per camera keeps the generic site mapping (site_of_building = site)
    topo = Topology(n_cams, D, overlap_groups(n_cams, sorted(overlaps)), np.arange(n_cams),
                    np.ones(n_cams, bool) if entrance is None else np.asarray(entrance),
                    site, edges)
    return topo


def to_stream(tr: pd.DataFrame, vecs: np.ndarray, topo: Topology, fps: float,
              revise_vecs: np.ndarray | None = None, revise_delay: float = 10.0) -> Stream:
    """tr sorted by t_end (tid = row index). revise_vecs (optional): provisional vectors
    (e.g. from the first half of the tracklet); the final vector arrives revise_delay s
    later as a revision event."""
    n = len(tr)
    te = tr.t_end.to_numpy(float)
    ev_t, ev_type, ev_tid, ev_vec = [te], [np.zeros(n, np.int8)], [np.arange(n)], []
    if revise_vecs is not None:
        ev_vec = [revise_vecs, vecs]
        ev_t.append(te + revise_delay)
        ev_type.append(np.ones(n, np.int8))
        ev_tid.append(np.arange(n))
    else:
        ev_vec = [vecs]
    t, et, tid = np.concatenate(ev_t), np.concatenate(ev_type), np.concatenate(ev_tid)
    v = np.concatenate(ev_vec).astype(np.float32)
    o = np.lexsort((tid, et, t))
    return Stream(topo, t[o], et[o], tid[o], v[o], tr.cam.to_numpy(np.int64),
                  tr.t_start.to_numpy(float), te, tr.gt_pid.to_numpy(np.int64),
                  {"gen": {"fps": fps}, "n_tracklets": n, "source": "real"})


def choose_thresholds(stream_train: Stream, base_cfg: dict, grid=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75)):
    """Pick tau_accept on the training split by pairwise F1; freeze it for test."""
    from ..eval.metrics import evaluate
    from ..sync.sim import run_single
    best = None
    for tau in grid:
        cfg = dict(base_cfg, tau_accept=tau, tau_keep=min(base_cfg.get("tau_keep", 0.5), tau))
        node, _ = run_single(stream_train, cfg)
        f1 = evaluate(node.engine.partition(), stream_train)["pw_f1"]
        if best is None or f1 > best[1]:
            best = (tau, f1)
    return best
