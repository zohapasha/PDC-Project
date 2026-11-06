"""Clustering / tracking metrics computed from contingency tables (never enumerating pairs)."""
from __future__ import annotations

import numpy as np

from ..common.constraints import Checker


def _labels(part, tids):
    """partition dict tid->label  ->  int array aligned with tids."""
    return np.fromiter((part[t] for t in tids), dtype=np.int64, count=len(tids))


def _contingency(pred, gt, weight=None):
    _, p = np.unique(pred, return_inverse=True)
    _, g = np.unique(gt, return_inverse=True)
    key = p.astype(np.int64) * (g.max() + 1) + g
    uk, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv, weights=weight)
    return p, g, uk // (g.max() + 1), uk % (g.max() + 1), cnt


def _c2(x):
    x = np.asarray(x, dtype=np.float64)
    return (x * (x - 1) / 2).sum()


def pairwise(pred, gt) -> dict:
    p, g, _, _, nij = _contingency(pred, gt)
    tp = _c2(nij)
    pp = _c2(np.bincount(p))
    gp = _c2(np.bincount(g))
    P = tp / pp if pp else 1.0
    R = tp / gp if gp else 1.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"pw_p": P, "pw_r": R, "pw_f1": F}


def bcubed(pred, gt) -> dict:
    p, g, pi, gi, nij = _contingency(pred, gt)
    a, b = np.bincount(p), np.bincount(g)
    N = len(pred)
    P = (nij * nij / a[pi]).sum() / N
    R = (nij * nij / b[gi]).sum() / N
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"b3_p": P, "b3_r": R, "b3_f1": F}


def idf1(pred, gt, frames) -> dict:
    """IDF1 with GT boxes: every detection is a frame of some tracklet, so
    IDF1 = IDTP / total_frames where IDTP comes from the optimal one-to-one assignment
    of predicted identities to GT identities (weights = shared frames)."""
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import coo_matrix
    p, g, pi, gi, w = _contingency(pred, gt, weight=frames)
    n_p, n_g = p.max() + 1, g.max() + 1
    M = coo_matrix((w, (pi, gi)), shape=(n_p, n_g)).tocsc()
    if n_p * n_g > 4e7:      # keep, per GT id, its 8 heaviest predicted ids (approximation)
        rows = set()
        for j in range(n_g):
            col = M.getcol(j)
            if col.nnz:
                top = col.indices[np.argsort(-col.data)[:8]]
                rows.update(top.tolist())
        rows = np.array(sorted(rows))
        M = M.tocsr()[rows]
    dense = M.toarray()
    r, c = linear_sum_assignment(-dense)
    idtp = dense[r, c].sum()
    return {"idf1": idtp / float(np.sum(frames))}


def violations(part: dict, visits_of: dict, checker: Checker) -> int:
    """Number of predicted persons whose full visit list is physically infeasible."""
    groups = {}
    for t, lab in part.items():
        groups.setdefault(lab, []).append(visits_of[t])
    bad = 0
    for vs in groups.values():
        if len(vs) > 1:
            vs.sort()
            if not checker.feasible(vs):
                bad += 1
    return bad


def agreement(part_a: dict, part_b: dict) -> float:
    """Pairwise F1 between two predicted partitions over their common tracklets."""
    tids = [t for t in part_a if t in part_b]
    return pairwise(_labels(part_a, tids), _labels(part_b, tids))["pw_f1"]


def evaluate(part: dict, stream, checker: Checker | None = None) -> dict:
    tids = np.arange(stream.n_tracklets)
    tids = tids[[t in part for t in tids]] if len(part) != stream.n_tracklets else tids
    pred = _labels(part, tids)
    gt = stream.gt[tids]
    out = {}
    out.update(pairwise(pred, gt))
    out.update(bcubed(pred, gt))
    fps = stream.meta.get("gen", {}).get("fps", 10.0)
    frames = np.maximum(1.0, np.round((stream.te[tids] - stream.ts[tids]) * fps))
    out.update(idf1(pred, gt, frames))
    out["n_pred"] = int(len(np.unique(pred)))
    out["n_gt"] = int(len(np.unique(gt)))
    if checker is not None:
        vis = {int(t): (stream.ts[t], stream.te[t], int(stream.cam[t]), int(t)) for t in tids}
        out["violations"] = violations(part, vis, checker)
    return out
