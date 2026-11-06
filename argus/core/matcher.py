"""Candidate generation + cosine scoring over a sliding window.

For a new tracklet t at camera c, a stored tracklet u at camera c' is a candidate iff
    t.ts - D[c',c] - W <= u.te <= t.ts - D[c',c]      (different, non-overlapping cams)
    t.ts - W <= u.te <= t.ts                          (same camera: fragment)
    t.ts - W <= u.te                                  (overlap-group camera)
i.e. W is the walking *slack* on top of the minimum walking time.

Each camera keeps a ring buffer (numpy arrays sorted by t_end, because tracklets arrive
in t_end order): the window is two searchsorted calls and the scores are one
matrix-vector product per neighbouring camera. Vectors are dropped as soon as a
tracklet can no longer be a candidate of any future tracklet.
"""
from __future__ import annotations

import numpy as np

from ..common.topology import Topology


class _CamBuf:
    __slots__ = ("te", "ts", "tid", "vec", "h", "n", "base")

    def __init__(self, dim, cap=64):
        self.te = np.empty(cap)
        self.ts = np.empty(cap)
        self.tid = np.empty(cap, np.int64)
        self.vec = np.empty((cap, dim), np.float32)
        self.h = 0          # first live slot
        self.n = 0          # one past last slot
        self.base = 0       # absolute index of slot 0

    def append(self, tid, ts, te, v) -> int:
        if self.n == len(self.te):
            live = self.n - self.h
            if self.h >= live:                       # compact in place
                for a in ("te", "ts", "tid", "vec"):
                    arr = getattr(self, a)
                    arr[:live] = arr[self.h:self.n]
            else:                                    # grow
                cap = 2 * len(self.te)
                for a in ("te", "ts", "tid", "vec"):
                    arr = getattr(self, a)
                    new = np.empty((cap,) + arr.shape[1:], arr.dtype)
                    new[:live] = arr[self.h:self.n]
                    setattr(self, a, new)
            self.base += self.h
            self.n, self.h = live, 0
        i = self.n
        self.te[i], self.ts[i], self.tid[i] = te, ts, tid
        self.vec[i] = v
        self.n += 1
        return self.base + i


class Matcher:
    def __init__(self, topo: Topology, W: float, tau_keep: float, cams=None,
                 cross_site_only: bool = False, D_max: float = 600.0):
        self.W = W
        self.tau_keep = tau_keep
        cams = list(range(topo.n_cams)) if cams is None else list(cams)
        self.cams = set(cams)
        self.buf: dict[int, _CamBuf] = {}
        self.loc: dict[int, tuple] = {}             # tid -> (cam, abs index, ts, te)
        self.max_lag = 0.0
        self.nbrs = {}
        D, site = topo.D, topo.site
        for c in cams:
            lst = []
            for c2 in cams:
                if cross_site_only and site[c2] == site[c]:
                    continue
                if D[c2, c] <= D_max:
                    lst.append((c2, float(D[c2, c]), bool(topo.overlap(c2, c))))
                    self.max_lag = max(self.max_lag, float(D[c2, c]))
            self.nbrs[c] = lst
        self.n_scored = 0

    # ------------------------------------------------------------------ lookups
    def get_vec(self, tid):
        loc = self.loc.get(tid)
        if loc is None:
            return None
        b = self.buf[loc[0]]
        i = loc[1] - b.base
        return b.vec[i] if i >= b.h else None

    @property
    def vec(self):            # mapping-like view used by callers: .get(tid)
        return self

    def get(self, tid, default=None):
        v = self.get_vec(tid)
        return default if v is None else v

    def meta(self, tid):
        loc = self.loc[tid]
        return loc[0], loc[2], loc[3]

    # ------------------------------------------------------------------ scoring
    def _score(self, b, lo_i, hi_i, vec, exclude, out):
        if hi_i <= lo_i:
            return
        sc = b.vec[lo_i:hi_i] @ vec
        self.n_scored += hi_i - lo_i
        for k in np.nonzero(sc >= self.tau_keep)[0]:
            t = int(b.tid[lo_i + k])
            if exclude is None or t not in exclude:
                out.append((t, float(sc[k])))

    def candidates(self, cam, ts, vec, exclude=None):
        """Earlier tracklets that may be the same person. Returns [(tid, score)] with
        score >= tau_keep, sorted descending."""
        if cam not in self.nbrs:
            return []
        W = self.W
        out = []
        for c2, d, ov in self.nbrs[cam]:
            b = self.buf.get(c2)
            if b is None or b.n == b.h:
                continue
            te = b.te[b.h:b.n]
            if ov:
                lo = np.searchsorted(te, ts - W, "left")
                hi = len(te)
            else:
                lo = np.searchsorted(te, ts - d - W, "left")
                hi = np.searchsorted(te, ts - d, "right")
            self._score(b, b.h + lo, b.h + hi, vec, exclude, out)
        out.sort(key=lambda x: -x[1])
        return out

    def candidates_fwd(self, tid, vec, exclude=None):
        """Later tracklets whose window contains `tid` (used after a score revision)."""
        cam, ts0, te0 = self.meta(tid)
        if cam not in self.nbrs:
            return []
        W = self.W
        out = []
        for c2, d, ov in self.nbrs[cam]:
            b = self.buf.get(c2)
            if b is None or b.n == b.h:
                continue
            lo = b.h + int(np.searchsorted(b.te[b.h:b.n], te0, "left"))
            if lo >= b.n:
                continue
            ts_u = b.ts[lo:b.n]
            if ov:
                m = te0 >= ts_u - W
            else:
                m = (ts_u - d - W <= te0) & (te0 <= ts_u - d)
            idx = np.nonzero(m)[0]
            if len(idx) == 0:
                continue
            sc = b.vec[lo + idx] @ vec
            self.n_scored += len(idx)
            for k in np.nonzero(sc >= self.tau_keep)[0]:
                t = int(b.tid[lo + idx[k]])
                if t != tid and (exclude is None or t not in exclude):
                    out.append((t, float(sc[k])))
        return out

    # ------------------------------------------------------------------ state
    def add(self, tid, cam, ts, te, vec):
        if cam not in self.cams:
            return
        b = self.buf.get(cam)
        if b is None:
            b = self.buf[cam] = _CamBuf(len(vec))
        self.loc[tid] = (cam, b.append(tid, ts, te, vec), ts, te)

    def update_vec(self, tid, vec) -> bool:
        loc = self.loc.get(tid)
        if loc is None:
            return False
        b = self.buf[loc[0]]
        i = loc[1] - b.base
        if i < b.h:
            return False
        b.vec[i] = vec
        return True

    def prune(self, now, max_dur):
        """Drop vectors that can no longer be candidates of any future tracklet."""
        horizon = now - max_dur - self.max_lag - self.W - 1.0
        for b in self.buf.values():
            if b.n == b.h:
                continue
            k = int(np.searchsorted(b.te[b.h:b.n], horizon, "left"))
            if k:
                for t in b.tid[b.h:b.h + k].tolist():
                    self.loc.pop(t, None)
                b.h += k

    def n_live(self) -> int:
        return sum(b.n - b.h for b in self.buf.values())
