"""Candidate generation + cosine scoring over a sliding window.

For a new tracklet t at camera c, a stored tracklet u at camera c' is a candidate iff
    t.ts - D[c',c] - W <= u.te <= t.ts - D[c',c]      (different, non-overlapping cams)
    t.ts - W <= u.te <= t.ts                          (same camera: fragment)
    t.ts - W <= u.te                                  (overlap-group camera)
i.e. W is the walking *slack* on top of the minimum walking time.

Stored tracklets live in one ring buffer per *site* (numpy arrays sorted by t_end:
tracklets of one site arrive in t_end order, also at the multi-process hub where sites
interleave). A query is one searchsorted window per buffer, one vectorised mask with
per-camera bounds, and one matrix-vector product over the surviving rows. Vectors are
dropped as soon as a tracklet can no longer be a candidate of any future tracklet.
"""
from __future__ import annotations

import numpy as np

from ..common.topology import Topology


class _Buf:
    __slots__ = ("te", "ts", "tid", "cam", "vec", "h", "n", "base")

    def __init__(self, dim, cap=256):
        self.te = np.empty(cap)
        self.ts = np.empty(cap)
        self.tid = np.empty(cap, np.int64)
        self.cam = np.empty(cap, np.int64)
        self.vec = np.empty((cap, dim), np.float32)
        self.h = 0          # first live slot
        self.n = 0          # one past last slot
        self.base = 0       # absolute index of slot 0

    def append(self, tid, cam, ts, te, v) -> int:
        if self.n == len(self.te):
            live = self.n - self.h
            if self.h >= live:                       # compact in place
                for a in ("te", "ts", "tid", "cam", "vec"):
                    arr = getattr(self, a)
                    arr[:live] = arr[self.h:self.n]
            else:                                    # grow
                cap = 2 * len(self.te)
                for a in ("te", "ts", "tid", "cam", "vec"):
                    arr = getattr(self, a)
                    new = np.empty((cap,) + arr.shape[1:], arr.dtype)
                    new[:live] = arr[self.h:self.n]
                    setattr(self, a, new)
            self.base += self.h
            self.n, self.h = live, 0
        i = self.n
        self.te[i], self.ts[i], self.tid[i], self.cam[i] = te, ts, tid, cam
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
        n = topo.n_cams
        self.site = topo.site.astype(np.int64)
        self.buf: dict[int, _Buf] = {}               # site -> buffer
        self.loc: dict[int, tuple] = {}              # tid -> (cam, abs index, ts, te)
        self.max_lag = 0.0
        D = topo.D
        # per query camera c, per stored camera c2: u.te in [ts - lo[c2], ts - hi[c2]]
        self.lo = {}
        self.hi = {}
        self.dd = {}                                 # D[c2, c] for neighbours, nan otherwise
        self.ov = {}
        self.srcs = {}                               # sites holding neighbour cameras
        self.span = {}                               # (max lo, min hi) over neighbours
        for c in cams:
            lo = np.full(n, -np.inf)
            hi = np.full(n, np.inf)
            dd = np.full(n, np.nan)
            ov = np.zeros(n, bool)
            for c2 in cams:
                if cross_site_only and topo.site[c2] == topo.site[c]:
                    continue
                if D[c2, c] <= D_max:
                    d = float(D[c2, c])
                    dd[c2] = d
                    if topo.overlap(c2, c):
                        ov[c2] = True
                        lo[c2], hi[c2] = W, -np.inf
                    else:
                        lo[c2], hi[c2] = d + W, d
                    self.max_lag = max(self.max_lag, d)
            nb = np.isfinite(lo)
            self.lo[c], self.hi[c], self.dd[c], self.ov[c] = lo, hi, dd, ov
            self.srcs[c] = sorted({int(self.site[c2]) for c2 in np.nonzero(nb)[0]})
            self.span[c] = ((float(lo[nb].max()), float(hi[nb].min())) if nb.any()
                            else (-np.inf, np.inf))
        self.n_scored = 0

    # ------------------------------------------------------------------ lookups
    def get_vec(self, tid):
        loc = self.loc.get(tid)
        if loc is None:
            return None
        b = self.buf[int(self.site[loc[0]])]
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
    def _emit(self, b, s0, s1, mask, vec, exclude, out, skip=None):
        """Score the contiguous rows s0:s1 (a view, no copy); keep masked rows."""
        if s1 <= s0 or not mask.any():
            return
        sc = b.vec[s0:s1] @ vec
        self.n_scored += int(mask.sum())
        keep = np.nonzero(mask & (sc >= self.tau_keep))[0]
        if len(keep) == 0:
            return
        tids = b.tid[s0 + keep].tolist()
        scs = sc[keep].tolist()
        for t, sv in zip(tids, scs):
            if t != skip and (exclude is None or t not in exclude):
                out.append((t, sv))

    def candidates(self, cam, ts, vec, exclude=None):
        """Earlier tracklets that may be the same person. Returns [(tid, score)] with
        score >= tau_keep, sorted descending."""
        srcs = self.srcs.get(cam)
        if not srcs:
            return []
        lo_c, hi_c = self.lo[cam], self.hi[cam]
        span_lo, span_hi = self.span[cam]
        out = []
        for sid in srcs:
            b = self.buf.get(sid)
            if b is None or b.n == b.h:
                continue
            te = b.te[b.h:b.n]
            i = int(np.searchsorted(te, ts - span_lo, "left"))
            j = len(te) if span_hi == -np.inf else int(np.searchsorted(te, ts - span_hi, "right"))
            if j <= i:
                continue
            cs = b.cam[b.h + i:b.h + j]
            tw = te[i:j]
            m = (tw >= ts - lo_c[cs]) & (tw <= ts - hi_c[cs])
            self._emit(b, b.h + i, b.h + j, m, vec, exclude, out)
        out.sort(key=lambda x: -x[1])
        return out

    def candidates_fwd(self, tid, vec, exclude=None):
        """Later tracklets whose window contains `tid` (used after a score revision)."""
        cam, ts0, te0 = self.meta(tid)
        srcs = self.srcs.get(cam)
        if not srcs:
            return []
        W = self.W
        dd, ov = self.dd[cam], self.ov[cam]
        out = []
        for sid in srcs:
            b = self.buf.get(sid)
            if b is None or b.n == b.h:
                continue
            lo = b.h + int(np.searchsorted(b.te[b.h:b.n], te0, "left"))
            if lo >= b.n:
                continue
            cs = b.cam[lo:b.n]
            ts_u = b.ts[lo:b.n]
            d = dd[cs]                               # nan for non-neighbours -> False
            o = ov[cs]
            m = np.where(o, te0 >= ts_u - W, (ts_u - d - W <= te0) & (te0 <= ts_u - d))
            m &= ~np.isnan(d)
            self._emit(b, lo, b.n, m, vec, exclude, out, skip=tid)
        return out

    # ------------------------------------------------------------------ state
    def add(self, tid, cam, ts, te, vec):
        if cam not in self.cams:
            return
        sid = int(self.site[cam])
        b = self.buf.get(sid)
        if b is None:
            b = self.buf[sid] = _Buf(len(vec))
        self.loc[tid] = (cam, b.append(tid, cam, ts, te, vec), ts, te)

    def update_vec(self, tid, vec) -> bool:
        loc = self.loc.get(tid)
        if loc is None:
            return False
        b = self.buf[int(self.site[loc[0]])]
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
