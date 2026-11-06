"""Site and hub logic, independent of transport (used in-process and by mp.py)."""
from __future__ import annotations

import numpy as np

from ..common.topology import Topology
from ..common.types import INF
from ..core.node import IdentityNode
from . import protocol as P


class Site:
    def __init__(self, topo: Topology, sid: int, cfg: dict, mode: str = "full"):
        self.sid = sid
        cams = np.nonzero(topo.site == sid)[0].tolist()
        self.node = IdentityNode(topo, cfg, match_cams=cams, emit=True, mode=mode)
        self.border = topo.border

    def _drain(self, out):
        eng = self.node.engine
        for d in eng.out:
            if d[0] == "-":
                out.append([P.ED, d[1], d[2]])
            else:                                 # "+" new tree match, "w" its new score
                out.append([P.EA, d[1], d[2], float(d[3])])
        eng.out = []
        return out

    def on_tracklet(self, tid, cam, ts, te, vec):
        self.node.on_tracklet(tid, cam, ts, te, vec)
        if self.border[cam]:
            out = [[P.BT, tid, cam, ts, te, P.vec_bytes(vec)]]
        else:
            out = [[P.TR, tid, cam, ts, te]]
        return self._drain(out)

    def on_revise(self, tid, cam, vec):
        self.node.on_revise(tid, vec)
        out = self._drain([])
        if self.border[cam]:
            out.append([P.BR, tid, P.vec_bytes(vec)])
        return out

    def on_split(self, a, b):
        """Hub SplitRequest: delete and block the match; return the resulting deltas."""
        self.node.engine.remove_match(a, b, block=True)
        return self._drain([])

    def table(self):
        """Full person table (B6)."""
        return [[t, v.person] for t, v in self.node.engine.V.items()]


class Hub:
    """Two-level hub: same engine; vertices = all tracklets (metadata), cross-site
    matching over border tracklets only, plus every site tree match.

    ownership="b" (default): site matches carry their real score; the hub resolves
    conflicts like a single instance and returns SplitRequests for site matches it rejects.
    ownership="a": site matches are +inf edges (never cut by the hub)."""

    def __init__(self, topo: Topology, cfg: dict, emit_belongs: bool = False):
        border = np.nonzero(topo.border)[0].tolist()
        self.own = cfg.get("ownership", "b")
        self.node = IdentityNode(topo, cfg, match_cams=border, cross_site_only=True,
                                 emit=(self.own == "b"), emit_belongs=emit_belongs)
        self.eng = self.node.engine
        self.site_of = topo.site
        self.owned = {}                  # (min, max) -> site, for site tree matches
        self.n_split = 0
        if self.own == "b":
            self.node.fixed = lambda a, b: self._key(a, b) in self.owned

    @staticmethod
    def _key(a, b):
        return (a, b) if a < b else (b, a)

    def handle(self, m):
        """Process one site message. Returns [(site, SplitRequest), ...] to send back."""
        k = m[0]
        eng = self.eng
        if k == P.EA:
            a, b = m[1], m[2]
            if self.own == "a":
                eng.add_match(a, b, INF)
                return []
            key = self._key(a, b)
            self.owned[key] = int(self.site_of[eng.V[a].cam])
            eng.add_match(a, b, m[3] if len(m) > 3 else INF)
            return self._splits(extra=key)
        if k == P.ED:
            self.owned.pop(self._key(m[1], m[2]), None)
            eng.remove_match(m[1], m[2])
        elif k == P.TR:                  # tag = site: enables the site-pure fast path
            eng.add_vertex(m[1], m[2], m[3], m[4], int(self.site_of[m[2]]))
        elif k == P.BT:
            self.node.on_tracklet(m[1], m[2], m[3], m[4], P.bytes_vec(m[5]),
                                  tag=int(self.site_of[m[2]]))
        elif k == P.BR:
            self.node.on_revise(m[1], P.bytes_vec(m[2]))
        return self._splits() if self.own == "b" else []

    def _splits(self, extra=None):
        """SplitRequests for site matches the hub rejected: every site match the hub cut
        (its own decision; ED-driven cuts are no longer owned), and a new site match the
        hub could not accept into one person."""
        eng = self.eng
        V = eng.V
        cut = [self._key(d[1], d[2]) for d in eng.out if d[0] == "-"]
        eng.out = [d for d in eng.out if d[0] == "B"]      # keep BelongsTo for a sink
        if extra is not None and V[extra[0]].person != V[extra[1]].person:
            cut.append(extra)
        out = []
        for key in cut:
            sid = self.owned.pop(key, None)
            if sid is None:
                continue
            eng.remove_match(key[0], key[1], block=True)   # not accepted here: no-op on the forest
            out.append((sid, [P.SR, key[0], key[1]]))
            self.n_split += 1
        return out


class CentralHub:
    """B5: one full instance at the hub; sites only forward."""

    def __init__(self, topo: Topology, cfg: dict, mode: str = "full"):
        self.node = IdentityNode(topo, cfg, mode=mode)
        self.eng = self.node.engine

    def handle(self, m):
        k = m[0]
        if k == P.FT:
            self.node.on_tracklet(m[1], m[2], m[3], m[4], P.bytes_vec(m[5]))
        elif k == P.FR:
            self.node.on_revise(m[1], P.bytes_vec(m[2]))


def forward_msgs(etype, tid, cam, ts, te, vec):
    if etype == 0:
        return [[P.FT, tid, cam, ts, te, P.vec_bytes(vec)]]
    return [[P.FR, tid, P.vec_bytes(vec)]]
