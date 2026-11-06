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
            out.append([P.EA, d[1], d[2]] if d[0] == "+" else [P.ED, d[1], d[2]])
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

    def table(self):
        """Full person table (B6)."""
        return [[t, v.person] for t, v in self.node.engine.V.items()]


class Hub:
    """Two-level hub: same engine; vertices = all tracklets (metadata), cross-site
    matching over border tracklets only, site tree matches as +inf edges."""

    def __init__(self, topo: Topology, cfg: dict, emit_belongs: bool = False):
        border = np.nonzero(topo.border)[0].tolist()
        self.node = IdentityNode(topo, cfg, match_cams=border, cross_site_only=True,
                                 emit_belongs=emit_belongs)
        self.eng = self.node.engine

    def handle(self, m):
        k = m[0]
        if k == P.EA:
            self.eng.add_match(m[1], m[2], INF)
        elif k == P.ED:
            self.eng.remove_match(m[1], m[2])
        elif k == P.TR:
            self.eng.add_vertex(m[1], m[2], m[3], m[4])
        elif k == P.BT:
            self.node.on_tracklet(m[1], m[2], m[3], m[4], P.bytes_vec(m[5]))
        elif k == P.BR:
            self.node.on_revise(m[1], P.bytes_vec(m[2]))


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
