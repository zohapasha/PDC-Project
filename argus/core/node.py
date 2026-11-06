"""IdentityNode = Matcher + IdentityEngine. Used for a single (centralized) instance,
for each site, and (with cross-site matching only) for the hub."""
from __future__ import annotations

from ..common.topology import Topology
from ..common.types import INF
from .engine import IdentityEngine
from .matcher import Matcher


class IdentityNode:
    def __init__(self, topo: Topology, cfg: dict, match_cams=None, cross_site_only=False,
                 mode: str = "full", emit: bool = False, emit_belongs: bool = False,
                 log_ops: bool = True):
        self.engine = IdentityEngine(topo, cfg["tau_accept"], cfg["tau_keep"], mode=mode,
                                     emit=emit, emit_belongs=emit_belongs, log_ops=log_ops)
        self.matcher = Matcher(topo, cfg["W"], cfg["tau_keep"], cams=match_cams,
                               cross_site_only=cross_site_only,
                               D_max=cfg.get("D_max", 600.0))
        self.fixed = None                        # hub: predicate for site-owned matches
        self.max_dur = 1.0
        self._n = 0

    def on_tracklet(self, tid, cam, ts, te, vec=None, tag=-1):
        eng, m = self.engine, self.matcher
        eng.add_vertex(tid, cam, ts, te, tag)
        if te - ts > self.max_dur:
            self.max_dur = te - ts
        if vec is None or cam not in m.cams:
            return
        cands = m.candidates(cam, ts, vec)
        m.add(tid, cam, ts, te, vec)
        for o, sc in cands:                      # strongest first
            eng.add_match(tid, o, sc)
        self._n += 1
        if self._n & 255 == 0:
            m.prune(te, self.max_dur)

    def on_revise(self, tid, vec):
        """The tracklet's final appearance vector replaces a provisional one: re-score its
        stored matches (may trigger evidence-driven deletions) and look for new ones."""
        eng, m = self.engine, self.matcher
        if not m.update_vec(tid, vec):
            return
        adj = eng.adj.get(tid)
        if adj is None:
            return
        for o, old in list(adj.items()):
            if old == INF or (self.fixed is not None and self.fixed(tid, o)):
                continue                         # site-owned link at the hub: site scores it
            ov = m.get(o)
            if ov is not None:
                eng.update_match(tid, o, float(ov @ vec))
        cam, ts, _ = m.meta(tid)
        ex = set(eng.adj[tid])
        ex.add(tid)
        new = m.candidates(cam, ts, vec, exclude=ex) + m.candidates_fwd(tid, vec, exclude=ex)
        new.sort(key=lambda x: -x[1])
        for o, sc in new:
            eng.add_match(tid, o, sc)
