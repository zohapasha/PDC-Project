"""B1 (full recomputation / offline reference) and B4 (no constraints).

B1: sort *all* stored matches by score (desc); accept a match iff the two persons can
merge without violating a constraint (Kruskal-style). This is the "full recomputation"
reference: online agreement is reported against it, and its run time at graph size n is
the per-update cost of a from-scratch approach (E2).

B4: plain connected components over all matches with score >= tau_accept.
"""
from __future__ import annotations

import heapq
import time

from ..common.constraints import Checker


def _accepted_edges(engine):
    acc = engine.tau_accept
    blocked = engine.blocked
    E = []
    for a, nb in engine.adj.items():
        for b, w in nb.items():
            if a < b and w >= acc and (a, b) not in blocked and (b, a) not in blocked:
                E.append((w, a, b))
    return E


def full_recompute(engine, checker: Checker | None = None):
    """Returns (partition, seconds)."""
    t0 = time.perf_counter()
    chk = checker or engine.chk
    V = engine.V
    E = _accepted_edges(engine)
    E.sort(key=lambda x: -x[0])
    parent = {t: t for t in V}
    visits = {t: [(v.ts, v.te, v.cam, t)] for t, v in V.items()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for w, a, b in E:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        va, vb = visits[ra], visits[rb]
        if chk.can_merge(va, vb)[0]:
            if len(va) > len(vb):
                ra, rb, va, vb = rb, ra, vb, va
            parent[ra] = rb
            visits[rb] = list(heapq.merge(va, vb))
            del visits[ra]
    part = {t: find(t) for t in V}
    return part, time.perf_counter() - t0


def connected_components(engine):
    V = engine.V
    parent = {t: t for t in V}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for w, a, b in _accepted_edges(engine):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return {t: find(t) for t in V}
