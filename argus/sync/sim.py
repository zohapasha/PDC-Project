"""Deterministic in-process drivers (single instance, two-level, byte accounting).
Used for accuracy / agreement / byte experiments and tests. The multi-process runtime
(mp.py) is used for throughput and latency."""
from __future__ import annotations

import time
from collections import defaultdict, deque

import numpy as np

from ..core.node import IdentityNode
from . import protocol as P
from .nodes import CentralHub, Hub, Site, forward_msgs


def _events(stream):
    """Yields python-typed events: (etype, tid, cam, ts, te, vec, time)."""
    cam, ts, te = stream.cam, stream.ts, stream.te
    for i in range(len(stream.time)):
        t = int(stream.tid[i])
        yield (int(stream.etype[i]), t, int(cam[t]), float(ts[t]), float(te[t]),
               stream.vec[i], float(stream.time[i]))


def run_single(stream, cfg: dict, mode: str = "full", op_rate: float = 0.0, seed: int = 0,
               checkpoints=(), on_checkpoint=None, time_events: bool = False):
    """One central instance over all cameras. op_rate = simulated operator corrections
    per minute (each deletes one currently-accepted wrong match, drawn using GT)."""
    node = IdentityNode(stream.topo, cfg, mode=mode)
    eng = node.engine
    gt = stream.gt
    rng = np.random.default_rng(seed + 991)
    next_op = 60.0 / op_rate if op_rate > 0 else float("inf")
    cps = sorted(checkpoints)
    lat = [] if time_events else None
    n_tr = 0
    pc = time.perf_counter_ns
    for et, tid, cam, ts, te, vec, now in _events(stream):
        t0 = pc() if time_events else 0
        if et == 0:
            node.on_tracklet(tid, cam, ts, te, vec)
            n_tr += 1
        else:
            node.on_revise(tid, vec)
        if time_events:
            lat.append((et, pc() - t0, n_tr))
        while now >= next_op:
            _operator_correction(eng, gt, rng)
            next_op += 60.0 / op_rate
        if cps and et == 0 and n_tr >= cps[0]:
            cps.pop(0)
            if on_checkpoint:
                on_checkpoint(n_tr, node)
    return node, lat


def _operator_correction(eng, gt, rng):
    wrong = []
    for t, v in eng.V.items():
        if v.parent != -1 and gt[t] != gt[v.parent]:
            wrong.append((t, v.parent))
        if v.backups:
            for o in v.backups:
                if t < o and gt[t] != gt[o]:
                    wrong.append((t, o))
    if wrong:
        a, b = wrong[int(rng.integers(len(wrong)))]
        eng.remove_match(a, b, block=True)


def run_two_level(stream, cfg: dict, topo=None, mode: str = "full"):
    """N sites + hub, messages delivered immediately in stream order (deterministic).
    Returns dict with hub, sites, bytes per message type, message counts."""
    topo = topo or stream.topo
    sites = [Site(topo, s, cfg, mode=mode) for s in range(topo.n_sites)]
    hub = Hub(topo, cfg)
    nbytes = defaultdict(int)
    nmsgs = defaultdict(int)
    site_of = topo.site
    for et, tid, cam, ts, te, vec, now in _events(stream):
        site = sites[site_of[cam]]
        msgs = site.on_tracklet(tid, cam, ts, te, vec) if et == 0 else site.on_revise(tid, cam, vec)
        work = deque(msgs)
        while work:                      # site deltas; SplitRequest replies go to the front
            m = work.popleft()
            nbytes[m[0]] += len(P.pack(m))
            nmsgs[m[0]] += 1
            back = []
            for sid, sr in hub.handle(m):
                nbytes[sr[0]] += len(P.pack(sr))
                nmsgs[sr[0]] += 1
                back.extend(sites[sid].on_split(sr[1], sr[2]))
            work.extendleft(reversed(back))
    return {"hub": hub, "sites": sites,
            "bytes": {P.NAMES[k]: v for k, v in nbytes.items()},
            "msgs": {P.NAMES[k]: v for k, v in nmsgs.items()}}


def run_centralized_hub(stream, cfg: dict, topo=None):
    """B5 in-process: everything forwarded with vectors."""
    topo = topo or stream.topo
    hub = CentralHub(topo, cfg)
    nbytes = defaultdict(int)
    for et, tid, cam, ts, te, vec, now in _events(stream):
        for m in forward_msgs(et, tid, cam, ts, te, vec):
            nbytes[m[0]] += len(P.pack(m))
            hub.handle(m)
    return {"hub": hub, "bytes": {P.NAMES[k]: v for k, v in nbytes.items()}}


def b6_bytes(stream, cfg: dict, period: float, topo=None):
    """B6 periodic full-state sync: border tracklets/vectors as in ours plus every
    `period` seconds each site sends its whole person table. Returns bytes by type and
    staleness (seconds until the hub learns about each tracklet's identity)."""
    topo = topo or stream.topo
    sites = [Site(topo, s, cfg) for s in range(topo.n_sites)]
    nbytes = defaultdict(int)
    stale = []
    next_sync = period
    site_of = topo.site
    for et, tid, cam, ts, te, vec, now in _events(stream):
        while now >= next_sync:
            for s in sites:
                nbytes[P.ST] += len(P.pack([P.ST, s.sid, s.table()]))
            next_sync += period
        site = sites[site_of[cam]]
        msgs = site.on_tracklet(tid, cam, ts, te, vec) if et == 0 else site.on_revise(tid, cam, vec)
        for m in msgs:
            if m[0] in (P.BT, P.BR):
                nbytes[m[0]] += len(P.pack(m))
        if et == 0:
            stale.append(next_sync - now)
    for s in sites:
        nbytes[P.ST] += len(P.pack([P.ST, s.sid, s.table()]))
    return {"bytes": {P.NAMES[k]: v for k, v in nbytes.items()},
            "staleness_mean": float(np.mean(stale)) if stale else 0.0}
