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

from ..common.constraints import ConstraintChecker


def _accepted_matches(engine):
    """[(score, tracklet_a, tracklet_b)] for every stored, unblocked match >= tau_accept."""
    tau_accept = engine.tau_accept
    blocked_pairs = engine.blocked_pairs
    accepted = []
    for tracklet_a, scores_of_a in engine.match_scores.items():
        for tracklet_b, score in scores_of_a.items():
            if (tracklet_a < tracklet_b and score >= tau_accept
                    and (tracklet_a, tracklet_b) not in blocked_pairs
                    and (tracklet_b, tracklet_a) not in blocked_pairs):
                accepted.append((score, tracklet_a, tracklet_b))
    return accepted


def _make_find_root(union_parent):
    def find_root(tracklet_id):
        while union_parent[tracklet_id] != tracklet_id:
            union_parent[tracklet_id] = union_parent[union_parent[tracklet_id]]
            tracklet_id = union_parent[tracklet_id]
        return tracklet_id
    return find_root


def full_recompute(engine, checker: ConstraintChecker | None = None):
    """Returns (partition, seconds)."""
    start = time.perf_counter()
    checker = checker or engine.checker
    vertices = engine.vertices
    accepted = _accepted_matches(engine)
    accepted.sort(key=lambda match: -match[0])
    union_parent = {tracklet_id: tracklet_id for tracklet_id in vertices}
    visits_of_root = {tracklet_id: [(vertex.start_time, vertex.end_time, vertex.camera_id, tracklet_id)]
                      for tracklet_id, vertex in vertices.items()}
    find_root = _make_find_root(union_parent)

    for _, tracklet_a, tracklet_b in accepted:
        root_a, root_b = find_root(tracklet_a), find_root(tracklet_b)
        if root_a == root_b:
            continue
        visits_a, visits_b = visits_of_root[root_a], visits_of_root[root_b]
        if checker.can_merge(visits_a, visits_b)[0]:
            if len(visits_a) > len(visits_b):
                root_a, root_b, visits_a, visits_b = root_b, root_a, visits_b, visits_a
            union_parent[root_a] = root_b
            visits_of_root[root_b] = list(heapq.merge(visits_a, visits_b))
            del visits_of_root[root_a]
    partition = {tracklet_id: find_root(tracklet_id) for tracklet_id in vertices}
    return partition, time.perf_counter() - start


def connected_components(engine):
    vertices = engine.vertices
    union_parent = {tracklet_id: tracklet_id for tracklet_id in vertices}
    find_root = _make_find_root(union_parent)

    for _, tracklet_a, tracklet_b in _accepted_matches(engine):
        root_a, root_b = find_root(tracklet_a), find_root(tracklet_b)
        if root_a != root_b:
            union_parent[root_a] = root_b
    return {tracklet_id: find_root(tracklet_id) for tracklet_id in vertices}
