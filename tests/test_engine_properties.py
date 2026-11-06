"""Property-based tests: random topologies + random event sequences; after every
operation invariants I1-I3 (+ backup links inside persons) must hold."""
import numpy as np
from hypothesis import given, settings, strategies as st

from argus.baselines.offline import full_recompute
from argus.common.constraints import Checker
from argus.common.topology import Topology, overlap_groups, shortest_paths
from argus.common.types import INF
from argus.core.engine import IdentityEngine


def make_topo(n, edges, overlaps):
    D, _ = shortest_paths(n, edges, overlaps)
    D[np.isinf(D)] = 1e6
    return Topology(n, D, overlap_groups(n, overlaps), np.zeros(n, int),
                    np.zeros(n, bool), np.zeros(1, int), edges)


@st.composite
def scenario(draw):
    n = draw(st.integers(1, 5))
    edges = [(i, draw(st.integers(0, i - 1)), float(draw(st.integers(1, 20))))
             for i in range(1, n)]
    overlaps = [(a, b) for a, b, _ in edges if draw(st.integers(0, 3)) == 0]
    topo = make_topo(n, edges, overlaps)
    k = draw(st.integers(2, 14))
    verts = []
    for t in range(k):
        s = float(draw(st.integers(0, 120)))
        verts.append((t, draw(st.integers(0, n - 1)), s, s + float(draw(st.integers(1, 15)))))
    ops = []
    for _ in range(draw(st.integers(1, 40))):
        kind = draw(st.sampled_from(["add", "add", "add", "update", "remove", "block", "inf"]))
        a = draw(st.integers(0, k - 1))
        b = draw(st.integers(0, k - 1))
        w = draw(st.integers(40, 100)) / 100.0
        ops.append((kind, a, b, w))
    return topo, verts, ops


def _run(topo, verts, ops, mode):
    eng = IdentityEngine(topo, tau_accept=0.7, tau_keep=0.5, mode=mode, emit=True,
                         emit_belongs=True, check=True)
    for t, c, s, e in verts:
        eng.add_vertex(t, c, s, e)
    for kind, a, b, w in ops:
        if kind == "add":
            eng.add_match(a, b, w)
        elif kind == "update":
            if a in eng.adj and b in eng.adj[a]:
                eng.update_match(a, b, w)
        elif kind == "remove":
            eng.remove_match(a, b)
        elif kind == "block":
            eng.remove_match(a, b, block=True)
        elif kind == "inf" and mode == "full":
            # +inf links must never be rejected when the endpoints' union is feasible
            # (hub semantics); otherwise they are just stored.
            pa, pb = eng.V[a].person, eng.V[b].person
            fresh = (a, b) not in eng.blocked and (b, a) not in eng.blocked and b not in eng.adj[a]
            feasible = fresh and (pa == pb or eng.chk.can_merge(eng.P[pa].visits, eng.P[pb].visits)[0])
            eng.add_match(a, b, INF)
            if feasible and a != b:
                assert eng.V[a].person == eng.V[b].person
    eng.check_invariants()
    return eng


@settings(max_examples=1500, deadline=None)
@given(scenario())
def test_invariants_full(sc):
    _run(*sc, "full")


@settings(max_examples=600, deadline=None)
@given(scenario())
def test_invariants_component(sc):
    _run(*sc, "component")


@settings(max_examples=600, deadline=None)
@given(scenario())
def test_invariants_no_correction(sc):
    _run(*sc, "no_correction")


@settings(max_examples=500, deadline=None)
@given(scenario())
def test_offline_reference_is_feasible(sc):
    eng = _run(*sc, "full")
    part, _ = full_recompute(eng)
    chk = Checker(sc[0])
    groups = {}
    for t, lab in part.items():
        v = eng.V[t]
        groups.setdefault(lab, []).append((v.ts, v.te, v.cam, t))
    for vs in groups.values():
        assert chk.feasible_bruteforce(vs)


@settings(max_examples=500, deadline=None)
@given(scenario())
def test_belongs_deltas_replay_to_partition(sc):
    """Replaying emitted BelongsTo deltas reproduces the engine's partition."""
    eng = _run(*sc, "full")
    lab = {}
    for d in eng.out:
        if d[0] == "B":
            lab[d[1]] = d[3]
    assert lab == eng.partition()
