import numpy as np
from hypothesis import given, settings, strategies as st

from argus.common.constraints import Checker
from argus.common.topology import Topology, overlap_groups, shortest_paths


def make_topo(n, edges, overlaps):
    D, _ = shortest_paths(n, edges, overlaps)
    D[np.isinf(D)] = 1e6
    g = overlap_groups(n, overlaps)
    return Topology(n, D, g, np.zeros(n, int), np.zeros(n, bool), np.zeros(1, int), edges)


@st.composite
def topo_and_visits(draw):
    n = draw(st.integers(2, 6))
    edges = []
    for i in range(1, n):
        j = draw(st.integers(0, i - 1))
        edges.append((i, j, float(draw(st.integers(1, 30)))))
    overlaps = [(a, b) for a, b, _ in edges if draw(st.booleans()) and draw(st.booleans())]
    topo = make_topo(n, edges, overlaps)
    k = draw(st.integers(1, 9))
    vis = []
    for t in range(k):
        s = float(draw(st.integers(0, 200)))
        d = float(draw(st.integers(1, 30)))
        vis.append((s, s + d, draw(st.integers(0, n - 1)), t))
    return topo, sorted(vis)


@settings(max_examples=3000, deadline=None)
@given(topo_and_visits())
def test_linear_check_equals_all_pairs(tv):
    topo, vis = tv
    chk = Checker(topo)
    assert chk.feasible(vis) == chk.feasible_bruteforce(vis)


@settings(max_examples=2000, deadline=None)
@given(topo_and_visits(), st.data())
def test_can_merge_witness_crosses(tv, data):
    topo, vis = tv
    chk = Checker(topo)
    mask = [data.draw(st.booleans()) for _ in vis]
    A = [v for v, m in zip(vis, mask) if m]
    B = [v for v, m in zip(vis, mask) if not m]
    if not (chk.feasible_bruteforce(A) and chk.feasible_bruteforce(B)):
        return
    ok, w = chk.can_merge(A, B)
    assert ok == chk.feasible_bruteforce(vis)
    if not ok:
        ta, tb = {v[3] for v in A}, {v[3] for v in B}
        assert (w[0] in ta and w[1] in tb) or (w[0] in tb and w[1] in ta)


def test_overlap_same_camera_regression():
    # cams 0 and 1 overlap; a later visit on cam 0 overlaps an earlier cam-0 visit while
    # the latest-ending visit is on cam 1 (the case the naive consecutive check misses)
    topo = make_topo(2, [(0, 1, 10.0)], [(0, 1)])
    chk = Checker(topo)
    vis = [(0.0, 15.0, 0, 0), (5.0, 40.0, 1, 1), (12.0, 20.0, 0, 2)]
    assert not chk.feasible_bruteforce(vis)
    assert not chk.feasible(vis)


def test_walking_time():
    topo = make_topo(3, [(0, 1, 10.0), (1, 2, 10.0)], [])
    chk = Checker(topo)
    assert chk.feasible([(0, 5, 0, 0), (25, 30, 2, 1)])        # gap 20 >= D=20
    assert not chk.feasible([(0, 5, 0, 0), (24, 30, 2, 1)])    # gap 19 < 20
    assert not chk.feasible([(0, 5, 0, 0), (4, 6, 0, 1)])      # same camera overlap


@st.composite
def two_feasible_lists(draw):
    """A long feasible list (> 16 visits, triggers windowing) and a short one."""
    n = draw(st.integers(2, 6))
    edges = []
    for i in range(1, n):
        j = draw(st.integers(0, i - 1))
        edges.append((i, j, float(draw(st.integers(1, 40)))))
    overlaps = [(a, b) for a, b, _ in edges if draw(st.booleans()) and draw(st.booleans())]
    topo = make_topo(n, edges, overlaps)
    chk = Checker(topo)

    def walk(k, t0, tid0):
        out, t, cam = [], t0, draw(st.integers(0, n - 1))
        for i in range(k):
            nxt = draw(st.integers(0, n - 1))
            if out:
                t = out[-1][1] + topo.D[cam, nxt] + draw(st.integers(0, 60))
            d = float(draw(st.integers(1, 30)))
            out.append((float(t), float(t) + d, nxt, tid0 + i))
            cam = nxt
        return out

    B = walk(draw(st.integers(17, 40)), 0.0, 0)
    A = walk(draw(st.integers(1, 4)), float(draw(st.integers(0, 2000))), 1000)
    for v in A + B:
        chk.note_duration(v[1] - v[0])
    return chk, sorted(A), sorted(B)


@settings(max_examples=2000, deadline=None)
@given(two_feasible_lists())
def test_windowed_can_merge_equals_all_pairs(x):
    chk, A, B = x
    assert chk.feasible_bruteforce(A) and chk.feasible_bruteforce(B)
    ok, wit = chk.can_merge(A, B)
    assert ok == chk.feasible_bruteforce(A + B)
    if not ok:
        va = {v[3]: v for v in A + B}
        u, v = sorted((va[wit[0]], va[wit[1]]))
        assert not chk.compatible(u, v)
        assert (wit[0] < 1000) != (wit[1] < 1000)       # witness crosses the two lists
