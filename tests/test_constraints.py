import numpy as np
from hypothesis import given, settings, strategies as st

from argus.common.constraints import ConstraintChecker
from argus.common.topology import Topology, overlap_groups, shortest_paths


def make_topology(n_cameras, corridors, overlap_pairs):
    walk_time, _ = shortest_paths(n_cameras, corridors, overlap_pairs)
    walk_time[np.isinf(walk_time)] = 1e6
    overlap_group_of_camera = overlap_groups(n_cameras, overlap_pairs)
    return Topology(n_cameras, walk_time, overlap_group_of_camera, np.zeros(n_cameras, int),
                    np.zeros(n_cameras, bool), np.zeros(1, int), corridors)


@st.composite
def topology_and_visits(draw):
    n_cameras = draw(st.integers(2, 6))
    corridors = []
    for camera in range(1, n_cameras):
        neighbour = draw(st.integers(0, camera - 1))
        corridors.append((camera, neighbour, float(draw(st.integers(1, 30)))))
    overlap_pairs = [(camera_a, camera_b) for camera_a, camera_b, _ in corridors
                     if draw(st.booleans()) and draw(st.booleans())]
    topology = make_topology(n_cameras, corridors, overlap_pairs)
    n_visits = draw(st.integers(1, 9))
    visits = []
    for tracklet_id in range(n_visits):
        start = float(draw(st.integers(0, 200)))
        duration = float(draw(st.integers(1, 30)))
        visits.append((start, start + duration, draw(st.integers(0, n_cameras - 1)), tracklet_id))
    return topology, sorted(visits)


@settings(max_examples=3000, deadline=None)
@given(topology_and_visits())
def test_linear_check_equals_all_pairs(topology_with_visits):
    topology, visits = topology_with_visits
    checker = ConstraintChecker(topology)
    assert checker.feasible(visits) == checker.feasible_bruteforce(visits)


@settings(max_examples=2000, deadline=None)
@given(topology_and_visits(), st.data())
def test_can_merge_witness_crosses(topology_with_visits, data):
    topology, visits = topology_with_visits
    checker = ConstraintChecker(topology)
    in_list_a = [data.draw(st.booleans()) for _ in visits]
    visits_a = [visit for visit, in_a in zip(visits, in_list_a) if in_a]
    visits_b = [visit for visit, in_a in zip(visits, in_list_a) if not in_a]
    if not (checker.feasible_bruteforce(visits_a) and checker.feasible_bruteforce(visits_b)):
        return
    feasible, witness = checker.can_merge(visits_a, visits_b)
    assert feasible == checker.feasible_bruteforce(visits)
    if not feasible:
        tracklets_a = {visit[3] for visit in visits_a}
        tracklets_b = {visit[3] for visit in visits_b}
        assert ((witness[0] in tracklets_a and witness[1] in tracklets_b)
                or (witness[0] in tracklets_b and witness[1] in tracklets_a))


def test_overlap_same_camera_regression():
    # cams 0 and 1 overlap; a later visit on cam 0 overlaps an earlier cam-0 visit while
    # the latest-ending visit is on cam 1 (the case the naive consecutive check misses)
    topology = make_topology(2, [(0, 1, 10.0)], [(0, 1)])
    checker = ConstraintChecker(topology)
    visits = [(0.0, 15.0, 0, 0), (5.0, 40.0, 1, 1), (12.0, 20.0, 0, 2)]
    assert not checker.feasible_bruteforce(visits)
    assert not checker.feasible(visits)


def test_walking_time():
    topology = make_topology(3, [(0, 1, 10.0), (1, 2, 10.0)], [])
    checker = ConstraintChecker(topology)
    assert checker.feasible([(0, 5, 0, 0), (25, 30, 2, 1)])        # gap 20 >= walk time 20
    assert not checker.feasible([(0, 5, 0, 0), (24, 30, 2, 1)])    # gap 19 < 20
    assert not checker.feasible([(0, 5, 0, 0), (4, 6, 0, 1)])      # same camera overlap


@st.composite
def two_feasible_lists(draw):
    """A long feasible list (> 16 visits, triggers windowing) and a short one."""
    n_cameras = draw(st.integers(2, 6))
    corridors = []
    for camera in range(1, n_cameras):
        neighbour = draw(st.integers(0, camera - 1))
        corridors.append((camera, neighbour, float(draw(st.integers(1, 40)))))
    overlap_pairs = [(camera_a, camera_b) for camera_a, camera_b, _ in corridors
                     if draw(st.booleans()) and draw(st.booleans())]
    topology = make_topology(n_cameras, corridors, overlap_pairs)
    checker = ConstraintChecker(topology)

    def walk(n_visits, first_start, first_tracklet_id):
        visits, start, camera = [], first_start, draw(st.integers(0, n_cameras - 1))
        for i in range(n_visits):
            next_camera = draw(st.integers(0, n_cameras - 1))
            if visits:
                start = visits[-1][1] + topology.walk_time[camera, next_camera] + draw(st.integers(0, 60))
            duration = float(draw(st.integers(1, 30)))
            visits.append((float(start), float(start) + duration, next_camera, first_tracklet_id + i))
            camera = next_camera
        return visits

    long_list = walk(draw(st.integers(17, 40)), 0.0, 0)
    short_list = walk(draw(st.integers(1, 4)), float(draw(st.integers(0, 2000))), 1000)
    for visit in short_list + long_list:
        checker.note_duration(visit[1] - visit[0])
    return checker, sorted(short_list), sorted(long_list)


@settings(max_examples=2000, deadline=None)
@given(two_feasible_lists())
def test_windowed_can_merge_equals_all_pairs(checker_and_lists):
    checker, short_list, long_list = checker_and_lists
    assert checker.feasible_bruteforce(short_list) and checker.feasible_bruteforce(long_list)
    feasible, witness = checker.can_merge(short_list, long_list)
    assert feasible == checker.feasible_bruteforce(short_list + long_list)
    if not feasible:
        visit_of_tracklet = {visit[3]: visit for visit in short_list + long_list}
        earlier, later = sorted((visit_of_tracklet[witness[0]], visit_of_tracklet[witness[1]]))
        assert not checker.compatible(earlier, later)
        assert (witness[0] < 1000) != (witness[1] < 1000)       # witness crosses the two lists
