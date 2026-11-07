"""Property-based tests: random topologies + random event sequences; after every
operation invariants I1-I3 (+ backup links inside persons) must hold."""
import numpy as np
from hypothesis import given, settings, strategies as st

from argus.baselines.offline import full_recompute
from argus.common.constraints import ConstraintChecker
from argus.common.topology import Topology, overlap_groups, shortest_paths
from argus.common.types import DELTA_BELONGS_TO, INF
from argus.core.engine import IdentityEngine


def make_topology(n_cameras, corridors, overlap_pairs):
    walk_time, _ = shortest_paths(n_cameras, corridors, overlap_pairs)
    walk_time[np.isinf(walk_time)] = 1e6
    return Topology(n_cameras, walk_time, overlap_groups(n_cameras, overlap_pairs),
                    np.zeros(n_cameras, int), np.zeros(n_cameras, bool), np.zeros(1, int), corridors)


@st.composite
def scenario(draw):
    n_cameras = draw(st.integers(1, 5))
    corridors = [(camera, draw(st.integers(0, camera - 1)), float(draw(st.integers(1, 20))))
                 for camera in range(1, n_cameras)]
    overlap_pairs = [(camera_a, camera_b) for camera_a, camera_b, _ in corridors
                     if draw(st.integers(0, 3)) == 0]
    topology = make_topology(n_cameras, corridors, overlap_pairs)
    n_tracklets = draw(st.integers(2, 14))
    tracklets = []
    for tracklet_id in range(n_tracklets):
        start = float(draw(st.integers(0, 120)))
        tracklets.append((tracklet_id, draw(st.integers(0, n_cameras - 1)), start,
                          start + float(draw(st.integers(1, 15)))))
    operations = []
    for _ in range(draw(st.integers(1, 40))):
        kind = draw(st.sampled_from(["add", "add", "add", "update", "remove", "block", "inf"]))
        tracklet_a = draw(st.integers(0, n_tracklets - 1))
        tracklet_b = draw(st.integers(0, n_tracklets - 1))
        score = draw(st.integers(40, 100)) / 100.0
        operations.append((kind, tracklet_a, tracklet_b, score))
    return topology, tracklets, operations


def _run(topology, tracklets, operations, mode):
    engine = IdentityEngine(topology, tau_accept=0.7, tau_keep=0.5, mode=mode,
                            emit_tree_deltas=True, emit_belongs_to=True, verify_invariants=True)
    for tracklet_id, camera_id, start, end in tracklets:
        engine.add_vertex(tracklet_id, camera_id, start, end)
    for kind, tracklet_a, tracklet_b, score in operations:
        if kind == "add":
            engine.add_match(tracklet_a, tracklet_b, score)
        elif kind == "update":
            if tracklet_a in engine.match_scores and tracklet_b in engine.match_scores[tracklet_a]:
                engine.update_match(tracklet_a, tracklet_b, score)
        elif kind == "remove":
            engine.remove_match(tracklet_a, tracklet_b)
        elif kind == "block":
            engine.remove_match(tracklet_a, tracklet_b, block=True)
        elif kind == "inf" and mode == "full":
            # +inf links must never be rejected when the endpoints' union is feasible
            # (hub semantics); otherwise they are just stored.
            person_a = engine.vertices[tracklet_a].person_id
            person_b = engine.vertices[tracklet_b].person_id
            is_new = ((tracklet_a, tracklet_b) not in engine.blocked_pairs
                      and (tracklet_b, tracklet_a) not in engine.blocked_pairs
                      and tracklet_b not in engine.match_scores[tracklet_a])
            feasible = is_new and (person_a == person_b or engine.checker.can_merge(
                engine.persons[person_a].visits, engine.persons[person_b].visits)[0])
            engine.add_match(tracklet_a, tracklet_b, INF)
            if feasible and tracklet_a != tracklet_b:
                assert engine.vertices[tracklet_a].person_id == engine.vertices[tracklet_b].person_id
    engine.check_invariants()
    return engine


@settings(max_examples=1500, deadline=None)
@given(scenario())
def test_invariants_full(generated):
    _run(*generated, "full")


@settings(max_examples=600, deadline=None)
@given(scenario())
def test_invariants_component(generated):
    _run(*generated, "component")


@settings(max_examples=600, deadline=None)
@given(scenario())
def test_invariants_no_correction(generated):
    _run(*generated, "no_correction")


@settings(max_examples=500, deadline=None)
@given(scenario())
def test_offline_reference_is_feasible(generated):
    engine = _run(*generated, "full")
    partition, _ = full_recompute(engine)
    checker = ConstraintChecker(generated[0])
    visits_of_label = {}
    for tracklet_id, label in partition.items():
        vertex = engine.vertices[tracklet_id]
        visits_of_label.setdefault(label, []).append(
            (vertex.start_time, vertex.end_time, vertex.camera_id, tracklet_id))
    for visits in visits_of_label.values():
        assert checker.feasible_bruteforce(visits)


@settings(max_examples=500, deadline=None)
@given(scenario())
def test_belongs_deltas_replay_to_partition(generated):
    """Replaying emitted BelongsTo deltas reproduces the engine's partition."""
    engine = _run(*generated, "full")
    person_of_tracklet = {}
    for delta in engine.emitted_deltas:
        if delta[0] == DELTA_BELONGS_TO:
            person_of_tracklet[delta[1]] = delta[3]
    assert person_of_tracklet == engine.partition()
