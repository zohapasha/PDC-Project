"""Hand-built cases for merge / reroot / conflict resolution / backup reconnection."""
import numpy as np

from argus.common.topology import Topology, shortest_paths
from argus.common.types import OP_CONFLICT_REJECTED, OP_CONFLICT_RESOLVED, OP_DELETE_TREE_EDGE
from argus.core.engine import IdentityEngine


def line_topology():
    # 3 cameras on a line, 10 s apart
    corridors = [(0, 1, 10.0), (1, 2, 10.0)]
    walk_time, _ = shortest_paths(3, corridors)
    return Topology(3, walk_time, np.arange(3), np.zeros(3, int), np.zeros(3, bool),
                    np.zeros(1, int), corridors)


def make_engine():
    return IdentityEngine(line_topology(), tau_accept=0.7, tau_keep=0.5, verify_invariants=True)


def person_ids(engine, tracklet_ids):
    return {engine.vertices[tracklet_id].person_id for tracklet_id in tracklet_ids}


def test_merge_and_backup():
    engine = make_engine()
    for tracklet_id, (camera_id, start) in enumerate([(0, 0), (1, 20), (2, 40)]):
        engine.add_vertex(tracklet_id, camera_id, start, start + 5)
    engine.add_match(0, 1, 0.9)
    engine.add_match(1, 2, 0.8)
    assert len(person_ids(engine, range(3))) == 1
    engine.add_match(0, 2, 0.75)                       # same person -> backup link
    assert engine.vertices[0].backup_links == {2: 0.75}
    # deleting tree edge (1,2) is repaired by the backup (0,2): no split
    engine.update_match(1, 2, 0.3)
    assert len(person_ids(engine, range(3))) == 1
    last_op = engine.op_log[-1]
    assert last_op.kind == OP_DELETE_TREE_EDGE and last_op.extra == 1


def test_split_without_backup():
    engine = make_engine()
    for tracklet_id, (camera_id, start) in enumerate([(0, 0), (1, 20), (2, 40)]):
        engine.add_vertex(tracklet_id, camera_id, start, start + 5)
    engine.add_match(0, 1, 0.9)
    engine.add_match(1, 2, 0.8)
    engine.remove_match(1, 2, block=True)
    vertices = engine.vertices
    assert vertices[0].person_id == vertices[1].person_id != vertices[2].person_id
    engine.add_match(1, 2, 0.99)                       # blocked -> ignored
    assert vertices[1].person_id != vertices[2].person_id


def test_conflict_cuts_weaker_match():
    """Fig. 2: x(cam0,t=0) ~ a (weak), new strong match a-b, but x conflicts with y in b's person."""
    engine = make_engine()
    engine.add_vertex(0, 0, 0, 5)      # x
    engine.add_vertex(1, 1, 20, 25)    # a
    engine.add_vertex(2, 2, 3, 8)      # y: cam 2 at t=3 conflicts with x (cam 0 at t<=5, walk 20)
    engine.add_vertex(3, 2, 40, 45)    # b
    engine.add_match(0, 1, 0.72)       # weak x-a
    engine.add_match(2, 3, 0.95)       # strong y-b
    engine.add_match(1, 3, 0.9)        # new a-b: conflicts via x,y; weakest on path is x-a
    assert engine.op_log[-1].kind == OP_CONFLICT_RESOLVED
    vertices = engine.vertices
    assert vertices[1].person_id == vertices[3].person_id == vertices[2].person_id
    assert vertices[0].person_id != vertices[1].person_id


def test_conflict_rejects_weakest_new_match_and_rolls_back():
    engine = make_engine()
    engine.add_vertex(0, 0, 0, 5)
    engine.add_vertex(1, 1, 20, 25)
    engine.add_vertex(2, 2, 3, 8)
    engine.add_vertex(3, 2, 40, 45)
    engine.add_match(0, 1, 0.95)
    engine.add_match(2, 3, 0.95)
    partition_before = engine.partition()
    engine.add_match(1, 3, 0.8)
    assert engine.op_log[-1].kind == OP_CONFLICT_REJECTED
    assert engine.partition() == partition_before


def test_touched_vertices_bounded_by_person():
    engine = make_engine()
    n_tracklets = 40
    for tracklet_id in range(n_tracklets):
        engine.add_vertex(tracklet_id, tracklet_id % 3, 30.0 * tracklet_id, 30.0 * tracklet_id + 5)
    for tracklet_id in range(1, n_tracklets // 2):
        engine.add_match(tracklet_id - 1, tracklet_id, 0.9)
    for tracklet_id in range(n_tracklets // 2 + 1, n_tracklets):
        engine.add_match(tracklet_id - 1, tracklet_id, 0.9)
    engine.update_match(5, 6, 0.1)                     # cut inside the first person (size 20)
    last_op = engine.op_log[-1]
    assert last_op.kind == OP_DELETE_TREE_EDGE and last_op.person_size == n_tracklets // 2
    assert last_op.n_touched <= last_op.person_size
