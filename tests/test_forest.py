"""Hand-built cases for merge / reroot / conflict resolution / backup reconnection."""
import numpy as np

from argus.common.topology import Topology, shortest_paths
from argus.common.types import K_CONFLICT_OK, K_CONFLICT_REJ, K_DEL_TREE
from argus.core.engine import IdentityEngine


def line_topo():
    # 3 cameras on a line, 10 s apart
    D, _ = shortest_paths(3, [(0, 1, 10.0), (1, 2, 10.0)])
    return Topology(3, D, np.arange(3), np.zeros(3, int), np.zeros(3, bool),
                    np.zeros(1, int), [(0, 1, 10.0), (1, 2, 10.0)])


def eng():
    return IdentityEngine(line_topo(), tau_accept=0.7, tau_keep=0.5, check=True)


def test_merge_and_backup():
    e = eng()
    for t, (c, s) in enumerate([(0, 0), (1, 20), (2, 40)]):
        e.add_vertex(t, c, s, s + 5)
    e.add_match(0, 1, 0.9)
    e.add_match(1, 2, 0.8)
    assert len({e.V[t].person for t in range(3)}) == 1
    e.add_match(0, 2, 0.75)                       # same person -> backup link
    assert e.V[0].backups == {2: 0.75}
    # deleting tree edge (1,2) is repaired by the backup (0,2): no split
    e.update_match(1, 2, 0.3)
    assert len({e.V[t].person for t in range(3)}) == 1
    assert e.oplog[-1][0] == K_DEL_TREE and e.oplog[-1][6] == 1


def test_split_without_backup():
    e = eng()
    for t, (c, s) in enumerate([(0, 0), (1, 20), (2, 40)]):
        e.add_vertex(t, c, s, s + 5)
    e.add_match(0, 1, 0.9)
    e.add_match(1, 2, 0.8)
    e.remove_match(1, 2, block=True)
    assert e.V[0].person == e.V[1].person != e.V[2].person
    e.add_match(1, 2, 0.99)                       # blocked -> ignored
    assert e.V[1].person != e.V[2].person


def test_conflict_cuts_weaker_match():
    """Fig. 2: x(cam0,t=0) ~ a (weak), new strong match a-b, but x conflicts with y in b's person."""
    e = eng()
    e.add_vertex(0, 0, 0, 5)      # x
    e.add_vertex(1, 1, 20, 25)    # a
    e.add_vertex(2, 2, 3, 8)      # y: cam 2 at t=3 conflicts with x (cam 0 at t<=5, D=20)
    e.add_vertex(3, 2, 40, 45)    # b
    e.add_match(0, 1, 0.72)       # weak x-a
    e.add_match(2, 3, 0.95)       # strong y-b
    e.add_match(1, 3, 0.9)        # new a-b: conflicts via x,y; weakest on path is x-a
    assert e.oplog[-1][0] == K_CONFLICT_OK
    assert e.V[1].person == e.V[3].person == e.V[2].person
    assert e.V[0].person != e.V[1].person


def test_conflict_rejects_weakest_new_match_and_rolls_back():
    e = eng()
    e.add_vertex(0, 0, 0, 5)
    e.add_vertex(1, 1, 20, 25)
    e.add_vertex(2, 2, 3, 8)
    e.add_vertex(3, 2, 40, 45)
    e.add_match(0, 1, 0.95)
    e.add_match(2, 3, 0.95)
    before = e.partition()
    e.add_match(1, 3, 0.8)
    assert e.oplog[-1][0] == K_CONFLICT_REJ
    assert e.partition() == before


def test_touched_vertices_bounded_by_person():
    e = eng()
    n = 40
    for t in range(n):
        e.add_vertex(t, t % 3, 30.0 * t, 30.0 * t + 5)
    for t in range(1, n // 2):
        e.add_match(t - 1, t, 0.9)
    for t in range(n // 2 + 1, n):
        e.add_match(t - 1, t, 0.9)
    e.update_match(5, 6, 0.1)                     # cut inside the first person (size 20)
    kind, msgs, touched, size, nv, ns, extra = e.oplog[-1]
    assert kind == K_DEL_TREE and size == n // 2 and touched <= size
