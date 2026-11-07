"""Core vertex / person records and shared constants. slots=True roughly halves
per-object memory."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

INF = float("inf")
NO_PARENT = -1                  # parent_id of a tree root

# A visit is a plain tuple (start_time, end_time, camera_id, tracklet_id): tuples sort by
# start time natively and are cheap in the constraint checker's hot loop.
VISIT_START, VISIT_END, VISIT_CAMERA, VISIT_TRACKLET = 0, 1, 2, 3

# Stream event types
EVENT_TRACKLET = 0              # a finished tracklet (arrives at its end time)
EVENT_REVISION = 1              # the final appearance vector of an earlier tracklet


@dataclass(slots=True)
class Vertex:
    """A tracklet as a vertex of the identity graph (no appearance vector here: the
    matcher owns vectors and drops them once a tracklet leaves the matching window)."""
    tracklet_id: int
    camera_id: int
    start_time: float
    end_time: float
    person_id: int = -1
    parent_id: int = NO_PARENT          # NO_PARENT = root of its person's tree
    parent_edge_score: float = 0.0      # score of the tree edge to the parent
    child_ids: list = field(default_factory=list)
    backup_links: dict | None = None    # partner tracklet_id -> score (accepted, non-tree,
                                        # same person)


@dataclass(slots=True)
class Person:
    person_id: int
    root_id: int
    size: int
    visits: list                # sorted [(start_time, end_time, camera_id, tracklet_id), ...]
    site_tag: int = -1          # hub: site id if every member is from that site and was
                                # joined by that site's own matches ("site-pure"); else -1


# --------------------------------------------------------------- op-log kinds
OP_MERGE = 1                 # accepted match joined two persons
OP_BACKUP = 2                # accepted match inside one person -> backup link
OP_CONFLICT_RESOLVED = 3     # conflict resolved by cutting weaker match(es), new match accepted
OP_CONFLICT_REJECTED = 4     # conflict: new match was the weakest -> rejected (no change)
OP_DELETE_TREE_EDGE = 5      # deletion of a tree edge (evidence / operator / site)
OP_DELETE_BACKUP = 6         # deletion of a backup link (cost ~ 0)
OP_REJECT = 7                # rejected without resolution (no-correction baseline)

OP_KIND_NAMES = {OP_MERGE: "merge", OP_BACKUP: "backup", OP_CONFLICT_RESOLVED: "conflict_ok",
                 OP_CONFLICT_REJECTED: "conflict_rej", OP_DELETE_TREE_EDGE: "del_tree",
                 OP_DELETE_BACKUP: "del_backup", OP_REJECT: "reject"}
CORRECTION_OP_KINDS = (OP_CONFLICT_RESOLVED, OP_DELETE_TREE_EDGE)


class OpRecord(NamedTuple):
    """One logged engine operation."""
    kind: int                # one of the OP_* constants
    n_messages: int          # vertex messages sent
    n_touched: int           # distinct vertices touched
    person_size: int         # size of the person(s) involved
    graph_size: int          # number of vertices in the engine at that time
    latency_ns: int
    extra: int               # deletions: 1 if repaired by a backup link; conflicts: #cuts


# --------------------------------------------------------------- emitted deltas
# Engine output tuples (tag first):
DELTA_TREE_EDGE_ADDED = "+"      # ("+", child_id, parent_id, score)
DELTA_TREE_EDGE_REMOVED = "-"    # ("-", child_id, parent_id)
DELTA_TREE_EDGE_SCORE = "w"      # ("w", tracklet_a, tracklet_b, new_score)
DELTA_BELONGS_TO = "B"           # ("B", tracklet_id, old_person_id, new_person_id)
