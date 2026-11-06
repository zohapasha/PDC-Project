"""Core vertex / person records. slots=True roughly halves per-object memory."""
from __future__ import annotations

from dataclasses import dataclass, field

INF = float("inf")


@dataclass(slots=True)
class Vertex:
    """A tracklet as a vertex of the identity graph (no appearance vector here: the
    matcher owns vectors and drops them once a tracklet leaves the matching window)."""
    tid: int
    cam: int
    ts: float
    te: float
    person: int = -1
    parent: int = -1            # -1 = root of its person's tree
    pw: float = 0.0             # score of the tree edge to parent
    children: list = field(default_factory=list)
    backups: dict | None = None  # partner tid -> score (accepted, non-tree, same person)


@dataclass(slots=True)
class Person:
    pid: int
    root: int
    size: int
    visits: list                # sorted [(ts, te, cam, tid), ...]


# --------------------------------------------------------------- op-log kinds
K_MERGE = 1          # accepted match joined two persons
K_BACKUP = 2         # accepted match inside one person -> backup link
K_CONFLICT_OK = 3    # conflict resolved by cutting weaker match(es), new match accepted
K_CONFLICT_REJ = 4   # conflict: new match was the weakest -> rejected (no change)
K_DEL_TREE = 5       # deletion of a tree edge (evidence / operator / site)
K_DEL_BACKUP = 6     # deletion of a backup link (cost ~ 0)
K_REJECT = 7         # rejected without resolution (no-correction baseline)

KIND_NAMES = {K_MERGE: "merge", K_BACKUP: "backup", K_CONFLICT_OK: "conflict_ok",
              K_CONFLICT_REJ: "conflict_rej", K_DEL_TREE: "del_tree",
              K_DEL_BACKUP: "del_backup", K_REJECT: "reject"}
CORRECTION_KINDS = (K_CONFLICT_OK, K_DEL_TREE)
