"""Camera topology: walking-time metric, overlap groups, sites and border cameras.

D[c1, c2] is the all-pairs *shortest* walking time, so it satisfies the triangle
inequality. Cameras in the same overlap group are at distance 0 and may see the
same person simultaneously. Same camera: D = 0, but intervals may not overlap.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Topology:
    n_cams: int
    D: np.ndarray                      # (n_cams, n_cams) float64 shortest walking times
    group: np.ndarray                  # overlap-group id per camera (singleton groups allowed)
    building: np.ndarray               # physical building per camera
    entrance: np.ndarray               # bool per camera: building entrance
    site_of_building: np.ndarray       # logical site per building
    edges: list = field(default_factory=list)   # (c1, c2, walk_time) raw corridor graph
    border_radius: float = 0.0         # hub matches cross-site on cams within this walking
                                       # time of a cross-site entrance (0 = entrances only)
    # derived
    site: np.ndarray = None            # site per camera
    border: np.ndarray = None          # bool per camera: sends vectors to the hub

    def __post_init__(self):
        self.D = np.asarray(self.D, dtype=np.float64)
        self.group = np.asarray(self.group, dtype=np.int64)
        self.building = np.asarray(self.building, dtype=np.int64)
        self.entrance = np.asarray(self.entrance, dtype=bool)
        self.site_of_building = np.asarray(self.site_of_building, dtype=np.int64)
        self.site = self.site_of_building[self.building]
        self.border = np.zeros(self.n_cams, dtype=bool)
        for c1, c2, _ in self.edges:
            if self.site[c1] != self.site[c2]:
                self.border[c1] = self.border[c2] = True
        if self.border_radius > 0 and self.border.any():
            self.border = (self.D[:, self.border] <= self.border_radius).any(axis=1)
        self.n_sites = int(self.site.max()) + 1 if self.n_cams else 0

    # ------------------------------------------------------------------ helpers
    def overlap(self, c1: int, c2: int) -> bool:
        """Different cameras whose views overlap (simultaneous presence allowed)."""
        return c1 != c2 and self.group[c1] == self.group[c2]

    def resite(self, site_of_building) -> "Topology":
        """Same physical campus, different logical split into sites (strong scaling)."""
        return Topology(self.n_cams, self.D, self.group, self.building, self.entrance,
                        np.asarray(site_of_building), list(self.edges), self.border_radius)

    def with_border_radius(self, r: float) -> "Topology":
        return Topology(self.n_cams, self.D, self.group, self.building, self.entrance,
                        self.site_of_building, list(self.edges), float(r))

    def to_json(self) -> str:
        return json.dumps({
            "n_cams": self.n_cams, "D": self.D.tolist(), "group": self.group.tolist(),
            "building": self.building.tolist(), "entrance": self.entrance.tolist(),
            "site_of_building": self.site_of_building.tolist(),
            "edges": [list(e) for e in self.edges], "border_radius": self.border_radius,
        })

    @staticmethod
    def from_json(s: str) -> "Topology":
        d = json.loads(s)
        return Topology(d["n_cams"], np.array(d["D"]), np.array(d["group"]),
                        np.array(d["building"]), np.array(d["entrance"]),
                        np.array(d["site_of_building"]), [tuple(e) for e in d["edges"]],
                        float(d.get("border_radius", 0.0)))


def shortest_paths(n: int, edges, overlap_pairs=()) -> tuple[np.ndarray, np.ndarray]:
    """Floyd-Warshall. Returns (D, next_hop) where next_hop[i, j] is the first hop on a
    shortest i->j path (-1 if unreachable). Overlap pairs get distance 0."""
    D = np.full((n, n), np.inf)
    nxt = np.full((n, n), -1, dtype=np.int64)
    for i in range(n):
        D[i, i] = 0.0
        nxt[i, i] = i
    for a, b, w in edges:
        if w < D[a, b]:
            D[a, b] = D[b, a] = w
            nxt[a, b], nxt[b, a] = b, a
    for a, b in overlap_pairs:
        D[a, b] = D[b, a] = 0.0
        nxt[a, b], nxt[b, a] = b, a
    for k in range(n):
        via = D[:, k:k + 1] + D[k:k + 1, :]
        better = via < D
        D = np.where(better, via, D)
        nxt = np.where(better, nxt[:, k:k + 1], nxt)
    return D, nxt


def overlap_groups(n: int, overlap_pairs) -> np.ndarray:
    """Connected components of the overlap relation -> group id per camera."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in overlap_pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = [find(i) for i in range(n)]
    remap = {r: i for i, r in enumerate(dict.fromkeys(roots))}
    return np.array([remap[r] for r in roots], dtype=np.int64)
