"""Camera topology: walking-time metric, overlap groups, sites and border cameras.

walk_time[c1, c2] is the all-pairs *shortest* walking time, so it satisfies the triangle
inequality. Cameras in the same overlap group are at distance 0 and may see the
same person simultaneously. Same camera: walking time 0, but intervals may not overlap.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Topology:
    n_cameras: int
    walk_time: np.ndarray                   # (n_cameras, n_cameras) float64 shortest walking times
    overlap_group_of_camera: np.ndarray     # overlap-group id per camera (singletons allowed)
    building_of_camera: np.ndarray          # physical building per camera
    is_entrance: np.ndarray                 # bool per camera: building entrance
    site_of_building: np.ndarray            # logical site per building
    corridors: list = field(default_factory=list)   # (camera_a, camera_b, walk_time) raw graph
    border_radius: float = 0.0              # hub matches cross-site on cameras within this
                                            # walking time of a cross-site entrance
                                            # (0 = entrances only)
    # derived
    site_of_camera: np.ndarray = None       # site per camera
    is_border_camera: np.ndarray = None     # bool per camera: sends vectors to the hub

    def __post_init__(self):
        self.walk_time = np.asarray(self.walk_time, dtype=np.float64)
        self.overlap_group_of_camera = np.asarray(self.overlap_group_of_camera, dtype=np.int64)
        self.building_of_camera = np.asarray(self.building_of_camera, dtype=np.int64)
        self.is_entrance = np.asarray(self.is_entrance, dtype=bool)
        self.site_of_building = np.asarray(self.site_of_building, dtype=np.int64)
        self.site_of_camera = self.site_of_building[self.building_of_camera]
        self.is_border_camera = np.zeros(self.n_cameras, dtype=bool)
        for camera_a, camera_b, _ in self.corridors:
            if self.site_of_camera[camera_a] != self.site_of_camera[camera_b]:
                self.is_border_camera[camera_a] = self.is_border_camera[camera_b] = True
        if self.border_radius > 0 and self.is_border_camera.any():
            self.is_border_camera = (
                self.walk_time[:, self.is_border_camera] <= self.border_radius).any(axis=1)
        self.n_sites = int(self.site_of_camera.max()) + 1 if self.n_cameras else 0

    # ------------------------------------------------------------------ helpers
    def views_overlap(self, camera_a: int, camera_b: int) -> bool:
        """Different cameras whose views overlap (simultaneous presence allowed)."""
        return (camera_a != camera_b
                and self.overlap_group_of_camera[camera_a] == self.overlap_group_of_camera[camera_b])

    def with_site_assignment(self, site_of_building) -> "Topology":
        """Same physical campus, different logical split into sites (strong scaling)."""
        return Topology(self.n_cameras, self.walk_time, self.overlap_group_of_camera,
                        self.building_of_camera, self.is_entrance, np.asarray(site_of_building),
                        list(self.corridors), self.border_radius)

    def with_border_radius(self, radius: float) -> "Topology":
        return Topology(self.n_cameras, self.walk_time, self.overlap_group_of_camera,
                        self.building_of_camera, self.is_entrance, self.site_of_building,
                        list(self.corridors), float(radius))

    # The JSON keys are a file format (saved next to every stream); they keep their
    # original short names so existing files still load.
    def to_json(self) -> str:
        return json.dumps({
            "n_cams": self.n_cameras, "D": self.walk_time.tolist(),
            "group": self.overlap_group_of_camera.tolist(),
            "building": self.building_of_camera.tolist(), "entrance": self.is_entrance.tolist(),
            "site_of_building": self.site_of_building.tolist(),
            "edges": [list(corridor) for corridor in self.corridors],
            "border_radius": self.border_radius,
        })

    @staticmethod
    def from_json(text: str) -> "Topology":
        data = json.loads(text)
        return Topology(data["n_cams"], np.array(data["D"]), np.array(data["group"]),
                        np.array(data["building"]), np.array(data["entrance"]),
                        np.array(data["site_of_building"]),
                        [tuple(corridor) for corridor in data["edges"]],
                        float(data.get("border_radius", 0.0)))


def shortest_paths(n_cameras: int, corridors, overlap_pairs=()) -> tuple[np.ndarray, np.ndarray]:
    """Floyd-Warshall. Returns (walk_time, next_hop) where next_hop[i, j] is the first hop
    on a shortest i->j path (-1 if unreachable). Overlap pairs get walking time 0."""
    walk_time = np.full((n_cameras, n_cameras), np.inf)
    next_hop = np.full((n_cameras, n_cameras), -1, dtype=np.int64)
    for camera in range(n_cameras):
        walk_time[camera, camera] = 0.0
        next_hop[camera, camera] = camera
    for camera_a, camera_b, corridor_time in corridors:
        if corridor_time < walk_time[camera_a, camera_b]:
            walk_time[camera_a, camera_b] = walk_time[camera_b, camera_a] = corridor_time
            next_hop[camera_a, camera_b], next_hop[camera_b, camera_a] = camera_b, camera_a
    for camera_a, camera_b in overlap_pairs:
        walk_time[camera_a, camera_b] = walk_time[camera_b, camera_a] = 0.0
        next_hop[camera_a, camera_b], next_hop[camera_b, camera_a] = camera_b, camera_a
    for via_camera in range(n_cameras):
        via_time = walk_time[:, via_camera:via_camera + 1] + walk_time[via_camera:via_camera + 1, :]
        is_shorter = via_time < walk_time
        walk_time = np.where(is_shorter, via_time, walk_time)
        next_hop = np.where(is_shorter, next_hop[:, via_camera:via_camera + 1], next_hop)
    return walk_time, next_hop


def overlap_groups(n_cameras: int, overlap_pairs) -> np.ndarray:
    """Connected components of the overlap relation -> group id per camera."""
    union_parent = list(range(n_cameras))

    def find_root(camera):
        while union_parent[camera] != camera:
            union_parent[camera] = union_parent[union_parent[camera]]
            camera = union_parent[camera]
        return camera

    for camera_a, camera_b in overlap_pairs:
        root_a, root_b = find_root(camera_a), find_root(camera_b)
        if root_a != root_b:
            union_parent[root_a] = root_b
    roots = [find_root(camera) for camera in range(n_cameras)]
    group_of_root = {root: group for group, root in enumerate(dict.fromkeys(roots))}
    return np.array([group_of_root[root] for root in roots], dtype=np.int64)
