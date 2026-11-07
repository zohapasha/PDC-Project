"""Synthetic campus generator with perfect ground truth.

Buildings with corridor camera graphs, entrance cameras linked by outdoor paths, people
walking shortest paths between destinations (with rests), single-camera fragmentation,
look-alike people, camera bias, and provisional (noisier) appearance vectors that are
later revised.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from ..common.topology import Topology, overlap_groups, shortest_paths
from ..common.types import EVENT_REVISION, EVENT_TRACKLET


@dataclass
class GenConfig:
    # Field names are configuration keys (the `gen:` block of the YAML configs) and part
    # of the cache key, so they are kept stable.
    n_buildings: int = 4
    cams_per_building: int = 8
    n_sites: int = 0                 # 0 -> one site per building; else building % n_sites
    n_people: int = 1000
    duration: float = 3600.0
    p_cross: float = 0.2             # probability that a trip goes to another building
    sigma: float = 0.5               # re-ID noise (norm of noise relative to identity)
    sigma_cam: float = 0.3           # camera bias
    dim: int = 128                   # appearance vector dimension
    lookalike_frac: float = 0.1      # fraction of people in look-alike groups
    lookalike_spread: float = 0.35
    p_frag: float = 0.1              # probability a visit is split into two tracklets
    p_revise: float = 0.0            # probability a tracklet gets a provisional vector
    revise_noise: float = 2.0        # provisional vector noise multiplier
    overlap_frac: float = 0.1        # fraction of corridors whose cameras overlap
    extra_edge_frac: float = 0.3     # extra corridors per building, relative to cameras
    walk_in: tuple = (5.0, 40.0)     # indoor corridor walking time range (s)
    walk_out: tuple = (60.0, 300.0)  # outdoor path walking time range (s)
    walk_slack: tuple = (1.0, 1.4)   # actual / minimum walking time
    dwell_mean: float = 10.0
    dwell_max: float = 120.0
    rest_prob: float = 0.5
    rest_mean: float = 120.0
    start_frac: float = 0.5          # people appear uniformly in [0, start_frac*duration]
    fps: float = 10.0                # only used to weight IDF1 by "frames"
    seed: int = 0

    def cache_key(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class Stream:
    """Event stream. Event type EVENT_TRACKLET (at the tracklet's end time) or
    EVENT_REVISION (its final vector)."""
    topology: Topology
    event_times: np.ndarray
    event_types: np.ndarray
    event_tracklet_ids: np.ndarray
    event_vectors: np.ndarray            # (n_events, dim) float32
    # per-tracklet tables (indexed by tracklet_id)
    camera_ids: np.ndarray
    start_times: np.ndarray
    end_times: np.ndarray
    gt_person_ids: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def n_tracklets(self):
        return len(self.camera_ids)

    def prefix(self, n_tracklets: int) -> "Stream":
        """Prefix with the first n tracklets (and revisions of them)."""
        keep = self.event_tracklet_ids < n_tracklets
        return Stream(self.topology, self.event_times[keep], self.event_types[keep],
                      self.event_tracklet_ids[keep], self.event_vectors[keep],
                      self.camera_ids[:n_tracklets], self.start_times[:n_tracklets],
                      self.end_times[:n_tracklets], self.gt_person_ids[:n_tracklets],
                      dict(self.metadata))

    def with_topology(self, topology: Topology) -> "Stream":
        return Stream(topology, self.event_times, self.event_types, self.event_tracklet_ids,
                      self.event_vectors, self.camera_ids, self.start_times, self.end_times,
                      self.gt_person_ids, dict(self.metadata))

    # ------------------------------------------------------------ persistence
    # The parquet column names are a file format and keep their original spelling.
    def save(self, path: str):
        import pyarrow as pa
        import pyarrow.parquet as pq
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        dim = self.event_vectors.shape[1]
        flat_vectors = pa.array(self.event_vectors.reshape(-1), type=pa.float32())
        tracklet_ids = self.event_tracklet_ids
        table = pa.table({
            "time": self.event_times, "event_type": self.event_types, "tid": tracklet_ids,
            "cam": self.camera_ids[tracklet_ids],
            "site": self.topology.site_of_camera[self.camera_ids[tracklet_ids]],
            "t_start": self.start_times[tracklet_ids], "t_end": self.end_times[tracklet_ids],
            "gt_pid": self.gt_person_ids[tracklet_ids],
            "vec": pa.FixedSizeListArray.from_arrays(flat_vectors, dim),
        })
        pq.write_table(table, path)
        with open(path.replace(".parquet", ".topology.json"), "w") as file:
            file.write(self.topology.to_json())
        with open(path.replace(".parquet", ".meta.json"), "w") as file:
            json.dump(self.metadata, file)

    @staticmethod
    def load(path: str) -> "Stream":
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        with open(path.replace(".parquet", ".topology.json")) as file:
            topology = Topology.from_json(file.read())
        metadata = {}
        metadata_path = path.replace(".parquet", ".meta.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as file:
                metadata = json.load(file)

        def column(name):
            return table.column(name).to_numpy()

        vector_column = table.column("vec").combine_chunks()
        dim = vector_column.type.list_size
        event_vectors = vector_column.values.to_numpy().reshape(-1, dim).astype(np.float32)
        event_tracklet_ids = column("tid").astype(np.int64)
        n_tracklets = int(event_tracklet_ids.max()) + 1
        camera_ids = np.zeros(n_tracklets, np.int64)
        start_times = np.zeros(n_tracklets)
        end_times = np.zeros(n_tracklets)
        gt_person_ids = np.zeros(n_tracklets, np.int64)
        camera_ids[event_tracklet_ids] = column("cam")
        start_times[event_tracklet_ids] = column("t_start")
        end_times[event_tracklet_ids] = column("t_end")
        gt_person_ids[event_tracklet_ids] = column("gt_pid")
        return Stream(topology, column("time").astype(np.float64),
                      column("event_type").astype(np.int8), event_tracklet_ids, event_vectors,
                      camera_ids, start_times, end_times, gt_person_ids, metadata)


# ====================================================================== topology
def make_topology(config: GenConfig, rng: np.random.Generator):
    n_buildings, cameras_per_building = config.n_buildings, config.cams_per_building
    n_cameras = n_buildings * cameras_per_building
    corridors, overlap_pairs = [], []
    building_of_camera = np.repeat(np.arange(n_buildings), cameras_per_building)
    is_entrance = np.zeros(n_cameras, bool)
    for building in range(n_buildings):
        first_camera = building * cameras_per_building
        # random spanning tree + extra corridors
        for i in range(1, cameras_per_building):
            j = int(rng.integers(0, i))
            corridors.append((first_camera + i, first_camera + j,
                              float(rng.uniform(*config.walk_in))))
        for _ in range(int(config.extra_edge_frac * cameras_per_building)):
            i, j = rng.choice(cameras_per_building, 2, replace=False)
            corridors.append((first_camera + int(i), first_camera + int(j),
                              float(rng.uniform(*config.walk_in))))
        n_entrances = int(rng.integers(1, 3)) if cameras_per_building > 1 else 1
        for offset in rng.choice(cameras_per_building, n_entrances, replace=False):
            is_entrance[first_camera + int(offset)] = True
    indoor_corridors = list(corridors)
    for (camera_a, camera_b, _) in indoor_corridors:
        if rng.random() < config.overlap_frac and not (is_entrance[camera_a] or is_entrance[camera_b]):
            overlap_pairs.append((camera_a, camera_b))
    overlap_pair_set = {tuple(sorted(pair)) for pair in overlap_pairs}
    corridors = [corridor for corridor in corridors
                 if tuple(sorted(corridor[:2])) not in overlap_pair_set]
    entrances_of_building = [np.nonzero(is_entrance & (building_of_camera == building))[0]
                             for building in range(n_buildings)]
    for building_a in range(n_buildings):
        for building_b in range(building_a + 1, n_buildings):
            entrance_a = int(rng.choice(entrances_of_building[building_a]))
            entrance_b = int(rng.choice(entrances_of_building[building_b]))
            corridors.append((entrance_a, entrance_b, float(rng.uniform(*config.walk_out))))
    walk_time, next_hop = shortest_paths(n_cameras, corridors, overlap_pairs)
    overlap_group_of_camera = overlap_groups(n_cameras, overlap_pairs)
    n_sites = config.n_sites or n_buildings
    topology = Topology(n_cameras, walk_time, overlap_group_of_camera, building_of_camera,
                        is_entrance, np.arange(n_buildings) % n_sites, corridors)
    return topology, next_hop


def _shortest_path(next_hop, source_camera, target_camera):
    path = [source_camera]
    while source_camera != target_camera:
        source_camera = int(next_hop[source_camera, target_camera])
        path.append(source_camera)
    return path


# ====================================================================== people
def generate(config: GenConfig) -> Stream:
    rng = np.random.default_rng(config.seed)
    topology, next_hop = make_topology(config, rng)
    walk_time = topology.walk_time
    n_buildings, cameras_per_building = config.n_buildings, config.cams_per_building
    cameras_of_building = [list(range(building * cameras_per_building,
                                      (building + 1) * cameras_per_building))
                           for building in range(n_buildings)]
    # lognormal(sigma=0.5) with the given mean
    dwell_log_mean = np.log(config.dwell_mean) - 0.125
    rest_log_mean = np.log(config.rest_mean) - 0.125

    visits = []   # (person_id, camera_id, start_time, end_time)
    for person_id in range(config.n_people):
        home_building = int(rng.integers(0, n_buildings))
        current_camera = int(rng.choice([camera for camera in cameras_of_building[home_building]
                                         if topology.is_entrance[camera]]))
        start_time = float(rng.uniform(0, config.start_frac * config.duration))
        dwell = min(config.dwell_max, float(rng.lognormal(dwell_log_mean, 0.5)))
        visits.append((person_id, current_camera, start_time, start_time + dwell))
        free_at = start_time + dwell                      # when the person can walk on
        last_start, last_dwell = start_time, dwell
        while free_at < config.duration:
            if rng.random() < config.p_cross and n_buildings > 1:
                other_building = int(rng.choice([building for building in range(n_buildings)
                                                 if building != home_building]))
                destination_camera = int(rng.choice(cameras_of_building[other_building]))
            else:
                destination_camera = int(rng.choice(cameras_of_building[home_building]))
            if destination_camera == current_camera:
                continue
            path = _shortest_path(next_hop, current_camera, destination_camera)
            for next_camera in path[1:]:
                step_walk_time = walk_time[current_camera, next_camera]
                dwell = min(config.dwell_max, float(rng.lognormal(dwell_log_mean, 0.5)))
                if step_walk_time == 0.0:   # overlapping view: appears while still visible
                    visit_start = last_start + float(rng.uniform(0.2, 0.8)) * last_dwell
                else:
                    visit_start = free_at + step_walk_time * float(rng.uniform(*config.walk_slack))
                visit_end = visit_start + dwell
                visits.append((person_id, next_camera, visit_start, visit_end))
                free_at = max(free_at, visit_end)
                last_start, last_dwell = visit_start, dwell
                current_camera = next_camera
            if rng.random() < config.rest_prob:
                free_at += float(rng.lognormal(rest_log_mean, 0.5))
    # fragmentation (single-camera tracker splits a visit into two tracklets)
    tracklet_records = []      # (person_id, camera_id, start_time, end_time)
    for person_id, camera_id, visit_start, visit_end in visits:
        if visit_end - visit_start > 4.0 and rng.random() < config.p_frag:
            split_time = float(rng.uniform(visit_start + 1.0, visit_end - 3.0))
            gap = float(rng.uniform(0.5, 2.0))
            tracklet_records.append((person_id, camera_id, visit_start, split_time))
            tracklet_records.append((person_id, camera_id, split_time + gap, visit_end))
        else:
            tracklet_records.append((person_id, camera_id, visit_start, visit_end))
    tracklet_records = [record for record in tracklet_records if record[3] <= config.duration]
    # arrival order = end time
    tracklet_records.sort(key=lambda record: (record[3], record[2], record[1], record[0]))
    n_tracklets = len(tracklet_records)
    gt_person_ids = np.array([record[0] for record in tracklet_records], np.int64)
    camera_ids = np.array([record[1] for record in tracklet_records], np.int64)
    start_times = np.array([record[2] for record in tracklet_records], np.float64)
    end_times = np.array([record[3] for record in tracklet_records], np.float64)

    # appearance
    dim = config.dim

    def gaussian(*shape):
        return rng.standard_normal(shape).astype(np.float32) / np.sqrt(dim)

    def normalize(vectors):
        return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)

    identity_vectors = normalize(gaussian(config.n_people, dim) * np.sqrt(dim))
    n_lookalikes = int(config.lookalike_frac * config.n_people)
    lookalike_people = rng.permutation(config.n_people)[:n_lookalikes]
    i = 0
    while i < len(lookalike_people):
        group_size = int(rng.integers(2, 5))
        group_members = lookalike_people[i:i + group_size]
        center = identity_vectors[group_members[0]].copy()
        for member in group_members:
            identity_vectors[member] = normalize(center + config.lookalike_spread * gaussian(dim))
        i += group_size
    camera_bias = config.sigma_cam * gaussian(topology.n_cameras, dim) * 1.0
    final_vectors = normalize(identity_vectors[gt_person_ids] + camera_bias[camera_ids]
                              + config.sigma * gaussian(n_tracklets, dim))
    is_revised = rng.random(n_tracklets) < config.p_revise
    provisional_vectors = final_vectors.copy()
    if is_revised.any():
        n_revised = int(is_revised.sum())
        provisional_vectors[is_revised] = normalize(
            identity_vectors[gt_person_ids[is_revised]] + camera_bias[camera_ids[is_revised]]
            + config.sigma * config.revise_noise * gaussian(n_revised, dim))
    revision_times = end_times + rng.uniform(5.0, 30.0, n_tracklets)
    # events
    event_times = np.concatenate([end_times, revision_times[is_revised]])
    event_types = np.concatenate([np.full(n_tracklets, EVENT_TRACKLET, np.int8),
                                  np.full(int(is_revised.sum()), EVENT_REVISION, np.int8)])
    event_tracklet_ids = np.concatenate([np.arange(n_tracklets), np.nonzero(is_revised)[0]])
    event_vectors = np.concatenate([provisional_vectors, final_vectors[is_revised]]).astype(np.float32)
    order = np.lexsort((event_tracklet_ids, event_types, event_times))
    metadata = {"gen": asdict(config), "n_tracklets": n_tracklets,
                "n_revisions": int(is_revised.sum())}
    return Stream(topology, event_times[order], event_types[order], event_tracklet_ids[order],
                  event_vectors[order], camera_ids, start_times, end_times, gt_person_ids, metadata)


def cache_path(config: GenConfig, cache_dir: str = "data/cache") -> str:
    return os.path.join(cache_dir, f"synth_{config.cache_key()}.parquet")


def load_or_generate(config: GenConfig, cache_dir: str = "data/cache") -> Stream:
    path = cache_path(config, cache_dir)
    if os.path.exists(path):
        return Stream.load(path)
    stream = generate(config)
    stream.save(path)
    return stream
