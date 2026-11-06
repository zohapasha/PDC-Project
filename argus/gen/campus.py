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


@dataclass
class GenConfig:
    n_buildings: int = 4
    cams_per_building: int = 8
    n_sites: int = 0                 # 0 -> one site per building; else building % n_sites
    n_people: int = 1000
    duration: float = 3600.0
    p_cross: float = 0.2
    sigma: float = 0.5               # re-ID noise (norm of noise relative to identity)
    sigma_cam: float = 0.3           # camera bias
    dim: int = 128
    lookalike_frac: float = 0.1
    lookalike_spread: float = 0.35
    p_frag: float = 0.1
    p_revise: float = 0.0
    revise_noise: float = 2.0        # provisional vector noise multiplier
    overlap_frac: float = 0.1
    extra_edge_frac: float = 0.3
    walk_in: tuple = (5.0, 40.0)
    walk_out: tuple = (60.0, 300.0)
    walk_slack: tuple = (1.0, 1.4)
    dwell_mean: float = 10.0
    dwell_max: float = 120.0
    rest_prob: float = 0.5
    rest_mean: float = 120.0
    start_frac: float = 0.5          # people appear uniformly in [0, start_frac*duration]
    fps: float = 10.0                # only used to weight IDF1 by "frames"
    seed: int = 0

    def key(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class Stream:
    """Event stream. etype 0 = tracklet (at te), 1 = revision (final vector)."""
    topo: Topology
    time: np.ndarray
    etype: np.ndarray
    tid: np.ndarray
    vec: np.ndarray                  # (n_events, dim) float32
    # per-tracklet tables (indexed by tid)
    cam: np.ndarray
    ts: np.ndarray
    te: np.ndarray
    gt: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_tracklets(self):
        return len(self.cam)

    def head(self, n_tracklets: int) -> "Stream":
        """Prefix with the first n tracklets (and revisions of them)."""
        keep = self.tid < n_tracklets
        return Stream(self.topo, self.time[keep], self.etype[keep], self.tid[keep],
                      self.vec[keep], self.cam[:n_tracklets], self.ts[:n_tracklets],
                      self.te[:n_tracklets], self.gt[:n_tracklets], dict(self.meta))

    def with_topology(self, topo: Topology) -> "Stream":
        return Stream(topo, self.time, self.etype, self.tid, self.vec, self.cam, self.ts,
                      self.te, self.gt, dict(self.meta))

    # ------------------------------------------------------------ persistence
    def save(self, path: str):
        import pyarrow as pa
        import pyarrow.parquet as pq
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        dim = self.vec.shape[1]
        flat = pa.array(self.vec.reshape(-1), type=pa.float32())
        t = pa.table({
            "time": self.time, "event_type": self.etype, "tid": self.tid,
            "cam": self.cam[self.tid], "site": self.topo.site[self.cam[self.tid]],
            "t_start": self.ts[self.tid], "t_end": self.te[self.tid],
            "gt_pid": self.gt[self.tid],
            "vec": pa.FixedSizeListArray.from_arrays(flat, dim),
        })
        pq.write_table(t, path)
        with open(path.replace(".parquet", ".topology.json"), "w") as f:
            f.write(self.topo.to_json())
        with open(path.replace(".parquet", ".meta.json"), "w") as f:
            json.dump(self.meta, f)

    @staticmethod
    def load(path: str) -> "Stream":
        import pyarrow.parquet as pq
        t = pq.read_table(path)
        with open(path.replace(".parquet", ".topology.json")) as f:
            topo = Topology.from_json(f.read())
        meta = {}
        mp = path.replace(".parquet", ".meta.json")
        if os.path.exists(mp):
            with open(mp) as f:
                meta = json.load(f)
        col = lambda n: t.column(n).to_numpy()
        vcol = t.column("vec").combine_chunks()
        dim = vcol.type.list_size
        vec = vcol.values.to_numpy().reshape(-1, dim).astype(np.float32)
        tid = col("tid").astype(np.int64)
        n = int(tid.max()) + 1
        cam = np.zeros(n, np.int64); ts = np.zeros(n); te = np.zeros(n); gt = np.zeros(n, np.int64)
        cam[tid] = col("cam"); ts[tid] = col("t_start"); te[tid] = col("t_end"); gt[tid] = col("gt_pid")
        return Stream(topo, col("time").astype(np.float64), col("event_type").astype(np.int8),
                      tid, vec, cam, ts, te, gt, meta)


# ====================================================================== topology
def make_topology(cfg: GenConfig, rng: np.random.Generator):
    nb, k = cfg.n_buildings, cfg.cams_per_building
    n = nb * k
    edges, overlap_pairs = [], []
    building = np.repeat(np.arange(nb), k)
    entrance = np.zeros(n, bool)
    for b in range(nb):
        base = b * k
        # random spanning tree + extra corridors
        for i in range(1, k):
            j = int(rng.integers(0, i))
            edges.append((base + i, base + j, float(rng.uniform(*cfg.walk_in))))
        for _ in range(int(cfg.extra_edge_frac * k)):
            i, j = rng.choice(k, 2, replace=False)
            edges.append((base + int(i), base + int(j), float(rng.uniform(*cfg.walk_in))))
        n_ent = int(rng.integers(1, 3)) if k > 1 else 1
        for e in rng.choice(k, n_ent, replace=False):
            entrance[base + int(e)] = True
    intra = list(edges)
    for (a, b_, _) in intra:
        if rng.random() < cfg.overlap_frac and not (entrance[a] or entrance[b_]):
            overlap_pairs.append((a, b_))
    ov_set = {tuple(sorted(p)) for p in overlap_pairs}
    edges = [e for e in edges if tuple(sorted(e[:2])) not in ov_set]
    ents = [np.nonzero(entrance & (building == b))[0] for b in range(nb)]
    for b1 in range(nb):
        for b2 in range(b1 + 1, nb):
            e1, e2 = int(rng.choice(ents[b1])), int(rng.choice(ents[b2]))
            edges.append((e1, e2, float(rng.uniform(*cfg.walk_out))))
    D, nxt = shortest_paths(n, edges, overlap_pairs)
    group = overlap_groups(n, overlap_pairs)
    ns = cfg.n_sites or nb
    topo = Topology(n, D, group, building, entrance, np.arange(nb) % ns, edges)
    return topo, nxt


def _path(nxt, a, b):
    p = [a]
    while a != b:
        a = int(nxt[a, b])
        p.append(a)
    return p


# ====================================================================== people
def generate(cfg: GenConfig) -> Stream:
    rng = np.random.default_rng(cfg.seed)
    topo, nxt = make_topology(cfg, rng)
    D = topo.D
    nb, k = cfg.n_buildings, cfg.cams_per_building
    cams_of_b = [list(range(b * k, (b + 1) * k)) for b in range(nb)]
    ln_mu = np.log(cfg.dwell_mean) - 0.125         # lognormal(sigma=0.5) with given mean
    rest_mu = np.log(cfg.rest_mean) - 0.125

    visits = []   # (pid, cam, ts, te)
    for pid in range(cfg.n_people):
        home = int(rng.integers(0, nb))
        cur = int(rng.choice([c for c in cams_of_b[home] if topo.entrance[c]]))
        t = float(rng.uniform(0, cfg.start_frac * cfg.duration))
        dwell = min(cfg.dwell_max, float(rng.lognormal(ln_mu, 0.5)))
        visits.append((pid, cur, t, t + dwell))
        t_free = t + dwell
        last_start, last_dwell = t, dwell
        while t_free < cfg.duration:
            if rng.random() < cfg.p_cross and nb > 1:
                ob = int(rng.choice([b for b in range(nb) if b != home]))
                dest = int(rng.choice(cams_of_b[ob]))
            else:
                dest = int(rng.choice(cams_of_b[home]))
            if dest == cur:
                continue
            path = _path(nxt, cur, dest)
            for nx in path[1:]:
                d = D[cur, nx]
                dwell = min(cfg.dwell_max, float(rng.lognormal(ln_mu, 0.5)))
                if d == 0.0:      # overlapping view: appears while still visible in cur
                    s = last_start + float(rng.uniform(0.2, 0.8)) * last_dwell
                else:
                    s = t_free + d * float(rng.uniform(*cfg.walk_slack))
                e = s + dwell
                visits.append((pid, nx, s, e))
                t_free = max(t_free, e)
                last_start, last_dwell = s, dwell
                cur = nx
            if rng.random() < cfg.rest_prob:
                t_free += float(rng.lognormal(rest_mu, 0.5))
    # fragmentation (single-camera tracker splits a visit into two tracklets)
    tr = []
    for pid, cam, s, e in visits:
        if e - s > 4.0 and rng.random() < cfg.p_frag:
            cut = float(rng.uniform(s + 1.0, e - 3.0))
            gap = float(rng.uniform(0.5, 2.0))
            tr.append((pid, cam, s, cut))
            tr.append((pid, cam, cut + gap, e))
        else:
            tr.append((pid, cam, s, e))
    tr = [x for x in tr if x[3] <= cfg.duration]
    tr.sort(key=lambda x: (x[3], x[2], x[1], x[0]))     # arrival order = te
    n = len(tr)
    gt = np.array([x[0] for x in tr], np.int64)
    cam = np.array([x[1] for x in tr], np.int64)
    ts = np.array([x[2] for x in tr], np.float64)
    te = np.array([x[3] for x in tr], np.float64)

    # appearance
    dim = cfg.dim
    g = lambda *shape: rng.standard_normal(shape).astype(np.float32) / np.sqrt(dim)
    unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    Z = unit(g(cfg.n_people, dim) * np.sqrt(dim))
    n_look = int(cfg.lookalike_frac * cfg.n_people)
    order = rng.permutation(cfg.n_people)[:n_look]
    i = 0
    while i < len(order):
        size = int(rng.integers(2, 5))
        grp = order[i:i + size]
        center = Z[grp[0]].copy()
        for p in grp:
            Z[p] = unit(center + cfg.lookalike_spread * g(dim))
        i += size
    Bc = cfg.sigma_cam * g(topo.n_cams, dim) * 1.0
    final = unit(Z[gt] + Bc[cam] + cfg.sigma * g(n, dim))
    revise = rng.random(n) < cfg.p_revise
    prov = final.copy()
    if revise.any():
        m = int(revise.sum())
        prov[revise] = unit(Z[gt[revise]] + Bc[cam[revise]] +
                            cfg.sigma * cfg.revise_noise * g(m, dim))
    t_rev = te + rng.uniform(5.0, 30.0, n)
    # events
    ev_t = np.concatenate([te, t_rev[revise]])
    ev_type = np.concatenate([np.zeros(n, np.int8), np.ones(int(revise.sum()), np.int8)])
    ev_tid = np.concatenate([np.arange(n), np.nonzero(revise)[0]])
    ev_vec = np.concatenate([prov, final[revise]]).astype(np.float32)
    o = np.lexsort((ev_tid, ev_type, ev_t))
    meta = {"gen": asdict(cfg), "n_tracklets": n, "n_revisions": int(revise.sum())}
    return Stream(topo, ev_t[o], ev_type[o], ev_tid[o], ev_vec[o], cam, ts, te, gt, meta)


def load_or_generate(cfg: GenConfig, cache_dir: str = "data/cache") -> Stream:
    path = os.path.join(cache_dir, f"synth_{cfg.key()}.parquet")
    if os.path.exists(path):
        return Stream.load(path)
    s = generate(cfg)
    s.save(path)
    return s
