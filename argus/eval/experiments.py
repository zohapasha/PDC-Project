"""Experiments E1-E7. Each function takes (params, seed) and returns a list of result
rows (dicts). The harness adds provenance and appends rows to results/<exp>.csv."""
from __future__ import annotations

import os
import time
from dataclasses import replace

import numpy as np

from ..baselines.offline import connected_components, full_recompute
from ..common.constraints import Checker
from ..common.types import (CORRECTION_KINDS, K_CONFLICT_OK, K_CONFLICT_REJ, K_DEL_BACKUP,
                            K_DEL_TREE, KIND_NAMES)
from ..gen.campus import GenConfig, Stream, load_or_generate
from ..sync.sim import b6_bytes, run_centralized_hub, run_single, run_two_level
from .metrics import agreement, evaluate

PRESETS = {
    "small":  dict(n_buildings=2, cams_per_building=4, n_people=200, duration=1800.0),
    "medium": dict(n_buildings=4, cams_per_building=8, n_people=1000, duration=3600.0),
    "large":  dict(n_buildings=8, cams_per_building=12, n_people=3000, duration=5400.0),
}
DEFAULT_ENGINE = dict(W=300.0, tau_keep=0.5, tau_accept=0.6, D_max=600.0)


def gen_cfg(p: dict, seed: int) -> GenConfig:
    base = dict(PRESETS[p.get("preset", "medium")])
    base.update(p.get("gen", {}))
    return GenConfig(seed=seed, **base)


def get_stream(p: dict, seed: int) -> Stream:
    """Real-data configs give `stream: path.parquet` (prepared by scripts/prepare_real.py);
    otherwise the synthetic generator is used with the given seed."""
    if p.get("stream"):
        return Stream.load(p["stream"])
    return load_or_generate(gen_cfg(p, seed))


def eng_cfg(p: dict) -> dict:
    c = dict(DEFAULT_ENGINE)
    c.update(p.get("engine", {}))
    return c


def _pct(a, qs=(50, 95, 99)):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {f"p{q}": float("nan") for q in qs}
    return {f"p{q}": float(np.percentile(a, q)) for q in qs}


def _acc_row(part, stream, chk, ref=None):
    m = evaluate(part, stream, chk)
    if ref is not None:
        m["agree_offline"] = agreement(part, ref)
    return m


def _stream_info(s: Stream):
    return {"n_tracklets": s.n_tracklets, "n_events": len(s.time),
            "n_cams": s.topo.n_cams, "n_sites": s.topo.n_sites}


# ============================================================================ E1
def e1_accuracy(p, seed):
    s = get_stream(p, seed)
    cfg = eng_cfg(p)
    chk = Checker(s.topo)
    rows = []
    node, _ = run_single(s, cfg, mode="full")
    eng = node.engine
    ref, _ = full_recompute(eng)
    base = _stream_info(s)
    rows.append({"method": "ours", **base, **_acc_row(eng.partition(), s, chk, ref)})
    rows.append({"method": "B1_offline", **base, **_acc_row(ref, s, chk, ref)})
    rows.append({"method": "B4_no_constraints", **base,
                 **_acc_row(connected_components(eng), s, chk, ref)})
    n3, _ = run_single(s, cfg, mode="no_correction")
    rows.append({"method": "B3_no_correction", **base, **_acc_row(n3.engine.partition(), s, chk, ref)})
    if s.topo.n_sites > 1:
        tl = run_two_level(s, cfg)
        part = tl["hub"].eng.partition()
        r = _acc_row(part, s, chk, ref)
        r["agree_centralized"] = agreement(part, eng.partition())
        rows.append({"method": "ours_two_level", **base, **r})
    return rows


# ============================================================================ E2
def e2_incremental(p, seed):
    """Per-update latency of ours vs. the cost of one full recomputation (B1) at the same
    graph size."""
    s = load_or_generate(gen_cfg(p, seed))
    cfg = eng_cfg(p)
    cps = [c for c in p.get("checkpoints", [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000])
           if c <= s.n_tracklets]
    b1 = {}

    def on_cp(n, node):
        _, sec = full_recompute(node.engine)
        b1[n] = (sec, sum(len(a) for a in node.engine.adj.values()) // 2)

    node, lat = run_single(s, cfg, checkpoints=cps, on_checkpoint=on_cp, time_events=True)
    lat = np.array(lat, dtype=np.float64)              # (etype, ns, n_tracklets)
    rows, prev = [], 0
    for n in cps:
        sel = lat[(lat[:, 2] > max(prev, n * 0.8)) & (lat[:, 2] <= n), 1] / 1e6
        rows.append({"n_tracklets": n, "n_stored_matches": b1[n][1], "method": "ours",
                     "update_ms_mean": float(sel.mean()), **{f"update_ms_{k}": v for k, v in _pct(sel).items()}})
        rows.append({"n_tracklets": n, "n_stored_matches": b1[n][1], "method": "B1_full_recompute",
                     "update_ms_mean": b1[n][0] * 1e3, "update_ms_p50": b1[n][0] * 1e3,
                     "update_ms_p95": b1[n][0] * 1e3, "update_ms_p99": b1[n][0] * 1e3})
        prev = n
    return rows


# ============================================================================ E3
def e3_corrections(p, seed):
    """Per-correction cost: ours (subtree + backup links) vs B2 (component rebuild).
    Returns summary rows; per-op rows are written to results/e3_ops_<seed>.parquet."""
    s = load_or_generate(gen_cfg(p, seed))
    cfg = eng_cfg(p)
    op_rate = p.get("op_rate", 0.0)
    rows, ops = [], []
    for method, mode in (("ours", "full"), ("B2_component", "component")):
        node, _ = run_single(s, cfg, mode=mode, op_rate=op_rate, seed=seed)
        log = node.engine.oplog
        n_v = len(node.engine.V)
        for (kind, msgs, touched, size, nV, ns, extra) in log:
            if kind in (K_CONFLICT_OK, K_CONFLICT_REJ, K_DEL_TREE, K_DEL_BACKUP):
                ops.append({"method": method, "seed": seed, "kind": KIND_NAMES[kind], "msgs": msgs,
                            "touched": touched, "person_size": size, "graph_size": nV,
                            "latency_us": ns / 1e3, "extra": extra})
        corr = [o for o in log if o[0] in CORRECTION_KINDS]
        dels = [o for o in log if o[0] in (K_DEL_TREE, K_DEL_BACKUP)]
        tree_dels = [o for o in log if o[0] == K_DEL_TREE]
        row = {"method": method, "n_vertices": n_v, "n_corrections": len(corr),
               "n_conflict_ok": sum(o[0] == K_CONFLICT_OK for o in log),
               "n_conflict_rej": sum(o[0] == K_CONFLICT_REJ for o in log),
               "n_deletions": len(dels),
               "frac_del_hit_backup": (sum(o[0] == K_DEL_BACKUP for o in dels) / len(dels)) if dels else float("nan"),
               "frac_treedel_repaired_by_backup": (sum(o[6] for o in tree_dels) / len(tree_dels)) if tree_dels else float("nan"),
               "touched_mean": float(np.mean([o[2] for o in corr])) if corr else float("nan"),
               "touched_over_size_max": float(max((o[2] / max(o[3], 1) for o in corr), default=float("nan"))),
               "msgs_mean": float(np.mean([o[1] for o in corr])) if corr else float("nan"),
               **{f"corr_us_{k}": v / 1e3 for k, v in _pct([o[5] for o in corr]).items()}}
        depths = node.engine.depths()
        row.update({"tree_depth_mean": float(np.mean(depths)), "tree_depth_max": int(np.max(depths))})
        row.update(_acc_row(node.engine.partition(), s, Checker(s.topo)))
        rows.append(row)
    os.makedirs("results", exist_ok=True)
    import pandas as pd
    pd.DataFrame(ops).to_parquet(f"results/e3_ops_{p.get('tag', 'default')}_{seed}.parquet")
    return rows


# ============================================================================ E4
def e4_scaling(p, seed):
    """Throughput and end-to-end latency vs #sites (multi-process). kind=weak|strong."""
    from ..sync.mp import run_mp
    kind = p.get("kind", "weak")
    rows = []
    for n_sites in p.get("sites", [1, 2, 4, 8]):
        if kind == "weak":
            g = gen_cfg(p, seed)
            g = replace(g, n_buildings=n_sites, n_sites=n_sites,
                        n_people=p.get("people_per_site", 250) * n_sites)
        else:
            g = replace(gen_cfg(p, seed), n_sites=n_sites)
        s = load_or_generate(g)
        path = f"data/cache/synth_{g.key()}.parquet"
        for method in p.get("methods", ["two_level", "central"]):
            for speed_name, speed in (("max", None), ("fixed", p.get("speed", 20.0))):
                r = run_mp(path, s.topo, eng_cfg(p), mode=method, speed=speed)
                lat = r["hub"]["e2e_lat_ns"] / 1e6
                site_lat = np.concatenate([x["site_lat_ns"] for x in r["sites"]]) / 1e6
                rows.append({
                    "kind": kind, "n_sites": n_sites, "method": "ours" if method == "two_level" else "B5_centralized",
                    "load": speed_name, "speed": speed or 0, **_stream_info(s),
                    "throughput_ev_s": r["throughput"], "elapsed_s": r["elapsed_s"],
                    **{f"e2e_ms_{k}": v for k, v in _pct(lat).items()},
                    **{f"site_ms_{k}": v for k, v in _pct(site_lat).items()},
                    "hub_cpu_util": r["hub"]["cpu_s"] / max(r["elapsed_s"], 1e-9),
                    "hub_rss_mb": r["hub"]["rss_peak"] / 2**20,
                    "site_rss_mb_max": max(x["rss_peak"] for x in r["sites"]) / 2**20,
                    "hub_msgs": r["hub"]["n_msgs"],
                })
    return rows


# ============================================================================ E5
def e5_bytes(p, seed):
    s = load_or_generate(gen_cfg(p, seed))
    cfg = eng_cfg(p)
    rows = []
    tl = run_two_level(s, cfg)
    base = _stream_info(s)
    rows.append({"method": "ours", **base, "total_bytes": sum(tl["bytes"].values()),
                 "staleness_s": 0.0, **{f"bytes_{k}": v for k, v in tl["bytes"].items()}})
    c = run_centralized_hub(s, cfg)
    rows.append({"method": "B5_centralized", **base, "total_bytes": sum(c["bytes"].values()),
                 "staleness_s": 0.0, **{f"bytes_{k}": v for k, v in c["bytes"].items()}})
    for T in p.get("periods", [1, 5, 30]):
        b = b6_bytes(s, cfg, float(T))
        rows.append({"method": f"B6_periodic_T{T}", **base, "total_bytes": sum(b["bytes"].values()),
                     "staleness_s": b["staleness_mean"], **{f"bytes_{k}": v for k, v in b["bytes"].items()}})
    return rows


# ============================================================================ E6
def e6_noise(p, seed):
    rows = []
    for sigma in p.get("sigmas", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        q = dict(p)
        q["gen"] = dict(p.get("gen", {}), sigma=sigma)
        s = load_or_generate(gen_cfg(q, seed))
        cfg = eng_cfg(q)
        chk = Checker(s.topo)
        dur_min = s.meta["gen"]["duration"] / 60.0
        node, _ = run_single(s, cfg)
        eng = node.engine
        corr = [o for o in eng.oplog if o[0] in CORRECTION_KINDS]
        r = _acc_row(eng.partition(), s, chk)
        rows.append({"sigma": sigma, "method": "ours", **r,
                     "corrections_per_min": len(corr) / dur_min,
                     **{f"corr_us_{k}": v / 1e3 for k, v in _pct([o[5] for o in corr]).items()}})
        rows.append({"sigma": sigma, "method": "B4_no_constraints",
                     **_acc_row(connected_components(eng), s, chk)})
        n3, _ = run_single(s, cfg, mode="no_correction")
        rows.append({"sigma": sigma, "method": "B3_no_correction", **_acc_row(n3.engine.partition(), s, chk)})
    return rows


# ============================================================================ E7
def e7_cross(p, seed):
    rows = []
    for pc in p.get("p_cross", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
        q = dict(p)
        q["gen"] = dict(p.get("gen", {}), p_cross=pc)
        s = load_or_generate(gen_cfg(q, seed))
        cfg = eng_cfg(q)
        chk = Checker(s.topo)
        node, _ = run_single(s, cfg)
        central = node.engine.partition()
        t0 = time.perf_counter()
        tl = run_two_level(s, cfg)
        dt = time.perf_counter() - t0
        hub = tl["hub"].eng
        part = hub.partition()
        dur = s.meta["gen"]["duration"]
        hub_ops = len(hub.oplog)
        rows.append({"p_cross": pc, "method": "ours_two_level_a", **_stream_info(s),
                     **_acc_row(part, s, chk), "agree_centralized": agreement(part, central),
                     "hub_msgs_per_stream_s": sum(tl["msgs"].values()) / dur,
                     "hub_ops_per_stream_s": hub_ops / dur,
                     "hub_cross_cuts": sum(1 for o in hub.oplog if o[0] == K_CONFLICT_OK),
                     "total_bytes": sum(tl["bytes"].values()), "sim_wall_s": dt})
        rows.append({"p_cross": pc, "method": "centralized", **_stream_info(s), **_acc_row(central, s, chk),
                     "agree_centralized": 1.0})
    return rows


EXPERIMENTS = {"e1": e1_accuracy, "e2": e2_incremental, "e3": e3_corrections, "e4": e4_scaling,
               "e5": e5_bytes, "e6": e6_noise, "e7": e7_cross}
