"""Experiments E1-E9. Each function takes (params, seed) and returns a list of result
rows (dicts). The harness adds provenance and appends rows to results/<exp>.csv.
Row keys are result-CSV column names (read by scripts/make_figures.py)."""
from __future__ import annotations

import os
import time
from dataclasses import replace

import numpy as np

from ..baselines.offline import connected_components, full_recompute
from ..common.constraints import ConstraintChecker
from ..common.types import (CORRECTION_OP_KINDS, OP_CONFLICT_REJECTED, OP_CONFLICT_RESOLVED,
                            OP_DELETE_BACKUP, OP_DELETE_TREE_EDGE, OP_KIND_NAMES)
from ..gen.campus import GenConfig, Stream, cache_path, load_or_generate
from ..sync.sim import b6_periodic_sync_bytes, run_centralized_hub, run_single, run_two_level
from .metrics import agreement, evaluate

PRESETS = {
    "small":  dict(n_buildings=2, cams_per_building=4, n_people=200, duration=1800.0),
    "medium": dict(n_buildings=4, cams_per_building=8, n_people=1000, duration=3600.0),
    "large":  dict(n_buildings=8, cams_per_building=12, n_people=3000, duration=5400.0),
}
# window_slack = W (walking slack on top of the minimum walking time, s);
# max_walk_time = D_max (camera pairs farther apart are never matched, s)
DEFAULT_ENGINE_CONFIG = dict(window_slack=300.0, tau_keep=0.5, tau_accept=0.6, max_walk_time=600.0)


def generator_config(params: dict, seed: int) -> GenConfig:
    settings = dict(PRESETS[params.get("preset", "medium")])
    settings.update(params.get("gen", {}))
    return GenConfig(seed=seed, **settings)


def load_stream(params: dict, seed: int) -> Stream:
    """Real-data configs give `stream: path.parquet` (prepared by scripts/prepare_real.py);
    otherwise the synthetic generator is used with the given seed."""
    if params.get("stream"):
        return _apply_border_radius(Stream.load(params["stream"]), params)
    return _load_synthetic(generator_config(params, seed), params)


def _apply_border_radius(stream: Stream, params: dict) -> Stream:
    radius = float(params.get("border_radius", 0.0))
    return stream.with_topology(stream.topology.with_border_radius(radius)) if radius > 0 else stream


def _load_synthetic(config: GenConfig, params: dict) -> Stream:
    """Synthetic stream with the config's hub matching radius applied to its topology."""
    return _apply_border_radius(load_or_generate(config), params)


def engine_config(params: dict) -> dict:
    config = dict(DEFAULT_ENGINE_CONFIG)
    config.update(params.get("engine", {}))
    return config


def _percentiles(values, quantiles=(50, 95, 99)):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {f"p{q}": float("nan") for q in quantiles}
    return {f"p{q}": float(np.percentile(values, q)) for q in quantiles}


def _accuracy_metrics(partition, stream, checker, offline_partition=None):
    metrics = evaluate(partition, stream, checker)
    if offline_partition is not None:
        metrics["agree_offline"] = agreement(partition, offline_partition)
    return metrics


def _stream_info(stream: Stream):
    return {"n_tracklets": stream.n_tracklets, "n_events": len(stream.event_times),
            "n_cams": stream.topology.n_cameras, "n_sites": stream.topology.n_sites}


def _mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


# ============================================================================ E1
def e1_accuracy(params, seed):
    stream = load_stream(params, seed)
    config = engine_config(params)
    checker = ConstraintChecker(stream.topology)
    rows = []
    node, _ = run_single(stream, config, mode="full")
    engine = node.engine
    offline_partition, _ = full_recompute(engine)
    info = _stream_info(stream)
    rows.append({"method": "ours", **info,
                 **_accuracy_metrics(engine.partition(), stream, checker, offline_partition)})
    rows.append({"method": "B1_offline", **info,
                 **_accuracy_metrics(offline_partition, stream, checker, offline_partition)})
    rows.append({"method": "B4_no_constraints", **info,
                 **_accuracy_metrics(connected_components(engine), stream, checker, offline_partition)})
    no_correction_node, _ = run_single(stream, config, mode="no_correction")
    rows.append({"method": "B3_no_correction", **info,
                 **_accuracy_metrics(no_correction_node.engine.partition(), stream, checker,
                                     offline_partition)})
    if stream.topology.n_sites > 1:
        for ownership, method in (("b", "ours_two_level"), ("a", "ours_two_level_a")):
            two_level = run_two_level(stream, dict(config, ownership=ownership))
            hub_partition = two_level["hub"].engine.partition()
            row = _accuracy_metrics(hub_partition, stream, checker, offline_partition)
            row["agree_centralized"] = agreement(hub_partition, engine.partition())
            row["split_requests"] = two_level["hub"].n_split_requests
            row["total_bytes"] = sum(two_level["bytes"].values())
            rows.append({"method": method, **info, **row})
    return rows


# ============================================================================ E2
def e2_incremental(params, seed):
    """Per-update latency of ours vs. the cost of one full recomputation (B1) at the same
    graph size."""
    stream = _load_synthetic(generator_config(params, seed), params)
    config = engine_config(params)
    checkpoints = [n for n in params.get("checkpoints",
                                         [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000])
                   if n <= stream.n_tracklets]
    recompute_at = {}          # n_tracklets -> (B1 seconds, n stored matches)

    def on_checkpoint(n_tracklets, node):
        _, seconds = full_recompute(node.engine)
        n_stored_matches = sum(len(scores) for scores in node.engine.match_scores.values()) // 2
        recompute_at[n_tracklets] = (seconds, n_stored_matches)

    node, latencies = run_single(stream, config, checkpoints=checkpoints,
                                 on_checkpoint=on_checkpoint, time_events=True)
    latencies = np.array(latencies, dtype=np.float64)    # (event_type, ns, n_tracklets)
    rows, previous_checkpoint = [], 0
    for n_tracklets in checkpoints:
        in_range = ((latencies[:, 2] > max(previous_checkpoint, n_tracklets * 0.8))
                    & (latencies[:, 2] <= n_tracklets))
        update_ms = latencies[in_range, 1] / 1e6
        recompute_seconds, n_stored_matches = recompute_at[n_tracklets]
        rows.append({"n_tracklets": n_tracklets, "n_stored_matches": n_stored_matches,
                     "method": "ours", "update_ms_mean": float(update_ms.mean()),
                     **{f"update_ms_{k}": v for k, v in _percentiles(update_ms).items()}})
        recompute_ms = recompute_seconds * 1e3
        rows.append({"n_tracklets": n_tracklets, "n_stored_matches": n_stored_matches,
                     "method": "B1_full_recompute", "update_ms_mean": recompute_ms,
                     "update_ms_p50": recompute_ms, "update_ms_p95": recompute_ms,
                     "update_ms_p99": recompute_ms})
        previous_checkpoint = n_tracklets
    return rows


# ============================================================================ E3
def e3_corrections(params, seed):
    """Per-correction cost: ours (subtree + backup links) vs B2 (component rebuild).
    Returns summary rows; per-op rows are written to results/e3_ops_<tag>_<seed>.parquet."""
    stream = _load_synthetic(generator_config(params, seed), params)
    config = engine_config(params)
    operator_rate = params.get("op_rate", 0.0)        # operator corrections per minute
    rows, op_rows = [], []
    for method, mode in (("ours", "full"), ("B2_component", "component")):
        node, _ = run_single(stream, config, mode=mode,
                             operator_corrections_per_minute=operator_rate, seed=seed)
        op_log = node.engine.op_log
        n_vertices = len(node.engine.vertices)
        for record in op_log:
            if record.kind in (OP_CONFLICT_RESOLVED, OP_CONFLICT_REJECTED, OP_DELETE_TREE_EDGE,
                               OP_DELETE_BACKUP):
                op_rows.append({"method": method, "seed": seed, "kind": OP_KIND_NAMES[record.kind],
                                "msgs": record.n_messages, "touched": record.n_touched,
                                "person_size": record.person_size, "graph_size": record.graph_size,
                                "latency_us": record.latency_ns / 1e3, "extra": record.extra})
        corrections = [record for record in op_log if record.kind in CORRECTION_OP_KINDS]
        deletions = [record for record in op_log
                     if record.kind in (OP_DELETE_TREE_EDGE, OP_DELETE_BACKUP)]
        tree_deletions = [record for record in op_log if record.kind == OP_DELETE_TREE_EDGE]
        row = {"method": method, "n_vertices": n_vertices, "n_corrections": len(corrections),
               "n_conflict_ok": sum(record.kind == OP_CONFLICT_RESOLVED for record in op_log),
               "n_conflict_rej": sum(record.kind == OP_CONFLICT_REJECTED for record in op_log),
               "n_deletions": len(deletions),
               "frac_del_hit_backup": (sum(record.kind == OP_DELETE_BACKUP for record in deletions)
                                       / len(deletions)) if deletions else float("nan"),
               "frac_treedel_repaired_by_backup": (sum(record.extra for record in tree_deletions)
                                                   / len(tree_deletions))
               if tree_deletions else float("nan"),
               "touched_mean": _mean_or_nan([record.n_touched for record in corrections]),
               "touched_over_size_max": float(max(
                   (record.n_touched / max(record.person_size, 1) for record in corrections),
                   default=float("nan"))),
               "msgs_mean": _mean_or_nan([record.n_messages for record in corrections]),
               **{f"corr_us_{k}": v / 1e3 for k, v in
                  _percentiles([record.latency_ns for record in corrections]).items()}}
        depths = node.engine.depths()
        row.update({"tree_depth_mean": float(np.mean(depths)), "tree_depth_max": int(np.max(depths))})
        row.update(_accuracy_metrics(node.engine.partition(), stream,
                                     ConstraintChecker(stream.topology)))
        rows.append(row)
    os.makedirs("results", exist_ok=True)
    import pandas as pd
    pd.DataFrame(op_rows).to_parquet(f"results/e3_ops_{params.get('tag', 'default')}_{seed}.parquet")
    return rows


# ============================================================================ E4
def e4_scaling(params, seed):
    """Throughput and end-to-end latency vs #sites (multi-process). kind=weak|strong.
    Real data: `streams: {n_sites: path}` gives one prepared stream per site count (the
    same scene split into that many sites); the generator is not used."""
    from ..sync.mp import run_mp
    kind = params.get("kind", "weak")
    rows = []
    real_streams = params.get("streams")        # real data: {n_sites: prepared stream path}
    for n_sites in params.get("sites", [1, 2, 4, 8]):
        if real_streams:
            path = real_streams[n_sites] if n_sites in real_streams else real_streams[str(n_sites)]
            stream = _apply_border_radius(Stream.load(path), params)
        else:
            if kind == "weak":
                config = replace(generator_config(params, seed), n_buildings=n_sites,
                                 n_sites=n_sites,
                                 n_people=params.get("people_per_site", 250) * n_sites)
            else:
                config = replace(generator_config(params, seed), n_sites=n_sites)
            stream = _load_synthetic(config, params)
            path = cache_path(config)
        for method in params.get("methods", ["two_level", "central"]):
            for load_name, speed in (("max", None), ("fixed", params.get("speed", 20.0))):
                run = run_mp(path, stream.topology, engine_config(params), mode=method, speed=speed,
                             blas_threads=params.get("blas_threads"))
                hub, sites = run["hub"], run["sites"]
                e2e_ms = hub["e2e_latency_ns"] / 1e6
                site_ms = np.concatenate([site["site_latency_ns"] for site in sites]) / 1e6
                rows.append({
                    "kind": kind, "n_sites": n_sites,
                    "method": "ours" if method == "two_level" else "B5_centralized",
                    "load": load_name, "speed": speed or 0, **_stream_info(stream),
                    "throughput_ev_s": run["throughput"], "elapsed_s": run["elapsed_s"],
                    **{f"e2e_ms_{k}": v for k, v in _percentiles(e2e_ms).items()},
                    **{f"site_ms_{k}": v for k, v in _percentiles(site_ms).items()},
                    "hub_cpu_util": hub["cpu_seconds"] / max(run["elapsed_s"], 1e-9),
                    "hub_rss_mb": hub["peak_rss_bytes"] / 2**20,
                    "site_rss_mb_max": max(site["peak_rss_bytes"] for site in sites) / 2**20,
                    "hub_msgs": hub["n_messages"],
                    "split_requests": hub.get("n_split_requests", 0),
                    "sr_bytes": hub.get("split_request_bytes", 0),
                })
    return rows


# ============================================================================ E5
def e5_bytes(params, seed):
    stream = load_stream(params, seed)
    config = engine_config(params)
    rows = []
    two_level = run_two_level(stream, config)
    info = _stream_info(stream)
    rows.append({"method": "ours", **info, "total_bytes": sum(two_level["bytes"].values()),
                 "staleness_s": 0.0, **{f"bytes_{k}": v for k, v in two_level["bytes"].items()}})
    centralized = run_centralized_hub(stream, config)
    rows.append({"method": "B5_centralized", **info, "total_bytes": sum(centralized["bytes"].values()),
                 "staleness_s": 0.0, **{f"bytes_{k}": v for k, v in centralized["bytes"].items()}})
    for period in params.get("periods", [1, 5, 30]):
        periodic = b6_periodic_sync_bytes(stream, config, float(period))
        rows.append({"method": f"B6_periodic_T{period}", **info,
                     "total_bytes": sum(periodic["bytes"].values()),
                     "staleness_s": periodic["staleness_mean"],
                     **{f"bytes_{k}": v for k, v in periodic["bytes"].items()}})
    return rows


# ============================================================================ E6
def e6_noise(params, seed):
    rows = []
    for sigma in params.get("sigmas", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        sweep_params = dict(params)
        sweep_params["gen"] = dict(params.get("gen", {}), sigma=sigma)
        stream = _load_synthetic(generator_config(sweep_params, seed), sweep_params)
        config = engine_config(sweep_params)
        checker = ConstraintChecker(stream.topology)
        duration_min = stream.metadata["gen"]["duration"] / 60.0
        node, _ = run_single(stream, config)
        engine = node.engine
        corrections = [record for record in engine.op_log if record.kind in CORRECTION_OP_KINDS]
        row = _accuracy_metrics(engine.partition(), stream, checker)
        rows.append({"sigma": sigma, "method": "ours", **row,
                     "corrections_per_min": len(corrections) / duration_min,
                     **{f"corr_us_{k}": v / 1e3 for k, v in
                        _percentiles([record.latency_ns for record in corrections]).items()}})
        rows.append({"sigma": sigma, "method": "B4_no_constraints",
                     **_accuracy_metrics(connected_components(engine), stream, checker)})
        no_correction_node, _ = run_single(stream, config, mode="no_correction")
        rows.append({"sigma": sigma, "method": "B3_no_correction",
                     **_accuracy_metrics(no_correction_node.engine.partition(), stream, checker)})
    return rows


# ============================================================================ E7
def e7_cross(params, seed):
    rows = []
    for p_cross in params.get("p_cross", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
        sweep_params = dict(params)
        sweep_params["gen"] = dict(params.get("gen", {}), p_cross=p_cross)
        stream = _load_synthetic(generator_config(sweep_params, seed), sweep_params)
        config = engine_config(sweep_params)
        checker = ConstraintChecker(stream.topology)
        node, _ = run_single(stream, config)
        central_partition = node.engine.partition()
        for ownership, method in (("b", "ours_two_level"), ("a", "ours_two_level_a")):
            start = time.perf_counter()
            two_level = run_two_level(stream, dict(config, ownership=ownership))
            sim_wall_s = time.perf_counter() - start
            hub_engine = two_level["hub"].engine
            hub_partition = hub_engine.partition()
            duration = stream.metadata["gen"]["duration"]
            rows.append({"p_cross": p_cross, "method": method, **_stream_info(stream),
                         **_accuracy_metrics(hub_partition, stream, checker),
                         "agree_centralized": agreement(hub_partition, central_partition),
                         "hub_msgs_per_stream_s": sum(two_level["message_counts"].values()) / duration,
                         "hub_ops_per_stream_s": len(hub_engine.op_log) / duration,
                         "hub_cross_cuts": sum(1 for record in hub_engine.op_log
                                               if record.kind == OP_CONFLICT_RESOLVED),
                         "split_requests": two_level["hub"].n_split_requests,
                         "total_bytes": sum(two_level["bytes"].values()), "sim_wall_s": sim_wall_s})
        rows.append({"p_cross": p_cross, "method": "centralized", **_stream_info(stream),
                     **_accuracy_metrics(central_partition, stream, checker),
                     "agree_centralized": 1.0})
    return rows


# ============================================================================ E8
def e8_neo4j(params, seed):
    """Persistence sink: replay the hub's BelongsTo deltas (two-level, option b) into
    Neo4j with batched UNWIND writes; throughput per batch size, compared with the rate
    the stream produces them; the final graph must equal the hub partition."""
    from ..sink.neo4j_sink import Neo4jSink
    stream = load_stream(params, seed)
    config = engine_config(params)
    deltas = []
    two_level = run_two_level(stream, config,
                              on_belongs_to=lambda event_time, new_deltas: deltas.extend(new_deltas))
    hub_partition = two_level["hub"].engine.partition()
    duration = (stream.metadata["gen"].get("duration")
                or float(stream.end_times.max() - stream.start_times.min()))
    rows = []
    uri = params.get("uri", "bolt://localhost:7687")
    auth = tuple(params.get("auth", ["neo4j", "argus-pass"]))
    for batch_size in params.get("batches", [500, 2000, 5000, 20000]):
        sink = Neo4jSink(uri, auth, batch_size=batch_size)
        with sink.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        start = time.perf_counter()
        flush_ms = []
        for batch_start in range(0, len(deltas), batch_size):
            flush_start = time.perf_counter()
            sink.push(deltas[batch_start:batch_start + batch_size])
            sink.flush()
            flush_ms.append((time.perf_counter() - flush_start) * 1e3)
        sink_wall_s = time.perf_counter() - start
        with sink.driver.session() as session:
            stored_partition = {record["tid"]: record["pid"] for record in session.run(
                "MATCH (t:Tracklet)-[:BELONGS_TO]->(p:Person) RETURN t.tid AS tid, p.pid AS pid")}
            n_relationships = session.run(
                "MATCH (:Tracklet)-[r:BELONGS_TO]->() RETURN count(r) AS c").single()["c"]
        sink.close()
        rows.append({"method": "ours_two_level", "batch": batch_size, **_stream_info(stream),
                     "n_deltas": len(deltas), "deltas_per_stream_s": len(deltas) / duration,
                     "sink_deltas_per_s": len(deltas) / sink_wall_s, "sink_wall_s": sink_wall_s,
                     "headroom": (len(deltas) / sink_wall_s) / (len(deltas) / duration),
                     **{f"flush_ms_{k}": v for k, v in _percentiles(flush_ms).items()},
                     "graph_equals_hub": bool(stored_partition == hub_partition),
                     "n_belongs_rel": int(n_relationships)})
    return rows


# ============================================================================ E9
def e9_radius(params, seed):
    """Accuracy vs bytes as the hub's cross-site matching radius grows (option b), with
    option (a) at radius 0 and the centralized instance as references."""
    base_stream = load_or_generate(generator_config(params, seed))
    config = engine_config(params)
    checker = ConstraintChecker(base_stream.topology)
    central_partition = run_single(base_stream, config)[0].engine.partition()
    rows = [{"method": "centralized", "radius": float("nan"), **_stream_info(base_stream),
             **_accuracy_metrics(central_partition, base_stream, checker), "agree_centralized": 1.0,
             "total_bytes": sum(run_centralized_hub(base_stream, config)["bytes"].values())}]
    for ownership in ("a", "b"):
        radii = params.get("radii", [0, 10, 20, 30, 45, 1e9]) if ownership == "b" else [0]
        for radius in radii:
            stream = base_stream.with_topology(base_stream.topology.with_border_radius(radius))
            two_level = run_two_level(stream, dict(config, ownership=ownership))
            hub_partition = two_level["hub"].engine.partition()
            rows.append({"method": f"ours_two_level_{ownership}", "radius": float(radius),
                         **_stream_info(stream),
                         "vector_cams": int(stream.topology.is_border_camera.sum()),
                         **_accuracy_metrics(hub_partition, stream, checker),
                         "agree_centralized": agreement(hub_partition, central_partition),
                         "split_requests": two_level["hub"].n_split_requests,
                         "total_bytes": sum(two_level["bytes"].values()),
                         **{f"bytes_{k}": v for k, v in two_level["bytes"].items()}})
    return rows


EXPERIMENTS = {"e1": e1_accuracy, "e2": e2_incremental, "e3": e3_corrections, "e4": e4_scaling,
               "e5": e5_bytes, "e6": e6_noise, "e7": e7_cross, "e8": e8_neo4j, "e9": e9_radius}
