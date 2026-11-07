"""Deterministic in-process drivers (single instance, two-level, byte accounting).
Used for accuracy / agreement / byte experiments and tests. The multi-process runtime
(mp.py) is used for throughput and latency."""
from __future__ import annotations

import time
from collections import defaultdict, deque

import numpy as np

from ..common.types import DELTA_BELONGS_TO, EVENT_TRACKLET, NO_PARENT
from ..core.node import IdentityNode
from . import protocol
from .nodes import CentralHub, Hub, Site, forward_messages


def _iterate_events(stream):
    """Yields python-typed events:
    (event_type, tracklet_id, camera_id, start_time, end_time, vector, event_time)."""
    camera_ids, start_times, end_times = stream.camera_ids, stream.start_times, stream.end_times
    for i in range(len(stream.event_times)):
        tracklet_id = int(stream.event_tracklet_ids[i])
        yield (int(stream.event_types[i]), tracklet_id, int(camera_ids[tracklet_id]),
               float(start_times[tracklet_id]), float(end_times[tracklet_id]),
               stream.event_vectors[i], float(stream.event_times[i]))


def _by_type_name(counts):
    return {protocol.MESSAGE_TYPE_NAMES[message_type]: value
            for message_type, value in counts.items()}


def run_single(stream, engine_config: dict, mode: str = "full",
               operator_corrections_per_minute: float = 0.0, seed: int = 0, checkpoints=(),
               on_checkpoint=None, time_events: bool = False):
    """One central instance over all cameras. operator_corrections_per_minute = simulated
    operator corrections (each deletes one currently-accepted wrong match, drawn using GT).
    Returns (node, latencies); latencies = [(event_type, ns, n_tracklets_so_far)] if
    time_events else None."""
    node = IdentityNode(stream.topology, engine_config, mode=mode)
    engine = node.engine
    gt_person_ids = stream.gt_person_ids
    rng = np.random.default_rng(seed + 991)
    correction_interval = (60.0 / operator_corrections_per_minute
                           if operator_corrections_per_minute > 0 else float("inf"))
    next_correction_time = correction_interval
    pending_checkpoints = sorted(checkpoints)
    latencies = [] if time_events else None
    n_tracklets = 0
    clock_ns = time.perf_counter_ns
    for event_type, tracklet_id, camera_id, start_time, end_time, vector, event_time in \
            _iterate_events(stream):
        event_start_ns = clock_ns() if time_events else 0
        if event_type == EVENT_TRACKLET:
            node.on_tracklet(tracklet_id, camera_id, start_time, end_time, vector)
            n_tracklets += 1
        else:
            node.on_revise(tracklet_id, vector)
        if time_events:
            latencies.append((event_type, clock_ns() - event_start_ns, n_tracklets))
        while event_time >= next_correction_time:
            _operator_correction(engine, gt_person_ids, rng)
            next_correction_time += correction_interval
        if (pending_checkpoints and event_type == EVENT_TRACKLET
                and n_tracklets >= pending_checkpoints[0]):
            pending_checkpoints.pop(0)
            if on_checkpoint:
                on_checkpoint(n_tracklets, node)
    return node, latencies


def _operator_correction(engine, gt_person_ids, rng):
    """Delete (and block) one random accepted match that joins two different GT people."""
    wrong_matches = []
    for tracklet_id, vertex in engine.vertices.items():
        if (vertex.parent_id != NO_PARENT
                and gt_person_ids[tracklet_id] != gt_person_ids[vertex.parent_id]):
            wrong_matches.append((tracklet_id, vertex.parent_id))
        if vertex.backup_links:
            for partner_id in vertex.backup_links:
                if tracklet_id < partner_id and gt_person_ids[tracklet_id] != gt_person_ids[partner_id]:
                    wrong_matches.append((tracklet_id, partner_id))
    if wrong_matches:
        tracklet_a, tracklet_b = wrong_matches[int(rng.integers(len(wrong_matches)))]
        engine.remove_match(tracklet_a, tracklet_b, block=True)


def run_two_level(stream, engine_config: dict, topology=None, mode: str = "full",
                  on_belongs_to=None):
    """N sites + hub, messages delivered immediately in stream order (deterministic).
    Returns dict with hub, sites, bytes per message type, message counts per type."""
    topology = topology or stream.topology
    sites = [Site(topology, site_id, engine_config, mode=mode) for site_id in range(topology.n_sites)]
    hub = Hub(topology, engine_config, emit_belongs_to=on_belongs_to is not None)
    bytes_by_type = defaultdict(int)
    count_by_type = defaultdict(int)
    site_of_camera = topology.site_of_camera
    for event_type, tracklet_id, camera_id, start_time, end_time, vector, event_time in \
            _iterate_events(stream):
        site = sites[site_of_camera[camera_id]]
        if event_type == EVENT_TRACKLET:
            messages = site.on_tracklet(tracklet_id, camera_id, start_time, end_time, vector)
        else:
            messages = site.on_revise(tracklet_id, camera_id, vector)
        work = deque(messages)
        while work:                      # site deltas; SplitRequest replies go to the front
            message = work.popleft()
            bytes_by_type[message[0]] += len(protocol.pack(message))
            count_by_type[message[0]] += 1
            replies = []
            for site_id, split_request in hub.handle(message):
                bytes_by_type[split_request[0]] += len(protocol.pack(split_request))
                count_by_type[split_request[0]] += 1
                replies.extend(sites[site_id].on_split_request(split_request[1], split_request[2]))
            work.extendleft(reversed(replies))
        if on_belongs_to is not None and hub.engine.emitted_deltas:   # BelongsTo deltas for a sink
            on_belongs_to(event_time, [delta for delta in hub.engine.emitted_deltas
                                       if delta[0] == DELTA_BELONGS_TO])
            hub.engine.emitted_deltas = []
    return {"hub": hub, "sites": sites, "bytes": _by_type_name(bytes_by_type),
            "message_counts": _by_type_name(count_by_type)}


def run_centralized_hub(stream, engine_config: dict, topology=None):
    """B5 in-process: everything forwarded with vectors."""
    topology = topology or stream.topology
    hub = CentralHub(topology, engine_config)
    bytes_by_type = defaultdict(int)
    for event_type, tracklet_id, camera_id, start_time, end_time, vector, _ in _iterate_events(stream):
        for message in forward_messages(event_type, tracklet_id, camera_id, start_time, end_time, vector):
            bytes_by_type[message[0]] += len(protocol.pack(message))
            hub.handle(message)
    return {"hub": hub, "bytes": _by_type_name(bytes_by_type)}


def b6_periodic_sync_bytes(stream, engine_config: dict, period: float, topology=None):
    """B6 periodic full-state sync: border tracklets/vectors as in ours plus every
    `period` seconds each site sends its whole person table. Returns bytes by type and
    staleness (seconds until the hub learns about each tracklet's identity)."""
    topology = topology or stream.topology
    sites = [Site(topology, site_id, engine_config) for site_id in range(topology.n_sites)]
    bytes_by_type = defaultdict(int)
    staleness = []
    next_sync_time = period
    site_of_camera = topology.site_of_camera
    for event_type, tracklet_id, camera_id, start_time, end_time, vector, event_time in \
            _iterate_events(stream):
        while event_time >= next_sync_time:
            for site in sites:
                bytes_by_type[protocol.MSG_SITE_TABLE] += len(protocol.pack(
                    [protocol.MSG_SITE_TABLE, site.site_id, site.person_table()]))
            next_sync_time += period
        site = sites[site_of_camera[camera_id]]
        if event_type == EVENT_TRACKLET:
            messages = site.on_tracklet(tracklet_id, camera_id, start_time, end_time, vector)
        else:
            messages = site.on_revise(tracklet_id, camera_id, vector)
        for message in messages:
            if message[0] in (protocol.MSG_BORDER_TRACKLET, protocol.MSG_BORDER_REVISE):
                bytes_by_type[message[0]] += len(protocol.pack(message))
        if event_type == EVENT_TRACKLET:
            staleness.append(next_sync_time - event_time)
    for site in sites:
        bytes_by_type[protocol.MSG_SITE_TABLE] += len(protocol.pack(
            [protocol.MSG_SITE_TABLE, site.site_id, site.person_table()]))
    return {"bytes": _by_type_name(bytes_by_type),
            "staleness_mean": float(np.mean(staleness)) if staleness else 0.0}
