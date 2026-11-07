"""Multi-process runtime: player -> N site processes -> hub process (pyzmq PUSH/PULL).

One OS process per site and one for the hub (spawn start method; no GIL sharing).
Per event the site sends one multipart message: [instrumentation, msg1, msg2, ...].
The instrumentation frame (arrival timestamp at the site) is excluded from byte counts.
Timestamps use time.perf_counter_ns(), which is one system-wide clock on one machine.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time

import msgpack
import numpy as np

from ..common.types import EVENT_TRACKLET
from . import protocol

_HUB_PORT_OFFSET = 1          # hub PULL socket: base_port + 1
_SITE_PORT_OFFSET = 2         # site k PULL socket: base_port + 2 + k

# Control messages on the player/hub -> site sockets and the site -> hub socket
_PLAYER_END = -1              # player -> site: no more events; site -> hub: [-1, site_id]
_HUB_SPLIT_REQUEST = -2       # hub -> site: [-2, tracklet_a, tracklet_b]
_HUB_FINISH = -3              # hub -> site: hub has everything; exit
_SPLIT_REPLY_EVENT = 2        # event type in the instrumentation frame of a SplitRequest reply

_BLAS_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _address(port):
    return f"tcp://127.0.0.1:{port}"


class _PeakRssSampler:
    """Samples this process's RSS every 0.5 s; keeps the peak."""

    def __init__(self):
        import psutil
        self.process = psutil.Process()
        self.peak_bytes = self.process.memory_info().rss
        self._stopped = False
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def _sample_loop(self):
        while not self._stopped:
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
            time.sleep(0.5)

    def stop(self):
        self._stopped = True
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
        return self.peak_bytes


def _cpu_seconds(process):
    cpu_times = process.cpu_times()
    return cpu_times.user + cpu_times.system


# ======================================================================== site
def _site_main(site_id, topology_json, engine_config, mode, base_port, ready_event,
               start_event, result_queue):
    import psutil
    import zmq
    from ..common.topology import Topology
    from .nodes import Site
    topology = Topology.from_json(topology_json)
    zmq_context = zmq.Context()
    pull_socket = zmq_context.socket(zmq.PULL)
    pull_socket.bind(_address(base_port + _SITE_PORT_OFFSET + site_id))
    push_socket = zmq_context.socket(zmq.PUSH)
    push_socket.connect(_address(base_port + _HUB_PORT_OFFSET))
    site = Site(topology, site_id, engine_config) if mode == "two_level" else None
    rss_sampler = _PeakRssSampler()
    process = psutil.Process()
    ready_event.set()
    start_event.wait()
    latencies_ns = []
    bytes_by_type = {}
    cpu_start, wall_start = _cpu_seconds(process), time.perf_counter()
    unpack, clock_ns = msgpack.unpackb, time.perf_counter_ns

    def send_to_hub(arrival_ns, event_type, messages):
        frames = [msgpack.packb([arrival_ns, event_type])]
        for message in messages:
            packed = protocol.pack(message)
            bytes_by_type[message[0]] = bytes_by_type.get(message[0], 0) + len(packed)
            frames.append(packed)
        push_socket.send_multipart(frames)

    # Player events: [event_type, tracklet_id, camera_id, start, end, vector_bytes].
    while True:
        raw = pull_socket.recv()
        arrival_ns = clock_ns()
        event = unpack(raw)
        if event[0] == _PLAYER_END:
            push_socket.send_multipart([msgpack.packb([_PLAYER_END, site_id])])
            continue
        if event[0] == _HUB_FINISH:
            break
        if event[0] == _HUB_SPLIT_REQUEST:
            send_to_hub(arrival_ns, _SPLIT_REPLY_EVENT, site.on_split_request(event[1], event[2]))
            continue
        event_type, tracklet_id, camera_id, start_time, end_time, vector_bytes = event
        if site is not None:
            vector = np.frombuffer(vector_bytes, np.float32)
            if event_type == EVENT_TRACKLET:
                messages = site.on_tracklet(tracklet_id, camera_id, start_time, end_time, vector)
            else:
                messages = site.on_revise(tracklet_id, camera_id, vector)
        elif event_type == EVENT_TRACKLET:           # centralized: forward everything
            messages = [[protocol.MSG_FULL_TRACKLET, tracklet_id, camera_id, start_time,
                         end_time, vector_bytes]]
        else:
            messages = [[protocol.MSG_FULL_REVISE, tracklet_id, vector_bytes]]
        latencies_ns.append(clock_ns() - arrival_ns)
        send_to_hub(arrival_ns, event_type, messages)
    wall_seconds = time.perf_counter() - wall_start
    result = {"site_id": site_id, "site_latency_ns": np.array(latencies_ns, np.int64),
              "bytes": bytes_by_type, "peak_rss_bytes": rss_sampler.stop(),
              "cpu_seconds": _cpu_seconds(process) - cpu_start, "wall_seconds": wall_seconds}
    if site is not None:
        result["op_log"] = site.node.engine.op_log
        result["partition"] = site.node.engine.partition()
    result_queue.put(("site", result))
    time.sleep(0.2)
    push_socket.close(linger=-1)                    # never drop the END marker
    pull_socket.close()
    zmq_context.term()


# ======================================================================== hub
def _hub_main(n_sites, topology_json, engine_config, mode, base_port, ready_event,
              start_event, result_queue):
    import psutil
    import zmq
    from ..common.topology import Topology
    from .nodes import CentralHub, Hub
    topology = Topology.from_json(topology_json)
    zmq_context = zmq.Context()
    pull_socket = zmq_context.socket(zmq.PULL)
    pull_socket.bind(_address(base_port + _HUB_PORT_OFFSET))
    hub = Hub(topology, engine_config) if mode == "two_level" else CentralHub(topology, engine_config)
    site_sockets = []
    for site_id in range(n_sites):
        site_socket = zmq_context.socket(zmq.PUSH)
        site_socket.connect(_address(base_port + _SITE_PORT_OFFSET + site_id))
        site_sockets.append(site_socket)
    rss_sampler = _PeakRssSampler()
    process = psutil.Process()
    ready_event.set()
    start_event.wait()
    latencies = []                                  # (end-to-end ns, event_type)
    n_sites_ended = 0
    n_messages = 0
    outstanding_split_requests = 0                  # SplitRequests not yet answered
    split_request_bytes = 0
    cpu_start, wall_start = _cpu_seconds(process), time.perf_counter()
    first_message_ns = None
    unpack, clock_ns = protocol.unpack, time.perf_counter_ns
    sends_split_requests = mode == "two_level"
    while n_sites_ended < n_sites or outstanding_split_requests > 0:
        frames = pull_socket.recv_multipart()
        arrival_ns, event_type = msgpack.unpackb(frames[0])
        if arrival_ns == _PLAYER_END:
            n_sites_ended += 1
            continue
        if first_message_ns is None:
            first_message_ns = clock_ns()
        if event_type == _SPLIT_REPLY_EVENT:
            outstanding_split_requests -= 1
        for packed in frames[1:]:
            split_requests = hub.handle(unpack(packed))
            if sends_split_requests and split_requests:
                for site_id, split_request in split_requests:
                    split_request_bytes += len(protocol.pack(split_request))
                    site_sockets[site_id].send(msgpack.packb(
                        [_HUB_SPLIT_REQUEST, split_request[1], split_request[2]]))
                    outstanding_split_requests += 1
        n_messages += len(frames) - 1
        if event_type != _SPLIT_REPLY_EVENT:
            latencies.append((clock_ns() - arrival_ns, event_type))
    for site_socket in site_sockets:
        site_socket.send(msgpack.packb([_HUB_FINISH]))
    end_ns = clock_ns()
    wall_seconds = time.perf_counter() - wall_start
    engine = hub.engine
    result = {"e2e_latency_ns": np.array([latency for latency, _ in latencies], np.int64),
              "e2e_event_type": np.array([event_type for _, event_type in latencies], np.int8),
              "end_ns": end_ns, "first_message_ns": first_message_ns, "n_messages": n_messages,
              "peak_rss_bytes": rss_sampler.stop(), "cpu_seconds": _cpu_seconds(process) - cpu_start,
              "wall_seconds": wall_seconds, "partition": engine.partition(), "op_log": engine.op_log,
              "n_split_requests": getattr(hub, "n_split_requests", 0),
              "split_request_bytes": split_request_bytes}
    result_queue.put(("hub", result))
    for site_socket in site_sockets:
        site_socket.close(linger=-1)
    pull_socket.close()
    zmq_context.term()


# ======================================================================== player
def _player_main(stream_path, topology_json, base_port, speed, ready_event, start_event,
                 result_queue):
    import zmq
    from ..common.topology import Topology
    from ..gen.campus import Stream
    stream = Stream.load(stream_path)
    topology = Topology.from_json(topology_json)
    zmq_context = zmq.Context()
    site_sockets = []
    for site_id in range(topology.n_sites):
        site_socket = zmq_context.socket(zmq.PUSH)
        site_socket.connect(_address(base_port + _SITE_PORT_OFFSET + site_id))
        site_sockets.append(site_socket)
    site_of_camera = topology.site_of_camera
    # pre-encode events (excluded from timing)
    encoded_events, destination_sites, event_times = [], [], []
    for i in range(len(stream.event_times)):
        tracklet_id = int(stream.event_tracklet_ids[i])
        camera_id = int(stream.camera_ids[tracklet_id])
        encoded_events.append(msgpack.packb([
            int(stream.event_types[i]), tracklet_id, camera_id,
            float(stream.start_times[tracklet_id]), float(stream.end_times[tracklet_id]),
            stream.event_vectors[i].tobytes()]))
        destination_sites.append(int(site_of_camera[camera_id]))
        event_times.append(float(stream.event_times[i]))
    first_event_time = event_times[0] if event_times else 0.0
    ready_event.set()
    start_event.wait()
    start_ns = time.perf_counter_ns()
    wall_start = time.perf_counter()
    for i in range(len(encoded_events)):
        if speed:
            target = wall_start + (event_times[i] - first_event_time) / speed
            remaining = target - time.perf_counter()
            if remaining > 0.002:
                time.sleep(remaining - 0.001)
            while time.perf_counter() < target:      # spin for the last millisecond
                pass
        site_sockets[destination_sites[i]].send(encoded_events[i])
    for site_socket in site_sockets:
        site_socket.send(msgpack.packb([_PLAYER_END]))
    result_queue.put(("player", {"start_ns": start_ns, "n_events": len(encoded_events),
                                 "send_seconds": time.perf_counter() - wall_start}))
    for site_socket in site_sockets:
        site_socket.close(linger=-1)
    zmq_context.term()


def _report_errors(target, result_queue, *args):
    try:
        target(*args)
    except BaseException:
        import traceback
        result_queue.put(("error", traceback.format_exc()))
        raise


# ======================================================================== driver
def run_mp(stream_path: str, topology, engine_config: dict, mode: str = "two_level", speed=None,
           base_port: int | None = None, timeout: float = 1200.0,
           blas_threads: int | None = None) -> dict:
    """mode: 'two_level' (ours) or 'central' (B5). speed=None replays as fast as possible
    (max throughput); speed=k replays k x real time (latency at a given load).
    blas_threads=k caps the BLAS/OpenMP threads of every child process (the children are
    spawned, so they read the environment when they import numpy). On a machine with few
    cores, uncapped BLAS pools of the hub, every site and the player oversubscribe the CPU.
    None leaves the library defaults (as in the synthetic runs)."""
    mp_context = mp.get_context("spawn")
    base_port = base_port or (20000 + (os.getpid() * 7 + int(time.time())) % 20000)
    n_sites = topology.n_sites
    topology_json = topology.to_json()
    result_queue = mp_context.Queue()
    start_event = mp_context.Event()
    ready_events = [mp_context.Event() for _ in range(n_sites + 2)]   # hub, sites..., player
    processes = [mp_context.Process(target=_report_errors, args=(
        _hub_main, result_queue, n_sites, topology_json, engine_config, mode, base_port,
        ready_events[0], start_event, result_queue))]
    for site_id in range(n_sites):
        processes.append(mp_context.Process(target=_report_errors, args=(
            _site_main, result_queue, site_id, topology_json, engine_config, mode, base_port,
            ready_events[1 + site_id], start_event, result_queue)))
    processes.append(mp_context.Process(target=_report_errors, args=(
        _player_main, result_queue, stream_path, topology_json, base_port, speed,
        ready_events[-1], start_event, result_queue)))
    saved_env = {name: os.environ.get(name) for name in _BLAS_THREAD_ENV_VARS}
    if blas_threads:
        for name in _BLAS_THREAD_ENV_VARS:
            os.environ[name] = str(int(blas_threads))
    try:
        for process in processes:
            process.start()
    finally:
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    for ready_event in ready_events:
        if not ready_event.wait(timeout=300):
            for process in processes:
                process.kill()
            raise RuntimeError("process failed to become ready")
    time.sleep(0.5)                                 # let zmq connections settle
    start_event.set()
    results = {"sites": []}
    deadline = time.time() + timeout
    import queue
    try:
        while len(results["sites"]) < n_sites or "hub" not in results or "player" not in results:
            try:
                kind, result = result_queue.get(timeout=max(1.0, deadline - time.time()))
            except queue.Empty:
                alive = [process.name for process in processes if process.is_alive()]
                raise TimeoutError(f"run_mp timed out after {timeout}s; got "
                                   f"{sorted(k for k in results if results[k])}, alive={alive}")
            if kind == "error":
                raise RuntimeError("child process failed: " + result)
            if kind == "site":
                results["sites"].append(result)
            else:
                results[kind] = result
        for process in processes:
            process.join(timeout=30)
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
    hub_result, player_result = results["hub"], results["player"]
    results["elapsed_s"] = (hub_result["end_ns"] - player_result["start_ns"]) / 1e9
    results["throughput"] = player_result["n_events"] / results["elapsed_s"]
    return results
