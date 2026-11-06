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

from . import protocol as P

_SITE_PORT0 = 0
_HUB_PORT = 1


class _RSS:
    """Samples this process's RSS every 0.5 s; keeps the peak."""

    def __init__(self):
        import psutil
        self.p = psutil.Process()
        self.peak = self.p.memory_info().rss
        self._stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self._stop:
            self.peak = max(self.peak, self.p.memory_info().rss)
            time.sleep(0.5)

    def stop(self):
        self._stop = True
        self.peak = max(self.peak, self.p.memory_info().rss)
        return self.peak


def _cpu(p):
    c = p.cpu_times()
    return c.user + c.system


# ======================================================================== site
def _site_main(sid, topo_json, cfg, mode, base_port, ready, start, q):
    import psutil
    import zmq
    from ..common.topology import Topology
    from .nodes import Site
    topo = Topology.from_json(topo_json)
    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    pull.bind(f"tcp://127.0.0.1:{base_port + 2 + sid}")
    push = ctx.socket(zmq.PUSH)
    push.connect(f"tcp://127.0.0.1:{base_port + _HUB_PORT}")
    site = Site(topo, sid, cfg) if mode == "two_level" else None
    rss = _RSS()
    proc = psutil.Process()
    ready.set()
    start.wait()
    lat = []
    nbytes = {}
    cpu0, w0 = _cpu(proc), time.perf_counter()
    unpack, pc = msgpack.unpackb, time.perf_counter_ns
    def ship(t_arr, et, msgs):
        frames = [msgpack.packb([t_arr, et])]
        for m in msgs:
            b = P.pack(m)
            nbytes[m[0]] = nbytes.get(m[0], 0) + len(b)
            frames.append(b)
        push.send_multipart(frames)

    # Player events: [etype, tid, cam, ts, te, vec]; player END: [-1].
    # Hub messages: [-2, a, b] SplitRequest; [-3] FIN (hub has everything; exit).
    while True:
        raw = pull.recv()
        t_arr = pc()
        ev = unpack(raw)
        if ev[0] == -1:
            push.send_multipart([msgpack.packb([-1, sid])])
            continue
        if ev[0] == -3:
            break
        if ev[0] == -2:                             # SplitRequest; et=2 marks the reply
            ship(t_arr, 2, site.on_split(ev[1], ev[2]))
            continue
        et, tid, cam, ts, te, vb = ev
        if site is not None:
            vec = np.frombuffer(vb, np.float32)
            msgs = site.on_tracklet(tid, cam, ts, te, vec) if et == 0 else site.on_revise(tid, cam, vec)
        else:                                       # centralized: forward everything
            msgs = [[P.FT, tid, cam, ts, te, vb]] if et == 0 else [[P.FR, tid, vb]]
        lat.append(pc() - t_arr)
        ship(t_arr, et, msgs)
    wall = time.perf_counter() - w0
    res = {"sid": sid, "site_lat_ns": np.array(lat, np.int64), "bytes": nbytes,
           "rss_peak": rss.stop(), "cpu_s": _cpu(proc) - cpu0, "wall_s": wall}
    if site is not None:
        res["oplog"] = site.node.engine.oplog
        res["partition"] = site.node.engine.partition()
    q.put(("site", res))
    time.sleep(0.2)
    push.close(linger=-1)                           # never drop the END marker
    pull.close()
    ctx.term()


# ======================================================================== hub
def _hub_main(n_sites, topo_json, cfg, mode, base_port, ready, start, q):
    import psutil
    import zmq
    from ..common.topology import Topology
    from .nodes import CentralHub, Hub
    topo = Topology.from_json(topo_json)
    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    pull.bind(f"tcp://127.0.0.1:{base_port + _HUB_PORT}")
    hub = Hub(topo, cfg) if mode == "two_level" else CentralHub(topo, cfg)
    to_site = []
    for sid in range(n_sites):
        so = ctx.socket(zmq.PUSH)
        so.connect(f"tcp://127.0.0.1:{base_port + 2 + sid}")
        to_site.append(so)
    rss = _RSS()
    proc = psutil.Process()
    ready.set()
    start.wait()
    lat = []
    ended = 0
    n_msgs = 0
    outstanding = 0                                 # SplitRequests not yet answered
    sr_bytes = 0
    cpu0, w0 = _cpu(proc), time.perf_counter()
    t_first = None
    unpack, pc = P.unpack, time.perf_counter_ns
    split = mode == "two_level"
    while ended < n_sites or outstanding > 0:
        frames = pull.recv_multipart()
        head = msgpack.unpackb(frames[0])
        if head[0] == -1:
            ended += 1
            continue
        if t_first is None:
            t_first = pc()
        if head[1] == 2:
            outstanding -= 1
        for b in frames[1:]:
            back = hub.handle(unpack(b))
            if split and back:
                for sid, sr in back:
                    sr_bytes += len(P.pack(sr))
                    to_site[sid].send(msgpack.packb([-2, sr[1], sr[2]]))
                    outstanding += 1
        n_msgs += len(frames) - 1
        if head[1] != 2:
            lat.append((pc() - head[0], head[1]))
    for so in to_site:
        so.send(msgpack.packb([-3]))
    t_end = pc()
    wall = time.perf_counter() - w0
    eng = hub.eng
    res = {"e2e_lat_ns": np.array([x[0] for x in lat], np.int64),
           "e2e_etype": np.array([x[1] for x in lat], np.int8),
           "t_end_ns": t_end, "t_first_ns": t_first, "n_msgs": n_msgs,
           "rss_peak": rss.stop(), "cpu_s": _cpu(proc) - cpu0, "wall_s": wall,
           "partition": eng.partition(), "oplog": eng.oplog,
           "n_split": getattr(hub, "n_split", 0), "sr_bytes": sr_bytes}
    q.put(("hub", res))
    for so in to_site:
        so.close(linger=-1)
    pull.close()
    ctx.term()


# ======================================================================== player
def _player_main(stream_path, topo_json, base_port, speed, ready, start, q):
    import zmq
    from ..common.topology import Topology
    from ..gen.campus import Stream
    s = Stream.load(stream_path)
    topo = Topology.from_json(topo_json)
    ctx = zmq.Context()
    socks = []
    for sid in range(topo.n_sites):
        so = ctx.socket(zmq.PUSH)
        so.connect(f"tcp://127.0.0.1:{base_port + 2 + sid}")
        socks.append(so)
    site_of = topo.site
    # pre-encode events (excluded from timing)
    enc, dest, when = [], [], []
    for i in range(len(s.time)):
        t = int(s.tid[i])
        c = int(s.cam[t])
        enc.append(msgpack.packb([int(s.etype[i]), t, c, float(s.ts[t]), float(s.te[t]),
                                  s.vec[i].tobytes()]))
        dest.append(int(site_of[c]))
        when.append(float(s.time[i]))
    t_base = when[0] if when else 0.0
    ready.set()
    start.wait()
    t0 = time.perf_counter_ns()
    w0 = time.perf_counter()
    for i in range(len(enc)):
        if speed:
            target = w0 + (when[i] - t_base) / speed
            dt = target - time.perf_counter()
            if dt > 0.002:
                time.sleep(dt - 0.001)
            while time.perf_counter() < target:
                pass
        socks[dest[i]].send(enc[i])
    for so in socks:
        so.send(msgpack.packb([-1]))
    q.put(("player", {"t0_ns": t0, "n_events": len(enc),
                      "send_s": time.perf_counter() - w0}))
    for so in socks:
        so.close(linger=-1)
    ctx.term()


def _guard(fn, q, *args):
    try:
        fn(*args)
    except BaseException:
        import traceback
        q.put(("error", traceback.format_exc()))
        raise


# ======================================================================== driver
def run_mp(stream_path: str, topo, cfg: dict, mode: str = "two_level", speed=None,
           base_port: int | None = None, timeout: float = 1200.0) -> dict:
    """mode: 'two_level' (ours) or 'central' (B5). speed=None replays as fast as possible
    (max throughput); speed=k replays k x real time (latency at a given load)."""
    ctxm = mp.get_context("spawn")
    base_port = base_port or (20000 + (os.getpid() * 7 + int(time.time())) % 20000)
    n = topo.n_sites
    tj = topo.to_json()
    q = ctxm.Queue()
    start = ctxm.Event()
    readies = [ctxm.Event() for _ in range(n + 2)]
    procs = [ctxm.Process(target=_guard, args=(_hub_main, q, n, tj, cfg, mode, base_port,
                                               readies[0], start, q))]
    for sid in range(n):
        procs.append(ctxm.Process(target=_guard, args=(_site_main, q, sid, tj, cfg, mode,
                                                       base_port, readies[1 + sid], start, q)))
    procs.append(ctxm.Process(target=_guard, args=(_player_main, q, stream_path, tj, base_port,
                                                   speed, readies[-1], start, q)))
    for p in procs:
        p.start()
    for r in readies:
        if not r.wait(timeout=300):
            for p in procs:
                p.kill()
            raise RuntimeError("process failed to become ready")
    time.sleep(0.5)                                 # let zmq connections settle
    start.set()
    out = {"sites": []}
    deadline = time.time() + timeout
    import queue as _queue
    try:
        while len(out["sites"]) < n or "hub" not in out or "player" not in out:
            try:
                kind, res = q.get(timeout=max(1.0, deadline - time.time()))
            except _queue.Empty:
                alive = [p.name for p in procs if p.is_alive()]
                raise TimeoutError(f"run_mp timed out after {timeout}s; got "
                                   f"{sorted(k for k in out if out[k])}, alive={alive}")
            if kind == "error":
                raise RuntimeError("child process failed: " + res)
            if kind == "site":
                out["sites"].append(res)
            else:
                out[kind] = res
        for p in procs:
            p.join(timeout=30)
    finally:
        for p in procs:
            if p.is_alive():
                p.kill()
    hub, pl = out["hub"], out["player"]
    out["elapsed_s"] = (hub["t_end_ns"] - pl["t0_ns"]) / 1e9
    out["throughput"] = pl["n_events"] / out["elapsed_s"]
    return out
