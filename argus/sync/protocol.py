"""Site -> hub protocol (msgpack). Byte counts are len(packed) of each message.

Two-level (ours): sites send identity *changes*, never full state.
  TR  TrackletRecord  [1, tid, cam, ts, te]             every tracklet, no vector
  BT  BorderTracklet  [2, tid, cam, ts, te, vec]        border-camera tracklets only
  BR  BorderRevise    [3, tid, vec]                     final vector of a border tracklet
  EA  EdgeAdd         [4, a, b]                         site added a tree match (a, b)
  ED  EdgeDel         [5, a, b]                         site removed a tree match (a, b)
The hub embeds every site tree match as an edge of weight +inf: the site owns it, the hub
can never cut it (ownership rule, option (a)), and site persons are always contained in
hub (global) persons (invariant I4) by construction.

Centralized baseline (B5): sites forward everything.
  FT  FullTracklet    [6, tid, cam, ts, te, vec]
  FR  FullRevise      [7, tid, vec]
Periodic sync baseline (B6): BT/BR as above plus every T seconds
  ST  SiteTable       [8, site, [[tid, site_pid], ...]]
"""
from __future__ import annotations

import msgpack
import numpy as np

TR, BT, BR, EA, ED, FT, FR, ST, END = 1, 2, 3, 4, 5, 6, 7, 8, 9
NAMES = {TR: "TrackletRecord", BT: "BorderTracklet", BR: "BorderRevise", EA: "EdgeAdd",
         ED: "EdgeDel", FT: "FullTracklet", FR: "FullRevise", ST: "SiteTable", END: "End"}


def pack(msg) -> bytes:
    return msgpack.packb(msg, use_bin_type=True)


def unpack(b: bytes):
    return msgpack.unpackb(b, raw=False)


def vec_bytes(v: np.ndarray) -> bytes:
    return np.ascontiguousarray(v, dtype=np.float32).tobytes()


def bytes_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)
