"""Site -> hub protocol (msgpack). Byte counts are len(packed) of each message.

Two-level (ours): sites send identity *changes*, never full state.
  TR  TrackletRecord  [1, tid, cam, ts, te]             every tracklet, no vector
  BT  BorderTracklet  [2, tid, cam, ts, te, vec]        border-camera tracklets only
  BR  BorderRevise    [3, tid, vec]                     final vector of a border tracklet
  EA  EdgeAdd         [4, a, b, w]                      site added a tree match (a, b), score w
  ED  EdgeDel         [5, a, b]                         site removed a tree match (a, b)
Hub -> site:
  SR  SplitRequest    [10, a, b]                        hub rejected site match (a, b)

Ownership option (b) (default): the hub embeds site tree matches with their real scores
and resolves conflicts exactly like a single instance. When the hub cuts or rejects a
site-owned match whose endpoints end up in different hub persons, it sends SplitRequest;
the site deletes (and blocks) the match and streams the resulting deltas back. Invariant
I4 (site persons inside hub persons) holds whenever no SplitRequest is outstanding.
Option (a) (cfg ownership="a"): site matches are +inf edges; the hub can never cut them
and I4 holds by construction, but site mistakes cannot be corrected with global evidence.

Centralized baseline (B5): sites forward everything.
  FT  FullTracklet    [6, tid, cam, ts, te, vec]
  FR  FullRevise      [7, tid, vec]
Periodic sync baseline (B6): BT/BR as above plus every T seconds
  ST  SiteTable       [8, site, [[tid, site_pid], ...]]
"""
from __future__ import annotations

import msgpack
import numpy as np

TR, BT, BR, EA, ED, FT, FR, ST, END, SR = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
NAMES = {TR: "TrackletRecord", BT: "BorderTracklet", BR: "BorderRevise", EA: "EdgeAdd",
         ED: "EdgeDel", FT: "FullTracklet", FR: "FullRevise", ST: "SiteTable", END: "End",
         SR: "SplitRequest"}


def pack(msg) -> bytes:
    return msgpack.packb(msg, use_bin_type=True)


def unpack(b: bytes):
    return msgpack.unpackb(b, raw=False)


def vec_bytes(v: np.ndarray) -> bytes:
    return np.ascontiguousarray(v, dtype=np.float32).tobytes()


def bytes_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)
