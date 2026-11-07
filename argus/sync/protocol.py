"""Site -> hub protocol (msgpack). Byte counts are len(packed) of each message.

Two-level (ours): sites send identity *changes*, never full state.
  TrackletRecord  [1, tracklet_id, camera_id, start, end]          every tracklet, no vector
  BorderTracklet  [2, tracklet_id, camera_id, start, end, vector]  border-camera tracklets only
  BorderRevise    [3, tracklet_id, vector]                         final vector of a border tracklet
  EdgeAdd         [4, tracklet_a, tracklet_b, score]               site added a tree match
  EdgeDel         [5, tracklet_a, tracklet_b]                      site removed a tree match
Hub -> site:
  SplitRequest    [10, tracklet_a, tracklet_b]                     hub rejected a site match

Ownership option (b) (default): the hub embeds site tree matches with their real scores
and resolves conflicts exactly like a single instance. When the hub cuts or rejects a
site-owned match whose endpoints end up in different hub persons, it sends SplitRequest;
the site deletes (and blocks) the match and streams the resulting deltas back. Invariant
I4 (site persons inside hub persons) holds whenever no SplitRequest is outstanding.
Option (a) (config ownership="a"): site matches are +inf edges; the hub can never cut them
and I4 holds by construction, but site mistakes cannot be corrected with global evidence.

Centralized baseline (B5): sites forward everything.
  FullTracklet    [6, tracklet_id, camera_id, start, end, vector]
  FullRevise      [7, tracklet_id, vector]
Periodic sync baseline (B6): BorderTracklet/BorderRevise as above plus every T seconds
  SiteTable       [8, site_id, [[tracklet_id, site_person_id], ...]]
"""
from __future__ import annotations

import msgpack
import numpy as np

# Message type codes (the first element of every message; part of the wire format)
MSG_TRACKLET_RECORD = 1
MSG_BORDER_TRACKLET = 2
MSG_BORDER_REVISE = 3
MSG_EDGE_ADD = 4
MSG_EDGE_DELETE = 5
MSG_FULL_TRACKLET = 6
MSG_FULL_REVISE = 7
MSG_SITE_TABLE = 8
MSG_END = 9
MSG_SPLIT_REQUEST = 10

# Names used in byte/message-count reports (result columns are "bytes_<name>")
MESSAGE_TYPE_NAMES = {
    MSG_TRACKLET_RECORD: "TrackletRecord", MSG_BORDER_TRACKLET: "BorderTracklet",
    MSG_BORDER_REVISE: "BorderRevise", MSG_EDGE_ADD: "EdgeAdd", MSG_EDGE_DELETE: "EdgeDel",
    MSG_FULL_TRACKLET: "FullTracklet", MSG_FULL_REVISE: "FullRevise",
    MSG_SITE_TABLE: "SiteTable", MSG_END: "End", MSG_SPLIT_REQUEST: "SplitRequest"}


def pack(message) -> bytes:
    return msgpack.packb(message, use_bin_type=True)


def unpack(packed: bytes):
    return msgpack.unpackb(packed, raw=False)


def vector_to_bytes(vector: np.ndarray) -> bytes:
    return np.ascontiguousarray(vector, dtype=np.float32).tobytes()


def bytes_to_vector(packed: bytes) -> np.ndarray:
    return np.frombuffer(packed, dtype=np.float32)
