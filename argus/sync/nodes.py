"""Site and hub logic, independent of transport (used in-process and by mp.py)."""
from __future__ import annotations

import numpy as np

from ..common.topology import Topology
from ..common.types import (DELTA_BELONGS_TO, DELTA_TREE_EDGE_REMOVED, EVENT_TRACKLET, INF)
from ..core.node import IdentityNode
from . import protocol


class Site:
    def __init__(self, topology: Topology, site_id: int, engine_config: dict, mode: str = "full"):
        self.site_id = site_id
        camera_ids = np.nonzero(topology.site_of_camera == site_id)[0].tolist()
        self.node = IdentityNode(topology, engine_config, match_camera_ids=camera_ids,
                                 emit_tree_deltas=True, mode=mode)
        self.is_border_camera = topology.is_border_camera

    def _drain_deltas(self, messages):
        """Turn the engine's emitted tree deltas into EdgeAdd / EdgeDel messages."""
        engine = self.node.engine
        for delta in engine.emitted_deltas:
            if delta[0] == DELTA_TREE_EDGE_REMOVED:
                messages.append([protocol.MSG_EDGE_DELETE, delta[1], delta[2]])
            else:                          # new tree match, or a tree match's new score
                messages.append([protocol.MSG_EDGE_ADD, delta[1], delta[2], float(delta[3])])
        engine.emitted_deltas = []
        return messages

    def on_tracklet(self, tracklet_id, camera_id, start_time, end_time, vector):
        self.node.on_tracklet(tracklet_id, camera_id, start_time, end_time, vector)
        if self.is_border_camera[camera_id]:
            messages = [[protocol.MSG_BORDER_TRACKLET, tracklet_id, camera_id, start_time,
                         end_time, protocol.vector_to_bytes(vector)]]
        else:
            messages = [[protocol.MSG_TRACKLET_RECORD, tracklet_id, camera_id, start_time, end_time]]
        return self._drain_deltas(messages)

    def on_revise(self, tracklet_id, camera_id, vector):
        self.node.on_revise(tracklet_id, vector)
        messages = self._drain_deltas([])
        if self.is_border_camera[camera_id]:
            messages.append([protocol.MSG_BORDER_REVISE, tracklet_id, protocol.vector_to_bytes(vector)])
        return messages

    def on_split_request(self, tracklet_a, tracklet_b):
        """Hub SplitRequest: delete and block the match; return the resulting deltas."""
        self.node.engine.remove_match(tracklet_a, tracklet_b, block=True)
        return self._drain_deltas([])

    def person_table(self):
        """Full person table (B6)."""
        return [[tracklet_id, vertex.person_id]
                for tracklet_id, vertex in self.node.engine.vertices.items()]


class Hub:
    """Two-level hub: same engine; vertices = all tracklets (metadata), cross-site
    matching over border tracklets only, plus every site tree match.

    ownership="b" (default): site matches carry their real score; the hub resolves
    conflicts like a single instance and returns SplitRequests for site matches it rejects.
    ownership="a": site matches are +inf edges (never cut by the hub)."""

    def __init__(self, topology: Topology, engine_config: dict, emit_belongs_to: bool = False):
        border_camera_ids = np.nonzero(topology.is_border_camera)[0].tolist()
        self.ownership = engine_config.get("ownership", "b")
        self.node = IdentityNode(topology, engine_config, match_camera_ids=border_camera_ids,
                                 cross_site_only=True, emit_tree_deltas=(self.ownership == "b"),
                                 emit_belongs_to=emit_belongs_to)
        self.engine = self.node.engine
        self.site_of_camera = topology.site_of_camera
        self.site_owned_matches = {}     # (min, max) tracklet pair -> owning site_id
        self.n_split_requests = 0
        if self.ownership == "b":
            self.node.is_site_owned_match = (
                lambda tracklet_a, tracklet_b:
                self._pair_key(tracklet_a, tracklet_b) in self.site_owned_matches)

    @staticmethod
    def _pair_key(tracklet_a, tracklet_b):
        return (tracklet_a, tracklet_b) if tracklet_a < tracklet_b else (tracklet_b, tracklet_a)

    def handle(self, message):
        """Process one site message. Returns [(site_id, SplitRequest), ...] to send back."""
        message_type = message[0]
        engine = self.engine
        if message_type == protocol.MSG_EDGE_ADD:
            tracklet_a, tracklet_b = message[1], message[2]
            if self.ownership == "a":
                engine.add_match(tracklet_a, tracklet_b, INF)
                return []
            key = self._pair_key(tracklet_a, tracklet_b)
            owning_site = int(self.site_of_camera[engine.vertices[tracklet_a].camera_id])
            self.site_owned_matches[key] = owning_site
            engine.add_match(tracklet_a, tracklet_b, message[3] if len(message) > 3 else INF)
            return self._collect_split_requests(new_site_match=key)
        if message_type == protocol.MSG_EDGE_DELETE:
            self.site_owned_matches.pop(self._pair_key(message[1], message[2]), None)
            engine.remove_match(message[1], message[2])
        elif message_type == protocol.MSG_TRACKLET_RECORD:
            # site_tag = the tracklet's site: enables the site-pure fast path
            _, tracklet_id, camera_id, start_time, end_time = message
            engine.add_vertex(tracklet_id, camera_id, start_time, end_time,
                              int(self.site_of_camera[camera_id]))
        elif message_type == protocol.MSG_BORDER_TRACKLET:
            _, tracklet_id, camera_id, start_time, end_time, vector_bytes = message
            self.node.on_tracklet(tracklet_id, camera_id, start_time, end_time,
                                  protocol.bytes_to_vector(vector_bytes),
                                  site_tag=int(self.site_of_camera[camera_id]))
        elif message_type == protocol.MSG_BORDER_REVISE:
            self.node.on_revise(message[1], protocol.bytes_to_vector(message[2]))
        return self._collect_split_requests() if self.ownership == "b" else []

    def _collect_split_requests(self, new_site_match=None):
        """SplitRequests for site matches the hub rejected: every site match the hub cut
        (its own decision; EdgeDel-driven cuts are no longer owned), and a new site match
        the hub could not accept into one person."""
        engine = self.engine
        vertices = engine.vertices
        rejected = [self._pair_key(delta[1], delta[2]) for delta in engine.emitted_deltas
                    if delta[0] == DELTA_TREE_EDGE_REMOVED]
        # keep BelongsTo deltas for a sink
        engine.emitted_deltas = [delta for delta in engine.emitted_deltas
                                 if delta[0] == DELTA_BELONGS_TO]
        if (new_site_match is not None and vertices[new_site_match[0]].person_id
                != vertices[new_site_match[1]].person_id):
            rejected.append(new_site_match)
        split_requests = []
        for key in rejected:
            site_id = self.site_owned_matches.pop(key, None)
            if site_id is None:
                continue
            engine.remove_match(key[0], key[1], block=True)   # not accepted here: no-op on the forest
            split_requests.append((site_id, [protocol.MSG_SPLIT_REQUEST, key[0], key[1]]))
            self.n_split_requests += 1
        return split_requests


class CentralHub:
    """B5: one full instance at the hub; sites only forward."""

    def __init__(self, topology: Topology, engine_config: dict, mode: str = "full"):
        self.node = IdentityNode(topology, engine_config, mode=mode)
        self.engine = self.node.engine

    def handle(self, message):
        message_type = message[0]
        if message_type == protocol.MSG_FULL_TRACKLET:
            _, tracklet_id, camera_id, start_time, end_time, vector_bytes = message
            self.node.on_tracklet(tracklet_id, camera_id, start_time, end_time,
                                  protocol.bytes_to_vector(vector_bytes))
        elif message_type == protocol.MSG_FULL_REVISE:
            self.node.on_revise(message[1], protocol.bytes_to_vector(message[2]))


def forward_messages(event_type, tracklet_id, camera_id, start_time, end_time, vector):
    """B5: what a site forwards for one event."""
    if event_type == EVENT_TRACKLET:
        return [[protocol.MSG_FULL_TRACKLET, tracklet_id, camera_id, start_time, end_time,
                 protocol.vector_to_bytes(vector)]]
    return [[protocol.MSG_FULL_REVISE, tracklet_id, protocol.vector_to_bytes(vector)]]
