# Reference (original per-camera) matcher, kept only for the parity test.
"""Candidate generation + cosine scoring over a sliding window.

For a new tracklet t at camera c, a stored tracklet u at camera c' is a candidate iff
    t.start - D[c',c] - W <= u.end <= t.start - D[c',c]   (different, non-overlapping cams)
    t.start - W <= u.end <= t.start                       (same camera: fragment)
    t.start - W <= u.end                                  (overlap-group camera)
i.e. W is the walking *slack* on top of the minimum walking time.

Each camera keeps a ring buffer (numpy arrays sorted by end time, because tracklets arrive
in end-time order): the window is two searchsorted calls and the scores are one
matrix-vector product per neighbouring camera. Vectors are dropped as soon as a
tracklet can no longer be a candidate of any future tracklet.
"""
from __future__ import annotations

import numpy as np

from argus.common.topology import Topology


class _CameraBuffer:
    __slots__ = ("end_times", "start_times", "tracklet_ids", "vectors", "head", "tail", "base_index")
    _COLUMNS = ("end_times", "start_times", "tracklet_ids", "vectors")

    def __init__(self, dim, capacity=64):
        self.end_times = np.empty(capacity)
        self.start_times = np.empty(capacity)
        self.tracklet_ids = np.empty(capacity, np.int64)
        self.vectors = np.empty((capacity, dim), np.float32)
        self.head = 0           # first live slot
        self.tail = 0           # one past the last used slot
        self.base_index = 0     # absolute index of slot 0

    def append(self, tracklet_id, start_time, end_time, vector) -> int:
        if self.tail == len(self.end_times):
            n_live = self.tail - self.head
            if self.head >= n_live:                      # compact in place
                for column in self._COLUMNS:
                    array = getattr(self, column)
                    array[:n_live] = array[self.head:self.tail]
            else:                                        # grow
                capacity = 2 * len(self.end_times)
                for column in self._COLUMNS:
                    array = getattr(self, column)
                    grown = np.empty((capacity,) + array.shape[1:], array.dtype)
                    grown[:n_live] = array[self.head:self.tail]
                    setattr(self, column, grown)
            self.base_index += self.head
            self.tail, self.head = n_live, 0
        slot = self.tail
        self.end_times[slot], self.start_times[slot], self.tracklet_ids[slot] = end_time, start_time, tracklet_id
        self.vectors[slot] = vector
        self.tail += 1
        return self.base_index + slot


class Matcher:
    def __init__(self, topology: Topology, window_slack: float, tau_keep: float, camera_ids=None,
                 cross_site_only: bool = False, max_walk_time: float = 600.0):
        self.window_slack = window_slack
        self.tau_keep = tau_keep
        camera_ids = list(range(topology.n_cameras)) if camera_ids is None else list(camera_ids)
        self.camera_ids = set(camera_ids)
        self.buffers: dict[int, _CameraBuffer] = {}
        # tracklet_id -> (camera_id, absolute slot, start_time, end_time)
        self.location: dict[int, tuple] = {}
        self.max_neighbour_walk_time = 0.0
        self.neighbours = {}     # query camera -> [(stored camera, walk time, views overlap)]
        walk_time, site_of_camera = topology.walk_time, topology.site_of_camera
        for query_camera in camera_ids:
            neighbours = []
            for stored_camera in camera_ids:
                if cross_site_only and site_of_camera[stored_camera] == site_of_camera[query_camera]:
                    continue
                if walk_time[stored_camera, query_camera] <= max_walk_time:
                    neighbours.append((stored_camera, float(walk_time[stored_camera, query_camera]),
                                       bool(topology.views_overlap(stored_camera, query_camera))))
                    self.max_neighbour_walk_time = max(self.max_neighbour_walk_time,
                                                       float(walk_time[stored_camera, query_camera]))
            self.neighbours[query_camera] = neighbours
        self.n_scored = 0

    # ------------------------------------------------------------------ lookups
    def get_vector(self, tracklet_id):
        location = self.location.get(tracklet_id)
        if location is None:
            return None
        buffer = self.buffers[location[0]]
        slot = location[1] - buffer.base_index
        return buffer.vectors[slot] if slot >= buffer.head else None

    def metadata(self, tracklet_id):
        location = self.location[tracklet_id]
        return location[0], location[2], location[3]

    # ------------------------------------------------------------------ scoring
    def _score_rows(self, buffer, row_start, row_end, vector, exclude, candidates):
        if row_end <= row_start:
            return
        scores = buffer.vectors[row_start:row_end] @ vector
        self.n_scored += row_end - row_start
        for row in np.nonzero(scores >= self.tau_keep)[0]:
            tracklet_id = int(buffer.tracklet_ids[row_start + row])
            if exclude is None or tracklet_id not in exclude:
                candidates.append((tracklet_id, float(scores[row])))

    def candidates(self, camera_id, start_time, vector, exclude=None):
        """Earlier tracklets that may be the same person. Returns [(tracklet_id, score)]
        with score >= tau_keep, sorted descending."""
        if camera_id not in self.neighbours:
            return []
        window_slack = self.window_slack
        candidates = []
        for stored_camera, walk, overlapping in self.neighbours[camera_id]:
            buffer = self.buffers.get(stored_camera)
            if buffer is None or buffer.tail == buffer.head:
                continue
            end_times = buffer.end_times[buffer.head:buffer.tail]
            if overlapping:
                first = np.searchsorted(end_times, start_time - window_slack, "left")
                last = len(end_times)
            else:
                first = np.searchsorted(end_times, start_time - walk - window_slack, "left")
                last = np.searchsorted(end_times, start_time - walk, "right")
            self._score_rows(buffer, buffer.head + first, buffer.head + last, vector, exclude,
                             candidates)
        candidates.sort(key=lambda candidate: -candidate[1])
        return candidates

    def forward_candidates(self, tracklet_id, vector, exclude=None):
        """Later tracklets whose window contains `tracklet_id` (used after a score revision)."""
        camera_id, _, end_time = self.metadata(tracklet_id)
        if camera_id not in self.neighbours:
            return []
        window_slack = self.window_slack
        candidates = []
        for stored_camera, walk, overlapping in self.neighbours[camera_id]:
            buffer = self.buffers.get(stored_camera)
            if buffer is None or buffer.tail == buffer.head:
                continue
            first_row = buffer.head + int(
                np.searchsorted(buffer.end_times[buffer.head:buffer.tail], end_time, "left"))
            if first_row >= buffer.tail:
                continue
            later_start_times = buffer.start_times[first_row:buffer.tail]
            if overlapping:
                in_window = end_time >= later_start_times - window_slack
            else:
                in_window = ((later_start_times - walk - window_slack <= end_time)
                             & (end_time <= later_start_times - walk))
            rows = np.nonzero(in_window)[0]
            if len(rows) == 0:
                continue
            scores = buffer.vectors[first_row + rows] @ vector
            self.n_scored += len(rows)
            for k in np.nonzero(scores >= self.tau_keep)[0]:
                other_id = int(buffer.tracklet_ids[first_row + rows[k]])
                if other_id != tracklet_id and (exclude is None or other_id not in exclude):
                    candidates.append((other_id, float(scores[k])))
        return candidates

    # ------------------------------------------------------------------ state
    def add(self, tracklet_id, camera_id, start_time, end_time, vector):
        if camera_id not in self.camera_ids:
            return
        buffer = self.buffers.get(camera_id)
        if buffer is None:
            buffer = self.buffers[camera_id] = _CameraBuffer(len(vector))
        self.location[tracklet_id] = (camera_id, buffer.append(tracklet_id, start_time, end_time, vector),
                                      start_time, end_time)

    def update_vector(self, tracklet_id, vector) -> bool:
        location = self.location.get(tracklet_id)
        if location is None:
            return False
        buffer = self.buffers[location[0]]
        slot = location[1] - buffer.base_index
        if slot < buffer.head:
            return False
        buffer.vectors[slot] = vector
        return True

    def prune(self, now, max_tracklet_duration):
        """Drop vectors that can no longer be candidates of any future tracklet."""
        horizon = now - max_tracklet_duration - self.max_neighbour_walk_time - self.window_slack - 1.0
        for buffer in self.buffers.values():
            if buffer.tail == buffer.head:
                continue
            n_expired = int(np.searchsorted(buffer.end_times[buffer.head:buffer.tail], horizon, "left"))
            if n_expired:
                for tracklet_id in buffer.tracklet_ids[buffer.head:buffer.head + n_expired].tolist():
                    self.location.pop(tracklet_id, None)
                buffer.head += n_expired

    def n_live(self) -> int:
        return sum(buffer.tail - buffer.head for buffer in self.buffers.values())
