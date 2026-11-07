"""Candidate generation + cosine scoring over a sliding window.

For a new tracklet t at camera c, a stored tracklet u at camera c' is a candidate iff
    t.start - D[c',c] - W <= u.end <= t.start - D[c',c]   (different, non-overlapping cams)
    t.start - W <= u.end <= t.start                       (same camera: fragment)
    t.start - W <= u.end                                  (overlap-group camera)
where D is the walking time and W the window slack, i.e. W is the walking *slack* on top
of the minimum walking time.

Stored tracklets live in one ring buffer per *site* (numpy arrays sorted by end time:
tracklets of one site arrive in end-time order, also at the multi-process hub where sites
interleave). A query is one searchsorted window per buffer, one vectorised mask with
per-camera bounds, and one matrix-vector product over the surviving rows. Vectors are
dropped as soon as a tracklet can no longer be a candidate of any future tracklet.
"""
from __future__ import annotations

import numpy as np

from ..common.topology import Topology


class _SiteBuffer:
    """Ring buffer of one site's stored tracklets, in end-time order."""
    __slots__ = ("end_times", "start_times", "tracklet_ids", "camera_ids", "vectors",
                 "head", "tail", "base_index")
    _COLUMNS = ("end_times", "start_times", "tracklet_ids", "camera_ids", "vectors")

    def __init__(self, dim, capacity=256):
        self.end_times = np.empty(capacity)
        self.start_times = np.empty(capacity)
        self.tracklet_ids = np.empty(capacity, np.int64)
        self.camera_ids = np.empty(capacity, np.int64)
        self.vectors = np.empty((capacity, dim), np.float32)
        self.head = 0           # first live slot
        self.tail = 0           # one past the last used slot
        self.base_index = 0     # absolute index of slot 0

    def append(self, tracklet_id, camera_id, start_time, end_time, vector) -> int:
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
        self.end_times[slot], self.start_times[slot] = end_time, start_time
        self.tracklet_ids[slot], self.camera_ids[slot] = tracklet_id, camera_id
        self.vectors[slot] = vector
        self.tail += 1
        return self.base_index + slot


class Matcher:
    def __init__(self, topology: Topology, window_slack: float, tau_keep: float,
                 camera_ids=None, cross_site_only: bool = False, max_walk_time: float = 600.0):
        self.window_slack = window_slack
        self.tau_keep = tau_keep
        camera_ids = list(range(topology.n_cameras)) if camera_ids is None else list(camera_ids)
        self.camera_ids = set(camera_ids)
        n_cameras = topology.n_cameras
        self.site_of_camera = topology.site_of_camera.astype(np.int64)
        self.buffers: dict[int, _SiteBuffer] = {}        # site_id -> buffer
        # tracklet_id -> (camera_id, absolute slot, start_time, end_time)
        self.location: dict[int, tuple] = {}
        self.max_neighbour_walk_time = 0.0
        walk_time = topology.walk_time
        # Per query camera c, per stored camera c2: a candidate's end time lies in
        # [query start - max_lag[c2], query start - min_lag[c2]].
        self.max_lag = {}
        self.min_lag = {}
        self.neighbour_walk_time = {}     # walk_time[c2, c] for neighbours, nan otherwise
        self.overlaps_with = {}           # bool per stored camera: views overlap
        self.source_sites = {}            # sites holding neighbour cameras
        self.lag_span = {}                # (largest max_lag, smallest min_lag) over neighbours
        for query_camera in camera_ids:
            max_lag = np.full(n_cameras, -np.inf)
            min_lag = np.full(n_cameras, np.inf)
            neighbour_walk_time = np.full(n_cameras, np.nan)
            overlaps_with = np.zeros(n_cameras, bool)
            for stored_camera in camera_ids:
                if (cross_site_only and topology.site_of_camera[stored_camera]
                        == topology.site_of_camera[query_camera]):
                    continue
                if walk_time[stored_camera, query_camera] <= max_walk_time:
                    walk = float(walk_time[stored_camera, query_camera])
                    neighbour_walk_time[stored_camera] = walk
                    if topology.views_overlap(stored_camera, query_camera):
                        overlaps_with[stored_camera] = True
                        max_lag[stored_camera], min_lag[stored_camera] = window_slack, -np.inf
                    else:
                        max_lag[stored_camera], min_lag[stored_camera] = walk + window_slack, walk
                    self.max_neighbour_walk_time = max(self.max_neighbour_walk_time, walk)
            is_neighbour = np.isfinite(max_lag)
            self.max_lag[query_camera], self.min_lag[query_camera] = max_lag, min_lag
            self.neighbour_walk_time[query_camera] = neighbour_walk_time
            self.overlaps_with[query_camera] = overlaps_with
            self.source_sites[query_camera] = sorted(
                {int(self.site_of_camera[camera]) for camera in np.nonzero(is_neighbour)[0]})
            self.lag_span[query_camera] = (
                (float(max_lag[is_neighbour].max()), float(min_lag[is_neighbour].min()))
                if is_neighbour.any() else (-np.inf, np.inf))
        self.n_scored = 0

    # ------------------------------------------------------------------ lookups
    def get_vector(self, tracklet_id):
        location = self.location.get(tracklet_id)
        if location is None:
            return None
        buffer = self.buffers[int(self.site_of_camera[location[0]])]
        slot = location[1] - buffer.base_index
        return buffer.vectors[slot] if slot >= buffer.head else None

    def metadata(self, tracklet_id):
        """(camera_id, start_time, end_time) of a stored tracklet."""
        location = self.location[tracklet_id]
        return location[0], location[2], location[3]

    # ------------------------------------------------------------------ scoring
    def _score_rows(self, buffer, row_start, row_end, mask, vector, exclude, candidates,
                    skip_tracklet_id=None):
        """Score the contiguous rows row_start:row_end (a view, no copy); keep masked rows."""
        if row_end <= row_start or not mask.any():
            return
        scores = buffer.vectors[row_start:row_end] @ vector
        self.n_scored += int(mask.sum())
        kept_rows = np.nonzero(mask & (scores >= self.tau_keep))[0]
        if len(kept_rows) == 0:
            return
        tracklet_ids = buffer.tracklet_ids[row_start + kept_rows].tolist()
        kept_scores = scores[kept_rows].tolist()
        for tracklet_id, score in zip(tracklet_ids, kept_scores):
            if tracklet_id != skip_tracklet_id and (exclude is None or tracklet_id not in exclude):
                candidates.append((tracklet_id, score))

    def candidates(self, camera_id, start_time, vector, exclude=None):
        """Earlier tracklets that may be the same person. Returns [(tracklet_id, score)]
        with score >= tau_keep, sorted descending."""
        source_sites = self.source_sites.get(camera_id)
        if not source_sites:
            return []
        max_lag, min_lag = self.max_lag[camera_id], self.min_lag[camera_id]
        largest_max_lag, smallest_min_lag = self.lag_span[camera_id]
        candidates = []
        for site_id in source_sites:
            buffer = self.buffers.get(site_id)
            if buffer is None or buffer.tail == buffer.head:
                continue
            end_times = buffer.end_times[buffer.head:buffer.tail]
            first = int(np.searchsorted(end_times, start_time - largest_max_lag, "left"))
            last = (len(end_times) if smallest_min_lag == -np.inf
                    else int(np.searchsorted(end_times, start_time - smallest_min_lag, "right")))
            if last <= first:
                continue
            stored_cameras = buffer.camera_ids[buffer.head + first:buffer.head + last]
            window_end_times = end_times[first:last]
            in_window = ((window_end_times >= start_time - max_lag[stored_cameras])
                         & (window_end_times <= start_time - min_lag[stored_cameras]))
            self._score_rows(buffer, buffer.head + first, buffer.head + last, in_window, vector,
                             exclude, candidates)
        candidates.sort(key=lambda candidate: -candidate[1])
        return candidates

    def forward_candidates(self, tracklet_id, vector, exclude=None):
        """Later tracklets whose window contains `tracklet_id` (used after a score revision)."""
        camera_id, _, end_time = self.metadata(tracklet_id)
        source_sites = self.source_sites.get(camera_id)
        if not source_sites:
            return []
        window_slack = self.window_slack
        neighbour_walk_time = self.neighbour_walk_time[camera_id]
        overlaps_with = self.overlaps_with[camera_id]
        candidates = []
        for site_id in source_sites:
            buffer = self.buffers.get(site_id)
            if buffer is None or buffer.tail == buffer.head:
                continue
            first_row = buffer.head + int(
                np.searchsorted(buffer.end_times[buffer.head:buffer.tail], end_time, "left"))
            if first_row >= buffer.tail:
                continue
            stored_cameras = buffer.camera_ids[first_row:buffer.tail]
            later_start_times = buffer.start_times[first_row:buffer.tail]
            walk = neighbour_walk_time[stored_cameras]     # nan for non-neighbours -> False
            overlapping = overlaps_with[stored_cameras]
            in_window = np.where(overlapping, end_time >= later_start_times - window_slack,
                                 (later_start_times - walk - window_slack <= end_time)
                                 & (end_time <= later_start_times - walk))
            in_window &= ~np.isnan(walk)
            self._score_rows(buffer, first_row, buffer.tail, in_window, vector, exclude,
                             candidates, skip_tracklet_id=tracklet_id)
        return candidates

    # ------------------------------------------------------------------ state
    def add(self, tracklet_id, camera_id, start_time, end_time, vector):
        if camera_id not in self.camera_ids:
            return
        site_id = int(self.site_of_camera[camera_id])
        buffer = self.buffers.get(site_id)
        if buffer is None:
            buffer = self.buffers[site_id] = _SiteBuffer(len(vector))
        slot = buffer.append(tracklet_id, camera_id, start_time, end_time, vector)
        self.location[tracklet_id] = (camera_id, slot, start_time, end_time)

    def update_vector(self, tracklet_id, vector) -> bool:
        location = self.location.get(tracklet_id)
        if location is None:
            return False
        buffer = self.buffers[int(self.site_of_camera[location[0]])]
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
