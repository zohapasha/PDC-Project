"""Physical constraint check on person visit lists.

A visit is a tuple (start_time, end_time, camera_id, tracklet_id). Two visits u, v with
s_u <= s_v are compatible iff
  * cameras differ and are in the same overlap group (simultaneous presence ok), or
  * s_v - e_u >= D[c_u, c_v]          (same camera: D = 0, so no time overlap)
where D is the walking-time matrix.

Lemma (linear check). Because D is a shortest-path metric and overlap cameras are at
distance 0, a visit list sorted by start time is feasible iff every visit v is
compatible with
  (i)  the latest-ending visit that started before v, and
  (ii) the latest-ending earlier visit on v's own camera.
(ii) is needed only with overlap groups: if the latest-ending visit p is on a camera
overlapping v's, it says nothing about an earlier visit on v's own camera. Every other
earlier visit u is covered by (i) through the triangle inequality:
  s_v - e_u >= (s_v - e_p) + (s_p - e_u) >= D[p,v] + D[u,p] >= D[u,v].
The check is a single linear merge of the two sorted visit lists: O(|A| + |B|).

Windowing (can_merge only). A and B are each feasible, so only cross pairs can violate.
Let S be the smaller list, L the larger, Dmax = max_{c,c'} D[c,c'] and dmax the longest
visit. An L-visit u with s_u < s_min(S) - Dmax - dmax ends before s_min(S) - Dmax, so it
is compatible with every S-visit; an L-visit u with s_u > e_max(S) + Dmax likewise. The
lemma applied to S plus the remaining slice of L therefore decides the merge exactly, in
O(|S| + |slice| + log |L|).
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from operator import itemgetter

from .topology import Topology

_TOLERANCE = 1e-9
_WINDOWING_MIN_LENGTH = 16       # window the larger list only when it is longer than this


class ConstraintChecker:
    __slots__ = ("walk_time", "overlap_group", "max_walk_time", "max_visit_duration")

    def __init__(self, topology: Topology):
        # nested lists: faster scalar access than numpy
        self.walk_time = topology.walk_time.tolist()
        self.overlap_group = topology.overlap_group_of_camera.tolist()
        largest = float(topology.walk_time.max()) if topology.n_cameras else 0.0
        # unreachable cameras (infinite walking time): no windowing
        self.max_walk_time = largest if math.isfinite(largest) else None
        self.max_visit_duration = None      # longest visit seen; set via note_duration()

    def note_duration(self, duration: float):
        if self.max_visit_duration is None or duration > self.max_visit_duration:
            self.max_visit_duration = duration

    def compatible(self, earlier, later) -> bool:
        """`earlier` starts no later than `later`."""
        earlier_camera, later_camera = earlier[2], later[2]
        if (earlier_camera != later_camera
                and self.overlap_group[earlier_camera] == self.overlap_group[later_camera]):
            return True
        return later[0] - earlier[1] >= self.walk_time[earlier_camera][later_camera] - _TOLERANCE

    def can_merge(self, visits_a: list, visits_b: list):
        """Returns (True, None) or (False, (tracklet_u, tracklet_v)) with u in one list and
        v in the other -- a witness pair of incompatible visits. Both lists must each be
        feasible."""
        if not visits_a or not visits_b:
            return True, None
        if len(visits_a) > len(visits_b):
            smaller, larger = visits_b, visits_a
        else:
            smaller, larger = visits_a, visits_b
        if (len(larger) > _WINDOWING_MIN_LENGTH and self.max_walk_time is not None
                and self.max_visit_duration is not None):
            window_start = smaller[0][0] - self.max_walk_time - self.max_visit_duration - 1e-6
            window_end = max(visit[1] for visit in smaller) + self.max_walk_time + 1e-6
            first = bisect_left(larger, window_start, key=itemgetter(0))
            last = bisect_right(larger, window_end, key=itemgetter(0))
            if first == last:
                return True, None
            larger = larger[first:last]
        return self._scan_sorted(sorted(smaller + larger))

    def _scan_sorted(self, visits):
        # Hot loop: visits are (start_time, end_time, camera_id, tracklet_id) tuples and
        # are indexed directly ([0] start, [1] end, [2] camera, [3] tracklet).
        walk_time, overlap_group = self.walk_time, self.overlap_group
        visit_iter = iter(visits)
        latest = next(visit_iter)                      # latest-ending visit so far
        latest_on_camera = {latest[2]: latest}         # latest-ending visit per camera
        for current in visit_iter:
            latest_camera, current_camera = latest[2], current[2]
            if not (latest_camera != current_camera
                    and overlap_group[latest_camera] == overlap_group[current_camera]):
                if current[0] - latest[1] < walk_time[latest_camera][current_camera] - _TOLERANCE:
                    return False, (latest[3], current[3])
            elif current_camera in latest_on_camera:   # latest overlaps current's view:
                same_camera = latest_on_camera[current_camera]   # check its own camera
                if current[0] < same_camera[1] - _TOLERANCE:
                    return False, (same_camera[3], current[3])
            if current[1] > latest[1]:
                latest = current
            same_camera = latest_on_camera.get(current_camera)
            if same_camera is None or current[1] > same_camera[1]:
                latest_on_camera[current_camera] = current
        return True, None

    def feasible(self, visits: list) -> bool:
        """Feasibility of a single (sorted) visit list."""
        return self._scan_sorted(visits)[0] if len(visits) > 1 else True

    def feasible_bruteforce(self, visits: list) -> bool:
        """All-pairs reference used by tests."""
        sorted_visits = sorted(visits)
        for i in range(len(sorted_visits)):
            for j in range(i + 1, len(sorted_visits)):
                if not self.compatible(sorted_visits[i], sorted_visits[j]):
                    return False
        return True
