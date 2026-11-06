"""Physical constraint check on person visit lists.

A visit is a tuple (t_start, t_end, cam, tid). Two visits u, v with s_u <= s_v are
compatible iff
  * cameras differ and are in the same overlap group (simultaneous presence ok), or
  * s_v - e_u >= D[c_u, c_v]          (same camera: D = 0, so no time overlap)

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


class Checker:
    __slots__ = ("D", "group", "Dmax", "dmax")

    def __init__(self, topo: Topology):
        self.D = topo.D.tolist()            # nested lists: faster scalar access than numpy
        self.group = topo.group.tolist()
        dm = float(topo.D.max()) if topo.n_cams else 0.0
        self.Dmax = dm if math.isfinite(dm) else None   # unreachable cams: no windowing
        self.dmax = None                    # longest visit seen; set via note_duration()

    def note_duration(self, d: float):
        if self.dmax is None or d > self.dmax:
            self.dmax = d

    def compatible(self, u, v) -> bool:
        """u starts no later than v."""
        cu, cv = u[2], v[2]
        if cu != cv and self.group[cu] == self.group[cv]:
            return True
        return v[0] - u[1] >= self.D[cu][cv] - 1e-9

    def can_merge(self, A: list, B: list):
        """Returns (True, None) or (False, (u_tid, v_tid)) with u in A, v in B (or vice
        versa) -- a witness pair of incompatible visits. A and B must each be feasible."""
        if not A or not B:
            return True, None
        if len(A) > len(B):
            A, B = B, A
        if len(B) > 16 and self.Dmax is not None and self.dmax is not None:
            lo = A[0][0] - self.Dmax - self.dmax - 1e-6
            hi = max(v[1] for v in A) + self.Dmax + 1e-6
            i = bisect_left(B, lo, key=itemgetter(0))
            j = bisect_right(B, hi, key=itemgetter(0))
            if i == j:
                return True, None
            B = B[i:j]
        return self._scan(sorted(A + B))

    def _scan(self, seq):
        D, group = self.D, self.group
        it = iter(seq)
        prev = next(it)
        last = {prev[2]: prev}                 # latest-ending visit per camera
        for cur in it:
            cp, cc = prev[2], cur[2]
            if not (cp != cc and group[cp] == group[cc]):
                if cur[0] - prev[1] < D[cp][cc] - 1e-9:
                    return False, (prev[3], cur[3])
            elif cc in last:                   # prev overlaps cur's view: check own camera
                lc = last[cc]
                if cur[0] < lc[1] - 1e-9:
                    return False, (lc[3], cur[3])
            if cur[1] > prev[1]:
                prev = cur
            lc = last.get(cc)
            if lc is None or cur[1] > lc[1]:
                last[cc] = cur
        return True, None

    def feasible(self, visits: list) -> bool:
        """Feasibility of a single (sorted) visit list."""
        return self._scan(visits)[0] if len(visits) > 1 else True

    def feasible_bruteforce(self, visits: list) -> bool:
        """All-pairs reference used by tests."""
        vs = sorted(visits)
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                if not self.compatible(vs[i], vs[j]):
                    return False
        return True
