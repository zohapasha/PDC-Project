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
"""
from __future__ import annotations

import heapq

from .topology import Topology


class Checker:
    __slots__ = ("D", "group")

    def __init__(self, topo: Topology):
        self.D = topo.D.tolist()            # nested lists: faster scalar access than numpy
        self.group = topo.group.tolist()

    def compatible(self, u, v) -> bool:
        """u starts no later than v."""
        cu, cv = u[2], v[2]
        if cu != cv and self.group[cu] == self.group[cv]:
            return True
        return v[0] - u[1] >= self.D[cu][cv] - 1e-9

    def can_merge(self, A: list, B: list):
        """Returns (True, None) or (False, (u_tid, v_tid)) with u in A, v in B (or vice
        versa) -- a witness pair of incompatible visits."""
        if not A or not B:
            return True, None
        D, group = self.D, self.group
        it = heapq.merge(A, B)
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
        return self.can_merge(visits[:1], visits[1:])[0] if len(visits) > 1 else True

    def feasible_bruteforce(self, visits: list) -> bool:
        """All-pairs reference used by tests."""
        vs = sorted(visits)
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                if not self.compatible(vs[i], vs[j]):
                    return False
        return True
