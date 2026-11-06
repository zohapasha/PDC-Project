"""In-memory vertex-centric identity engine.

State
  * every tracklet is a vertex; accepted matches inside a person form one tree
    (parent pointers + children lists); other accepted matches are *backup links*;
  * every person keeps a summary at its root: size and its sorted visit list;
  * all matches with score >= tau_keep are stored (adjacency dicts).

Operations (each is one logged "op")
  add_match    -> accept as backup / merge (union by size, reroot one path) /
                  conflict resolution (cut weakest match on the conflict path)
  update_match -> score revision; may delete a tree edge (cut + backup reconnection)
  remove_match -> deletion (operator correction or a site deleting a match at the hub)

Conflict resolution is atomic: if the new match turns out to be the weakest after some
cuts, every change made during the attempt is rolled back.

Modes
  full          : the proposed method
  no_correction : B3 -- constraint check at merge, never undo/cut (union-find style)
  component     : B2 -- on any cut, rebuild the whole person's tree from its matches
"""
from __future__ import annotations

import heapq
import os
import time

from ..common.constraints import Checker
from ..common.topology import Topology
from ..common.types import (INF, K_BACKUP, K_CONFLICT_OK, K_CONFLICT_REJ, K_DEL_BACKUP,
                            K_DEL_TREE, K_MERGE, K_REJECT, Person, Vertex)
from .vertex import CHECK_BACKUP, CLIMB, MARK, RELABEL, REROOT, VertexScheduler

_CHECK = os.environ.get("ARGUS_CHECK", "0") == "1"


class IdentityEngine:
    def __init__(self, topo: Topology, tau_accept: float, tau_keep: float,
                 mode: str = "full", emit: bool = False, emit_belongs: bool = False,
                 log_ops: bool = True, check: bool | None = None):
        assert mode in ("full", "no_correction", "component")
        self.chk = Checker(topo)
        self.tau_accept = tau_accept
        self.tau_keep = tau_keep
        self.mode = mode
        self.emit = emit
        self.emit_belongs = emit_belongs
        self.log_ops = log_ops
        self.check = _CHECK if check is None else check
        self.V: dict[int, Vertex] = {}
        self.P: dict[int, Person] = {}
        self.adj: dict[int, dict[int, float]] = {}
        self.blocked: set = set()
        self._next_pid = 0
        self.out: list = []          # committed deltas
        self.oplog: list = []        # (kind, msgs, touched, size, nV, ns, extra)
        # per-op scratch
        self._pend: list = []
        self._relabel: dict = {}
        self._newpids: list = []
        self._t0 = 0
        s = self.sched = VertexScheduler()
        s.register(CLIMB, self._h_climb)
        s.register(MARK, self._h_mark)
        s.register(RELABEL, self._h_relabel)
        s.register(REROOT, self._h_reroot)
        s.register(CHECK_BACKUP, self._h_check_backup)

    # ================================================================ handlers
    # Each handler reads/writes only the vertex it runs on, then messages neighbours.
    def _h_climb(self, v, ctx):                 # ctx = [path, stop_set]
        ctx[0].append(v)
        if ctx[1] is not None and v in ctx[1]:
            return
        p = self.V[v].parent
        if p != -1:
            self.sched.send(p, CLIMB, ctx)

    def _h_mark(self, v, out):
        out.append(v)
        for ch in self.V[v].children:
            self.sched.send(ch, MARK, out)

    def _h_relabel(self, v, pid):
        vx = self.V[v]
        if v not in self._relabel:
            self._relabel[v] = vx.person
        vx.person = pid
        for ch in vx.children:
            self.sched.send(ch, RELABEL, pid)

    def _h_reroot(self, v, payload):           # payload = (new_parent, weight)
        prev, prevw = payload
        vx = self.V[v]
        if prev != -1:
            vx.children.remove(prev)            # my old child becomes my parent
        nxt, nw = vx.parent, vx.pw
        vx.parent, vx.pw = prev, prevw
        if nxt != -1:
            vx.children.append(nxt)
            self.sched.send(nxt, REROOT, (v, nw))

    def _h_check_backup(self, v, ctx):         # ctx = [marked_set, best]
        bk = self.V[v].backups
        if not bk:
            return
        S = ctx[0]
        for o, w in bk.items():
            if o not in S and (ctx[1] is None or w > ctx[1][2]):
                ctx[1] = (v, o, w)

    # ================================================================ op framing
    def _begin(self):
        self.sched.reset()
        self._pend = []
        self._relabel = {}
        self._newpids = []
        self._t0 = time.perf_counter_ns()

    def _commit(self, kind, size, extra=0):
        ns = time.perf_counter_ns() - self._t0
        if self.emit:
            self.out.extend(self._pend)
        if self.emit_belongs:
            V = self.V
            for t, orig in self._relabel.items():
                new = V[t].person
                if new != orig:
                    self.out.append(("B", t, orig, new))
        if self.log_ops:
            self.oplog.append((kind, self.sched.msgs, len(self.sched.touched), size,
                               len(self.V), ns, extra))
        if self.check:
            self.check_invariants()

    def _new_pid(self):
        pid = self._next_pid
        self._next_pid += 1
        self._newpids.append(pid)
        return pid

    # ================================================================ public API
    def add_vertex(self, tid: int, cam: int, ts: float, te: float):
        if tid in self.V:
            return
        pid = self._next_pid
        self._next_pid += 1
        self.V[tid] = Vertex(tid, cam, ts, te, person=pid)
        self.P[pid] = Person(pid, tid, 1, [(ts, te, cam, tid)])
        self.adj[tid] = {}
        if self.emit_belongs:
            self.out.append(("B", tid, -1, pid))

    def add_match(self, a: int, b: int, w: float):
        if a == b or a not in self.V or b not in self.V:
            return
        if (a, b) in self.blocked or (b, a) in self.blocked:
            return
        adj_a = self.adj[a]
        if b in adj_a:
            return self.update_match(a, b, w)
        if w < self.tau_keep:
            return
        adj_a[b] = w
        self.adj[b][a] = w
        if w >= self.tau_accept:
            self._try_accept(a, b, w)

    def update_match(self, a: int, b: int, w: float):
        """Score revision of a stored match."""
        old = self.adj[a].get(b)
        if old is None:
            return self.add_match(a, b, w)
        if w < self.tau_keep:
            self._demote(a, b)
            del self.adj[a][b], self.adj[b][a]
            return
        self.adj[a][b] = self.adj[b][a] = w
        va, vb = self.V[a], self.V[b]
        if va.parent == b or vb.parent == a:
            if w >= self.tau_accept:
                (va if va.parent == b else vb).pw = w
            else:
                self._demote(a, b)
        elif va.backups and b in va.backups:
            if w >= self.tau_accept:
                va.backups[b] = vb.backups[a] = w
            else:
                self._demote(a, b)
        elif w >= self.tau_accept and old < self.tau_accept:
            self._try_accept(a, b, w)

    def remove_match(self, a: int, b: int, block: bool = False):
        """Delete a match (operator correction if block=True; it can never come back)."""
        if block:
            self.blocked.add((a, b))
        if a in self.adj and b in self.adj[a]:
            self._demote(a, b)
            del self.adj[a][b], self.adj[b][a]

    # ================================================================ internals
    def _demote(self, a, b):
        """Accepted match (tree or backup) is no longer accepted."""
        va, vb = self.V[a], self.V[b]
        if va.backups and b in va.backups:
            self._begin()
            del va.backups[b], vb.backups[a]
            self.sched.local(a)
            self.sched.local(b)
            self._commit(K_DEL_BACKUP, self.P[va.person].size)
        elif va.parent == b or vb.parent == a:
            self._begin()
            size = self.P[va.person].size
            child = a if va.parent == b else b
            rec = self._cut(child)
            self._commit(K_DEL_TREE, size, int(rec))

    def _try_accept(self, a, b, w):
        self._begin()
        va, vb = self.V[a], self.V[b]
        if va.person == vb.person:
            if va.backups is None:
                va.backups = {}
            if vb.backups is None:
                vb.backups = {}
            va.backups[b] = w
            vb.backups[a] = w
            self.sched.local(a)
            self.sched.local(b)
            self._commit(K_BACKUP, self.P[va.person].size)
            return True
        A, B = self.P[va.person], self.P[vb.person]
        size = A.size + B.size
        ok, conf = self.chk.can_merge(A.visits, B.visits)
        if ok:
            self._merge(a, b, w)
            self._commit(K_MERGE, size)
            return True
        if self.mode == "no_correction":
            self._commit(K_REJECT, size)
            return False
        return self._resolve(a, b, w, conf, size)

    def _merge(self, a, b, w):
        V, P, s = self.V, self.P, self.sched
        A, B = P[V[a].person], P[V[b].person]
        if A.size > B.size:                       # union by size: relabel the smaller
            a, b, A, B = b, a, B, A
        s.send(a, REROOT, (-1, 0.0))              # a becomes root of A
        s.run()
        va = V[a]
        va.parent, va.pw = b, w
        V[b].children.append(a)
        s.local(b)
        if self.emit:
            self._pend.append(("+", a, b, w))
        s.send(a, RELABEL, B.pid)
        s.run()
        B.visits = list(heapq.merge(A.visits, B.visits))
        B.size += A.size
        del P[A.pid]

    def _path_min(self, u, v):
        """Weakest tree edge on the u..v path (same tree). Returns (w, child) or None."""
        if u == v:
            return None
        s = self.sched
        cu = [[], None]
        s.send(u, CLIMB, cu)
        s.run()
        idx = {x: i for i, x in enumerate(cu[0])}
        cv = [[], idx]
        s.send(v, CLIMB, cv)
        s.run()
        lca = cv[0][-1]
        V = self.V
        best = None
        for x in cu[0][:idx[lca]]:
            if best is None or V[x].pw < best[0]:
                best = (V[x].pw, x)
        for x in cv[0][:-1]:
            if best is None or V[x].pw < best[0]:
                best = (V[x].pw, x)
        return best

    def _resolve(self, a, b, w, conf, size):
        """Fig. 2 case. Cut the weakest match on the cycle the new match would close
        through the conflicting pair, until the merge becomes feasible or the new match
        is itself the weakest (then roll back everything)."""
        V, P = self.V, self.P
        snap = self._snapshot((V[a].person, V[b].person))
        cuts = 0
        while True:
            x, y = conf
            if V[x].person != V[a].person:
                x, y = y, x
            if V[x].person != V[a].person or V[y].person != V[b].person:
                raise AssertionError("conflict witness does not cross the two persons")
            m1 = self._path_min(x, a)
            m2 = self._path_min(b, y)
            cand = [m for m in (m1, m2) if m is not None]
            weakest = min(cand) if cand else None
            if weakest is None or weakest[0] >= w:
                self._rollback(snap)
                self._commit(K_CONFLICT_REJ, size, cuts)
                return False
            self._cut(weakest[1])
            cuts += 1
            A, B = P[V[a].person], P[V[b].person]
            if A is B:                            # cannot happen (backups stay inside a person)
                self._commit(K_CONFLICT_OK, size, cuts)
                return True
            ok, conf = self.chk.can_merge(A.visits, B.visits)
            if ok:
                self._merge(a, b, w)
                self._commit(K_CONFLICT_OK, size, cuts)
                return True

    def _cut(self, c) -> bool:
        """Delete tree edge c -> parent(c). Mark the subtree below it, look for the best
        backup link back to the rest of the person; else split the person.
        Returns True if the person stayed whole via a backup link."""
        V, P, s = self.V, self.P, self.sched
        vc = V[c]
        p = vc.parent
        V[p].children.remove(c)
        s.local(p)
        vc.parent, vc.pw = -1, 0.0
        del_idx = len(self._pend)
        if self.emit:
            self._pend.append(("-", c, p))
        if self.mode == "component":
            return self._rebuild_component(c, p)
        pid = vc.person
        Pobj = P[pid]
        S = []
        s.send(c, MARK, S)
        s.run()
        Sset = set(S)
        ctx = [Sset, None]
        for x in S:
            s.send(x, CHECK_BACKUP, ctx)
        s.run()
        best = ctx[1]
        if best is not None and best[2] >= self.tau_accept:
            x, o, bw = best
            del V[x].backups[o], V[o].backups[x]
            s.send(x, REROOT, (-1, 0.0))
            s.run()
            vx = V[x]
            vx.parent, vx.pw = o, bw
            V[o].children.append(x)
            s.local(o)
            if self.emit:                         # announce the replacement before the cut
                self._pend.insert(del_idx, ("+", x, o, bw))
            return True
        # split: relabel the smaller side
        Svis, Rvis = [], []
        for vis in Pobj.visits:
            (Svis if vis[3] in Sset else Rvis).append(vis)
        newpid = self._new_pid()
        nS = len(S)
        if nS <= Pobj.size - nS:
            P[newpid] = Person(newpid, c, nS, Svis)
            Pobj.size -= nS
            Pobj.visits = Rvis
            s.send(c, RELABEL, newpid)
        else:
            rest_root = Pobj.root
            P[newpid] = Person(newpid, rest_root, Pobj.size - nS, Rvis)
            Pobj.root, Pobj.size, Pobj.visits = c, nS, Svis
            s.send(rest_root, RELABEL, newpid)
        s.run()
        return False

    # ---------------------------------------------------------------- B2 baseline
    def _rebuild_component(self, c, p) -> bool:
        """B2: BFS the whole person, rebuild its maximum spanning forest from all its
        accepted matches (Kruskal), split into components, relabel."""
        V, P, s = self.V, self.P, self.sched
        pid = V[c].person
        Pobj = P[pid]
        members = []
        s.send(Pobj.root, MARK, members)
        s.send(c, MARK, members)
        s.run()
        edges = []
        for m in members:
            vm = V[m]
            if vm.parent != -1:
                edges.append((vm.pw, m, vm.parent))
            if vm.backups:
                for o, w in vm.backups.items():
                    if m < o:
                        edges.append((w, m, o))
        edges.sort(reverse=True)
        uf = {m: m for m in members}

        def find(x):
            while uf[x] != x:
                uf[x] = uf[uf[x]]
                x = uf[x]
            return x

        tree_adj = {m: [] for m in members}
        back = []
        for w, x, y in edges:
            rx, ry = find(x), find(y)
            if rx != ry:
                uf[rx] = ry
                tree_adj[x].append((y, w))
                tree_adj[y].append((x, w))
            else:
                back.append((x, y, w))
        s.msgs += 2 * len(members)                 # rebuild pass over every member
        for m in members:
            vm = V[m]
            vm.parent, vm.pw, vm.children, vm.backups = -1, 0.0, [], None
        for x, y, w in back:
            for u, v in ((x, y), (y, x)):
                if V[u].backups is None:
                    V[u].backups = {}
                V[u].backups[v] = w
        comps = []
        seen = set()
        for m in members:
            if m in seen:
                continue
            seen.add(m)
            comp, stack = [m], [m]
            while stack:
                u = stack.pop()
                for v, w in tree_adj[u]:
                    if v not in seen:
                        seen.add(v)
                        V[v].parent, V[v].pw = u, w
                        V[u].children.append(v)
                        comp.append(v)
                        stack.append(v)
            comps.append(comp)
        comps.sort(key=len, reverse=True)
        vis_of = {v[3]: v for v in Pobj.visits}
        first = comps[0]
        Pobj.root, Pobj.size = first[0], len(first)
        Pobj.visits = sorted(vis_of[t] for t in first)
        for comp in comps[1:]:
            npid = self._new_pid()
            P[npid] = Person(npid, comp[0], len(comp), sorted(vis_of[t] for t in comp))
            for t in comp:
                if t not in self._relabel:
                    self._relabel[t] = V[t].person
                V[t].person = npid
        return len(comps) == 1

    # ---------------------------------------------------------------- rollback
    def _snapshot(self, pids):
        V, P = self.V, self.P
        vs, ps = {}, {}
        for pid in set(pids):
            Pobj = P[pid]
            ps[pid] = (Pobj.root, Pobj.size, Pobj.visits)
            for vis in Pobj.visits:
                vx = V[vis[3]]
                vs[vx.tid] = (vx.person, vx.parent, vx.pw, list(vx.children),
                              dict(vx.backups) if vx.backups else None)
        return vs, ps

    def _rollback(self, snap):
        vs, ps = snap
        V, P = self.V, self.P
        for pid in self._newpids:
            P.pop(pid, None)
        for t, (person, parent, pw, children, backups) in vs.items():
            vx = V[t]
            vx.person, vx.parent, vx.pw, vx.children, vx.backups = (
                person, parent, pw, children, backups)
        for pid, (root, size, visits) in ps.items():
            P[pid] = Person(pid, root, size, visits)
        self._pend = []
        self._relabel = {}

    # ================================================================ inspection
    def partition(self) -> dict:
        return {t: v.person for t, v in self.V.items()}

    def depths(self) -> list:
        memo = {}
        V = self.V
        for t in V:
            path = []
            x = t
            while x not in memo and V[x].parent != -1:
                path.append(x)
                x = V[x].parent
            d = memo.get(x, 0)
            memo[x] = d
            for y in reversed(path):
                d += 1
                memo[y] = d
        return list(memo.values())

    def n_violations(self) -> int:
        return sum(0 if self.chk.feasible(Pobj.visits) else 1 for Pobj in self.P.values())

    def check_invariants(self):
        """I1 (one tree per person), I2 (labels = root's person), I3 (feasible),
        plus: backup links stay inside a person and are symmetric."""
        V, P = self.V, self.P
        seen = 0
        for pid, Pobj in P.items():
            root = Pobj.root
            assert V[root].parent == -1, f"root {root} has a parent"
            assert V[root].person == pid
            stack, cnt = [root], 0
            visited = set()
            while stack:
                u = stack.pop()
                assert u not in visited, "cycle"
                visited.add(u)
                cnt += 1
                vu = V[u]
                assert vu.person == pid, f"I2 broken at {u}"
                for ch in vu.children:
                    assert V[ch].parent == u, "child/parent mismatch"
                    stack.append(ch)
                if vu.backups:
                    for o, w in vu.backups.items():
                        assert V[o].person == pid, "backup crosses persons"
                        assert V[o].backups[u] == w
            assert cnt == Pobj.size == len(Pobj.visits), f"I1 size mismatch pid={pid}"
            assert {v[3] for v in Pobj.visits} == visited
            assert self.chk.feasible(Pobj.visits), f"I3 violated pid={pid}"
            seen += cnt
        assert seen == len(V), "vertices not covered by persons"
