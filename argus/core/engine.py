"""In-memory vertex-centric identity engine.

State
  * every tracklet is a vertex; accepted matches inside a person form one tree
    (parent pointers + child lists); other accepted matches are *backup links*;
  * every person keeps a summary at its root: size and its sorted visit list;
  * all matches with score >= tau_keep are stored (symmetric score dicts).

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

import os
import time

from ..common.constraints import ConstraintChecker
from ..common.topology import Topology
from ..common.types import (DELTA_BELONGS_TO, DELTA_TREE_EDGE_ADDED, DELTA_TREE_EDGE_REMOVED,
                            DELTA_TREE_EDGE_SCORE, NO_PARENT, OP_BACKUP, OP_CONFLICT_REJECTED,
                            OP_CONFLICT_RESOLVED, OP_DELETE_BACKUP, OP_DELETE_TREE_EDGE,
                            OP_MERGE, OP_REJECT, VISIT_TRACKLET, OpRecord, Person, Vertex)
from .vertex import CHECK_BACKUP, CLIMB, MARK, RELABEL, REROOT, VertexScheduler

_VERIFY_INVARIANTS_BY_DEFAULT = os.environ.get("ARGUS_CHECK", "0") == "1"


class IdentityEngine:
    def __init__(self, topology: Topology, tau_accept: float, tau_keep: float,
                 mode: str = "full", emit_tree_deltas: bool = False,
                 emit_belongs_to: bool = False, log_ops: bool = True,
                 verify_invariants: bool | None = None):
        assert mode in ("full", "no_correction", "component")
        self.checker = ConstraintChecker(topology)
        self.tau_accept = tau_accept
        self.tau_keep = tau_keep
        self.mode = mode
        self.emit_tree_deltas = emit_tree_deltas
        self.emit_belongs_to = emit_belongs_to
        self.log_ops = log_ops
        self.verify_invariants = (_VERIFY_INVARIANTS_BY_DEFAULT if verify_invariants is None
                                  else verify_invariants)
        self.vertices: dict[int, Vertex] = {}
        self.persons: dict[int, Person] = {}
        self.match_scores: dict[int, dict[int, float]] = {}   # stored matches, both directions
        self.blocked_pairs: set = set()
        self._next_person_id = 0
        self.emitted_deltas: list = []           # committed deltas (see DELTA_* in types)
        self.op_log: list[OpRecord] = []
        # per-op scratch
        self._pending_deltas: list = []
        self._original_person_of: dict = {}      # tracklet_id -> person_id before this op
        self._new_person_ids: list = []
        self._op_start_ns = 0
        scheduler = self.scheduler = VertexScheduler()
        scheduler.register(CLIMB, self._handle_climb)
        scheduler.register(MARK, self._handle_mark)
        scheduler.register(RELABEL, self._handle_relabel)
        scheduler.register(REROOT, self._handle_reroot)
        scheduler.register(CHECK_BACKUP, self._handle_check_backup)

    # ================================================================ handlers
    # Each handler reads/writes only the vertex it runs on, then messages neighbours.
    def _handle_climb(self, vertex_id, context):      # context = [path, stop_at]
        path, stop_at = context
        path.append(vertex_id)
        if stop_at is not None and vertex_id in stop_at:
            return
        parent_id = self.vertices[vertex_id].parent_id
        if parent_id != NO_PARENT:
            self.scheduler.send(parent_id, CLIMB, context)

    def _handle_mark(self, vertex_id, marked):
        marked.append(vertex_id)
        for child_id in self.vertices[vertex_id].child_ids:
            self.scheduler.send(child_id, MARK, marked)

    def _handle_relabel(self, vertex_id, person_id):
        vertex = self.vertices[vertex_id]
        if vertex_id not in self._original_person_of:
            self._original_person_of[vertex_id] = vertex.person_id
        vertex.person_id = person_id
        for child_id in vertex.child_ids:
            self.scheduler.send(child_id, RELABEL, person_id)

    def _handle_reroot(self, vertex_id, payload):     # payload = (new_parent_id, edge_score)
        new_parent_id, new_parent_score = payload
        vertex = self.vertices[vertex_id]
        if new_parent_id != NO_PARENT:
            vertex.child_ids.remove(new_parent_id)    # my old child becomes my parent
        old_parent_id, old_parent_score = vertex.parent_id, vertex.parent_edge_score
        vertex.parent_id, vertex.parent_edge_score = new_parent_id, new_parent_score
        if old_parent_id != NO_PARENT:
            vertex.child_ids.append(old_parent_id)
            self.scheduler.send(old_parent_id, REROOT, (vertex_id, old_parent_score))

    def _handle_check_backup(self, vertex_id, search):  # search = [marked_set, best_link]
        backup_links = self.vertices[vertex_id].backup_links
        if not backup_links:
            return
        marked = search[0]
        for partner_id, score in backup_links.items():
            if partner_id not in marked and (search[1] is None or score > search[1][2]):
                search[1] = (vertex_id, partner_id, score)

    # ================================================================ op framing
    def _begin_op(self):
        self.scheduler.reset()
        self._pending_deltas = []
        self._original_person_of = {}
        self._new_person_ids = []
        self._op_start_ns = time.perf_counter_ns()

    def _commit_op(self, kind, person_size, extra=0):
        latency_ns = time.perf_counter_ns() - self._op_start_ns
        if self.emit_tree_deltas:
            self.emitted_deltas.extend(self._pending_deltas)
        if self.emit_belongs_to:
            vertices = self.vertices
            for tracklet_id, original_person_id in self._original_person_of.items():
                new_person_id = vertices[tracklet_id].person_id
                if new_person_id != original_person_id:
                    self.emitted_deltas.append(
                        (DELTA_BELONGS_TO, tracklet_id, original_person_id, new_person_id))
        if self.log_ops:
            self.op_log.append(OpRecord(kind, self.scheduler.n_messages,
                                        len(self.scheduler.touched_vertices), person_size,
                                        len(self.vertices), latency_ns, extra))
        if self.verify_invariants:
            self.check_invariants()

    def _allocate_person_id(self):
        person_id = self._next_person_id
        self._next_person_id += 1
        self._new_person_ids.append(person_id)
        return person_id

    # ================================================================ public API
    def add_vertex(self, tracklet_id: int, camera_id: int, start_time: float, end_time: float,
                   site_tag: int = -1):
        if tracklet_id in self.vertices:
            return
        person_id = self._next_person_id
        self._next_person_id += 1
        self.vertices[tracklet_id] = Vertex(tracklet_id, camera_id, start_time, end_time,
                                            person_id=person_id)
        self.persons[person_id] = Person(person_id, tracklet_id, 1,
                                         [(start_time, end_time, camera_id, tracklet_id)], site_tag)
        self.checker.note_duration(end_time - start_time)
        self.match_scores[tracklet_id] = {}
        if self.emit_belongs_to:
            self.emitted_deltas.append((DELTA_BELONGS_TO, tracklet_id, -1, person_id))

    def add_match(self, tracklet_a: int, tracklet_b: int, score: float):
        if tracklet_a == tracklet_b or tracklet_a not in self.vertices or tracklet_b not in self.vertices:
            return
        if (tracklet_a, tracklet_b) in self.blocked_pairs or (tracklet_b, tracklet_a) in self.blocked_pairs:
            return
        scores_of_a = self.match_scores[tracklet_a]
        if tracklet_b in scores_of_a:
            return self.update_match(tracklet_a, tracklet_b, score)
        if score < self.tau_keep:
            return
        scores_of_a[tracklet_b] = score
        self.match_scores[tracklet_b][tracklet_a] = score
        if score >= self.tau_accept:
            self._try_accept_match(tracklet_a, tracklet_b, score)

    def update_match(self, tracklet_a: int, tracklet_b: int, score: float):
        """Score revision of a stored match."""
        old_score = self.match_scores[tracklet_a].get(tracklet_b)
        if old_score is None:
            return self.add_match(tracklet_a, tracklet_b, score)
        if score < self.tau_keep:
            self._demote_match(tracklet_a, tracklet_b)
            del self.match_scores[tracklet_a][tracklet_b], self.match_scores[tracklet_b][tracklet_a]
            return
        self.match_scores[tracklet_a][tracklet_b] = self.match_scores[tracklet_b][tracklet_a] = score
        vertex_a, vertex_b = self.vertices[tracklet_a], self.vertices[tracklet_b]
        if vertex_a.parent_id == tracklet_b or vertex_b.parent_id == tracklet_a:
            if score >= self.tau_accept:
                child = vertex_a if vertex_a.parent_id == tracklet_b else vertex_b
                child.parent_edge_score = score
                if self.emit_tree_deltas:              # score change of a tree match
                    self.emitted_deltas.append(
                        (DELTA_TREE_EDGE_SCORE, tracklet_a, tracklet_b, score))
            else:
                self._demote_match(tracklet_a, tracklet_b)
        elif vertex_a.backup_links and tracklet_b in vertex_a.backup_links:
            if score >= self.tau_accept:
                vertex_a.backup_links[tracklet_b] = vertex_b.backup_links[tracklet_a] = score
            else:
                self._demote_match(tracklet_a, tracklet_b)
        elif score >= self.tau_accept and old_score < self.tau_accept:
            self._try_accept_match(tracklet_a, tracklet_b, score)

    def remove_match(self, tracklet_a: int, tracklet_b: int, block: bool = False):
        """Delete a match (operator correction if block=True; it can never come back)."""
        if block:
            self.blocked_pairs.add((tracklet_a, tracklet_b))
        if tracklet_a in self.match_scores and tracklet_b in self.match_scores[tracklet_a]:
            self._demote_match(tracklet_a, tracklet_b)
            del self.match_scores[tracklet_a][tracklet_b], self.match_scores[tracklet_b][tracklet_a]

    # ================================================================ internals
    def _demote_match(self, tracklet_a, tracklet_b):
        """Accepted match (tree or backup) is no longer accepted."""
        vertex_a, vertex_b = self.vertices[tracklet_a], self.vertices[tracklet_b]
        if vertex_a.backup_links and tracklet_b in vertex_a.backup_links:
            self._begin_op()
            del vertex_a.backup_links[tracklet_b], vertex_b.backup_links[tracklet_a]
            self.scheduler.record_local_work(tracklet_a)
            self.scheduler.record_local_work(tracklet_b)
            self._commit_op(OP_DELETE_BACKUP, self.persons[vertex_a.person_id].size)
        elif vertex_a.parent_id == tracklet_b or vertex_b.parent_id == tracklet_a:
            self._begin_op()
            person_size = self.persons[vertex_a.person_id].size
            child_id = tracklet_a if vertex_a.parent_id == tracklet_b else tracklet_b
            repaired_by_backup = self._cut_tree_edge(child_id)
            self._commit_op(OP_DELETE_TREE_EDGE, person_size, int(repaired_by_backup))

    def _try_accept_match(self, tracklet_a, tracklet_b, score):
        self._begin_op()
        vertex_a, vertex_b = self.vertices[tracklet_a], self.vertices[tracklet_b]
        if vertex_a.person_id == vertex_b.person_id:
            if vertex_a.backup_links is None:
                vertex_a.backup_links = {}
            if vertex_b.backup_links is None:
                vertex_b.backup_links = {}
            vertex_a.backup_links[tracklet_b] = score
            vertex_b.backup_links[tracklet_a] = score
            self.scheduler.record_local_work(tracklet_a)
            self.scheduler.record_local_work(tracklet_b)
            self._commit_op(OP_BACKUP, self.persons[vertex_a.person_id].size)
            return True
        person_a, person_b = self.persons[vertex_a.person_id], self.persons[vertex_b.person_id]
        merged_size = person_a.size + person_b.size
        if person_a.site_tag >= 0 and person_a.site_tag == person_b.site_tag:
            # both persons lie inside one site person (site-pure, joined by that site's
            # matches): the union is a subset of a feasible set -> feasible, no check
            feasible = True
        else:
            feasible, witness = self.checker.can_merge(person_a.visits, person_b.visits)
        if feasible:
            self._merge_persons(tracklet_a, tracklet_b, score)
            self._commit_op(OP_MERGE, merged_size)
            return True
        if self.mode == "no_correction":
            self._commit_op(OP_REJECT, merged_size)
            return False
        return self._resolve_conflict(tracklet_a, tracklet_b, score, witness, merged_size)

    def _merge_persons(self, tracklet_a, tracklet_b, score):
        vertices, persons, scheduler = self.vertices, self.persons, self.scheduler
        small_id, large_id = tracklet_a, tracklet_b
        small_person = persons[vertices[small_id].person_id]
        large_person = persons[vertices[large_id].person_id]
        if small_person.size > large_person.size:          # union by size: relabel the smaller
            small_id, large_id = large_id, small_id
            small_person, large_person = large_person, small_person
        scheduler.send(small_id, REROOT, (NO_PARENT, 0.0))    # small_id becomes its tree's root
        scheduler.run()
        small_vertex = vertices[small_id]
        small_vertex.parent_id, small_vertex.parent_edge_score = large_id, score
        vertices[large_id].child_ids.append(small_id)
        scheduler.record_local_work(large_id)
        if self.emit_tree_deltas:
            self._pending_deltas.append((DELTA_TREE_EDGE_ADDED, small_id, large_id, score))
        scheduler.send(small_id, RELABEL, large_person.person_id)
        scheduler.run()
        # timsort: linear merge of two sorted runs, in C
        large_person.visits = sorted(small_person.visits + large_person.visits)
        large_person.size += small_person.size
        if large_person.site_tag != small_person.site_tag:
            large_person.site_tag = -1
        del persons[small_person.person_id]

    def _weakest_tree_edge_on_path(self, start_id, end_id):
        """Weakest tree edge on the start..end path (same tree).
        Returns (score, child_id) or None."""
        if start_id == end_id:
            return None
        scheduler = self.scheduler
        start_climb = [[], None]
        scheduler.send(start_id, CLIMB, start_climb)
        scheduler.run()
        start_path = start_climb[0]
        position_on_start_path = {vertex_id: i for i, vertex_id in enumerate(start_path)}
        end_climb = [[], position_on_start_path]          # stop at the first common vertex
        scheduler.send(end_id, CLIMB, end_climb)
        scheduler.run()
        end_path = end_climb[0]
        lowest_common_ancestor = end_path[-1]
        vertices = self.vertices
        weakest = None
        for vertex_id in start_path[:position_on_start_path[lowest_common_ancestor]]:
            edge_score = vertices[vertex_id].parent_edge_score
            if weakest is None or edge_score < weakest[0]:
                weakest = (edge_score, vertex_id)
        for vertex_id in end_path[:-1]:
            edge_score = vertices[vertex_id].parent_edge_score
            if weakest is None or edge_score < weakest[0]:
                weakest = (edge_score, vertex_id)
        return weakest

    def _resolve_conflict(self, tracklet_a, tracklet_b, score, witness, merged_size):
        """Fig. 2 case. Cut the weakest match on the cycle the new match would close
        through the conflicting pair, until the merge becomes feasible or the new match
        is itself the weakest (then roll back everything)."""
        vertices, persons = self.vertices, self.persons
        snapshot = self._snapshot((vertices[tracklet_a].person_id, vertices[tracklet_b].person_id))
        n_cuts = 0
        while True:
            witness_a, witness_b = witness
            if vertices[witness_a].person_id != vertices[tracklet_a].person_id:
                witness_a, witness_b = witness_b, witness_a
            if (vertices[witness_a].person_id != vertices[tracklet_a].person_id
                    or vertices[witness_b].person_id != vertices[tracklet_b].person_id):
                raise AssertionError("conflict witness does not cross the two persons")
            weakest_on_side_a = self._weakest_tree_edge_on_path(witness_a, tracklet_a)
            weakest_on_side_b = self._weakest_tree_edge_on_path(tracklet_b, witness_b)
            candidate_edges = [edge for edge in (weakest_on_side_a, weakest_on_side_b)
                               if edge is not None]
            weakest = min(candidate_edges) if candidate_edges else None
            if weakest is None or weakest[0] >= score:
                self._rollback(snapshot)
                self._commit_op(OP_CONFLICT_REJECTED, merged_size, n_cuts)
                return False
            self._cut_tree_edge(weakest[1])
            n_cuts += 1
            person_a = persons[vertices[tracklet_a].person_id]
            person_b = persons[vertices[tracklet_b].person_id]
            if person_a is person_b:          # cannot happen (backups stay inside a person)
                self._commit_op(OP_CONFLICT_RESOLVED, merged_size, n_cuts)
                return True
            feasible, witness = self.checker.can_merge(person_a.visits, person_b.visits)
            if feasible:
                self._merge_persons(tracklet_a, tracklet_b, score)
                self._commit_op(OP_CONFLICT_RESOLVED, merged_size, n_cuts)
                return True

    def _cut_tree_edge(self, child_id) -> bool:
        """Delete tree edge child -> parent(child). Mark the subtree below it, look for the
        best backup link back to the rest of the person; else split the person.
        Returns True if the person stayed whole via a backup link."""
        vertices, persons, scheduler = self.vertices, self.persons, self.scheduler
        child = vertices[child_id]
        parent_id = child.parent_id
        vertices[parent_id].child_ids.remove(child_id)
        scheduler.record_local_work(parent_id)
        child.parent_id, child.parent_edge_score = NO_PARENT, 0.0
        removal_index = len(self._pending_deltas)
        if self.emit_tree_deltas:
            self._pending_deltas.append((DELTA_TREE_EDGE_REMOVED, child_id, parent_id))
        if self.mode == "component":
            return self._rebuild_person_tree(child_id)
        person = persons[child.person_id]
        subtree = []
        scheduler.send(child_id, MARK, subtree)
        scheduler.run()
        subtree_set = set(subtree)
        backup_search = [subtree_set, None]
        for vertex_id in subtree:
            scheduler.send(vertex_id, CHECK_BACKUP, backup_search)
        scheduler.run()
        best_backup = backup_search[1]
        if best_backup is not None and best_backup[2] >= self.tau_accept:
            inside_id, outside_id, backup_score = best_backup
            del vertices[inside_id].backup_links[outside_id], vertices[outside_id].backup_links[inside_id]
            scheduler.send(inside_id, REROOT, (NO_PARENT, 0.0))
            scheduler.run()
            inside = vertices[inside_id]
            inside.parent_id, inside.parent_edge_score = outside_id, backup_score
            vertices[outside_id].child_ids.append(inside_id)
            scheduler.record_local_work(outside_id)
            if self.emit_tree_deltas:          # announce the replacement before the cut
                self._pending_deltas.insert(
                    removal_index, (DELTA_TREE_EDGE_ADDED, inside_id, outside_id, backup_score))
            return True
        # split: relabel the smaller side
        subtree_visits, rest_visits = [], []
        for visit in person.visits:
            (subtree_visits if visit[VISIT_TRACKLET] in subtree_set else rest_visits).append(visit)
        new_person_id = self._allocate_person_id()
        subtree_size = len(subtree)
        if subtree_size <= person.size - subtree_size:
            persons[new_person_id] = Person(new_person_id, child_id, subtree_size, subtree_visits,
                                            person.site_tag)
            person.size -= subtree_size
            person.visits = rest_visits
            scheduler.send(child_id, RELABEL, new_person_id)
        else:
            rest_root_id = person.root_id
            persons[new_person_id] = Person(new_person_id, rest_root_id, person.size - subtree_size,
                                            rest_visits, person.site_tag)
            person.root_id, person.size, person.visits = child_id, subtree_size, subtree_visits
            scheduler.send(rest_root_id, RELABEL, new_person_id)
        scheduler.run()
        return False

    # ---------------------------------------------------------------- B2 baseline
    def _rebuild_person_tree(self, child_id) -> bool:
        """B2: BFS the whole person, rebuild its maximum spanning forest from all its
        accepted matches (Kruskal), split into components, relabel."""
        vertices, persons, scheduler = self.vertices, self.persons, self.scheduler
        person = persons[vertices[child_id].person_id]
        members = []
        scheduler.send(person.root_id, MARK, members)
        scheduler.send(child_id, MARK, members)
        scheduler.run()
        accepted_matches = []
        for member_id in members:
            member = vertices[member_id]
            if member.parent_id != NO_PARENT:
                accepted_matches.append((member.parent_edge_score, member_id, member.parent_id))
            if member.backup_links:
                for partner_id, score in member.backup_links.items():
                    if member_id < partner_id:
                        accepted_matches.append((score, member_id, partner_id))
        accepted_matches.sort(reverse=True)
        union_parent = {member_id: member_id for member_id in members}

        def find_root(vertex_id):
            while union_parent[vertex_id] != vertex_id:
                union_parent[vertex_id] = union_parent[union_parent[vertex_id]]
                vertex_id = union_parent[vertex_id]
            return vertex_id

        tree_neighbours = {member_id: [] for member_id in members}
        backup_matches = []
        for score, tracklet_x, tracklet_y in accepted_matches:
            root_x, root_y = find_root(tracklet_x), find_root(tracklet_y)
            if root_x != root_y:
                union_parent[root_x] = root_y
                tree_neighbours[tracklet_x].append((tracklet_y, score))
                tree_neighbours[tracklet_y].append((tracklet_x, score))
            else:
                backup_matches.append((tracklet_x, tracklet_y, score))
        scheduler.n_messages += 2 * len(members)          # rebuild pass over every member
        for member_id in members:
            member = vertices[member_id]
            member.parent_id, member.parent_edge_score = NO_PARENT, 0.0
            member.child_ids, member.backup_links = [], None
        for tracklet_x, tracklet_y, score in backup_matches:
            for owner_id, partner_id in ((tracklet_x, tracklet_y), (tracklet_y, tracklet_x)):
                if vertices[owner_id].backup_links is None:
                    vertices[owner_id].backup_links = {}
                vertices[owner_id].backup_links[partner_id] = score
        components = []
        visited = set()
        for member_id in members:
            if member_id in visited:
                continue
            visited.add(member_id)
            component, stack = [member_id], [member_id]
            while stack:
                current_id = stack.pop()
                for neighbour_id, score in tree_neighbours[current_id]:
                    if neighbour_id not in visited:
                        visited.add(neighbour_id)
                        neighbour = vertices[neighbour_id]
                        neighbour.parent_id, neighbour.parent_edge_score = current_id, score
                        vertices[current_id].child_ids.append(neighbour_id)
                        component.append(neighbour_id)
                        stack.append(neighbour_id)
            components.append(component)
        components.sort(key=len, reverse=True)
        visit_of_tracklet = {visit[VISIT_TRACKLET]: visit for visit in person.visits}
        largest_component = components[0]
        person.root_id, person.size = largest_component[0], len(largest_component)
        person.visits = sorted(visit_of_tracklet[tracklet_id] for tracklet_id in largest_component)
        for component in components[1:]:
            new_person_id = self._allocate_person_id()
            persons[new_person_id] = Person(
                new_person_id, component[0], len(component),
                sorted(visit_of_tracklet[tracklet_id] for tracklet_id in component), person.site_tag)
            for tracklet_id in component:
                if tracklet_id not in self._original_person_of:
                    self._original_person_of[tracklet_id] = vertices[tracklet_id].person_id
                vertices[tracklet_id].person_id = new_person_id
        return len(components) == 1

    # ---------------------------------------------------------------- rollback
    def _snapshot(self, person_ids):
        vertices, persons = self.vertices, self.persons
        vertex_states, person_states = {}, {}
        for person_id in set(person_ids):
            person = persons[person_id]
            person_states[person_id] = (person.root_id, person.size, person.visits, person.site_tag)
            for visit in person.visits:
                vertex = vertices[visit[VISIT_TRACKLET]]
                vertex_states[vertex.tracklet_id] = (
                    vertex.person_id, vertex.parent_id, vertex.parent_edge_score,
                    list(vertex.child_ids),
                    dict(vertex.backup_links) if vertex.backup_links else None)
        return vertex_states, person_states

    def _rollback(self, snapshot):
        vertex_states, person_states = snapshot
        vertices, persons = self.vertices, self.persons
        for person_id in self._new_person_ids:
            persons.pop(person_id, None)
        for tracklet_id, (person_id, parent_id, parent_edge_score, child_ids,
                          backup_links) in vertex_states.items():
            vertex = vertices[tracklet_id]
            vertex.person_id, vertex.parent_id, vertex.parent_edge_score = (
                person_id, parent_id, parent_edge_score)
            vertex.child_ids, vertex.backup_links = child_ids, backup_links
        for person_id, (root_id, size, visits, site_tag) in person_states.items():
            persons[person_id] = Person(person_id, root_id, size, visits, site_tag)
        self._pending_deltas = []
        self._original_person_of = {}

    # ================================================================ inspection
    def partition(self) -> dict:
        """tracklet_id -> person_id."""
        return {tracklet_id: vertex.person_id for tracklet_id, vertex in self.vertices.items()}

    def depths(self) -> list:
        depth_of = {}
        vertices = self.vertices
        for tracklet_id in vertices:
            path = []
            current_id = tracklet_id
            while current_id not in depth_of and vertices[current_id].parent_id != NO_PARENT:
                path.append(current_id)
                current_id = vertices[current_id].parent_id
            depth = depth_of.get(current_id, 0)
            depth_of[current_id] = depth
            for vertex_id in reversed(path):
                depth += 1
                depth_of[vertex_id] = depth
        return list(depth_of.values())

    def n_violations(self) -> int:
        return sum(0 if self.checker.feasible(person.visits) else 1
                   for person in self.persons.values())

    def check_invariants(self):
        """I1 (one tree per person), I2 (labels = root's person), I3 (feasible),
        plus: backup links stay inside a person and are symmetric."""
        vertices, persons = self.vertices, self.persons
        n_covered = 0
        for person_id, person in persons.items():
            root_id = person.root_id
            assert vertices[root_id].parent_id == NO_PARENT, f"root {root_id} has a parent"
            assert vertices[root_id].person_id == person_id
            stack, n_members = [root_id], 0
            visited = set()
            while stack:
                vertex_id = stack.pop()
                assert vertex_id not in visited, "cycle"
                visited.add(vertex_id)
                n_members += 1
                vertex = vertices[vertex_id]
                assert vertex.person_id == person_id, f"I2 broken at {vertex_id}"
                for child_id in vertex.child_ids:
                    assert vertices[child_id].parent_id == vertex_id, "child/parent mismatch"
                    stack.append(child_id)
                if vertex.backup_links:
                    for partner_id, score in vertex.backup_links.items():
                        assert vertices[partner_id].person_id == person_id, "backup crosses persons"
                        assert vertices[partner_id].backup_links[vertex_id] == score
            assert n_members == person.size == len(person.visits), \
                f"I1 size mismatch person_id={person_id}"
            assert {visit[VISIT_TRACKLET] for visit in person.visits} == visited
            assert self.checker.feasible(person.visits), f"I3 violated person_id={person_id}"
            n_covered += n_members
        assert n_covered == len(vertices), "vertices not covered by persons"
