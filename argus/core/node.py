"""IdentityNode = Matcher + IdentityEngine. Used for a single (centralized) instance,
for each site, and (with cross-site matching only) for the hub."""
from __future__ import annotations

from ..common.topology import Topology
from ..common.types import INF
from .engine import IdentityEngine
from .matcher import Matcher

_PRUNE_EVERY = 256       # prune the matcher every this many matched tracklets (power of 2)


class IdentityNode:
    def __init__(self, topology: Topology, engine_config: dict, match_camera_ids=None,
                 cross_site_only=False, mode: str = "full", emit_tree_deltas: bool = False,
                 emit_belongs_to: bool = False, log_ops: bool = True):
        self.engine = IdentityEngine(topology, engine_config["tau_accept"],
                                     engine_config["tau_keep"], mode=mode,
                                     emit_tree_deltas=emit_tree_deltas,
                                     emit_belongs_to=emit_belongs_to, log_ops=log_ops)
        self.matcher = Matcher(topology, engine_config["window_slack"], engine_config["tau_keep"],
                               camera_ids=match_camera_ids, cross_site_only=cross_site_only,
                               max_walk_time=engine_config.get("max_walk_time", 600.0))
        self.is_site_owned_match = None          # hub: predicate for site-owned matches
        self.max_tracklet_duration = 1.0
        self._n_matched_tracklets = 0

    def on_tracklet(self, tracklet_id, camera_id, start_time, end_time, vector=None, site_tag=-1):
        engine, matcher = self.engine, self.matcher
        engine.add_vertex(tracklet_id, camera_id, start_time, end_time, site_tag)
        if end_time - start_time > self.max_tracklet_duration:
            self.max_tracklet_duration = end_time - start_time
        if vector is None or camera_id not in matcher.camera_ids:
            return
        candidates = matcher.candidates(camera_id, start_time, vector)
        matcher.add(tracklet_id, camera_id, start_time, end_time, vector)
        for other_id, score in candidates:            # strongest first
            engine.add_match(tracklet_id, other_id, score)
        self._n_matched_tracklets += 1
        if self._n_matched_tracklets & (_PRUNE_EVERY - 1) == 0:
            matcher.prune(end_time, self.max_tracklet_duration)

    def on_revise(self, tracklet_id, vector):
        """The tracklet's final appearance vector replaces a provisional one: re-score its
        stored matches (may trigger evidence-driven deletions) and look for new ones."""
        engine, matcher = self.engine, self.matcher
        if not matcher.update_vector(tracklet_id, vector):
            return
        stored_scores = engine.match_scores.get(tracklet_id)
        if stored_scores is None:
            return
        for other_id, old_score in list(stored_scores.items()):
            if old_score == INF or (self.is_site_owned_match is not None
                                    and self.is_site_owned_match(tracklet_id, other_id)):
                continue                   # site-owned link at the hub: the site scores it
            other_vector = matcher.get_vector(other_id)
            if other_vector is not None:
                engine.update_match(tracklet_id, other_id, float(other_vector @ vector))
        camera_id, start_time, _ = matcher.metadata(tracklet_id)
        exclude = set(engine.match_scores[tracklet_id])
        exclude.add(tracklet_id)
        new_candidates = (matcher.candidates(camera_id, start_time, vector, exclude=exclude)
                          + matcher.forward_candidates(tracklet_id, vector, exclude=exclude))
        new_candidates.sort(key=lambda candidate: -candidate[1])
        for other_id, score in new_candidates:
            engine.add_match(tracklet_id, other_id, score)
