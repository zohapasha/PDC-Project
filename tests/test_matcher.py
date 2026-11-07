"""The per-site vectorised matcher returns exactly the candidates of the original
per-camera reference implementation (tests/_ref_matcher.py)."""
import numpy as np
import pytest

from argus.common.types import EVENT_TRACKLET
from argus.core.matcher import Matcher
from argus.gen.campus import GenConfig, generate
from tests._ref_matcher import Matcher as ReferenceMatcher


@pytest.mark.parametrize("cross_site_only,border_radius", [(False, 0.0), (True, 0.0), (True, 30.0)])
def test_matcher_parity(cross_site_only, border_radius):
    stream = generate(GenConfig(n_buildings=3, cams_per_building=5, n_people=150, duration=900,
                                overlap_frac=0.4, p_revise=0.3, seed=5))
    topology = stream.topology.with_border_radius(border_radius)
    camera_ids = np.nonzero(topology.is_border_camera)[0].tolist() if cross_site_only else None
    matcher = Matcher(topology, 300.0, 0.3, camera_ids=camera_ids, cross_site_only=cross_site_only)
    reference = ReferenceMatcher(topology, 300.0, 0.3, camera_ids=camera_ids,
                                 cross_site_only=cross_site_only)
    n_compared = 0
    for i in range(len(stream.event_times)):
        tracklet_id = int(stream.event_tracklet_ids[i])
        camera_id = int(stream.camera_ids[tracklet_id])
        vector = stream.event_vectors[i]
        start_time, end_time = stream.start_times[tracklet_id], stream.end_times[tracklet_id]
        if stream.event_types[i] == EVENT_TRACKLET:
            if camera_id not in matcher.camera_ids:
                continue
            found = matcher.candidates(camera_id, start_time, vector)
            expected = reference.candidates(camera_id, start_time, vector)
            matcher.add(tracklet_id, camera_id, start_time, end_time, vector)
            reference.add(tracklet_id, camera_id, start_time, end_time, vector)
        else:
            if not matcher.update_vector(tracklet_id, vector):
                assert not reference.update_vector(tracklet_id, vector)
                continue
            reference.update_vector(tracklet_id, vector)
            found = matcher.forward_candidates(tracklet_id, vector)
            expected = reference.forward_candidates(tracklet_id, vector)
        found_scores, expected_scores = dict(found), dict(expected)
        assert set(found_scores) == set(expected_scores)
        assert all(abs(found_scores[other_id] - expected_scores[other_id]) < 1e-5
                   for other_id in found_scores)
        n_compared += len(found)
        if i % 200 == 0:
            matcher.prune(stream.event_times[i], 120.0)
            reference.prune(stream.event_times[i], 120.0)
    assert n_compared > 100
