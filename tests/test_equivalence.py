"""Two-level vs centralized (plan Section 13.5) and invariant I4."""
import numpy as np
import pandas as pd
import pytest

from argus.data.tracklets import boxes_to_tracklets, estimate_topology
from argus.eval.metrics import agreement
from argus.gen.campus import GenConfig, generate
from argus.sync.sim import run_single, run_two_level

ENGINE_CONFIG = dict(window_slack=300.0, tau_keep=0.5, tau_accept=0.6, max_walk_time=600.0)


def _same_partition(partition_a, partition_b):
    # identical up to renaming of labels
    label_b_of_label_a = {}
    for tracklet_id, label_a in partition_a.items():
        if label_b_of_label_a.setdefault(label_a, partition_b[tracklet_id]) != partition_b[tracklet_id]:
            return False
    return len(set(label_b_of_label_a.values())) == len(label_b_of_label_a)


def _assert_site_persons_inside_hub_persons(site_partitions, hub_partition):
    """Invariant I4."""
    for site_partition in site_partitions:
        hub_persons_of_site_person = {}
        for tracklet_id, site_person_id in site_partition.items():
            hub_persons_of_site_person.setdefault(site_person_id, set()).add(hub_partition[tracklet_id])
        assert all(len(hub_persons) == 1 for hub_persons in hub_persons_of_site_person.values()), \
            "a site person was split at the hub"


@pytest.mark.parametrize("ownership", ["a", "b"])
def test_two_level_equals_centralized_without_cross_site_candidates(ownership):
    # buildings far apart (> max_walk_time) and nobody crosses -> no cross-site matches exist
    config = GenConfig(n_buildings=3, cams_per_building=5, n_people=120, duration=1200,
                       p_cross=0.0, walk_out=(900.0, 1200.0), p_revise=0.2, seed=7)
    stream = generate(config)
    central_partition = run_single(stream, ENGINE_CONFIG)[0].engine.partition()
    two_level = run_two_level(stream, dict(ENGINE_CONFIG, ownership=ownership))
    hub_partition = two_level["hub"].engine.partition()
    assert _same_partition(hub_partition, central_partition)
    assert agreement(hub_partition, central_partition) == 1.0
    # and every site's partition equals the hub's restricted to that site
    for site in two_level["sites"]:
        site_partition = site.node.engine.partition()
        assert _same_partition(site_partition,
                               {tracklet_id: hub_partition[tracklet_id] for tracklet_id in site_partition})


@pytest.mark.parametrize("ownership,seed,radius",
                         [("a", 3, 0), ("b", 3, 0), ("b", 4, 0), ("b", 5, 30.0)])
def test_invariant_I4_site_persons_inside_global_persons(ownership, seed, radius):
    config = GenConfig(n_buildings=3, cams_per_building=5, n_people=150, duration=1200,
                       p_cross=0.4, p_revise=0.3, lookalike_frac=0.3, sigma=0.7, seed=seed)
    stream = generate(config)
    stream = stream.with_topology(stream.topology.with_border_radius(radius))
    two_level = run_two_level(stream, dict(ENGINE_CONFIG, ownership=ownership))
    hub_engine = two_level["hub"].engine
    hub_engine.check_invariants()                      # I1-I3 at the hub, incl. constraints
    for site in two_level["sites"]:
        site.node.engine.check_invariants()
    _assert_site_persons_inside_hub_persons(
        [site.node.engine.partition() for site in two_level["sites"]], hub_engine.partition())
    assert hub_engine.n_violations() == 0


def test_real_pipeline_on_synthetic_boxes():
    rows = []
    # person 0: cam 0 frames 0-9, cam 1 frames 40-49; person 1 on cam 1 frames 0-5 and 30-35
    for frame in range(10):
        rows.append((0, frame, 0))
    for frame in range(40, 50):
        rows.append((1, frame, 0))
    for frame in list(range(6)) + list(range(30, 36)):
        rows.append((1, frame, 1))
    boxes = pd.DataFrame(rows, columns=["camera_id", "frame", "person_id"])
    boxes[["x1", "y1", "x2", "y2"]] = [0, 0, 10, 20]
    tracklet_table = boxes_to_tracklets(boxes, fps=10.0, gap_s=1.0)
    assert len(tracklet_table) == 4                    # person 1 split by the 2.4 s gap
    topology = estimate_topology(tracklet_table, 2, [0, 0], margin=0.1)
    assert np.isclose(topology.walk_time[0, 1], (4.0 - 1.0) * 0.9)  # cam0 ends t=1.0, cam1 starts t=4.0


def test_multiprocess_two_level_split_requests(tmp_path):
    """Real processes + pyzmq: SplitRequests flow hub -> site and the run terminates
    with I4 intact (site persons inside hub persons)."""
    from argus.sync.mp import run_mp
    config = GenConfig(n_buildings=3, cams_per_building=4, n_people=120, duration=900,
                       p_cross=0.4, p_revise=0.3, lookalike_frac=0.3, sigma=0.7, seed=11)
    stream = generate(config)
    path = str(tmp_path / "stream.parquet")
    stream.save(path)
    run = run_mp(path, stream.topology, dict(ENGINE_CONFIG, ownership="b"), mode="two_level",
                 timeout=300)
    hub_partition = run["hub"]["partition"]
    assert run["hub"]["n_split_requests"] > 0, "scenario should exercise SplitRequests"
    assert len(hub_partition) == stream.n_tracklets
    _assert_site_persons_inside_hub_persons([site["partition"] for site in run["sites"]],
                                            hub_partition)
