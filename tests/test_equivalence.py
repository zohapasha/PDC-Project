"""Two-level vs centralized (plan Section 13.5) and invariant I4."""
import numpy as np
import pandas as pd

from argus.data.tracklets import boxes_to_tracklets, estimate_topology
from argus.eval.metrics import agreement
from argus.gen.campus import GenConfig, generate
from argus.sync.sim import run_single, run_two_level

CFG = dict(W=300.0, tau_keep=0.5, tau_accept=0.6, D_max=600.0)


def _same_partition(a, b):
    # identical up to renaming of labels
    m = {}
    for t, la in a.items():
        if m.setdefault(la, b[t]) != b[t]:
            return False
    return len(set(m.values())) == len(m)


def test_two_level_equals_centralized_without_cross_site_candidates():
    # buildings far apart (> D_max) and nobody crosses -> no cross-site matches exist
    g = GenConfig(n_buildings=3, cams_per_building=5, n_people=120, duration=1200,
                  p_cross=0.0, walk_out=(900.0, 1200.0), p_revise=0.2, seed=7)
    s = generate(g)
    central = run_single(s, CFG)[0].engine.partition()
    tl = run_two_level(s, CFG)
    hub = tl["hub"].eng.partition()
    assert _same_partition(hub, central)
    assert agreement(hub, central) == 1.0
    # and every site's partition equals the hub's restricted to that site
    for site in tl["sites"]:
        sp = site.node.engine.partition()
        assert _same_partition(sp, {t: hub[t] for t in sp})


def test_invariant_I4_site_persons_inside_global_persons():
    g = GenConfig(n_buildings=3, cams_per_building=5, n_people=150, duration=1200,
                  p_cross=0.4, p_revise=0.3, lookalike_frac=0.3, seed=3)
    s = generate(g)
    tl = run_two_level(s, CFG)
    hub_eng = tl["hub"].eng
    hub_eng.check_invariants()                      # I1-I3 at the hub, incl. constraints
    hub = hub_eng.partition()
    for site in tl["sites"]:
        site.node.engine.check_invariants()
        groups = {}
        for t, p in site.node.engine.partition().items():
            groups.setdefault(p, set()).add(hub[t])
        assert all(len(v) == 1 for v in groups.values()), "a site person was split at the hub"
    assert hub_eng.n_violations() == 0


def test_real_pipeline_on_synthetic_boxes():
    rows = []
    # person 0: cam 0 frames 0-9, cam 1 frames 40-49; person 1 on cam 1 frames 0-5 and 30-35
    for f in range(10):
        rows.append((0, f, 0))
    for f in range(40, 50):
        rows.append((1, f, 0))
    for f in list(range(6)) + list(range(30, 36)):
        rows.append((1, f, 1))
    b = pd.DataFrame(rows, columns=["cam", "frame", "pid"])
    b[["x1", "y1", "x2", "y2"]] = [0, 0, 10, 20]
    tr = boxes_to_tracklets(b, fps=10.0, gap_s=1.0)
    assert len(tr) == 4                               # person 1 split by the 2.4 s gap
    topo = estimate_topology(tr, 2, [0, 0], margin=0.1)
    assert np.isclose(topo.D[0, 1], (4.0 - 1.0) * 0.9)  # cam0 ends t=1.0, cam1 starts t=4.0
