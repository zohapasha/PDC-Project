"""The per-site vectorised matcher returns exactly the candidates of the original
per-camera reference implementation (tests/_ref_matcher.py)."""
import numpy as np
import pytest

from argus.core.matcher import Matcher
from argus.gen.campus import GenConfig, generate
from tests._ref_matcher import Matcher as RefMatcher


@pytest.mark.parametrize("cross,radius", [(False, 0.0), (True, 0.0), (True, 30.0)])
def test_matcher_parity(cross, radius):
    s = generate(GenConfig(n_buildings=3, cams_per_building=5, n_people=150, duration=900,
                           overlap_frac=0.4, p_revise=0.3, seed=5))
    topo = s.topo.with_border_radius(radius)
    cams = np.nonzero(topo.border)[0].tolist() if cross else None
    new = Matcher(topo, 300.0, 0.3, cams=cams, cross_site_only=cross)
    ref = RefMatcher(topo, 300.0, 0.3, cams=cams, cross_site_only=cross)
    n_cmp = 0
    for i in range(len(s.time)):
        t = int(s.tid[i]); c = int(s.cam[t]); v = s.vec[i]
        if s.etype[i] == 0:
            if c not in new.cams:
                continue
            a = new.candidates(c, s.ts[t], v)
            b = ref.candidates(c, s.ts[t], v)
            new.add(t, c, s.ts[t], s.te[t], v)
            ref.add(t, c, s.ts[t], s.te[t], v)
        else:
            if not new.update_vec(t, v):
                assert not ref.update_vec(t, v)
                continue
            ref.update_vec(t, v)
            a = new.candidates_fwd(t, v)
            b = ref.candidates_fwd(t, v)
        da, db = dict(a), dict(b)
        assert set(da) == set(db)
        assert all(abs(da[k] - db[k]) < 1e-5 for k in da)
        n_cmp += len(a)
        if i % 200 == 0:
            new.prune(s.time[i], 120.0)
            ref.prune(s.time[i], 120.0)
    assert n_cmp > 100
