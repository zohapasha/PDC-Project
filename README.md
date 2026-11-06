# ARGUS — Parallel Vertex-Centric Synchronization Toward a Central Graph

Implementation and experiment harness for the ARGUS paper (FAST-NUCES Lahore:
Ammar Bin Mudassar, Muhammad Ammar Hussain, Zoha Pasha).

Every tracklet is a vertex. Accepted matches inside a person form one tree, with spare
accepted matches kept as *backup links*. Merges, conflict resolution and deletions are
vertex programs that touch only the persons involved. Each site runs this engine on its
own cameras and streams only the **changes** to a hub. The hub runs the **same engine**
to build the central (global) graph.

## Install

```bash
uv venv .venv --python 3.12          # or: python -m venv .venv
uv pip install -r requirements.txt    # or: pip install -r requirements.txt
python -m pytest -q                   # 17 tests, ~10k Hypothesis cases, ~1 min
```

## Reproduce every number and figure

```bash
bash scripts/run_all.sh               # all experiments (5 seeds) + figures, ~2 h on 12 cores
# or one experiment at a time:
python scripts/run_experiment.py configs/e1_accuracy.yaml --seeds 0-4 --jobs 5 --fresh
python scripts/make_figures.py        # reads results/*.csv only -> results/figures/*.pdf, results/tables/
python scripts/calibrate.py           # similarity histograms (add --real <train.parquet>)
```

Each result row stores the seed, the config hash, the git commit (`nogit` until the
repo has a commit), the CPU, core counts, RAM, and the Python version.
Timed experiments (E2, E3, E4) always run seeds one after another; close other apps
while they run. Untimed ones (E1, E5, E6, E7) run seeds in parallel.

| Config | Exp. | Claim | Output |
|---|---|---|---|
| `e1_accuracy.yaml`, `e1_accuracy_hard.yaml` | E1 accuracy | C1 | `tables/e1_*.{md,tex}` |
| `e2_incremental.yaml` | E2 incremental vs full recompute | C2 | `fig_e2_incremental` |
| `e3_corrections.yaml` | E3 cost of corrections vs B2 | C2 | `fig_e3_corrections`, `tables/e3_summary` |
| `e4_weak.yaml`, `e4_strong.yaml` | E4 scaling with sites (multi-process) | C3 | `fig_e4_{weak,strong}` |
| `e5_bytes.yaml` | E5 bytes to hub vs B5/B6 | C3 | `fig_e5_bytes` |
| `e6_noise.yaml` | E6 re-ID noise sweep | C1, C2 | `fig_e6_noise` |
| `e7_cross.yaml` | E7 cross-site movement, rule (a) vs centralized | C3, C4 | `fig_e7_cross` |

## Headline results (synthetic, 5 seeds, mean; 12-core / 32 GB Windows machine, 2026-11-06)

| Exp | Result |
|---|---|
| E1 (medium, ~46k tracklets) | Ours pairwise F1 **0.941**, IDF1 0.926, **0 violations**. B1 offline 0.943, B3 no-correction 0.928, B4 no-constraints 0.896 with ~367 violations. Agreement with B1: 0.989. |
| E1 hard | Ours 0.891 vs B1 0.900, B3 0.868, B4 0.778. **Two-level only 0.809.** |
| E2 | Per-update latency at 200k tracklets: **2.1 ms mean / 6.5 ms p99**, nearly flat with graph size. One full recompute (B1) takes 14.2 s and grows super-linearly. |
| E3 | Corrections vs B2 (whole-person rebuild): p50 0.23 vs 0.71 ms, p99 3.5 vs 22.5 ms, 3.4× fewer vertex messages, 16 vs 24 vertices touched, identical F1. About 70% of deleted tree matches are repaired via a backup link. |
| E4 weak | Max throughput 2.6k → 7.6k ev/s from 1 to 4 sites, then flat at 8 sites because the hub saturates (CPU ≈ 1.0). B5 centralized *drops* from 3.8k to 0.9k ev/s. Fixed-load p99 for ours stays at about 2 ms; B5 reaches 30 s at 8 sites. |
| E4 strong | 1.2k → 5.9k ev/s (1 → 4 sites), flat after that. B5 stays at about 1.5k ev/s. p99 for ours is about 2 ms vs 1.7–5 s for B5. |
| E5 | Bytes to hub: ours **8.8 MiB**, B5 25.8 MiB, B6 22–443 MiB (T = 30 s … 1 s), with 0.5–15 s staleness for B6. |
| E6 | F1 stays at ≥ 0.92 up to σ = 0.7, then collapses for all methods (by σ = 0.9). Ours ≥ B3 ≥ B4 throughout. |
| E7 | Two-level agreement with centralized falls from 0.98 to 0.94 as p_cross goes from 0.05 to 0.5. Two-level F1 is 0.91–0.92 vs centralized 0.92–0.95. |

Limits to report: the hub saturates beyond about 4 sites, and the two-level design loses
accuracy compared with centralized (worse with heavy cross-site traffic or noise; see E1-hard and E7).

## Layout

```
argus/common/   topology (shortest walking-time metric, overlap groups, sites, border cams),
                constraints (linear feasibility check), types
argus/core/     engine.py (merge / conflict resolution / cut + backup reconnection / B2, B3 modes),
                vertex.py (vertex-program scheduler + message counters), matcher.py (windowed
                cosine candidates on per-camera ring buffers), node.py (matcher + engine)
argus/gen/      campus.py synthetic campus generator -> Parquet stream (+ topology.json)
argus/data/     wildtrack.py, mtmmc.py, tracklets.py (GT boxes -> tracklets, topology
                estimation, threshold choice), features.py (OSNet on Colab)
argus/sync/     protocol.py (msgpack messages), nodes.py (Site, Hub, CentralHub),
                sim.py (deterministic in-process drivers), mp.py (multi-process pyzmq runtime)
argus/baselines/offline.py  B1 full recompute (also the offline reference), B4 no constraints
argus/eval/     metrics (pairwise, B-cubed, IDF1, violations, agreement), experiments E1–E7,
                harness, sysinfo
argus/sink/     neo4j_sink.py (optional, batched UNWIND; off for all timed runs)
```

Baselines: **B1** full recompute (Kruskal over all stored matches), **B2** `mode="component"`
(on any cut, rebuild the whole person), **B3** `mode="no_correction"`, **B4** connected
components without constraints, **B5** `CentralHub` (sites forward everything with vectors),
**B6** `sim.b6_bytes` (periodic full-table sync).

## Where this differs from the written plan (update the Methodology section)

1. **Hub protocol: site tree matches are +∞ edges at the hub.** Sites send
   `TrackletRecord` (every tracklet, no vector), `BorderTracklet`/`BorderRevise` (vectors,
   border cameras only), and `EdgeAdd`/`EdgeDel` for every change to their match trees. The hub
   runs the unmodified engine, with every site tree match added at weight +∞. Consequences:
   - The hub can never cut a site-owned match, so it only ever removes the weakest
     *cross-site* match. That is ownership rule (a), and it falls out of the weights.
   - Every site person is always inside one global person. That is invariant I4, and it holds
     by construction (tested).
   - A site merge costs one `EdgeAdd` rather than |A| `BelongsTo` messages.

   `BelongsTo` deltas still exist (`emit_belongs=True`) and feed the Neo4j sink.
   With 1 site the two-level result equals the centralized result exactly, and also with any
   number of sites when no cross-site candidate exists (`tests/test_equivalence.py`).
2. **The constraint lemma needed a fix.** "Check each pair of consecutive visits" is wrong
   once overlap groups exist. Take cam 0 then cam 1 (which overlaps cam 0) ending later, then
   cam 0 again while the first visit is still running. The latest-ending earlier visit is on
   cam 1, so that check misses the same-camera overlap. The correct linear check compares each
   visit with (i) the latest-ending earlier visit and (ii) the latest-ending earlier visit on
   the same camera. The proof is in `argus/common/constraints.py`. The check is tested against
   an all-pairs brute force on 3,000 random cases. Hypothesis found a real engine failure
   caused by the original version.
3. **Conflict resolution is atomic.** If the new match turns out to be the weakest after some
   cuts, every cut is rolled back from a snapshot of the two persons involved. The snapshot
   costs O(|A|+|B|) copies. It counts toward latency but not toward vertex messages.
4. **The smaller side is relabelled on a split** (union-by-size in reverse). The deltas are
   still exactly that side's tracklets.
5. **Matching window.** A candidate must satisfy
   `t.ts − D(c′,c) − W ≤ u.te ≤ t.ts − D(c′,c)`, so W is slack *on top of* the minimum
   walking time. With the plan's `W = 120 s` taken as an absolute window, people could never
   be matched between buildings 60–300 s apart. Defaults: `W = 300 s`, `τ_keep = 0.5`,
   `τ_accept = 0.6`, `D_max = 600 s`.
6. **Rejected matches are not retried** when a later split would make them feasible. The
   offline reference (B1) does retry them, which accounts for the remaining disagreement
   (agreement ≈ 0.99).
7. **Synthetic defaults:** σ = 0.5, σ_cam = 0.3, rests with a mean of 120 s,
   `p_frag = 0.1`. A true pair scores ≈ 0.75 on average; a random pair ≈ 0.
8. **Two runtimes.** `sim.py` delivers site messages to the hub immediately and in stream
   order, so it is deterministic; E1, E5 and E7 use it. `mp.py` is the real multi-process
   system used for E4. Its cross-site interleaving is nondeterministic, so the hub result
   varies slightly between runs (agreement with `sim` ≈ 0.96).
9. **Not implemented:** option (b) `SplitRequest`, `tc netem` network delay, and E8 (Neo4j
   throughput). The sink code exists but has not been benchmarked.
10. **IDF1 with GT boxes:** every frame belongs to exactly one tracklet, so
    IDF1 = IDTP / total frames, with IDTP taken from the optimal assignment between
    predicted and GT identities. Synthetic "frames" are duration × 10 fps. For very large
    instances, the assignment keeps only the 8 heaviest predicted IDs per GT ID, which
    gives a lower bound.

## Real data (written, NOT yet run — datasets were not available here)

1. Request MTMMC access. Download WILDTRACK.
2. On Colab: build the tracklet table and extract OSNet features
   (`argus/data/features.py` docstring). Save the `.npy` files.
3. `python scripts/prepare_real.py wildtrack --root ... --vec ... --prov ... --out data/real/wildtrack`
   This splits train/test in time, estimates walking times and overlap groups from train,
   and picks `τ_accept` on train.
4. Make a config with `experiment: e1`, `stream: data/real/wildtrack_test.parquet`, and
   `engine: {tau_accept: <chosen>}`. Run it like the others.
5. `python scripts/calibrate.py --real data/real/wildtrack_train.parquet`. Adjust σ until
   the synthetic histograms roughly match the real ones, then re-run the synthetic experiments.

`mtmmc.load_boxes` *assumes* a MOT-format `gt/gt.txt` per camera with global IDs. Check
this against the actual release.

## Limitations to state in the paper

Python implementation on one machine. GT tracklets, not tracker output. The hub processes
metadata for every tracklet, so it becomes the bottleneck as sites are added (see E4).
Greedy online decisions (agreement with offline ≈ 0.99). Synthetic workloads are the main
scaling evidence until the real-data calibration is done.
