# ARGUS — Parallel Vertex-Centric Synchronization Toward a Central Graph

Implementation and experiment harness for the ARGUS paper (FAST-NUCES Lahore:
Ammar Bin Mudassar, Muhammad Ammar Hussain, Zoha Pasha).

Every tracklet is a vertex. Accepted matches inside a person form one tree, with spare
accepted matches kept as *backup links*. Merges, conflict resolution and deletions are
vertex programs that touch only the persons involved. Each site runs this engine on its
own cameras and streams only the **changes** to a hub. The hub runs the **same engine**
to build the central (global) graph, and sends a `SplitRequest` back when global
evidence shows a site match is wrong.

## Install

```bash
uv venv .venv --python 3.12          # or: python -m venv .venv
uv pip install -r requirements.txt    # or: pip install -r requirements.txt
python -m pytest -q                   # 26 tests, ~12k Hypothesis cases, ~1 min
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
| `e7_cross.yaml` | E7 cross-site movement, option (b) / (a) vs centralized | C3, C4 | `fig_e7_cross` |
| `e8_neo4j.yaml` | E8 Neo4j sink throughput, stored graph = hub partition | C3 | `results/e8_neo4j.csv` |
| `e9_radius.yaml`, `e9_radius_hard.yaml` | E9 hub matching radius: bytes vs accuracy | C1, C3 | `fig_e9_radius*` |
| `real_wildtrack.yaml` | E1 on WILDTRACK (OSNet features) | C1 | `tables/real_wildtrack` |

## Headline results (synthetic, 5 seeds, mean; 12-core / 32 GB Windows machine, 2026-11-06)

Two-level = site engines plus a hub (option b, SplitRequest) unless stated otherwise. Full tables are in `results/tables/`.

| Exp | Result |
|---|---|
| E1 (medium, ~46k tracklets) | Ours pairwise F1 **0.941**, IDF1 0.926, **0 violations**. B1 offline 0.943, B3 no-correction 0.928, B4 no-constraints 0.896 with ~367 violations. Agreement with B1: 0.989. Two-level: option b 0.933, option a 0.921. |
| E1 hard | Ours 0.891 vs B1 0.900, B3 0.868, B4 0.778. Two-level at the minimum-bytes radius: option b 0.827, option a 0.809. E9 shows this gap closing with a larger radius. |
| E2 | Per-update latency at 200k tracklets: **3.7 ms mean / 8.6 ms p99**, growing slowly with graph size. One full recompute (B1) takes **14.3 s** and grows super-linearly. |
| E3 | Corrections vs B2 (whole-person rebuild): p50 **0.18 vs 0.66 ms**, p99 **3.1 vs 21.8 ms**, 3.4× fewer vertex messages (201 vs 679), 16.5 vs 24.2 vertices touched, identical F1 (0.909). About 69% of deleted tree matches are repaired via a backup link. |
| E4 weak (250 people/site) | Max throughput **3.3k → 6.7k → 7.7k ev/s** at 1 → 2 → 4 sites, then 5.9k at 8 sites because the hub saturates (CPU ≈ 1.0). B5 centralized *drops* from 5.4k to 1.2k ev/s. At fixed load (50× real time), p99 for ours stays at **1.7–9 ms**; B5 reaches **20 s** at 8 sites. |
| E4 strong (2000 people, 32 cams) | **1.2k → 2.4k → 4.7k ev/s** (1 → 2 → 4 sites), 4.1k at 8 sites. B5 stays at 1.0–1.6k ev/s. At fixed load, p99 for ours is 2.4–5.3 ms vs 2.6–15 s for B5. SplitRequests are about 1.1–1.4k per run (~10 KB). |
| E5 | Bytes to hub: ours **9.25 MiB** (SplitRequest 0.01 MiB), B5 25.8 MiB, B6 22–443 MiB (T = 30 s … 1 s), with 0.5–15 s staleness for B6. |
| E6 | F1 stays at ≥ 0.92 up to σ = 0.7, then collapses for all methods (by σ = 0.9). Ours ≥ B3 ≥ B4 up to σ = 0.8. |
| E7 | As p_cross goes from 0.05 to 0.5, agreement with centralized is 0.98 → 0.96 for option b and 0.98 → 0.94 for option a. Option b F1 is 0.916–0.943 vs centralized 0.919–0.954. |
| E8 (Neo4j 5, batched UNWIND) | The stream produces 26 BelongsTo deltas/s (94k per run). The sink sustains **17k–40k deltas/s** (batch 500 → 20 000): 670–1500× headroom, flush p99 74–548 ms. The stored graph equals the hub partition in **5/5** runs. |
| E9 (hub matching radius) | Normal: option b 0.933 → 0.941 F1 as bytes go from 9.2 to 26.8 MiB, matching centralized (0.941 at 25.8 MiB). Hard: 0.827 → 0.889 (centralized 0.891); 45 s radius gives 0.886 at 25.6 MiB vs 30.4 MiB centralized; 30 s gives 0.876 at 21.7 MiB. |
| Real: WILDTRACK test (671 tracklets, 125 people) | Ours F1 0.314 / IDF1 0.540, 0 violations; B1 0.315; B3 0.305; B4 0.034 with 12 violations; two-level b 0.316 (agreement 0.994). Low absolute accuracy because association is appearance-only (see below). |

Limits to report: the hub saturates beyond about 4 sites (E4). At the minimum-bytes radius the
two-level design loses accuracy compared with centralized in the hard setting (E1-hard). Buying it
back costs bytes (E9). Absolute accuracy on WILDTRACK is low.

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

1. **Hub protocol: the hub mirrors site matches and can send them back (option b).** Sites send
   `TrackletRecord` (every tracklet, no vector), `BorderTracklet`/`BorderRevise` (vectors,
   only for cameras within `border_radius` of a cross-site entrance) and
   `EdgeAdd(a, b, score)`/`EdgeDel` for every change to their match trees. The hub runs the
   unmodified engine on all of this, with site matches at their **real scores**. So it
   resolves conflicts exactly as a single instance would. When it cuts or rejects a site
   match, it sends `SplitRequest(a, b)`. The site deletes and blocks the match, then
   streams the resulting deltas back. Invariant I4 (every site person lies inside one
   global person) holds whenever no `SplitRequest` is outstanding. Tests cover this for the
   in-process and the multi-process runtime.
   - The site is authoritative for its scores: the hub never re-scores a site match, and the
     site sends score changes as `EdgeAdd`.
   - Option (a) (`engine: {ownership: a}`) mirrors site matches as +∞ edges. The hub then
     never cuts them, and I4 holds by construction. But a site's mistakes (it cannot see a
     person's visits in other buildings) become permanent. In E1-hard this cost
     0.08 pairwise F1. Option (a) is kept as an ablation (E1, E7, E9).
   - With no cross-site candidates, two-level equals centralized exactly under both options
     (`tests/test_equivalence.py`).
   - Hub fast path: a merge of two *site-pure* hub persons (one site, joined only by that
     site's matches) needs no constraint check, because the union is a subset of a feasible
     site person.

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
9. **Not implemented:** `tc netem` network delay (single Windows machine; all processes talk over
   loopback TCP). MTMMC is not run: the team has to request access. The loader assumes a
   MOT-style layout.
10. **IDF1 with GT boxes:** every frame belongs to exactly one tracklet, so
    IDF1 = IDTP / total frames, with IDTP taken from the optimal assignment between
    predicted and GT identities. Synthetic "frames" are duration × 10 fps. For very large
    instances, the assignment keeps only the 8 heaviest predicted IDs per GT ID, which
    gives a lower bound.
11. **Constraint check windowing.** When merging a small person S into a large one L, only
    L's visits within max walking time + longest visit of S's visits can conflict. The check
    runs on that slice. This is exact (proof in `constraints.py`, tested against brute force).

## Real data: WILDTRACK (run), MTMMC (needs access)

```bash
# 1. download + extract (the official zip has >4 GB offsets written without zip64;
#    extract per entry at offset ± 2^32, every file is CRC-checked)
# 2. OSNet-AIN x1.0 trained on all of MSMT17 (cross-domain), CPU is enough (~8 min)
.venv-feat/Scripts/python.exe scripts/extract_features.py wildtrack     --root data/raw/Wildtrack_extract/Wildtrack_dataset --out data/real/wildtrack
# 3. tracklets from GT boxes, temporal split 100 s train / 100 s test, walking times and
#    overlap groups from train, features centred with the train mean, tau chosen on train
python scripts/prepare_real.py wildtrack --root data/raw/Wildtrack_extract/Wildtrack_dataset     --vec data/real/wildtrack_vec.npy --prov data/real/wildtrack_prov.npy     --sites "0,1,2,3;4,5,6" --out data/real/wildtrack          # -> tau_accept = 0.5
python scripts/run_experiment.py configs/real_wildtrack.yaml --seeds 0 --fresh
```

WILDTRACK test split (671 tracklets, 125 people, 7 overlapping cameras split into 2 sites):

| Method | Pairwise P / R / F1 | B³ F1 | IDF1 | Violations | Agreement w/ B1 |
|---|---|---|---|---|---|
| Ours | 0.338 / 0.293 / **0.314** | 0.521 | 0.540 | 0 | 0.884 |
| Ours two-level (option b) | 0.342 / 0.293 / 0.316 | 0.522 | 0.541 | 0 | 0.890 |
| Ours two-level (option a) | 0.331 / 0.291 / 0.310 | 0.519 | 0.539 | 0 | 0.867 |
| B1 offline | 0.340 / 0.294 / 0.315 | 0.520 | 0.536 | 0 | 1.000 |
| B3 no correction | 0.322 / 0.289 / 0.305 | 0.518 | 0.539 | 0 | 0.717 |
| B4 no constraints | 0.018 / 0.438 / 0.034 | 0.488 | 0.398 | 12 | 0.068 |

The relative claims hold on real appearance features:
- ours ≈ the offline optimum;
- corrections help (B3 is lower);
- without constraints, merges chain through the whole crowd (B4: F1 0.03, 12 physical violations);
- two-level ≈ centralized.

**Absolute accuracy is low, and the paper should say why.** WILDTRACK is one plaza seen
by 7 overlapping cameras. Every camera pair overlaps (D = 0, one overlap group), so the
walking-time constraints carry no information beyond "not twice on the same camera". Association
is then appearance-only, across very different viewpoints of a dense crowd. True-match AUC of
the centred OSNet features is 0.83 (mean cosine 0.27 for true pairs vs 0.00 for false pairs).
Multi-view methods on WILDTRACK use ground-plane geometry, which we do not. **MTMMC** is the
dataset that matches the setting (a multi-building campus with real walking times). Request
access, then use `argus/data/mtmmc.py`. It *assumes* a MOT-format `gt/gt.txt` per camera
with global IDs, so check that against the release.

## Limitations to state in the paper

- Python implementation on one machine. Sites and hub are separate processes over ZeroMQ, but there is no real network: we did not use `tc netem` delay or loss.
- GT tracklets, not tracker output.
- The hub processes metadata for every tracklet, so it saturates at about 4 sites (E4; hub CPU ≈ 1.0). Values slightly above 1.0 are BLAS threads.
- Greedy online decisions (agreement with offline ≈ 0.99 normal, 0.98 hard).
- Two-level vs centralized accuracy is a bytes trade-off (E9). At the cheapest radius, the hard setting loses 0.06 F1.
- Real data: only WILDTRACK, which is small (2 × 100 s, 7 overlapping cameras) and appearance-only. MTMMC (multi-building, real walking times) is pending access. Synthetic workloads remain the main scaling evidence.
