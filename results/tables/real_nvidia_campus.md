Campus = test scenes 071–078 combined (8 buildings × 16 cameras = 128 cameras; camera 649 of scene_071 dropped, its video is a broken 262 KB file in the release), 255 people, 12,892 test tracklets. Buildings are disjoint (nobody moves between them), so no camera is a border camera and two-level equals centralized by construction.

### E1 accuracy (4 sites)

| Method | Pairwise P / R / F1 | B³ F1 | IDF1 | Violations | Agreement w/ B1 |
|---|---|---|---|---|---|
| Ours | 0.880 / 0.647 / 0.746 | 0.750 | 0.908 | 0 | 0.934 |
| Ours two-level (option b) | 0.880 / 0.647 / 0.746 | 0.750 | 0.908 | 0 | 0.934 |
| B1 offline | 0.852 / 0.652 / 0.739 | 0.748 | 0.911 | 0 | 1.000 |
| B3 no correction | 0.696 / 0.583 / 0.635 | 0.671 | 0.818 | 0 | 0.754 |
| B4 no constraints | 0.048 / 0.903 / 0.091 | 0.277 | 0.222 | 16 | 0.078 |

### E5 bytes to the hub (4 sites)

| Method | MiB | Staleness (s) |
|---|---|---|
| Ours | 0.78 | 0.0 |
| B5 centralized | 50.79 | 0.0 |
| B6 periodic, T = 1 s | 15.46 | 0.5 |
| B6 periodic, T = 5 s | 3.12 | 2.4 |
| B6 periodic, T = 30 s | 0.55 | 14.9 |

### E4 strong scaling (multi-process, Kaggle 4 vCPU Intel(R) Xeon(R) CPU @ 2.20GHz, 1 BLAS thread per process, 3 seeds, mean ± std)

| Sites | Load | Ours ev/s | B5 ev/s | Ours e2e p50 / p99 (ms) | B5 e2e p50 / p99 (ms) | Ours hub CPU |
|---|---|---|---|---|---|---|
| 1 | max | 932 ± 12 | 1016 ± 4 | 1.2 / 4.1 | 4118 / 8729 | 0.14 |
| 2 | max | 2211 ± 34 | 1447 ± 17 | 1.0 / 4.5 | 3772 / 8607 | 0.35 |
| 4 | max | 3531 ± 20 | 1866 ± 13 | 1.5 / 29.5 | 3930 / 10980 | 0.43 |
| 1 | 20× real time | 828 ± 5 | 864 ± 23 | 1.3 / 4.2 | 1008 / 8397 | 0.14 |
| 2 | 20× real time | 1238 ± 1 | 1099 ± 15 | 1.1 / 5.3 | 53 / 3475 | 0.18 |
| 4 | 20× real time | 1244 ± 1 | 1230 ± 1 | 1.0 / 6.1 | 3 / 1500 | 0.17 |

An earlier E4 run with the library-default BLAS thread pools (every process starts one pool per core on a 4-vCPU machine) oversubscribed the CPU and showed ours *falling* from 996 to 432 ev/s as sites were added; it is kept as `results/real_nvidia_e4_blas_default.csv`.
