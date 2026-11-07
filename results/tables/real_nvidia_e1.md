NVIDIA MTMC_Tracking_2024, test scenes 061–070 (10 scenes, 10 cameras each, 20–25 people, 800 s). Each scene: temporal split, walking times / overlap groups / centering / tau fitted on the first half, evaluated on the second half (6637 test tracklets in total); cameras split into 2 sites by floor position. Mean ± std over scenes.

| Method | Pairwise P / R / F1 | B³ F1 | IDF1 | Violations | Agreement w/ B1 | Agreement w/ centralized |
|---|---|---|---|---|---|---|
| Ours | 0.844 / 0.658 / **0.738** ± 0.086 | 0.760 | 0.929 | 0 total | 0.922 | – |
| Ours two-level (option b) | 0.846 / 0.656 / 0.738 ± 0.085 | 0.759 | 0.928 | 0 total | 0.925 | 0.987 |
| Ours two-level (option a) | 0.818 / 0.643 / 0.718 ± 0.092 | 0.743 | 0.919 | 0 total | 0.885 | 0.917 |
| B1 offline | 0.808 / 0.666 / 0.729 ± 0.095 | 0.760 | 0.928 | 0 total | 1.000 | – |
| B3 no correction | 0.705 / 0.610 / 0.653 ± 0.114 | 0.702 | 0.872 | 0 total | 0.777 | – |
| B4 no constraints | 0.117 / 0.859 / 0.183 ± 0.170 | 0.379 | 0.309 | 20 total | 0.188 | – |

Per-scene pairwise F1:

| Scene | People | Test tracklets | Overlap groups | τ | Ours | Two-level b | B1 | B3 | B4 |
|---|---|---|---|---|---|---|---|---|---|
| scene_061 | 20 | 617 | 2 | 0.55 | 0.800 | 0.800 | 0.782 | 0.706 | 0.147 |
| scene_062 | 24 | 704 | 2 | 0.5 | 0.663 | 0.658 | 0.657 | 0.481 | 0.103 |
| scene_063 | 24 | 1168 | 2 | 0.5 | 0.652 | 0.661 | 0.648 | 0.537 | 0.083 |
| scene_064 | 23 | 722 | 1 | 0.5 | 0.719 | 0.719 | 0.690 | 0.631 | 0.089 |
| scene_065 | 23 | 677 | 1 | 0.6 | 0.736 | 0.736 | 0.735 | 0.680 | 0.344 |
| scene_066 | 25 | 511 | 2 | 0.5 | 0.910 | 0.910 | 0.910 | 0.840 | 0.612 |
| scene_067 | 22 | 619 | 3 | 0.5 | 0.808 | 0.801 | 0.831 | 0.771 | 0.088 |
| scene_068 | 21 | 580 | 1 | 0.5 | 0.728 | 0.728 | 0.714 | 0.632 | 0.157 |
| scene_069 | 23 | 538 | 1 | 0.5 | 0.618 | 0.614 | 0.580 | 0.528 | 0.113 |
| scene_070 | 24 | 501 | 3 | 0.5 | 0.750 | 0.750 | 0.741 | 0.721 | 0.094 |
