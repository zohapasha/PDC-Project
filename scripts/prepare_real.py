"""Build ARGUS streams from a real dataset (after features were extracted on a GPU).

WILDTRACK:
  python scripts/prepare_real.py wildtrack --root /data/Wildtrack \
      --vec wildtrack_vec.npy --prov wildtrack_prov.npy --out data/real/wildtrack
MTMMC (one recording; camera->site groups by floor/area, e.g. "0,1,2,3;4,5,6,7;..."):
  python scripts/prepare_real.py mtmmc --root /data/MTMMC/campus/s01 --sites "0,1,2,3;4,5,6,7" \
      --vec s01_vec.npy --prov s01_prov.npy --out data/real/mtmmc_s01

Writes <out>_train.parquet and <out>_test.parquet (+ topology/meta JSON) and prints the
tau_accept chosen on the training split. Then run E1 with a config containing
  experiment: e1
  stream: data/real/wildtrack_test.parquet
  engine: {tau_accept: <chosen>}
The tracklet table must be built with the same arguments as used for feature extraction
(the .npy rows are aligned with it).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from argus.data import mtmmc, tracklets  # noqa: E402
from argus.eval.experiments import DEFAULT_ENGINE_CONFIG  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["wildtrack", "mtmmc"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--vec", required=True, dest="vectors_path")
    parser.add_argument("--prov", default=None, dest="provisional_vectors_path")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--sites", default=None, help='camera groups per site, e.g. "0,1;2,3"')
    parser.add_argument("--gap", type=float, default=1.0, dest="gap_s")
    parser.add_argument("--p_frag", type=float, default=0.0)
    parser.add_argument("--train_frac", type=float, default=0.5)
    parser.add_argument("--no_center", action="store_true",
                        help="skip mean-centering (OSNet features are non-negative after ReLU, "
                             "which inflates every cosine; the mean is estimated on train only)")
    parser.add_argument("--out", required=True, dest="output_prefix")
    args = parser.parse_args()

    if args.dataset == "wildtrack":
        from argus.data import wildtrack
        boxes = wildtrack.load_boxes(args.root)
        fps = args.fps or wildtrack.ANNOTATION_FPS
    else:
        boxes = mtmmc.load_boxes(args.root)
        fps = args.fps or 23.0
    tracklet_table = tracklets.boxes_to_tracklets(boxes, fps=fps, gap_s=args.gap_s,
                                                  p_frag=args.p_frag)
    vectors = np.load(args.vectors_path).astype(np.float32)
    provisional_vectors = (np.load(args.provisional_vectors_path).astype(np.float32)
                           if args.provisional_vectors_path else None)
    n_cameras = int(boxes.camera_id.max()) + 1
    if args.sites:
        camera_groups = [[int(camera) for camera in group.split(",")]
                         for group in args.sites.split(";")]
        site_of_camera = mtmmc.site_map_from_groups(camera_groups, n_cameras)
    else:
        site_of_camera = [0] * n_cameras
    tracklets.build_split_streams(tracklet_table, vectors, provisional_vectors, fps, n_cameras,
                                  site_of_camera, args.output_prefix, train_frac=args.train_frac,
                                  center=not args.no_center, engine_config=DEFAULT_ENGINE_CONFIG)
