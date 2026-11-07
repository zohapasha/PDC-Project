"""End-to-end real-data run on NVIDIA MTMC_Tracking_2024. Meant to run ON KAGGLE (GPU +
internet), launched by kaggle-run/run.py; nothing here needs the dataset locally.

Per scene, one at a time so disk use stays at about one scene (~3 GB):
  1. download ground truth + videos from Hugging Face
  2. GT boxes -> tracklets; cameras -> sites by floor position (site_map_by_fov)
  3. OSNet-AIN (MSMT17) features from the videos on the GPU; videos are deleted after
  4. temporal split, walking times / overlap groups / centering / tau on the train half
  5. experiments on the test half:
       E1 accuracy (+ two-level option b / a) and E5 bytes, each E1 scene split into 2 sites
       E4 strong scaling on a "campus" of several scenes (one building each) split
       into 1, 2, 4 ... sites (multi-process, timed); E1 + E5 also on that campus

Outputs (under the project root): results/real_nvidia_e1.csv, real_nvidia_e5.csv,
real_nvidia_e4.csv, real_nvidia_campus_{e1,e5}.csv, results/real_nvidia_profile.json, results/tables/real_nvidia_e1.md.

    python scripts/nvidia_pipeline.py --split test --e1_scenes scene_061,scene_062 \
        --e4_scenes scene_071,scene_072,scene_073,scene_074 --e4_sites 1,2,4 --e4_seeds 0-2
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from argus.data import features  # noqa: E402
from argus.data import nvidia_mtmc  # noqa: E402
from argus.data import tracklets  # noqa: E402
from argus.eval.experiments import DEFAULT_ENGINE_CONFIG  # noqa: E402
from argus.eval.harness import run as run_config  # noqa: E402

PERSON_ID_OFFSET_PER_SCENE = 10**6      # keeps GT person ids of different scenes disjoint


def log(*parts):
    print(time.strftime("[%H:%M:%S]"), *parts, flush=True)


def parse_seeds(spec):
    """'0-2,5' -> [0, 1, 2, 5]."""
    seeds = []
    for part in spec.split(","):
        first, _, last = part.partition("-")
        seeds += list(range(int(first), int(last or first) + 1))
    return seeds


def probe_video(path):
    """(fps, frame count) as reported by the container; (0, 0) if unreadable."""
    import cv2
    capture = cv2.VideoCapture(path)
    fps, n_frames = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    return float(fps or 0.0), int(n_frames or 0)


_EXTRACTOR = []


def get_extractor(args):
    """Created on first use, so a run on cached features needs no GPU or weights."""
    if not _EXTRACTOR:
        _EXTRACTOR.append(features.make_extractor(args.model, args.weights, args.device))
    return _EXTRACTOR[0]


def scene_features(args, scene, profile):
    """Steps 1-3. Returns (boxes, tracklet_table, fps, vectors, provisional_vectors);
    features are cached in <artifacts_dir>/<scene>_{vec,prov}.npy so a re-run skips the
    video work."""
    vectors_path = os.path.join(args.artifacts_dir, f"{scene}_vec.npy")
    provisional_path = os.path.join(args.artifacts_dir, f"{scene}_prov.npy")
    dropped_path = os.path.join(args.artifacts_dir, f"{scene}_dropped.npy")
    has_cached_features = os.path.exists(vectors_path) and os.path.exists(provisional_path)
    log(f"== {scene}: downloading "
        f"{'ground truth' if has_cached_features else 'ground truth + videos'}")
    start = time.time()
    scene_dir = nvidia_mtmc.download_scene(args.split, scene, args.download_dir,
                                           videos=not has_cached_features)
    boxes, dataset_camera_ids = nvidia_mtmc.load_boxes(scene_dir, frame_step=args.frame_step)
    fps = nvidia_mtmc.FPS / args.frame_step
    # profile JSON keys are kept stable (results/real_nvidia_profile.json)
    info = {"scene": scene, "split": args.split, "cam_ids": dataset_camera_ids,
            "download_s": time.time() - start}
    if not has_cached_features:
        # Some released videos are broken (e.g. test/scene_071/camera_0649 is a 262 KB stub
        # without a moov atom). Drop such cameras: their boxes and tracklets are removed,
        # the camera index stays (it simply has no tracklets).
        dropped_cameras = [camera_id for camera_id, dataset_id in enumerate(dataset_camera_ids)
                           if probe_video(nvidia_mtmc.video_path(scene_dir, dataset_id))[1] <= 0]
        if dropped_cameras:
            log(f"   WARNING: unreadable video, dropping camera(s) "
                f"{[dataset_camera_ids[c] for c in dropped_cameras]} "
                f"({int(boxes.camera_id.isin(dropped_cameras).sum())} GT boxes)")
            boxes = boxes[~boxes.camera_id.isin(dropped_cameras)].reset_index(drop=True)
            info["dropped_cam_ids"] = [dataset_camera_ids[c] for c in dropped_cameras]
        np.save(dropped_path, np.array(dropped_cameras, np.int64))
    else:
        dropped_cameras = np.load(dropped_path).tolist() if os.path.exists(dropped_path) else []
        if dropped_cameras:
            boxes = boxes[~boxes.camera_id.isin(dropped_cameras)].reset_index(drop=True)
    if not has_cached_features:
        readable_camera_ids = [dataset_id for camera_id, dataset_id in enumerate(dataset_camera_ids)
                               if camera_id not in dropped_cameras]
        video_fps, video_frames = probe_video(nvidia_mtmc.video_path(scene_dir, readable_camera_ids[0]))
        info.update(video_fps=video_fps, video_frames=video_frames)
        if video_fps and abs(video_fps - nvidia_mtmc.FPS) > 0.5:
            log(f"   WARNING: video reports {video_fps:.2f} fps, loader assumes {nvidia_mtmc.FPS}")
    info.update(nvidia_mtmc.summarize_boxes(boxes, fps))
    tracklet_table = tracklets.boxes_to_tracklets(boxes, fps=fps, gap_s=args.gap_s)
    info["n_tracklets"] = int(len(tracklet_table))
    log(f"   {info['n_boxes']} boxes, {info['n_cams']} cams, {info['n_people']} people, "
        f"{info['duration_s']:.0f} s, {len(tracklet_table)} tracklets, "
        f"{info['cams_per_person_mean']:.1f} cams/person")
    if has_cached_features:
        vectors, provisional_vectors = np.load(vectors_path), np.load(provisional_path)
    else:
        features_start = time.time()
        video_paths = [nvidia_mtmc.video_path(scene_dir, dataset_id) for dataset_id in dataset_camera_ids]
        vectors, provisional_vectors = features.embed_tracklets_video(
            tracklet_table, boxes, video_paths, frames_per_tracklet=args.frames_per_tracklet,
            extractor=get_extractor(args), workers=args.workers, log=log)
        info["features_s"] = time.time() - features_start
        np.save(vectors_path, vectors)
        np.save(provisional_path, provisional_vectors)
        log(f"   features in {info['features_s']:.0f} s")
    if not args.keep_videos:
        shutil.rmtree(scene_dir, ignore_errors=True)
    profile.setdefault(scene, {}).update(info)
    return boxes, tracklet_table, fps, vectors, provisional_vectors


def write_config(name, body):
    path = os.path.join("configs", "generated", f"{name}.yaml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        yaml.safe_dump(body, file, sort_keys=False)
    return path


def concat_csvs(pattern, output_path):
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None
    combined = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    combined.to_csv(output_path, index=False)
    return combined


def write_e1_table(results, path):
    columns = ["pw_p", "pw_r", "pw_f1", "b3_f1", "idf1", "violations", "agree_offline",
               "agree_centralized"]
    columns = [column for column in columns if column in results.columns]
    method_order = ["ours", "ours_two_level", "ours_two_level_a", "B1_offline",
                    "B3_no_correction", "B4_no_constraints"]
    means = results.groupby("method")[columns].mean().reindex(
        [method for method in method_order if method in set(results.method)])
    n_test_tracklets = int(results.groupby("scene").n_tracklets.first().sum())
    lines = [f"Mean over {results.scene.nunique()} scenes "
             f"({', '.join(sorted(results.scene.unique()))}); "
             f"{n_test_tracklets} test tracklets in total.", "",
             "| Method | " + " | ".join(columns) + " |", "|---" * (len(columns) + 1) + "|"]
    for method, row in means.iterrows():
        lines.append(f"| {method} | " + " | ".join(
            f"{value:.3f}" if column != "violations" else f"{value:.1f}"
            for column, value in row.items()) + " |")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        file.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def frozen_engine_config(split_summary):
    """Engine thresholds chosen on the training split."""
    tau_accept = split_summary["tau_accept"]
    return {"tau_accept": tau_accept, "tau_keep": min(0.5, tau_accept)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--e1_scenes", default="scene_061,scene_062,scene_063,scene_064")
    parser.add_argument("--e1_sites", type=int, default=2)
    parser.add_argument("--e4_scenes", default="scene_071,scene_072,scene_073,scene_074",
                        help="scenes forming the E4 campus (one building each); '' to skip E4")
    parser.add_argument("--e4_sites", default="1,2,4")
    parser.add_argument("--e4_seeds", default="0-2")
    parser.add_argument("--e4_speed", type=float, default=20.0)
    parser.add_argument("--blas_threads", type=int, default=1,
                        help="BLAS threads per E4 process (0 = library default)")
    parser.add_argument("--e4_only", action="store_true",
                        help="skip the campus E1/E5 (e.g. when re-running E4 on cached features)")
    parser.add_argument("--work", default="/tmp/nvidia_mtmc", dest="download_dir",
                        help="download dir (scratch)")
    parser.add_argument("--art", default="data/real/nvidia", dest="artifacts_dir",
                        help="features + streams")
    parser.add_argument("--weights", default="data/models/osnet_ain_x1_0_msmt17_combineall.pth")
    parser.add_argument("--model", default="osnet_ain_x1_0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--K", type=int, default=8, dest="frames_per_tracklet",
                        help="frames sampled per tracklet")
    parser.add_argument("--gap", type=float, default=1.0, dest="gap_s")
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4, help="parallel video decoders")
    parser.add_argument("--keep_videos", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    profile_path = "results/real_nvidia_profile.json"
    profile = json.load(open(profile_path)) if os.path.exists(profile_path) else {}

    def save_profile():
        with open(profile_path, "w") as file:
            json.dump(profile, file, indent=1)

    # ------------------------------------------------------------- E1 + E5 per scene
    for scene in [name for name in args.e1_scenes.split(",") if name]:
        boxes, tracklet_table, fps, vectors, provisional_vectors = scene_features(args, scene, profile)
        n_cameras = int(boxes.camera_id.max()) + 1
        site_of_camera = nvidia_mtmc.site_map_by_fov(boxes, n_cameras, args.e1_sites)
        split_summary = tracklets.build_split_streams(
            tracklet_table, vectors, provisional_vectors, fps, n_cameras, site_of_camera,
            os.path.join(args.artifacts_dir, f"{scene}_s{args.e1_sites}"),
            engine_config=DEFAULT_ENGINE_CONFIG, log=log)
        profile[scene]["e1_split"] = {**split_summary, "site_of_cam": site_of_camera}
        log(f"   topology: {split_summary['n_overlap_groups']} overlap groups over {n_cameras} cams, "
            f"{split_summary['n_border_cams']} border cams, "
            f"median walking time {split_summary['D_finite_median_s']:.1f} s")
        save_profile()
        engine_overrides = frozen_engine_config(split_summary)
        for experiment_id in ("e1", "e5"):
            name = f"real_nvidia_{experiment_id}_{scene}"
            run_config(write_config(name, {"experiment": experiment_id,
                                           "stream": split_summary["test_path"],
                                           "engine": engine_overrides, "name": name}), [0])
            scene_results = pd.read_csv(f"results/{name}.csv")
            scene_results["scene"] = scene
            scene_results.to_csv(f"results/{name}.csv", index=False)
        log(f"   {scene} done")

    # ------------------------------------------------------------- E4 campus of scenes
    # Each scene is one building (disjoint cameras and people, all on the same clock);
    # for n sites, buildings are assigned to sites in contiguous blocks. Strong scaling:
    # the campus is fixed, only the number of sites changes.
    e4_scenes = [name for name in args.e4_scenes.split(",") if name]
    if e4_scenes:
        scene_parts, camera_offset, n_buildings = [], 0, len(e4_scenes)
        building_of_camera = []
        for building, scene in enumerate(e4_scenes):
            boxes, tracklet_table, fps, vectors, provisional_vectors = scene_features(args, scene, profile)
            n_scene_cameras = max(len(profile[scene]["cam_ids"]), int(boxes.camera_id.max()) + 1)
            scene_table = tracklet_table[["camera_id", "gt_person_id", "start_frame", "end_frame",
                                          "start_time", "end_time"]].copy()
            scene_table["camera_id"] += camera_offset
            scene_table["gt_person_id"] += building * PERSON_ID_OFFSET_PER_SCENE
            scene_parts.append((scene_table, vectors, provisional_vectors))
            building_of_camera += [building] * n_scene_cameras
            camera_offset += n_scene_cameras
            del boxes
            save_profile()
        tracklet_table = pd.concat([table for table, _, _ in scene_parts], ignore_index=True)
        vectors = np.concatenate([part_vectors for _, part_vectors, _ in scene_parts])
        provisional_vectors = np.concatenate([part_provisional for _, _, part_provisional in scene_parts])
        order = np.lexsort((tracklet_table.camera_id.to_numpy(), tracklet_table.start_time.to_numpy(),
                            tracklet_table.end_time.to_numpy()))
        tracklet_table = tracklet_table.iloc[order].reset_index(drop=True)
        vectors, provisional_vectors = vectors[order], provisional_vectors[order]
        log(f"== campus: {n_buildings} buildings, {camera_offset} cams, {len(tracklet_table)} tracklets, "
            f"{tracklet_table.gt_person_id.nunique()} people")
        test_stream_of_site_count, engine_overrides = {}, None
        campus = {"scenes": e4_scenes, "n_cams": camera_offset}
        for n_sites in [int(count) for count in args.e4_sites.split(",")]:
            site_of_camera = [building_of_camera[camera] * n_sites // n_buildings
                              for camera in range(camera_offset)]
            split_summary = tracklets.build_split_streams(
                tracklet_table, vectors, provisional_vectors, fps, camera_offset, site_of_camera,
                os.path.join(args.artifacts_dir, f"campus{n_buildings}_s{n_sites}"),
                engine_config=DEFAULT_ENGINE_CONFIG if engine_overrides is None else None, log=log)
            if engine_overrides is None:
                engine_overrides = frozen_engine_config(split_summary)
            test_stream_of_site_count[n_sites] = split_summary["test_path"]
            campus[f"s{n_sites}"] = split_summary
        profile["campus"] = campus
        save_profile()
        max_sites = max(test_stream_of_site_count)
        log(f"   E4 on the campus: sites {list(test_stream_of_site_count)}, "
            f"{os.cpu_count()} CPUs visible")
        for experiment_id in ([] if args.e4_only else ["e1", "e5"]):   # campus accuracy + bytes at max sites
            name = f"real_nvidia_campus_{experiment_id}"
            run_config(write_config(name, {"experiment": experiment_id,
                                           "stream": test_stream_of_site_count[max_sites],
                                           "engine": engine_overrides, "name": name}), [0])
        name = "real_nvidia_e4"
        if os.path.exists(f"results/{name}.csv"):
            os.remove(f"results/{name}.csv")
        run_config(write_config(name, {"experiment": "e4", "kind": "real_strong",
                                       "sites": list(test_stream_of_site_count),
                                       "streams": test_stream_of_site_count,
                                       "speed": args.e4_speed, "methods": ["two_level", "central"],
                                       "blas_threads": args.blas_threads or None,
                                       "engine": engine_overrides, "name": name}),
                   parse_seeds(args.e4_seeds))

    # ------------------------------------------------------------- collect
    save_profile()
    e1_results = concat_csvs("results/real_nvidia_e1_scene_*.csv", "results/real_nvidia_e1.csv")
    concat_csvs("results/real_nvidia_e5_scene_*.csv", "results/real_nvidia_e5.csv")
    if e1_results is not None:
        write_e1_table(e1_results, "results/tables/real_nvidia_e1.md")
    log("pipeline finished")
