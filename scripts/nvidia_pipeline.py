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

from argus.data import features as F  # noqa: E402
from argus.data import nvidia_mtmc as N  # noqa: E402
from argus.data import tracklets as T  # noqa: E402
from argus.eval.experiments import DEFAULT_ENGINE  # noqa: E402
from argus.eval.harness import run as run_config  # noqa: E402


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def seeds(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def probe_fps(path):
    import cv2
    cap = cv2.VideoCapture(path)
    fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return float(fps or 0.0), int(n or 0)


_EXTRACTOR = []


def get_extractor(a):
    """Created on first use, so a run on cached features needs no GPU or weights."""
    if not _EXTRACTOR:
        _EXTRACTOR.append(F.make_extractor(a.model, a.weights, a.device))
    return _EXTRACTOR[0]


def scene_features(a, scene, profile):
    """Steps 1-3. Returns (boxes, tracklets, fps, vec, prov); features are cached in
    <art>/<scene>_{vec,prov}.npy so a re-run skips the video work."""
    vec_p = os.path.join(a.art, f"{scene}_vec.npy")
    prov_p = os.path.join(a.art, f"{scene}_prov.npy")
    have = os.path.exists(vec_p) and os.path.exists(prov_p)
    log(f"== {scene}: downloading {'ground truth' if have else 'ground truth + videos'}")
    t0 = time.time()
    sdir = N.download_scene(a.split, scene, a.work, videos=not have)
    boxes, ids = N.load_boxes(sdir, frame_step=a.frame_step)
    fps = N.FPS / a.frame_step
    info = {"scene": scene, "split": a.split, "cam_ids": ids, "download_s": time.time() - t0}
    if not have:
        # Some released videos are broken (e.g. test/scene_071/camera_0649 is a 262 KB stub
        # without a moov atom). Drop such cameras: their boxes and tracklets are removed,
        # the camera index stays (it simply has no tracklets).
        bad = [k for k, c in enumerate(ids) if probe_fps(N.video_path(sdir, c))[1] <= 0]
        if bad:
            log(f"   WARNING: unreadable video, dropping camera(s) "
                f"{[ids[k] for k in bad]} ({int(boxes.cam.isin(bad).sum())} GT boxes)")
            boxes = boxes[~boxes.cam.isin(bad)].reset_index(drop=True)
            info["dropped_cam_ids"] = [ids[k] for k in bad]
        np.save(os.path.join(a.art, f"{scene}_dropped.npy"), np.array(bad, np.int64))
    else:
        dp = os.path.join(a.art, f"{scene}_dropped.npy")
        bad = np.load(dp).tolist() if os.path.exists(dp) else []
        if bad:
            boxes = boxes[~boxes.cam.isin(bad)].reset_index(drop=True)
    if not have:
        good = [c for k, c in enumerate(ids) if k not in bad]
        vfps, nfr = probe_fps(N.video_path(sdir, good[0]))
        info.update(video_fps=vfps, video_frames=nfr)
        if vfps and abs(vfps - N.FPS) > 0.5:
            log(f"   WARNING: video reports {vfps:.2f} fps, loader assumes {N.FPS}")
    info.update(N.profile(boxes, fps))
    tr = T.boxes_to_tracklets(boxes, fps=fps, gap_s=a.gap)
    info["n_tracklets"] = int(len(tr))
    log(f"   {info['n_boxes']} boxes, {info['n_cams']} cams, {info['n_people']} people, "
        f"{info['duration_s']:.0f} s, {len(tr)} tracklets, "
        f"{info['cams_per_person_mean']:.1f} cams/person")
    if have:
        vec, prov = np.load(vec_p), np.load(prov_p)
    else:
        t1 = time.time()
        paths = [N.video_path(sdir, c) for c in ids]
        vec, prov = F.embed_tracklets_video(tr, boxes, paths, K=a.K, extractor=get_extractor(a),
                                            workers=a.workers, log=log)
        info["features_s"] = time.time() - t1
        np.save(vec_p, vec)
        np.save(prov_p, prov)
        log(f"   features in {info['features_s']:.0f} s")
    if not a.keep_videos:
        shutil.rmtree(sdir, ignore_errors=True)
    profile.setdefault(scene, {}).update(info)
    return boxes, tr, fps, vec, prov


def write_cfg(name, body):
    path = os.path.join("configs", "generated", f"{name}.yaml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(body, f, sort_keys=False)
    return path


def concat(pattern, out):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(out, index=False)
    return df


def e1_table(df, path):
    cols = ["pw_p", "pw_r", "pw_f1", "b3_f1", "idf1", "violations", "agree_offline",
            "agree_centralized"]
    cols = [c for c in cols if c in df.columns]
    order = ["ours", "ours_two_level", "ours_two_level_a", "B1_offline", "B3_no_correction",
             "B4_no_constraints"]
    g = df.groupby("method")[cols].mean().reindex([m for m in order if m in set(df.method)])
    lines = [f"Mean over {df.scene.nunique()} scenes ({', '.join(sorted(df.scene.unique()))}); "
             f"{int(df.groupby('scene').n_tracklets.first().sum())} test tracklets in total.", "",
             "| Method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for m, r in g.iterrows():
        lines.append(f"| {m} | " + " | ".join(f"{v:.3f}" if c != "violations" else f"{v:.1f}"
                                            for c, v in r.items()) + " |")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--e1_scenes", default="scene_061,scene_062,scene_063,scene_064")
    ap.add_argument("--e1_sites", type=int, default=2)
    ap.add_argument("--e4_scenes", default="scene_071,scene_072,scene_073,scene_074",
                    help="scenes forming the E4 campus (one building each); '' to skip E4")
    ap.add_argument("--e4_sites", default="1,2,4")
    ap.add_argument("--e4_seeds", default="0-2")
    ap.add_argument("--e4_speed", type=float, default=20.0)
    ap.add_argument("--blas_threads", type=int, default=1,
                    help="BLAS threads per E4 process (0 = library default)")
    ap.add_argument("--e4_only", action="store_true",
                    help="skip the campus E1/E5 (e.g. when re-running E4 on cached features)")
    ap.add_argument("--work", default="/tmp/nvidia_mtmc", help="download dir (scratch)")
    ap.add_argument("--art", default="data/real/nvidia", help="features + streams")
    ap.add_argument("--weights", default="data/models/osnet_ain_x1_0_msmt17_combineall.pth")
    ap.add_argument("--model", default="osnet_ain_x1_0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--frame_step", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4, help="parallel video decoders")
    ap.add_argument("--keep_videos", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.art, exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    prof_path = "results/real_nvidia_profile.json"
    profile = json.load(open(prof_path)) if os.path.exists(prof_path) else {}

    def save_profile():
        with open(prof_path, "w") as f:
            json.dump(profile, f, indent=1)

    # ------------------------------------------------------------- E1 + E5 per scene
    for scene in [s for s in a.e1_scenes.split(",") if s]:
        boxes, tr, fps, vec, prov = scene_features(a, scene, profile)
        n_cams = int(boxes.cam.max()) + 1
        site = N.site_map_by_fov(boxes, n_cams, a.e1_sites)
        res = T.build_split_streams(tr, vec, prov, fps, n_cams, site,
                                    os.path.join(a.art, f"{scene}_s{a.e1_sites}"),
                                    engine_cfg=DEFAULT_ENGINE, log=log)
        profile[scene]["e1_split"] = {**res, "site_of_cam": site}
        log(f"   topology: {res['n_overlap_groups']} overlap groups over {n_cams} cams, "
            f"{res['n_border_cams']} border cams, median walking time {res['D_finite_median_s']:.1f} s")
        save_profile()
        eng = {"tau_accept": res["tau_accept"], "tau_keep": min(0.5, res["tau_accept"])}
        for exp in ("e1", "e5"):
            name = f"real_nvidia_{exp}_{scene}"
            run_config(write_cfg(name, {"experiment": exp, "stream": res["test_path"],
                                        "engine": eng, "name": name}), [0])
            df = pd.read_csv(f"results/{name}.csv")
            df["scene"] = scene
            df.to_csv(f"results/{name}.csv", index=False)
        log(f"   {scene} done")

    # ------------------------------------------------------------- E4 campus of scenes
    # Each scene is one building (disjoint cameras and people, all on the same clock);
    # for n sites, buildings are assigned to sites in contiguous blocks. Strong scaling:
    # the campus is fixed, only the number of sites changes.
    e4_scenes = [x for x in a.e4_scenes.split(",") if x]
    if e4_scenes:
        parts, cam_off, n_b = [], 0, len(e4_scenes)
        building = []
        for k, scene in enumerate(e4_scenes):
            boxes, tr, fps, vec, prov = scene_features(a, scene, profile)
            n_cams_k = max(len(profile[scene]["cam_ids"]), int(boxes.cam.max()) + 1)
            t = tr[["cam", "gt_pid", "f_start", "f_end", "t_start", "t_end"]].copy()
            t["cam"] += cam_off
            t["gt_pid"] += k * 10**6
            parts.append((t, vec, prov))
            building += [k] * n_cams_k
            cam_off += n_cams_k
            del boxes
            save_profile()
        tr = pd.concat([t for t, _, _ in parts], ignore_index=True)
        vec = np.concatenate([v for _, v, _ in parts])
        prov = np.concatenate([q for _, _, q in parts])
        o = np.lexsort((tr.cam.to_numpy(), tr.t_start.to_numpy(), tr.t_end.to_numpy()))
        tr, vec, prov = tr.iloc[o].reset_index(drop=True), vec[o], prov[o]
        log(f"== campus: {n_b} buildings, {cam_off} cams, {len(tr)} tracklets, "
            f"{tr.gt_pid.nunique()} people")
        streams, eng, campus = {}, None, {"scenes": e4_scenes, "n_cams": cam_off}
        for n in [int(x) for x in a.e4_sites.split(",")]:
            site = [building[c] * n // n_b for c in range(cam_off)]
            res = T.build_split_streams(tr, vec, prov, fps, cam_off, site,
                                        os.path.join(a.art, f"campus{n_b}_s{n}"),
                                        engine_cfg=DEFAULT_ENGINE if eng is None else None, log=log)
            if eng is None:
                eng = {"tau_accept": res["tau_accept"], "tau_keep": min(0.5, res["tau_accept"])}
            streams[n] = res["test_path"]
            campus[f"s{n}"] = res
        profile["campus"] = campus
        save_profile()
        n_max = max(streams)
        log(f"   E4 on the campus: sites {list(streams)}, {os.cpu_count()} CPUs visible")
        for exp in ([] if a.e4_only else ["e1", "e5"]):   # campus accuracy + bytes at max sites
            name = f"real_nvidia_campus_{exp}"
            run_config(write_cfg(name, {"experiment": exp, "stream": streams[n_max],
                                        "engine": eng, "name": name}), [0])
        name = "real_nvidia_e4"
        if os.path.exists(f"results/{name}.csv"):
            os.remove(f"results/{name}.csv")
        run_config(write_cfg(name, {"experiment": "e4", "kind": "real_strong",
                                    "sites": list(streams), "streams": streams,
                                    "speed": a.e4_speed, "methods": ["two_level", "central"],
                                    "blas_threads": a.blas_threads or None,
                                    "engine": eng, "name": name}), seeds(a.e4_seeds))

    # ------------------------------------------------------------- collect
    save_profile()
    e1 = concat("results/real_nvidia_e1_scene_*.csv", "results/real_nvidia_e1.csv")
    concat("results/real_nvidia_e5_scene_*.csv", "results/real_nvidia_e5.csv")
    if e1 is not None:
        e1_table(e1, "results/tables/real_nvidia_e1.md")
    log("pipeline finished")
