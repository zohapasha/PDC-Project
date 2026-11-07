r"""Build a self-contained Kaggle script kernel from the current working tree.

    python kaggle-run/build.py smoke        # E1+E5 on 1 scene, E4 on a 2-scene campus (~25 min)
    python kaggle-run/build.py full         # E1+E5 on 10 scenes, E4 on an 8-scene campus (~4-5 h)
    kaggle kernels push -p kaggle-run/smoke
    kaggle kernels status <user>/argus-nvidia-mtmc-smoke
    PYTHONUTF8=1 kaggle kernels output <user>/argus-nvidia-mtmc-smoke -p kaggle-run/output/smoke         --file-pattern "(results/|\.log$)"     # results + log only, not the cached features

The code (tracked + untracked, not ignored) under argus/, scripts/, tests/, configs/ is
embedded as base64 in kaggle-run/<mode>/run.py, so the kernel runs exactly this tree
without GitHub. Data never touches this machine: the kernel downloads it on Kaggle.
"""
import base64
import io
import json
import os
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INCLUDE = ("argus/", "scripts/", "tests/", "configs/", "pyproject.toml", "requirements.txt")

MODES = {
    "smoke": "--split test --e1_scenes scene_061 --e4_scenes scene_061,scene_062 "
             "--e4_sites 1,2 --e4_seeds 0 --e4_speed 50",
    "full": "--split test --e1_scenes " + ",".join(f"scene_{i:03d}" for i in range(61, 71)) +
            " --e4_scenes " + ",".join(f"scene_{i:03d}" for i in range(71, 79)) +
            " --e4_sites 1,2,4 --e4_seeds 0-2 --e4_speed 20",
    # E4 only, on the features cached by the "campus" kernel (attached as input; CPU only)
    "e4": "--split test --e1_scenes '' --e4_only --e4_scenes " +
          ",".join(f"scene_{i:03d}" for i in range(71, 79)) +
          " --e4_sites 1,2,4 --e4_seeds 0-2 --e4_speed 20 --blas_threads 1",
    # campus part of "full" only (its per-scene E1/E5 part already ran)
    "campus": "--split test --e1_scenes '' --e4_scenes " +
              ",".join(f"scene_{i:03d}" for i in range(71, 79)) +
              " --e4_sites 1,2,4 --e4_seeds 0-2 --e4_speed 20",
}


# per-mode kernel settings: GPU, and earlier kernels whose output holds cached features
SETTINGS = {"e4": {"gpu": False, "cache_from": ["argus-nvidia-mtmc-campus"]}}


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout


def bundle() -> str:
    files = git("ls-files", "--cached", "--others", "--exclude-standard").split("\n")
    files = sorted(f for f in files if f and f.startswith(INCLUDE) and os.path.isfile(
        os.path.join(ROOT, f)) and not f.startswith("configs/generated/"))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for f in files:
            t.add(os.path.join(ROOT, f), arcname=f)
    print(f"bundled {len(files)} files, {buf.tell() / 1024:.0f} KiB compressed")
    return base64.b64encode(buf.getvalue()).decode()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode not in MODES:
        raise SystemExit(f"mode must be one of {list(MODES)}")
    user = None
    for p in ("~/.kaggle/kaggle.json", "~/.kaggle/credentials.json"):
        p = os.path.expanduser(p)
        if os.path.exists(p):
            user = json.load(open(p)).get("username") or user
    user = os.environ.get("KAGGLE_USERNAME", user)
    if not user:
        raise SystemExit("no Kaggle username found (~/.kaggle/kaggle.json or KAGGLE_USERNAME)")
    commit = git("rev-parse", "--short", "HEAD").strip() or "nogit"
    if git("status", "--porcelain", "--", "argus", "scripts"):
        commit += "-dirty"
    src = open(os.path.join(HERE, "run_template.py"), encoding="utf-8").read()
    src = (src.replace("@@BUNDLE@@", bundle()).replace("@@GIT@@", commit)
              .replace("@@MODE@@", mode).replace("@@ARGS@@", MODES[mode]))
    st = SETTINGS.get(mode, {})
    cache = [f"{user}/{k}" for k in st.get("cache_from", [])]
    src = src.replace('"@@CACHE@@"', repr([k.split("/")[1] for k in cache]))
    out = os.path.join(HERE, mode)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "run.py"), "w", encoding="utf-8") as f:
        f.write(src)
    slug = "argus-nvidia-mtmc" + ("" if mode == "full" else f"-{mode}")
    meta = {"id": f"{user}/{slug}", "title": slug, "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": st.get("gpu", True), "enable_internet": True, "dataset_sources": [],
            "competition_sources": [], "kernel_sources": cache, "model_sources": []}
    with open(os.path.join(out, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {out}/run.py and kernel-metadata.json (id {user}/{slug}, code {commit})")
    print(f"next: kaggle kernels push -p {os.path.relpath(out, ROOT)}")
    print(f"then: PYTHONUTF8=1 kaggle kernels output {user}/{slug} -p kaggle-run/output/{mode} "
          r'--file-pattern "(results/|\.log$)"')
