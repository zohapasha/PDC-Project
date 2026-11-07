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
INCLUDED_PREFIXES = ("argus/", "scripts/", "tests/", "configs/", "pyproject.toml", "requirements.txt")

PIPELINE_ARGS_OF_MODE = {
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
KERNEL_SETTINGS_OF_MODE = {"e4": {"gpu": False, "cache_from": ["argus-nvidia-mtmc-campus"]}}


def git(*git_args):
    return subprocess.run(["git", *git_args], cwd=ROOT, capture_output=True, text=True).stdout


def bundle() -> str:
    paths = git("ls-files", "--cached", "--others", "--exclude-standard").split("\n")
    paths = sorted(path for path in paths if path and path.startswith(INCLUDED_PREFIXES)
                   and os.path.isfile(os.path.join(ROOT, path))
                   and not path.startswith("configs/generated/"))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in paths:
            archive.add(os.path.join(ROOT, path), arcname=path)
    print(f"bundled {len(paths)} files, {buffer.tell() / 1024:.0f} KiB compressed")
    return base64.b64encode(buffer.getvalue()).decode()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode not in PIPELINE_ARGS_OF_MODE:
        raise SystemExit(f"mode must be one of {list(PIPELINE_ARGS_OF_MODE)}")
    username = None
    for credentials_path in ("~/.kaggle/kaggle.json", "~/.kaggle/credentials.json"):
        credentials_path = os.path.expanduser(credentials_path)
        if os.path.exists(credentials_path):
            username = json.load(open(credentials_path)).get("username") or username
    username = os.environ.get("KAGGLE_USERNAME", username)
    if not username:
        raise SystemExit("no Kaggle username found (~/.kaggle/kaggle.json or KAGGLE_USERNAME)")
    commit = git("rev-parse", "--short", "HEAD").strip() or "nogit"
    if git("status", "--porcelain", "--", "argus", "scripts"):
        commit += "-dirty"
    source = open(os.path.join(HERE, "run_template.py"), encoding="utf-8").read()
    source = (source.replace("@@BUNDLE@@", bundle()).replace("@@GIT@@", commit)
                    .replace("@@MODE@@", mode).replace("@@ARGS@@", PIPELINE_ARGS_OF_MODE[mode]))
    settings = KERNEL_SETTINGS_OF_MODE.get(mode, {})
    cache_kernels = [f"{username}/{slug}" for slug in settings.get("cache_from", [])]
    source = source.replace('"@@CACHE@@"', repr([kernel.split("/")[1] for kernel in cache_kernels]))
    output_dir = os.path.join(HERE, mode)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run.py"), "w", encoding="utf-8") as file:
        file.write(source)
    slug = "argus-nvidia-mtmc" + ("" if mode == "full" else f"-{mode}")
    metadata = {"id": f"{username}/{slug}", "title": slug, "code_file": "run.py",
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": settings.get("gpu", True), "enable_internet": True,
                "dataset_sources": [], "competition_sources": [], "kernel_sources": cache_kernels,
                "model_sources": []}
    with open(os.path.join(output_dir, "kernel-metadata.json"), "w") as file:
        json.dump(metadata, file, indent=1)
    print(f"wrote {output_dir}/run.py and kernel-metadata.json (id {username}/{slug}, code {commit})")
    print(f"next: kaggle kernels push -p {os.path.relpath(output_dir, ROOT)}")
    print(f"then: PYTHONUTF8=1 kaggle kernels output {username}/{slug} -p kaggle-run/output/{mode} "
          r'--file-pattern "(results/|\.log$)"')
