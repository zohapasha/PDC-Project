"""Hardware / software / git provenance stored with every result row."""
from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import subprocess
import sys


@functools.lru_cache(maxsize=1)
def system_info() -> dict:
    """The keys are result-CSV column names."""
    import psutil
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=10, cwd=_repo_root()).stdout.strip() or "nogit"
        dirty_files = subprocess.run(["git", "status", "--porcelain", "--", "argus"],
                                     capture_output=True, text=True, timeout=10,
                                     cwd=_repo_root()).stdout.strip()
        if dirty_files and commit != "nogit":
            commit += "-dirty"
    except Exception:
        commit = "nogit"
    if commit == "nogit" and os.environ.get("ARGUS_GIT"):
        commit = os.environ["ARGUS_GIT"]       # code bundle without .git (Kaggle run)
    cpu_name = platform.processor()
    try:
        if sys.platform == "win32":
            import winreg
            registry_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                          r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name = winreg.QueryValueEx(registry_key, "ProcessorNameString")[0].strip()
        elif sys.platform.startswith("linux"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    cpu_name = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    return {"git": commit, "cpu": cpu_name, "cores_phys": psutil.cpu_count(logical=False),
            "cores_log": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "python": platform.python_version(), "os": platform.platform(terse=True)}


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def config_hash(config) -> str:
    return hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:10]
