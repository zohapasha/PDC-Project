"""Hardware / software / git provenance stored with every result row."""
from __future__ import annotations

import functools
import hashlib
import json
import platform
import subprocess
import sys


@functools.lru_cache(maxsize=1)
def sysinfo() -> dict:
    import psutil
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=10, cwd=_root()).stdout.strip() or "nogit"
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "argus"],
                               capture_output=True, text=True, timeout=10, cwd=_root()).stdout.strip()
        if dirty and commit != "nogit":
            commit += "-dirty"
    except Exception:
        commit = "nogit"
    import os
    if commit == "nogit" and os.environ.get("ARGUS_GIT"):
        commit = os.environ["ARGUS_GIT"]       # code bundle without .git (Kaggle run)
    cpu = platform.processor()
    try:
        if sys.platform == "win32":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        elif sys.platform.startswith("linux"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    return {"git": commit, "cpu": cpu, "cores_phys": psutil.cpu_count(logical=False),
            "cores_log": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "python": platform.python_version(), "os": platform.platform(terse=True)}


def _root():
    import os
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cfg_hash(d) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]
