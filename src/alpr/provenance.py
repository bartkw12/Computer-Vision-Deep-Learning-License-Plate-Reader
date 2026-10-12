"""Recording where a result came from.

Every results file written by this package carries the output of
:func:`environment_info`, so a number in the README can be traced back to the
code revision, library versions and hardware that produced it.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

_PACKAGES = ("alpr", "torch", "ultralytics", "easyocr", "fast-plate-ocr", "opencv-python", "numpy")


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def environment_info(device: str | None = None) -> dict[str, Any]:
    """Describe the code revision, software and hardware in use.

    Args:
        device: The inference device a result was computed on, if any. When
            it is a CUDA device the GPU's name is included.

    Returns:
        A JSON-serializable dictionary.
    """
    status = _git("status", "--porcelain")
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        # True means the working tree had changes not captured by git_commit.
        "git_dirty": bool(status) if status is not None else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "packages": _package_versions(),
    }
    if device is not None:
        info["device"] = device
        if device.startswith("cuda"):
            import torch

            info["gpu"] = torch.cuda.get_device_name(torch.device(device))
    return info
