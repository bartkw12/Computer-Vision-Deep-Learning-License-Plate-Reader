"""Download the datasets this project uses into ``data/``.

Usage::

    python scripts/download_data.py openalpr
    python scripts/download_data.py plates            # needs ROBOFLOW_API_KEY
    python scripts/download_data.py plates --zip export.zip

``openalpr`` fetches the OpenALPR end-to-end benchmark at a pinned commit, so
everyone evaluates on exactly the same files.

``plates`` fetches the plate detection dataset from Roboflow Universe in YOLO
format. That needs a free Roboflow account: either set ``ROBOFLOW_API_KEY``,
or download the "YOLOv11" export from the dataset page yourself and pass it
with ``--zip``.

Only the standard library is used, so this runs before the project's
dependencies are installed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

OPENALPR_REPO = "openalpr/benchmarks"
OPENALPR_COMMIT = "9790ed20d475be1e940a7e2a3f492a97c2ebed9b"  # head of master, 2017-09-07
OPENALPR_REGIONS = ("us", "eu", "br")

ROBOFLOW_WORKSPACE = "roboflow-universe-projects"
ROBOFLOW_PROJECT = "license-plate-recognition-rxg4e"
ROBOFLOW_FORMAT = "yolov11"

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _download(url: str, destination: Path) -> None:
    """Stream ``url`` to ``destination`` with a simple progress readout."""
    request = urllib.request.Request(url, headers={"User-Agent": "alpr-download"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as file:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            file.write(chunk)
            done += len(chunk)
            size = f"{done / 1e6:.0f}" + (f" / {total / 1e6:.0f}" if total else "")
            print(f"\r  downloaded {size} MB", end="", flush=True)
    print()


def _safe_target(root: Path, relative: str) -> Path:
    """Resolve an archive member under ``root``, refusing paths that escape it."""
    target = (root / relative).resolve()
    if root.resolve() not in target.parents:
        raise ValueError(f"archive member escapes the destination: {relative}")
    return target


def download_openalpr(destination: Path) -> None:
    """Fetch ``endtoend/{us,eu,br}`` of the OpenALPR benchmark."""
    if (destination / "endtoend").exists():
        print(f"{destination / 'endtoend'} already exists; delete it to download again")
        return
    url = f"https://github.com/{OPENALPR_REPO}/archive/{OPENALPR_COMMIT}.zip"
    with tempfile.TemporaryDirectory() as temp:
        archive = Path(temp) / "benchmarks.zip"
        print(f"fetching {url}")
        _download(url, archive)
        counts = dict.fromkeys(OPENALPR_REGIONS, 0)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                parts = Path(member.filename).parts  # <repo>-<sha>/endtoend/<region>/<file>
                if member.is_dir() or len(parts) != 4 or parts[1] != "endtoend":
                    continue
                region, name = parts[2], parts[3]
                suffix = Path(name).suffix.lower()
                if region not in OPENALPR_REGIONS or suffix not in {*_IMAGE_SUFFIXES, ".txt"}:
                    continue
                target = _safe_target(destination, f"endtoend/{region}/{name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("wb") as file:
                    shutil.copyfileobj(source, file)
                counts[region] += suffix == ".txt"

    source = {
        "repository": f"https://github.com/{OPENALPR_REPO}",
        "commit": OPENALPR_COMMIT,
        "license": "AGPL-3.0",
        "labeled_plates": counts,
    }
    (destination / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n")
    print(f"labeled plates per region: {counts} -> {destination}")


def _roboflow_export_link(version: int, api_key: str) -> str:
    url = (
        f"https://api.roboflow.com/{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}/{version}/"
        f"{ROBOFLOW_FORMAT}?api_key={api_key}"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)["export"]["link"]


def _count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)


def download_plates(destination: Path, version: int, archive: Path | None) -> None:
    """Fetch (or unpack) the Roboflow plate dataset and write a clean ``data.yaml``."""
    if (destination / "data.yaml").exists():
        print(f"{destination / 'data.yaml'} already exists; delete the folder to start over")
        return
    with tempfile.TemporaryDirectory() as temp:
        if archive is None:
            api_key = os.environ.get("ROBOFLOW_API_KEY")
            if not api_key:
                sys.exit(
                    "Set ROBOFLOW_API_KEY (free account at roboflow.com), or download the "
                    f"{ROBOFLOW_FORMAT} export yourself and pass it with --zip."
                )
            archive = Path(temp) / "plates.zip"
            print(f"requesting version {version} of {ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}")
            _download(_roboflow_export_link(version, api_key), archive)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                if not member.is_dir():
                    target = _safe_target(destination, member.filename)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as source, target.open("wb") as file:
                        shutil.copyfileobj(source, file)

    # Roboflow's own data.yaml uses "../train/images"-style paths that only
    # resolve from one particular working directory. Write absolute-safe ones.
    splits = {"train": "train", "val": "valid", "test": "test"}
    lines = [f"path: {destination.resolve()}"]
    counts = {}
    for key, folder in splits.items():
        if (destination / folder / "images").is_dir():
            lines.append(f"{key}: {folder}/images")
            counts[key] = _count_images(destination / folder / "images")
    lines += ["names:", "  0: license_plate", ""]
    original = destination / "data.yaml"
    if original.exists():
        original.rename(destination / "data.roboflow.yaml")
    (destination / "data.yaml").write_text("\n".join(lines))
    source = {
        "source": f"https://universe.roboflow.com/{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}",
        "version": version,
        "format": ROBOFLOW_FORMAT,
        "images": counts,
    }
    (destination / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n")
    print(f"images per split: {counts} -> {destination}")


def main() -> None:
    """Parse arguments and download the requested dataset."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    datasets = parser.add_subparsers(dest="dataset", required=True)

    openalpr = datasets.add_parser("openalpr", help="OpenALPR end-to-end benchmark")
    openalpr.add_argument("--dest", type=Path, default=Path("data/openalpr"))

    plates = datasets.add_parser("plates", help="Roboflow plate detection dataset")
    plates.add_argument("--dest", type=Path, default=Path("data/plates"))
    plates.add_argument("--version", type=int, default=4, help="dataset version on Roboflow")
    plates.add_argument("--zip", type=Path, help="use this downloaded export instead of the API")

    args = parser.parse_args()
    if args.dataset == "openalpr":
        download_openalpr(args.dest)
    else:
        download_plates(args.dest, args.version, args.zip)


if __name__ == "__main__":
    main()
