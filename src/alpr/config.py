"""Pipeline configuration: typed defaults, YAML files and command-line overrides.

The dataclasses below are the single source of truth for default values.
``configs/pipeline.yaml`` documents the same settings; a test keeps the two
from drifting apart.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from alpr.aggregation import STRATEGIES
from alpr.ocr import BACKENDS
from alpr.ocr.preprocess import build_preprocessor
from alpr.plate_format import get_formats

PLATE_SEARCH_MODES = ("frame", "vehicle")


@dataclass(frozen=True)
class DetectorConfig:
    """Settings for one YOLO detector.

    Attributes:
        weights: Path to the model weights.
        conf: Minimum detection confidence.
        iou: IoU threshold for non-maximum suppression.
        imgsz: Inference image size in pixels.
    """

    weights: str = ""
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640


@dataclass(frozen=True)
class TrackerConfig:
    """Settings for the SORT tracker.

    Attributes:
        max_age: Frames a track survives without a matching detection.
        min_hits: Consecutive matches before a new track is reported.
        iou_threshold: Minimum overlap for a detection to continue a track.
    """

    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3


@dataclass(frozen=True)
class OcrConfig:
    """Settings for reading plate crops.

    Attributes:
        backend: OCR backend name; see :data:`alpr.ocr.BACKENDS`.
        preprocess: Preprocessing applied to each crop before OCR.
        every: Read a track's plate at most once every this many frames.
        min_plate_height: Skip plates shorter than this many pixels.
    """

    backend: str = "easyocr"
    preprocess: str = "none"
    every: int = 1
    min_plate_height: int = 0


@dataclass(frozen=True)
class PlateFormatConfig:
    """Settings for format-aware correction of OCR output.

    Attributes:
        name: Named format set; see :func:`alpr.plate_format.get_formats`.
        max_substitutions: Look-alike characters that may be corrected per
            read. ``0`` accepts exact format matches only.
    """

    name: str = "ontario"
    max_substitutions: int = 2


@dataclass(frozen=True)
class AggregationConfig:
    """Settings for combining a track's reads.

    Attributes:
        strategy: ``"max_conf"``, ``"string_vote"`` or ``"char_vote"``.
        min_reads: Tracks with fewer valid reads than this report no plate.
    """

    strategy: str = "string_vote"
    min_reads: int = 1


@dataclass(frozen=True)
class PipelineConfig:
    """All settings of the plate-reading pipeline.

    Attributes:
        device: ``"auto"``, ``"cpu"`` or a CUDA device such as ``"cuda:0"``.
        vehicle: Vehicle detector (COCO-pretrained).
        plate: License plate detector.
        plate_search: ``"frame"`` runs the plate detector on the whole frame;
            ``"vehicle"`` runs it on each tracked vehicle's crop.
        min_plate_overlap: Share of a plate box that must lie inside a vehicle
            box for the two to be matched.
        tracker: Tracker settings.
        ocr: OCR settings.
        plate_format: Plate format settings.
        aggregation: Settings for combining reads across frames.
    """

    device: str = "auto"
    vehicle: DetectorConfig = DetectorConfig(weights="weights/yolo11n.pt")
    plate: DetectorConfig = DetectorConfig(weights="weights/plate_yolo11n.pt")
    plate_search: str = "frame"
    min_plate_overlap: float = 0.7
    tracker: TrackerConfig = TrackerConfig()
    ocr: OcrConfig = OcrConfig()
    plate_format: PlateFormatConfig = PlateFormatConfig()
    aggregation: AggregationConfig = AggregationConfig()

    def validate(self) -> None:
        """Check that every named choice exists.

        Raises:
            ValueError: On the first invalid setting, naming it.
        """
        if self.plate_search not in PLATE_SEARCH_MODES:
            raise ValueError(
                f"plate_search must be one of {list(PLATE_SEARCH_MODES)}, got {self.plate_search!r}"
            )
        if self.ocr.backend not in BACKENDS:
            raise ValueError(
                f"ocr.backend must be one of {list(BACKENDS)}, got {self.ocr.backend!r}"
            )
        if self.aggregation.strategy not in STRATEGIES:
            raise ValueError(
                f"aggregation.strategy must be one of {sorted(STRATEGIES)}, "
                f"got {self.aggregation.strategy!r}"
            )
        if self.ocr.every < 1:
            raise ValueError(f"ocr.every must be at least 1, got {self.ocr.every}")
        build_preprocessor(self.ocr.preprocess)
        get_formats(self.plate_format.name)

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as nested plain dictionaries."""
        return dataclasses.asdict(self)


def _coerce(value: Any, current: Any, key: str) -> Any:
    """Check ``value`` against the type of the default it replaces."""
    expected = type(current)
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if type(value) is not expected:
        raise ValueError(
            f"{key} must be {expected.__name__}, got {type(value).__name__} ({value!r})"
        )
    return value


def _updated(instance: Any, data: Mapping[str, Any], prefix: str = "") -> Any:
    """Return a copy of a config dataclass with ``data`` applied on top."""
    if not isinstance(data, Mapping):
        raise ValueError(f"{prefix.rstrip('.') or 'config'} must be a mapping, got {data!r}")
    names = {f.name for f in dataclasses.fields(instance)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(
            f"unknown config key(s): {', '.join(prefix + key for key in unknown)}; "
            f"valid keys here are {sorted(names)}"
        )
    changes = {}
    for key, value in data.items():
        current = getattr(instance, key)
        if dataclasses.is_dataclass(current):
            changes[key] = _updated(current, value, f"{prefix}{key}.")
        else:
            changes[key] = _coerce(value, current, prefix + key)
    return dataclasses.replace(instance, **changes)


def parse_overrides(overrides: Sequence[str]) -> dict[str, Any]:
    """Turn ``["tracker.max_age=10", "ocr.backend=easyocr"]`` into a nested dict.

    Values are parsed as YAML, so numbers and booleans keep their type.

    Raises:
        ValueError: If an item is not of the form ``key=value``.
    """
    nested: dict[str, Any] = {}
    for item in overrides:
        key, separator, raw = item.partition("=")
        if not separator or not key:
            raise ValueError(f"override must look like key=value, got {item!r}")
        *parents, leaf = key.strip().split(".")
        node = nested
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = yaml.safe_load(raw)
    return nested


def load_config(path: str | Path | None = None, overrides: Sequence[str] = ()) -> PipelineConfig:
    """Build a configuration from defaults, an optional YAML file, and overrides.

    Later sources win: defaults, then the file, then the overrides.

    Args:
        path: YAML file with any subset of the settings.
        overrides: ``key=value`` strings with dotted keys.

    Raises:
        ValueError: On unknown keys, wrong types, or invalid choices.
    """
    config = PipelineConfig()
    if path is not None:
        config = _updated(config, yaml.safe_load(Path(path).read_text()) or {})
    if overrides:
        config = _updated(config, parse_overrides(overrides))
    config.validate()
    return config
