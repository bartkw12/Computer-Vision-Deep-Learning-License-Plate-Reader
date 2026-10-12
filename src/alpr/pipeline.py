"""The plate-reading pipeline: detect, track, associate, read, aggregate.

Per frame::

    vehicles = vehicle_detector(frame)
    tracks   = tracker.update(vehicles)
    plates   = plate_detector(frame)            # or per tracked vehicle crop
    matches  = assign plates to tracks
    for each track with a plate: OCR the crop, fit the text to a plate format

Every active track is logged in every frame as a
:class:`~alpr.types.TrackObservation`, with the *raw* OCR text alongside the
fitted plate. Because the raw text is kept, the format rules and the
aggregation strategy can be changed and re-evaluated from the log alone,
without running any model again; see :func:`read_tracks`.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from alpr.aggregation import Read, aggregate
from alpr.association import assign_plates_to_tracks
from alpr.config import PipelineConfig
from alpr.detection import VEHICLE_CLASSES, Detector, YoloDetector, resolve_device
from alpr.geometry import crop, crop_with_offset
from alpr.ocr import PlateOCR, build_ocr
from alpr.plate_format import PlateFormat, fit_plate, get_formats
from alpr.tracking import Sort
from alpr.types import Box, Detection, PlateResult, Track, TrackObservation

logger = logging.getLogger(__name__)

#: Stage names in pipeline order, used for timing reports.
STAGES = ("vehicle_detection", "tracking", "plate_detection", "association", "ocr")


class StageTimer:
    """Accumulates wall-clock time per named stage."""

    def __init__(self) -> None:
        self._seconds: dict[str, float] = defaultdict(float)
        self._calls: dict[str, int] = defaultdict(int)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the enclosed block and add it to ``name``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self._seconds[name] += time.perf_counter() - start
            self._calls[name] += 1

    def summary(self, frames: int) -> dict[str, object]:
        """Report time per stage, averaged over ``frames`` processed frames.

        Returns:
            ``{"frames", "ms_per_frame", "fps", "stages": {name: {...}}}``.
            Stage times are per *frame*, not per call, so they add up to the
            total; ``ms_per_call`` is given separately because OCR runs a
            varying number of times per frame.
        """
        frames = max(frames, 1)
        total = sum(self._seconds.values())
        stages = {
            name: {
                "ms_per_frame": 1000 * self._seconds[name] / frames,
                "calls": self._calls[name],
                "ms_per_call": 1000 * self._seconds[name] / max(self._calls[name], 1),
                "share": self._seconds[name] / total if total else 0.0,
            }
            for name in STAGES
            if name in self._seconds
        }
        return {
            "frames": frames,
            "ms_per_frame": 1000 * total / frames,
            "fps": frames / total if total else 0.0,
            "stages": stages,
        }


def read_tracks(
    observations: Iterable[TrackObservation],
    *,
    formats: Sequence[PlateFormat],
    max_substitutions: int = 2,
    strategy: str = "string_vote",
    min_reads: int = 1,
) -> list[PlateResult]:
    """Decide the final plate of every track from its logged OCR reads.

    The raw OCR text of each observation is fitted to the plate formats and
    the fits of a track are aggregated. This is the same computation the live
    pipeline performs, so calling it on a saved log with different arguments
    shows what those settings would have produced.

    Args:
        observations: Logged observations of one video.
        formats: Plate formats to fit reads to; empty accepts any text.
        max_substitutions: Look-alike corrections allowed per read.
        strategy: Aggregation strategy; see :mod:`alpr.aggregation`.
        min_reads: Tracks with fewer fitted reads report no plate.

    Returns:
        One result per track that produced a plate, ordered by track ID.
    """
    reads: dict[int, list[Read]] = defaultdict(list)
    spans: dict[int, tuple[int, int]] = {}
    for row in observations:
        first, last = spans.get(row.track_id, (row.frame, row.frame))
        spans[row.track_id] = (min(first, row.frame), max(last, row.frame))
        if row.ocr_text is None:
            continue
        fit = fit_plate(row.ocr_text, formats, max_substitutions=max_substitutions)
        if fit is not None:
            reads[row.track_id].append((fit.text, row.ocr_confidence or 0.0))

    results = []
    for track_id in sorted(spans):
        consensus = aggregate(reads.get(track_id, []), strategy)
        if consensus is None or consensus.n_reads < min_reads:
            continue
        first, last = spans[track_id]
        results.append(
            PlateResult(
                track_id=track_id,
                text=consensus.text,
                confidence=consensus.confidence,
                n_reads=consensus.n_reads,
                first_frame=first,
                last_frame=last,
            )
        )
    return results


class PlateReaderPipeline:
    """Runs the full pipeline over a sequence of frames.

    The detectors, tracker and OCR backend are passed in, so each can be
    replaced, including by lightweight stand-ins in tests.

    Args:
        vehicle_detector: Finds vehicles in a frame.
        plate_detector: Finds license plates in a frame or a vehicle crop.
        ocr: Reads plate crops.
        config: Pipeline settings.
        tracker: Tracker to use; by default a :class:`~alpr.tracking.Sort`
            built from ``config.tracker``.

    Attributes:
        observations: Everything logged so far, in frame order.
        timer: Per-stage timing.
        counters: Running totals such as plates detected and plates read.
        frames_processed: Number of frames seen.
    """

    def __init__(
        self,
        vehicle_detector: Detector,
        plate_detector: Detector,
        ocr: PlateOCR,
        config: PipelineConfig,
        *,
        tracker: Sort | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self._vehicle_detector = vehicle_detector
        self._plate_detector = plate_detector
        self._ocr = ocr
        self._tracker = tracker or Sort(
            max_age=config.tracker.max_age,
            min_hits=config.tracker.min_hits,
            iou_threshold=config.tracker.iou_threshold,
        )
        self._formats = get_formats(config.plate_format.name)
        self._last_read: dict[int, int] = {}
        self.observations: list[TrackObservation] = []
        self.timer = StageTimer()
        self.counters: Counter[str] = Counter()
        self.frames_processed = 0

    def process_frame(self, index: int, frame: np.ndarray) -> list[TrackObservation]:
        """Process one frame and return what was observed in it.

        Frames must be passed in order and none may be skipped, because the
        tracker advances its motion model once per call.

        Args:
            index: Zero-based frame index, recorded in the log.
            frame: BGR image.
        """
        with self.timer.stage("vehicle_detection"):
            vehicles = self._vehicle_detector.detect(frame)
        with self.timer.stage("tracking"):
            tracks = self._update_tracks(vehicles)
        with self.timer.stage("plate_detection"):
            plates = self._find_plates(frame, tracks)
        with self.timer.stage("association"):
            plate_of = assign_plates_to_tracks(
                plates, tracks, min_overlap=self.config.min_plate_overlap
            )

        observations = []
        for track in tracks:
            plate = plate_of.get(track.track_id)
            read = fit = None
            if plate is not None and self._should_read(track.track_id, index, plate):
                with self.timer.stage("ocr"):
                    read = self._ocr.read(crop(frame, plate.box))
                self._last_read[track.track_id] = index
                self.counters["ocr_calls"] += 1
                if read is not None:
                    fit = fit_plate(
                        read.text,
                        self._formats,
                        max_substitutions=self.config.plate_format.max_substitutions,
                    )
                    self.counters["reads_fitting_format"] += fit is not None
            observations.append(
                TrackObservation(
                    frame=index,
                    track_id=track.track_id,
                    vehicle_box=track.box,
                    plate_box=plate.box if plate else None,
                    plate_score=plate.score if plate else None,
                    ocr_text=read.text if read else None,
                    ocr_confidence=read.confidence if read else None,
                    plate_text=fit.text if fit else None,
                )
            )

        self.counters["vehicle_detections"] += len(vehicles)
        self.counters["plate_detections"] += len(plates)
        self.counters["plates_matched_to_a_track"] += len(plate_of)
        self.observations.extend(observations)
        self.frames_processed += 1
        return observations

    def run(self, frames: Iterable[tuple[int, np.ndarray]]) -> list[TrackObservation]:
        """Process ``(index, frame)`` pairs and return the full log."""
        for index, frame in frames:
            self.process_frame(index, frame)
        return self.observations

    def results(self) -> list[PlateResult]:
        """Final plate per track, from everything processed so far."""
        return read_tracks(
            self.observations,
            formats=self._formats,
            max_substitutions=self.config.plate_format.max_substitutions,
            strategy=self.config.aggregation.strategy,
            min_reads=self.config.aggregation.min_reads,
        )

    def _update_tracks(self, vehicles: Sequence[Detection]) -> list[Track]:
        detections = np.array([[*v.box.as_tuple(), v.score] for v in vehicles]).reshape(-1, 5)
        rows = self._tracker.update(detections)
        tracks = [Track(int(row[4]), Box(*(float(v) for v in row[:4]))) for row in rows]
        return sorted(tracks, key=lambda track: track.track_id)

    def _find_plates(self, frame: np.ndarray, tracks: Sequence[Track]) -> list[Detection]:
        if self.config.plate_search == "frame":
            return self._plate_detector.detect(frame)

        # Search each tracked vehicle's crop. A plate that is tiny in the full
        # frame fills much more of the detector's input this way.
        crops, offsets = [], []
        for track in tracks:
            vehicle_crop, offset = crop_with_offset(frame, track.box)
            if vehicle_crop.size:
                crops.append(vehicle_crop)
                offsets.append(offset)
        plates = []
        for detections, (dx, dy) in zip(
            self._plate_detector.detect_batch(crops), offsets, strict=True
        ):
            plates.extend(Detection(d.box.shifted(dx, dy), d.score, d.class_id) for d in detections)
        return plates

    def _should_read(self, track_id: int, index: int, plate: Detection) -> bool:
        if plate.box.height < self.config.ocr.min_plate_height:
            return False
        last = self._last_read.get(track_id)
        return last is None or index - last >= self.config.ocr.every


def build_pipeline(config: PipelineConfig) -> PlateReaderPipeline:
    """Create a pipeline with real models, as described by ``config``.

    Raises:
        FileNotFoundError: If the plate detector weights are missing. Unlike
            the COCO vehicle weights they cannot be downloaded automatically.
    """
    config.validate()
    if not Path(config.plate.weights).exists():
        raise FileNotFoundError(
            f"plate detector weights not found: {config.plate.weights}. "
            "Train them with `alpr train` or see weights/README.md."
        )
    device = resolve_device(config.device)
    logger.info("building pipeline on %s", device)
    vehicle_detector = YoloDetector(
        config.vehicle.weights,
        conf=config.vehicle.conf,
        iou=config.vehicle.iou,
        imgsz=config.vehicle.imgsz,
        classes=list(VEHICLE_CLASSES),
        agnostic_nms=True,
        device=device,
    )
    plate_detector = YoloDetector(
        config.plate.weights,
        conf=config.plate.conf,
        iou=config.plate.iou,
        imgsz=config.plate.imgsz,
        device=device,
    )
    ocr = build_ocr(
        config.ocr.backend, preprocess=config.ocr.preprocess, gpu=device.startswith("cuda")
    )
    return PlateReaderPipeline(vehicle_detector, plate_detector, ocr, config)
