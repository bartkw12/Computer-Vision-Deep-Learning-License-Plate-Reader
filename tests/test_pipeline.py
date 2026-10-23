import dataclasses
import itertools

import pytest

import synthetic
from alpr.config import PipelineConfig, load_config
from alpr.pipeline import STAGES, PlateReaderPipeline, read_tracks
from alpr.plate_format import get_formats
from alpr.types import Box

TRUTH = {v.plate for v in synthetic.VEHICLES}


def run_scene(overrides=(), *, ocr_kwargs=None, **scene):
    config = load_config(overrides=["tracker.max_age=5", *overrides])
    vehicle_detector, plate_detector, ocr = synthetic.stand_ins(**(ocr_kwargs or {}))
    pipeline = PlateReaderPipeline(vehicle_detector, plate_detector, ocr, config)
    pipeline.run(synthetic.frames(**scene))
    return pipeline, ocr


def test_every_vehicle_is_tracked_once_and_read_correctly():
    pipeline, _ = run_scene()
    results = pipeline.results()
    assert {r.text for r in results} == TRUTH
    assert len(results) == len(synthetic.VEHICLES)  # no duplicate or phantom tracks
    assert pipeline.frames_processed == synthetic.N_FRAMES


def test_frames_without_vehicles_are_handled():
    """Regression: an empty frame while tracks were alive raised IndexError."""
    pipeline, _ = run_scene()
    busy = {row.frame for row in pipeline.observations}
    assert not busy & set(range(41, 48)), "the scene is empty in frames 41-47"
    assert max(busy) > 48, "tracking resumed after the gap"


def test_every_tracked_frame_is_logged_with_consistent_boxes():
    pipeline, _ = run_scene()
    first = synthetic.VEHICLES[0]
    track_id = next(r.track_id for r in pipeline.results() if r.text == first.plate)
    rows = [row for row in pipeline.observations if row.track_id == track_id]
    frames = [row.frame for row in rows]
    assert frames == list(range(frames[0], frames[-1] + 1)), "one row per frame, no holes"
    assert frames[-1] == first.last_frame
    for row in rows:
        assert row.plate_box is not None
        plate_cx, plate_cy = row.plate_box.center
        assert row.vehicle_box.contains_point(plate_cx, plate_cy)
        truth = first.box(row.frame)
        assert abs(row.vehicle_box.x1 - truth.x1) < 4 and abs(row.vehicle_box.y2 - truth.y2) < 4


def test_result_spans_cover_the_track_lifetime():
    pipeline, _ = run_scene()
    by_text = {r.text: r for r in pipeline.results()}
    for vehicle in synthetic.VEHICLES:
        result = by_text[vehicle.plate]
        assert result.last_frame == vehicle.last_frame
        # A track is reported once confirmed, a few frames after it appears.
        assert vehicle.first_frame <= result.first_frame <= vehicle.first_frame + 4


def test_plate_with_no_vehicle_is_ignored():
    """Regression: stray plates used to be pooled under car ID -1 and read."""
    pipeline, _ = run_scene(stray_plate=Box(560, 320, 600, 336))
    assert {r.text for r in pipeline.results()} == TRUTH
    assert all(row.track_id > 0 for row in pipeline.observations)
    assert pipeline.counters["plate_detections"] > pipeline.counters["plates_matched_to_a_track"]


def test_both_plate_search_modes_agree():
    whole_frame, _ = run_scene(["plate_search=frame"])
    per_vehicle, _ = run_scene(["plate_search=vehicle"])
    key = lambda r: (r.text, r.n_reads, r.first_frame, r.last_frame)  # noqa: E731
    assert sorted(map(key, whole_frame.results())) == sorted(map(key, per_vehicle.results()))
    for a, b in zip(whole_frame.observations, per_vehicle.observations, strict=True):
        assert (a.frame, a.track_id) == (b.frame, b.track_id)
        assert a.plate_box == pytest.approx(b.plate_box, abs=1.0)


def test_voting_survives_confident_ocr_mistakes():
    """Every fourth read is wrong with 0.99 confidence; correct reads have 0.8."""
    noisy = {"ocr_kwargs": {"corrupt_every": 4}}
    voted, _ = run_scene(["aggregation.strategy=string_vote"], **noisy)
    assert {r.text for r in voted.results()} == TRUTH
    # What the original did: keep the single most confident read.
    single, _ = run_scene(["aggregation.strategy=max_conf"], **noisy)
    assert not {r.text for r in single.results()} & TRUTH


def test_ocr_can_be_throttled_per_track():
    _, ocr_all = run_scene()
    throttled, ocr_third = run_scene(["ocr.every=3"])
    assert ocr_third.calls < ocr_all.calls / 2.5
    assert {r.text for r in throttled.results()} == TRUTH
    read_frames = [r.frame for r in throttled.observations if r.ocr_text and r.track_id == 1]
    assert all(b - a >= 3 for a, b in itertools.pairwise(read_frames))


def test_small_plates_can_be_skipped():
    pipeline, ocr = run_scene(["ocr.min_plate_height=50"])
    assert ocr.calls == 0
    assert pipeline.results() == []
    assert pipeline.observations, "vehicles are still tracked and logged"


def test_min_reads_suppresses_poorly_supported_plates():
    pipeline, _ = run_scene(["aggregation.min_reads=1000"])
    assert pipeline.results() == []


def test_reads_that_do_not_fit_the_format_are_logged_but_not_used():
    config = load_config(overrides=["tracker.max_age=5"])
    vehicle_detector, plate_detector, _ = synthetic.stand_ins()
    not_ontario = synthetic.ColorOCR({v.key: "AB12" for v in synthetic.VEHICLES})
    pipeline = PlateReaderPipeline(vehicle_detector, plate_detector, not_ontario, config)
    pipeline.run(synthetic.frames())
    assert pipeline.results() == []
    logged = [row for row in pipeline.observations if row.ocr_text]
    assert logged and all(row.ocr_text == "AB12" and row.plate_text is None for row in logged)
    assert pipeline.counters["reads_fitting_format"] == 0


def test_log_can_be_replayed_with_other_settings():
    """The raw OCR text is logged, so format and voting choices can be revisited."""
    pipeline, _ = run_scene(ocr_kwargs={"corrupt_every": 4})
    replay = lambda **kw: {  # noqa: E731
        r.text for r in read_tracks(pipeline.observations, formats=get_formats("ontario"), **kw)
    }
    assert replay(strategy="string_vote") == TRUTH
    assert replay(strategy="char_vote") == TRUTH
    assert not replay(strategy="max_conf") & TRUTH
    live = pipeline.results()
    assert live == read_tracks(pipeline.observations, formats=get_formats("ontario"))


def test_timing_and_counters_are_reported():
    pipeline, ocr = run_scene()
    summary = pipeline.timer.summary(pipeline.frames_processed)
    assert list(summary["stages"]) == list(STAGES)
    assert summary["frames"] == synthetic.N_FRAMES
    assert summary["fps"] > 0
    total = sum(stage["ms_per_frame"] for stage in summary["stages"].values())
    assert total == pytest.approx(summary["ms_per_frame"])
    assert sum(stage["share"] for stage in summary["stages"].values()) == pytest.approx(1.0)
    assert summary["stages"]["ocr"]["calls"] == ocr.calls == pipeline.counters["ocr_calls"]


def test_invalid_config_is_rejected_before_anything_runs():
    config = dataclasses.replace(PipelineConfig(), plate_search="satellite")
    with pytest.raises(ValueError, match="plate_search"):
        PlateReaderPipeline(*synthetic.stand_ins(), config)
