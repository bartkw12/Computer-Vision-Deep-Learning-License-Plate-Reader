import csv

import pytest

import synthetic
from alpr.config import load_config
from alpr.evaluation.ablation import (
    Variant,
    default_variants,
    format_modes,
    pipeline_grid,
    replay_grid,
)
from alpr.evaluation.video import read_labels, write_label_sheet
from alpr.io import VideoReader, VideoWriter
from alpr.pipeline import PlateReaderPipeline


@pytest.fixture(scope="module")
def labeled_scene(tmp_path_factory):
    """The synthetic scene as a video file, plus a filled-in label sheet for it."""
    folder = tmp_path_factory.mktemp("ablation")
    video = folder / "scene.mp4"
    with VideoWriter(video, fps=25, size=(synthetic.WIDTH, synthetic.HEIGHT)) as writer:
        for _, frame in synthetic.frames():
            writer.write(frame)

    pipeline = PlateReaderPipeline(
        *synthetic.stand_ins(), load_config(overrides=["tracker.max_age=5"])
    )
    with VideoReader(video) as reader:
        pipeline.run(reader)
    sheet = folder / "labels.csv"
    write_label_sheet(pipeline.observations, video, sheet)
    with sheet.open(newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        frame, x, y = int(row["frame"]), float(row["x"]), float(row["y"])
        row["plate"] = next(
            v.plate
            for v in synthetic.VEHICLES
            if v.box(frame) is not None and v.box(frame).contains_point(x, y)
        )
    with sheet.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return video, read_labels(sheet)


def noisy_factory(config):
    """Stand-in pipeline whose OCR is confidently wrong on every fourth read."""
    return PlateReaderPipeline(*synthetic.stand_ins(corrupt_every=4), config)


def test_format_modes_include_the_original_regex_only_for_ontario():
    assert [mode[0] for mode in format_modes("ontario")] == [
        "off", "regex", "strict", "lookalike-1", "lookalike-2",
    ]  # fmt: skip
    assert [mode[0] for mode in format_modes("3L4D")] == [
        "off", "strict", "lookalike-1", "lookalike-2",
    ]  # fmt: skip
    assert format_modes("ontario")[1] == ("regex", "4L3D", 0)


def test_replay_grid_covers_every_combination(labeled_scene):
    video, labels = labeled_scene
    pipeline = noisy_factory(load_config(overrides=["tracker.max_age=5"]))
    with VideoReader(video) as reader:
        pipeline.run(reader)

    rows = replay_grid(pipeline.observations, labels, "ontario")
    assert len(rows) == 5 * 3
    by_key = {(row["format_mode"], row["aggregation"]): row for row in rows}
    # Voting recovers every plate; trusting the single most confident read recovers none.
    assert by_key["lookalike-2", "string_vote"]["exact"] == 3
    assert by_key["lookalike-2", "char_vote"]["exact"] == 3
    assert by_key["lookalike-2", "max_conf"]["exact"] == 0
    assert all(row["n"] == 3 and row["tracked"] == 3 for row in rows)
    assert all(row["tracks_per_vehicle"] == 1.0 for row in rows)


def test_pipeline_grid_runs_each_variant_and_reports_cost(labeled_scene):
    video, labels = labeled_scene
    variants = [
        Variant("baseline", "defaults"),
        Variant("ocr_cost", "every=4", ("ocr.every=4",)),
        Variant("tracker", "max_age=1", ("tracker.max_age=1",)),
        Variant("plate_search", "per-vehicle crops", ("plate_search=vehicle",)),
    ]
    rows = pipeline_grid(
        video,
        labels,
        variants,
        lambda config: PlateReaderPipeline(*synthetic.stand_ins(), config),
        base_overrides=["tracker.max_age=5"],
        progress=False,
    )
    assert [(row["group"], row["variant"]) for row in rows] == [(v.group, v.name) for v in variants]
    baseline, throttled, short_lived, per_vehicle = rows
    assert all(row["exact"] == 3 and row["precision"] == 1.0 for row in rows)
    assert throttled["ocr_calls"] < baseline["ocr_calls"] / 3
    assert throttled["overrides"] == {"ocr": {"every": 4}}
    assert short_lived["overrides"] == {"tracker": {"max_age": 1}}
    assert per_vehicle["ocr_calls"] == baseline["ocr_calls"]
    for row in rows:
        assert row["fps"] > 0
        assert set(row["stage_ms_per_frame"]) == {
            "vehicle_detection", "tracking", "plate_detection", "association", "ocr",
        }  # fmt: skip


def test_pipeline_grid_can_stop_early(labeled_scene):
    video, labels = labeled_scene
    (row,) = pipeline_grid(
        video,
        labels,
        [Variant("baseline", "defaults")],
        lambda config: PlateReaderPipeline(*synthetic.stand_ins(), config),
        max_frames=45,
        progress=False,
    )
    # The third vehicle only appears at frame 48.
    assert row["n"] == 3 and row["tracked"] == 2 and row["exact"] == 2


def test_default_variants_are_valid_configurations():
    variants = default_variants()
    assert variants[0] == Variant("baseline", "defaults")
    assert {v.group for v in variants} == {
        "baseline", "tracker", "plate_search", "ocr_backend", "ocr_preprocess", "ocr_cost",
    }  # fmt: skip
    assert len({(v.group, v.name) for v in variants}) == len(variants)
    for variant in variants:
        load_config(overrides=list(variant.overrides))  # raises on a typo
