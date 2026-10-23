import csv

import cv2
import numpy as np
import pytest

import synthetic
from alpr.config import load_config
from alpr.evaluation.images import evaluate_end_to_end, evaluate_recognition, load_openalpr
from alpr.evaluation.video import (
    Label,
    evaluate_video,
    match_labels_to_tracks,
    read_labels,
    write_label_sheet,
)
from alpr.io import VideoReader, VideoWriter
from alpr.pipeline import PlateReaderPipeline
from alpr.plate_format import get_formats
from alpr.types import Box, Detection, PlateRead, PlateResult, TrackObservation

# --- still images ---------------------------------------------------------------


class FixedOCR:
    """Reads a crop by its brightness: a lookup that stands in for a real model."""

    name = "fixed"
    preprocess_name = "none"

    def __init__(self, by_brightness):
        self.by_brightness = by_brightness

    def read(self, crop):
        text = self.by_brightness.get(int(round(crop.mean() / 10) * 10))
        return PlateRead(text, 0.9) if text else None


REGIONS = ["us", "eu"]


@pytest.fixture
def benchmark(tmp_path):
    """Three labeled images in OpenALPR's layout; each plate is a flat gray patch."""
    plates = [
        ("us", "a.jpg", 60, "ABC123", (100, 80, 60, 24)),
        ("us", "b.jpg", 120, "XYZ789", (40, 40, 120, 50)),
        ("eu", "c.jpg", 180, "KA PL 1234", (10, 10, 200, 80)),
    ]
    for region, name, brightness, text, (x, y, w, h) in plates:
        folder = tmp_path / "endtoend" / region
        folder.mkdir(parents=True, exist_ok=True)
        image = np.zeros((240, 320, 3), np.uint8)
        image[y : y + h, x : x + w] = brightness
        cv2.imwrite(str(folder / name), image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        (folder / name).with_suffix(".txt").write_text(f"{name}\t{x}\t{y}\t{w}\t{h}\t{text}\n")
    return tmp_path


def test_load_openalpr(benchmark):
    images = load_openalpr(benchmark, REGIONS)
    assert [(i.region, i.path.name, i.text) for i in images] == [
        ("us", "a.jpg", "ABC123"),
        ("us", "b.jpg", "XYZ789"),
        ("eu", "c.jpg", "KAPL1234"),
    ]
    assert images[0].box == Box(100, 80, 160, 104)
    assert len(load_openalpr(benchmark / "endtoend", ["eu"])) == 1


def test_load_openalpr_reports_a_missing_region(benchmark):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_openalpr(benchmark, ["br"])


def test_load_openalpr_skips_broken_labels(benchmark):
    folder = benchmark / "endtoend" / "us"
    (folder / "broken.txt").write_text("broken.jpg\t1\t2\n")
    (folder / "orphan.txt").write_text("orphan.jpg\t1\t2\t3\t4\tABC123\n")
    assert len(load_openalpr(benchmark, ["us"])) == 2


def test_recognition_is_scored_overall_by_region_and_by_size(benchmark):
    ocr = FixedOCR({60: "ABC123", 120: "XYZ780", 180: None})  # right, one char off, unread
    report = evaluate_recognition(load_openalpr(benchmark, REGIONS), ocr, progress=False)

    assert report["overall"]["n"] == 3 and report["overall"]["exact"] == 1
    assert report["overall"]["cer"] == pytest.approx((0 + 1 + 8) / (6 + 6 + 8))
    assert report["by_region"]["us"]["exact"] == 1 and report["by_region"]["us"]["n"] == 2
    assert report["by_region"]["eu"]["accuracy"] == 0
    assert set(report["by_plate_height"]) == {"20-30px", "45-70px", "70px+"}
    assert report["top_confusions"] == [{"truth": "9", "predicted": "0", "count": 1}]
    assert report["ms_per_plate"] >= 0
    assert [r["prediction"] for r in report["records"]] == ["ABC123", "XYZ780", None]


class OneBoxDetector:
    """Returns a preset box per image height-independent call order."""

    def __init__(self, boxes):
        self.boxes = iter(boxes)

    def detect(self, image):
        box = next(self.boxes)
        return [] if box is None else [Detection(box, 0.9)]


def test_end_to_end_counts_undetected_plates_as_wrong(benchmark):
    images = load_openalpr(benchmark, REGIONS)
    detector = OneBoxDetector(
        [
            Box(102, 81, 161, 105),  # a: good overlap -> read
            Box(40, 40, 70, 50),  # b: far too small -> IoU below 0.5, not detected
            None,  # c: nothing found
        ]
    )
    ocr = FixedOCR({60: "ABC123", 120: "XYZ789", 180: "KAPL1234"})
    report = evaluate_end_to_end(images, detector, ocr, progress=False)

    assert report["detection"]["overall"] == {
        "n": 3,
        "detected": 1,
        "recall": pytest.approx(1 / 3),
        "mean_iou_when_detected": pytest.approx(0.85, abs=0.06),
    }
    assert report["detection"]["by_region"]["eu"]["recall"] == 0
    assert report["reading"]["overall"]["exact"] == 1
    assert [r["prediction"] for r in report["records"]] == ["ABC123", None, None]


# --- video ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def scene_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "scene.mp4"
    with VideoWriter(path, fps=25, size=(synthetic.WIDTH, synthetic.HEIGHT)) as writer:
        for _, frame in synthetic.frames():
            writer.write(frame)
    return path


def run_scene(video_path, overrides=("tracker.max_age=5",), **ocr_kwargs):
    pipeline = PlateReaderPipeline(
        *synthetic.stand_ins(**ocr_kwargs), load_config(overrides=list(overrides))
    )
    with VideoReader(video_path) as video:
        pipeline.run(video)
    return pipeline


def truth_at(frame, x, y):
    for vehicle in synthetic.VEHICLES:
        box = vehicle.box(frame)
        if box is not None and box.contains_point(x, y):
            return vehicle.plate
    raise AssertionError(f"no vehicle at frame {frame}, ({x}, {y})")


def fill_in(csv_path, answers=None):
    """Type the true plate into every row, as a person would."""
    with csv_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        truth = truth_at(int(row["frame"]), float(row["x"]), float(row["y"]))
        row["plate"] = (answers or {}).get(truth, truth)
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_label_sheet_lists_every_vehicle_without_revealing_predictions(scene_video, tmp_path):
    pipeline = run_scene(scene_video)
    sheet = tmp_path / "labels" / "scene.csv"
    assert write_label_sheet(pipeline.observations, scene_video, sheet) == 3

    text = sheet.read_text()
    assert text.splitlines()[0] == "label_id,frame,x,y,plate"
    assert not any(vehicle.plate in text for vehicle in synthetic.VEHICLES)
    assert all(label.plate == "" and not label.readable for label in read_labels(sheet))

    image = cv2.imread(str(tmp_path / "labels" / "scene_sheet_01.jpg"))
    assert image is not None and image.shape[0] == 3 * 180


def test_label_sheet_is_never_overwritten_by_accident(scene_video, tmp_path):
    pipeline = run_scene(scene_video)
    sheet = tmp_path / "scene.csv"
    write_label_sheet(pipeline.observations, scene_video, sheet)
    fill_in(sheet)
    typed = sheet.read_text()
    with pytest.raises(FileExistsError, match="already exists"):
        write_label_sheet(pipeline.observations, scene_video, sheet)
    assert sheet.read_text() == typed
    write_label_sheet(pipeline.observations, scene_video, sheet, overwrite=True)
    assert sheet.read_text() != typed


def test_perfect_run_scores_perfectly(scene_video, tmp_path):
    pipeline = run_scene(scene_video)
    sheet = tmp_path / "scene.csv"
    write_label_sheet(pipeline.observations, scene_video, sheet)
    fill_in(sheet)

    evaluation = evaluate_video(
        pipeline.observations,
        pipeline.results(),
        read_labels(sheet),
        formats=get_formats("ontario"),
    )
    summary = evaluation["summary"]
    assert summary["reading"]["exact"] == summary["readable"] == 3
    assert summary["reading"]["accuracy"] == 1.0 and summary["reading"]["cer"] == 0.0
    assert summary["precision"] == 1.0
    assert summary["tracks_per_vehicle"] == 1.0
    assert summary["true_plates_outside_assumed_format"] == 0
    assert summary["plates_emitted_on_unlabeled_tracks"] == 0


def test_summary_contains_no_plate_strings(scene_video, tmp_path):
    pipeline = run_scene(scene_video)
    sheet = tmp_path / "scene.csv"
    write_label_sheet(pipeline.observations, scene_video, sheet)
    fill_in(sheet)
    evaluation = evaluate_video(pipeline.observations, pipeline.results(), read_labels(sheet))
    published = repr(evaluation["summary"])
    assert not any(vehicle.plate in published for vehicle in synthetic.VEHICLES)
    assert {r["truth"] for r in evaluation["records"]} == {v.plate for v in synthetic.VEHICLES}


def test_labels_stay_valid_when_the_tracker_changes(scene_video, tmp_path):
    """Labels are anchored to a frame and pixel, so they survive new track IDs."""
    baseline = run_scene(scene_video)
    sheet = tmp_path / "scene.csv"
    write_label_sheet(baseline.observations, scene_video, sheet)
    fill_in(sheet)
    labels = read_labels(sheet)

    other = run_scene(scene_video, overrides=("tracker.max_age=1", "tracker.min_hits=1"))
    assert other.observations != baseline.observations
    summary = evaluate_video(other.observations, other.results(), labels)["summary"]
    assert summary["tracked"] == 3 and summary["reading"]["accuracy"] == 1.0


def test_wrong_reads_and_special_rows_are_counted(scene_video, tmp_path):
    # Every read of every plate is corrupted, so the system is confidently wrong.
    pipeline = run_scene(scene_video, corrupt_every=1)
    sheet = tmp_path / "scene.csv"
    write_label_sheet(pipeline.observations, scene_video, sheet)
    fill_in(sheet, answers={"CKLM456": "?", "DRST789": ""})

    summary = evaluate_video(pipeline.observations, pipeline.results(), read_labels(sheet))[
        "summary"
    ]
    assert (summary["labels"], summary["readable"]) == (3, 1)
    assert (summary["unreadable"], summary["unlabeled"]) == (1, 1)
    assert summary["reading"]["exact"] == 0
    assert summary["reading"]["cer"] == pytest.approx(1 / 7)
    assert summary["emitted"] == 1 and summary["precision"] == 0.0
    assert summary["top_confusions"] == [{"truth": "3", "predicted": "8", "count": 1}]
    assert summary["plates_emitted_on_unlabeled_tracks"] == 2


def observation(frame, track_id, x):
    return TrackObservation(frame, track_id, Box(x, 100, x + 100, 160))


def test_vehicle_split_across_two_tracks_is_measured():
    rows = [observation(f, 1, 10 * f) for f in range(5)]
    rows += [observation(f, 2, 10 * f) for f in range(8, 13)]  # same car, new ID
    labels = [Label(1, 2, 70, 130, "BXYZ123"), Label(2, 10, 150, 130, "BXYZ123")]
    results = [PlateResult(1, "BXYZ123", 1.0, 5, 0, 4), PlateResult(2, "BXYZ128", 1.0, 5, 8, 12)]
    summary = evaluate_video(rows, results, labels)["summary"]
    assert summary["tracks_per_vehicle"] == 2.0
    assert summary["unique_plates"] == 1 and summary["unique_plates_read_correctly"] == 1
    assert summary["reading"]["exact"] == 1 and summary["reading"]["n"] == 2


def test_label_matching_looks_at_nearby_frames_and_prefers_the_smallest_box():
    rows = [
        TrackObservation(10, 1, Box(0, 0, 400, 300)),
        TrackObservation(10, 2, Box(100, 100, 200, 180)),
        TrackObservation(13, 3, Box(500, 100, 600, 180)),
    ]
    labels = [
        Label(1, 10, 150, 150, "A"),  # inside both boxes -> the smaller, track 2
        Label(2, 11, 550, 150, "B"),  # nothing at frame 11; track 3 is 2 frames later
        Label(3, 10, 630, 350, "C"),  # on no vehicle
        Label(4, 30, 150, 150, "D"),  # too far away in time
    ]
    assert match_labels_to_tracks(labels, rows, window=5) == {1: 2, 2: 3, 3: None, 4: None}
    assert match_labels_to_tracks(labels, rows, window=1)[2] is None


def test_true_plates_outside_the_assumed_format_are_reported():
    rows = [observation(0, 1, 0), observation(0, 2, 300)]
    labels = [Label(1, 0, 50, 130, "BXYZ123"), Label(2, 0, 350, 130, "AB12345")]
    summary = evaluate_video(rows, [], labels, formats=get_formats("ontario"))["summary"]
    assert summary["true_plates_outside_assumed_format"] == 1
    assert evaluate_video(rows, [], labels)["summary"]["true_plates_outside_assumed_format"] is None
