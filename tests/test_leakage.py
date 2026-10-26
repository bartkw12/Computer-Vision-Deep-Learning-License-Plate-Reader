import cv2
import numpy as np
import pytest

from alpr.evaluation.leakage import dhash, leakage_report, nearest, split_images


def scene(seed, size=(240, 320)):
    """A smooth random picture: blurred noise, so a few pixels of shift keep it similar."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
    return cv2.resize(small, size[::-1], interpolation=cv2.INTER_CUBIC)


def distance(a, b):
    distances, _ = nearest(dhash(a)[None], dhash(b)[None])
    return int(distances[0])


def test_identical_images_hash_identically():
    assert distance(scene(0), scene(0)) == 0


def test_resized_and_recompressed_copy_stays_close():
    image = scene(1)
    smaller = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
    _, jpeg = cv2.imencode(".jpg", smaller, [cv2.IMWRITE_JPEG_QUALITY, 40])
    assert distance(image, cv2.imdecode(jpeg, cv2.IMREAD_COLOR)) <= 4


def test_slightly_shifted_frame_stays_close():
    image = scene(2)
    shifted = np.roll(image, 2, axis=1)
    assert distance(image, shifted) <= 10


def test_unrelated_images_are_far_apart():
    assert min(distance(scene(3), scene(seed)) for seed in range(10, 20)) > 10


def test_nearest_picks_the_closest_reference():
    references = np.stack([dhash(scene(seed)) for seed in (5, 6, 7)])
    distances, indices = nearest(np.stack([dhash(scene(6)), dhash(scene(7))]), references, chunk=1)
    assert indices.tolist() == [1, 2]
    assert distances.tolist() == [0, 0]


@pytest.fixture
def dataset(tmp_path):
    """train has seeds 0-5; valid repeats seed 0; test repeats seed 1 (re-encoded) and valid's 9."""
    layout = {"train": range(6), "valid": [0, 9, 20], "test": [1, 9, 30, 31]}
    for split, seeds in layout.items():
        folder = tmp_path / split / "images"
        folder.mkdir(parents=True)
        for seed in seeds:
            cv2.imwrite(str(folder / f"s{seed}.jpg"), scene(seed), [cv2.IMWRITE_JPEG_QUALITY, 70])
    return tmp_path


def test_split_images_reads_yolo_and_flat_layouts(tmp_path, dataset):
    assert [p.name for p in split_images(dataset, "valid")] == ["s0.jpg", "s20.jpg", "s9.jpg"]
    flat = tmp_path / "flat" / "test"
    flat.mkdir(parents=True)
    cv2.imwrite(str(flat / "a.png"), scene(0))
    (flat / "a.txt").write_text("not an image")
    assert [p.name for p in split_images(tmp_path / "flat", "test")] == ["a.png"]
    assert split_images(tmp_path / "flat", "train") == []


def test_report_flags_exactly_the_shared_images(dataset):
    report = leakage_report(dataset, thresholds=(0, 4), progress=False)
    assert report["split_sizes"] == {"train": 6, "valid": 3, "test": 4}
    pairs = report["pairs"]
    assert set(pairs) == {"valid_vs_train", "test_vs_train", "test_vs_valid"}

    assert pairs["valid_vs_train"]["within"]["4"]["count"] == 1
    assert pairs["test_vs_train"]["within"]["4"]["count"] == 1
    assert pairs["test_vs_valid"]["within"]["4"]["count"] == 1
    assert pairs["test_vs_valid"]["within"]["4"]["share"] == pytest.approx(0.25)

    closest = pairs["test_vs_train"]["closest_pairs"][0]
    assert closest == {
        "query": "test/images/s1.jpg",
        "reference": "train/images/s1.jpg",
        "distance": closest["distance"],
    }
    assert closest["distance"] <= 4


def test_missing_split_is_reported_not_fatal(dataset):
    for path in (dataset / "valid" / "images").iterdir():
        path.unlink()
    report = leakage_report(dataset, progress=False)
    assert report["pairs"]["valid_vs_train"] == {"queries": 0, "references": 6, "within": {}}
