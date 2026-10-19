"""Tests for the detection pipeline (model-free parts + stubbed inference).

These tests do **not** download weights: the YOLO backend is replaced with a
stub so the detection flow (filtering, NMS, drawing, counting) can be verified
offline.  Real-model behaviour is covered by the documented smoke test.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src import detector as det_mod
from src.detector import (
    Detection,
    DetectionResult,
    WeightsNotFoundError,
    filter_detections,
    is_tree_class_name,
    nms,
    resolve_weights,
    validate_class_names,
)
from src.preprocessing import InputValidationError, list_images_in_folder, load_image
from src.visualization import draw_detections, side_by_side


# --------------------------------------------------------------------------- #
# Detection container & filtering
# --------------------------------------------------------------------------- #

def _det(x1, y1, x2, y2, conf=0.9, cls="tree_trunk", name="img.jpg") -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf,
                     class_id=0, class_name=cls, source_name=name)


def test_detection_bbox_properties() -> None:
    d = _det(10, 20, 50, 80, conf=0.75)
    assert d.width == 40
    assert d.height == 60
    assert d.area == 2400
    assert d.center == (30.0, 50.0)
    assert d.to_dict()["confidence"] == pytest.approx(0.75)


def test_confidence_filtering() -> None:
    dets = [_det(0, 0, 10, 10, conf=0.9), _det(20, 20, 30, 30, conf=0.3)]
    result = DetectionResult("a.jpg", dets, 12.0, 100, 100)
    assert result.count == 2
    filtered = result.filtered(0.5)
    assert filtered.count == 1
    assert filtered.detections[0].confidence == pytest.approx(0.9)
    # Original result must not be mutated.
    assert result.count == 2


def test_empty_result_is_safe() -> None:
    result = DetectionResult("a.jpg", [], 5.0, 64, 64)
    assert result.count == 0
    assert result.boxes.shape == (0, 4)
    assert result.confidences.shape == (0,)
    assert result.to_dict()["detections"] == []
    # Drawing with no detections must return an image, not None.
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    annotated = draw_detections(img, result)
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == img.shape
    assert np.array_equal(img, np.zeros_like(img))  # input untouched


def test_nms_suppresses_overlaps() -> None:
    boxes = np.array(
        [[0, 0, 100, 100], [5, 5, 105, 105], [200, 200, 300, 300]], dtype=np.float32
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert keep == [0, 2]


def test_nms_empty_input() -> None:
    assert nms(np.zeros((0, 4)), np.zeros((0,))) == []


def test_filter_detections_combines_conf_and_nms() -> None:
    dets = [
        _det(0, 0, 100, 100, conf=0.9),
        _det(5, 5, 105, 105, conf=0.6),   # overlaps -> suppressed
        _det(0, 0, 50, 50, conf=0.2),      # below threshold -> dropped
    ]
    kept = filter_detections(dets, confidence_threshold=0.4, iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0].confidence == pytest.approx(0.9)


def test_iou_edge_cases() -> None:
    from src.detector import box_iou

    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3, abs=1e-5)


# --------------------------------------------------------------------------- #
# Class capability validation
# --------------------------------------------------------------------------- #

def test_class_name_detection() -> None:
    assert is_tree_class_name("tree_trunk")
    assert is_tree_class_name("Tree Trunk")
    assert not is_tree_class_name("person")
    assert not is_tree_class_name("car")


def test_validate_class_names_coco_vs_custom() -> None:
    is_tree, names = validate_class_names({0: "person", 1: "bicycle", 2: "car"})
    assert not is_tree and len(names) == 3

    is_tree, names = validate_class_names({0: "tree_trunk"})
    assert is_tree and names == ["tree_trunk"]


def test_resolve_weights_missing_gives_helpful_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(det_mod, "models_dir", lambda: tmp_path)
    with pytest.raises(WeightsNotFoundError) as exc:
        resolve_weights(None)
    assert "No model weights" in str(exc.value)

    with pytest.raises(WeightsNotFoundError):
        resolve_weights("does_not_exist.pt")


def test_list_available_weights(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(det_mod, "models_dir", lambda: tmp_path)
    assert det_mod.list_available_weights() == []
    (tmp_path / "tree_trunk.pt").write_bytes(b"stub")
    (tmp_path / "notes.txt").write_text("ignore")
    found = det_mod.list_available_weights()
    assert [p.name for p in found] == ["tree_trunk.pt"]
    assert resolve_weights("tree_trunk.pt") == tmp_path / "tree_trunk.pt"


# --------------------------------------------------------------------------- #
# Detector flow with a stubbed backend (no weights download)
# --------------------------------------------------------------------------- #

class _StubBoxes:
    def __init__(self, xyxy, conf, cls) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.conf)


class _StubModel:
    """Mimics the slice of the Ultralytics API used by TreeDetector."""

    def __init__(self, names, boxes) -> None:
        self.names = names
        self._boxes = boxes

    def predict(self, source, **kwargs):
        return [SimpleNamespace(boxes=self._boxes)]


def _make_detector(monkeypatch, tmp_path, names, boxes) -> "det_mod.TreeDetector":
    weights = tmp_path / "stub.pt"
    weights.write_bytes(b"stub")
    monkeypatch.setattr(det_mod, "_load_model", lambda path: _StubModel(names, boxes))
    return det_mod.TreeDetector(weights=weights)


def test_detector_end_to_end_with_stub(monkeypatch, tmp_path) -> None:
    boxes = _StubBoxes([[10, 10, 60, 200], [100, 10, 150, 210], [12, 12, 62, 205]],
                       [0.9, 0.8, 0.5], [0, 0, 0])
    det = _make_detector(monkeypatch, tmp_path, {0: "tree_trunk"}, boxes)
    assert det.supports_tree_class and det.is_tree_model

    image = np.full((240, 320, 3), 120, dtype=np.uint8)
    result = det.detect(image, source_name="synthetic.jpg", frame_index=0)
    # NMS should collapse the third overlapping box; two trees remain.
    assert result.count == 2
    assert result.model_name == "stub.pt"
    assert result.is_tree_model is True
    assert result.inference_time_ms >= 0
    assert all(d.class_name == "tree_trunk" for d in result.detections)
    assert all(0 <= d.confidence <= 1 for d in result.detections)
    assert result.boxes.shape == (2, 4)
    assert result.to_dict()["count"] == 2


def test_detector_flags_non_tree_model(monkeypatch, tmp_path) -> None:
    boxes = _StubBoxes([[0, 0, 30, 30]], [0.7], [0])
    det = _make_detector(monkeypatch, tmp_path, {0: "person"}, boxes)
    assert not det.supports_tree_class
    result = det.detect(np.zeros((50, 50, 3), dtype=np.uint8))
    assert result.is_tree_model is False
    assert "person" in result.class_names


def test_detector_respects_confidence_threshold(monkeypatch, tmp_path) -> None:
    boxes = _StubBoxes([[0, 0, 30, 30], [60, 60, 90, 90]], [0.9, 0.2], [0, 0])
    det = _make_detector(monkeypatch, tmp_path, {0: "tree_trunk"}, boxes)
    det.confidence_threshold = 0.5
    result = det.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result.count == 1
    assert result.detections[0].confidence == pytest.approx(0.9)


def test_detector_empty_predictions(monkeypatch, tmp_path) -> None:
    det = _make_detector(monkeypatch, tmp_path, {0: "tree_trunk"}, _StubBoxes([], [], []))
    result = det.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    assert result.count == 0
    assert result.boxes.shape == (0, 4)


def test_model_cache_avoids_reloading(monkeypatch, tmp_path) -> None:
    """Two constructions from the same file must load weights from disk once."""
    weights = tmp_path / "cached.pt"
    weights.write_bytes(b"stub")
    load_calls: list[str] = []

    class FakeYOLO:
        def __init__(self, path):
            load_calls.append(path)

    monkeypatch.setattr(det_mod, "_import_ultralytics", lambda: FakeYOLO)
    det_mod._MODEL_CACHE.clear()
    try:
        a = det_mod.TreeDetector(weights=weights)
        b = det_mod.TreeDetector(weights=weights)
        assert len(load_calls) == 1, "weights must be loaded only once and then cached"
        assert a.names == b.names == {}
        # Cache key includes mtime: touching the file forces a reload.
        import os
        import time

        os.utime(weights, (time.time() + 10, time.time() + 10))
        det_mod.TreeDetector(weights=weights)
        assert len(load_calls) == 2
    finally:
        det_mod._MODEL_CACHE.clear()


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

def test_draw_detections_adds_boxes_and_count(synthetic_forest) -> None:
    image, boxes = synthetic_forest
    dets = [_det(*b, conf=0.8) for b in boxes]
    result = DetectionResult("synth.jpg", dets, 10.0, image.shape[1], image.shape[0],
                             is_tree_model=True)
    annotated = draw_detections(image, result)
    assert annotated.shape == image.shape
    # Annotation must change pixels (boxes/count text drawn).
    assert not np.array_equal(annotated, image)
    # Original image must stay unchanged.
    assert image.dtype == np.uint8
    combined = side_by_side(image, annotated)
    assert combined.shape[1] == image.shape[1] * 2


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #

def test_load_image_missing_and_corrupt(tmp_path) -> None:
    with pytest.raises(InputValidationError):
        load_image(tmp_path / "nope.jpg")
    bad = tmp_path / "corrupt.jpg"
    bad.write_bytes(b"this is not an image")
    with pytest.raises(InputValidationError):
        load_image(bad)
    wrong = tmp_path / "notes.txt"
    wrong.write_text("hello")
    with pytest.raises(InputValidationError):
        load_image(wrong)


def test_list_images_in_folder(image_dir, tmp_path) -> None:
    imgs = list_images_in_folder(image_dir)
    assert len(imgs) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(InputValidationError):
        list_images_in_folder(empty)
    with pytest.raises(InputValidationError):
        list_images_in_folder(tmp_path / "missing_dir")


def test_video_roundtrip(tmp_path) -> None:
    """Write a small synthetic video, process it with a stub detector, verify output."""
    from src.video_processor import process_video

    src = tmp_path / "in.mp4"
    size = (96, 64)
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    if not writer.isOpened():
        pytest.skip("VideoWriter unavailable in this environment")
    for i in range(8):
        frame = np.full((size[1], size[0], 3), 30 + i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    if not src.exists() or src.stat().st_size == 0:
        pytest.skip("Video encoding produced no data in this environment")

    class StubDetector:
        backend_device = "cpu"
        model_name = "stub"

        def detect(self, frame, source_name="", frame_index=-1):
            return DetectionResult(
                source_name=source_name,
                detections=[_det(5, 5, 30, 60, conf=0.88, name=source_name)],
                inference_time_ms=3.0,
                image_width=frame.shape[1],
                image_height=frame.shape[0],
                frame_index=frame_index,
            )

    out = tmp_path / "out.mp4"
    summary = process_video(StubDetector(), src, out, max_frames=8)
    assert summary.frames_processed == 8
    assert summary.total_detections == 8
    assert summary.mean_inference_ms > 0
    assert out.exists() and out.stat().st_size > 0

    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None
