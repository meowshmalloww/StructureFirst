from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from detector import MODEL_SIZE, _preprocess, detect_data_url, detector_status


def test_detector_model_is_available_and_identified() -> None:
    status = detector_status()
    assert status["available"] is True
    assert status["model"] == "YOLO26 Nano"
    assert status["license"] == "AGPL-3.0"
    assert status["output"] == "2D view-space observations"


def test_detector_accepts_a_local_rendered_frame() -> None:
    image = np.full((240, 320, 3), 114, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    data_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode()
    result = detect_data_url(data_url)
    assert result["model"] == "YOLO26 Nano"
    assert result["imageWidth"] == 320
    assert result["imageHeight"] == 240
    assert isinstance(result["detections"], list)


def test_detector_rejects_invalid_frames() -> None:
    with pytest.raises(ValueError, match="base64"):
        detect_data_url("data:image/jpeg;base64,not-valid")


def test_yolo26_preprocess_centers_letterbox_and_normalizes_rgb() -> None:
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    image[:, :, 2] = 255  # red in OpenCV BGR ordering

    tensor, ratio, pad_x, pad_y = _preprocess(image)

    assert tensor.shape == (1, 3, MODEL_SIZE, MODEL_SIZE)
    assert tensor.dtype == np.float32
    assert ratio == pytest.approx(1.6)
    assert pad_x == 0
    assert pad_y == 160
    assert tensor[0, 0, 320, 320] == pytest.approx(1.0)
    assert tensor[0, 1, 320, 320] == pytest.approx(0.0)
    assert tensor[0, 2, 320, 320] == pytest.approx(0.0)
    assert tensor[0, :, 0, 0].tolist() == pytest.approx([114 / 255] * 3)
