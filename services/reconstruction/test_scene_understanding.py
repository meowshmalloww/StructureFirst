from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from scene_understanding import CaptureMetadata, recognize_scene_frames


def test_scene_recognition_nominates_capture_continuity_not_outliers() -> None:
    base = datetime(2026, 7, 19, 12, 18, 3)
    captures = [
        CaptureMetadata("apple iphone", base),
        CaptureMetadata("apple iphone", base + timedelta(seconds=3)),
        CaptureMetadata("apple iphone", base + timedelta(seconds=7)),
        CaptureMetadata("apple iphone", base + timedelta(seconds=15)),
        CaptureMetadata(None, None),
        CaptureMetadata(None, None),
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.20, 0.0],
            [0.96, 0.25, 0.0],
            [0.70, 0.20, 0.10],
            [0.15, 0.98, 0.0],
            [-0.4, 0.1, 0.9],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    selected, reports, _ = recognize_scene_frames(
        [0, 1, 2],
        embeddings,
        captures,
    )

    assert selected == [0, 1, 2, 3]
    assert reports[0]["captureSessionMatch"] is True
    assert reports[1]["accepted"] is False
    assert reports[2]["accepted"] is False


def test_exact_capture_burst_can_cross_a_low_texture_wall() -> None:
    base = datetime(2026, 7, 19, 12, 18, 3)
    captures = [
        CaptureMetadata("apple iphone 15", base),
        CaptureMetadata("apple iphone 15", base + timedelta(seconds=4)),
        CaptureMetadata("apple iphone 15", base + timedelta(seconds=9)),
    ]
    embeddings = np.eye(3, dtype=np.float32)

    selected, reports, _ = recognize_scene_frames([0, 1], embeddings, captures)

    assert selected == [0, 1, 2]
    assert reports[0]["captureSessionMatch"] is True
    assert reports[0]["accepted"] is True
    assert "joint geometry" in reports[0]["reason"]
