from pathlib import Path
import os
import sys

import numpy as np
import cv2
import pytest

SERVICE_ROOT = Path(__file__).resolve().parent
LUCIDFRAME_BACKEND = Path(
    os.getenv("LUCIDFRAME_ROOT", str(SERVICE_ROOT.parent.parent.parent / "LucidFrame"))
) / "backend"
sys.path.insert(0, str(LUCIDFRAME_BACKEND))
sys.path.insert(0, str(SERVICE_ROOT))

import smart_connect


def _frame(index: int) -> smart_connect.PreflightFrame:
    return smart_connect.PreflightFrame(
        index=index,
        path=Path(f"frame_{index}.jpg"),
        keypoints=[],
        descriptors=None,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
        source_size=(32, 32),
    )


def test_capture_set_rejects_a_singleton_without_geometric_overlap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = [_frame(index) for index in range(10)]
    monkeypatch.setattr(
        smart_connect,
        "_preflight_selection",
        lambda _paths: (frames, [], [], [3], [3], {}),
    )

    with pytest.raises(smart_connect.RegistrationError) as caught:
        smart_connect.reconstruct_connected(
            [Path(f"frame_{index}.jpg") for index in range(10)],
            tmp_path,
        )

    assert caught.value.report["connectedFrameCount"] == 1
    assert caught.value.report["minimumConnectedFrames"] == 2
    assert caught.value.report["connectedFrames"] == [3]
    assert "Only 1/10 photographs" in str(caught.value)


@pytest.mark.parametrize(
    ("frame_count", "required"),
    [(2, 2), (3, 2), (7, 2), (10, 2), (20, 2)],
)
def test_minimum_connected_frames_accepts_a_verified_partial_scene(
    frame_count: int,
    required: int,
) -> None:
    assert smart_connect.minimum_connected_frames(frame_count) == required


@pytest.mark.parametrize(
    ("frame_count", "required"),
    [(2, 2), (3, 2), (7, 4), (10, 5), (20, 10)],
)
def test_representative_connected_frames_keeps_half_the_capture_set(
    frame_count: int,
    required: int,
) -> None:
    assert smart_connect.representative_connected_frames(frame_count) == required


def test_planar_repeating_matches_are_detected_as_ambiguous() -> None:
    grid = np.asarray(
        [(x, y) for y in range(20, 220, 20) for x in range(20, 300, 20)],
        dtype=np.float32,
    )
    transform = np.asarray(
        [[1.02, 0.04, 18.0], [-0.03, 0.98, 11.0], [0.0002, 0.0001, 1.0]],
        dtype=np.float32,
    )
    projected = cv2.perspectiveTransform(grid[None, :, :], transform)[0]
    assert smart_connect._homography_dominance(
        grid, projected, threshold_px=2.0
    ) > 0.95


def test_match_spread_distinguishes_room_coverage_from_one_floor_patch() -> None:
    broad = np.asarray(
        [(x, y) for y in (20, 120, 220, 300) for x in (20, 120, 220, 300)],
        dtype=np.float32,
    )
    patch = np.asarray(
        [(x, y) for y in (210, 240) for x in (30, 80, 130, 180)],
        dtype=np.float32,
    )
    broad_spread = smart_connect._match_spread(
        broad, broad + 2, (320, 320), (320, 320)
    )
    patch_spread = smart_connect._match_spread(
        patch, patch + 2, (320, 320), (320, 320)
    )
    assert broad_spread["minimumGridCells"] >= 8
    assert broad_spread["minimumHullCoverage"] >= 0.2
    assert patch_spread["minimumHullCoverage"] < 0.1
