from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from joint_geometry import (
    JointCameraEstimate,
    MetricPoseConstraint,
    calibrate_joint_cameras,
)


def _camera_to_world(yaw_degrees: float, translation: list[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler(
        "y", yaw_degrees, degrees=True
    ).as_matrix()
    transform[:3, 3] = translation
    return transform


def test_joint_camera_solution_is_metric_calibrated_by_verified_pair() -> None:
    raw = {
        0: _camera_to_world(0.0, [0.0, 0.0, 0.0]),
        1: _camera_to_world(42.0, [0.10, 0.0, -0.08]),
        2: _camera_to_world(88.0, [0.20, 0.0, -0.14]),
        3: _camera_to_world(-31.0, [0.0, -0.01, -0.04]),
    }
    expected_metric_scale = 3.25
    measured = np.linalg.inv(raw[0]) @ raw[3]
    measured[:3, 3] *= expected_metric_scale
    estimate = JointCameraEstimate(
        frame_indices=[0, 1, 2, 3],
        camera_to_world=raw,
        intrinsics={index: np.eye(3) for index in raw},
        depth_confidence={0: 10.0, 1: 9.0, 2: 8.0, 3: 9.5},
        report={"available": True, "accepted": False},
    )

    calibrated = calibrate_joint_cameras(
        estimate,
        [MetricPoseConstraint(0, 3, measured, 0.95)],
        0,
    )

    assert calibrated is not None
    assert calibrated.report["accepted"] is True
    assert math.isclose(
        calibrated.report["metricScale"],
        expected_metric_scale,
        rel_tol=1e-6,
    )
    assert sorted(calibrated.transforms_to_anchor) == [0, 1, 2, 3]
    np.testing.assert_allclose(
        calibrated.transforms_to_anchor[2][:3, 3],
        raw[2][:3, 3] * expected_metric_scale,
        atol=1e-7,
    )


def test_joint_camera_solution_rejects_rotation_that_conflicts_with_measurement() -> None:
    raw = {
        0: _camera_to_world(0.0, [0.0, 0.0, 0.0]),
        1: _camera_to_world(75.0, [0.1, 0.0, 0.0]),
    }
    measured = _camera_to_world(5.0, [0.3, 0.0, 0.0])
    estimate = JointCameraEstimate(
        frame_indices=[0, 1],
        camera_to_world=raw,
        intrinsics={0: np.eye(3), 1: np.eye(3)},
        depth_confidence={0: 10.0, 1: 10.0},
        report={"available": True, "accepted": False},
    )

    calibrated = calibrate_joint_cameras(
        estimate,
        [MetricPoseConstraint(0, 1, measured, 0.9)],
        0,
    )

    assert calibrated is None
    assert estimate.report["accepted"] is False
    assert estimate.report["validatedPairs"] == []
