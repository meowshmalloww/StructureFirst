from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

SERVICE_ROOT = Path(__file__).resolve().parent
LUCIDFRAME_BACKEND = Path(
    os.getenv("LUCIDFRAME_ROOT", str(SERVICE_ROOT.parent.parent.parent / "LucidFrame"))
) / "backend"
sys.path.insert(0, str(LUCIDFRAME_BACKEND))
sys.path.insert(0, str(SERVICE_ROOT))

from reconstruction_stage import GaussianData  # noqa: E402
from smart_connect import (  # noqa: E402
    Alignment,
    FrameData,
    PreflightAlignment,
    _apply_points,
    _connection_tree,
    _cross_view_cleanup,
    _refine_pose_graph,
    _strongest_preflight_component,
    _transform_cloud,
    _umeyama,
)


def test_umeyama_recovers_metric_similarity() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(80, 3))
    angle = math.radians(23)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    expected_scale = 1.13
    translation = np.array([0.4, -0.2, 1.7])
    target = source @ (expected_scale * rotation).T + translation

    transform, scale = _umeyama(source, target)

    assert abs(scale - expected_scale) < 1e-7
    np.testing.assert_allclose(_apply_points(source, transform), target, atol=1e-7)


def test_connection_tree_keeps_unconnected_frames_out() -> None:
    transform = np.eye(4)
    edge = Alignment(0, 1, transform, 100, 90, 80, 0.88, 0.02, 1.0, 0.91)
    transforms, tree, anchor = _connection_tree(3, [edge])
    assert sorted(transforms) == [0, 1]
    assert tree == [edge]
    assert anchor == 0


def test_connection_tree_uses_valid_group_when_first_frame_is_disconnected() -> None:
    transform = np.eye(4)
    weak_pair = Alignment(1, 2, transform, 90, 75, 50, 0.67, 0.04, 1.0, 0.72)
    strong_pair = Alignment(2, 3, transform, 120, 100, 86, 0.86, 0.02, 1.0, 0.94)

    transforms, tree, anchor = _connection_tree(4, [weak_pair, strong_pair])

    assert sorted(transforms) == [1, 2, 3]
    assert tree == [strong_pair, weak_pair]
    assert anchor == 2


def test_preflight_selects_largest_verified_overlap_group() -> None:
    road_pair = PreflightAlignment(0, 1, 90, 62, 0.69, 0.82)
    facade_pair_a = PreflightAlignment(2, 3, 110, 78, 0.71, 0.9)
    facade_pair_b = PreflightAlignment(3, 4, 105, 74, 0.7, 0.88)

    selected = _strongest_preflight_component(
        6,
        [road_pair, facade_pair_a, facade_pair_b],
    )

    assert selected == [2, 3, 4]


def test_cloud_transform_updates_position_scale_and_rotation() -> None:
    cloud = GaussianData(
        positions=np.array([[1.0, 0.0, 2.0]], dtype=np.float32),
        scales=np.zeros((1, 3), dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        colors=np.ones((1, 3), dtype=np.float32),
        opacities=np.ones((1, 1), dtype=np.float32),
    )
    transform = np.eye(4)
    transform[:3, :3] *= 2.0
    transform[:3, 3] = [1.0, 2.0, 3.0]

    transformed = _transform_cloud(cloud, transform)

    np.testing.assert_allclose(transformed.positions, [[3.0, 2.0, 7.0]])
    np.testing.assert_allclose(transformed.scales, math.log(2.0))


def test_pose_graph_uses_loop_edge_to_reduce_joint_residual() -> None:
    identity = np.eye(4)
    one = np.eye(4)
    one[0, 3] = 1.0
    two = np.eye(4)
    two[0, 3] = 2.2
    alignments = [
        Alignment(0, 1, one, 100, 90, 80, 0.8, 0.02, 1.0, 0.95),
        Alignment(1, 2, one, 100, 90, 80, 0.8, 0.02, 1.0, 0.92),
        Alignment(0, 2, two, 100, 90, 80, 0.8, 0.02, 1.0, 0.75),
    ]
    initial, _, anchor = _connection_tree(3, alignments)

    refined, report = _refine_pose_graph(initial, alignments, anchor)

    assert report["optimized"] is True
    assert report["loopEdgeCount"] == 1
    assert report["afterRmse"] < report["beforeRmse"]
    assert refined[2][0, 3] > initial[2][0, 3]
    np.testing.assert_allclose(refined[anchor], identity)


def test_cross_view_cleanup_only_removes_repeated_front_conflict() -> None:
    cloud = GaussianData(
        positions=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        scales=np.zeros((1, 3), dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        colors=np.ones((1, 3), dtype=np.float32),
        opacities=np.ones((1, 1), dtype=np.float32),
    )

    def frame(index: int, depth: float) -> FrameData:
        return FrameData(
            source_index=index,
            path=Path(f"frame-{index}.jpg"),
            cloud=cloud,
            image=np.zeros((1, 1, 3), dtype=np.uint8),
            focal_px=1.0,
            keypoints=[],
            descriptors=None,
            xyz_map=np.array([[[0.0, 0.0, depth]]], dtype=np.float32),
        )

    cleaned, report = _cross_view_cleanup(
        {0: cloud},
        {0: frame(0, 1.0), 1: frame(1, 2.0), 2: frame(2, 2.0)},
        {0: np.eye(4), 1: np.eye(4), 2: np.eye(4)},
    )

    assert cleaned[0].count == 0
    assert report["removed"] == 1
    assert report["requiredObservations"] == 2


def test_cross_view_cleanup_uses_the_other_camera_for_a_two_view_scene() -> None:
    cloud = GaussianData(
        positions=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        scales=np.zeros((1, 3), dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        colors=np.ones((1, 3), dtype=np.float32),
        opacities=np.ones((1, 1), dtype=np.float32),
    )

    def frame(index: int, depth: float) -> FrameData:
        return FrameData(
            source_index=index,
            path=Path(f"frame-{index}.jpg"),
            cloud=cloud,
            image=np.zeros((1, 1, 3), dtype=np.uint8),
            focal_px=1.0,
            keypoints=[],
            descriptors=None,
            xyz_map=np.array([[[0.0, 0.0, depth]]], dtype=np.float32),
        )

    cleaned, report = _cross_view_cleanup(
        {0: cloud},
        {0: frame(0, 1.0), 1: frame(1, 2.0)},
        {0: np.eye(4), 1: np.eye(4)},
    )

    assert cleaned[0].count == 0
    assert report["removed"] == 1
    assert report["requiredObservations"] == 1
