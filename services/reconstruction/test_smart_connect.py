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
    _dense_surface_alignments,
    _joint_color_balance,
    _joint_scene_quality_gate,
    _refine_pose_graph,
    _refine_with_dense_surface_constraints,
    _regularize_extreme_scales,
    _regularize_source_footprints,
    _strongest_preflight_component,
    _transform_cloud,
    _umeyama,
)


def test_joint_scene_quality_gate_rejects_a_weak_overlay() -> None:
    result = _joint_scene_quality_gate(
        {"crossViewSupportedFraction": 0.26070772},
        {"denseConstraintCount": 2},
    )

    assert result["accepted"] is False


def test_joint_scene_quality_gate_accepts_either_measured_signal() -> None:
    supported = _joint_scene_quality_gate(
        {"crossViewSupportedFraction": 0.31},
        {"denseConstraintCount": 0},
    )
    dense = _joint_scene_quality_gate(
        {"crossViewSupportedFraction": 0.1},
        {"denseConstraintCount": 24},
    )

    assert supported["accepted"] is True
    assert dense["accepted"] is True


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


def test_pose_graph_uses_retained_metric_points_to_reduce_bending() -> None:
    world = np.column_stack(
        [
            np.linspace(-0.6, 0.7, 24),
            np.linspace(-0.2, 0.4, 24),
            np.linspace(1.2, 2.4, 24),
        ]
    )

    def translation(x: float) -> np.ndarray:
        transform = np.eye(4)
        transform[0, 3] = x
        return transform

    frame_points = {
        0: world,
        1: world - np.array([1.0, 0.0, 0.0]),
        2: world - np.array([2.0, 0.0, 0.0]),
    }

    def edge(a: int, b: int, measured: float, confidence: float) -> Alignment:
        return Alignment(
            a,
            b,
            translation(measured),
            120,
            100,
            90,
            0.9,
            0.02,
            1.0,
            confidence,
            points_a=frame_points[a],
            points_b=frame_points[b],
        )

    alignments = [
        edge(0, 1, 1.1, 0.96),
        edge(1, 2, 1.1, 0.94),
        edge(0, 2, 2.0, 0.80),
    ]
    initial, _, anchor = _connection_tree(3, alignments)

    refined, report = _refine_pose_graph(initial, alignments, anchor)

    assert report["optimized"] is True
    assert report["pointConstraintCount"] == 72
    assert (
        report["pointAfterRmseMeters"]
        < report["pointBeforeRmseMeters"]
    )
    expected_relative_x = 2.0 - float(anchor)
    assert abs(refined[2][0, 3] - expected_relative_x) < abs(
        initial[2][0, 3] - expected_relative_x
    )


def test_dense_surface_refinement_tightens_a_verified_pair() -> None:
    height = 14
    width = 14
    pixel_x, pixel_y = np.meshgrid(np.arange(width), np.arange(height))
    world = np.stack(
        [
            (pixel_x - width / 2) * 0.03,
            (pixel_y - height / 2) * 0.03,
            1.5
            + 0.015 * np.sin(pixel_x * 0.7)
            + 0.01 * np.cos(pixel_y * 0.4),
        ],
        axis=-1,
    ).astype(np.float32)
    translation_true = np.array([0.1, 0.0, 0.0], dtype=np.float32)
    image = np.stack(
        [
            40 + pixel_x * 8,
            50 + pixel_y * 7,
            80 + (pixel_x + pixel_y) * 4,
        ],
        axis=-1,
    ).clip(0, 255).astype(np.uint8)

    def make_frame(index: int, xyz_map: np.ndarray) -> FrameData:
        positions = xyz_map.reshape(-1, 3).copy()
        cloud = GaussianData(
            positions=positions,
            scales=np.full((len(positions), 3), math.log(0.01), np.float32),
            rotations=np.tile(
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                (len(positions), 1),
            ),
            colors=(image.reshape(-1, 3) / 255.0).astype(np.float32),
            opacities=np.ones((len(positions), 1), dtype=np.float32),
        )
        return FrameData(
            source_index=index,
            path=Path(f"frame-{index}.jpg"),
            cloud=cloud,
            image=image,
            focal_px=100.0,
            keypoints=[],
            descriptors=None,
            xyz_map=xyz_map,
        )

    frames = {
        0: make_frame(0, world),
        1: make_frame(1, world - translation_true),
    }
    measured = np.eye(4)
    measured[0, 3] = 0.108
    sparse_rows = np.array([20, 55, 100, 150])
    alignment = Alignment(
        0,
        1,
        measured,
        120,
        100,
        90,
        0.9,
        0.01,
        1.0,
        0.95,
        points_a=world.reshape(-1, 3)[sparse_rows],
        points_b=(world - translation_true).reshape(-1, 3)[sparse_rows],
    )
    initial = {0: np.eye(4), 1: measured.copy()}

    augmented, dense, reports = _dense_surface_alignments(
        frames,
        initial,
        [alignment],
        maximum_samples_per_frame=500,
    )
    refined, report = _refine_with_dense_surface_constraints(
        frames,
        initial,
        [alignment],
        [],
        0,
    )

    assert len(augmented[0].points_a) > len(alignment.points_a)
    assert len(dense[0].points_a) >= 24
    assert reports[0]["accepted"] >= 24
    assert report["optimized"] is True
    assert report["denseAfterRmseMeters"] < report["denseBeforeRmseMeters"]
    assert abs(refined[1][0, 3] - 0.1) < abs(initial[1][0, 3] - 0.1)


def test_source_footprint_regularization_preserves_thin_axis_and_count() -> None:
    positions = np.array(
        [[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]],
        dtype=np.float32,
    )
    linear_scales = np.array(
        [[0.002, 0.004, 0.02], [0.012, 0.015, 0.02]],
        dtype=np.float32,
    )
    cloud = GaussianData(
        positions=positions,
        scales=np.log(linear_scales),
        rotations=np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (2, 1),
        ),
        colors=np.ones((2, 3), dtype=np.float32),
        opacities=np.ones((2, 1), dtype=np.float32),
    )
    frame = FrameData(
        source_index=0,
        path=Path("frame-0.jpg"),
        cloud=cloud,
        image=np.zeros((1, 2, 3), dtype=np.uint8),
        focal_px=100.0,
        keypoints=[],
        descriptors=None,
        xyz_map=positions.reshape(1, 2, 3),
    )

    regularized, report = _regularize_source_footprints(
        {0: cloud},
        {0: frame},
        {0: np.eye(4)},
    )
    result = np.exp(regularized[0].scales)

    assert regularized[0].count == cloud.count
    assert report["gaussianCountBefore"] == report["gaussianCountAfter"]
    assert report["affected"] == 1
    np.testing.assert_allclose(result[0, 0], linear_scales[0, 0], rtol=1e-6)
    np.testing.assert_allclose(result[0, 1], 0.0046, rtol=1e-6)
    np.testing.assert_allclose(result[0, 2], linear_scales[0, 2], rtol=1e-6)
    np.testing.assert_allclose(result[1], linear_scales[1], rtol=1e-6)


def test_scale_regularizer_removes_only_rare_single_axis_needles() -> None:
    linear_scales = np.array(
        [
            [1.0, 0.9, 0.001],
            [1.0, 0.01, 0.009],
            [0.1, 0.09, 0.08],
        ],
        dtype=np.float32,
    )
    cloud = GaussianData(
        positions=np.zeros((3, 3), dtype=np.float32),
        scales=np.log(linear_scales),
        rotations=np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (3, 1),
        ),
        colors=np.ones((3, 3), dtype=np.float32),
        opacities=np.ones((3, 1), dtype=np.float32),
    )

    regularized, report = _regularize_extreme_scales({0: cloud})
    result = np.exp(regularized[0].scales)
    sorted_result = np.sort(result, axis=1)

    assert regularized[0].count == cloud.count
    assert report["gaussianCountBefore"] == report["gaussianCountAfter"]
    assert report["affected"] == 1
    np.testing.assert_allclose(result[0], linear_scales[0], rtol=1e-6)
    assert sorted_result[1, 2] / sorted_result[1, 1] <= 12.0001


def test_color_balance_uses_verified_overlap_pixels() -> None:
    pixels = np.asarray(
        [[x, y] for y in range(4) for x in range(5)],
        dtype=np.float32,
    )

    def cloud(color: np.ndarray) -> GaussianData:
        return GaussianData(
            positions=np.zeros((2, 3), dtype=np.float32),
            scales=np.zeros((2, 3), dtype=np.float32),
            rotations=np.tile(
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                (2, 1),
            ),
            colors=np.tile(color[None] / 255.0, (2, 1)).astype(np.float32),
            opacities=np.ones((2, 1), dtype=np.float32),
        )

    def frame(index: int, color: np.ndarray) -> FrameData:
        return FrameData(
            source_index=index,
            path=Path(f"frame-{index}.jpg"),
            cloud=cloud(color),
            image=np.tile(color[None, None], (8, 8, 1)).astype(np.uint8),
            focal_px=8.0,
            keypoints=[],
            descriptors=None,
            xyz_map=np.zeros((8, 8, 3), dtype=np.float32),
        )

    bright = np.array([120, 140, 160], dtype=np.float32)
    dark = np.array([96, 112, 128], dtype=np.float32)
    frames = {0: frame(0, bright), 1: frame(1, dark)}
    alignment = Alignment(
        0,
        1,
        np.eye(4),
        30,
        20,
        20,
        1.0,
        0.01,
        1.0,
        0.95,
        pixels_a=pixels,
        pixels_b=pixels,
    )

    balanced, report = _joint_color_balance(
        {0: frames[0].cloud, 1: frames[1].cloud},
        frames,
        [alignment],
    )

    assert report["applied"] is True
    assert report["pairCount"] == 1
    assert report["overlapSampleCount"] == 20
    assert report["overlapLogRmseAfter"] < report["overlapLogRmseBefore"]
    assert balanced[1].colors.mean() > frames[1].cloud.colors.mean()


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
    assert report["crossViewSupportedFraction"] == 0.0


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
