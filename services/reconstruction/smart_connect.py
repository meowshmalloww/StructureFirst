"""Confidence-gated registration for unordered indoor photographs.

LucidFrame's SHARP output is metric and camera-centred. A learned indoor
matcher and SIFT verify overlapping views first. Only frames with verified
cross-image correspondences and a stable metric transform join the merged
Gaussian scene. Scene recognition can nominate another room view, but it can
never place that view by appearance or point-cloud shape alone.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from joint_geometry import (
    MetricPoseConstraint,
    calibrate_joint_cameras,
    estimate_joint_cameras,
)
from reconstruction_stage import GaussianData
from scene_understanding import (
    CaptureMetadata,
    capture_metadata,
    load_indoor_matcher,
    match_indoor_pair,
    recognize_scene_frames,
    release_matcher,
    scene_embeddings,
)
from sharp_wrapper import (
    _matrices_to_wxyz,
    _wxyz_to_matrices,
    reconstruct_sharp,
    unload as unload_sharp,
)

LOGGER = logging.getLogger("structurefirst.smart_connect")

# Half-set coverage is the threshold for describing a result as representative
# of the submitted capture set. It is not a reason to discard a smaller,
# geometrically verified room component. Partial components are exported as
# separate scenes and must remain visibly labelled partial in the product.
REPRESENTATIVE_CAPTURE_SET_COVERAGE = 0.5
MIN_CROSS_VIEW_SUPPORTED_FRACTION = 0.30
MIN_DENSE_SURFACE_CONSTRAINTS = 24


def minimum_connected_frames(frame_count: int) -> int:
    """Minimum geometric-core size required for one honest partial scene."""
    del frame_count
    return 2


def representative_connected_frames(frame_count: int) -> int:
    """Frames required before a component represents the submitted set."""
    return max(2, math.ceil(frame_count * REPRESENTATIVE_CAPTURE_SET_COVERAGE))


@dataclass
class FrameData:
    source_index: int
    path: Path
    cloud: GaussianData
    image: np.ndarray
    focal_px: float
    keypoints: list[Any]
    descriptors: np.ndarray | None
    xyz_map: np.ndarray


@dataclass
class Alignment:
    frame_a: int
    frame_b: int
    transform_b_to_a: np.ndarray
    feature_matches: int
    metric_matches: int
    inliers: int
    inlier_ratio: float
    rmse_m: float
    scale: float
    confidence: float
    method: str = "sift_metric_similarity"
    points_a: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64), repr=False
    )
    points_b: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64), repr=False
    )
    pixels_a: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32), repr=False
    )
    pixels_b: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32), repr=False
    )


@dataclass
class PreflightFrame:
    index: int
    path: Path
    keypoints: list[Any]
    descriptors: np.ndarray | None
    image: np.ndarray | None = field(default=None, repr=False)
    source_size: tuple[int, int] = (0, 0)
    capture: CaptureMetadata = field(
        default_factory=lambda: CaptureMetadata(None, None)
    )


@dataclass
class PreflightAlignment:
    frame_a: int
    frame_b: int
    feature_matches: int
    inliers: int
    inlier_ratio: float
    confidence: float
    method: str = "sift_ransac"
    points_a: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32), repr=False
    )
    points_b: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32), repr=False
    )


class RegistrationError(RuntimeError):
    def __init__(
        self,
        message: str,
        report: dict[str, Any],
        fallback_cloud: GaussianData | None = None,
    ):
        super().__init__(message)
        self.report = report
        self.fallback_cloud = fallback_cloud


def reconstruct_connected(
    image_paths: list[Path],
    output_dir: Path,
) -> tuple[GaussianData, dict[str, Any]]:
    if len(image_paths) < 2:
        raise ValueError("Smart connect requires at least two photographs")

    (
        preflight_frames,
        preflight_edges,
        preflight_rejected,
        selected_indices,
        core_indices,
        scene_report,
    ) = _preflight_selection(image_paths)
    recognized_set = set(selected_indices)
    core_set = set(core_indices)
    frame_reports: list[dict[str, Any]] = [
        {
            "index": frame.index,
            "file": frame.path.name,
            "featureCount": len(frame.keypoints),
            "selectedForSharp": frame.index in core_set,
            "classification": (
                "geometric_core"
                if frame.index in core_set
                else "same_scene_candidate"
                if frame.index in recognized_set
                else "outlier"
            ),
        }
        for frame in preflight_frames
    ]
    preflight_report = {
        "method": "SIFT + indoor LoFTR + robust two-view geometry",
        "acceptedPairs": [_preflight_report(item) for item in preflight_edges],
        "rejectedPairs": preflight_rejected,
        "geometricCoreFrames": core_indices,
        "recognizedSceneFrames": selected_indices,
        "outlierFrames": [
            index for index in range(len(image_paths)) if index not in recognized_set
        ],
        "selectedFrames": core_indices,
        "sceneRecognition": scene_report,
    }
    required_connected_frames = minimum_connected_frames(len(image_paths))
    representative_frames = representative_connected_frames(len(image_paths))
    if len(core_indices) < required_connected_frames:
        report = {
            "schemaVersion": 1,
            "method": "SIFT and LoFTR preflight before LucidFrame SHARP metric registration",
            "status": "failed",
            "frameCount": len(image_paths),
            "connectedFrameCount": len(core_indices),
            "anchorFrame": core_indices[0] if core_indices else 0,
            "frames": frame_reports,
            "preflight": preflight_report,
            "acceptedPairs": [],
            "rejectedPairs": [],
            "treePairs": [],
            "connectedFrames": core_indices,
            "disconnectedFrames": [
                index
                for index in range(len(image_paths))
                if index not in core_set
            ],
            "minimumConnectedFrames": required_connected_frames,
            "representativeConnectedFrames": representative_frames,
            "representativeCaptureSetCoverage": REPRESENTATIVE_CAPTURE_SET_COVERAGE,
            "representativeCaptureSetCoverageMet": False,
            "confidenceScore": 0.0,
        }
        raise RegistrationError(
            (
                f"Only {len(core_indices)}/{len(image_paths)} photographs formed "
                f"one geometric capture path; at least {required_connected_frames} "
                "are required. Capture adjacent views with 60-80% overlap and "
                "bridge every doorway or stair."
            ),
            report,
        )

    joint_estimate = None
    joint_error: str | None = None
    try:
        # A previous serialized job may still have SHARP cached. VGGT needs
        # most of the 12 GB RTX budget, so the two predictors run sequentially.
        unload_sharp()
        joint_estimate = estimate_joint_cameras(
            [image_paths[index] for index in core_indices],
            core_indices,
            output_dir,
        )
    except Exception as exc:
        joint_error = str(exc)
        LOGGER.warning("Joint camera inference unavailable: %s", exc)

    frames: list[FrameData] = []
    for index in core_indices:
        path = image_paths[index]
        frame_dir = output_dir / "frames" / f"{index:02d}"
        cloud = reconstruct_sharp(path, frame_dir)
        image, focal_px = _sharp_source(path, frame_dir)
        sift = cv2.SIFT_create(nfeatures=7000, contrastThreshold=0.025)
        keypoints, descriptors = sift.detectAndCompute(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None
        )
        xyz_map = _project_cloud(cloud, image.shape[1], image.shape[0], focal_px)
        frames.append(
            FrameData(
                source_index=index,
                path=path,
                cloud=cloud,
                image=image,
                focal_px=focal_px,
                keypoints=keypoints,
                descriptors=descriptors,
                xyz_map=xyz_map,
            )
        )
        frame_reports[index]["gaussianCount"] = cloud.count
        frame_reports[index]["metricFeatureCount"] = len(keypoints)

    alignments: list[Alignment] = []
    rejected: list[dict[str, Any]] = []
    preflight_by_pair = {
        (edge.frame_a, edge.frame_b): edge for edge in preflight_edges
    }
    for frame_a in range(len(frames)):
        for frame_b in range(frame_a + 1, len(frames)):
            source_a = frames[frame_a].source_index
            source_b = frames[frame_b].source_index
            alignment, reason = _align_pair(
                frames[frame_a],
                frames[frame_b],
                source_a,
                source_b,
                preflight_by_pair.get((source_a, source_b)),
            )
            if alignment is not None:
                alignments.append(alignment)
            else:
                rejected.append(
                    {
                        "frameA": source_a,
                        "frameB": source_b,
                        "reason": reason,
                    }
                )

    metric_transforms, metric_tree, anchor_frame = _connection_tree(
        len(image_paths), alignments
    )
    calibrated_joint = calibrate_joint_cameras(
        joint_estimate,
        [
            MetricPoseConstraint(
                frame_a=item.frame_a,
                frame_b=item.frame_b,
                transform_b_to_a=item.transform_b_to_a,
                confidence=item.confidence,
            )
            for item in alignments
        ],
        anchor_frame,
    )
    joint_alignments: list[Alignment] = []
    if calibrated_joint is not None:
        for index, transform in calibrated_joint.transforms_to_anchor.items():
            # Joint cameras refine a transform only after that frame already
            # belongs to the SHARP metric component. A single agreeing pair
            # must never pull an unverified semantic candidate into the room.
            if index == anchor_frame or index not in metric_transforms:
                continue
            joint_alignments.append(
                Alignment(
                    frame_a=anchor_frame,
                    frame_b=index,
                    transform_b_to_a=transform,
                    feature_matches=0,
                    metric_matches=0,
                    inliers=0,
                    inlier_ratio=0.0,
                    rmse_m=0.0,
                    scale=1.0,
                    confidence=calibrated_joint.confidence_by_frame[index],
                    method="vggt_joint_camera_metric_calibrated",
                )
            )
    pose_edges = [*alignments, *joint_alignments]
    transforms, tree, anchor_frame = _connection_tree(
        len(image_paths), pose_edges
    )
    transforms, pose_optimization = _refine_pose_graph(
        transforms,
        pose_edges,
        anchor_frame,
    )
    (
        transforms,
        dense_surface_refinement,
    ) = _refine_with_dense_surface_constraints(
        frames_by_index={frame.source_index: frame for frame in frames},
        initial=transforms,
        metric_alignments=alignments,
        auxiliary_alignments=joint_alignments,
        anchor=anchor_frame,
    )
    connected_indices = sorted(transforms)
    for frame in frame_reports:
        index = int(frame["index"])
        if index in transforms:
            frame["classification"] = "registered"
        elif index in recognized_set:
            frame["classification"] = "same_scene_unregistered"
            frame["registrationReason"] = (
                "Same-room evidence was found, but no safe cross-view transform was verified"
            )
    frames_by_index = {frame.source_index: frame for frame in frames}
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "method": (
            "VGGT shared cameras + SIFT/LoFTR correspondences + "
            "SHARP metric similarity + robust global pose graph"
        ),
        "frameCount": len(image_paths),
        "connectedFrameCount": len(connected_indices),
        "anchorFrame": anchor_frame,
        "frames": frame_reports,
        "preflight": preflight_report,
        "acceptedPairs": [_alignment_report(item) for item in alignments],
        "jointCameraEdges": [
            _alignment_report(item) for item in joint_alignments
        ],
        "rejectedPairs": rejected,
        "treePairs": [_alignment_report(item) for item in tree],
        "poseOptimization": pose_optimization,
        "denseSurfaceRefinement": dense_surface_refinement,
        "jointGeometry": (
            calibrated_joint.report
            if calibrated_joint is not None
            else joint_estimate.report
            if joint_estimate is not None
            else {
                "available": False,
                "accepted": False,
                "error": joint_error or "Joint camera inference was unavailable.",
            }
        ),
        "metricCoreFrames": sorted(metric_transforms),
        "metricTreePairs": [
            _alignment_report(item) for item in metric_tree
        ],
        "connectedFrames": connected_indices,
        "disconnectedFrames": [
            index for index in range(len(image_paths)) if index not in transforms
        ],
        "minimumConnectedFrames": required_connected_frames,
        "representativeConnectedFrames": representative_frames,
        "representativeCaptureSetCoverage": REPRESENTATIVE_CAPTURE_SET_COVERAGE,
        "representativeCaptureSetCoverageMet": (
            len(connected_indices) >= representative_frames
        ),
    }
    if len(connected_indices) < 2:
        report["status"] = "failed"
        report["confidenceScore"] = 0.0
        raise RegistrationError(
            "No two photographs had enough verified metric overlap. Capture adjacent views with 60–80% overlap.",
            report,
            frames[0].cloud if frames[0].source_index == 0 else None,
        )

    transformed_clouds: dict[int, GaussianData] = {}
    for index in connected_indices:
        transformed_clouds[index] = _transform_cloud(
            frames_by_index[index].cloud,
            transforms[index],
        )

    transformed_clouds, footprint_report = _regularize_source_footprints(
        transformed_clouds,
        frames_by_index,
        transforms,
    )
    transformed_clouds, cleanup_report = _cross_view_cleanup(
        transformed_clouds,
        frames_by_index,
        transforms,
    )
    merge_validation = _joint_scene_quality_gate(
        cleanup_report,
        dense_surface_refinement,
    )
    if not merge_validation["accepted"]:
        report.update(
            {
                "status": "failed",
                "confidenceScore": 0.0,
                "gaussianCount": 0,
                "sourceCoverageRegularization": footprint_report,
                "artifactCleanup": cleanup_report,
                "mergeValidation": merge_validation,
            }
        )
        raise RegistrationError(
            (
                "The registered cameras did not produce one reliable shared "
                "surface. Their exact SHARP views are kept as separate scenes "
                "instead of being overlaid into a broken Gaussian room."
            ),
            report,
        )
    transformed_clouds, scale_report = _regularize_extreme_scales(
        transformed_clouds
    )
    transformed_clouds, color_report = _joint_color_balance(
        transformed_clouds,
        frames_by_index,
        alignments,
    )
    merged: GaussianData | None = None
    for index in connected_indices:
        transformed = transformed_clouds[index]
        merged = transformed if merged is None else _concatenate(merged, transformed)

    confidence = float(np.mean([item.confidence for item in tree])) if tree else 0.0
    report["status"] = (
        "connected" if len(connected_indices) == len(image_paths) else "partial"
    )
    report["confidenceScore"] = round(confidence, 4)
    report["gaussianCount"] = merged.count if merged else 0
    report["sourceCoverageRegularization"] = footprint_report
    report["artifactCleanup"] = cleanup_report
    report["scaleRegularization"] = scale_report
    report["jointColorBalance"] = color_report
    report["mergeValidation"] = merge_validation
    report["cameraPoses"] = [
        _camera_pose_report(index, transforms[index], index in metric_transforms)
        for index in connected_indices
    ]
    return merged or frames_by_index[connected_indices[0]].cloud, report


def _joint_scene_quality_gate(
    cleanup_report: dict[str, Any],
    dense_surface_refinement: dict[str, Any],
) -> dict[str, Any]:
    """Require measured shared geometry before concatenating SHARP scenes."""
    supported_fraction = float(
        cleanup_report.get("crossViewSupportedFraction", 0.0)
    )
    dense_constraints = int(
        dense_surface_refinement.get("denseConstraintCount", 0)
    )
    accepted = bool(
        supported_fraction >= MIN_CROSS_VIEW_SUPPORTED_FRACTION
        or dense_constraints >= MIN_DENSE_SURFACE_CONSTRAINTS
    )
    return {
        "accepted": accepted,
        "crossViewSupportedFraction": round(supported_fraction, 8),
        "minimumCrossViewSupportedFraction": (
            MIN_CROSS_VIEW_SUPPORTED_FRACTION
        ),
        "denseSurfaceConstraintCount": dense_constraints,
        "minimumDenseSurfaceConstraints": MIN_DENSE_SURFACE_CONSTRAINTS,
        "rule": "cross-view depth support OR reciprocal dense surface constraints",
    }


def _preflight_selection(
    image_paths: list[Path],
) -> tuple[
    list[PreflightFrame],
    list[PreflightAlignment],
    list[dict[str, Any]],
    list[int],
    list[int],
    dict[str, Any],
]:
    """Find the strongest visual overlap group before spending GPU time."""
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.025)
    frames: list[PreflightFrame] = []
    for index, path in enumerate(image_paths):
        with Image.open(path) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
            source_size = source.size
            longest = max(source.size)
            if longest > 1400:
                ratio = 1400 / longest
                source = source.resize(
                    (
                        max(1, round(source.width * ratio)),
                        max(1, round(source.height * ratio)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            image = np.asarray(source)
        keypoints, descriptors = sift.detectAndCompute(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None
        )
        frames.append(
            PreflightFrame(
                index=index,
                path=path,
                keypoints=keypoints,
                descriptors=descriptors,
                image=image,
                source_size=source_size,
                capture=capture_metadata(path),
            )
        )

    accepted: list[PreflightAlignment] = []
    rejection_reasons: dict[tuple[int, int], str] = {}
    for frame_a in range(len(frames)):
        for frame_b in range(frame_a + 1, len(frames)):
            alignment, reason = _preflight_pair(frames[frame_a], frames[frame_b])
            if alignment is not None:
                accepted.append(alignment)
            else:
                rejection_reasons[(frame_a, frame_b)] = reason

    embeddings: np.ndarray | None = None
    embedding_error: str | None = None
    try:
        embeddings = scene_embeddings(
            [frame.image for frame in frames if frame.image is not None]
        )
    except Exception as exc:  # The geometric fallback remains available offline.
        embedding_error = str(exc)
        LOGGER.warning("DINOv2 scene descriptors unavailable: %s", exc)

    learned_error: str | None = None
    learned_checkpoint: str | None = None
    matcher: Any | None = None
    try:
        matcher, device, learned_checkpoint = load_indoor_matcher()
        already_accepted = {
            (edge.frame_a, edge.frame_b) for edge in accepted
        }
        for frame_a in range(len(frames)):
            for frame_b in range(frame_a + 1, len(frames)):
                if (frame_a, frame_b) in already_accepted:
                    continue
                image_a = frames[frame_a].image
                image_b = frames[frame_b].image
                if image_a is None or image_b is None:
                    continue
                learned = match_indoor_pair(
                    matcher,
                    device,
                    image_a,
                    image_b,
                    frames[frame_a].source_size,
                    frames[frame_b].source_size,
                )
                homography_dominance = _homography_dominance(
                    learned.points_a,
                    learned.points_b,
                    threshold_px=6.0,
                )
                spread = _match_spread(
                    learned.points_a,
                    learned.points_b,
                    frames[frame_a].source_size,
                    frames[frame_b].source_size,
                )
                distributed_room_evidence = (
                    learned.inlier_ratio >= 0.68
                    # LoFTR is slightly direction-sensitive. Seven occupied
                    # 4x4 cells plus broad hull coverage remains room-scale,
                    # while the repeated bathroom-floor false match occupies
                    # only six cells and also fails the ratio/hull gates.
                    and spread["minimumGridCells"] >= 7
                    and spread["minimumHullCoverage"] >= 0.20
                )
                if (
                    learned.raw_matches >= 20
                    and learned.inliers >= 18
                    and learned.inlier_ratio >= 0.45
                    and (
                        homography_dominance < 0.58
                        or distributed_room_evidence
                    )
                ):
                    confidence = float(
                        np.clip(
                            0.55 * min(1.0, learned.inliers / 80.0)
                            + 0.35 * min(1.0, learned.inlier_ratio / 0.8)
                            + 0.10 * min(1.0, learned.mean_confidence / 0.5),
                            0.0,
                            1.0,
                        )
                    )
                    accepted.append(
                        PreflightAlignment(
                            frame_a=frame_a,
                            frame_b=frame_b,
                            feature_matches=learned.raw_matches,
                            inliers=learned.inliers,
                            inlier_ratio=learned.inlier_ratio,
                            confidence=confidence,
                            method="loftr_indoor_fundamental_magsac",
                            points_a=learned.points_a,
                            points_b=learned.points_b,
                        )
                    )
                    rejection_reasons.pop((frame_a, frame_b), None)
                else:
                    original = rejection_reasons.get((frame_a, frame_b), "")
                    learned_reason = (
                        f"indoor matcher kept {learned.inliers}/"
                        f"{learned.raw_matches} geometric inliers; "
                        f"planar/repetitive dominance={homography_dominance:.3f}; "
                        f"minimum grid cells={spread['minimumGridCells']}; "
                        f"minimum hull coverage={spread['minimumHullCoverage']:.3f}"
                    )
                    rejection_reasons[(frame_a, frame_b)] = "; ".join(
                        part for part in (original, learned_reason) if part
                    )
    except Exception as exc:
        learned_error = str(exc)
        LOGGER.warning("Indoor learned matching unavailable: %s", exc)
    finally:
        if matcher is not None:
            release_matcher(matcher)

    core = _strongest_preflight_component(len(frames), accepted)
    if accepted:
        recognized, candidate_reports, affinity_threshold = recognize_scene_frames(
            core,
            embeddings,
            [frame.capture for frame in frames],
        )
    else:
        # Semantics must never bootstrap a scene without a verified pair.
        recognized = core
        candidate_reports = []
        affinity_threshold = None
    rejected = [
        {"frameA": pair[0], "frameB": pair[1], "reason": reason}
        for pair, reason in sorted(rejection_reasons.items())
    ]
    scene_report = {
        "method": "geometric core + DINOv2 place affinity + EXIF capture continuity",
        "embeddingModel": "timm/vit_small_patch14_dinov2",
        "embeddingAvailable": embeddings is not None,
        "embeddingError": embedding_error,
        "indoorMatcher": "Kornia LoFTR indoor_new",
        "indoorMatcherAvailable": learned_error is None,
        "indoorMatcherError": learned_error,
        "indoorMatcherCheckpoint": (
            Path(learned_checkpoint).name if learned_checkpoint else None
        ),
        "captureWindowSeconds": 120,
        "affinityThreshold": (
            round(affinity_threshold, 4)
            if affinity_threshold is not None
            else None
        ),
        "candidates": candidate_reports,
        "note": "Scene recognition only nominates frames; geometric registration decides what enters the splat.",
    }
    return frames, accepted, rejected, recognized, core, scene_report


def _preflight_pair(
    a: PreflightFrame,
    b: PreflightFrame,
) -> tuple[PreflightAlignment | None, str]:
    if a.descriptors is None or b.descriptors is None:
        return None, "Not enough visual features"
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(a.descriptors, b.descriptors, k=2)
    reverse = matcher.knnMatch(b.descriptors, a.descriptors, k=2)
    forward_good = {
        match.queryIdx: match
        for pair in forward
        if len(pair) == 2
        for match, second in [pair]
        if match.distance < 0.72 * second.distance
    }
    reverse_good = {
        match.queryIdx: match
        for pair in reverse
        if len(pair) == 2
        for match, second in [pair]
        if match.distance < 0.72 * second.distance
    }
    symmetric = [
        match
        for match in forward_good.values()
        if reverse_good.get(match.trainIdx) is not None
        and reverse_good[match.trainIdx].trainIdx == match.queryIdx
    ]
    if len(symmetric) < 18:
        return None, f"Only {len(symmetric)} symmetric feature matches"
    points_a = np.float32(
        [a.keypoints[match.queryIdx].pt for match in symmetric]
    ).reshape(-1, 1, 2)
    points_b = np.float32(
        [b.keypoints[match.trainIdx].pt for match in symmetric]
    ).reshape(-1, 1, 2)
    _, fundamental_mask = cv2.findFundamentalMat(
        points_a.reshape(-1, 2),
        points_b.reshape(-1, 2),
        cv2.USAC_MAGSAC,
        1.5,
        0.999,
        10_000,
    )
    if fundamental_mask is None:
        return None, "MAGSAC could not estimate stable two-view geometry"
    method, mask = "sift_fundamental_magsac", fundamental_mask
    inliers = int(mask.ravel().sum())
    ratio = inliers / len(symmetric)
    if inliers < 15 or ratio < 0.22:
        return None, f"Visual geometry kept {inliers}/{len(symmetric)} inliers"
    inlier_points_a = points_a.reshape(-1, 2)[mask.ravel().astype(bool)]
    inlier_points_b = points_b.reshape(-1, 2)[mask.ravel().astype(bool)]
    homography_dominance = _homography_dominance(
        inlier_points_a,
        inlier_points_b,
        threshold_px=5.0,
    )
    spread = _match_spread(
        inlier_points_a,
        inlier_points_b,
        (a.image.shape[1], a.image.shape[0]),
        (b.image.shape[1], b.image.shape[0]),
    )
    distributed_room_evidence = (
        ratio >= 0.45
        and spread["minimumGridCells"] >= 8
        and spread["minimumHullCoverage"] >= 0.20
    )
    if homography_dominance >= 0.72 and not distributed_room_evidence:
        return None, (
            "Matches are dominated by one planar or repeating pattern "
            f"({homography_dominance:.3f}) with only "
            f"{spread['minimumGridCells']} occupied image cells; no safe "
            "room-scale translation"
        )
    confidence = float(
        np.clip(
            0.55 * min(1.0, inliers / 80.0)
            + 0.45 * min(1.0, ratio / 0.65),
            0.0,
            1.0,
        )
    )
    return (
        PreflightAlignment(
            frame_a=a.index,
            frame_b=b.index,
            feature_matches=len(symmetric),
            inliers=inliers,
            inlier_ratio=ratio,
            confidence=confidence,
            method=method,
            points_a=(
                inlier_points_a
                * np.array(
                    [
                        a.source_size[0] / max(a.image.shape[1], 1),
                        a.source_size[1] / max(a.image.shape[0], 1),
                    ],
                    dtype=np.float32,
                )
            ),
            points_b=(
                inlier_points_b
                * np.array(
                    [
                        b.source_size[0] / max(b.image.shape[1], 1),
                        b.source_size[1] / max(b.image.shape[0], 1),
                    ],
                    dtype=np.float32,
                )
            ),
        ),
        "",
    )


def _homography_dominance(
    points_a: np.ndarray,
    points_b: np.ndarray,
    *,
    threshold_px: float,
) -> float:
    """Fraction of epipolar inliers explained by one planar transform.

    A high value is ambiguous for sparse property photos: repeated tile, a
    painting, or one wall can satisfy a homography even when the photographs
    come from different room instances. Such pairs remain separate scenes.
    """
    if len(points_a) < 8 or len(points_a) != len(points_b):
        return 0.0
    _, mask = cv2.findHomography(
        np.asarray(points_a, dtype=np.float32).reshape(-1, 2),
        np.asarray(points_b, dtype=np.float32).reshape(-1, 2),
        cv2.USAC_MAGSAC,
        threshold_px,
        maxIters=10_000,
        confidence=0.999,
    )
    if mask is None:
        return 0.0
    return float(mask.ravel().astype(bool).mean())


def _match_spread(
    points_a: np.ndarray,
    points_b: np.ndarray,
    size_a: tuple[int, int],
    size_b: tuple[int, int],
) -> dict[str, float | int]:
    """Measure whether correspondences span the room instead of one motif."""

    def one(points: np.ndarray, size: tuple[int, int]) -> tuple[int, float]:
        if len(points) < 3 or size[0] <= 0 or size[1] <= 0:
            return 0, 0.0
        values = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        grid_x = np.clip((values[:, 0] / size[0] * 4).astype(int), 0, 3)
        grid_y = np.clip((values[:, 1] / size[1] * 4).astype(int), 0, 3)
        cells = len(set(zip(grid_x.tolist(), grid_y.tolist(), strict=True)))
        hull = cv2.convexHull(values)
        hull_coverage = float(cv2.contourArea(hull) / (size[0] * size[1]))
        return cells, hull_coverage

    cells_a, hull_a = one(points_a, size_a)
    cells_b, hull_b = one(points_b, size_b)
    return {
        "minimumGridCells": min(cells_a, cells_b),
        "minimumHullCoverage": min(hull_a, hull_b),
        "gridCellsA": cells_a,
        "gridCellsB": cells_b,
        "hullCoverageA": hull_a,
        "hullCoverageB": hull_b,
    }


def _strongest_preflight_component(
    frame_count: int,
    alignments: list[PreflightAlignment],
) -> list[int]:
    adjacency: dict[int, set[int]] = {index: set() for index in range(frame_count)}
    for edge in alignments:
        adjacency[edge.frame_a].add(edge.frame_b)
        adjacency[edge.frame_b].add(edge.frame_a)
    unseen = set(range(frame_count))
    components: list[set[int]] = []
    while unseen:
        first = min(unseen)
        nodes = {first}
        queue = [first]
        unseen.remove(first)
        while queue:
            current = queue.pop(0)
            for neighbor in adjacency[current]:
                if neighbor in nodes:
                    continue
                nodes.add(neighbor)
                unseen.discard(neighbor)
                queue.append(neighbor)
        components.append(nodes)
    selected = max(
        components,
        key=lambda nodes: (
            len(nodes),
            sum(
                edge.confidence
                for edge in alignments
                if edge.frame_a in nodes and edge.frame_b in nodes
            ),
            -min(nodes),
        ),
    )
    return sorted(selected)


def _sharp_source(path: Path, frame_dir: Path) -> tuple[np.ndarray, float]:
    quality = json.loads((frame_dir / "sharp_quality.json").read_text("utf-8"))
    width, height = quality["source_resolution"]
    focal_px = float(quality["focal_px"])
    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source)
        image = np.asarray(
            source.convert("RGB").resize((int(width), int(height)), Image.Resampling.LANCZOS)
        )
    return image, focal_px


def _project_cloud(
    cloud: GaussianData,
    width: int,
    height: int,
    focal_px: float,
) -> np.ndarray:
    positions = cloud.positions
    z = positions[:, 2]
    u = np.rint(positions[:, 0] / z * focal_px + width / 2.0).astype(np.int32)
    v = np.rint(positions[:, 1] / z * focal_px + height / 2.0).astype(np.int32)
    valid = (
        np.isfinite(positions).all(axis=1)
        & (z > 0.01)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    valid_indices = np.flatnonzero(valid)
    pixels = v[valid] * width + u[valid]
    depths = z[valid]
    order = np.argsort(depths)
    sorted_pixels = pixels[order]
    _, first = np.unique(sorted_pixels, return_index=True)
    selected = valid_indices[order[first]]
    selected_pixels = pixels[order[first]]
    xyz = np.full((height * width, 3), np.nan, dtype=np.float32)
    xyz[selected_pixels] = positions[selected]
    return xyz.reshape(height, width, 3)


def _align_pair(
    a: FrameData,
    b: FrameData,
    frame_a: int,
    frame_b: int,
    preflight: PreflightAlignment | None = None,
) -> tuple[Alignment | None, str]:
    candidates: list[tuple[str, np.ndarray, np.ndarray, int]] = []
    reasons: list[str] = []
    if a.descriptors is not None and b.descriptors is not None:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        forward = matcher.knnMatch(a.descriptors, b.descriptors, k=2)
        reverse = matcher.knnMatch(b.descriptors, a.descriptors, k=2)
        forward_good = {
            match.queryIdx: match
            for pair in forward
            if len(pair) == 2
            for match, second in [pair]
            if match.distance < 0.72 * second.distance
        }
        reverse_good = {
            match.queryIdx: match
            for pair in reverse
            if len(pair) == 2
            for match, second in [pair]
            if match.distance < 0.72 * second.distance
        }
        symmetric = [
            match
            for match in forward_good.values()
            if reverse_good.get(match.trainIdx) is not None
            and reverse_good[match.trainIdx].trainIdx == match.queryIdx
        ]
        if len(symmetric) >= 24:
            candidates.append(
                (
                    "sift_metric_similarity",
                    np.asarray(
                        [a.keypoints[match.queryIdx].pt for match in symmetric],
                        dtype=np.float32,
                    ),
                    np.asarray(
                        [b.keypoints[match.trainIdx].pt for match in symmetric],
                        dtype=np.float32,
                    ),
                    len(symmetric),
                )
            )
        else:
            reasons.append(f"Only {len(symmetric)} symmetric SIFT matches")
    else:
        reasons.append("Not enough SIFT features")
    if preflight is not None and len(preflight.points_a) >= 12:
        candidates.append(
            (
                f"{preflight.method}_metric_similarity",
                preflight.points_a,
                preflight.points_b,
                preflight.feature_matches,
            )
        )

    for method, pixel_points_a, pixel_points_b, feature_matches in candidates:
        alignment, reason = _align_pixel_correspondences(
            a,
            b,
            frame_a,
            frame_b,
            pixel_points_a,
            pixel_points_b,
            feature_matches,
            method,
        )
        if alignment is not None:
            return alignment, ""
        reasons.append(reason)
    return None, "; ".join(dict.fromkeys(reasons)) or "No verified correspondences"


def _align_pixel_correspondences(
    a: FrameData,
    b: FrameData,
    frame_a: int,
    frame_b: int,
    pixel_points_a: np.ndarray,
    pixel_points_b: np.ndarray,
    feature_matches: int,
    method: str,
) -> tuple[Alignment | None, str]:
    points_a: list[np.ndarray] = []
    points_b: list[np.ndarray] = []
    valid_pixels_a: list[np.ndarray] = []
    valid_pixels_b: list[np.ndarray] = []
    for pixel_a, pixel_b in zip(pixel_points_a, pixel_points_b, strict=True):
        point_a = _nearby_xyz(a.xyz_map, pixel_a)
        point_b = _nearby_xyz(b.xyz_map, pixel_b)
        if point_a is not None and point_b is not None:
            points_a.append(point_a)
            points_b.append(point_b)
            valid_pixels_a.append(pixel_a)
            valid_pixels_b.append(pixel_b)
    if len(points_a) < 12:
        return None, f"Only {len(points_a)} matches had metric SHARP geometry"

    target = np.asarray(points_a, dtype=np.float64)
    source = np.asarray(points_b, dtype=np.float64)
    try:
        transform, inliers, scale, rmse = _ransac_similarity(source, target)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return None, f"Metric similarity failed: {exc}"
    inlier_count = int(inliers.sum())
    ratio = inlier_count / len(source)
    minimum_inliers = 18 if len(source) >= 40 else 10
    if inlier_count < minimum_inliers or ratio < 0.25:
        return None, f"Metric registration kept {inlier_count}/{len(source)} inliers"
    if not 0.5 <= scale <= 2.0:
        return None, f"Predicted scale {scale:.2f} is outside the safe range"
    depth_scale = max(0.08, float(np.median(target[:, 2])) * 0.05)
    if rmse > depth_scale * 1.5:
        return None, f"Metric alignment error {rmse:.3f} m is too high"
    confidence = float(
        np.clip(
            0.35 * min(1.0, inlier_count / 80.0)
            + 0.45 * min(1.0, ratio / 0.65)
            + 0.20 * max(0.0, 1.0 - rmse / (depth_scale * 1.5)),
            0.0,
            1.0,
        )
    )
    inlier_indices = np.flatnonzero(inliers)
    if len(inlier_indices) > 512:
        inlier_indices = inlier_indices[
            np.linspace(0, len(inlier_indices) - 1, 512, dtype=np.int32)
        ]
    retained_points_a = target[inlier_indices]
    retained_points_b = source[inlier_indices]
    retained_pixels_a = np.asarray(valid_pixels_a, dtype=np.float32)[
        inlier_indices
    ]
    retained_pixels_b = np.asarray(valid_pixels_b, dtype=np.float32)[
        inlier_indices
    ]
    return (
        Alignment(
            frame_a=frame_a,
            frame_b=frame_b,
            transform_b_to_a=transform,
            feature_matches=feature_matches,
            metric_matches=len(source),
            inliers=inlier_count,
            inlier_ratio=ratio,
            rmse_m=rmse,
            scale=scale,
            confidence=confidence,
            method=method,
            points_a=retained_points_a,
            points_b=retained_points_b,
            pixels_a=retained_pixels_a,
            pixels_b=retained_pixels_b,
        ),
        "",
    )


def _nearby_xyz(xyz_map: np.ndarray, point: tuple[float, float]) -> np.ndarray | None:
    x = int(round(point[0]))
    y = int(round(point[1]))
    height, width = xyz_map.shape[:2]
    for radius in range(0, 5):
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        patch = xyz_map[y0:y1, x0:x1].reshape(-1, 3)
        valid = patch[np.isfinite(patch).all(axis=1)]
        if len(valid):
            return valid[np.argmin(valid[:, 2])]
    return None


def _ransac_similarity(
    source: np.ndarray,
    target: np.ndarray,
    iterations: int = 1200,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    rng = np.random.default_rng(20260716)
    threshold = max(0.08, float(np.median(target[:, 2])) * 0.05)
    best_inliers = np.zeros(len(source), dtype=bool)
    best_error = float("inf")
    for _ in range(iterations):
        sample = rng.choice(len(source), 3, replace=False)
        try:
            transform, _ = _umeyama(source[sample], target[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        predicted = _apply_points(source, transform)
        errors = np.linalg.norm(predicted - target, axis=1)
        inliers = errors <= threshold
        count = int(inliers.sum())
        error = float(np.mean(errors[inliers])) if count else float("inf")
        if count > int(best_inliers.sum()) or (
            count == int(best_inliers.sum()) and error < best_error
        ):
            best_inliers = inliers
            best_error = error
    if int(best_inliers.sum()) < 3:
        raise ValueError("No stable similarity transform")
    transform, scale = _umeyama(source[best_inliers], target[best_inliers])
    residuals = np.linalg.norm(
        _apply_points(source[best_inliers], transform) - target[best_inliers], axis=1
    )
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return transform, best_inliers, scale, rmse


def _umeyama(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = float(np.sum(source_centered**2) / len(source))
    if variance < 1e-10:
        raise ValueError("Degenerate registration points")
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    parity = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        parity[-1] = -1
    rotation = u @ np.diag(parity) @ vt
    scale = float(np.sum(singular * parity) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform, scale


def _apply_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _connection_tree(
    frame_count: int,
    alignments: list[Alignment],
) -> tuple[dict[int, np.ndarray], list[Alignment], int]:
    """Return transforms for the strongest connected capture component.

    A bad first capture must not discard a valid overlap group later in the
    sequence. We build a maximum-confidence spanning forest, choose its largest
    component (confidence breaks ties), and anchor that component at its most
    strongly connected frame.
    """
    parent = list(range(frame_count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    tree: list[Alignment] = []
    for edge in sorted(alignments, key=lambda item: item.confidence, reverse=True):
        root_a, root_b = find(edge.frame_a), find(edge.frame_b)
        if root_a == root_b:
            continue
        parent[root_b] = root_a
        tree.append(edge)

    adjacency: dict[int, list[tuple[int, np.ndarray]]] = {
        index: [] for index in range(frame_count)
    }
    for edge in tree:
        adjacency[edge.frame_a].append((edge.frame_b, edge.transform_b_to_a))
        adjacency[edge.frame_b].append(
            (edge.frame_a, np.linalg.inv(edge.transform_b_to_a))
        )
    unseen = set(range(frame_count))
    components: list[tuple[set[int], list[Alignment]]] = []
    while unseen:
        first = min(unseen)
        nodes = {first}
        queue = [first]
        unseen.remove(first)
        while queue:
            current = queue.pop(0)
            for neighbor, _ in adjacency[current]:
                if neighbor in nodes:
                    continue
                nodes.add(neighbor)
                unseen.discard(neighbor)
                queue.append(neighbor)
        edges = [
            edge
            for edge in tree
            if edge.frame_a in nodes and edge.frame_b in nodes
        ]
        components.append((nodes, edges))

    selected_nodes, selected_edges = max(
        components,
        key=lambda component: (
            len(component[0]),
            sum(edge.confidence for edge in component[1]),
            -min(component[0]),
        ),
    )
    anchor = max(
        selected_nodes,
        key=lambda node: (
            sum(
                edge.confidence
                for edge in selected_edges
                if node in {edge.frame_a, edge.frame_b}
            ),
            -node,
        ),
    )

    transforms = {anchor: np.eye(4, dtype=np.float64)}
    queue = [anchor]
    while queue:
        current = queue.pop(0)
        for neighbor, transform_neighbor_to_current in adjacency[current]:
            if neighbor in transforms:
                continue
            transforms[neighbor] = transforms[current] @ transform_neighbor_to_current
            queue.append(neighbor)
    connected_tree = [
        edge
        for edge in selected_edges
        if edge.frame_a in transforms and edge.frame_b in transforms
    ]
    return transforms, connected_tree, anchor


def _refine_pose_graph(
    initial: dict[int, np.ndarray],
    alignments: list[Alignment],
    anchor: int,
    *,
    require_loop: bool = True,
    transform_rmse_tolerance: float = 0.0,
    point_weight_multiplier: float = 1.5,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Jointly refine every connected camera against transforms and points.

    The maximum-confidence tree supplies a stable initialization. Non-tree
    edges then close loops and distribute residual error instead of letting a
    sequence of pairwise transforms bend the room. Verified SHARP inlier
    points remain in the optimization instead of being reduced to one Sim(3)
    summary per pair. The anchor stays fixed so the exported coordinate frame
    remains deterministic.
    """
    nodes = sorted(initial)
    edges = [
        edge
        for edge in alignments
        if edge.frame_a in initial and edge.frame_b in initial
    ]
    movable = [node for node in nodes if node != anchor]
    if not movable or (require_loop and len(edges) < len(nodes)):
        return initial, {
            "method": "correspondence-aware robust Sim(3) pose graph",
            "optimized": False,
            "reason": (
                "No loop-closing edge was available"
                if movable
                else "No movable camera was available"
            ),
            "nodeCount": len(nodes),
            "edgeCount": len(edges),
            "pointConstraintCount": 0,
        }

    offsets = {node: index * 7 for index, node in enumerate(movable)}
    initial_vector = np.concatenate(
        [_similarity_parameters(initial[node]) for node in movable]
    )
    translation_scale = max(
        0.25,
        float(
            np.median(
                [np.linalg.norm(edge.transform_b_to_a[:3, 3]) for edge in edges]
            )
        ),
    )

    def unpack(parameters: np.ndarray) -> dict[int, np.ndarray]:
        transforms = {anchor: initial[anchor].copy()}
        for node in movable:
            start = offsets[node]
            transforms[node] = _parameters_to_similarity(parameters[start : start + 7])
        return transforms

    point_edges = [
        edge
        for edge in edges
        if len(edge.points_a) >= 3 and len(edge.points_a) == len(edge.points_b)
    ]
    point_constraint_count = int(
        sum(len(edge.points_a) for edge in point_edges)
    )

    def transform_residuals(parameters: np.ndarray) -> np.ndarray:
        transforms = unpack(parameters)
        values: list[np.ndarray] = []
        for edge in edges:
            predicted = np.linalg.inv(transforms[edge.frame_a]) @ transforms[edge.frame_b]
            observed = edge.transform_b_to_a
            pred_scale, pred_rotation = _similarity_scale_rotation(predicted)
            obs_scale, obs_rotation = _similarity_scale_rotation(observed)
            rotation_error = Rotation.from_matrix(
                pred_rotation @ obs_rotation.T
            ).as_rotvec()
            translation_error = (
                predicted[:3, 3] - observed[:3, 3]
            ) / translation_scale
            scale_error = np.array(
                [np.log(max(pred_scale, 1e-8) / max(obs_scale, 1e-8))],
                dtype=np.float64,
            )
            weight = np.sqrt(max(0.05, edge.confidence))
            values.append(
                weight
                * np.concatenate(
                    [rotation_error, translation_error, scale_error]
                )
            )
        return np.concatenate(values)

    def residuals(parameters: np.ndarray) -> np.ndarray:
        transforms = unpack(parameters)
        values: list[np.ndarray] = [transform_residuals(parameters)]
        for edge in point_edges:
            shared_a = _apply_points(
                edge.points_a,
                transforms[edge.frame_a],
            )
            shared_b = _apply_points(
                edge.points_b,
                transforms[edge.frame_b],
            )
            weight = np.sqrt(max(0.05, edge.confidence))
            point_weight = (
                point_weight_multiplier
                * weight
                / np.sqrt(float(len(edge.points_a)))
                / translation_scale
            )
            values.append(
                point_weight * (shared_a - shared_b).reshape(-1)
            )
        return np.concatenate(values)

    def point_rmse(transforms: dict[int, np.ndarray]) -> float | None:
        squared: list[np.ndarray] = []
        for edge in point_edges:
            shared_a = _apply_points(
                edge.points_a,
                transforms[edge.frame_a],
            )
            shared_b = _apply_points(
                edge.points_b,
                transforms[edge.frame_b],
            )
            squared.append(np.sum((shared_a - shared_b) ** 2, axis=1))
        if not squared:
            return None
        return float(np.sqrt(np.mean(np.concatenate(squared))))

    initial_transforms = unpack(initial_vector)
    point_before = point_rmse(initial_transforms)
    before = residuals(initial_vector)
    transform_before = transform_residuals(initial_vector)
    result = least_squares(
        residuals,
        initial_vector,
        loss="soft_l1",
        f_scale=0.5,
        max_nfev=250,
    )
    after = residuals(result.x)
    transform_after = transform_residuals(result.x)
    combined_before_rmse = float(np.sqrt(np.mean(before**2)))
    combined_after_rmse = float(np.sqrt(np.mean(after**2)))
    before_rmse = float(np.sqrt(np.mean(transform_before**2)))
    after_rmse = float(np.sqrt(np.mean(transform_after**2)))
    refined_transforms = unpack(result.x)
    point_after = point_rmse(refined_transforms)
    accepted = bool(
        result.success
        and np.isfinite(result.x).all()
        and combined_after_rmse <= combined_before_rmse + 1e-8
        and after_rmse <= before_rmse + transform_rmse_tolerance + 1e-8
        and (
            point_before is None
            or point_after is not None
            and point_after <= point_before + 1e-8
        )
    )
    return (refined_transforms if accepted else initial), {
        "method": "correspondence-aware robust Sim(3) pose graph",
        "optimized": accepted,
        "nodeCount": len(nodes),
        "edgeCount": len(edges),
        "loopEdgeCount": max(0, len(edges) - (len(nodes) - 1)),
        "pointConstraintCount": point_constraint_count,
        "beforeRmse": round(before_rmse, 7),
        "afterRmse": round(after_rmse if accepted else before_rmse, 7),
        "candidateAfterRmse": round(after_rmse, 7),
        "transformRmseTolerance": transform_rmse_tolerance,
        "pointWeightMultiplier": point_weight_multiplier,
        "combinedBeforeRmse": round(combined_before_rmse, 7),
        "combinedCandidateAfterRmse": round(combined_after_rmse, 7),
        "pointBeforeRmseMeters": (
            round(point_before, 7) if point_before is not None else None
        ),
        "pointAfterRmseMeters": (
            round(point_after if accepted else point_before, 7)
            if point_before is not None
            else None
        ),
        "candidatePointAfterRmseMeters": (
            round(point_after, 7) if point_after is not None else None
        ),
        "rejectionReason": (
            None
            if accepted
            else "Candidate did not improve transform and point residuals together."
        ),
        "iterations": int(result.nfev),
    }


def _refine_with_dense_surface_constraints(
    frames_by_index: dict[int, FrameData],
    initial: dict[int, np.ndarray],
    metric_alignments: list[Alignment],
    auxiliary_alignments: list[Alignment],
    anchor: int,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Tighten registered cameras with reciprocal, measured surface matches.

    The feature pose graph establishes identity and coarse placement first.
    This second pass samples only the SHARP depth surfaces belonging to an
    already verified image pair. Reciprocal nearest-neighbor, normal, color,
    and spatial-diversity checks prevent an unrelated wall or repeated object
    from becoming a new registration edge.
    """

    (
        augmented_alignments,
        dense_only_alignments,
        match_report,
    ) = _dense_surface_alignments(
        frames_by_index,
        initial,
        metric_alignments,
    )
    dense_count = int(
        sum(len(edge.points_a) for edge in dense_only_alignments)
    )
    if dense_count < 24:
        return initial, {
            "method": "reciprocal measured-surface Sim(3) refinement",
            "optimized": False,
            "reason": "Too few safe dense surface correspondences were found.",
            "denseConstraintCount": dense_count,
            "pairs": match_report,
        }

    sparse_before = _alignment_point_rmse(metric_alignments, initial)
    dense_before = _alignment_point_rmse(dense_only_alignments, initial)
    candidate, optimization = _refine_pose_graph(
        initial,
        [*augmented_alignments, *auxiliary_alignments],
        anchor,
        require_loop=False,
        transform_rmse_tolerance=0.05,
        point_weight_multiplier=4.0,
    )
    sparse_after = _alignment_point_rmse(metric_alignments, candidate)
    dense_after = _alignment_point_rmse(dense_only_alignments, candidate)

    maximum_translation_delta = 0.0
    maximum_rotation_delta = 0.0
    for node in initial:
        if node not in candidate:
            continue
        delta = np.linalg.inv(initial[node]) @ candidate[node]
        _, rotation = _similarity_scale_rotation(delta)
        maximum_translation_delta = max(
            maximum_translation_delta,
            float(np.linalg.norm(delta[:3, 3])),
        )
        maximum_rotation_delta = max(
            maximum_rotation_delta,
            float(
                np.degrees(
                    np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec())
                )
            ),
        )

    sparse_tolerance = max(
        0.001,
        (sparse_before or 0.0) * 0.02,
    )
    accepted = bool(
        optimization.get("optimized") is True
        and dense_before is not None
        and dense_after is not None
        and dense_after <= dense_before + 1e-8
        and (
            sparse_before is None
            or sparse_after is not None
            and sparse_after <= sparse_before + sparse_tolerance
        )
        and maximum_translation_delta <= 0.15
        and maximum_rotation_delta <= 3.0
    )
    if not accepted:
        candidate = initial
        dense_after = dense_before
        sparse_after = sparse_before

    return candidate, {
        "method": "reciprocal measured-surface Sim(3) refinement",
        "optimized": accepted,
        "denseConstraintCount": dense_count,
        "denseBeforeRmseMeters": (
            round(dense_before, 7) if dense_before is not None else None
        ),
        "denseAfterRmseMeters": (
            round(dense_after, 7) if dense_after is not None else None
        ),
        "sparseBeforeRmseMeters": (
            round(sparse_before, 7) if sparse_before is not None else None
        ),
        "sparseAfterRmseMeters": (
            round(sparse_after, 7) if sparse_after is not None else None
        ),
        "maximumCameraTranslationDeltaMeters": round(
            maximum_translation_delta, 7
        ),
        "maximumCameraRotationDeltaDegrees": round(
            maximum_rotation_delta, 5
        ),
        "pairs": match_report,
        "optimizer": optimization,
        "rejectionReason": (
            None
            if accepted
            else (
                "Candidate failed dense improvement, sparse preservation, "
                "or bounded camera-motion checks."
            )
        ),
    }


def _dense_surface_alignments(
    frames: dict[int, FrameData],
    transforms: dict[int, np.ndarray],
    alignments: list[Alignment],
    *,
    maximum_samples_per_frame: int = 30_000,
    maximum_matches_per_pair: int = 1_200,
) -> tuple[list[Alignment], list[Alignment], list[dict[str, Any]]]:
    sample_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    augmented: list[Alignment] = []
    dense_only: list[Alignment] = []
    reports: list[dict[str, Any]] = []

    for alignment in alignments:
        frame_a = frames.get(alignment.frame_a)
        frame_b = frames.get(alignment.frame_b)
        if (
            frame_a is None
            or frame_b is None
            or alignment.frame_a not in transforms
            or alignment.frame_b not in transforms
        ):
            augmented.append(alignment)
            continue
        if alignment.frame_a not in sample_cache:
            sample_cache[alignment.frame_a] = _sample_surface_points(
                frame_a,
                transforms[alignment.frame_a],
                maximum_samples_per_frame,
            )
        if alignment.frame_b not in sample_cache:
            sample_cache[alignment.frame_b] = _sample_surface_points(
                frame_b,
                transforms[alignment.frame_b],
                maximum_samples_per_frame,
            )
        local_a, world_a, normals_a, colors_a = sample_cache[alignment.frame_a]
        local_b, world_b, normals_b, colors_b = sample_cache[alignment.frame_b]
        if len(world_a) < 24 or len(world_b) < 24:
            augmented.append(alignment)
            reports.append(
                {
                    "frameA": alignment.frame_a,
                    "frameB": alignment.frame_b,
                    "accepted": 0,
                    "reason": "Insufficient continuous depth surface samples.",
                }
            )
            continue

        distance_limit = float(
            np.clip(max(0.035, alignment.rmse_m * 2.5), 0.035, 0.1)
        )
        tree_b = cKDTree(world_b)
        distances, indices_b = tree_b.query(world_a, k=1, workers=-1)
        tree_a = cKDTree(world_a)
        _, reciprocal_a = tree_a.query(
            world_b[indices_b],
            k=1,
            workers=-1,
        )
        rows_a = np.arange(len(world_a))
        rows_b = indices_b
        normal_agreement = np.abs(
            np.sum(normals_a * normals_b[rows_b], axis=1)
        )
        color_distance = np.linalg.norm(colors_a - colors_b[rows_b], axis=1)
        valid = (
            (reciprocal_a == rows_a)
            & np.isfinite(distances)
            & (distances <= distance_limit)
            & (normal_agreement >= 0.6)
            & (color_distance <= 0.35)
        )
        candidates = np.flatnonzero(valid)
        if len(candidates):
            candidates = candidates[np.argsort(distances[candidates])]
            midpoint = (world_a[candidates] + world_b[rows_b[candidates]]) * 0.5
            voxel_size = max(0.025, distance_limit * 0.6)
            cells = np.floor(midpoint / voxel_size).astype(np.int64)
            _, first = np.unique(cells, axis=0, return_index=True)
            candidates = candidates[np.sort(first)]
            candidates = candidates[:maximum_matches_per_pair]
        selected_b = rows_b[candidates]
        points_a = local_a[candidates].astype(np.float64, copy=False)
        points_b = local_b[selected_b].astype(np.float64, copy=False)
        accepted_count = len(points_a)
        reports.append(
            {
                "frameA": alignment.frame_a,
                "frameB": alignment.frame_b,
                "accepted": accepted_count,
                "distanceLimitMeters": round(distance_limit, 5),
                "medianDistanceMeters": (
                    round(float(np.median(distances[candidates])), 6)
                    if accepted_count
                    else None
                ),
                "checks": (
                    "reciprocal nearest neighbor + surface normal + RGB + "
                    "spatial diversity"
                ),
            }
        )
        if accepted_count < 12:
            augmented.append(alignment)
            continue

        augmented.append(
            replace(
                alignment,
                method=f"{alignment.method}+dense_surface",
                points_a=np.concatenate([alignment.points_a, points_a]),
                points_b=np.concatenate([alignment.points_b, points_b]),
            )
        )
        dense_only.append(
            replace(
                alignment,
                method="reciprocal_dense_surface",
                points_a=points_a,
                points_b=points_b,
                pixels_a=np.empty((0, 2), dtype=np.float32),
                pixels_b=np.empty((0, 2), dtype=np.float32),
            )
        )
    return augmented, dense_only, reports


def _sample_surface_points(
    frame: FrameData,
    transform: np.ndarray,
    maximum_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cloud = frame.cloud
    valid = (
        np.isfinite(cloud.positions).all(axis=1)
        & np.isfinite(cloud.scales).all(axis=1)
        & np.isfinite(cloud.rotations).all(axis=1)
        & np.isfinite(cloud.colors).all(axis=1)
        & (cloud.positions[:, 2] > 0.02)
    )
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, empty, empty, empty
    if len(valid_indices) > maximum_samples:
        sample_rows = np.linspace(
            0,
            len(valid_indices) - 1,
            maximum_samples,
            dtype=np.int64,
        )
        selected = valid_indices[sample_rows]
    else:
        selected = valid_indices

    local = cloud.positions[selected].astype(np.float64, copy=False)
    rotations = _wxyz_to_matrices(cloud.rotations[selected]).astype(
        np.float64,
        copy=False,
    )
    normal_axes = np.argmin(cloud.scales[selected], axis=1)
    normals = rotations[np.arange(len(selected)), :, normal_axes]
    _, rotation = _similarity_scale_rotation(transform)
    world = _apply_points(local, transform)
    world_normals = normals @ rotation.T
    world_normals /= np.maximum(
        np.linalg.norm(world_normals, axis=1, keepdims=True),
        1e-8,
    )
    colors = cloud.colors[selected].astype(np.float64, copy=False)
    return local, world, world_normals, colors


def _alignment_point_rmse(
    alignments: list[Alignment],
    transforms: dict[int, np.ndarray],
) -> float | None:
    squared: list[np.ndarray] = []
    for edge in alignments:
        if (
            edge.frame_a not in transforms
            or edge.frame_b not in transforms
            or len(edge.points_a) == 0
            or len(edge.points_a) != len(edge.points_b)
        ):
            continue
        shared_a = _apply_points(edge.points_a, transforms[edge.frame_a])
        shared_b = _apply_points(edge.points_b, transforms[edge.frame_b])
        squared.append(np.sum((shared_a - shared_b) ** 2, axis=1))
    if not squared:
        return None
    return float(np.sqrt(np.mean(np.concatenate(squared))))


def _similarity_scale_rotation(transform: np.ndarray) -> tuple[float, np.ndarray]:
    linear = transform[:3, :3]
    scale = float(np.cbrt(abs(np.linalg.det(linear))))
    rotation = linear / max(scale, 1e-8)
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return scale, rotation


def _similarity_parameters(transform: np.ndarray) -> np.ndarray:
    scale, rotation = _similarity_scale_rotation(transform)
    return np.concatenate(
        [
            Rotation.from_matrix(rotation).as_rotvec(),
            transform[:3, 3],
            np.array([np.log(max(scale, 1e-8))], dtype=np.float64),
        ]
    )


def _parameters_to_similarity(parameters: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    scale = float(np.exp(np.clip(parameters[6], -2.0, 2.0)))
    transform[:3, :3] = scale * Rotation.from_rotvec(parameters[:3]).as_matrix()
    transform[:3, 3] = parameters[3:6]
    return transform


def _regularize_source_footprints(
    transformed_clouds: dict[int, GaussianData],
    frames: dict[int, FrameData],
    transforms: dict[int, np.ndarray],
) -> tuple[dict[int, GaussianData], dict[str, Any]]:
    """Close sub-pixel pinholes without changing source detail or geometry.

    Every SHARP Gaussian is tied to a source ray. Its projected world-space
    pixel footprint is therefore measured from source depth and focal length.
    Only the two broad surface axes may grow, by at most 15 percent, when they
    undersample that measured footprint. The thin normal axis, positions,
    colors, opacity, rotations, count, and image resolution remain unchanged.
    """

    regularized: dict[int, GaussianData] = {}
    frame_reports: list[dict[str, Any]] = []
    total_count = 0
    total_affected = 0
    maximum_factor = 1.0
    target_pixel_sigma = 0.55
    maximum_growth_factor = 1.15

    for index, cloud in transformed_clouds.items():
        frame = frames.get(index)
        transform = transforms.get(index)
        if frame is None or transform is None or cloud.count == 0:
            regularized[index] = cloud
            continue
        similarity_scale, _ = _similarity_scale_rotation(transform)
        local_depth = frame.cloud.positions[:, 2].astype(np.float64)
        if len(local_depth) != cloud.count:
            regularized[index] = cloud
            frame_reports.append(
                {
                    "frame": index,
                    "gaussianCount": cloud.count,
                    "affected": 0,
                    "reason": "Source ray count did not match Gaussian count.",
                }
            )
            total_count += cloud.count
            continue

        linear = np.exp(
            np.clip(cloud.scales.astype(np.float64), -20.0, 20.0)
        )
        order = np.argsort(linear, axis=1)
        rows = np.arange(cloud.count)
        tangent_axes = order[:, 1:]
        source_to_gaussian_pitch = float(
            np.sqrt(
                (frame.image.shape[0] * frame.image.shape[1])
                / max(1, cloud.count)
            )
        )
        desired = (
            np.maximum(local_depth, 0.0)
            * similarity_scale
            / max(frame.focal_px, 1e-6)
            * source_to_gaussian_pitch
            * target_pixel_sigma
        )
        updated = linear.copy()
        growth = np.ones((cloud.count, 2), dtype=np.float64)
        valid = np.isfinite(desired) & (desired > 0.0)
        for tangent_index in range(2):
            axes = tangent_axes[:, tangent_index]
            current = linear[rows, axes]
            target = np.minimum(
                np.maximum(current, desired),
                current * maximum_growth_factor,
            )
            target = np.where(valid, target, current)
            updated[rows, axes] = target
            growth[:, tangent_index] = target / np.maximum(current, 1e-8)
        affected = np.max(growth, axis=1) > 1.00001
        affected_count = int(affected.sum())
        frame_maximum = (
            float(np.max(growth[affected])) if affected_count else 1.0
        )
        scales = (
            np.log(np.maximum(updated, 1e-8)).astype(np.float32)
            if affected_count
            else cloud.scales
        )
        regularized[index] = GaussianData(
            positions=cloud.positions,
            scales=scales,
            rotations=cloud.rotations,
            colors=cloud.colors,
            opacities=cloud.opacities,
            sh_coeffs=cloud.sh_coeffs,
            errors=cloud.errors,
        )
        total_count += cloud.count
        total_affected += affected_count
        maximum_factor = max(maximum_factor, frame_maximum)
        frame_reports.append(
            {
                "frame": index,
                "gaussianCount": cloud.count,
                "affected": affected_count,
                "affectedFraction": round(
                    affected_count / max(1, cloud.count), 8
                ),
                "sourceToGaussianPixelPitch": round(
                    source_to_gaussian_pitch, 5
                ),
                "maximumGrowthFactor": round(frame_maximum, 5),
            }
        )

    return regularized, {
        "method": "source-ray projected footprint regularization",
        "applied": total_affected > 0,
        "targetPixelSigma": target_pixel_sigma,
        "maximumGrowthFactorAllowed": maximum_growth_factor,
        "maximumGrowthFactorApplied": round(maximum_factor, 5),
        "gaussianCountBefore": total_count,
        "gaussianCountAfter": total_count,
        "affected": total_affected,
        "affectedFraction": round(total_affected / max(1, total_count), 8),
        "frames": frame_reports,
        "note": (
            "Only undersized source-supported tangent footprints grow; no "
            "Gaussian, source pixel, or thin surface axis is removed."
        ),
    }


def _cross_view_cleanup(
    transformed_clouds: dict[int, GaussianData],
    frames: dict[int, FrameData],
    transforms: dict[int, np.ndarray],
) -> tuple[dict[int, GaussianData], dict[str, Any]]:
    """Conservatively remove Gaussians contradicted by multiple measured views.

    A point is never removed merely because another camera cannot see it.
    Pruning requires at least two overlapping observations, no supporting depth,
    and a consistent in-front-of-surface contradiction in every observation.
    This targets floaters while retaining occluded and single-view detail.
    """
    cleaned: dict[int, GaussianData] = {}
    frame_reports: list[dict[str, Any]] = []
    total_before = 0
    total_observed = 0
    total_supported = 0
    total_supported_twice = 0
    required_observations = min(2, max(1, len(transforms) - 1))
    for source_index, cloud in transformed_clouds.items():
        observations = np.zeros(cloud.count, dtype=np.uint8)
        supports = np.zeros(cloud.count, dtype=np.uint8)
        contradictions = np.zeros(cloud.count, dtype=np.uint8)
        world_positions = cloud.positions.astype(np.float64)
        for target_index, target in frames.items():
            if target_index == source_index or target_index not in transforms:
                continue
            world_to_target = np.linalg.inv(transforms[target_index])
            target_positions = _apply_points(world_positions, world_to_target)
            z = target_positions[:, 2]
            width = target.xyz_map.shape[1]
            height = target.xyz_map.shape[0]
            valid_projection = np.isfinite(target_positions).all(axis=1) & (z > 0.01)
            projected_x = np.full(len(z), -1.0, dtype=np.float64)
            projected_y = np.full(len(z), -1.0, dtype=np.float64)
            projected_x[valid_projection] = (
                target_positions[valid_projection, 0]
                / z[valid_projection]
                * target.focal_px
                + width / 2.0
            )
            projected_y[valid_projection] = (
                target_positions[valid_projection, 1]
                / z[valid_projection]
                * target.focal_px
                + height / 2.0
            )
            x = np.rint(np.clip(projected_x, -1.0, float(width))).astype(np.int32)
            y = np.rint(np.clip(projected_y, -1.0, float(height))).astype(np.int32)
            inside = (
                valid_projection
                & (x >= 0)
                & (x < width)
                & (y >= 0)
                & (y < height)
            )
            candidates = np.flatnonzero(inside)
            if len(candidates) == 0:
                continue
            reference = target.xyz_map[y[candidates], x[candidates], 2]
            measured = np.isfinite(reference)
            candidates = candidates[measured]
            reference = reference[measured]
            if len(candidates) == 0:
                continue
            predicted = z[candidates]
            tolerance = np.maximum(0.08, reference * 0.06)
            observations[candidates] += 1
            supports[candidates] += (np.abs(predicted - reference) <= tolerance).astype(
                np.uint8
            )
            contradictions[candidates] += (
                predicted < reference - tolerance
            ).astype(np.uint8)

        remove = (
            (observations >= required_observations)
            & (supports == 0)
            & (contradictions == observations)
        )
        keep = ~remove
        cleaned[source_index] = _select_cloud(cloud, keep)
        observed_count = int((observations >= 1).sum())
        supported_count = int((supports >= 1).sum())
        supported_twice_count = int((supports >= 2).sum())
        total_before += cloud.count
        total_observed += observed_count
        total_supported += supported_count
        total_supported_twice += supported_twice_count
        frame_reports.append(
            {
                "frame": source_index,
                "before": cloud.count,
                "after": int(keep.sum()),
                "removed": int(remove.sum()),
                "multiViewObserved": int(
                    (observations >= required_observations).sum()
                ),
                "projectedIntoAnotherMeasuredView": observed_count,
                "crossViewSupported": supported_count,
                "supportedByTwoOtherViews": supported_twice_count,
            }
        )
    return cleaned, {
        "method": "conservative cross-view depth-consistency pruning",
        "requiredObservations": required_observations,
        "removed": int(
            sum(report["removed"] for report in frame_reports)
        ),
        "projectedIntoAnotherMeasuredView": total_observed,
        "crossViewSupported": total_supported,
        "supportedByTwoOtherViews": total_supported_twice,
        "crossViewObservedFraction": round(
            total_observed / max(1, total_before), 8
        ),
        "crossViewSupportedFraction": round(
            total_supported / max(1, total_before), 8
        ),
        "frames": frame_reports,
        "note": "Unseen and occluded Gaussians are retained; only repeated front-surface contradictions are pruned.",
    }


def _regularize_extreme_scales(
    transformed_clouds: dict[int, GaussianData],
) -> tuple[dict[int, GaussianData], dict[str, Any]]:
    """Shrink only rare single-axis needles while preserving every Gaussian.

    A surface splat is expected to be thin on one axis, so the ordinary
    largest/smallest aspect ratio is not a safe artifact test. A needle has
    only one abnormally broad principal axis. This pass changes that axis only
    when it is both above the per-view 99.5th percentile and more than 12 times
    the second-largest axis. Positions, colors, opacities, rotations, source
    resolution, and Gaussian count are unchanged.
    """

    regularized: dict[int, GaussianData] = {}
    frame_reports: list[dict[str, Any]] = []
    total_affected = 0
    total_count = 0
    maximum_before = 0.0
    maximum_after = 0.0
    ratio_limit = 12.0
    tail_quantile = 0.995
    for index, cloud in transformed_clouds.items():
        linear = np.exp(
            np.clip(cloud.scales.astype(np.float64), -20.0, 20.0)
        )
        order = np.argsort(linear, axis=1)
        rows = np.arange(cloud.count)
        largest_axis = order[:, 2]
        largest = linear[rows, largest_axis]
        second = linear[rows, order[:, 1]]
        ratios = largest / np.maximum(second, 1e-8)
        tail = float(np.quantile(largest, tail_quantile))
        affected = (ratios > ratio_limit) & (largest >= tail)
        affected_count = int(affected.sum())
        scales = cloud.scales
        after_ratios = ratios
        if affected_count:
            updated_linear = linear.copy()
            affected_rows = rows[affected]
            affected_axes = largest_axis[affected]
            updated_linear[affected_rows, affected_axes] = np.minimum(
                largest[affected],
                second[affected] * ratio_limit,
            )
            scales = np.log(np.maximum(updated_linear, 1e-8)).astype(
                np.float32
            )
            updated_sorted = np.sort(updated_linear, axis=1)
            after_ratios = updated_sorted[:, 2] / np.maximum(
                updated_sorted[:, 1], 1e-8
            )
        regularized[index] = GaussianData(
            positions=cloud.positions,
            scales=scales,
            rotations=cloud.rotations,
            colors=cloud.colors,
            opacities=cloud.opacities,
            sh_coeffs=cloud.sh_coeffs,
            errors=cloud.errors,
        )
        before_max = float(np.max(ratios)) if cloud.count else 0.0
        after_max = float(np.max(after_ratios)) if cloud.count else 0.0
        maximum_before = max(maximum_before, before_max)
        maximum_after = max(maximum_after, after_max)
        total_affected += affected_count
        total_count += cloud.count
        frame_reports.append(
            {
                "frame": index,
                "gaussianCount": cloud.count,
                "affected": affected_count,
                "affectedFraction": round(
                    affected_count / max(1, cloud.count), 8
                ),
                "tailLinearScaleMeters": round(tail, 7),
                "maximumPrincipalRatioBefore": round(before_max, 4),
                "maximumPrincipalRatioAfter": round(after_max, 4),
            }
        )
    return regularized, {
        "method": "rare single-principal-axis scale regularization",
        "applied": total_affected > 0,
        "ratioLimit": ratio_limit,
        "tailQuantile": tail_quantile,
        "gaussianCountBefore": total_count,
        "gaussianCountAfter": total_count,
        "affected": total_affected,
        "affectedFraction": round(total_affected / max(1, total_count), 8),
        "maximumPrincipalRatioBefore": round(maximum_before, 4),
        "maximumPrincipalRatioAfter": round(maximum_after, 4),
        "frames": frame_reports,
        "note": (
            "No Gaussian was culled; ordinary two-axis surface splats are "
            "preserved."
        ),
    }


def _sample_image_pixels(
    image: np.ndarray,
    pixels: np.ndarray,
) -> np.ndarray:
    coordinates = np.rint(pixels).astype(np.int64)
    x = np.clip(coordinates[:, 0], 0, image.shape[1] - 1)
    y = np.clip(coordinates[:, 1], 0, image.shape[0] - 1)
    return image[y, x].astype(np.float64) / 255.0


def _joint_color_balance(
    transformed_clouds: dict[int, GaussianData],
    frames: dict[int, FrameData],
    alignments: list[Alignment],
) -> tuple[dict[int, GaussianData], dict[str, Any]]:
    """Reduce seams using only verified views of the same physical features.

    Global image medians are biased when two cameras see different parts of a
    room. Instead, each accepted alignment contributes robust RGB ratios at its
    retained inlier pixels. A regularized graph solve estimates one bounded
    gain per registered source and keeps a strongly connected frame fixed.
    """

    nodes = sorted(transformed_clouds)
    if len(nodes) < 2:
        return transformed_clouds, {
            "method": "verified-overlap RGB gain graph",
            "applied": False,
            "reason": "At least two registered views are required.",
        }
    usable_edges: list[tuple[Alignment, np.ndarray, int]] = []
    degree = {node: 0.0 for node in nodes}
    for alignment in alignments:
        if (
            alignment.frame_a not in transformed_clouds
            or alignment.frame_b not in transformed_clouds
            or len(alignment.pixels_a) < 12
            or len(alignment.pixels_a) != len(alignment.pixels_b)
        ):
            continue
        colors_a = _sample_image_pixels(
            frames[alignment.frame_a].image,
            alignment.pixels_a,
        )
        colors_b = _sample_image_pixels(
            frames[alignment.frame_b].image,
            alignment.pixels_b,
        )
        luminance_a = colors_a @ np.asarray([0.2126, 0.7152, 0.0722])
        luminance_b = colors_b @ np.asarray([0.2126, 0.7152, 0.0722])
        valid = (
            (luminance_a > 0.08)
            & (luminance_a < 0.94)
            & (luminance_b > 0.08)
            & (luminance_b < 0.94)
            & (colors_a.min(axis=1) > 0.025)
            & (colors_b.min(axis=1) > 0.025)
            & (colors_a.max(axis=1) < 0.98)
            & (colors_b.max(axis=1) < 0.98)
        )
        if int(valid.sum()) < 12:
            continue
        log_ratio = np.median(
            np.log(np.maximum(colors_a[valid], 1e-4))
            - np.log(np.maximum(colors_b[valid], 1e-4)),
            axis=0,
        )
        sample_count = int(valid.sum())
        usable_edges.append((alignment, log_ratio, sample_count))
        degree[alignment.frame_a] += alignment.confidence * sample_count
        degree[alignment.frame_b] += alignment.confidence * sample_count
    if not usable_edges:
        return transformed_clouds, {
            "method": "verified-overlap RGB gain graph",
            "applied": False,
            "reason": "No accepted pair retained enough non-clipped RGB inliers.",
        }

    anchor = max(nodes, key=lambda node: (degree[node], -node))
    movable = [node for node in nodes if node != anchor]
    offsets = {node: index for index, node in enumerate(movable)}
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    pair_reports: list[dict[str, Any]] = []
    total_samples = 0
    for alignment, log_ratio, sample_count in usable_edges:
        row = np.zeros(len(movable), dtype=np.float64)
        if alignment.frame_a != anchor:
            row[offsets[alignment.frame_a]] -= 1.0
        if alignment.frame_b != anchor:
            row[offsets[alignment.frame_b]] += 1.0
        weight = np.sqrt(
            max(0.05, alignment.confidence)
            * min(1.0, sample_count / 100.0)
        )
        rows.append(row * weight)
        targets.append(log_ratio * weight)
        total_samples += sample_count
        pair_reports.append(
            {
                "frameA": alignment.frame_a,
                "frameB": alignment.frame_b,
                "samples": sample_count,
                "medianLogRatioRgb": np.round(log_ratio, 6).tolist(),
            }
        )
    regularization = 0.35
    for index in range(len(movable)):
        row = np.zeros(len(movable), dtype=np.float64)
        row[index] = np.sqrt(regularization)
        rows.append(row)
        targets.append(np.zeros(3, dtype=np.float64))
    matrix = np.stack(rows)
    target = np.stack(targets)
    solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    strength = 0.65
    gains: dict[int, np.ndarray] = {
        anchor: np.ones(3, dtype=np.float64)
    }
    for index in movable:
        gains[index] = np.clip(
            np.exp(strength * solution[offsets[index]]),
            0.90,
            1.10,
        )

    pair_rows = matrix[: len(usable_edges)]
    pair_targets = target[: len(usable_edges)]
    residual_before = float(np.sqrt(np.mean(pair_targets**2)))
    residual_after = float(
        np.sqrt(
            np.mean(
                (
                    pair_rows
                    @ np.log(
                        np.stack([gains[index] for index in movable])
                    )
                    - pair_targets
                )
                ** 2
            )
        )
    )
    balanced: dict[int, GaussianData] = {}
    frame_reports: list[dict[str, Any]] = []
    for index, cloud in transformed_clouds.items():
        gain = gains[index]
        balanced[index] = GaussianData(
            positions=cloud.positions,
            scales=cloud.scales,
            rotations=cloud.rotations,
            colors=np.clip(
                cloud.colors.astype(np.float32) * gain[None, :],
                0.0,
                1.0,
            ),
            opacities=cloud.opacities,
            sh_coeffs=cloud.sh_coeffs,
            errors=cloud.errors,
        )
        frame_reports.append(
            {
                "frame": index,
                "appliedGainRgb": np.round(gain, 5).tolist(),
            }
        )
    return balanced, {
        "method": "verified-overlap RGB gain graph",
        "applied": True,
        "anchorFrame": anchor,
        "strength": strength,
        "gainBounds": [0.90, 1.10],
        "pairCount": len(usable_edges),
        "overlapSampleCount": total_samples,
        "overlapLogRmseBefore": round(residual_before, 7),
        "overlapLogRmseAfter": round(residual_after, 7),
        "pairs": pair_reports,
        "frames": frame_reports,
        "note": (
            "Only geometrically verified overlap pixels influence appearance; "
            "geometry, source resolution, and Gaussian count are unchanged."
        ),
    }


def _camera_pose_report(
    index: int,
    transform: np.ndarray,
    metric_verified: bool,
) -> dict[str, Any]:
    scale, rotation = _similarity_scale_rotation(transform)
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return {
        "frame": index,
        "position": np.round(transform[:3, 3], 7).tolist(),
        "rotationWxyz": np.round(
            np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]]),
            7,
        ).tolist(),
        "scale": round(scale, 7),
        "placement": (
            "measured_feature_and_sharp_metric"
            if metric_verified
            else "joint_camera_calibrated_by_metric_core"
        ),
    }


def _select_cloud(cloud: GaussianData, keep: np.ndarray) -> GaussianData:
    return GaussianData(
        positions=cloud.positions[keep],
        scales=cloud.scales[keep],
        rotations=cloud.rotations[keep],
        colors=cloud.colors[keep],
        opacities=cloud.opacities[keep],
        sh_coeffs=(
            cloud.sh_coeffs[keep] if cloud.sh_coeffs is not None else None
        ),
        errors=cloud.errors,
    )


def _transform_cloud(cloud: GaussianData, transform: np.ndarray) -> GaussianData:
    linear = transform[:3, :3]
    scale = float(np.cbrt(abs(np.linalg.det(linear))))
    rotation = linear / max(scale, 1e-8)
    positions = _apply_points(cloud.positions.astype(np.float64), transform).astype(
        np.float32
    )
    covariance_rotations = _wxyz_to_matrices(cloud.rotations)
    rotated = rotation[None].astype(np.float32) @ covariance_rotations
    return GaussianData(
        positions=positions,
        scales=(cloud.scales + np.log(max(scale, 1e-8))).astype(np.float32),
        rotations=_matrices_to_wxyz(rotated),
        colors=cloud.colors,
        opacities=cloud.opacities,
        sh_coeffs=cloud.sh_coeffs,
        errors=cloud.errors,
    )


def _concatenate(a: GaussianData, b: GaussianData) -> GaussianData:
    return GaussianData(
        positions=np.vstack([a.positions, b.positions]),
        scales=np.vstack([a.scales, b.scales]),
        rotations=np.vstack([a.rotations, b.rotations]),
        colors=np.vstack([a.colors, b.colors]),
        opacities=np.vstack([a.opacities, b.opacities]),
        sh_coeffs=(
            np.vstack([a.sh_coeffs, b.sh_coeffs])
            if a.sh_coeffs is not None and b.sh_coeffs is not None
            else None
        ),
        errors=a.errors + b.errors,
    )


def _alignment_report(item: Alignment) -> dict[str, Any]:
    return {
        "frameA": item.frame_a,
        "frameB": item.frame_b,
        "method": item.method,
        "featureMatches": item.feature_matches,
        "metricMatches": item.metric_matches,
        "inliers": item.inliers,
        "inlierRatio": round(item.inlier_ratio, 4),
        "rmseMeters": round(item.rmse_m, 5),
        "scale": round(item.scale, 5),
        "confidence": round(item.confidence, 4),
        "transformBToA": np.round(item.transform_b_to_a, 7).tolist(),
    }


def _preflight_report(item: PreflightAlignment) -> dict[str, Any]:
    return {
        "frameA": item.frame_a,
        "frameB": item.frame_b,
        "method": item.method,
        "featureMatches": item.feature_matches,
        "inliers": item.inliers,
        "inlierRatio": round(item.inlier_ratio, 4),
        "confidence": round(item.confidence, 4),
    }
