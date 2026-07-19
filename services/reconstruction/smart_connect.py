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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

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
from sharp_wrapper import _matrices_to_wxyz, _wxyz_to_matrices, reconstruct_sharp

LOGGER = logging.getLogger("structurefirst.smart_connect")


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
    selected_set = set(selected_indices)
    core_set = set(core_indices)
    frame_reports: list[dict[str, Any]] = [
        {
            "index": frame.index,
            "file": frame.path.name,
            "featureCount": len(frame.keypoints),
            "selectedForSharp": frame.index in selected_set,
            "classification": (
                "geometric_core"
                if frame.index in core_set
                else "same_scene_candidate"
                if frame.index in selected_set
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
            index for index in range(len(image_paths)) if index not in selected_set
        ],
        "selectedFrames": selected_indices,
        "sceneRecognition": scene_report,
    }
    if len(selected_indices) < 2:
        report = {
            "schemaVersion": 1,
            "method": "SIFT and LoFTR preflight before LucidFrame SHARP metric registration",
            "status": "failed",
            "frameCount": len(image_paths),
            "connectedFrameCount": 1,
            "anchorFrame": 0,
            "frames": frame_reports,
            "preflight": preflight_report,
            "acceptedPairs": [],
            "rejectedPairs": [],
            "treePairs": [],
            "connectedFrames": [0],
            "disconnectedFrames": list(range(1, len(image_paths))),
            "confidenceScore": 0.0,
        }
        raise RegistrationError(
            "No two photographs passed visual overlap preflight. Capture adjacent views with 60–80% overlap.",
            report,
        )

    frames: list[FrameData] = []
    for index in selected_indices:
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

    transforms, tree, anchor_frame = _connection_tree(len(image_paths), alignments)
    transforms, pose_optimization = _refine_pose_graph(
        transforms,
        alignments,
        anchor_frame,
    )
    connected_indices = sorted(transforms)
    for frame in frame_reports:
        index = int(frame["index"])
        if index in transforms:
            frame["classification"] = "registered"
        elif index in selected_set:
            frame["classification"] = "same_scene_unregistered"
            frame["registrationReason"] = (
                "Same-room evidence was found, but no safe cross-view transform was verified"
            )
    frames_by_index = {frame.source_index: frame for frame in frames}
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "method": "SIFT/LoFTR correspondences + SHARP metric similarity",
        "frameCount": len(image_paths),
        "connectedFrameCount": len(connected_indices),
        "anchorFrame": anchor_frame,
        "frames": frame_reports,
        "preflight": preflight_report,
        "acceptedPairs": [_alignment_report(item) for item in alignments],
        "rejectedPairs": rejected,
        "treePairs": [_alignment_report(item) for item in tree],
        "poseOptimization": pose_optimization,
        "connectedFrames": connected_indices,
        "disconnectedFrames": [
            index for index in range(len(image_paths)) if index not in transforms
        ],
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

    transformed_clouds, cleanup_report = _cross_view_cleanup(
        transformed_clouds,
        frames_by_index,
        transforms,
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
    report["artifactCleanup"] = cleanup_report
    return merged or frames_by_index[connected_indices[0]].cloud, report


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
                if (
                    learned.raw_matches >= 20
                    and learned.inliers >= 18
                    and learned.inlier_ratio >= 0.45
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
                        f"{learned.raw_matches} geometric inliers"
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
    _, homography_mask = cv2.findHomography(
        points_b,
        points_a,
        cv2.RANSAC,
        5.0,
        maxIters=3000,
        confidence=0.995,
    )
    _, fundamental_mask = cv2.findFundamentalMat(
        points_a.reshape(-1, 2),
        points_b.reshape(-1, 2),
        cv2.USAC_MAGSAC,
        1.5,
        0.999,
        10_000,
    )
    candidates = [
        ("sift_homography_ransac", homography_mask),
        ("sift_fundamental_magsac", fundamental_mask),
    ]
    candidates = [(method, mask) for method, mask in candidates if mask is not None]
    if not candidates:
        return None, "RANSAC could not estimate a stable image transform"
    method, mask = max(candidates, key=lambda item: int(item[1].ravel().sum()))
    inliers = int(mask.ravel().sum())
    ratio = inliers / len(symmetric)
    if inliers < 15 or ratio < 0.22:
        return None, f"Visual geometry kept {inliers}/{len(symmetric)} inliers"
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
                points_a.reshape(-1, 2)[mask.ravel().astype(bool)]
                * np.array(
                    [
                        a.source_size[0] / max(a.image.shape[1], 1),
                        a.source_size[1] / max(a.image.shape[0], 1),
                    ],
                    dtype=np.float32,
                )
            ),
            points_b=(
                points_b.reshape(-1, 2)[mask.ravel().astype(bool)]
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
    for pixel_a, pixel_b in zip(pixel_points_a, pixel_points_b, strict=True):
        point_a = _nearby_xyz(a.xyz_map, pixel_a)
        point_b = _nearby_xyz(b.xyz_map, pixel_b)
        if point_a is not None and point_b is not None:
            points_a.append(point_a)
            points_b.append(point_b)
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
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Jointly refine every connected camera against all accepted pair edges.

    The maximum-confidence tree supplies a stable initialization. Non-tree
    edges then close loops and distribute residual error instead of letting a
    sequence of pairwise transforms bend the room. The anchor stays fixed so
    the exported coordinate frame remains deterministic.
    """
    nodes = sorted(initial)
    edges = [
        edge
        for edge in alignments
        if edge.frame_a in initial and edge.frame_b in initial
    ]
    movable = [node for node in nodes if node != anchor]
    if not movable or len(edges) < len(nodes):
        return initial, {
            "method": "robust Sim(3) pose graph",
            "optimized": False,
            "reason": "No loop-closing edge was available",
            "nodeCount": len(nodes),
            "edgeCount": len(edges),
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

    def residuals(parameters: np.ndarray) -> np.ndarray:
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

    before = residuals(initial_vector)
    result = least_squares(
        residuals,
        initial_vector,
        loss="soft_l1",
        f_scale=0.5,
        max_nfev=250,
    )
    after = residuals(result.x)
    before_rmse = float(np.sqrt(np.mean(before**2)))
    after_rmse = float(np.sqrt(np.mean(after**2)))
    accepted = bool(
        result.success
        and np.isfinite(result.x).all()
        and after_rmse <= before_rmse + 1e-8
    )
    return (unpack(result.x) if accepted else initial), {
        "method": "robust Sim(3) pose graph",
        "optimized": accepted,
        "nodeCount": len(nodes),
        "edgeCount": len(edges),
        "loopEdgeCount": max(0, len(edges) - (len(nodes) - 1)),
        "beforeRmse": round(before_rmse, 7),
        "afterRmse": round(after_rmse if accepted else before_rmse, 7),
        "iterations": int(result.nfev),
    }


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
            x = np.rint(
                target_positions[:, 0] / np.maximum(z, 1e-8) * target.focal_px
                + width / 2.0
            ).astype(np.int32)
            y = np.rint(
                target_positions[:, 1] / np.maximum(z, 1e-8) * target.focal_px
                + height / 2.0
            ).astype(np.int32)
            inside = (
                (z > 0.01)
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
        frame_reports.append(
            {
                "frame": source_index,
                "before": cloud.count,
                "after": int(keep.sum()),
                "removed": int(remove.sum()),
                "multiViewObserved": int(
                    (observations >= required_observations).sum()
                ),
            }
        )
    return cleaned, {
        "method": "conservative cross-view depth-consistency pruning",
        "requiredObservations": required_observations,
        "removed": int(
            sum(report["removed"] for report in frame_reports)
        ),
        "frames": frame_reports,
        "note": "Unseen and occluded Gaussians are retained; only repeated front-surface contradictions are pruned.",
    }


def _select_cloud(cloud: GaussianData, keep: np.ndarray) -> GaussianData:
    return GaussianData(
        positions=cloud.positions[keep],
        scales=cloud.scales[keep],
        rotations=cloud.rotations[keep],
        colors=cloud.colors[keep],
        opacities=cloud.opacities[keep],
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
        errors=cloud.errors,
    )


def _concatenate(a: GaussianData, b: GaussianData) -> GaussianData:
    return GaussianData(
        positions=np.vstack([a.positions, b.positions]),
        scales=np.vstack([a.scales, b.scales]),
        rotations=np.vstack([a.rotations, b.rotations]),
        colors=np.vstack([a.colors, b.colors]),
        opacities=np.vstack([a.opacities, b.opacities]),
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
