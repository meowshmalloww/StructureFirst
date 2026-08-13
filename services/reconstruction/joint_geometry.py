"""Joint camera geometry for unordered responder photographs.

SHARP remains the source of high-resolution, metric Gaussian fields.  This
module supplies the part a monocular predictor cannot: one camera solution
estimated from all views at once.  VGGT runs on EXIF-normalized copies, predicts
shared camera poses and depth confidence, and is then metric-calibrated against
the pairwise SHARP correspondences already accepted by ``smart_connect``.

The joint estimate is deliberately evidence gated.  It may extend a verified
capture session, but it is never allowed to bootstrap a room without at least
one measured SHARP/feature constraint.
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class MetricPoseConstraint:
    frame_a: int
    frame_b: int
    transform_b_to_a: np.ndarray
    confidence: float


@dataclass
class JointCameraEstimate:
    frame_indices: list[int]
    camera_to_world: dict[int, np.ndarray]
    intrinsics: dict[int, np.ndarray]
    depth_confidence: dict[int, float]
    report: dict[str, Any]


@dataclass
class CalibratedJointGeometry:
    transforms_to_anchor: dict[int, np.ndarray]
    confidence_by_frame: dict[int, float]
    report: dict[str, Any]


def joint_geometry_available() -> tuple[bool, str | None]:
    if os.getenv("STRUCTUREFIRST_JOINT_GEOMETRY", "vggt").casefold() == "off":
        return False, "Joint geometry is disabled by STRUCTUREFIRST_JOINT_GEOMETRY."
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "VGGT joint geometry requires CUDA."
        from vggt.models.vggt import VGGT  # noqa: F401
        from vggt.utils.load_fn import load_and_preprocess_images  # noqa: F401
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: F401
    except Exception as exc:
        return False, f"VGGT is unavailable: {exc}"
    return True, None


def estimate_joint_cameras(
    image_paths: list[Path],
    frame_indices: list[int],
    output_dir: Path,
) -> JointCameraEstimate | None:
    """Estimate all camera poses jointly on the NVIDIA CUDA device.

    The source bytes are never rewritten.  Temporary PNG copies apply EXIF
    orientation because geometry models otherwise see portrait phone captures
    rotated by 90 degrees, which changes predicted camera axes.
    """

    available, reason = joint_geometry_available()
    if not available:
        return None
    if len(image_paths) < 2 or len(image_paths) != len(frame_indices):
        return None

    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    started = time.perf_counter()
    joint_dir = output_dir / "joint_geometry"
    normalized_dir = joint_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized_paths: list[Path] = []
    normalized_sizes: list[list[int]] = []
    for order, path in enumerate(image_paths):
        destination = normalized_dir / f"{order:02d}.png"
        with Image.open(path) as opened:
            oriented = ImageOps.exif_transpose(opened).convert("RGB")
            normalized_sizes.append([oriented.width, oriented.height])
            oriented.save(destination, format="PNG", compress_level=2)
        normalized_paths.append(destination)

    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model: Any | None = None
    images: Any | None = None
    aggregated: Any | None = None
    depth_conf: Any | None = None
    try:
        local_only = os.getenv("STRUCTUREFIRST_VGGT_LOCAL_ONLY", "false").casefold() not in {
            "0",
            "false",
            "no",
        }
        model = VGGT.from_pretrained(
            "facebook/VGGT-1B",
            local_files_only=local_only,
        ).eval().to(device)
        images = load_and_preprocess_images(
            [str(path) for path in normalized_paths]
        ).to(device)
        capability = torch.cuda.get_device_capability(device)
        dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
            aggregated, patch_start_index = model.aggregator(images[None])
            pose_encoding = model.camera_head(aggregated)[-1]
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                pose_encoding,
                images.shape[-2:],
            )
            _, depth_conf = model.depth_head(
                aggregated,
                images[None],
                patch_start_index,
            )

        extrinsics_np = extrinsics[0].float().cpu().numpy()
        intrinsics_np = intrinsics[0].float().cpu().numpy()
        frame_confidence = depth_conf[0].float()
        if frame_confidence.ndim < 2:
            raise RuntimeError(
                f"VGGT returned an unexpected confidence shape: {tuple(depth_conf.shape)}"
            )
        depth_conf_np = (
            frame_confidence.mean(
                dim=tuple(range(1, frame_confidence.ndim))
            )
            .cpu()
            .numpy()
        )
        camera_to_world: dict[int, np.ndarray] = {}
        intrinsics_by_frame: dict[int, np.ndarray] = {}
        confidence_by_frame: dict[int, float] = {}
        for order, frame_index in enumerate(frame_indices):
            world_to_camera = np.eye(4, dtype=np.float64)
            world_to_camera[:3, :4] = extrinsics_np[order]
            camera_to_world[frame_index] = np.linalg.inv(world_to_camera)
            intrinsics_by_frame[frame_index] = intrinsics_np[order].astype(
                np.float64
            )
            confidence_by_frame[frame_index] = float(depth_conf_np[order])

        elapsed = time.perf_counter() - started
        report = {
            "available": True,
            "accepted": False,
            "method": "VGGT shared camera/depth inference + measured SHARP metric calibration",
            "model": "facebook/VGGT-1B",
            "modelVersion": "0.0.1",
            "device": torch.cuda.get_device_name(device),
            "cuda": str(torch.version.cuda),
            "dtype": str(dtype).replace("torch.", ""),
            "frameIndices": frame_indices,
            "normalizedSourceSizes": normalized_sizes,
            "inferenceShape": [int(value) for value in images.shape],
            "rawDepthConfidence": {
                str(frame): round(confidence_by_frame[frame], 5)
                for frame in frame_indices
            },
            "elapsedSeconds": round(elapsed, 3),
            "peakVramMb": round(
                float(torch.cuda.max_memory_allocated() / 1024**2), 1
            ),
            "note": (
                "Camera geometry was inferred jointly from EXIF-normalized views. "
                "It is not accepted until it agrees with measured image/SHARP geometry."
            ),
        }
        return JointCameraEstimate(
            frame_indices=frame_indices,
            camera_to_world=camera_to_world,
            intrinsics=intrinsics_by_frame,
            depth_confidence=confidence_by_frame,
            report=report,
        )
    finally:
        del depth_conf, aggregated, images, model
        gc.collect()
        torch.cuda.empty_cache()


def calibrate_joint_cameras(
    estimate: JointCameraEstimate | None,
    constraints: Iterable[MetricPoseConstraint],
    anchor_frame: int,
    *,
    maximum_rotation_disagreement_deg: float = 18.0,
) -> CalibratedJointGeometry | None:
    """Calibrate VGGT's arbitrary translation scale with verified metric edges."""

    if estimate is None or anchor_frame not in estimate.camera_to_world:
        return None
    usable: list[dict[str, float | int]] = []
    for item in constraints:
        if (
            item.frame_a not in estimate.camera_to_world
            or item.frame_b not in estimate.camera_to_world
        ):
            continue
        predicted = (
            np.linalg.inv(estimate.camera_to_world[item.frame_a])
            @ estimate.camera_to_world[item.frame_b]
        )
        observed = item.transform_b_to_a
        predicted_rotation = _proper_rotation(predicted[:3, :3])
        observed_rotation = _proper_rotation(observed[:3, :3])
        disagreement = float(
            np.degrees(
                Rotation.from_matrix(
                    observed_rotation @ predicted_rotation.T
                ).magnitude()
            )
        )
        predicted_baseline = float(np.linalg.norm(predicted[:3, 3]))
        observed_baseline = float(np.linalg.norm(observed[:3, 3]))
        if (
            disagreement <= maximum_rotation_disagreement_deg
            and predicted_baseline > 1e-6
            and observed_baseline > 1e-6
        ):
            usable.append(
                {
                    "frameA": item.frame_a,
                    "frameB": item.frame_b,
                    "rotationDisagreementDeg": disagreement,
                    "metricBaseline": observed_baseline,
                    "jointBaseline": predicted_baseline,
                    "scaleFactor": observed_baseline / predicted_baseline,
                    "confidence": item.confidence,
                }
            )
    if not usable:
        estimate.report.update(
            {
                "accepted": False,
                "rejectionReason": (
                    "No joint camera pair agreed with verified SHARP geometry "
                    f"within {maximum_rotation_disagreement_deg:.1f} degrees."
                ),
                "validatedPairs": [],
            }
        )
        return None

    factors = np.asarray([float(item["scaleFactor"]) for item in usable])
    weights = np.asarray(
        [max(0.05, float(item["confidence"])) for item in usable]
    )
    metric_scale = _weighted_median(factors, weights)
    anchor_to_world = estimate.camera_to_world[anchor_frame]
    transforms: dict[int, np.ndarray] = {}
    raw_scores = np.asarray(
        [
            max(0.0, estimate.depth_confidence.get(frame, 0.0))
            for frame in estimate.frame_indices
        ],
        dtype=np.float64,
    )
    score_median = max(float(np.median(raw_scores)), 1e-6)
    confidence_by_frame: dict[int, float] = {}
    for frame in estimate.frame_indices:
        relative = (
            np.linalg.inv(anchor_to_world) @ estimate.camera_to_world[frame]
        )
        transform = relative.astype(np.float64)
        transform[:3, 3] *= metric_scale
        transforms[frame] = transform
        normalized_depth_confidence = estimate.depth_confidence.get(
            frame, 0.0
        ) / score_median
        confidence_by_frame[frame] = float(
            np.clip(0.55 + 0.12 * normalized_depth_confidence, 0.58, 0.78)
        )
    confidence_by_frame[anchor_frame] = 1.0

    rotation_disagreements = np.asarray(
        [float(item["rotationDisagreementDeg"]) for item in usable]
    )
    estimate.report.update(
        {
            "accepted": True,
            "anchorFrame": anchor_frame,
            "metricScale": round(float(metric_scale), 7),
            "validatedPairCount": len(usable),
            "rotationAgreementMedianDeg": round(
                float(np.median(rotation_disagreements)), 4
            ),
            "rotationAgreementMaxDeg": round(
                float(np.max(rotation_disagreements)), 4
            ),
            "validatedPairs": [
                {
                    **item,
                    "rotationDisagreementDeg": round(
                        float(item["rotationDisagreementDeg"]), 4
                    ),
                    "metricBaseline": round(
                        float(item["metricBaseline"]), 6
                    ),
                    "jointBaseline": round(
                        float(item["jointBaseline"]), 6
                    ),
                    "scaleFactor": round(float(item["scaleFactor"]), 7),
                    "confidence": round(float(item["confidence"]), 4),
                }
                for item in usable
            ],
            "cameraPoses": [
                {
                    "frame": frame,
                    "position": np.round(
                        transforms[frame][:3, 3], 7
                    ).tolist(),
                    "rotationWxyz": np.round(
                        _matrix_to_wxyz(
                            _proper_rotation(transforms[frame][:3, :3])
                        ),
                        7,
                    ).tolist(),
                    "confidence": round(confidence_by_frame[frame], 4),
                    "source": (
                        "anchor"
                        if frame == anchor_frame
                        else "vggt_joint_camera"
                    ),
                }
                for frame in estimate.frame_indices
            ],
        }
    )
    return CalibratedJointGeometry(
        transforms_to_anchor=transforms,
        confidence_by_frame=confidence_by_frame,
        report=estimate.report,
    )


def _proper_rotation(linear: np.ndarray) -> np.ndarray:
    scale = float(np.cbrt(abs(np.linalg.det(linear))))
    normalized = linear / max(scale, 1e-8)
    u, _, vt = np.linalg.svd(normalized)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def _matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    threshold = float(sorted_weights.sum()) * 0.5
    index = int(np.searchsorted(np.cumsum(sorted_weights), threshold))
    return float(sorted_values[min(index, len(sorted_values) - 1)])
