"""Learned scene evidence used before LucidFrame reconstruction.

These signals may nominate photographs for geometric registration, but they
never place or merge a frame by themselves.  Spatial acceptance remains gated
by verified two-view correspondences and metric geometry in ``smart_connect``.
"""

from __future__ import annotations

import copy
import gc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

DINO_MODEL = "vit_small_patch14_dinov2"
LOFTR_REPOSITORY = "kornia/loftr"
LOFTR_CHECKPOINT = "loftr_indoor_ds_new.ckpt"


@dataclass(frozen=True)
class CaptureMetadata:
    device: str | None
    captured_at: datetime | None


@dataclass
class LearnedMatchResult:
    points_a: np.ndarray
    points_b: np.ndarray
    raw_matches: int
    inliers: int
    inlier_ratio: float
    mean_confidence: float


def capture_metadata(path: Path) -> CaptureMetadata:
    """Read general EXIF capture continuity without relying on filenames."""
    with Image.open(path) as opened:
        exif = opened.getexif()
        make = str(exif.get(271, "")).strip()
        model = str(exif.get(272, "")).strip()
        timestamp = str(
            exif.get(36867) or exif.get(36868) or exif.get(306) or ""
        ).strip()
    device = " ".join(part for part in (make, model) if part).casefold() or None
    captured_at: datetime | None = None
    for pattern in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            captured_at = datetime.strptime(timestamp, pattern)
            break
        except ValueError:
            continue
    return CaptureMetadata(device=device, captured_at=captured_at)


def scene_embeddings(images: list[np.ndarray]) -> np.ndarray:
    """Return normalized DINOv2 descriptors for visual-place candidacy."""
    import timm
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        DINO_MODEL,
        pretrained=True,
        num_classes=0,
    ).eval().to(device)
    tensors: list[torch.Tensor] = []
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    for image in images:
        height, width = image.shape[:2]
        scale = 518.0 / max(width, height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
        canvas = np.full((518, 518, 3), (124, 116, 104), dtype=np.uint8)
        offset_x = (518 - resized_width) // 2
        offset_y = (518 - resized_height) // 2
        canvas[
            offset_y : offset_y + resized_height,
            offset_x : offset_x + resized_width,
        ] = resized
        tensor = torch.from_numpy(canvas.copy()).permute(2, 0, 1).float() / 255.0
        tensors.append((tensor - mean) / std)
    try:
        with torch.inference_mode():
            tokens = model.forward_features(torch.stack(tensors).to(device))
            descriptors = torch.nn.functional.normalize(tokens[:, 0], dim=-1)
        return descriptors.float().cpu().numpy()
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_indoor_matcher() -> tuple[Any, Any, str]:
    """Load the official indoor LoFTR weights over Hugging Face HTTPS."""
    import kornia.feature as feature
    import torch
    from huggingface_hub import hf_hub_download
    from kornia.feature.loftr.loftr import default_cfg

    checkpoint_path = hf_hub_download(
        repo_id=LOFTR_REPOSITORY,
        filename=LOFTR_CHECKPOINT,
    )
    config = copy.deepcopy(default_cfg)
    config["coarse"]["temp_bug_fix"] = True
    matcher = feature.LoFTR(pretrained=None, config=config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    matcher.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return matcher.eval().to(device), device, checkpoint_path


def match_indoor_pair(
    matcher: Any,
    device: Any,
    image_a: np.ndarray,
    image_b: np.ndarray,
    source_size_a: tuple[int, int],
    source_size_b: tuple[int, int],
) -> LearnedMatchResult:
    """Match a low-texture indoor pair and verify epipolar geometry."""
    import torch

    target_width, target_height = 640, 480

    def prepare(image: np.ndarray) -> Any:
        resized = cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        return torch.from_numpy(gray.copy())[None, None].float().to(device) / 255.0

    with torch.inference_mode():
        result = matcher(
            {"image0": prepare(image_a), "image1": prepare(image_b)}
        )
    points_a_small = result["keypoints0"].float().cpu().numpy().astype(np.float32)
    points_b_small = result["keypoints1"].float().cpu().numpy().astype(np.float32)
    confidences = result["confidence"].float().cpu().numpy()
    raw_matches = len(points_a_small)
    if raw_matches < 8:
        return LearnedMatchResult(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            raw_matches,
            0,
            0.0,
            float(confidences.mean()) if raw_matches else 0.0,
        )
    _, mask = cv2.findFundamentalMat(
        points_a_small,
        points_b_small,
        cv2.USAC_MAGSAC,
        1.5,
        0.999,
        10_000,
    )
    if mask is None:
        inlier_mask = np.zeros(raw_matches, dtype=bool)
    else:
        inlier_mask = mask.ravel().astype(bool)
    points_a = points_a_small[inlier_mask].copy()
    points_b = points_b_small[inlier_mask].copy()
    points_a[:, 0] *= source_size_a[0] / target_width
    points_a[:, 1] *= source_size_a[1] / target_height
    points_b[:, 0] *= source_size_b[0] / target_width
    points_b[:, 1] *= source_size_b[1] / target_height
    inliers = len(points_a)
    return LearnedMatchResult(
        points_a=points_a,
        points_b=points_b,
        raw_matches=raw_matches,
        inliers=inliers,
        inlier_ratio=inliers / raw_matches,
        mean_confidence=float(confidences.mean()) if raw_matches else 0.0,
    )


def release_matcher(matcher: Any) -> None:
    import torch

    del matcher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def recognize_scene_frames(
    core_indices: list[int],
    embeddings: np.ndarray | None,
    captures: list[CaptureMetadata],
) -> tuple[list[int], list[dict[str, Any]], float | None]:
    """Nominate same-scene frames; geometry must still register each nominee."""
    recognized = set(core_indices)
    reports: list[dict[str, Any]] = []
    if not core_indices:
        return [], reports, None
    similarity = embeddings @ embeddings.T if embeddings is not None else None
    internal_values = (
        [
            float(similarity[a, b])
            for offset, a in enumerate(core_indices)
            for b in core_indices[offset + 1 :]
        ]
        if similarity is not None
        else []
    )
    internal_affinity = float(np.median(internal_values)) if internal_values else None
    session_threshold = (
        max(0.12, 0.65 * internal_affinity)
        if internal_affinity is not None
        else None
    )
    visual_threshold = (
        max(0.22, 0.85 * internal_affinity)
        if internal_affinity is not None
        else None
    )
    for index, capture in enumerate(captures):
        if index in recognized:
            continue
        affinity = (
            float(np.mean(similarity[index, core_indices]))
            if similarity is not None
            else None
        )
        gaps = [
            abs((capture.captured_at - captures[core].captured_at).total_seconds())
            for core in core_indices
            if capture.device
            and capture.device == captures[core].device
            and capture.captured_at is not None
            and captures[core].captured_at is not None
        ]
        nearest_capture_seconds = min(gaps) if gaps else None
        session_match = (
            nearest_capture_seconds is not None and nearest_capture_seconds <= 120.0
        )
        accepted_by_session = bool(
            session_match
            and (
                affinity is None
                or session_threshold is None
                or affinity >= session_threshold
            )
        )
        accepted_by_visual = bool(
            affinity is not None
            and visual_threshold is not None
            and affinity >= visual_threshold
        )
        accepted = accepted_by_session or accepted_by_visual
        if accepted:
            recognized.add(index)
        reports.append(
            {
                "frame": index,
                "accepted": accepted,
                "captureSessionMatch": session_match,
                "nearestCaptureSeconds": nearest_capture_seconds,
                "coreAffinity": round(affinity, 4) if affinity is not None else None,
                "reason": (
                    "capture continuity and visual-place affinity"
                    if accepted_by_session and affinity is not None
                    else "capture continuity"
                    if accepted_by_session
                    else "strong visual-place affinity; geometry still required"
                    if accepted_by_visual
                    else "not enough same-scene evidence"
                ),
            }
        )
    return sorted(recognized), reports, session_threshold
