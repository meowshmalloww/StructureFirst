"""Local YOLO26 object detection for rendered Rescue View frames.

The official YOLO26 Nano end-to-end ONNX model returns NMS-free ``xyxy``
detections.  Frames stay on the local reconstruction worker.  These 2D boxes
are observational aids, never verified hazards or structural-risk estimates.

Ultralytics YOLO26 weights are AGPL-3.0 by default.  See ``models/README.md``
before distributing StructureFirst under a different license.
"""

from __future__ import annotations

import base64
import hashlib
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

MODEL_SIZE = 640
MODEL_PATH = Path(__file__).resolve().parent / "models" / "yolo26n.onnx"
MODEL_SHA256 = "2e947b787d9e787b93a16772a5f55b1d4d8c4d86f53146149c5d6a642442d6f7"
MAX_IMAGE_BYTES = 2 * 1024 * 1024

COCO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)

TACTICAL_TAGS = {
    "person": ("occupant", "critical"),
    "bed": ("possible concealment area", "attention"),
    "couch": ("possible concealment area", "attention"),
    "knife": ("sharp object", "critical"),
    "scissors": ("sharp object", "attention"),
    "oven": ("heat / gas appliance", "attention"),
    "toaster": ("heat appliance", "attention"),
    "microwave": ("electrical appliance", "attention"),
    "refrigerator": ("large appliance", "attention"),
}

_SESSION: ort.InferenceSession | None = None
_SESSION_LOCK = threading.Lock()


def detector_status() -> dict[str, Any]:
    return {
        "available": MODEL_PATH.is_file() and _model_hash_matches(),
        "model": "YOLO26 Nano",
        "model_sha256": MODEL_SHA256,
        "license": "AGPL-3.0",
        "output": "2D view-space observations",
    }


def detect_data_url(image_data_url: str, score_threshold: float = 0.34) -> dict[str, Any]:
    started = time.perf_counter()
    image = _decode_image(image_data_url)
    original_height, original_width = image.shape[:2]
    tensor, ratio, pad_x, pad_y = _preprocess(image)
    session = _session()
    raw = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
    if raw.ndim != 2 or raw.shape[1] != 6:
        raise RuntimeError(
            f"YOLO26 end-to-end output must have shape (N, 6), received {raw.shape}"
        )

    # The official end-to-end export is already confidence-sorted, NMS-free,
    # and encoded as x1, y1, x2, y2, confidence, class_id.
    detections = []
    for x1, y1, x2, y2, score, raw_class_id in raw:
        if float(score) < score_threshold:
            continue
        class_id = int(raw_class_id)
        if not 0 <= class_id < len(COCO_LABELS):
            continue
        left = min(float(original_width), max(0.0, (float(x1) - pad_x) / ratio))
        top = min(float(original_height), max(0.0, (float(y1) - pad_y) / ratio))
        right = min(float(original_width), max(0.0, (float(x2) - pad_x) / ratio))
        bottom = min(float(original_height), max(0.0, (float(y2) - pad_y) / ratio))
        width = max(0.0, right - left)
        height = max(0.0, bottom - top)
        if width < 1 or height < 1:
            continue
        label = COCO_LABELS[class_id]
        tactical_label, priority = TACTICAL_TAGS.get(label, ("observed object", "standard"))
        detections.append(
            {
                "label": label,
                "score": round(float(score), 4),
                "x": round(left / original_width, 6),
                "y": round(top / original_height, 6),
                "width": round(width / original_width, 6),
                "height": round(height / original_height, 6),
                "tacticalLabel": tactical_label,
                "priority": priority,
            }
        )
        if len(detections) >= 30:
            break

    return {
        "detections": detections,
        "model": "YOLO26 Nano",
        "provider": session.get_providers()[0],
        "inferenceMs": round((time.perf_counter() - started) * 1000, 1),
        "imageWidth": original_width,
        "imageHeight": original_height,
    }


def _session() -> ort.InferenceSession:
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            if not MODEL_PATH.is_file():
                raise RuntimeError("YOLO26 Nano model is not installed")
            if not _model_hash_matches():
                raise RuntimeError("YOLO26 Nano model failed its SHA-256 integrity check")
            # ONNX Runtime can reuse the CUDA/cuDNN libraries already shipped
            # with the CUDA-enabled PyTorch installation on the workstation.
            # On CPU-only hosts preload_dlls is harmless and CPU remains the
            # final provider fallback.
            preload_dlls = getattr(ort, "preload_dlls", None)
            if preload_dlls is not None:
                preload_dlls()
            available = ort.get_available_providers()
            preferred = [
                provider
                for provider in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
                if provider in available
            ]
            _SESSION = ort.InferenceSession(str(MODEL_PATH), providers=preferred)
        return _SESSION


def _decode_image(image_data_url: str) -> np.ndarray:
    encoded = image_data_url.split(",", 1)[1] if "," in image_data_url else image_data_url
    try:
        payload = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("frame is not valid base64") from exc
    if not payload or len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("frame must be between 1 byte and 2 MB")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("frame is not a supported image")
    return image


def _preprocess(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    ratio = min(MODEL_SIZE / height, MODEL_SIZE / width)
    resized_width = round(width * ratio)
    resized_height = round(height * ratio)
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    padded = np.full((MODEL_SIZE, MODEL_SIZE, 3), 114, dtype=np.uint8)
    pad_x = round((MODEL_SIZE - resized_width) / 2 - 0.1)
    pad_y = round((MODEL_SIZE - resized_height) / 2 - 0.1)
    padded[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)[None]
    tensor /= 255.0
    return tensor, ratio, pad_x, pad_y


def _model_hash_matches() -> bool:
    if not MODEL_PATH.is_file():
        return False
    digest = hashlib.sha256()
    with MODEL_PATH.open("rb") as model_file:
        for block in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == MODEL_SHA256
