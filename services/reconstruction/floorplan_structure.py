"""Vectorize architectural plan images into a lightweight structural model.

This is intentionally separate from Gaussian reconstruction. A plan drawing is
top-down structural evidence, not a camera view, so sending it through SHARP or
feature-matching it against photographs would create false geometry.
"""

from __future__ import annotations

import json
import math
import re
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


_OCR_ENGINE: Any | None = None


def _read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode floorplan image: {path.name}")
    return image


def _ocr_entries(path: Path) -> list[dict[str, Any]]:
    """Read printed plan text and retain its position on the source sheet."""

    global _OCR_ENGINE
    try:
        if _OCR_ENGINE is None:
            from rapidocr import RapidOCR

            _OCR_ENGINE = RapidOCR()
        result = _OCR_ENGINE(str(path))
        entries = []
        boxes = result.boxes if result.boxes is not None else ()
        texts = result.txts if result.txts is not None else ()
        scores = result.scores if result.scores is not None else ()
        for box, value, score in zip(boxes, texts, scores, strict=True):
            text = str(value).strip()
            if not text or float(score) < 0.78:
                continue
            points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
            entries.append(
                {
                    "text": text,
                    "score": float(score),
                    "source": "plan_ocr",
                    "center": [float(points[:, 0].mean()), float(points[:, 1].mean())],
                    "box": [
                        float(points[:, 0].min()),
                        float(points[:, 1].min()),
                        float(points[:, 0].max()),
                        float(points[:, 1].max()),
                    ],
                }
            )
    except Exception:
        return []
    return entries


def _ocr_plan(path: Path) -> tuple[int | None, list[dict[str, Any]]]:
    """Read printed floor/room labels and retain their image positions."""

    entries = _ocr_entries(path)
    floor_number: int | None = None
    for entry in entries:
        text = entry["text"]
        normalized = re.sub(r"\s+", " ", text.upper())
        match = re.search(r"\b(\d{1,2})(?:ST|ND|RD|R|TH)?\s+FLOOR\s+PLAN\b", normalized)
        if match:
            floor_number = int(match.group(1))
            break
    ignored = re.compile(
        r"(?:FLOOR\s+PLAN|\d+\s*[\"'°]|ROOF(?:\s+BELOW)?$|^[\d.\-=/ ]+$)",
        re.IGNORECASE,
    )
    labels = [
        entry
        for entry in entries
        if not ignored.search(entry["text"])
        and len(entry["text"]) <= 48
        and any(character.isalpha() for character in entry["text"])
    ]
    return floor_number, labels[:60]


_LEVEL_ANCHORS: tuple[tuple[re.Pattern[str], int, str], ...] = (
    (re.compile(r"\b(?:BASEMENT|LOWER)\s+(?:LEVEL|FLOOR)\b", re.I), 0, "Lower level"),
    (re.compile(r"\b(?:MAIN|GROUND)\s+(?:LEVEL|FLOOR)\b", re.I), 1, "Main level"),
    (re.compile(r"\b(?:UPPER|SECOND|2ND)\s+(?:LEVEL|FLOOR)\b", re.I), 2, "Upper level"),
    (re.compile(r"\b(?:THIRD|3RD)\s+(?:LEVEL|FLOOR)\b", re.I), 3, "Third level"),
    (re.compile(r"\b1ST\s+FLOOR(?:\s+PLAN)?\b", re.I), 1, "First floor"),
    (re.compile(r"\b2ND\s+FLOOR(?:\s+PLAN)?\b", re.I), 2, "Second floor"),
    (re.compile(r"\b3RD\s+FLOOR(?:\s+PLAN)?\b", re.I), 3, "Third floor"),
)


def _level_anchors(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for entry in entries:
        normalized = re.sub(r"\s+", " ", entry["text"]).strip()
        for pattern, floor_number, label in _LEVEL_ANCHORS:
            if pattern.search(normalized):
                anchors.append(
                    {
                        **entry,
                        "floorNumber": floor_number,
                        "floorLabel": label,
                    }
                )
                break
    return anchors


def _is_site_reference(entries: Sequence[dict[str, Any]]) -> bool:
    text = " ".join(entry["text"] for entry in entries)
    return bool(re.search(r"\bSITE\s+PLAN\b", text, re.I)) and not _level_anchors(entries)


def _plan_components(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find dense, connected plan drawings while rejecting page borders/text blocks."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
    ink = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    height, width = gray.shape
    page_area = float(width * height)
    candidates: list[tuple[int, int, int, int, int]] = []
    for left, top, component_width, component_height, ink_area in stats[1:]:
        box_area = float(component_width * component_height)
        density = float(ink_area) / max(1.0, box_area)
        if box_area < page_area * 0.025:
            continue
        if component_width < width * 0.12 or component_height < height * 0.15:
            continue
        if density < 0.08:
            continue
        if component_width > width * 0.85 and component_height > height * 0.85:
            continue
        candidates.append(
            (
                int(left),
                int(top),
                int(left + component_width),
                int(top + component_height),
                int(ink_area),
            )
        )
    candidates.sort(key=lambda item: item[4], reverse=True)
    return [item[:4] for item in candidates[:8]]


def _distance_to_box(point: Sequence[float], box: Sequence[int]) -> float:
    x, y = point
    left, top, right, bottom = box
    dx = max(left - x, 0.0, x - right)
    dy = max(top - y, 0.0, y - bottom)
    return math.hypot(dx, dy)


def _crop_box(
    box: Sequence[int], width: int, height: int
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    padding = max(8, round(min(width, height) * 0.012))
    return (
        max(0, int(left) - padding),
        max(0, int(top) - padding),
        min(width, int(right) + padding),
        min(height, int(bottom) + padding),
    )


def _sheet_panels(
    image: np.ndarray,
    entries: Sequence[dict[str, Any]],
    fallback_floor_number: int,
) -> list[dict[str, Any]]:
    """Split a sheet containing several independently drawn floor levels."""

    height, width = image.shape[:2]
    anchors = _level_anchors(entries)
    components = _plan_components(image)
    if len(anchors) < 2 or len(components) < 2:
        detected_number = anchors[0]["floorNumber"] if anchors else fallback_floor_number
        detected_label = anchors[0]["floorLabel"] if anchors else f"Floor {detected_number}"
        return [
            {
                "floorNumber": int(detected_number),
                "floorLabel": detected_label,
                "floorNumberSource": "local_ocr" if anchors else "provided_order",
                "box": (0, 0, width, height),
            }
        ]

    remaining = list(components)
    panels: list[dict[str, Any]] = []
    for anchor in anchors:
        if not remaining:
            break
        component = min(
            remaining,
            key=lambda box: _distance_to_box(anchor["center"], box),
        )
        remaining.remove(component)
        panels.append(
            {
                "floorNumber": int(anchor["floorNumber"]),
                "floorLabel": anchor["floorLabel"],
                "floorNumberSource": "local_ocr_panel",
                "box": _crop_box(component, width, height),
            }
        )
    return panels


def _entries_in_crop(
    entries: Sequence[dict[str, Any]],
    box: Sequence[int],
) -> list[dict[str, Any]]:
    left, top, right, bottom = box
    output: list[dict[str, Any]] = []
    for entry in entries:
        x, y = entry["center"]
        if not (left <= x <= right and top <= y <= bottom):
            continue
        if _level_anchors([entry]):
            continue
        output.append(
            {
                **entry,
                "center": [x - left, y - top],
                "box": [
                    entry["box"][0] - left,
                    entry["box"][1] - top,
                    entry["box"][2] - left,
                    entry["box"][3] - top,
                ],
            }
        )
    return output


def _resize_for_analysis(image: np.ndarray, maximum: int = 1200) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, maximum / max(height, width))
    if scale == 1.0:
        return image, scale
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _line_mask(gray: np.ndarray) -> np.ndarray:
    # Architectural wall ink is deliberately much darker than wood, tile,
    # landscaping, furniture, and other presentation textures. A permissive
    # threshold makes those textures become fake walls in the 3D viewer.
    inverted = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)[1]
    return cv2.morphologyEx(
        inverted,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )


def _raw_lines(mask: np.ndarray) -> list[tuple[float, float, float, float]]:
    height, width = mask.shape
    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=max(18, min(height, width) // 32),
        minLineLength=max(18, min(height, width) // 24),
        maxLineGap=max(5, min(height, width) // 80),
    )
    output: list[tuple[float, float, float, float]] = []
    if lines is None:
        return output
    minimum = min(height, width) * 0.045
    for packed in lines:
        x1, y1, x2, y2 = (float(value) for value in packed[0])
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < minimum:
            continue
        if (x2, y2) < (x1, y1):
            x1, y1, x2, y2 = x2, y2, x1, y1
        output.append((x1, y1, x2, y2))
    return output


def _merge_lines(
    lines: Sequence[tuple[float, float, float, float]],
    coordinate_tolerance: float = 7.0,
    gap_tolerance: float = 14.0,
) -> list[tuple[float, float, float, float]]:
    """Merge collinear segments without imposing Manhattan-world geometry."""

    def properties(line: tuple[float, float, float, float]) -> tuple[float, np.ndarray, np.ndarray, tuple[float, float]]:
        start = np.asarray(line[:2], dtype=np.float64)
        end = np.asarray(line[2:], dtype=np.float64)
        direction = end - start
        length = float(np.linalg.norm(direction))
        direction /= max(length, 1e-9)
        if direction[0] < 0 or (abs(direction[0]) < 1e-9 and direction[1] < 0):
            direction *= -1
        normal = np.asarray([-direction[1], direction[0]])
        projections = (float(start @ direction), float(end @ direction))
        return math.atan2(direction[1], direction[0]) % math.pi, direction, normal, tuple(sorted(projections))

    groups: list[list[tuple[float, float, float, float]]] = []
    for line in sorted(lines, key=lambda item: -math.hypot(item[2] - item[0], item[3] - item[1])):
        angle, direction, normal, interval = properties(line)
        midpoint = (np.asarray(line[:2]) + np.asarray(line[2:])) / 2
        target = None
        for group in groups:
            reference = group[0]
            ref_angle, ref_direction, ref_normal, _ = properties(reference)
            angle_delta = abs(angle - ref_angle)
            angle_delta = min(angle_delta, math.pi - angle_delta)
            if angle_delta > math.radians(6):
                continue
            ref_midpoint = (np.asarray(reference[:2]) + np.asarray(reference[2:])) / 2
            if abs(float((midpoint - ref_midpoint) @ ref_normal)) > coordinate_tolerance:
                continue
            endpoints = [np.asarray(item[:2]) for item in group] + [np.asarray(item[2:]) for item in group]
            group_projection = [float(point @ ref_direction) for point in endpoints]
            projected = [float(np.asarray(line[:2]) @ ref_direction), float(np.asarray(line[2:]) @ ref_direction)]
            if min(projected) > max(group_projection) + gap_tolerance or max(projected) < min(group_projection) - gap_tolerance:
                continue
            target = group
            break
        if target is None:
            groups.append([line])
        else:
            target.append(line)

    merged: list[tuple[float, float, float, float]] = []
    for group in groups:
        points = np.asarray(
            [point for line in group for point in ((line[0], line[1]), (line[2], line[3]))],
            dtype=np.float64,
        )
        center = points.mean(axis=0)
        _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
        direction = vectors[0]
        if direction[0] < 0 or (abs(direction[0]) < 1e-9 and direction[1] < 0):
            direction *= -1
        projection = (points - center) @ direction
        start = center + direction * float(projection.min())
        end = center + direction * float(projection.max())
        merged.append((float(start[0]), float(start[1]), float(end[0]), float(end[1])))
    return merged


def _detected_stair_annotations(
    lines: Sequence[tuple[float, float, float, float]],
    shape: tuple[int, int],
) -> list[dict[str, Any]]:
    """Find compact runs of repeated parallel tread lines as stair symbols."""

    minimum = min(shape)
    candidates = []
    for line in lines:
        x1, y1, x2, y2 = line
        length = math.hypot(x2 - x1, y2 - y1)
        if not minimum * 0.018 <= length <= minimum * 0.16:
            continue
        angle = math.atan2(y2 - y1, x2 - x1) % math.pi
        candidates.append(
            {
                "line": line,
                "length": length,
                "angle": angle,
                "midpoint": np.asarray([(x1 + x2) / 2, (y1 + y2) / 2]),
            }
        )
    adjacent: list[set[int]] = [set() for _ in candidates]
    for first in range(len(candidates)):
        a = candidates[first]
        direction = np.asarray([math.cos(a["angle"]), math.sin(a["angle"])])
        for second in range(first + 1, len(candidates)):
            b = candidates[second]
            angle_delta = abs(a["angle"] - b["angle"])
            angle_delta = min(angle_delta, math.pi - angle_delta)
            if angle_delta > math.radians(6):
                continue
            ratio = a["length"] / max(b["length"], 1e-9)
            if not 0.58 <= ratio <= 1.72:
                continue
            delta = b["midpoint"] - a["midpoint"]
            along = abs(float(delta @ direction))
            across = abs(float(delta @ np.asarray([-direction[1], direction[0]])))
            if along <= max(a["length"], b["length"]) * 0.72 and 3 <= across <= minimum * 0.09:
                adjacent[first].add(second)
                adjacent[second].add(first)

    output: list[dict[str, Any]] = []
    visited: set[int] = set()
    for start in range(len(candidates)):
        if start in visited:
            continue
        stack = [start]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacent[current] - visited)
        if not 5 <= len(component) <= 15:
            continue
        endpoints = np.asarray(
            [
                point
                for index in component
                for point in (
                    candidates[index]["line"][:2],
                    candidates[index]["line"][2:],
                )
            ],
            dtype=np.float64,
        )
        left, top = endpoints.min(axis=0)
        right, bottom = endpoints.max(axis=0)
        if max(right - left, bottom - top) > minimum * 0.3:
            continue
        angles = np.asarray([candidates[index]["angle"] for index in component])
        mean_angle = 0.5 * math.atan2(
            float(np.sin(2 * angles).mean()),
            float(np.cos(2 * angles).mean()),
        )
        direction = np.asarray([math.cos(mean_angle), math.sin(mean_angle)])
        normal = np.asarray([-direction[1], direction[0]])
        along = endpoints @ direction
        across = np.asarray(
            [candidates[index]["midpoint"] @ normal for index in component]
        )
        along_span = float(along.max() - along.min())
        across_sorted = np.unique(np.round(np.sort(across), 1))
        gaps = np.diff(across_sorted)
        gaps = gaps[gaps >= 2]
        if len(gaps) < 3 or float(gaps.mean()) <= 0:
            continue
        if float(gaps.std() / gaps.mean()) > 0.46:
            continue
        across_span = float(across_sorted[-1] - across_sorted[0])
        if not along_span * 0.35 <= across_span <= along_span * 3.2:
            continue
        output.append(
            {
                "text": "Possible stairs (parallel tread symbol)",
                "score": 0.54,
                "source": "parallel_line_detector",
                "center": [float((left + right) / 2), float((top + bottom) / 2)],
                "box": [float(left), float(top), float(right), float(bottom)],
            }
        )
    consolidated: list[dict[str, Any]] = []
    for candidate in output:
        center = np.asarray(candidate["center"])
        if any(
            float(np.linalg.norm(center - np.asarray(existing["center"])))
            < minimum * 0.08
            for existing in consolidated
        ):
            continue
        consolidated.append(candidate)
    return consolidated[:3]


def _drawing_bounds(
    lines: Sequence[tuple[float, float, float, float]],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    if not lines:
        return 0.04 * width, 0.04 * height, 0.96 * width, 0.96 * height
    xs: list[float] = []
    ys: list[float] = []
    for x1, y1, x2, y2 in lines:
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    pad = max(4.0, min(width, height) * 0.015)
    return (
        max(0.0, min(xs) - pad),
        max(0.0, min(ys) - pad),
        min(float(width), max(xs) + pad),
        min(float(height), max(ys) + pad),
    )


def _world_transform(
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float, float]:
    left, top, right, bottom = bounds
    pixel_width = max(1.0, right - left)
    pixel_depth = max(1.0, bottom - top)
    meters_per_pixel = 14.0 / max(pixel_width, pixel_depth)
    return (
        (left + right) / 2,
        (top + bottom) / 2,
        meters_per_pixel,
        pixel_width * meters_per_pixel,
        pixel_depth * meters_per_pixel,
    )


def _shared_normalized_bounds(
    records: Sequence[dict[str, Any]],
) -> tuple[float, float, float, float]:
    """Keep every sheet in one page coordinate frame.

    Architectural floor sets are commonly drawn at one scale and aligned on
    matching sheets.  Normalizing each drawing independently destroys that
    relationship, so the union crop is calculated once and reused by every
    floor.  The result is still an estimate until a printed dimension or
    surveyed control point supplies metric scale.
    """

    normalized: list[tuple[float, float, float, float]] = []
    for record in records:
        image = record["image"]
        left, top, right, bottom = record["localBounds"]
        height, width = image.shape[:2]
        normalized.append(
            (left / width, top / height, right / width, bottom / height)
        )
    left = min(item[0] for item in normalized)
    top = min(item[1] for item in normalized)
    right = max(item[2] for item in normalized)
    bottom = max(item[3] for item in normalized)
    padding = 0.006
    return (
        max(0.0, left - padding),
        max(0.0, top - padding),
        min(1.0, right + padding),
        min(1.0, bottom + padding),
    )


def _pixel_bounds(
    normalized: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    return (
        normalized[0] * width,
        normalized[1] * height,
        normalized[2] * width,
        normalized[3] * height,
    )


def _structural_lines(
    gray: np.ndarray,
    lines: Sequence[tuple[float, float, float, float]],
    annotations: Sequence[dict[str, Any]] = (),
) -> list[tuple[float, float, float, float]]:
    """Reject thin drawing detail before treating a line as a candidate wall."""

    dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]
    height, width = gray.shape
    retained: list[tuple[float, float, float, float]] = []
    for line in lines:
        x1, y1, x2, y2 = line
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min(height, width) * 0.045:
            continue
        midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
        in_printed_text = any(
            item["box"][0] - 3 <= midpoint[0] <= item["box"][2] + 3
            and item["box"][1] - 3 <= midpoint[1] <= item["box"][3] + 3
            for item in annotations
        )
        if in_printed_text and length < min(height, width) * 0.38:
            continue
        in_stair_symbol = any(
            re.search(r"\b(?:STAIR|STAIRS|UP|DN|DOWN)\b", item["text"], re.IGNORECASE)
            and item["box"][0] - 18 <= midpoint[0] <= item["box"][2] + 18
            and item["box"][1] - 45 <= midpoint[1] <= item["box"][3] + 45
            for item in annotations
        )
        if in_stair_symbol and length < min(height, width) * 0.18:
            continue
        direction_x = (x2 - x1) / max(length, 1e-9)
        direction_y = (y2 - y1) / max(length, 1e-9)
        normal_x, normal_y = -direction_y, direction_x
        offset_support: list[float] = []
        for offset in range(-4, 5):
            sample = np.zeros_like(dark)
            cv2.line(
                sample,
                (round(x1 + normal_x * offset), round(y1 + normal_y * offset)),
                (round(x2 + normal_x * offset), round(y2 + normal_y * offset)),
                255,
                1,
            )
            values = dark[sample > 0]
            offset_support.append(
                float(np.mean(values > 0)) if values.size else 0.0
            )
        if float(np.mean(offset_support)) >= 0.24 and max(offset_support) >= 0.4:
            retained.append(line)
    return _connected_structural_lines(retained, gray.shape)


def _structural_wall_mask(gray: np.ndarray) -> np.ndarray:
    """Retain thick architectural ink without promoting text/furniture to walls.

    Presentation floor plans contain many dark strokes.  Line detection alone
    cannot distinguish a sofa, a car, or a glyph from a wall.  Wall strokes are
    normally substantially thicker, so the distance-transform core is used as
    the structural seed and expanded only to the observed stroke width.
    """

    dark = np.where(gray < 105, 255, 0).astype(np.uint8)
    stroke_distance = cv2.distanceTransform(dark, cv2.DIST_L2, 5)
    thick_core = np.where(stroke_distance >= 2.25, 255, 0).astype(np.uint8)
    wall_mask = cv2.dilate(
        thick_core,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    return cv2.morphologyEx(
        wall_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )


def _nearest_passable(
    passable: np.ndarray, point: tuple[int, int], maximum_radius: int = 24
) -> tuple[int, int] | None:
    x, y = point
    height, width = passable.shape
    if 0 <= x < width and 0 <= y < height and passable[y, x]:
        return x, y
    for radius in range(1, maximum_radius + 1):
        left, right = max(0, x - radius), min(width - 1, x + radius)
        top, bottom = max(0, y - radius), min(height - 1, y + radius)
        candidates: list[tuple[int, int]] = []
        for candidate_x in range(left, right + 1):
            candidates.extend(((candidate_x, top), (candidate_x, bottom)))
        for candidate_y in range(top + 1, bottom):
            candidates.extend(((left, candidate_y), (right, candidate_y)))
        available = [candidate for candidate in candidates if passable[candidate[1], candidate[0]]]
        if available:
            return min(
                available,
                key=lambda candidate: (candidate[0] - x) ** 2
                + (candidate[1] - y) ** 2,
            )
    return None


def _grid_distances(
    passable: np.ndarray,
    start: tuple[int, int],
    targets: dict[tuple[int, int], int],
) -> dict[int, int]:
    """Return obstacle-aware pixel distances to target room-label seeds."""

    height, width = passable.shape
    distances = np.full((height, width), -1, dtype=np.int32)
    distances[start[1], start[0]] = 0
    queue: deque[tuple[int, int]] = deque([start])
    reached: dict[int, int] = {}
    while queue and len(reached) < len(targets):
        x, y = queue.popleft()
        target_index = targets.get((x, y))
        if target_index is not None:
            reached[target_index] = int(distances[y, x])
        next_distance = int(distances[y, x]) + 1
        for next_x, next_y in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if not (0 <= next_x < width and 0 <= next_y < height):
                continue
            if not passable[next_y, next_x] or distances[next_y, next_x] >= 0:
                continue
            distances[next_y, next_x] = next_distance
            queue.append((next_x, next_y))
    return reached


def _space_graph(
    gray: np.ndarray,
    annotations: Sequence[dict[str, Any]],
    bounds: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Build a conservative, explicitly unverified room connectivity graph.

    Nodes come from printed architectural room labels.  Edges are the minimum
    set of obstacle-aware paths needed to connect reachable labels through gaps
    in the observed thick-wall mask.  They are *not* claimed as verified doors.
    """

    room_labels = [item for item in annotations if _is_room_label(item["text"])]
    nodes: list[dict[str, Any]] = []
    center_x, center_y, meters_per_pixel, _, _ = _world_transform(bounds)
    for item in room_labels[:36]:
        nodes.append(
            {
                "id": f"space_{len(nodes) + 1}",
                "label": item["text"],
                "position": [
                    round((item["center"][0] - center_x) * meters_per_pixel, 4),
                    round((item["center"][1] - center_y) * meters_per_pixel, 4),
                ],
                "confidence": round(float(item["score"]), 2),
                "source": item.get("source", "plan_ocr"),
            }
        )
    if len(nodes) < 2:
        return {
            "nodes": nodes,
            "edges": [],
            "connected": len(nodes) == 1,
            "verified": False,
            "method": "plan_label_opening_paths",
        }

    wall_mask = _structural_wall_mask(gray)
    maximum = max(gray.shape)
    analysis_scale = min(1.0, 480.0 / maximum)
    if analysis_scale < 1.0:
        target_size = (
            max(1, round(gray.shape[1] * analysis_scale)),
            max(1, round(gray.shape[0] * analysis_scale)),
        )
        wall_mask = cv2.resize(wall_mask, target_size, interpolation=cv2.INTER_NEAREST)
    points = cv2.findNonZero(wall_mask)
    if points is None or len(points) < 3:
        return {
            "nodes": nodes,
            "edges": [],
            "connected": False,
            "verified": False,
            "method": "plan_label_opening_paths",
        }

    # The hull limits pathfinding to the observed drawing.  It is deliberately
    # only a routing workspace; it is never exported as the building footprint.
    interior = np.zeros_like(wall_mask)
    cv2.fillConvexPoly(interior, cv2.convexHull(points), 255)
    margin = max(2, round(min(interior.shape) * 0.012))
    interior = cv2.erode(
        interior,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1,) * 2),
    )
    passable = (interior > 0) & (wall_mask == 0)

    seeds: list[tuple[int, int] | None] = []
    for item in room_labels[:36]:
        point = (
            round(item["center"][0] * analysis_scale),
            round(item["center"][1] * analysis_scale),
        )
        seeds.append(_nearest_passable(passable, point))

    pair_distances: dict[tuple[int, int], int] = {}
    for source_index, source in enumerate(seeds):
        if source is None:
            continue
        targets = {
            target: target_index
            for target_index, target in enumerate(seeds)
            if target is not None and target_index != source_index
        }
        reached = _grid_distances(passable, source, targets)
        for target_index, distance in reached.items():
            pair = tuple(sorted((source_index, target_index)))
            pair_distances[pair] = min(pair_distances.get(pair, distance), distance)

    # A minimum spanning forest avoids claiming every mutually reachable label
    # pair is a direct doorway while still exposing whether the floor connects.
    parent = list(range(len(nodes)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    edges: list[dict[str, Any]] = []
    for (first, second), distance in sorted(
        pair_distances.items(), key=lambda item: item[1]
    ):
        first_root, second_root = root(first), root(second)
        if first_root == second_root:
            continue
        parent[second_root] = first_root
        edges.append(
            {
                "id": f"connection_{len(edges) + 1}",
                "from": nodes[first]["id"],
                "to": nodes[second]["id"],
                "distanceMeters": round(
                    distance / max(analysis_scale, 1e-9) * meters_per_pixel, 2
                ),
                "kind": "inferred_traversable_path",
                "verified": False,
                "confidence": 0.38,
            }
        )
    roots = {root(index) for index, seed in enumerate(seeds) if seed is not None}
    return {
        "nodes": nodes,
        "edges": edges,
        "connected": len(roots) == 1 and all(seed is not None for seed in seeds),
        "verified": False,
        "method": "plan_label_opening_paths",
        "note": (
            "Room labels are observed plan text. Connections are candidate paths "
            "through gaps in thick wall ink, not verified door or egress geometry."
        ),
    }


def _connected_structural_lines(
    lines: Sequence[tuple[float, float, float, float]],
    shape: tuple[int, int],
) -> list[tuple[float, float, float, float]]:
    """Remove short, isolated object strokes while preserving a wall network."""

    connection_distance = max(6.0, min(shape) * 0.025)
    long_line = max(shape) * 0.30

    def point_segment_distance(
        point: np.ndarray, line: tuple[float, float, float, float]
    ) -> float:
        start = np.asarray(line[:2], dtype=np.float64)
        end = np.asarray(line[2:], dtype=np.float64)
        segment = end - start
        position = float((point - start) @ segment) / max(
            float(segment @ segment), 1e-9
        )
        projection = start + np.clip(position, 0.0, 1.0) * segment
        return float(np.linalg.norm(point - projection))

    output: list[tuple[float, float, float, float]] = []
    for index, line in enumerate(lines):
        length = math.hypot(line[2] - line[0], line[3] - line[1])
        connected = length >= long_line
        endpoints = (np.asarray(line[:2]), np.asarray(line[2:]))
        if not connected:
            for other_index, other in enumerate(lines):
                if other_index == index:
                    continue
                other_endpoints = (np.asarray(other[:2]), np.asarray(other[2:]))
                if min(
                    *(point_segment_distance(point, other) for point in endpoints),
                    *(point_segment_distance(point, line) for point in other_endpoints),
                ) <= connection_distance:
                    connected = True
                    break
        if connected:
            output.append(line)
    return output


def _wall_records(
    lines: Sequence[tuple[float, float, float, float]],
    bounds: tuple[float, float, float, float],
    gray: np.ndarray,
) -> list[dict[str, Any]]:
    center_x, center_y, meters_per_pixel, _, _ = _world_transform(bounds)
    dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]
    height, width = gray.shape
    records: list[dict[str, Any]] = []
    for index, (x1, y1, x2, y2) in enumerate(lines):
        first = [(x1 - center_x) * meters_per_pixel, (y1 - center_y) * meters_per_pixel]
        second = [(x2 - center_x) * meters_per_pixel, (y2 - center_y) * meters_per_pixel]
        length = math.dist(first, second)
        if length < 0.45:
            continue
        sample = np.zeros((height, width), dtype=np.uint8)
        cv2.line(sample, (round(x1), round(y1)), (round(x2), round(y2)), 255, 7)
        values = dark[sample > 0]
        support = float(np.mean(values > 0)) if values.size else 0.0
        records.append(
            {
                "id": f"wall_{index + 1}",
                "start": [round(first[0], 4), round(first[1], 4)],
                "end": [round(second[0], 4), round(second[1], 4)],
                "heightMeters": 2.7,
                "thicknessMeters": 0.14,
                "drawingSupport": round(support, 3),
                "confidence": round(
                    min(0.82, 0.42 + 0.34 * min(1.0, support / 0.6) + length / 80),
                    2,
                ),
            }
        )
    return records


def _room_zones(
    lines: Sequence[tuple[float, float, float, float]],
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
    annotations: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    height, width = shape
    wall_canvas = np.zeros((height, width), dtype=np.uint8)
    line_width = max(4, min(height, width) // 140)
    for x1, y1, x2, y2 in lines:
        cv2.line(wall_canvas, (round(x1), round(y1)), (round(x2), round(y2)), 255, line_width)
    wall_canvas = cv2.morphologyEx(
        wall_canvas,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=2,
    )
    left, top, right, bottom = (round(value) for value in bounds)
    free = cv2.bitwise_not(wall_canvas)
    roi = np.zeros_like(free)
    cv2.rectangle(roi, (left, top), (right, bottom), 255, -1)
    free = cv2.bitwise_and(free, roi)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free)
    area = max(1, (right - left) * (bottom - top))
    center_x, center_y, meters_per_pixel, _, _ = _world_transform(bounds)
    zones: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y, component_width, component_height, component_area = stats[label]
        touches_edge = (
            x <= left + 2
            or y <= top + 2
            or x + component_width >= right - 2
            or y + component_height >= bottom - 2
        )
        if touches_edge or component_area < area * 0.012 or component_area > area * 0.70:
            continue
        mask = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        epsilon = max(1.5, cv2.arcLength(contour, True) * 0.006)
        simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(simplified) < 3:
            continue
        polygon = [
            [
                round((float(point[0]) - center_x) * meters_per_pixel, 4),
                round((float(point[1]) - center_y) * meters_per_pixel, 4),
            ]
            for point in simplified
        ]
        labels_here = [
            item
            for item in annotations
            if cv2.pointPolygonTest(contour, tuple(item["center"]), False) >= 0
            and _is_room_label(item["text"])
        ]
        best_label = max(labels_here, key=lambda item: item["score"], default=None)
        zones.append(
            {
                "id": f"zone_{len(zones) + 1}",
                "label": best_label["text"] if best_label else f"Unlabeled space {len(zones) + 1}",
                "polygon": polygon,
                "areaMetersSquared": round(component_area * meters_per_pixel**2, 2),
                "confidence": round(0.58 + (0.2 * best_label["score"] if best_label else 0), 2),
                "labelSource": "plan_ocr" if best_label else "geometry_only",
            }
        )
    return sorted(zones, key=lambda zone: zone["areaMetersSquared"], reverse=True)[:30]


def _preview(
    image: np.ndarray,
    lines: Sequence[tuple[float, float, float, float]],
    path: Path,
) -> None:
    preview = image.copy()
    for x1, y1, x2, y2 in lines:
        first, second = (round(x1), round(y1)), (round(x2), round(y2))
        cv2.line(preview, first, second, (32, 86, 220), 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", preview)
    if ok:
        encoded.tofile(path)


def _transparent_overlay(image: np.ndarray, path: Path) -> None:
    """Save the exact plan ink without turning the white page into a slab."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    alpha = np.clip((242.0 - gray.astype(np.float32)) * 3.5, 0, 255).astype(
        np.uint8
    )
    overlay = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    overlay[:, :, 3] = alpha
    ok, encoded = cv2.imencode(".png", overlay)
    if ok:
        encoded.tofile(path)


def _source_plan_image(image: np.ndarray, path: Path) -> None:
    """Save a lossless, opaque copy for the authoritative 2D plan view."""

    ok, encoded = cv2.imencode(".png", image)
    if ok:
        encoded.tofile(path)


def _world_polygon(
    points: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> list[list[float]]:
    center_x, center_y, scale, _, _ = _world_transform(bounds)
    return [
        [
            round((float(point[0]) - center_x) * scale, 4),
            round((float(point[1]) - center_y) * scale, 4),
        ]
        for point in points.reshape(-1, 2)
    ]


def _footprint_polygon(
    lines: Sequence[tuple[float, float, float, float]],
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
) -> list[list[float]]:
    """Trace the outer connected wall network without replacing it by a box."""

    canvas = np.zeros(shape, dtype=np.uint8)
    width = max(5, min(shape) // 120)
    for x1, y1, x2, y2 in lines:
        cv2.line(canvas, (round(x1), round(y1)), (round(x2), round(y2)), 255, width)
    canvas = cv2.morphologyEx(
        canvas,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=2,
    )
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    simplified = cv2.approxPolyDP(
        contour,
        max(1.5, cv2.arcLength(contour, True) * 0.004),
        True,
    )
    return _world_polygon(simplified, bounds)


def _plan_features(
    annotations: Sequence[dict[str, Any]],
    bounds: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    center_x, center_y, scale, _, _ = _world_transform(bounds)
    features: list[dict[str, Any]] = []
    for item in annotations:
        if not re.search(r"\b(?:STAIR|STAIRS|UP|DN|DOWN)\b", item["text"], re.IGNORECASE):
            continue
        left, top, right, bottom = item["box"]
        pad_x = max(12.0, (right - left) * 0.7)
        pad_y = max(18.0, (bottom - top) * 1.2)
        polygon = [
            [round((left - pad_x - center_x) * scale, 4), round((top - pad_y - center_y) * scale, 4)],
            [round((right + pad_x - center_x) * scale, 4), round((top - pad_y - center_y) * scale, 4)],
            [round((right + pad_x - center_x) * scale, 4), round((bottom + pad_y - center_y) * scale, 4)],
            [round((left - pad_x - center_x) * scale, 4), round((bottom + pad_y - center_y) * scale, 4)],
        ]
        features.append(
            {
                "id": f"stairs_{len(features) + 1}",
                "kind": "stairs",
                "label": item["text"],
                "polygon": polygon,
                "confidence": round(float(item["score"]), 2),
                "source": item.get("source", "plan_ocr"),
            }
        )
    return features


def _plan_labels(
    annotations: Sequence[dict[str, Any]],
    bounds: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    center_x, center_y, scale, _, _ = _world_transform(bounds)
    labels: list[dict[str, Any]] = []
    for item in annotations:
        if not _is_room_label(item["text"]):
            continue
        labels.append(
            {
                "id": f"plan_label_{len(labels) + 1}",
                "label": item["text"],
                "position": [
                    round((item["center"][0] - center_x) * scale, 4),
                    round((item["center"][1] - center_y) * scale, 4),
                ],
                "confidence": round(float(item["score"]), 2),
                "source": item.get("source", "plan_ocr"),
            }
        )
    return labels


def _is_room_label(text: str) -> bool:
    """Keep architectural spaces and reject page metadata/appliance abbreviations."""

    return bool(
        re.search(
            r"\b(?:LIVING|FAMILY|DINING|KITCHEN|BED(?:ROOM)?|PRIMARY|MASTER|"
            r"BATH(?:ROOM)?|LAUNDRY|HALL(?:WAY)?|CORRIDOR|FOYER|ENTRY|CLOSET|WIC|"
            r"GARAGE|UTILITY|MECH(?:ANICAL)?|OFFICE|STUDY|WORKOUT|PATIO|TERRACE|"
            r"BALCONY|ROOF\s*DECK|BASEMENT|ATTIC|PANTRY|MUD\s*ROOM)\b",
            text,
            re.IGNORECASE,
        )
    )


def build_floorplan_structure(
    input_paths: Sequence[Path],
    evidence_ids: Sequence[str],
    floor_numbers: Sequence[int],
    output_directory: Path,
    case_id: str,
) -> dict[str, Any]:
    if not input_paths or len(input_paths) != len(evidence_ids):
        raise ValueError("Floorplan paths and evidence identifiers must match")
    if len(floor_numbers) != len(input_paths):
        raise ValueError("Every floorplan requires a floor number")
    output_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    reference_plans: list[dict[str, Any]] = []
    for order, (path, evidence_id, floor_number) in enumerate(
        zip(input_paths, evidence_ids, floor_numbers, strict=True)
    ):
        original = _read_image(path)
        entries = _ocr_entries(path)
        source_url = f"/assets/{case_id}/uploads/{path.name}"
        if _is_site_reference(entries):
            overlay_name = f"reference_{order + 1}_site_plan.png"
            _transparent_overlay(original, output_directory / overlay_name)
            reference_plans.append(
                {
                    "index": len(reference_plans),
                    "kind": "site_plan",
                    "label": "Site plan",
                    "sourceEvidenceId": evidence_id,
                    "sourceImageUrl": source_url,
                    "planOverlayUrl": (
                        f"/assets/{case_id}/reconstruction/"
                        f"{output_directory.name}/{overlay_name}"
                    ),
                    "structuralFloor": False,
                    "reason": (
                        "The sheet is explicitly labeled SITE PLAN and is retained "
                        "as parcel/exterior reference evidence, not an interior floor."
                    ),
                }
            )
            continue

        panels = _sheet_panels(original, entries, floor_number)
        source_height, source_width = original.shape[:2]
        for panel_order, panel in enumerate(panels):
            left, top, right, bottom = panel["box"]
            cropped = original[top:bottom, left:right].copy()
            annotations = _entries_in_crop(entries, panel["box"])
            image, analysis_scale = _resize_for_analysis(cropped)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = _line_mask(gray)
            all_lines = _merge_lines(_raw_lines(mask))
            scaled_annotations = [
                {
                    **item,
                    "center": [value * analysis_scale for value in item["center"]],
                    "box": [value * analysis_scale for value in item["box"]],
                }
                for item in annotations
            ]
            scaled_annotations.extend(
                _detected_stair_annotations(all_lines, gray.shape)
            )
            local_bounds = _drawing_bounds(
                all_lines, image.shape[1], image.shape[0]
            )
            records.append(
                {
                    "order": order,
                    "panelOrder": panel_order,
                    "panelCount": len(panels),
                    "path": path,
                    "evidenceId": evidence_id,
                    "floorNumber": panel["floorNumber"],
                    "floorLabel": panel["floorLabel"],
                    "floorNumberSource": panel["floorNumberSource"],
                    "annotations": scaled_annotations,
                    "sourceImageSize": {
                        "width": int(source_width),
                        "height": int(source_height),
                    },
                    "sourceCrop": {
                        "left": round(left / source_width, 6),
                        "top": round(top / source_height, 6),
                        "right": round(right / source_width, 6),
                        "bottom": round(bottom / source_height, 6),
                    },
                    "image": image,
                    "gray": gray,
                    "analysisScale": analysis_scale,
                    "allLines": all_lines,
                    "structuralLines": _structural_lines(
                        gray,
                        all_lines,
                        scaled_annotations,
                    ),
                    "localBounds": local_bounds,
                }
            )

    panel_split = any(record["panelCount"] > 1 for record in records)
    shared_bounds = (
        _shared_normalized_bounds(records) if records and not panel_split else None
    )
    floors: list[dict[str, Any]] = []
    for record in records:
        path = record["path"]
        evidence_id = record["evidenceId"]
        resolved_floor_number = record["floorNumber"]
        image = record["image"]
        gray = record["gray"]
        lines = record["structuralLines"]
        if shared_bounds is None:
            bounds = record["localBounds"]
            normalized_bounds = (
                bounds[0] / image.shape[1],
                bounds[1] / image.shape[0],
                bounds[2] / image.shape[1],
                bounds[3] / image.shape[0],
            )
        else:
            bounds = _pixel_bounds(
                shared_bounds, image.shape[1], image.shape[0]
            )
            normalized_bounds = shared_bounds
        _, _, _, width_meters, depth_meters = _world_transform(bounds)
        walls = _wall_records(lines, bounds, gray)
        rooms = _room_zones(lines, bounds, gray.shape, record["annotations"])
        footprint = _footprint_polygon(lines, bounds, gray.shape)
        features = _plan_features(record["annotations"], bounds)
        plan_labels = _plan_labels(record["annotations"], bounds)
        space_graph = _space_graph(
            gray,
            record["annotations"],
            bounds,
        )
        labeled_space_count = len(space_graph["nodes"])
        enclosed_label_count = sum(
            1 for room in rooms if room.get("labelSource") == "plan_ocr"
        )
        label_enclosure_ratio = (
            enclosed_label_count / labeled_space_count if labeled_space_count else 0.0
        )
        topology_status = (
            "candidate"
            if rooms and label_enclosure_ratio >= 0.55
            else "source_plan_only"
        )
        file_stem = f"floor_{resolved_floor_number}_{len(floors) + 1}"
        preview_name = f"{file_stem}_vectors.png"
        _preview(image, lines, output_directory / preview_name)
        overlay_name = f"{file_stem}_overlay.png"
        _transparent_overlay(image, output_directory / overlay_name)
        source_plan_name = f"{file_stem}_source.png"
        _source_plan_image(image, output_directory / source_plan_name)
        source_url = f"/assets/{case_id}/uploads/{path.name}"
        floors.append(
            {
                "index": len(floors),
                "floorNumber": int(resolved_floor_number),
                "floorNumberSource": record["floorNumberSource"],
                "label": record["floorLabel"],
                "elevationMeters": 0.0,
                "sourceEvidenceId": evidence_id,
                "sourceImageUrl": source_url,
                "vectorPreviewUrl": (
                    f"/assets/{case_id}/reconstruction/{output_directory.name}/{preview_name}"
                ),
                "planOverlayUrl": (
                    f"/assets/{case_id}/reconstruction/{output_directory.name}/{overlay_name}"
                ),
                "planImageUrl": (
                    f"/assets/{case_id}/reconstruction/{output_directory.name}/{source_plan_name}"
                ),
                "imageSize": {
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                },
                "sourceImageSize": record["sourceImageSize"],
                "sourceImageCrop": record["sourceCrop"],
                "analysisScale": record["analysisScale"],
                "textureCrop": {
                    "left": round(normalized_bounds[0], 5),
                    "top": round(normalized_bounds[1], 5),
                    "right": round(normalized_bounds[2], 5),
                    "bottom": round(normalized_bounds[3], 5),
                },
                "widthMeters": round(width_meters, 3),
                "depthMeters": round(depth_meters, 3),
                "wallHeightMeters": 2.7,
                "walls": walls,
                "rooms": rooms,
                "footprint": footprint,
                "features": features,
                "planLabels": plan_labels,
                "spaceGraph": space_graph,
                "vectorization": {
                    "status": topology_status,
                    "sourcePlanAuthoritative": True,
                    "safeForExtrusion": topology_status == "candidate",
                    "wallGeometryVerified": False,
                    "roomTopologyVerified": False,
                    "labeledSpaces": labeled_space_count,
                    "labelsInsideEnclosedZones": enclosed_label_count,
                    "labelEnclosureRatio": round(label_enclosure_ratio, 3),
                    "reason": (
                        "Candidate room zones enclose most printed space labels; "
                        "the exact supplied plan remains authoritative."
                        if topology_status == "candidate"
                        else "The wall network does not close enough labeled spaces. "
                        "Candidate walls must not replace or extrude over the supplied plan."
                    ),
                },
                "observedRoomLabels": [
                    item["text"]
                    for item in record["annotations"]
                    if item.get("source") == "plan_ocr"
                ],
                "metrics": {
                    "wallSegments": len(walls),
                    "enclosedZones": len(rooms),
                    "stairFeatures": len(features),
                    "labeledSpaces": labeled_space_count,
                    "candidateConnections": len(space_graph["edges"]),
                },
            }
        )
    floors.sort(key=lambda floor: (floor["floorNumber"], floor["index"]))
    floor_to_floor_meters = 3.05
    if floors:
        lowest_floor = min(floor["floorNumber"] for floor in floors)
        for index, floor in enumerate(floors):
            floor["index"] = index
            floor["elevationMeters"] = round(
                (floor["floorNumber"] - lowest_floor) * floor_to_floor_meters,
                3,
            )
        roof_elevation = round(
            max(floor["elevationMeters"] for floor in floors)
            + floor_to_floor_meters,
            3,
        )
    else:
        roof_elevation = 0.0
    model = {
        "schemaVersion": 5,
        "caseId": case_id,
        "coordinateSystem": (
            "independent plan-panel y-up; each floor is centered in its own observed drawing"
            if panel_split
            else "shared plan-sheet y-up; x/z use one normalized document frame"
        ),
        "alignment": {
            "method": (
                "independent_observed_panels"
                if panel_split
                else "shared_document_coordinates"
            ),
            "sharedNormalizedCrop": (
                [round(value, 6) for value in shared_bounds]
                if shared_bounds is not None
                else None
            ),
            "metricScaleSource": "unverified_display_scale",
            "metricScaleVerified": False,
            "floorToFloorMeters": floor_to_floor_meters,
            "confidence": 0.62 if panel_split else 0.58,
            "note": (
                "Each labeled floor panel is extracted and centered independently; "
                "the source sheet does not prove surveyed inter-floor alignment."
                if panel_split
                else "All supplied sheets preserve one page coordinate frame. "
                "Horizontal registration is not normalized independently per floor."
            ),
        },
        "floors": floors,
        "referencePlans": reference_plans,
        "roof": {
            "type": "unobserved",
            "elevationMeters": roof_elevation,
            "confidence": 0.0,
            "note": "No roof geometry is generated unless a roof plan or measured roof observation is supplied.",
        },
        "limitations": [
            "Plan coordinates use an unverified display scale because no machine-verified dimension was available.",
            "Arbitrary-angle candidate walls and enclosed polygons are extracted from supplied ink; openings and wall heights still require verification.",
            "Stair labels are annotations, not wall segments. A stair flight requires additional measured geometry before it is navigable.",
            "Roof geometry is unobserved and intentionally omitted.",
            "The shared sheet frame preserves relative floor placement, but it is not a surveyed inter-floor registration.",
            "A sheet explicitly labeled SITE PLAN is retained as reference evidence and never counted as an interior floor.",
            "Room photographs are not placed unless visual overlap or another measured constraint establishes a camera pose.",
            "Automatic label-to-label connections are routing candidates through observed wall gaps, not verified doors or egress routes.",
        ],
        "metrics": {
            "floorCount": len(floors),
            "wallSegments": sum(len(floor["walls"]) for floor in floors),
            "enclosedZones": sum(len(floor["rooms"]) for floor in floors),
        },
    }
    (output_directory / "structure.json").write_text(
        json.dumps(model, indent=2), encoding="utf-8"
    )
    return model
