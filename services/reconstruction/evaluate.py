"""Deterministic acceptance metrics for StructureFirst reconstruction reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def evaluate_report(
    manifest: dict[str, Any],
    report: dict[str, Any],
    *,
    manifest_path: Path | None = None,
    verify_inputs: bool = False,
) -> dict[str, Any]:
    frames = manifest.get("frames", [])
    expected = {
        int(frame["index"])
        for frame in frames
        if frame.get("expectedRegistered") is True
    }
    expected_rejected = {
        int(frame["index"])
        for frame in frames
        if frame.get("expectedRegistered") is False
    }
    registered = {
        int(index)
        for index in report.get("connectedFrames", [])
        if isinstance(index, int)
    }
    true_positive = len(registered & expected)
    false_positive = len(registered & expected_rejected)
    recall = true_positive / len(expected) if expected else 1.0
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 1.0
    )

    room_members: dict[str, set[int]] = {}
    for frame in frames:
        room_members.setdefault(str(frame["roomId"]), set()).add(
            int(frame["index"])
        )
    room_completeness = {
        room_id: len(indices & registered) / len(indices)
        for room_id, indices in room_members.items()
    }
    pose = report.get("poseOptimization") or {}
    joint = report.get("jointGeometry") or {}
    before_rmse = _finite_number(pose.get("beforeRmse"))
    after_rmse = _finite_number(pose.get("afterRmse"))
    point_before_rmse = _finite_number(pose.get("pointBeforeRmseMeters"))
    point_after_rmse = _finite_number(pose.get("pointAfterRmseMeters"))
    dense = report.get("denseSurfaceRefinement") or {}
    dense_before_rmse = _finite_number(dense.get("denseBeforeRmseMeters"))
    dense_after_rmse = _finite_number(dense.get("denseAfterRmseMeters"))
    rotation_agreement = _finite_number(
        joint.get("rotationAgreementMedianDeg")
    )
    color = report.get("jointColorBalance") or {}
    color_before_rmse = _finite_number(color.get("overlapLogRmseBefore"))
    color_after_rmse = _finite_number(color.get("overlapLogRmseAfter"))
    color_samples = _nonnegative_integer(color.get("overlapSampleCount")) or 0
    scale = report.get("scaleRegularization") or {}
    scale_before_count = _nonnegative_integer(
        scale.get("gaussianCountBefore")
    )
    scale_after_count = _nonnegative_integer(scale.get("gaussianCountAfter"))
    footprint = report.get("sourceCoverageRegularization") or {}
    footprint_before_count = _nonnegative_integer(
        footprint.get("gaussianCountBefore")
    )
    footprint_after_count = _nonnegative_integer(
        footprint.get("gaussianCountAfter")
    )
    cleanup = report.get("artifactCleanup") or {}
    cross_view_supported = _finite_number(
        cleanup.get("crossViewSupportedFraction")
    )
    gaussian_count = _nonnegative_integer(report.get("gaussianCount")) or 0
    thresholds = manifest.get("thresholds") or {}

    input_verification = _verify_inputs(
        manifest, manifest_path
    ) if verify_inputs else {"requested": False, "passed": None, "files": []}
    checks = {
        "registrationRecall": recall
        >= float(thresholds.get("minimumRegistrationRecall", 0)),
        "registrationPrecision": precision
        >= float(thresholds.get("minimumRegistrationPrecision", 0)),
        "rotationAgreement": rotation_agreement is not None
        and rotation_agreement
        <= float(
            thresholds.get("maximumRotationAgreementMedianDegrees", 180)
        ),
        "gaussianCount": gaussian_count
        >= int(thresholds.get("minimumGaussianCount", 0)),
        "jointCameraAccepted": (
            not thresholds.get("requireJointCameraAcceptance", False)
            or joint.get("accepted") is True
        ),
        "poseGraphImproved": (
            not thresholds.get("requirePoseGraphImprovement", False)
            or (
                before_rmse is not None
                and after_rmse is not None
                and after_rmse < before_rmse
            )
        ),
        "pointOptimization": (
            not thresholds.get("requirePointOptimization", False)
            or (
                pose.get("optimized") is True
                and point_before_rmse is not None
                and point_after_rmse is not None
                and point_after_rmse < point_before_rmse
            )
        ),
        "pointRmse": (
            "maximumPointRmseMeters" not in thresholds
            or (
                point_after_rmse is not None
                and point_after_rmse
                <= float(thresholds["maximumPointRmseMeters"])
            )
        ),
        "denseSurfaceImproved": (
            not thresholds.get("requireDenseSurfaceImprovement", False)
            or (
                dense.get("optimized") is True
                and dense_before_rmse is not None
                and dense_after_rmse is not None
                and dense_after_rmse < dense_before_rmse
            )
        ),
        "overlapColorImproved": (
            not thresholds.get("requireOverlapColorImprovement", False)
            or (
                color.get("applied") is True
                and color_before_rmse is not None
                and color_after_rmse is not None
                and color_after_rmse < color_before_rmse
            )
        ),
        "overlapColorSamples": color_samples
        >= int(thresholds.get("minimumOverlapColorSamples", 0)),
        "scaleCountPreserved": (
            not thresholds.get("requireScaleCountPreserved", False)
            or (
                scale_before_count is not None
                and scale_after_count == scale_before_count
            )
        ),
        "sourceCoverageCountPreserved": (
            not thresholds.get("requireSourceCoverageCountPreserved", False)
            or (
                footprint_before_count is not None
                and footprint_after_count == footprint_before_count
            )
        ),
        "crossViewSupport": (
            "minimumCrossViewSupportedRatio" not in thresholds
            or (
                cross_view_supported is not None
                and cross_view_supported
                >= float(thresholds["minimumCrossViewSupportedRatio"])
            )
        ),
        "inputHashes": input_verification["passed"] is not False,
    }
    baseline = manifest.get("legacyBaseline") or {}
    baseline_connected = _nonnegative_integer(
        baseline.get("connectedFrameCount")
    )
    result = {
        "schemaVersion": 1,
        "datasetId": manifest.get("id"),
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "expectedRegistered": sorted(expected),
            "registered": sorted(registered),
            "registrationRecall": round(recall, 6),
            "registrationPrecision": round(precision, 6),
            "roomCompleteness": {
                key: round(value, 6)
                for key, value in sorted(room_completeness.items())
            },
            "jointCameraAccepted": joint.get("accepted") is True,
            "rotationAgreementMedianDegrees": rotation_agreement,
            "poseGraphBeforeRmse": before_rmse,
            "poseGraphAfterRmse": after_rmse,
            "pointBeforeRmseMeters": point_before_rmse,
            "pointAfterRmseMeters": point_after_rmse,
            "pointConstraintCount": _nonnegative_integer(
                pose.get("pointConstraintCount")
            ),
            "denseSurfaceOptimized": dense.get("optimized") is True,
            "denseSurfaceBeforeRmseMeters": dense_before_rmse,
            "denseSurfaceAfterRmseMeters": dense_after_rmse,
            "denseSurfaceConstraintCount": _nonnegative_integer(
                dense.get("denseConstraintCount")
            ),
            "overlapColorBeforeRmse": color_before_rmse,
            "overlapColorAfterRmse": color_after_rmse,
            "overlapColorSamples": color_samples,
            "scaleAdjustedGaussians": _nonnegative_integer(
                scale.get("affected")
            ),
            "scaleGaussianCountPreserved": (
                scale_before_count is not None
                and scale_after_count == scale_before_count
            ),
            "sourceCoverageAdjustedGaussians": _nonnegative_integer(
                footprint.get("affected")
            ),
            "sourceCoverageGaussianCountPreserved": (
                footprint_before_count is not None
                and footprint_after_count == footprint_before_count
            ),
            "crossViewSupportedRatio": cross_view_supported,
            "gaussianCount": gaussian_count,
            "gpu": joint.get("device"),
            "peakVramMb": _finite_number(joint.get("peakVramMb")),
            "legacyConnectedFrameGain": (
                len(registered) - baseline_connected
                if baseline_connected is not None
                else None
            ),
        },
        "inputVerification": input_verification,
        "limitations": [
            (
                f"This dataset labels {len(room_members)} capture components; "
                "it does not establish an unobserved connection between them."
                if len(room_members) > 1
                else "This dataset validates one observed capture component."
            ),
            "Registration metrics do not validate unseen surfaces or safe routes.",
            "Room/floor labels require separate ground-truth evaluation.",
        ],
    }
    return result


def _verify_inputs(
    manifest: dict[str, Any], manifest_path: Path | None
) -> dict[str, Any]:
    if manifest_path is None:
        return {"requested": True, "passed": False, "files": []}
    root = (manifest_path.parent / str(manifest["inputRoot"])).resolve()
    files = []
    passed = True
    for frame in manifest.get("frames", []):
        path = root / str(frame["file"])
        exists = path.is_file()
        digest = _sha256(path) if exists else None
        matches = exists and digest == frame.get("sha256")
        passed = passed and matches
        files.append(
            {
                "index": frame["index"],
                "exists": exists,
                "sha256Matches": matches,
            }
        )
    return {"requested": True, "passed": passed, "files": files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and value == value:
        return float(value)
    return None


def _nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, int) and value >= 0:
        return value
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-inputs", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = json.loads(args.registration.read_text(encoding="utf-8"))
    result = evaluate_report(
        manifest,
        report,
        manifest_path=args.manifest,
        verify_inputs=args.verify_inputs,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
