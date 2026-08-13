from evaluate import evaluate_report


def test_acceptance_metrics_pass_for_complete_joint_reconstruction() -> None:
    manifest = {
        "id": "room",
        "frames": [
            {"index": 0, "roomId": "a", "expectedRegistered": True},
            {"index": 1, "roomId": "a", "expectedRegistered": True},
            {"index": 2, "roomId": "outlier", "expectedRegistered": False},
        ],
        "thresholds": {
            "minimumRegistrationRecall": 1,
            "minimumRegistrationPrecision": 1,
            "maximumRotationAgreementMedianDegrees": 10,
            "minimumGaussianCount": 100,
            "requireJointCameraAcceptance": True,
            "requirePoseGraphImprovement": True,
        },
    }
    report = {
        "connectedFrames": [0, 1],
        "gaussianCount": 200,
        "jointGeometry": {
            "accepted": True,
            "rotationAgreementMedianDeg": 2.2,
        },
        "poseOptimization": {"beforeRmse": 0.2, "afterRmse": 0.1},
    }

    result = evaluate_report(manifest, report)

    assert result["passed"] is True
    assert result["metrics"]["registrationRecall"] == 1
    assert result["metrics"]["registrationPrecision"] == 1
    assert result["metrics"]["roomCompleteness"] == {"a": 1.0, "outlier": 0.0}


def test_acceptance_metrics_fail_for_false_positive_and_missing_view() -> None:
    manifest = {
        "id": "room",
        "frames": [
            {"index": 0, "roomId": "a", "expectedRegistered": True},
            {"index": 1, "roomId": "a", "expectedRegistered": True},
            {"index": 2, "roomId": "outlier", "expectedRegistered": False},
        ],
        "thresholds": {
            "minimumRegistrationRecall": 1,
            "minimumRegistrationPrecision": 1,
            "maximumRotationAgreementMedianDegrees": 10,
        },
    }
    report = {
        "connectedFrames": [0, 2],
        "jointGeometry": {
            "accepted": True,
            "rotationAgreementMedianDeg": 2,
        },
    }

    result = evaluate_report(manifest, report)

    assert result["passed"] is False
    assert result["checks"]["registrationRecall"] is False
    assert result["checks"]["registrationPrecision"] is False


def test_quality_checks_require_points_overlap_color_and_preserved_count() -> None:
    manifest = {
        "id": "quality-room",
        "frames": [
            {"index": 0, "roomId": "a", "expectedRegistered": True},
            {"index": 1, "roomId": "a", "expectedRegistered": True},
        ],
        "thresholds": {
            "requirePointOptimization": True,
            "maximumPointRmseMeters": 0.2,
            "requireOverlapColorImprovement": True,
            "minimumOverlapColorSamples": 100,
            "requireScaleCountPreserved": True,
            "requireDenseSurfaceImprovement": True,
            "requireSourceCoverageCountPreserved": True,
            "minimumCrossViewSupportedRatio": 0.25,
        },
    }
    report = {
        "connectedFrames": [0, 1],
        "jointGeometry": {
            "accepted": True,
            "rotationAgreementMedianDeg": 2,
        },
        "poseOptimization": {
            "optimized": True,
            "pointConstraintCount": 200,
            "pointBeforeRmseMeters": 0.18,
            "pointAfterRmseMeters": 0.15,
        },
        "jointColorBalance": {
            "applied": True,
            "overlapSampleCount": 160,
            "overlapLogRmseBefore": 0.12,
            "overlapLogRmseAfter": 0.07,
        },
        "scaleRegularization": {
            "gaussianCountBefore": 500,
            "gaussianCountAfter": 500,
            "affected": 2,
        },
        "denseSurfaceRefinement": {
            "optimized": True,
            "denseConstraintCount": 240,
            "denseBeforeRmseMeters": 0.08,
            "denseAfterRmseMeters": 0.05,
        },
        "sourceCoverageRegularization": {
            "gaussianCountBefore": 500,
            "gaussianCountAfter": 500,
            "affected": 140,
        },
        "artifactCleanup": {
            "crossViewSupportedFraction": 0.42,
        },
    }

    result = evaluate_report(manifest, report)

    assert result["passed"] is True
    assert result["checks"]["pointOptimization"] is True
    assert result["checks"]["overlapColorImproved"] is True
    assert result["checks"]["scaleCountPreserved"] is True
    assert result["checks"]["denseSurfaceImproved"] is True
    assert result["checks"]["sourceCoverageCountPreserved"] is True
    assert result["checks"]["crossViewSupport"] is True
    assert result["metrics"]["pointConstraintCount"] == 200
    assert result["metrics"]["denseSurfaceConstraintCount"] == 240
