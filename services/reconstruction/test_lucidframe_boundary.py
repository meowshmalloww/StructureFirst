from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SERVICE_ROOT = Path(__file__).resolve().parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

import app as worker  # noqa: E402


def test_runtime_is_pinned_to_local_lucidframe_and_official_sharp() -> None:
    runtime = worker._lucidframe_runtime()
    checkpoint = worker._verify_official_sharp_checkpoint(require_present=True)

    assert runtime.reconstruct_sharp.__module__ == "sharp_wrapper"
    assert runtime.reconstruct_sharp360.__module__ == "sharp360_wrapper"
    assert runtime.compile_splat.__module__ == "splat_compiler"
    assert runtime.provenance["sharpModelUrl"] == worker.OFFICIAL_SHARP_MODEL_URL
    assert (
        runtime.provenance["sharpCheckpointSha256"]
        == worker.OFFICIAL_SHARP_CHECKPOINT_SHA256
    )
    assert checkpoint["verified"] is True
    assert checkpoint["sha256"] == worker.OFFICIAL_SHARP_CHECKPOINT_SHA256


def test_input_validation_requires_the_stored_exact_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases_root = (tmp_path / "cases").resolve()
    case_id = "case_12345678"
    upload_root = cases_root / case_id / "uploads"
    upload_root.mkdir(parents=True)
    source = upload_root / "original.jpg"
    source.write_bytes(b"original-image-file-bytes")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(worker, "CASES_ROOT", cases_root)

    request = worker.JobRequest(
        job_id="job_12345678",
        case_id=case_id,
        evidence_id="evidence_12345678",
        input_path=str(source),
        input_sha256=expected,
    )
    paths, fingerprints = worker._validated_inputs(request)

    assert paths == [source]
    assert fingerprints == [
        worker.InputFingerprint(sha256=expected, byte_size=source.stat().st_size)
    ]

    changed_request = request.model_copy(update={"input_sha256": "0" * 64})
    with pytest.raises(ValueError, match="no longer match"):
        worker._validated_inputs(changed_request)


def test_single_image_job_calls_lucidframe_with_the_original_path_and_no_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases_root = (tmp_path / "cases").resolve()
    case_id = "case_abcdefgh"
    upload_root = cases_root / case_id / "uploads"
    upload_root.mkdir(parents=True)
    source = upload_root / "capture.jpg"
    source.write_bytes(b"unchanged-responder-photo")
    fingerprint = worker._fingerprint(source)
    request = worker.JobRequest(
        job_id="artifact_abcdefgh",
        case_id=case_id,
        evidence_id="evidence_abcdefgh",
        input_path=str(source),
        input_sha256=fingerprint.sha256,
    )
    monkeypatch.setattr(worker, "CASES_ROOT", cases_root)
    with worker.JOBS_LOCK:
        worker.JOBS.clear()
    state = worker.JobState(
        job_id=request.job_id,
        case_id=request.case_id,
        evidence_id=request.evidence_id,
        status="queued",
        mode="single_image",
        created_at=worker.now_iso(),
        updated_at=worker.now_iso(),
    )
    worker._save_state(state)

    calls: list[Path] = []
    gaussians = SimpleNamespace(count=1_000)

    def reconstruct_sharp(path: Path, _output: Path) -> SimpleNamespace:
        calls.append(path)
        return gaussians

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("A non-single-image reconstruction path was called")

    def compile_splat(cloud: SimpleNamespace, output: Path) -> Path:
        output.write_bytes(b"\x00" * (cloud.count * 32))
        return output

    runtime = SimpleNamespace(
        reconstruct_sharp=reconstruct_sharp,
        reconstruct_sharp360=forbidden,
        compile_splat=compile_splat,
        provenance={"lucidFrameRevision": "test-revision"},
    )
    checkpoint = {
        "filename": worker.OFFICIAL_SHARP_CHECKPOINT,
        "sha256": worker.OFFICIAL_SHARP_CHECKPOINT_SHA256,
        "verified": True,
    }
    monkeypatch.setattr(worker, "_lucidframe_runtime", lambda: runtime)
    monkeypatch.setattr(
        worker,
        "_verify_official_sharp_checkpoint",
        lambda *, require_present: checkpoint,
    )

    worker._run_job(request, [source], [fingerprint])

    assert calls == [source]
    completed = worker._load_state(request.job_id)
    assert completed is not None and completed.status == "ready"
    manifest_path = (
        cases_root
        / case_id
        / "reconstruction"
        / request.job_id
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest["fallbackUsed"] is False
    assert manifest["pipelineEntrypoint"].endswith(
        "LucidFrame/backend/sharp_wrapper.py::reconstruct_sharp"
    )
    assert manifest["inputFiles"] == [
        {
            "evidenceId": request.evidence_id,
            "filename": source.name,
            "byteSize": source.stat().st_size,
            "sha256": fingerprint.sha256,
        }
    ]
    assert manifest["runtimeProvenance"]["checkpoint"]["verified"] is True
    assert "Original stored file path" in manifest["sourceFileDelivery"]


def test_multi_image_registration_failure_still_returns_exact_lucidframe_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import smart_connect

    cases_root = (tmp_path / "cases").resolve()
    case_id = "case_fallback1"
    upload_root = cases_root / case_id / "uploads"
    upload_root.mkdir(parents=True)
    sources = [upload_root / "front.jpg", upload_root / "side.jpg"]
    sources[0].write_bytes(b"first-exact-source-photo")
    sources[1].write_bytes(b"second-non-overlapping-photo")
    fingerprints = [worker._fingerprint(path) for path in sources]
    request = worker.JobRequest(
        job_id="artifact_fallback1",
        case_id=case_id,
        evidence_id="evidence_fallback1",
        evidence_ids=["evidence_fallback1", "evidence_fallback2"],
        input_path=str(sources[0]),
        input_paths=[str(path) for path in sources],
        input_sha256=fingerprints[0].sha256,
        input_sha256s=[item.sha256 for item in fingerprints],
        mode="multi_image",
    )
    monkeypatch.setattr(worker, "CASES_ROOT", cases_root)
    with worker.JOBS_LOCK:
        worker.JOBS.clear()
    worker._save_state(
        worker.JobState(
            job_id=request.job_id,
            case_id=request.case_id,
            evidence_id=request.evidence_id,
            status="queued",
            mode="multi_image",
            created_at=worker.now_iso(),
            updated_at=worker.now_iso(),
        )
    )

    report = {
        "schemaVersion": 1,
        "status": "failed",
        "frameCount": 2,
        "connectedFrameCount": 1,
        "confidenceScore": 0.0,
    }

    def registration_failure(*_args: object, **_kwargs: object) -> None:
        raise smart_connect.RegistrationError("No verified overlap", report)

    calls: list[Path] = []
    gaussians = SimpleNamespace(count=1_000)

    def reconstruct_sharp(path: Path, _output: Path) -> SimpleNamespace:
        calls.append(path)
        return gaussians

    def compile_splat(cloud: SimpleNamespace, output: Path) -> Path:
        output.write_bytes(b"\x00" * (cloud.count * 32))
        return output

    runtime = SimpleNamespace(
        reconstruct_sharp=reconstruct_sharp,
        reconstruct_sharp360=lambda *_args, **_kwargs: None,
        compile_splat=compile_splat,
        provenance={"lucidFrameRevision": "test-revision"},
    )
    checkpoint = {
        "filename": worker.OFFICIAL_SHARP_CHECKPOINT,
        "sha256": worker.OFFICIAL_SHARP_CHECKPOINT_SHA256,
        "verified": True,
    }
    monkeypatch.setattr(smart_connect, "reconstruct_connected", registration_failure)
    monkeypatch.setattr(worker, "_lucidframe_runtime", lambda: runtime)
    monkeypatch.setattr(
        worker,
        "_verify_official_sharp_checkpoint",
        lambda *, require_present: checkpoint,
    )

    worker._run_job(request, sources, fingerprints)

    completed = worker._load_state(request.job_id)
    assert completed is not None and completed.status == "ready"
    assert completed.fallback_used is True
    assert completed.registration_status == "partial"
    assert calls == [sources[0]]
    manifest_path = (
        cases_root
        / case_id
        / "reconstruction"
        / request.job_id
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest["fallbackUsed"] is True
    assert manifest["model"] == "Apple SHARP single-image fallback"
    assert manifest["registration"]["connectedFrames"] == [0]
