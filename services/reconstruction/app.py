"""Local RTX reconstruction boundary for StructureFirst.

This service deliberately imports only LucidFrame's reconstruction engine. It does
not reuse LucidFrame's UI, case model, or application server. Jobs are serialized
because Apple SHARP is GPU-heavy and LucidFrame already protects its cached model
with an inference lock.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from dotenv import load_dotenv

LOGGER = logging.getLogger("structurefirst.reconstruction")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

SERVICE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SERVICE_ROOT.parent.parent
load_dotenv(REPO_ROOT / ".env")
DATA_ROOT = Path(os.getenv("STRUCTUREFIRST_DATA_DIR", str(REPO_ROOT / "data"))).resolve()
CASES_ROOT = (DATA_ROOT / "cases").resolve()
LUCIDFRAME_ROOT = Path(
    os.getenv("LUCIDFRAME_ROOT", str(REPO_ROOT.parent / "LucidFrame"))
).resolve()
LUCIDFRAME_BACKEND = LUCIDFRAME_ROOT / "backend"
PANORAMA_QUALITY = os.getenv("STRUCTUREFIRST_PANORAMA_QUALITY", "detail")
if PANORAMA_QUALITY not in {"balanced", "detail"}:
    PANORAMA_QUALITY = "detail"

ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
OFFICIAL_SHARP_MODEL_URL = (
    "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
)
OFFICIAL_SHARP_CHECKPOINT = "sharp_2572gikvuh.pt"
OFFICIAL_SHARP_CHECKPOINT_SHA256 = (
    "94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d"
)
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="structurefirst-gpu")
JOBS: dict[str, "JobState"] = {}
JOBS_LOCK = threading.Lock()
CHECKPOINT_LOCK = threading.Lock()
CHECKPOINT_VERIFICATION: tuple[int, int, str] | None = None


def now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    case_id: str
    evidence_id: str
    input_path: str = Field(min_length=1, max_length=4096)
    input_sha256: str = Field(min_length=64, max_length=64)
    evidence_ids: list[str] | None = Field(default=None, min_length=2, max_length=12)
    input_paths: list[str] | None = Field(default=None, min_length=2, max_length=12)
    input_sha256s: list[str] | None = Field(default=None, min_length=2, max_length=12)
    mode: Literal["single_image", "panorama", "multi_image"] = "single_image"

    @field_validator("job_id", "case_id", "evidence_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not ID_PATTERN.fullmatch(value):
            raise ValueError("identifier contains unsupported characters")
        return value

    @field_validator("evidence_ids")
    @classmethod
    def validate_identifiers(cls, values: list[str] | None) -> list[str] | None:
        if values is not None and any(not ID_PATTERN.fullmatch(value) for value in values):
            raise ValueError("identifier contains unsupported characters")
        return values

    @field_validator("input_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("input_sha256 must be a 64-character SHA-256 digest")
        return normalized

    @field_validator("input_sha256s")
    @classmethod
    def validate_sha256s(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        normalized = [value.lower() for value in values]
        if any(not SHA256_PATTERN.fullmatch(value) for value in normalized):
            raise ValueError("every input SHA-256 must be a 64-character digest")
        return normalized

    @model_validator(mode="after")
    def validate_multi_input(self) -> "JobRequest":
        if self.mode == "multi_image":
            if not self.input_paths or not self.evidence_ids or not self.input_sha256s:
                raise ValueError(
                    "multi-image jobs require input_paths, input_sha256s, and evidence_ids"
                )
            if not (
                len(self.input_paths)
                == len(self.evidence_ids)
                == len(self.input_sha256s)
            ):
                raise ValueError(
                    "multi-image path, hash, and evidence counts must match"
                )
            if self.input_sha256 != self.input_sha256s[0]:
                raise ValueError("input_sha256 must match the first multi-image hash")
        elif self.input_sha256s is not None:
            raise ValueError("input_sha256s is only valid for multi-image jobs")
        return self


@dataclass(frozen=True)
class InputFingerprint:
    sha256: str
    byte_size: int


@dataclass(frozen=True)
class LucidFrameRuntime:
    reconstruct_sharp: Callable[..., Any]
    reconstruct_sharp360: Callable[..., Any]
    compile_splat: Callable[..., Path]
    is_sharp_available: Callable[[], bool]
    provenance: dict[str, object]


class JobState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    case_id: str
    evidence_id: str
    status: Literal["queued", "running", "ready", "failed"]
    mode: Literal["single_image", "panorama", "multi_image"]
    created_at: str
    updated_at: str
    splat_url: str | None = None
    manifest_url: str | None = None
    gaussian_count: int | None = None
    registration_report_url: str | None = None
    registration_status: Literal["connected", "partial", "failed"] | None = None
    connected_frame_count: int | None = None
    frame_count: int | None = None
    registration_confidence: float | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    error: str | None = None


app = FastAPI(
    title="StructureFirst Local Reconstruction",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
)


def _ensure_inside(candidate: Path, root: Path) -> Path:
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes the allowed case directory") from exc
    return resolved


def _validated_input(request: JobRequest) -> Path:
    case_upload_root = (CASES_ROOT / request.case_id / "uploads").resolve()
    path = _ensure_inside(Path(request.input_path), case_upload_root)
    if not path.is_file():
        raise ValueError("input image does not exist")
    if path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        raise ValueError("input must be JPEG, PNG, or WebP")
    return path


def _validated_inputs(
    request: JobRequest,
) -> tuple[list[Path], list[InputFingerprint]]:
    if request.mode != "multi_image":
        paths = [_validated_input(request)]
        expected_hashes = [request.input_sha256]
    else:
        assert request.input_paths is not None
        assert request.input_sha256s is not None
        paths = []
        case_upload_root = (CASES_ROOT / request.case_id / "uploads").resolve()
        for raw_path in request.input_paths:
            path = _ensure_inside(Path(raw_path), case_upload_root)
            if not path.is_file():
                raise ValueError("input image does not exist")
            if path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
                raise ValueError("input must be JPEG, PNG, or WebP")
            paths.append(path)
        if len({str(path) for path in paths}) != len(paths):
            raise ValueError("multi-image inputs must be unique")
        expected_hashes = request.input_sha256s

    fingerprints = [_fingerprint(path) for path in paths]
    for fingerprint, expected in zip(fingerprints, expected_hashes, strict=True):
        if fingerprint.sha256 != expected:
            raise ValueError(
                "input image bytes no longer match the stored evidence SHA-256"
            )
    return paths, fingerprints


def _job_directory(case_id: str, job_id: str) -> Path:
    return _ensure_inside(
        CASES_ROOT / case_id / "reconstruction" / job_id,
        CASES_ROOT,
    )


def _state_path(case_id: str, job_id: str) -> Path:
    return _job_directory(case_id, job_id) / "job.json"


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _save_state(state: JobState) -> None:
    with JOBS_LOCK:
        JOBS[state.job_id] = state
    _write_json_atomic(_state_path(state.case_id, state.job_id), state.model_dump())


def _load_state(job_id: str) -> JobState | None:
    with JOBS_LOCK:
        in_memory = JOBS.get(job_id)
    if in_memory:
        return in_memory
    if not ID_PATTERN.fullmatch(job_id) or not CASES_ROOT.exists():
        return None
    for candidate in CASES_ROOT.glob(f"*/reconstruction/{job_id}/job.json"):
        try:
            state = JobState.model_validate_json(candidate.read_text(encoding="utf-8"))
            with JOBS_LOCK:
                JOBS[job_id] = state
            return state
        except Exception:
            LOGGER.exception("Could not read persisted job %s", candidate)
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path: Path) -> InputFingerprint:
    return InputFingerprint(sha256=_sha256(path), byte_size=path.stat().st_size)


def _import_exact_lucidframe_module(name: str) -> ModuleType:
    if not LUCIDFRAME_BACKEND.is_dir():
        raise RuntimeError(f"LucidFrame backend was not found at {LUCIDFRAME_BACKEND}")
    backend_text = str(LUCIDFRAME_BACKEND)
    sys.path[:] = [entry for entry in sys.path if entry != backend_text]
    sys.path.insert(0, backend_text)
    module = importlib.import_module(name)
    module_path = Path(str(module.__file__)).resolve()
    expected_path = (LUCIDFRAME_BACKEND / f"{name}.py").resolve()
    if module_path != expected_path:
        raise RuntimeError(
            f"Refusing non-LucidFrame module {name}: loaded {module_path}, "
            f"expected {expected_path}"
        )
    return module


def _read_lucidframe_revision() -> str | None:
    git_directory = LUCIDFRAME_ROOT / ".git"
    if git_directory.is_file():
        pointer = git_directory.read_text("utf-8").strip()
        if pointer.startswith("gitdir:"):
            git_directory = (LUCIDFRAME_ROOT / pointer.split(":", 1)[1].strip()).resolve()
    head_path = git_directory / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text("utf-8").strip()
    if not head.startswith("ref:"):
        return head if re.fullmatch(r"[a-f0-9]{40}", head) else None
    reference = head.split(":", 1)[1].strip()
    reference_path = git_directory / reference
    if reference_path.is_file():
        revision = reference_path.read_text("utf-8").strip()
        return revision if re.fullmatch(r"[a-f0-9]{40}", revision) else None
    packed_refs = git_directory / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text("utf-8").splitlines():
            if line.endswith(f" {reference}"):
                revision = line.split(" ", 1)[0]
                return revision if re.fullmatch(r"[a-f0-9]{40}", revision) else None
    return None


def _verify_official_sharp_checkpoint(*, require_present: bool) -> dict[str, object]:
    import torch

    checkpoint_path = (
        Path(torch.hub.get_dir()) / "checkpoints" / OFFICIAL_SHARP_CHECKPOINT
    ).resolve()
    if not checkpoint_path.is_file():
        if require_present:
            raise RuntimeError(
                "The official Apple SHARP checkpoint was not cached after model loading"
            )
        return {
            "filename": OFFICIAL_SHARP_CHECKPOINT,
            "expectedSha256": OFFICIAL_SHARP_CHECKPOINT_SHA256,
            "present": False,
            "verified": False,
        }

    global CHECKPOINT_VERIFICATION
    with CHECKPOINT_LOCK:
        before = checkpoint_path.stat()
        cached = CHECKPOINT_VERIFICATION
        if cached and cached[:2] == (before.st_mtime_ns, before.st_size):
            actual_sha256 = cached[2]
        else:
            actual_sha256 = _sha256(checkpoint_path)
            after = checkpoint_path.stat()
            if (before.st_mtime_ns, before.st_size) != (
                after.st_mtime_ns,
                after.st_size,
            ):
                raise RuntimeError("Apple SHARP checkpoint changed during verification")
            CHECKPOINT_VERIFICATION = (
                after.st_mtime_ns,
                after.st_size,
                actual_sha256,
            )
    if actual_sha256 != OFFICIAL_SHARP_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Apple SHARP checkpoint SHA-256 does not match the official release"
        )
    return {
        "filename": OFFICIAL_SHARP_CHECKPOINT,
        "byteSize": checkpoint_path.stat().st_size,
        "sha256": actual_sha256,
        "expectedSha256": OFFICIAL_SHARP_CHECKPOINT_SHA256,
        "present": True,
        "verified": True,
    }


def _lucidframe_runtime() -> LucidFrameRuntime:
    sharp_wrapper = _import_exact_lucidframe_module("sharp_wrapper")
    sharp360_wrapper = _import_exact_lucidframe_module("sharp360_wrapper")
    splat_compiler = _import_exact_lucidframe_module("splat_compiler")
    sharp_predict = importlib.import_module("sharp.cli.predict")
    sharp_predict_path = Path(str(sharp_predict.__file__)).resolve()
    try:
        sharp_predict_relative = sharp_predict_path.relative_to(LUCIDFRAME_ROOT)
    except ValueError as exc:
        raise RuntimeError(
            "Apple SHARP resolved outside the configured LucidFrame repository: "
            f"{sharp_predict_path}"
        ) from exc
    model_url = str(getattr(sharp_predict, "DEFAULT_MODEL_URL", ""))
    if model_url != OFFICIAL_SHARP_MODEL_URL:
        raise RuntimeError(
            "LucidFrame is not configured for the pinned official Apple SHARP model"
        )

    module_paths = {
        "backend/sharp_wrapper.py": Path(str(sharp_wrapper.__file__)).resolve(),
        "backend/sharp360_wrapper.py": Path(str(sharp360_wrapper.__file__)).resolve(),
        "backend/splat_compiler.py": Path(str(splat_compiler.__file__)).resolve(),
    }
    provenance: dict[str, object] = {
        "lucidFrameRevision": _read_lucidframe_revision(),
        "backendModuleSha256": {
            name: _sha256(path) for name, path in module_paths.items()
        },
        "sharpImplementation": sharp_predict_relative.as_posix(),
        "sharpModelUrl": model_url,
        "sharpCheckpointSha256": OFFICIAL_SHARP_CHECKPOINT_SHA256,
    }

    return LucidFrameRuntime(
        reconstruct_sharp=getattr(sharp_wrapper, "reconstruct_sharp"),
        reconstruct_sharp360=getattr(sharp360_wrapper, "reconstruct_sharp360"),
        compile_splat=getattr(splat_compiler, "compile_splat"),
        is_sharp_available=getattr(sharp_wrapper, "is_available"),
        provenance=provenance,
    )


def _run_job(
    request: JobRequest,
    input_paths: list[Path],
    input_fingerprints: list[InputFingerprint],
) -> None:
    state = _load_state(request.job_id)
    if not state:
        return
    running = state.model_copy(update={"status": "running", "updated_at": now_iso()})
    _save_state(running)
    output_directory = _job_directory(request.case_id, request.job_id)
    output_directory.mkdir(parents=True, exist_ok=True)

    try:
        runtime = _lucidframe_runtime()
        _verify_official_sharp_checkpoint(require_present=False)
        registration: dict[str, object] | None = None
        fallback_used = False
        fallback_reason: str | None = None
        if request.mode == "multi_image":
            from smart_connect import RegistrationError, reconstruct_connected

            try:
                gaussians, registration = reconstruct_connected(
                    input_paths,
                    output_directory,
                )
            except RegistrationError as exc:
                fallback_used = True
                fallback_reason = str(exc)
                registration = {
                    **exc.report,
                    "status": "partial",
                    "connectedFrameCount": 1,
                    "connectedFrames": [0],
                    "disconnectedFrames": list(range(1, len(input_paths))),
                    "confidenceScore": 0.18,
                    "fallbackUsed": True,
                    "fallbackReason": fallback_reason,
                }
                LOGGER.warning(
                    "Smart connect failed for %s; producing the exact LucidFrame "
                    "single-image result from the first capture instead: %s",
                    request.job_id,
                    fallback_reason,
                )
                gaussians = exc.fallback_cloud or runtime.reconstruct_sharp(
                    input_paths[0],
                    output_directory / "single_image_fallback",
                )
            _write_json_atomic(output_directory / "registration.json", registration)
        elif request.mode == "panorama":
            gaussians = runtime.reconstruct_sharp360(
                input_paths[0],
                output_directory,
                quality_profile=PANORAMA_QUALITY,
            )
        else:
            gaussians = runtime.reconstruct_sharp(input_paths[0], output_directory)

        checkpoint = _verify_official_sharp_checkpoint(require_present=True)
        final_fingerprints = [_fingerprint(path) for path in input_paths]
        if final_fingerprints != input_fingerprints:
            raise RuntimeError(
                "Source image bytes changed while LucidFrame was reconstructing"
            )

        gaussian_count = int(gaussians.count)
        if gaussian_count < 1_000:
            raise RuntimeError("LucidFrame produced too few valid Gaussians")

        temporary_splat = output_directory / "scene.splat.partial"
        final_splat = output_directory / "scene.splat"
        runtime.compile_splat(gaussians, temporary_splat)
        expected_bytes = gaussian_count * 32
        actual_bytes = temporary_splat.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"Splat validation failed: expected {expected_bytes} bytes, got {actual_bytes}"
            )
        temporary_splat.replace(final_splat)

        manifest_confidence = (
            0.52
            if fallback_used
            else
            round(0.45 + 0.23 * float(registration["confidenceScore"]), 2)
            if registration
            else 0.58
        )
        limitations = [
            "A Gaussian splat is a visual evidence layer, not a traversable structure graph."
        ]
        if request.mode == "multi_image":
            limitations.append(
                "Only photographs joined by measured overlap are included; occluded space remains unknown."
            )
            if fallback_used:
                limitations.append(
                    "The selected photographs did not register, so this artifact uses only the first source photograph."
                )
        elif request.mode == "panorama":
            limitations.append(
                "Panorama output cannot establish rooms or surfaces absent from the source panorama."
            )
        else:
            limitations.append(
                "Single-image output supports nearby views and cannot establish occluded geometry."
            )
        limitations.append(
            "The operator must verify address relevance and current conditions."
        )
        pipeline_entrypoint = (
            "services/reconstruction/smart_connect.py::reconstruct_connected -> "
            "LucidFrame/backend/sharp_wrapper.py::reconstruct_sharp (single-image fallback)"
            if fallback_used
            else
            "services/reconstruction/smart_connect.py::reconstruct_connected -> "
            "LucidFrame/backend/sharp_wrapper.py::reconstruct_sharp"
            if request.mode == "multi_image"
            else "LucidFrame/backend/sharp360_wrapper.py::reconstruct_sharp360"
            if request.mode == "panorama"
            else "LucidFrame/backend/sharp_wrapper.py::reconstruct_sharp"
        )
        evidence_ids = request.evidence_ids or [request.evidence_id]
        input_files = [
            {
                "evidenceId": evidence_id,
                "filename": path.name,
                "byteSize": fingerprint.byte_size,
                "sha256": fingerprint.sha256,
            }
            for evidence_id, path, fingerprint in zip(
                evidence_ids,
                input_paths,
                input_fingerprints,
                strict=True,
            )
        ]

        manifest = {
            "schemaVersion": 1,
            "artifactId": request.job_id,
            "caseId": request.case_id,
            "evidenceId": request.evidence_id,
            "evidenceIds": evidence_ids,
            "createdAt": now_iso(),
            "adapter": "StructureFirst LucidFrame boundary 0.1.0",
            "engine": "LucidFrame",
            "model": (
                "Apple SHARP single-image fallback"
                if fallback_used
                else
                "LucidFrame SHARP smart connect"
                if request.mode == "multi_image"
                else "LucidFrame SHARP-360"
                if request.mode == "panorama"
                else "Apple SHARP"
            ),
            "modelLicense": (
                "Apple Machine Learning Research Model License "
                "(non-commercial research only)"
            ),
            "codeLicense": "LucidFrame code: Apache-2.0",
            "mode": request.mode,
            "qualityProfile": PANORAMA_QUALITY if request.mode == "panorama" else "1536-fixed",
            "pipelineEntrypoint": pipeline_entrypoint,
            "sourceFileDelivery": (
                "Original stored file path passed directly to LucidFrame; "
                "decoding and required 1536x1536 preprocessing occur inside SHARP."
            ),
            "fallbackUsed": fallback_used,
            "fallbackReason": fallback_reason,
            "inputFiles": input_files,
            "inputSha256": [item.sha256 for item in input_fingerprints]
            if request.mode == "multi_image"
            else input_fingerprints[0].sha256,
            "runtimeProvenance": {
                **runtime.provenance,
                "checkpoint": checkpoint,
            },
            "gaussianCount": gaussian_count,
            "splatBytes": actual_bytes,
            "coordinateSystem": "OpenCV camera axes: +X right, +Y down, +Z forward",
            "confidence": {
                "band": "reconstructed",
                "state": "derived",
                "score": manifest_confidence,
            },
            "limitations": limitations,
            **({"registration": registration} if registration else {}),
        }
        manifest_path = output_directory / "manifest.json"
        _write_json_atomic(manifest_path, manifest)
        base_url = f"/assets/{request.case_id}/reconstruction/{request.job_id}"
        registration_url = (
            f"{base_url}/registration.json" if registration is not None else None
        )
        ready = running.model_copy(
            update={
                "status": "ready",
                "updated_at": now_iso(),
                "splat_url": f"{base_url}/scene.splat",
                "manifest_url": f"{base_url}/manifest.json",
                "gaussian_count": gaussian_count,
                "registration_report_url": registration_url,
                "registration_status": registration.get("status") if registration else None,
                "connected_frame_count": registration.get("connectedFrameCount") if registration else None,
                "frame_count": registration.get("frameCount") if registration else None,
                "registration_confidence": registration.get("confidenceScore") if registration else None,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
            }
        )
        _save_state(ready)
        LOGGER.info("Reconstruction %s completed with %d Gaussians", request.job_id, gaussian_count)
    except Exception as exc:
        LOGGER.exception("Reconstruction %s failed", request.job_id)
        failed_update: dict[str, object] = {
            "status": "failed",
            "updated_at": now_iso(),
            "error": str(exc)[:2000],
        }
        registration_path = output_directory / "registration.json"
        if registration_path.is_file():
            try:
                registration = json.loads(registration_path.read_text("utf-8"))
                failed_update.update(
                    {
                        "registration_report_url": (
                            f"/assets/{request.case_id}/reconstruction/"
                            f"{request.job_id}/registration.json"
                        ),
                        "registration_status": registration.get("status", "failed"),
                        "connected_frame_count": registration.get(
                            "connectedFrameCount", 1
                        ),
                        "frame_count": registration.get("frameCount", 1),
                        "registration_confidence": registration.get(
                            "confidenceScore", 0.0
                        ),
                    }
                )
            except Exception:
                LOGGER.warning("Could not attach registration failure report", exc_info=True)
        failed = running.model_copy(
            update=failed_update
        )
        _save_state(failed)


@app.get("/health")
def health() -> dict[str, object]:
    cuda_available = False
    lucidframe_available = False
    sharp_available = False
    runtime: LucidFrameRuntime | None = None
    checkpoint: dict[str, object] = {
        "filename": OFFICIAL_SHARP_CHECKPOINT,
        "expectedSha256": OFFICIAL_SHARP_CHECKPOINT_SHA256,
        "present": False,
        "verified": False,
    }
    runtime_error: str | None = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        runtime = _lucidframe_runtime()
        lucidframe_available = True
        sharp_available = bool(runtime.is_sharp_available())
        checkpoint = _verify_official_sharp_checkpoint(require_present=False)
    except Exception as exc:
        runtime_error = str(exc)[:1000]
        LOGGER.debug("LucidFrame health probe failed", exc_info=True)

    checkpoint_verified = bool(checkpoint.get("verified"))

    return {
        "status": (
            "ready"
            if sharp_available and checkpoint_verified and runtime is not None
            else "degraded"
        ),
        "engine": "LucidFrame Apple SHARP",
        "gpu_available": cuda_available,
        "lucidframe_available": lucidframe_available,
        "sharp_available": sharp_available,
        "sharp_checkpoint_verified": checkpoint_verified,
        "sharp_checkpoint": checkpoint,
        "fallback_enabled": True,
        "runtime_provenance": runtime.provenance if runtime else None,
        "runtime_error": runtime_error,
        "panorama_quality": PANORAMA_QUALITY,
        "jobs_running": sum(
            1 for state in JOBS.values() if state.status in {"queued", "running"}
        ),
    }


@app.post("/jobs", response_model=JobState, status_code=status.HTTP_202_ACCEPTED)
def create_job(request: JobRequest) -> JobState:
    try:
        input_paths, input_fingerprints = _validated_inputs(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    existing = _load_state(request.job_id)
    if existing:
        if existing.case_id != request.case_id or existing.evidence_id != request.evidence_id:
            raise HTTPException(status_code=409, detail="job identifier is already in use")
        return existing

    created = now_iso()
    state = JobState(
        job_id=request.job_id,
        case_id=request.case_id,
        evidence_id=request.evidence_id,
        status="queued",
        mode=request.mode,
        created_at=created,
        updated_at=created,
    )
    _save_state(state)
    EXECUTOR.submit(_run_job, request, input_paths, input_fingerprints)
    return state


@app.get("/jobs/{job_id}", response_model=JobState)
def get_job(job_id: str) -> JobState:
    if not ID_PATTERN.fullmatch(job_id):
        raise HTTPException(status_code=400, detail="invalid job identifier")
    state = _load_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="job not found")
    return state
