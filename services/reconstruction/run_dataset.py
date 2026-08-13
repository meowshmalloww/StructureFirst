"""Run a reproducible full-resolution reconstruction evaluation dataset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from app import _lucidframe_runtime
from evaluate import evaluate_report

# Establish the same verified LucidFrame import boundary as the production
# worker before smart_connect imports LucidFrame's GaussianData type.
_RUNTIME = _lucidframe_runtime()

from smart_connect import RegistrationError, reconstruct_connected


def _load_manifest(path: Path) -> tuple[dict[str, Any], list[Path]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    root = (path.parent / str(manifest["inputRoot"])).resolve()
    images = [root / str(frame["file"]) for frame in manifest["frames"]]
    missing = [image for image in images if not image.is_file()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Dataset inputs are missing: {rendered}")
    return manifest, images


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compile-splat",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest, images = _load_manifest(manifest_path)

    failure: RegistrationError | None = None
    try:
        cloud, registration = reconstruct_connected(images, output)
    except RegistrationError as exc:
        failure = exc
        registration = exc.report
        cloud = exc.fallback_cloud

    registration["gaussianCount"] = int(cloud.count) if cloud is not None else 0
    registration_path = output / "registration.json"
    registration_path.write_text(
        json.dumps(registration, indent=2) + "\n",
        encoding="utf-8",
    )

    if cloud is not None and args.compile_splat:
        splat_path = output / "scene.splat"
        _RUNTIME.compile_splat(cloud, splat_path)
        expected_size = int(cloud.count) * 32
        if splat_path.stat().st_size != expected_size:
            raise RuntimeError(
                f"Splat size mismatch: expected {expected_size} bytes, "
                f"got {splat_path.stat().st_size}"
            )

    evaluation = evaluate_report(
        manifest,
        registration,
        manifest_path=manifest_path,
        verify_inputs=True,
    )
    (output / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evaluation, indent=2))
    if failure is not None:
        print(f"Registration failed: {failure}")
    return 0 if evaluation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
