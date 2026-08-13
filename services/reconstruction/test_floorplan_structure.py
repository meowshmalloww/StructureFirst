from pathlib import Path

import cv2
import numpy as np

import floorplan_structure
from floorplan_structure import build_floorplan_structure


def test_vectorizes_plan_as_structural_floors_not_gaussians(tmp_path: Path) -> None:
    plan = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.rectangle(plan, (70, 70), (570, 400), (25, 25, 25), 6)
    cv2.line(plan, (320, 70), (320, 400), (25, 25, 25), 6)
    cv2.line(plan, (70, 240), (570, 240), (25, 25, 25), 6)
    cv2.putText(
        plan,
        "1ST FLOOR PLAN",
        (190, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
    )
    source = tmp_path / "plan.png"
    assert cv2.imwrite(str(source), plan)

    output = tmp_path / "artifact"
    model = build_floorplan_structure(
        [source], ["evidence_plan0001"], [1], output, "case_floorplan0001"
    )

    assert model["metrics"]["floorCount"] == 1
    assert model["metrics"]["wallSegments"] >= 4
    assert model["floors"][0]["floorNumber"] == 1
    assert model["floors"][0]["sourceEvidenceId"] == "evidence_plan0001"
    assert model["roof"]["type"] == "unobserved"
    assert (output / "structure.json").is_file()
    assert (output / "floor_1_1_vectors.png").is_file()


def test_preserves_explicit_floor_order(tmp_path: Path) -> None:
    paths: list[Path] = []
    for index in range(2):
        plan = np.full((300, 400, 3), 255, dtype=np.uint8)
        cv2.rectangle(plan, (50, 50), (350, 250), (20, 20, 20), 5)
        path = tmp_path / f"plan-{index}.png"
        assert cv2.imwrite(str(path), plan)
        paths.append(path)

    model = build_floorplan_structure(
        paths,
        ["evidence_plan0004", "evidence_plan0002"],
        [4, 2],
        tmp_path / "ordered",
        "case_floorplan0002",
    )

    assert [floor["floorNumber"] for floor in model["floors"]] == [2, 4]
    assert [floor["elevationMeters"] for floor in model["floors"]] == [0.0, 6.1]
    assert model["alignment"]["method"] == "shared_document_coordinates"


def test_all_floors_share_one_horizontal_sheet_frame(tmp_path: Path) -> None:
    paths: list[Path] = []
    for index, top in enumerate((30, 90)):
        plan = np.full((360, 480, 3), 255, dtype=np.uint8)
        cv2.rectangle(plan, (80, top), (400, 300), (20, 20, 20), 7)
        path = tmp_path / f"shared-{index}.png"
        assert cv2.imwrite(str(path), plan)
        paths.append(path)

    output = tmp_path / "shared-output"
    model = build_floorplan_structure(
        paths,
        ["evidence_plan1001", "evidence_plan1002"],
        [1, 2],
        output,
        "case_floorplan1000",
    )

    first, second = model["floors"]
    assert first["textureCrop"] == second["textureCrop"]
    assert first["widthMeters"] == second["widthMeters"]
    assert first["depthMeters"] == second["depthMeters"]
    assert first["elevationMeters"] == 0.0
    assert second["elevationMeters"] == 3.05
    assert (output / "floor_1_1_overlay.png").is_file()
    assert (output / "floor_1_1_source.png").is_file()
    assert (output / "floor_2_2_overlay.png").is_file()
    assert (output / "floor_2_2_source.png").is_file()


def test_preserves_non_orthogonal_plan_geometry_without_inventing_a_roof(
    tmp_path: Path,
) -> None:
    plan = np.full((520, 700, 3), 255, dtype=np.uint8)
    polygon = np.asarray(
        [[80, 110], [430, 70], [620, 210], [560, 440], [180, 455]],
        dtype=np.int32,
    )
    cv2.polylines(plan, [polygon], True, (15, 15, 15), 9)
    cv2.line(plan, (430, 70), (390, 430), (15, 15, 15), 8)
    source = tmp_path / "angled-plan.png"
    assert cv2.imwrite(str(source), plan)

    model = build_floorplan_structure(
        [source], ["evidence_angled0001"], [1], tmp_path / "angled", "case_angled0001"
    )

    walls = model["floors"][0]["walls"]
    assert any(
        abs(wall["end"][0] - wall["start"][0]) > 0.2
        and abs(wall["end"][1] - wall["start"][1]) > 0.2
        for wall in walls
    )
    assert len(model["floors"][0]["footprint"]) > 4
    assert model["roof"]["type"] == "unobserved"


def test_splits_two_labeled_level_drawings_on_one_sheet(
    tmp_path: Path, monkeypatch
) -> None:
    plan = np.full((900, 700, 3), 255, dtype=np.uint8)
    cv2.rectangle(plan, (70, 110), (310, 720), (30, 30, 30), -1)
    cv2.rectangle(plan, (390, 160), (630, 500), (30, 30, 30), -1)
    source = tmp_path / "two-level-sheet.png"
    assert cv2.imwrite(str(source), plan)
    entries = [
        _ocr_entry("MAIN LEVEL", 170, 760),
        _ocr_entry("Living Room", 170, 500),
        _ocr_entry("UPPER LEVEL", 500, 540),
        _ocr_entry("Bedroom 2", 500, 330),
    ]
    monkeypatch.setattr(floorplan_structure, "_ocr_entries", lambda _: entries)

    output = tmp_path / "split"
    model = build_floorplan_structure(
        [source], ["evidence_sheet0001"], [1], output, "case_sheet0001"
    )

    assert model["metrics"]["floorCount"] == 2
    assert [floor["label"] for floor in model["floors"]] == [
        "Main level",
        "Upper level",
    ]
    assert model["alignment"]["method"] == "independent_observed_panels"
    assert model["floors"][0]["sourceImageCrop"] != model["floors"][1]["sourceImageCrop"]
    assert model["floors"][0]["sourceEvidenceId"] == "evidence_sheet0001"


def test_keeps_site_plan_as_reference_without_inventing_a_floor(
    tmp_path: Path, monkeypatch
) -> None:
    plan = np.full((500, 700, 3), 255, dtype=np.uint8)
    cv2.rectangle(plan, (80, 70), (620, 420), (40, 40, 40), 6)
    source = tmp_path / "site-plan.png"
    assert cv2.imwrite(str(source), plan)
    monkeypatch.setattr(
        floorplan_structure,
        "_ocr_entries",
        lambda _: [_ocr_entry("SITE PLAN", 540, 450)],
    )

    model = build_floorplan_structure(
        [source], ["evidence_site0001"], [1], tmp_path / "site", "case_site0001"
    )

    assert model["metrics"]["floorCount"] == 0
    assert model["floors"] == []
    assert model["referencePlans"][0]["kind"] == "site_plan"
    assert model["referencePlans"][0]["structuralFloor"] is False


def test_incomplete_room_topology_stays_source_plan_first(
    tmp_path: Path, monkeypatch
) -> None:
    plan = np.full((420, 640, 3), 255, dtype=np.uint8)
    cv2.rectangle(plan, (60, 60), (580, 360), (20, 20, 20), 8)
    cv2.line(plan, (320, 60), (320, 180), (20, 20, 20), 8)
    cv2.line(plan, (320, 240), (320, 360), (20, 20, 20), 8)
    source = tmp_path / "open-door-plan.png"
    assert cv2.imwrite(str(source), plan)
    monkeypatch.setattr(
        floorplan_structure,
        "_ocr_entries",
        lambda _: [
            _ocr_entry("Living Room", 180, 200),
            _ocr_entry("Bedroom", 460, 200),
        ],
    )

    model = build_floorplan_structure(
        [source],
        ["evidence_opening0001"],
        [1],
        tmp_path / "source-first",
        "case_sourcefirst0001",
    )
    floor = model["floors"][0]

    assert model["schemaVersion"] == 5
    assert floor["vectorization"]["sourcePlanAuthoritative"] is True
    assert floor["vectorization"]["roomTopologyVerified"] is False
    assert floor["spaceGraph"]["verified"] is False
    assert len(floor["spaceGraph"]["nodes"]) == 2
    assert len(floor["spaceGraph"]["edges"]) == 1


def _ocr_entry(text: str, x: float, y: float) -> dict[str, object]:
    return {
        "text": text,
        "score": 0.99,
        "source": "plan_ocr",
        "center": [x, y],
        "box": [x - 35, y - 10, x + 35, y + 10],
    }
    assert model["alignment"]["metricScaleVerified"] is False
