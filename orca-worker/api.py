from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Literal

import trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="Orca Worker API", version="1.2.0")

ORCA_PATH = "/opt/orca/squashfs-root/AppRun"
TEMPLATE_DIR = "/workspace/templates"
FILAMENT_DIAMETER_MM_DEFAULT = 1.75

TEMPLATE_MAP = {
    "abs": "abs_template.3mf",
    "abs_cf": "abs-cf_template.3mf",
    "abs_esd": "abs-esd_template.3mf",
    "asa": "asa_template.3mf",
    "pc": "pc_template.3mf",
    "pc_cf": "pc-cf_template.3mf",
    "pc_fr": "pc-fr_template.3mf",
    "tpu": "tpu_template.3mf",
}


class OrcaSliceError(Exception):
    pass


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/slice")
async def slice_model(
    file: UploadFile = File(...),
    material_profile: Literal["abs", "abs_cf", "abs_esd", "asa", "pc", "pc_cf", "pc_fr", "tpu"] = Form(default="abs"),
):
    filename = file.filename or "model.stl"
    suffix = os.path.splitext(filename)[1].lower()

    if suffix != ".stl":
        raise HTTPException(status_code=400, detail="Only STL files are currently supported for Orca slicing")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    tmp_input = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
            tmp.write(file_bytes)
            tmp_input = tmp.name

        result = run_orca_slice(tmp_input, material_profile)

        return {
            "success": True,
            "filename": filename,
            "method": "slice",
            "material_profile": material_profile,
            **result,
        }

    except OrcaSliceError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "SLICE_FAILED",
                "details": str(exc),
                "filename": filename,
            },
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "INTERNAL_SERVER_ERROR",
                "details": str(exc),
                "filename": filename,
            },
        )
    finally:
        if tmp_input and os.path.exists(tmp_input):
            try:
                os.unlink(tmp_input)
            except Exception:
                pass


def run_orca_slice(stl_path: str, material: str) -> Dict:
    if material not in TEMPLATE_MAP:
        raise OrcaSliceError(f"Unsupported material profile: {material}")

    template_file = os.path.join(TEMPLATE_DIR, TEMPLATE_MAP[material])
    if not os.path.exists(template_file):
        raise OrcaSliceError(f"Template not found: {template_file}")

    if not os.path.exists(stl_path):
        raise OrcaSliceError(f"STL not found: {stl_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "out")
        os.makedirs(output_dir, exist_ok=True)

        work_3mf = os.path.join(tmpdir, os.path.basename(template_file))
        shutil.copy2(template_file, work_3mf)

        inject_stl_into_3mf(work_3mf, stl_path)

        cmd = [
            ORCA_PATH,
            "--slice", "0",
            "--outputdir", output_dir,
            work_3mf,
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            raise OrcaSliceError(
                f"Orca failed with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

        gcode_path = os.path.join(output_dir, "plate_1.gcode")
        if not os.path.exists(gcode_path):
            raise OrcaSliceError("plate_1.gcode was not generated")

        return parse_gcode(gcode_path)


def inject_stl_into_3mf(template_3mf_path: str, stl_path: str) -> None:
    """
    Ersetzt in der Template-3MF die Datei 3D/3dmodel.model
    durch das Model aus der hochgeladenen STL.
    """
    temp_extract_dir = tempfile.mkdtemp(prefix="three_mf_extract_")
    temp_generated_dir = tempfile.mkdtemp(prefix="generated_3mf_")

    try:
        # 1) Template entpacken
        with zipfile.ZipFile(template_3mf_path, "r") as zf:
            zf.extractall(temp_extract_dir)

        extracted_template = Path(temp_extract_dir)
        target_model = extracted_template / "3D" / "3dmodel.model"

        if not target_model.exists():
            raise OrcaSliceError("Template 3MF does not contain 3D/3dmodel.model")

        # 2) Aus STL temporär eine 3MF erzeugen
        mesh = trimesh.load_mesh(stl_path, file_type="stl")
        if mesh is None or mesh.is_empty:
            raise OrcaSliceError("Uploaded STL could not be loaded")

        generated_3mf = Path(temp_generated_dir) / "generated.3mf"
        mesh.export(generated_3mf)

        # 3) Temporäre 3MF entpacken und ihr 3dmodel.model übernehmen
        generated_extract = Path(temp_generated_dir) / "unzipped"
        generated_extract.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(generated_3mf, "r") as zf:
            zf.extractall(generated_extract)

        generated_model = generated_extract / "3D" / "3dmodel.model"
        if not generated_model.exists():
            raise OrcaSliceError("Generated 3MF does not contain 3D/3dmodel.model")

        # 4) Model im Template ersetzen
        shutil.copy2(generated_model, target_model)

        # 5) Template neu packen
        rebuilt_3mf = template_3mf_path + ".rebuilt"
        with zipfile.ZipFile(rebuilt_3mf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in extracted_template.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(extracted_template).as_posix()
                    zf.write(file_path, arcname)

        os.replace(rebuilt_3mf, template_3mf_path)

    finally:
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
        shutil.rmtree(temp_generated_dir, ignore_errors=True)


def _split_csv_header_values(raw: str) -> List[str]:
    return [x.strip().strip('"') for x in raw.replace(";", ",").split(",") if x.strip()]


def _parse_header_list(raw: str, cast=float) -> List:
    values = []
    for item in _split_csv_header_values(raw):
        try:
            values.append(cast(item))
        except Exception:
            pass
    return values


def parse_gcode(gcode_path: str) -> Dict:
    current_tool = 0
    extrusion_per_tool_mm: Dict[int, float] = {}
    print_time_min = None

    filament_density_list: List[float] = []
    filament_diameter_list: List[float] = []
    filament_type_list: List[str] = []
    filament_cost_list: List[float] = []

    with open(gcode_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()

            if s.startswith("; filament_density:"):
                raw = s.split(":", 1)[1].strip()
                filament_density_list = _parse_header_list(raw, float)

            elif s.startswith("; filament_diameter:"):
                raw = s.split(":", 1)[1].strip()
                filament_diameter_list = _parse_header_list(raw, float)

            elif s.startswith("; filament_cost ="):
                raw = s.split("=", 1)[1].strip()
                filament_cost_list = _parse_header_list(raw, float)

            elif s.startswith("; filament_type ="):
                raw = s.split("=", 1)[1].strip()
                filament_type_list = _split_csv_header_values(raw)

            elif s.startswith("M73 ") and print_time_min is None:
                match = re.search(r"\bR(\d+)\b", s)
                if match:
                    print_time_min = int(match.group(1))

            elif re.fullmatch(r"T\d+", s):
                try:
                    current_tool = int(s[1:])
                except Exception:
                    current_tool = 0

            elif ("G0" in s or "G1" in s) and "E" in s:
                match = re.search(r"\bE(-?\d+(?:\.\d+)?)\b", s)
                if match:
                    e_val = float(match.group(1))
                    if e_val > 0:
                        extrusion_per_tool_mm[current_tool] = extrusion_per_tool_mm.get(current_tool, 0.0) + e_val

    if not filament_diameter_list:
        filament_diameter_list = [FILAMENT_DIAMETER_MM_DEFAULT]
    if not filament_density_list:
        filament_density_list = [1.0]

    def get_list_value(values: List, idx: int, default):
        if idx < len(values):
            return values[idx]
        if values:
            return values[0]
        return default

    per_tool = []
    total_length_mm = 0.0
    total_volume_mm3 = 0.0
    total_weight_g = 0.0
    total_material_cost_eur = 0.0

    for tool_idx in sorted(extrusion_per_tool_mm.keys()):
        length_mm = extrusion_per_tool_mm[tool_idx]
        diameter_mm = get_list_value(filament_diameter_list, tool_idx, FILAMENT_DIAMETER_MM_DEFAULT)
        density_g_cm3 = get_list_value(filament_density_list, tool_idx, 1.0)
        cost_per_kg_eur = get_list_value(filament_cost_list, tool_idx, 0.0)
        filament_type = get_list_value(filament_type_list, tool_idx, f"tool_{tool_idx}")

        radius_mm = diameter_mm / 2.0
        cross_section_mm2 = math.pi * (radius_mm ** 2)
        volume_mm3 = length_mm * cross_section_mm2
        volume_cm3 = volume_mm3 / 1000.0
        weight_g = volume_cm3 * density_g_cm3
        material_cost_eur = (weight_g / 1000.0) * cost_per_kg_eur

        total_length_mm += length_mm
        total_volume_mm3 += volume_mm3
        total_weight_g += weight_g
        total_material_cost_eur += material_cost_eur

        per_tool.append(
            {
                "tool": tool_idx,
                "filament_type": filament_type,
                "filament_density_g_cm3": round(density_g_cm3, 4),
                "filament_diameter_mm": round(diameter_mm, 4),
                "filament_cost_per_kg_eur": round(cost_per_kg_eur, 4),
                "filament_length_mm": round(length_mm, 3),
                "filament_volume_cm3": round(volume_cm3, 3),
                "filament_weight_g": round(weight_g, 3),
                "material_cost_eur": round(material_cost_eur, 2),
            }
        )

    return {
        "print_time_minutes": print_time_min or 0,
        "print_time_hours": round((print_time_min or 0) / 60.0, 4),
        "filament_length_mm_total": round(total_length_mm, 3),
        "filament_volume_cm3_total": round(total_volume_mm3 / 1000.0, 3),
        "filament_weight_g_total": round(total_weight_g, 3),
        "material_cost_eur_total": round(total_material_cost_eur, 2),
        "tools": per_tool,
    }