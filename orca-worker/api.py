from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from profile_loader import (
    ProfileLoaderError,
    build_minimal_filament_preset,
    build_minimal_printer_preset,
    build_minimal_process_preset,
    select_profile_set,
    write_json,
)

app = FastAPI(title="Orca Worker API", version="2.2.0")

ORCA_PATH = "/opt/orca/squashfs-root/AppRun"
FILAMENT_DIAMETER_MM_DEFAULT = 1.75


class OrcaSliceError(Exception):
    pass


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/slice")
async def slice_model(
    file: UploadFile = File(...),
    material_profile: Literal["abs", "abs_cf", "abs_esd", "asa", "pc", "pc_cf", "pc_fr", "tpu"] = Form(default="abs"),
    support_material_type: Literal["none", "breakaway"] = Form(default="none"),
    infill_percent: float = Form(default=20.0),
    perimeter_count: int = Form(default=5),
    top_layers: int = Form(default=5),
    bottom_layers: int = Form(default=5),
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

        result = run_orca_with_profiles(
            stl_path=tmp_input,
            material_profile=material_profile,
            support_material_type=support_material_type,
            infill_percent=infill_percent,
            perimeter_count=perimeter_count,
            top_layers=top_layers,
            bottom_layers=bottom_layers,
        )

        return {
            "success": True,
            "filename": filename,
            "method": "slice",
            "material_profile": material_profile,
            "support_material_type": support_material_type,
            "applied_slicer_settings": {
                "infill_percent": infill_percent,
                "perimeter_count": perimeter_count,
                "top_layers": top_layers,
                "bottom_layers": bottom_layers,
            },
            **result,
        }

    except (OrcaSliceError, ProfileLoaderError) as exc:
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


def run_orca_with_profiles(
    stl_path: str,
    material_profile: str,
    support_material_type: str,
    infill_percent: float,
    perimeter_count: int,
    top_layers: int,
    bottom_layers: int,
) -> Dict:
    selected = select_profile_set(material_profile, support_material_type)

    with tempfile.TemporaryDirectory() as tmpdir:
        printer_path = Path(tmpdir) / "printer.json"
        process_path = Path(tmpdir) / "process.json"
        filament_path = Path(tmpdir) / "filament.json"
        output_dir = Path(tmpdir) / "out"
        output_dir.mkdir(parents=True, exist_ok=True)

        printer = build_minimal_printer_preset(selected["machine"])
        process = build_minimal_process_preset(
            process_name=selected["process"],
            infill_percent=infill_percent,
            perimeter_count=perimeter_count,
            top_layers=top_layers,
            bottom_layers=bottom_layers,
        )
        filament = build_minimal_filament_preset(selected["filament"])

        write_json(printer_path, printer)
        write_json(process_path, process)
        write_json(filament_path, filament)

        debug_dir = Path("/workspace/debug-last")
        debug_dir.mkdir(parents=True, exist_ok=True)
        write_json(debug_dir / "printer.json", printer)
        write_json(debug_dir / "process.json", process)
        write_json(debug_dir / "filament.json", filament)

        cmd = [
            ORCA_PATH,
            "--load-settings", f"{printer_path};{process_path}",
            "--load-filaments", str(filament_path),
            "--outputdir", str(output_dir),
            "--slice", "0",
            stl_path,
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

        gcode_path = output_dir / "plate_1.gcode"
        if not gcode_path.exists():
            raise OrcaSliceError("plate_1.gcode was not generated")

        return parse_gcode(str(gcode_path))


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