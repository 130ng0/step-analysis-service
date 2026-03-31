from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from profile_loader import ProfileLoaderError, select_profile_set
from resolve_profiles import ResolveProfilesError, resolve_profile_set

app = FastAPI(title="Orca Worker API", version="2.3.0")

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

    except (OrcaSliceError, ProfileLoaderError, ResolveProfilesError) as exc:
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
        output_dir = Path(tmpdir) / "out"
        output_dir.mkdir(parents=True, exist_ok=True)

        resolved_dir = resolve_profile_set(
            machine_name=selected["machine"],
            process_name=selected["process"],
            filament_name=selected["filament"],
            output_name="last-run",
            infill_percent=infill_percent,
            perimeter_count=perimeter_count,
            top_layers=top_layers,
            bottom_layers=bottom_layers,
        )

        printer_path = resolved_dir / "printer.json"
        process_path = resolved_dir / "process.json"
        filament_path = resolved_dir / "filament.json"

        debug_dir = Path("/workspace/debug-last")
        if debug_dir.exists():
            shutil.rmtree(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(printer_path, debug_dir / "printer.json")
        shutil.copy2(process_path, debug_dir / "process.json")
        shutil.copy2(filament_path, debug_dir / "filament.json")

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

            elif s.startswith("; estimated printing time (normal mode) ="):
                raw = s.split("=", 1)[1].strip()
                print_time_min = _parse_time_to_minutes(raw)

            elif re.match(r"^T\d+$", s):
                try:
                    current_tool = int(s[1:])
                except Exception:
                    pass

            elif s.startswith("G1") and " E" in s:
                m = re.search(r"(?:^|\s)E(-?\d+(?:\.\d+)?)", s)
                if m:
                    delta_e = float(m.group(1))
                    if delta_e > 0:
                        extrusion_per_tool_mm[current_tool] = extrusion_per_tool_mm.get(current_tool, 0.0) + delta_e

    total_weight_g = 0.0
    total_cost = 0.0
    filament_usage = []

    for tool_index in sorted(extrusion_per_tool_mm.keys()):
        extruded_len_mm = extrusion_per_tool_mm[tool_index]
        diameter_mm = (
            filament_diameter_list[tool_index]
            if tool_index < len(filament_diameter_list)
            else FILAMENT_DIAMETER_MM_DEFAULT
        )
        density_g_cm3 = filament_density_list[tool_index] if tool_index < len(filament_density_list) else None
        cost_per_kg = filament_cost_list[tool_index] if tool_index < len(filament_cost_list) else None
        material_name = filament_type_list[tool_index] if tool_index < len(filament_type_list) else f"tool_{tool_index}"

        radius_cm = (diameter_mm / 10.0) / 2.0
        length_cm = extruded_len_mm / 10.0
        volume_cm3 = math.pi * (radius_cm ** 2) * length_cm
        weight_g = volume_cm3 * density_g_cm3 if density_g_cm3 is not None else None
        cost = (weight_g / 1000.0) * cost_per_kg if (weight_g is not None and cost_per_kg is not None) else None

        if weight_g is not None:
            total_weight_g += weight_g
        if cost is not None:
            total_cost += cost

        filament_usage.append(
            {
                "tool": tool_index,
                "material": material_name,
                "extruded_length_mm": round(extruded_len_mm, 2),
                "estimated_weight_g": round(weight_g, 2) if weight_g is not None else None,
                "estimated_cost": round(cost, 2) if cost is not None else None,
            }
        )

    return {
        "estimated_print_time_min": print_time_min,
        "estimated_total_filament_g": round(total_weight_g, 2) if total_weight_g > 0 else None,
        "estimated_total_cost": round(total_cost, 2) if total_cost > 0 else None,
        "filament_usage_by_tool": filament_usage,
        "gcode_generated": True,
    }



def _parse_time_to_minutes(raw: str) -> int | None:
    total = 0
    matches = re.findall(r"(\d+)\s*([dhms])", raw.lower())
    if not matches:
        return None
    for value, unit in matches:
        n = int(value)
        if unit == "d":
            total += n * 24 * 60
        elif unit == "h":
            total += n * 60
        elif unit == "m":
            total += n
        elif unit == "s":
            total += math.ceil(n / 60)
    return total
