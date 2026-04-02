from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from profile_loader import ProfileLoaderError, select_profile_set
from resolve_profiles import ResolveProfilesError, resolve_profile_set

app = FastAPI(title="Orca Worker API", version="2.3.0")

KEEP_TMP = os.getenv("KEEP_TMP", "false").lower() == "true"
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
    support_material_type: Literal["none", "breakaway", "hips", "soluble"] = Form(default="none"),
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


def load_filament_metadata(filament_json_path: Path) -> Dict:
    import json

    with open(filament_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def first_number(key: str):
        value = data.get(key)
        if isinstance(value, list) and value:
            try:
                return float(str(value[0]).replace(",", "."))
            except Exception:
                return None
        if value is not None:
            try:
                return float(str(value).replace(",", "."))
            except Exception:
                return None
        return None

    def first_text(key: str):
        value = data.get(key)
        if isinstance(value, list) and value:
            return str(value[0])
        if value is not None:
            return str(value)
        return None

    filament_type = first_text("filament_type")
    filament_settings_id = first_text("filament_settings_id")

    density = first_number("filament_density")
    diameter = first_number("filament_diameter")
    cost = first_number("filament_cost")

    # Fallback-Dichten, falls im Profil nichts gepflegt ist
    density_fallback_map = {
        "ABS": 1.04,
        "ABS PRO": 1.04,
        "ABS-CF": 1.10,
        "ABS-CF PRO": 1.10,
        "ABS-ESD": 1.08,
        "ABS-ESD PRO": 1.08,
        "ASA": 1.07,
        "ASA PRO": 1.07,
        "PC": 1.20,
        "PC PRO": 1.20,
        "PC-CF": 1.18,
        "PC-CF PRO": 1.18,
        "PC-FR": 1.20,
        "TPU": 1.21,
        "HIPS": 1.03,
        "NEVO SOLUBLE": 1.20,
    }

    if (density is None or density <= 0) and filament_type:
        density = density_fallback_map.get(filament_type.strip().upper())

    if diameter is None or diameter <= 0:
        diameter = FILAMENT_DIAMETER_MM_DEFAULT

    return {
        "filament_density_g_cm3": density,
        "filament_diameter_mm": diameter,
        "filament_cost_eur_per_kg": cost,
        "filament_type": filament_type,
        "filament_settings_id": filament_settings_id,
        "filament_vendor": first_text("filament_vendor"),
    }

    def first_text(key: str):
        value = data.get(key)
        if isinstance(value, list) and value:
            return str(value[0])
        if value is not None:
            return str(value)
        return None

    return {
        "filament_density_g_cm3": first_number("filament_density"),
        "filament_diameter_mm": first_number("filament_diameter"),
        "filament_cost_eur_per_kg": first_number("filament_cost"),
        "filament_type": first_text("filament_type"),
        "filament_settings_id": first_text("filament_settings_id"),
        "filament_vendor": first_text("filament_vendor"),
    }


def load_named_filament_metadata(profile_name: str) -> Dict:
    profiles_root = Path("/workspace/profiles")

    wanted = profile_name.strip().lower()

    for path in profiles_root.rglob("*.json"):
        if path.stem.strip().lower() == wanted:
            return load_filament_metadata(path)

        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if str(data.get("name", "")).strip().lower() == wanted:
                return load_filament_metadata(path)
        except Exception:
            pass

    raise FileNotFoundError(f"Support pricing filament profile not found: {profile_name}")


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

    tmpdir = tempfile.mkdtemp(prefix="orca-run-")
    tmpdir_path = Path(tmpdir)
    output_dir = tmpdir_path / "out"
    profiles_dir = tmpdir_path / "profiles"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        profiles_dir.mkdir(parents=True, exist_ok=True)

        resolved_dir = resolve_profile_set(
            machine_name=selected["machine"],
            process_name=selected["process"],
            filament_name=selected["filament"],
            output_name="profiles",
            support_material_type=support_material_type,
            infill_percent=infill_percent,
            perimeter_count=perimeter_count,
            top_layers=top_layers,
            bottom_layers=bottom_layers,
            out_base=tmpdir_path,
        )

        printer_path = resolved_dir / "printer.json"
        process_path = resolved_dir / "process.json"
        filament_path = resolved_dir / "filament.json"

        filament_metadata = load_filament_metadata(filament_path)
        support_filament_metadata = None
        support_price_filament = selected.get("support_price_filament")

        if support_price_filament:
            support_filament_metadata = load_named_filament_metadata(str(support_price_filament))

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
                f"Orca failed with code {proc.returncode}\n"
                f"Temp dir kept at: {tmpdir}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )

        gcode_path = output_dir / "plate_1.gcode"
        if not gcode_path.exists():
            raise OrcaSliceError(f"plate_1.gcode was not generated\nTemp dir kept at: {tmpdir}")

        result = parse_gcode(
            str(gcode_path),
            filament_metadata=filament_metadata,
            support_filament_metadata=support_filament_metadata,
        )

        # Nur bei Erfolg aufräumen? Falls du tmp behalten willst, auskommentieren:
        if not KEEP_TMP:
            shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            print(f"[DEBUG] tmp dir kept at: {tmpdir}")

        return result

    except Exception:
        # Bei Fehler absichtlich NICHT löschen
        raise


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


def parse_gcode(
    gcode_path: str,
    filament_metadata: Dict | None = None,
    support_filament_metadata: Dict | None = None,
) -> Dict:
    filament_metadata = filament_metadata or {}
    support_filament_metadata = support_filament_metadata or filament_metadata

    current_role = "unknown"

    total_filament_mm = None
    total_filament_cm3 = None
    total_weight_g_from_header = None
    total_cost_from_header = None
    print_time_min = None

    gcode_filament_density = None
    gcode_filament_diameter = None
    gcode_filament_cost_per_kg = None
    gcode_filament_type = None

    extrusion_by_role_mm: Dict[str, float] = {}

    with open(gcode_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()

            if s.startswith("; filament used [mm] ="):
                try:
                    total_filament_mm = float(s.split("=", 1)[1].strip())
                except Exception:
                    pass

            elif s.startswith("; filament used [cm3] ="):
                try:
                    total_filament_cm3 = float(s.split("=", 1)[1].strip())
                except Exception:
                    pass

            elif s.startswith("; total filament used [g] ="):
                try:
                    total_weight_g_from_header = float(s.split("=", 1)[1].strip())
                except Exception:
                    pass

            elif s.startswith("; total filament cost ="):
                try:
                    total_cost_from_header = float(s.split("=", 1)[1].strip())
                except Exception:
                    pass

            elif s.startswith("; estimated printing time (normal mode) ="):
                raw = s.split("=", 1)[1].strip()
                print_time_min = _parse_time_to_minutes(raw)

            elif s.startswith("; filament_density ="):
                try:
                    value = float(s.split("=", 1)[1].strip().strip('"'))
                    if value > 0:
                        gcode_filament_density = value
                except Exception:
                    pass

            elif s.startswith("; filament_diameter ="):
                try:
                    value = float(s.split("=", 1)[1].strip().strip('"'))
                    if value > 0:
                        gcode_filament_diameter = value
                except Exception:
                    pass

            elif s.startswith("; filament_cost ="):
                try:
                    value = float(s.split("=", 1)[1].strip().strip('"'))
                    if value > 0:
                        gcode_filament_cost_per_kg = value
                except Exception:
                    pass

            elif s.startswith("; filament_type ="):
                gcode_filament_type = s.split("=", 1)[1].strip().strip('"')

            elif s.startswith(";TYPE:"):
                current_role = s.split(":", 1)[1].strip()

            elif s.startswith("G1") and " E" in s:
                m = re.search(r"(?:^|\s)E(-?\d+(?:\.\d+)?)", s)
                if m:
                    try:
                        delta_e = float(m.group(1))
                    except Exception:
                        delta_e = 0.0

                    if delta_e > 0:
                        extrusion_by_role_mm[current_role] = extrusion_by_role_mm.get(current_role, 0.0) + delta_e

    if total_filament_mm is None:
        total_filament_mm = sum(extrusion_by_role_mm.values())

    model_density_g_cm3 = filament_metadata.get("filament_density_g_cm3") or gcode_filament_density
    model_diameter_mm = filament_metadata.get("filament_diameter_mm") or gcode_filament_diameter or FILAMENT_DIAMETER_MM_DEFAULT
    model_cost_per_kg = filament_metadata.get("filament_cost_eur_per_kg") or gcode_filament_cost_per_kg
    model_material_name = (
        filament_metadata.get("filament_type")
        or filament_metadata.get("filament_settings_id")
        or gcode_filament_type
        or "unknown"
    )

    support_density_g_cm3 = support_filament_metadata.get("filament_density_g_cm3") or model_density_g_cm3
    support_cost_per_kg = support_filament_metadata.get("filament_cost_eur_per_kg") or model_cost_per_kg
    support_material_name = (
        support_filament_metadata.get("filament_type")
        or support_filament_metadata.get("filament_settings_id")
        or model_material_name
    )

    if total_filament_cm3 is None and total_filament_mm is not None:
        radius_cm = (model_diameter_mm / 10.0) / 2.0
        length_cm = total_filament_mm / 10.0
        total_filament_cm3 = math.pi * (radius_cm ** 2) * length_cm

    support_mm = (
        extrusion_by_role_mm.get("Support", 0.0)
        + extrusion_by_role_mm.get("Support interface", 0.0)
    )
    model_mm = (total_filament_mm or 0.0) - support_mm

    support_cm3 = 0.0
    model_cm3 = 0.0
    if total_filament_mm and total_filament_mm > 0 and total_filament_cm3 is not None:
        support_ratio = support_mm / total_filament_mm
        model_ratio = model_mm / total_filament_mm
        support_cm3 = total_filament_cm3 * support_ratio
        model_cm3 = total_filament_cm3 * model_ratio

    model_g = model_cm3 * model_density_g_cm3 if (model_cm3 and model_density_g_cm3) else 0.0
    support_g = support_cm3 * support_density_g_cm3 if (support_cm3 and support_density_g_cm3) else 0.0

    model_cost = (model_g / 1000.0) * model_cost_per_kg if (model_g and model_cost_per_kg) else 0.0
    support_cost = (support_g / 1000.0) * support_cost_per_kg if (support_g and support_cost_per_kg) else 0.0

    total_weight_g = model_g + support_g
    total_cost = model_cost + support_cost

    tools = [
        {
            "tool": 0,
            "role": "model",
            "material": model_material_name,
            "filament_length_mm": round(model_mm, 2),
            "filament_volume_cm3": round(model_cm3, 3),
            "filament_weight_g": round(model_g, 3),
            "material_cost_eur": round(model_cost, 2),
        }
    ]

    if support_mm > 0:
        tools.append(
            {
                "tool": 0,
                "role": "support",
                "material": support_material_name,
                "filament_length_mm": round(support_mm, 2),
                "filament_volume_cm3": round(support_cm3, 3),
                "filament_weight_g": round(support_g, 3),
                "material_cost_eur": round(support_cost, 2),
            }
        )

    return {
        "print_time_minutes": int(print_time_min or 0),
        "print_time_hours": round((print_time_min or 0) / 60.0, 4),
        "filament_length_mm_total": round(total_filament_mm, 3) if total_filament_mm is not None else 0,
        "filament_volume_cm3_total": round(total_filament_cm3, 3) if total_filament_cm3 is not None else 0,
        "filament_weight_g_total": round(total_weight_g, 3),
        "material_cost_eur_total": round(total_cost, 2),
        "model_filament_length_mm": round(model_mm, 3),
        "support_filament_length_mm": round(support_mm, 3),
        "model_filament_volume_cm3": round(model_cm3, 3),
        "support_filament_volume_cm3": round(support_cm3, 3),
        "model_filament_weight_g": round(model_g, 3),
        "support_filament_weight_g": round(support_g, 3),
        "model_material_cost_eur": round(model_cost, 2),
        "support_material_cost_eur": round(support_cost, 2),
        "tools": tools,
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
