from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ProfileLoaderError(Exception):
    pass


def select_profile_set(material_profile: str, support_material_type: str) -> dict[str, str]:
    """
    Basis-Mapping auf Grundlage deiner aktuellen ZIP.

    Standard:
    - 0.6 mm
    - breakaway aktuell ebenfalls mit Standard-Prozess

    Spezialfälle:
    - tpu -> TPU 0.4
    - pc_fr -> Standard 0.4
    """
    if material_profile == "tpu":
        return {
            "machine": "EL-140V3 v.2.0",
            "process": "TPU 0.4mm v2.0",
            "filament": "TPU v.1.1 (REVO) 0.4mm",
        }

    if material_profile == "pc_fr":
        return {
            "machine": "EL-140V3 v.2.0",
            "process": "Standard 0.4mm v2.0",
            "filament": "PC-FR Ensinger",
        }

    filament_map = {
        "abs": "ABS PRO 0.6mm v2.0",
        "abs_cf": "ABS-CF PRO 0.6mm 2.0",
        "abs_esd": "ABS-ESD PRO 0.6mm v2.0",
        "asa": "ASA PRO 0.6mm v2.0",
        "pc": "PC PRO 0.6mm v2.0",
        "pc_cf": "PC-CF PRO 0.6mm v2.0",
    }

    filament_name = filament_map.get(material_profile)
    if not filament_name:
        raise ProfileLoaderError(f"No filament mapping configured for material: {material_profile}")

    return {
        "machine": "EL-140V3 v.2.0",
        "process": "Standard 0.6mm v2.0",
        "filament": filament_name,
    }


def build_minimal_printer_preset(machine_name: str) -> dict[str, Any]:
    """
    Minimaler Maschinen-Preset.
    Nur inherits + identity.
    Keine compatibility-Felder.
    """
    return {
        "type": "machine",
        "from": "User",
        "inherits": machine_name,
        "name": f"API {machine_name}",
        "printer_settings_id": f"API {machine_name}",
        "is_custom_defined": "0",
        "version": "2.3.1.10",
    }


def build_minimal_process_preset(
    process_name: str,
    infill_percent: float,
    perimeter_count: int,
    top_layers: int,
    bottom_layers: int,
) -> dict[str, Any]:
    """
    Minimaler Prozess-Preset mit nur den echten Overrides.
    Alle compatibility-Felder bewusst weggelassen.
    """
    return {
        "type": "process",
        "from": "User",
        "inherits": process_name,
        "name": f"API {process_name}",
        "print_settings_id": f"API {process_name}",
        "is_custom_defined": "0",
        "version": "2.3.1.10",
        "sparse_infill_density": f"{int(round(float(infill_percent)))}%",
        "wall_loops": str(int(perimeter_count)),
        "top_shell_layers": str(int(top_layers)),
        "bottom_shell_layers": str(int(bottom_layers)),
    }


def build_minimal_filament_preset(filament_name: str) -> dict[str, Any]:
    """
    Minimaler Filament-Preset.
    Materialdaten kommen aus dem geerbten Profil.
    Keine compatibility-Felder.
    """
    return {
        "type": "filament",
        "from": "User",
        "inherits": filament_name,
        "name": f"API {filament_name}",
        "filament_settings_id": [f"API {filament_name}"],
        "is_custom_defined": "0",
        "version": "2.3.1.10",
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )