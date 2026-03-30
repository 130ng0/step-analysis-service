from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROFILES_BASE = Path("/workspace/profiles")
ORCA_RESOURCES = Path("/opt/orca/squashfs-root/resources")


class ProfileResolutionError(Exception):
    pass


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_name(name: str) -> str:
    return name.strip().lower()


def candidate_files() -> list[Path]:
    files: list[Path] = []

    for base in [
        PROFILES_BASE,
        ORCA_RESOURCES / "profiles",
        ORCA_RESOURCES / "profiles_template",
    ]:
        if base.exists():
            files.extend(base.rglob("*.json"))

    return files


def find_profile_file(profile_name: str, candidates: list[Path]) -> Path:
    wanted = normalize_name(profile_name)
    for path in candidates:
        if normalize_name(path.stem) == wanted:
            return path
    raise ProfileResolutionError(f"Profile not found: {profile_name}")


def merge_dicts(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = dict(parent)
    for k, v in child.items():
        if k == "inherits":
            continue
        merged[k] = v
    return merged


def resolve_profile(profile_name: str, candidates: list[Path], seen: set[str] | None = None) -> dict[str, Any]:
    if seen is None:
        seen = set()

    key = normalize_name(profile_name)
    if key in seen:
        raise ProfileResolutionError(f"Circular inheritance detected: {profile_name}")
    seen.add(key)

    path = find_profile_file(profile_name, candidates)
    data = load_json(path)

    parent_name = data.get("inherits")
    if parent_name:
        parent = resolve_profile(parent_name, candidates, seen)
        return merge_dicts(parent, data)

    return data


def cleanup_profile(data: dict[str, Any], profile_kind: str, original_name: str) -> dict[str, Any]:
    cleaned = dict(data)

    cleaned.pop("inherits", None)
    cleaned.pop("compatible_printers_condition", None)
    cleaned.pop("compatible_prints_condition", None)

    cleaned["from"] = "User"
    cleaned["name"] = cleaned.get("name") or original_name
    cleaned["type"] = profile_kind

    return cleaned


def patch_printer(printer: dict[str, Any], printer_name: str) -> dict[str, Any]:
    patched = dict(printer)
    patched["name"] = printer_name
    patched["from"] = "User"
    patched["type"] = "machine"
    return patched


def patch_process(process: dict[str, Any], printer: dict[str, Any], process_name: str, overrides: dict[str, Any]) -> dict[str, Any]:
    patched = dict(process)

    printer_name = printer.get("name")
    printer_model = printer.get("printer_model")
    nozzle = printer.get("nozzle_diameter")

    if printer_name:
        patched["compatible_printers"] = [printer_name]
    if printer_model:
        patched["compatible_printer_model"] = [printer_model]

    patched["name"] = process_name
    patched["from"] = "User"
    patched["type"] = "process"

    if nozzle is not None:
        if isinstance(nozzle, list):
            patched["supported_nozzle_diameters"] = nozzle
        else:
            patched["supported_nozzle_diameters"] = [str(nozzle)]

    # Nur diese Overrides wollen wir bewusst extern setzen
    patched["sparse_infill_density"] = f"{int(round(float(overrides['infill_percent'])))}%"
    patched["wall_loops"] = int(overrides["perimeter_count"])
    patched["top_shell_layers"] = int(overrides["top_layers"])
    patched["bottom_shell_layers"] = int(overrides["bottom_layers"])

    return patched


def patch_filament(filament: dict[str, Any], printer: dict[str, Any], filament_name: str) -> dict[str, Any]:
    patched = dict(filament)

    printer_name = printer.get("name")
    printer_model = printer.get("printer_model")
    nozzle = printer.get("nozzle_diameter")

    if printer_name:
        patched["compatible_printers"] = [printer_name]
    if printer_model:
        patched["compatible_printer_model"] = [printer_model]

    patched["name"] = filament_name
    patched["from"] = "User"
    patched["type"] = "filament"

    if nozzle is not None:
        if isinstance(nozzle, list):
            patched["supported_nozzle_diameters"] = nozzle
        else:
            patched["supported_nozzle_diameters"] = [str(nozzle)]

    return patched


def build_resolved_profiles(
    machine_profile_name: str,
    process_profile_name: str,
    filament_profile_name: str,
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidates = candidate_files()

    printer = resolve_profile(machine_profile_name, candidates)
    process = resolve_profile(process_profile_name, candidates)
    filament = resolve_profile(filament_profile_name, candidates)

    printer = cleanup_profile(printer, "machine", machine_profile_name)
    process = cleanup_profile(process, "process", process_profile_name)
    filament = cleanup_profile(filament, "filament", filament_profile_name)

    printer = patch_printer(printer, machine_profile_name)
    process = patch_process(process, printer, process_profile_name, overrides)
    filament = patch_filament(filament, printer, filament_profile_name)

    return printer, process, filament