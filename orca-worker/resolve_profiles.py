from __future__ import annotations

import json
import pathlib
import sys
from typing import Any


USER_BASE = pathlib.Path("/workspace/profiles")
ORCA_RESOURCES = pathlib.Path("/opt/orca/squashfs-root/resources")
OUT_BASE = pathlib.Path("/workspace/resolved")


class ResolveProfilesError(Exception):
    pass


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_name(name: str) -> str:
    return name.strip().lower()


def find_candidate_files(base_dirs: list[pathlib.Path]) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for base in base_dirs:
        if base.exists():
            files.extend(base.rglob("*.json"))
    return sorted(set(files))


def _matches_profile_name(path: pathlib.Path, wanted: str) -> bool:
    if normalize_name(path.stem) == wanted:
        return True
    try:
        data = load_json(path)
    except Exception:
        return False
    return normalize_name(str(data.get("name", ""))) == wanted


def find_profile_file(profile_name: str, candidates: list[pathlib.Path]) -> pathlib.Path:
    wanted = normalize_name(profile_name)
    for path in candidates:
        if _matches_profile_name(path, wanted):
            return path
    raise FileNotFoundError(f"Profile not found: {profile_name}")


def merge_dicts(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = dict(parent)
    for k, v in child.items():
        if k == "inherits":
            continue
        merged[k] = v
    return merged


def resolve_profile(
    profile_name: str,
    candidates: list[pathlib.Path],
    seen: set[str] | None = None,
) -> dict[str, Any]:
    if seen is None:
        seen = set()

    key = normalize_name(profile_name)
    if key in seen:
        raise RuntimeError(f"Circular inheritance detected: {profile_name}")
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
    cleaned["from"] = "system"
    cleaned["name"] = cleaned.get("name") or original_name
    cleaned["type"] = profile_kind
    cleaned["is_custom_defined"] = "0"
    cleaned.setdefault("instantiation", "true")
    return cleaned


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if value is None:
        return []
    return [value]


def _repeat_to_length(values: list[Any], length: int, default: Any) -> list[Any]:
    if length <= 0:
        return []
    if not values:
        return [default] * length
    if len(values) >= length:
        return values[:length]
    last = values[-1]
    return values + [last] * (length - len(values))


def _ensure_list_length(data: dict[str, Any], key: str, length: int, default: Any) -> None:
    values = _as_list(data.get(key))
    data[key] = _repeat_to_length(values, length, default)


def _infer_extruder_count(printer: dict[str, Any]) -> int:
    nozzle = _as_list(printer.get("nozzle_diameter"))
    if nozzle:
        return len(nozzle)
    offsets = _as_list(printer.get("extruder_offset"))
    if offsets:
        return len(offsets)
    return 1


def patch_printer_for_orca(printer: dict[str, Any]) -> dict[str, Any]:
    patched = dict(printer)

    extruder_count = _infer_extruder_count(patched)

    _ensure_list_length(patched, "nozzle_diameter", extruder_count, "0.4")
    _ensure_list_length(patched, "extruder_offset", extruder_count, "0x0")
    _ensure_list_length(patched, "deretraction_speed", extruder_count, "35")
    _ensure_list_length(patched, "extruder_colour", extruder_count, "#FCE94F")
    _ensure_list_length(patched, "max_layer_height", extruder_count, "0.28")
    _ensure_list_length(patched, "min_layer_height", extruder_count, "0.08")
    _ensure_list_length(patched, "retraction_minimum_travel", extruder_count, "2")
    _ensure_list_length(patched, "retract_before_wipe", extruder_count, "100%")
    _ensure_list_length(patched, "retract_when_changing_layer", extruder_count, "0")
    _ensure_list_length(patched, "retraction_length", extruder_count, "0.4")
    _ensure_list_length(patched, "retract_length_toolchange", extruder_count, "3")
    _ensure_list_length(patched, "z_hop", extruder_count, "0.3")
    _ensure_list_length(patched, "retract_restart_extra", extruder_count, "0")
    _ensure_list_length(patched, "retract_restart_extra_toolchange", extruder_count, "0")
    _ensure_list_length(patched, "retraction_speed", extruder_count, "35")
    _ensure_list_length(patched, "wipe", extruder_count, "1")
    _ensure_list_length(patched, "long_retractions_when_cut", extruder_count, "0")
    _ensure_list_length(patched, "retract_lift_above", extruder_count, "0")
    _ensure_list_length(patched, "retract_lift_below", extruder_count, "0")
    _ensure_list_length(patched, "retract_lift_enforce", extruder_count, "Top Only")
    _ensure_list_length(patched, "retraction_distances_when_cut", extruder_count, "18")
    _ensure_list_length(patched, "travel_slope", extruder_count, "0")
    _ensure_list_length(patched, "wipe_distance", extruder_count, "1")
    _ensure_list_length(patched, "z_hop_types", extruder_count, "Normal Lift")

    _ensure_list_length(patched, "machine_max_acceleration_e", extruder_count, "5000")
    _ensure_list_length(patched, "machine_max_acceleration_extruding", extruder_count, "4000")
    _ensure_list_length(patched, "machine_max_acceleration_retracting", extruder_count, "5000")
    _ensure_list_length(patched, "machine_max_speed_e", extruder_count, "20")
    _ensure_list_length(patched, "machine_min_extruding_rate", extruder_count, "0")
    _ensure_list_length(patched, "machine_min_travel_rate", extruder_count, "0")

    nozzle_values = [str(v) for v in _as_list(patched.get("nozzle_diameter"))]
    patched["nozzle_diameter"] = _repeat_to_length(nozzle_values, extruder_count, "0.4")

    # Variant must match the real nozzle profile.
    patched["printer_variant"] = str(patched["nozzle_diameter"][0])

    # Critical Orca extruder metadata
    patched["extruder_type"] = ["Direct Drive"] * extruder_count
    patched["nozzle_volume_type"] = ["Standard"] * extruder_count
    patched["default_nozzle_volume_type"] = "Standard"

    # This is the missing companion field very likely required by Orca.
    patched["nozzle_volume"] = ["Standard"] * extruder_count

    # Explicit physical mapping for both tools.
    patched["physical_extruder_map"] = [str(i) for i in range(extruder_count)]

    # Sensible generic defaults
    patched.setdefault("printer_technology", "FFF")
    patched.setdefault("printer_model", "Generic Marlin Printer")
    patched.setdefault("setting_id", "GM001")
    patched.setdefault("silent_mode", "0")
    patched.setdefault("machine_pause_gcode", "M601")
    patched.setdefault("default_print_profile", "")
    patched.setdefault("before_layer_change_gcode", ";BEFORE_LAYER_CHANGE\n;[layer_z]\nG92 E0\n")

    return patched


def patch_process_for_printer(process: dict[str, Any], printer: dict[str, Any]) -> dict[str, Any]:
    patched = dict(process)

    printer_name = printer.get("name")
    printer_model = printer.get("printer_model")
    printer_variant = printer.get("printer_variant")
    printer_structure = printer.get("printer_structure")
    nozzle = printer.get("nozzle_diameter")

    if printer_name:
        patched["compatible_printers"] = [printer_name]
    patched["compatible_printers_condition"] = ""

    if printer_model:
        patched["compatible_printer_model"] = [printer_model]

    if printer_variant:
        patched["printer_variant"] = printer_variant

    if printer_structure:
        patched["printer_structure"] = printer_structure

    if nozzle is not None:
        if isinstance(nozzle, list):
            patched["supported_nozzle_diameters"] = [str(v) for v in nozzle]
        else:
            patched["supported_nozzle_diameters"] = [str(nozzle)]

    return patched


def patch_filament_for_printer(filament: dict[str, Any], printer: dict[str, Any]) -> dict[str, Any]:
    patched = dict(filament)

    printer_name = printer.get("name")
    printer_model = printer.get("printer_model")
    nozzle = printer.get("nozzle_diameter")
    extruder_count = _infer_extruder_count(printer)

    if printer_name:
        patched["compatible_printers"] = [printer_name]
    patched["compatible_printers_condition"] = ""

    if printer_model:
        patched["compatible_printer_model"] = [printer_model]

    if nozzle is not None:
        if isinstance(nozzle, list):
            patched["supported_nozzle_diameters"] = [str(v) for v in nozzle]
        else:
            patched["supported_nozzle_diameters"] = [str(nozzle)]

    # Expand all list-valued filament settings to match printer extruder count.
    for key, value in list(patched.items()):
        if isinstance(value, list):
            default = value[-1] if value else ""
            patched[key] = _repeat_to_length(value, extruder_count, default)

    # Critical pairing for Orca's extruder lookup.
    patched["filament_extruder_variant"] = ["Direct Drive Standard"] * extruder_count

    return patched


def apply_process_overrides(
    process: dict[str, Any],
    infill_percent: float,
    perimeter_count: int,
    top_layers: int,
    bottom_layers: int,
) -> dict[str, Any]:
    patched = dict(process)
    patched["sparse_infill_density"] = f"{int(round(float(infill_percent)))}%"
    patched["wall_loops"] = str(int(perimeter_count))
    patched["top_shell_layers"] = str(int(top_layers))
    patched["bottom_shell_layers"] = str(int(bottom_layers))
    return patched


def write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_profile_set(
    machine_name: str,
    process_name: str,
    filament_name: str,
    output_name: str,
    infill_percent: float | None = None,
    perimeter_count: int | None = None,
    top_layers: int | None = None,
    bottom_layers: int | None = None,
    user_base: pathlib.Path = USER_BASE,
    resources_base: pathlib.Path = ORCA_RESOURCES,
    out_base: pathlib.Path = OUT_BASE,
) -> pathlib.Path:
    candidates = find_candidate_files(
        [
            user_base,
            resources_base / "profiles",
            resources_base / "profiles_template",
        ]
    )
    if not candidates:
        raise ResolveProfilesError("No profile JSON files found in user or Orca resource directories")

    try:
        machine = resolve_profile(machine_name, candidates)
        process = resolve_profile(process_name, candidates)
        filament = resolve_profile(filament_name, candidates)
    except Exception as exc:
        raise ResolveProfilesError(str(exc)) from exc

    machine = cleanup_profile(machine, "machine", machine_name)
    process = cleanup_profile(process, "process", process_name)
    filament = cleanup_profile(filament, "filament", filament_name)

    machine = patch_printer_for_orca(machine)
    process = patch_process_for_printer(process, machine)
    filament = patch_filament_for_printer(filament, machine)

    if None not in (infill_percent, perimeter_count, top_layers, bottom_layers):
        process = apply_process_overrides(
            process,
            float(infill_percent),
            int(perimeter_count),
            int(top_layers),
            int(bottom_layers),
        )

    out_dir = out_base / output_name
    write_json(out_dir / "printer.json", machine)
    write_json(out_dir / "process.json", process)
    write_json(out_dir / "filament.json", filament)
    return out_dir


def main() -> int:
    if len(sys.argv) != 5:
        print(
            "Usage: python3 /workspace/resolve_profiles.py "
            "<machine_profile_name> <process_profile_name> <filament_profile_name> <output_name>"
        )
        return 2

    machine_name = sys.argv[1]
    process_name = sys.argv[2]
    filament_name = sys.argv[3]
    output_name = sys.argv[4]

    out_dir = resolve_profile_set(machine_name, process_name, filament_name, output_name)
    print(f"Resolved profiles written to: {out_dir}")
    print(f"  printer : {out_dir / 'printer.json'}")
    print(f"  process : {out_dir / 'process.json'}")
    print(f"  filament: {out_dir / 'filament.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())