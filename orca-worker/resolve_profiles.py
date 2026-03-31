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
    cleaned["type"] = profile_kind
    cleaned["name"] = cleaned.get("name") or original_name
    cleaned["from"] = "system"
    cleaned["is_custom_defined"] = "0"
    cleaned.setdefault("instantiation", "true")
    return cleaned


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if value is None:
        return []
    return [value]


def _repeat_last(values: list[Any], count: int) -> list[Any]:
    if count <= 0:
        return []
    if not values:
        return []
    if len(values) >= count:
        return values[:count]
    return values + [values[-1]] * (count - len(values))


def _infer_extruder_count(printer: dict[str, Any]) -> int:
    nozzle = _as_list(printer.get("nozzle_diameter"))
    if nozzle:
        return len(nozzle)

    extruder_ids = _as_list(printer.get("printer_extruder_id"))
    if extruder_ids:
        return len(extruder_ids)

    offsets = _as_list(printer.get("extruder_offset"))
    if offsets:
        return len(offsets)

    return 1


def _infer_target_nozzle_from_process(process: dict[str, Any], printer: dict[str, Any]) -> str:
    supported = _as_list(process.get("supported_nozzle_diameters"))
    if supported:
        return str(supported[0])

    line_width = process.get("line_width")
    if line_width is not None:
        try:
            lw = float(str(line_width).replace(",", "."))
            if lw >= 0.55:
                return "0.6"
            return "0.4"
        except ValueError:
            pass

    current = _as_list(printer.get("nozzle_diameter"))
    if current:
        return str(current[0])

    return "0.4"


def patch_printer_for_process(printer: dict[str, Any], process: dict[str, Any]) -> dict[str, Any]:
    patched = dict(printer)

    extruder_count = _infer_extruder_count(patched)
    target_nozzle = _infer_target_nozzle_from_process(process, patched)

    # Nur die Nozzle/Variant gezielt anpassen.
    patched["nozzle_diameter"] = [target_nozzle] * extruder_count

    if "printer_variant" in patched or "printer_model" in patched:
        patched["printer_variant"] = target_nozzle

    # Falls Orca-Export diese Felder schon enthält, konsistent halten.
    if "supported_nozzle_diameters" in patched:
        patched["supported_nozzle_diameters"] = [target_nozzle] * extruder_count

    if "default_nozzle_diameter" in patched:
        patched["default_nozzle_diameter"] = target_nozzle

    # Bereits exportierte Multi-Extruder-Felder auf korrekte Länge bringen,
    # aber keine neuen Orca-internen Felder erfinden.
    for key in (
        "extruder_colour",
        "extruder_offset",
        "printer_extruder_id",
        "extruder_type",
        "extruder_variant_list",
        "printer_extruder_variant",
        "default_nozzle_volume_type",
        "machine_max_acceleration_e",
        "machine_max_acceleration_extruding",
        "machine_max_acceleration_retracting",
        "machine_max_acceleration_x",
        "machine_max_acceleration_y",
        "machine_max_acceleration_z",
        "machine_max_speed_e",
        "machine_max_speed_x",
        "machine_max_speed_y",
        "machine_max_speed_z",
        "machine_max_jerk_e",
        "machine_max_jerk_x",
        "machine_max_jerk_y",
        "machine_max_jerk_z",
        "retraction_length",
        "retraction_speed",
        "deretraction_speed",
        "retract_length_toolchange",
        "retract_restart_extra",
        "retract_restart_extra_toolchange",
        "retract_when_changing_layer",
        "retraction_minimum_travel",
        "retraction_distances_when_cut",
        "long_retractions_when_cut",
        "z_hop",
        "z_hop_types",
        "wipe",
        "wipe_distance",
        "retract_before_wipe",
        "retract_lift_above",
        "retract_lift_below",
        "retract_lift_enforce",
        "max_layer_height",
        "min_layer_height",
        "travel_slope",
        "nozzle_type",
        "nozzle_volume",
        "nozzle_flush_dataset",
        "extruder_printable_height",
    ):
        if key in patched and isinstance(patched[key], list):
            patched[key] = _repeat_last(patched[key], extruder_count)

    # Kritisch für IDEX: physisches Extruder-Mapping muss zur Extruderanzahl passen.
    if extruder_count == 1:
        patched["physical_extruder_map"] = ["0"]
    else:
        patched["physical_extruder_map"] = [str(i) for i in range(extruder_count)]

    # Falls vorhanden, auf richtige Länge bringen
    if "default_nozzle_volume_type" in patched and isinstance(patched["default_nozzle_volume_type"], list):
        patched["default_nozzle_volume_type"] = _repeat_last(
            patched["default_nozzle_volume_type"], extruder_count
        )

    if "nozzle_volume" in patched and isinstance(patched["nozzle_volume"], list):
        patched["nozzle_volume"] = _repeat_last(patched["nozzle_volume"], extruder_count)

    # Falls Orca die Felder erwartet, aber sie im Export fehlen, nur minimal ergänzen.
    if "extruder_type" not in patched:
        patched["extruder_type"] = ["Direct Drive"] * extruder_count

    if "extruder_variant_list" not in patched:
        patched["extruder_variant_list"] = ["Direct Drive Standard"] * extruder_count

    if "printer_extruder_variant" not in patched:
        patched["printer_extruder_variant"] = ["Direct Drive Standard"] * extruder_count

    if "default_nozzle_volume_type" not in patched:
        patched["default_nozzle_volume_type"] = ["Standard"] * extruder_count

    # Nur wenn das Feld fehlt, minimal sinnvoll ergänzen.
    if "nozzle_volume" not in patched:
        patched["nozzle_volume"] = ["Standard"] * extruder_count

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
            patched["supported_nozzle_diameters"] = sorted({str(v) for v in nozzle})
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
            patched["supported_nozzle_diameters"] = sorted({str(v) for v in nozzle})
        else:
            patched["supported_nozzle_diameters"] = [str(nozzle)]

    # Nur bestehende Listen auf Extruderanzahl bringen.
    for key, value in list(patched.items()):
        if isinstance(value, list):
            patched[key] = _repeat_last(value, extruder_count)

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

    machine = patch_printer_for_process(machine, process)
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