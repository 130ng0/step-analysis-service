from __future__ import annotations

import json
import pathlib
import sys
from typing import Any


USER_BASE = pathlib.Path("/workspace/profiles")
ORCA_RESOURCES = pathlib.Path("/opt/orca/squashfs-root/resources")
OUT_BASE = pathlib.Path("/workspace/debug-last")


class ResolveProfilesError(Exception):
    pass


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_name(value: str) -> str:
    return value.strip().lower()


def find_candidate_files(base_dirs: list[pathlib.Path]) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for base in base_dirs:
        if base.exists():
            files.extend(base.rglob("*.json"))
    return sorted(set(files))


def _matches_profile(path: pathlib.Path, wanted: str) -> bool:
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
        if _matches_profile(path, wanted):
            return path
    raise FileNotFoundError(f"Profile not found: {profile_name}")


def resolve_profile(profile_name: str, candidates: list[pathlib.Path]) -> dict[str, Any]:
    path = find_profile_file(profile_name, candidates)
    return load_json(path)


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


def cleanup_profile(data: dict[str, Any], profile_kind: str, original_name: str) -> dict[str, Any]:
    cleaned = dict(data)
    cleaned.pop("inherits", None)
    cleaned["type"] = profile_kind
    cleaned["name"] = cleaned.get("name") or original_name
    cleaned["from"] = "system"
    cleaned.setdefault("instantiation", "true")
    cleaned["is_custom_defined"] = "0"
    return cleaned


def patch_process_for_printer(process: dict[str, Any], printer: dict[str, Any]) -> dict[str, Any]:
    patched = dict(process)

    printer_name = printer.get("name")
    printer_model = printer.get("printer_model")
    printer_variant = printer.get("printer_variant")
    nozzle = printer.get("nozzle_diameter")

    if printer_name:
        patched["compatible_printers"] = [printer_name]
    patched["compatible_printers_condition"] = ""

    if printer_model:
        patched["compatible_printer_model"] = [printer_model]

    if printer_variant:
        patched["printer_variant"] = str(printer_variant)

    if nozzle is not None:
        if isinstance(nozzle, list):
            patched["supported_nozzle_diameters"] = sorted({str(v) for v in nozzle})
        else:
            patched["supported_nozzle_diameters"] = [str(nozzle)]

    return patched


def patch_filament_for_printer(filament: dict[str, Any], printer: dict[str, Any]) -> dict[str, Any]:
    patched = dict(filament)

    extruder_count = _infer_extruder_count(printer)
    printer_name = printer.get("name")
    printer_model = printer.get("printer_model")
    nozzle = printer.get("nozzle_diameter")

    patched["type"] = "filament"
    patched["from"] = "system"
    patched["is_custom_defined"] = "0"

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

    for key, value in list(patched.items()):
        if isinstance(value, list):
            patched[key] = _repeat_last(value, extruder_count)

    return patched


def apply_process_overrides(
    process: dict[str, Any],
    infill_percent: float | None = None,
    perimeter_count: int | None = None,
    top_layers: int | None = None,
    bottom_layers: int | None = None,
) -> dict[str, Any]:
    patched = dict(process)

    if infill_percent is not None:
        patched["sparse_infill_density"] = f"{int(round(float(infill_percent)))}%"
    if perimeter_count is not None:
        patched["wall_loops"] = str(int(perimeter_count))
    if top_layers is not None:
        patched["top_shell_layers"] = str(int(top_layers))
    if bottom_layers is not None:
        patched["bottom_shell_layers"] = str(int(bottom_layers))

    return patched


def resolve_profile_set(
    machine_name: str,
    process_name: str,
    filament_name: str,
    output_name: str = "debug-last",
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
        raise ResolveProfilesError("No profile JSON files found")

    try:
        machine = resolve_profile(machine_name, candidates)
        process = resolve_profile(process_name, candidates)
        filament = resolve_profile(filament_name, candidates)
    except Exception as exc:
        raise ResolveProfilesError(str(exc)) from exc

    machine = cleanup_profile(machine, "machine", machine_name)
    process = cleanup_profile(process, "process", process_name)
    filament = cleanup_profile(filament, "filament", filament_name)

    process = patch_process_for_printer(process, machine)
    filament = patch_filament_for_printer(filament, machine)
    process = apply_process_overrides(
        process,
        infill_percent=infill_percent,
        perimeter_count=perimeter_count,
        top_layers=top_layers,
        bottom_layers=bottom_layers,
    )

    out_dir = out_base if out_base.name == output_name else (out_base / output_name)
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

    out_dir = resolve_profile_set(
        machine_name=machine_name,
        process_name=process_name,
        filament_name=filament_name,
        output_name=output_name,
    )
    print(f"Resolved profiles written to: {out_dir}")
    print(f"  printer : {out_dir / 'printer.json'}")
    print(f"  process : {out_dir / 'process.json'}")
    print(f"  filament: {out_dir / 'filament.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())