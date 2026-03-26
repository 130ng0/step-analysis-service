from __future__ import annotations

import json
import pathlib
import sys
from typing import Any


USER_BASE = pathlib.Path("/workspace/profiles")
ORCA_RESOURCES = pathlib.Path("/opt/orca/squashfs-root/resources")
OUT_BASE = pathlib.Path("/workspace/resolved")


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_name(name: str) -> str:
    return name.strip().lower()


def find_candidate_files(base_dirs: list[pathlib.Path]) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for base in base_dirs:
        if base.exists():
            files.extend(base.rglob("*.json"))
    return files


def find_profile_file(profile_name: str, candidates: list[pathlib.Path]) -> pathlib.Path:
    wanted = normalize_name(profile_name)
    for path in candidates:
        if normalize_name(path.stem) == wanted:
            return path
    raise FileNotFoundError(f"Profile not found: {profile_name}")


def merge_dicts(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = dict(parent)
    for k, v in child.items():
        if k == "inherits":
            continue
        merged[k] = v
    return merged


def resolve_profile(profile_name: str, candidates: list[pathlib.Path], seen: set[str] | None = None) -> dict[str, Any]:
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

    # Nur wirklich problematische Felder entfernen
    cleaned.pop("inherits", None)

    # Diese Metadaten explizit setzen
    cleaned["from"] = "User"
    cleaned["name"] = cleaned.get("name") or original_name
    cleaned["type"] = profile_kind

    # Orca mag oft Strings statt bool/int bei manchen Meta-Feldern
    if "instantiation" in cleaned:
        if isinstance(cleaned["instantiation"], bool):
            cleaned["instantiation"] = "true" if cleaned["instantiation"] else "false"

    return cleaned


def write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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

    candidates = find_candidate_files(
        [
            USER_BASE,
            ORCA_RESOURCES / "profiles",
            ORCA_RESOURCES / "profiles_template",
        ]
    )

    machine = resolve_profile(machine_name, candidates)
    process = resolve_profile(process_name, candidates)
    filament = resolve_profile(filament_name, candidates)

    machine = cleanup_profile(machine, "machine", machine_name)
    process = cleanup_profile(process, "process", process_name)
    filament = cleanup_profile(filament, "filament", filament_name)

    out_dir = OUT_BASE / output_name
    write_json(out_dir / "printer.json", machine)
    write_json(out_dir / "preset.json", process)
    write_json(out_dir / "filament.json", filament)

    print(f"Resolved profiles written to: {out_dir}")
    print(f"  printer : {out_dir / 'printer.json'}")
    print(f"  preset  : {out_dir / 'preset.json'}")
    print(f"  filament: {out_dir / 'filament.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())