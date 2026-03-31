from __future__ import annotations

from dataclasses import dataclass


class ProfileLoaderError(Exception):
    pass


@dataclass(frozen=True)
class SelectedProfiles:
    machine: str
    process: str
    filament: str

    def as_dict(self) -> dict[str, str]:
        return {
            "machine": self.machine,
            "process": self.process,
            "filament": self.filament,
        }


DEFAULT_MACHINE = "EL-140V3 v.2.0"


_FILAMENT_MAP_06 = {
    "abs": "ABS PRO 0.6mm v2.0",
    "abs_cf": "ABS-CF PRO 0.6mm 2.0",
    "abs_esd": "ABS-ESD PRO 0.6mm v2.0",
    "asa": "ASA PRO 0.6mm v2.0",
    "pc": "PC PRO 0.6mm v2.0",
    "pc_cf": "PC-CF PRO 0.6mm v2.0",
}


SPECIAL_CASES = {
    "pc_fr": SelectedProfiles(
        machine=DEFAULT_MACHINE,
        process="Standard 0.4mm v2.0",
        filament="PC-FR Ensinger",
    ),
    "tpu": SelectedProfiles(
        machine=DEFAULT_MACHINE,
        process="TPU 0.4mm v2.0",
        filament="TPU v.1.1 (REVO) 0.4mm",
    ),
}



def select_profile_set(material_profile: str, support_material_type: str) -> dict[str, str]:
    material_key = material_profile.strip().lower()
    support_key = support_material_type.strip().lower()

    if support_key not in {"none", "breakaway"}:
        raise ProfileLoaderError(f"Unsupported support material type: {support_material_type}")

    if material_key in SPECIAL_CASES:
        return SPECIAL_CASES[material_key].as_dict()

    filament_name = _FILAMENT_MAP_06.get(material_key)
    if not filament_name:
        raise ProfileLoaderError(f"No filament mapping configured for material: {material_profile}")

    # breakaway is intentionally mapped to the standard single-extruder process for now.
    return SelectedProfiles(
        machine=DEFAULT_MACHINE,
        process="Standard 0.6mm v2.0",
        filament=filament_name,
    ).as_dict()
