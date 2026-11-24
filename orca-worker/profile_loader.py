from __future__ import annotations


class ProfileLoaderError(Exception):
    pass


_FILAMENT_06 = {
    "abs": "ABS_PRO", "abs_pro": "ABS_PRO", "abs_cf": "ABS-CF_PRO",
    "abs_esd": "ABS-ESD_PRO", "asa": "ASA_PRO", "pc": "PC_PRO", "pc_pro": "PC_PRO",
    "pc_cf": "PC-CF_PRO",
}
_FILAMENT_04_ONLY = {"pc_fr": "PC-FR_Ensinger", "pcfr": "PC-FR_Ensinger", "tpu": "TPU"}


def select_profile_set(material_profile: str, support_material_type: str | None = None,
                       nozzle_diameter_mm: str | None = None) -> dict[str, str | None]:
    material = (material_profile or "").strip().lower().replace("-", "_")
    support = (support_material_type or "").strip().lower()
    if material not in _FILAMENT_06 and material not in _FILAMENT_04_ONLY:
        raise ProfileLoaderError(f"Unsupported material_profile: {material_profile}")
    nozzle = nozzle_diameter_mm or ("0.4" if material in _FILAMENT_04_ONLY else "0.6")
    if nozzle not in {"0.4", "0.6"} or (nozzle == "0.6" and material in _FILAMENT_04_ONLY):
        raise ProfileLoaderError(f"No {nozzle} mm profile for material {material_profile}")
    if support not in {"", "none", "breakaway", "hips", "soluble"}:
        raise ProfileLoaderError(f"Unsupported support_material_type: {support_material_type}")

    stem = _FILAMENT_04_ONLY.get(material) or _FILAMENT_06[material]
    filament = f"{stem}_{nozzle}mm"
    separate = support in {"hips", "soluble"}
    if separate:
        return {
            "machine": f"EL-140V3_{nozzle}", "process": f"Standard_{nozzle}mm",
            "filament": filament,
            "dual_model_profile": f"DUAL_{filament}" if nozzle == "0.6" or material == "tpu" else None,
            "support_filament": f"DUAL_{support.upper()}_{nozzle}mm",
            "nozzle_diameter_mm": nozzle,
        }
    return {
        "machine": f"EL-140V3_{nozzle}", "process": f"Standard_{nozzle}mm",
        "filament": filament, "dual_model_profile": None, "support_filament": None,
        "nozzle_diameter_mm": nozzle,
    }
