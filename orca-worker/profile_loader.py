from __future__ import annotations


class ProfileLoaderError(Exception):
    pass


def select_profile_set(material_profile: str, support_material_type: str | None = None) -> dict[str, str | None]:
    material = (material_profile or "").strip().lower()
    support = (support_material_type or "").strip().lower()

    machine_06 = "EL-140V3_0.6"
    process_06 = "Standard_0.6mm"

    machine_04 = "EL-140V3_0.4"
    process_04 = "Standard_0.4mm"

    filament_map_06 = {
        "abs": "ABS_PRO_0.6mm",
        "abs_pro": "ABS_PRO_0.6mm",
        "abs-cf": "ABS-CF_PRO_0.6mm",
        "abs_cf": "ABS-CF_PRO_0.6mm",
        "abs-esd": "ABS-ESD_PRO_0.6mm",
        "abs_esd": "ABS-ESD_PRO_0.6mm",
        "asa": "ASA_PRO_0.6mm",
        "pc": "PC_PRO_0.6mm",
        "pc_pro": "PC_PRO_0.6mm",
        "pc-cf": "PC-CF_PRO_0.6mm",
        "pc_cf": "PC-CF_PRO_0.6mm",
    }

    filament_map_04 = {
        "pc-fr": "PC-FR_Ensinger_0.4mm",
        "pc_fr": "PC-FR_Ensinger_0.4mm",
        "pcfr": "PC-FR_Ensinger_0.4mm",
        "tpu": "TPU_0.4mm",
    }

    support_price_filament = None
    if support == "hips":
        support_price_filament = "SUPP_HIPS_0.6mm"
    elif support == "soluble":
        support_price_filament = "SUPP_NEVO_SOLUBLE_0.6mm"
    elif support in ("none", "breakaway", ""):
        support_price_filament = None
    else:
        raise ProfileLoaderError(f"Unsupported support_material_type: {support_material_type}")

    if material in filament_map_04:
        return {
            "machine": machine_04,
            "process": process_04,
            "filament": filament_map_04[material],
            "support_price_filament": support_price_filament,
        }

    if material in filament_map_06:
        return {
            "machine": machine_06,
            "process": process_06,
            "filament": filament_map_06[material],
            "support_price_filament": support_price_filament,
        }

    raise ProfileLoaderError(f"Unsupported material_profile: {material_profile}")