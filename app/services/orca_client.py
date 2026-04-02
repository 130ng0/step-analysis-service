from __future__ import annotations

import os
from typing import Any

import requests


class OrcaClientError(Exception):
    pass


ORCA_WORKER_URL = os.getenv("ORCA_WORKER_URL", "http://orca-worker:8090").rstrip("/")
ORCA_WORKER_TIMEOUT = int(os.getenv("ORCA_WORKER_TIMEOUT", "1800"))


def slice_with_worker(stl_bytes: bytes, stl_filename: str, payload: dict[str, Any]) -> dict[str, Any]:
    files = {
        "file": (stl_filename, stl_bytes, "application/octet-stream"),
    }

    data = {
        "material_profile": str(payload.get("material_profile", "abs")),
        "support_material_type": str(payload.get("support_material_type", "none")),
        "infill_percent": str(payload.get("infill_percent", 20.0)),
        "perimeter_count": str(payload.get("perimeter_count", 5)),
        "top_layers": str(payload.get("top_layers", 5)),
        "bottom_layers": str(payload.get("bottom_layers", 5)),
        "material_density_g_cm3": str(payload.get("material_density_g_cm3", 0.0)),
        "material_price_eur_per_kg": str(payload.get("material_price_eur_per_kg", 0.0)),
        "support_density_g_cm3": str(payload.get("support_density_g_cm3", 0.0)),
        "support_price_eur_per_kg": str(payload.get("support_price_eur_per_kg", 0.0)),
        "material_display_name": str(payload.get("material_display_name", "")),
        "support_material_display_name": str(payload.get("support_material_display_name", "")),
    }

    try:
        response = requests.post(
            f"{ORCA_WORKER_URL}/slice",
            files=files,
            data=data,
            timeout=ORCA_WORKER_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise OrcaClientError(f"orca worker request failed: {exc}") from exc

    try:
        payload = response.json()
    except Exception:
        raise OrcaClientError(f"invalid worker response: {response.text}")

    if response.status_code >= 400 or not payload.get("success"):
        details = payload.get("details") or payload.get("error") or response.text
        raise OrcaClientError(f"orca worker failed: {details}")

    return payload