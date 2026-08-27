from __future__ import annotations

import asyncio
import json
import logging
import math
import time

from app.services.model_analysis import render_preview_from_converted_stl_bytes
from app.services.orca_client import OrcaClientError, slice_with_worker
from app.services.slice_input_converter import SliceInputConversionError, convert_upload_to_stl_bytes
from app.job_store import (
    claim_next_job,
    cleanup_job_files,
    get_job_dir,
    mark_done,
    mark_error,
    update_job_progress,
)

logger = logging.getLogger("step-analysis-service.worker")


def process_job_sync(job: dict) -> None:
    job_id = job["job_id"]
    request_payload = job["request_json"]
    filename = job["filename"]

    job_dir = get_job_dir(job_id)
    input_path = job_dir / "input.bin"

    started_ts = time.time()

    try:
        update_job_progress(job_id, phase="converting", progress_percent=5, eta_seconds=90)

        file_bytes = input_path.read_bytes()

        stl_bytes, stl_filename = convert_upload_to_stl_bytes(file_bytes, filename)

        update_job_progress(job_id, phase="rendering_preview", progress_percent=20, eta_seconds=70)

        preview_png_base64 = render_preview_from_converted_stl_bytes(stl_bytes)
        if preview_png_base64:
            (job_dir / "preview_base64.txt").write_text(preview_png_base64, encoding="utf-8")

        update_job_progress(job_id, phase="slicing", progress_percent=35, eta_seconds=60)

        worker_payload = slice_with_worker(
            stl_bytes=stl_bytes,
            stl_filename=stl_filename,
            payload=request_payload,
        )

        elapsed = max(1, int(time.time() - started_ts))
        remaining_guess = max(0, 5 - min(5, elapsed // 10))
        update_job_progress(
            job_id,
            phase="finalizing",
            progress_percent=90,
            eta_seconds=remaining_guess,
        )

        print_time_hours = float(worker_payload.get("print_time_hours") or 0.0)
        machine_hour_rate_eur = float(request_payload.get("machine_hour_rate_eur") or 8.0)
        margin_factor = float(request_payload.get("margin_factor") or 1.0)
        material_cost_eur_total = float(worker_payload.get("material_cost_eur_total") or 0.0)

        machine_cost_eur = round(print_time_hours * machine_hour_rate_eur, 2)
        subtotal_cost_eur = round(machine_cost_eur + material_cost_eur_total, 2)
        total_price_eur = round(subtotal_cost_eur * margin_factor, 2)

        result = {
            "success": True,
            "filename": filename,
            "method": "slice",
            "material_profile": request_payload.get("material_profile"),
            "support_material_type": request_payload.get("support_material_type"),
            "unit": "mm",
            "machine_hour_rate_eur": machine_hour_rate_eur,
            "margin_factor": margin_factor,
            "print_time_minutes": worker_payload.get("print_time_minutes", 0),
            "print_time_hours": print_time_hours,
            "filament_length_mm_total": round(float(worker_payload.get("filament_length_mm_total", 0) or 0), 3),
            "filament_volume_cm3_total": round(float(worker_payload.get("filament_volume_cm3_total", 0) or 0), 3),
            "filament_weight_g_total": round(float(worker_payload.get("filament_weight_g_total", 0) or 0), 3),
            "material_cost_eur_total": round(material_cost_eur_total, 2),
            "machine_cost_eur": machine_cost_eur,
            "subtotal_cost_eur": subtotal_cost_eur,
            "total_price_eur": total_price_eur,
            "model_filament_length_mm": round(float(worker_payload.get("model_filament_length_mm", 0) or 0), 3),
            "support_filament_length_mm": round(float(worker_payload.get("support_filament_length_mm", 0) or 0), 3),
            "model_filament_volume_cm3": round(float(worker_payload.get("model_filament_volume_cm3", 0) or 0), 3),
            "support_filament_volume_cm3": round(float(worker_payload.get("support_filament_volume_cm3", 0) or 0), 3),
            "model_filament_weight_g": round(float(worker_payload.get("model_filament_weight_g", 0) or 0), 3),
            "support_filament_weight_g": round(float(worker_payload.get("support_filament_weight_g", 0) or 0), 3),
            "model_material_cost_eur": round(float(worker_payload.get("model_material_cost_eur", 0) or 0), 2),
            "support_material_cost_eur": round(float(worker_payload.get("support_material_cost_eur", 0) or 0), 2),
            "applied_slicer_settings": {
                "infill_percent": request_payload.get("infill_percent"),
                "perimeter_count": request_payload.get("perimeter_count"),
                "top_layers": request_payload.get("top_layers"),
                "bottom_layers": request_payload.get("bottom_layers"),
            },
            "tools": worker_payload.get("tools", []),
            "preview_png_base64": preview_png_base64,
        }

        mark_done(job_id, result)

    except SliceInputConversionError as exc:
        mark_error(job_id, "SLICE_INPUT_CONVERSION_FAILED", str(exc))
    except OrcaClientError as exc:
        mark_error(job_id, "SLICE_FAILED", str(exc))
    except Exception as exc:
        logger.exception("analysis_job_failed job_id=%s filename=%s", job_id, filename)
        mark_error(job_id, "INTERNAL_SERVER_ERROR", str(exc))
    finally:
        # Binary input, converted artefacts and preview/result helper files must
        # never accumulate on the service. The terminal result/error itself is
        # already persisted in SQLite before this cleanup and can still be
        # fetched idempotently by Odoo.
        cleanup_job_files(job_id)


async def worker_loop(worker_id: int, poll_seconds: float = 1.0) -> None:
    logger.info("analysis_worker_started worker_id=%s", worker_id)
    try:
        while True:
            job = claim_next_job()
            if not job:
                await asyncio.sleep(poll_seconds)
                continue

            logger.info(
                "analysis_worker_claimed worker_id=%s job_id=%s filename=%s",
                worker_id,
                job.get("job_id"),
                job.get("filename"),
            )
            await asyncio.to_thread(process_job_sync, job)
    except asyncio.CancelledError:
        logger.info("analysis_worker_stopped worker_id=%s", worker_id)
        raise
