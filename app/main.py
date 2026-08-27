from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Literal

import asyncio
import requests
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import (
    ALLOWED_EXTENSIONS,
    ANALYSIS_MAX_WORKERS,
    JOB_CLEANUP_INTERVAL_SECONDS,
    JOB_RESULT_RETENTION_SECONDS,
    MAX_FILE_SIZE_BYTES,
)
from app.schemas import ErrorResponse
from app.security import verify_api_key
from app.services.slice_input_converter import SliceInputConversionError, convert_upload_to_stl_bytes
from app.services.model_analysis import render_preview_from_converted_stl_bytes

from app.job_store import (
    cleanup_expired_terminal_jobs,
    create_job,
    delete_job,
    get_job,
    get_job_dir,
    get_queue_position,
    init_db,
    mark_consumed,
    requeue_interrupted_jobs,
)

from app.worker_loop import worker_loop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("step-analysis-service")

app = FastAPI(
    title="3D Model Analysis Service",
    version="2.10.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

ORCA_WORKER_URL = os.getenv("ORCA_WORKER_URL", "http://orca-worker:8090")
ORCA_WORKER_TIMEOUT = int(os.getenv("ORCA_WORKER_TIMEOUT", "1800"))


async def terminal_job_cleanup_loop() -> None:
    while True:
        await asyncio.sleep(JOB_CLEANUP_INTERVAL_SECONDS)
        try:
            deleted = await asyncio.to_thread(
                cleanup_expired_terminal_jobs, JOB_RESULT_RETENTION_SECONDS
            )
            if deleted:
                logger.info("terminal_job_retention_cleanup deleted=%s", deleted)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("terminal_job_retention_cleanup_failed")


@app.on_event("startup")
async def startup_event():
    init_db()
    requeued = requeue_interrupted_jobs()
    if requeued:
        logger.warning("analysis_jobs_requeued_after_restart count=%s", requeued)

    app.state.analysis_worker_tasks = [
        asyncio.create_task(worker_loop(worker_id))
        for worker_id in range(1, ANALYSIS_MAX_WORKERS + 1)
    ]
    app.state.job_cleanup_task = asyncio.create_task(terminal_job_cleanup_loop())
    logger.info(
        "analysis_worker_pool_started workers=%s result_retention_seconds=%s",
        ANALYSIS_MAX_WORKERS,
        JOB_RESULT_RETENTION_SECONDS,
    )


@app.on_event("shutdown")
async def shutdown_event():
    tasks = list(getattr(app.state, "analysis_worker_tasks", []))
    cleanup_task = getattr(app.state, "job_cleanup_task", None)
    if cleanup_task:
        tasks.append(cleanup_task)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("analysis_worker_pool_stopped tasks=%s", len(tasks))


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.exception(
            "request_failed request_id=%s path=%s duration_ms=%s",
            request_id,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_done request_id=%s method=%s path=%s status=%s duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/health")
def health():
    worker_status = "unknown"
    try:
        r = requests.get(f"{ORCA_WORKER_URL}/health", timeout=5)
        worker_status = "ok" if r.ok else f"error:{r.status_code}"
    except Exception:
        worker_status = "unreachable"

    tasks = list(getattr(app.state, "analysis_worker_tasks", []))
    alive_workers = sum(1 for task in tasks if not task.done())

    return {
        "status": "ok",
        "orca_worker": worker_status,
        "analysis_workers_configured": ANALYSIS_MAX_WORKERS,
        "analysis_workers_alive": alive_workers,
        "job_result_retention_seconds": JOB_RESULT_RETENTION_SECONDS,
    }


@app.post(
    "/analyze-model/jobs",
    response_model=None,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    dependencies=[Depends(verify_api_key)],
)
async def create_analysis_job(
    file: UploadFile = File(...),
    material_profile: Literal["abs", "abs_cf", "abs_esd", "asa", "pc", "pc_cf", "pc_fr", "tpu"] = Form(default="abs"),
    support_material_type: Literal["none", "breakaway", "hips", "soluble"] = Form(default="breakaway"),
    infill_percent: float = Form(default=20.0),
    perimeter_count: int = Form(default=5),
    top_layers: int = Form(default=5),
    bottom_layers: int = Form(default=5),
    machine_hour_rate_eur: float = Form(default=8.0),
    margin_factor: float = Form(default=1.0),
    material_density_g_cm3: float = Form(default=0.0),
    material_price_eur_per_kg: float = Form(default=0.0),
    support_density_g_cm3: float = Form(default=0.0),
    support_price_eur_per_kg: float = Form(default=0.0),
    material_display_name: str = Form(default=""),
    support_material_display_name: str = Form(default=""),
):
    filename = file.filename or "model.step"

    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "UNSUPPORTED_FILE_FORMAT",
                "details": f"Only {', '.join(ALLOWED_EXTENSIONS)} files are supported",
                "filename": filename,
            },
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "FILE_TOO_LARGE",
                "details": f"Maximum allowed size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB",
                "filename": filename,
            },
        )

    request_payload = {
        "material_profile": material_profile,
        "support_material_type": support_material_type,
        "infill_percent": infill_percent,
        "perimeter_count": perimeter_count,
        "top_layers": top_layers,
        "bottom_layers": bottom_layers,
        "machine_hour_rate_eur": machine_hour_rate_eur,
        "margin_factor": margin_factor,
        "material_density_g_cm3": material_density_g_cm3,
        "material_price_eur_per_kg": material_price_eur_per_kg,
        "support_density_g_cm3": support_density_g_cm3,
        "support_price_eur_per_kg": support_price_eur_per_kg,
        "material_display_name": material_display_name,
        "support_material_display_name": support_material_display_name,
    }

    job_id = create_job(
        filename=filename,
        request_payload=request_payload,
        file_bytes=file_bytes,
    )

    return {
        "success": True,
        "job_id": job_id,
        "status": "queued",
    }


@app.get(
    "/analyze-model/jobs/{job_id}",
    response_model=None,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    dependencies=[Depends(verify_api_key)],
)
def get_analysis_job(job_id: str):
    job = get_job(job_id)
    if not job:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": "JOB_NOT_FOUND",
                "details": f"Job not found: {job_id}",
            },
        )

    status = job["status"]

    if status == "queued":
        job_dir = get_job_dir(job_id)
        preview_png_base64 = None
        preview_path = job_dir / "preview_base64.txt"
        if preview_path.exists():
            preview_png_base64 = preview_path.read_text(encoding="utf-8")

        return {
            "success": True,
            "job_id": job_id,
            "status": "queued",
            "phase": job.get("phase") or "queued",
            "progress_percent": job.get("progress_percent") or 0,
            "eta_seconds": job.get("eta_seconds"),
            "queue_position": get_queue_position(job_id),
            "preview_png_base64": preview_png_base64,
        }

    if status == "processing":
        job_dir = get_job_dir(job_id)
        preview_png_base64 = None
        preview_path = job_dir / "preview_base64.txt"
        if preview_path.exists():
            preview_png_base64 = preview_path.read_text(encoding="utf-8")

        return {
            "success": True,
            "job_id": job_id,
            "status": "processing",
            "phase": job.get("phase") or "processing",
            "progress_percent": job.get("progress_percent") or 0,
            "eta_seconds": job.get("eta_seconds"),
            "preview_png_base64": preview_png_base64,
        }

    if status == "done":
        result = job.get("result_json") or {}
        return {
            "success": True,
            "job_id": job_id,
            "status": "done",
            "result": result,
        }

    return {
        "success": False,
        "job_id": job_id,
        "status": "error",
        "error": job.get("error_code") or "JOB_FAILED",
        "details": job.get("error_details") or "Unknown job error",
    }


@app.delete(
    "/analyze-model/jobs/{job_id}",
    response_model=None,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    dependencies=[Depends(verify_api_key)],
)
def delete_analysis_job(job_id: str):
    job = get_job(job_id)
    if not job:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": "JOB_NOT_FOUND",
                "details": f"Job not found: {job_id}",
            },
        )

    mark_consumed(job_id)
    deleted = delete_job(job_id)

    return {
        "success": True,
        "job_id": job_id,
        "deleted": bool(deleted),
    }


@app.post(
    "/analyze-model",
    response_model=None,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    dependencies=[Depends(verify_api_key)],
)
async def analyze_model(
        file: UploadFile = File(...),
        material_profile: Literal["abs", "abs_cf", "abs_esd", "asa", "pc", "pc_cf", "pc_fr", "tpu"] = Form(
            default="abs"),
        support_material_type: Literal["none", "breakaway", "hips", "soluble"] = Form(default="breakaway"),
        infill_percent: float = Form(default=20.0),
        perimeter_count: int = Form(default=5),
        top_layers: int = Form(default=5),
        bottom_layers: int = Form(default=5),
        machine_hour_rate_eur: float = Form(default=8.0),
        margin_factor: float = Form(default=1.0),
        material_density_g_cm3: float = Form(default=0.0),
        material_price_eur_per_kg: float = Form(default=0.0),
        support_density_g_cm3: float = Form(default=0.0),
        support_price_eur_per_kg: float = Form(default=0.0),
        material_display_name: str = Form(default=""),
        support_material_display_name: str = Form(default=""),
):
    filename = file.filename or "model.step"

    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "UNSUPPORTED_FILE_FORMAT",
                "details": f"Only {', '.join(ALLOWED_EXTENSIONS)} files are supported",
                "filename": filename,
            },
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "FILE_TOO_LARGE",
                "details": f"Maximum allowed size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB",
                "filename": filename,
            },
        )

    try:
        stl_bytes, stl_filename = convert_upload_to_stl_bytes(file_bytes, filename)
        preview_png_base64 = render_preview_from_converted_stl_bytes(stl_bytes)

        worker_support_mode = support_material_type

        files = {
            "file": (stl_filename, stl_bytes, "application/octet-stream"),
        }
        data = {
            "material_profile": material_profile,
            "support_material_type": worker_support_mode,
            "infill_percent": str(infill_percent),
            "perimeter_count": str(perimeter_count),
            "top_layers": str(top_layers),
            "bottom_layers": str(bottom_layers),

            "material_density_g_cm3": str(material_density_g_cm3),
            "material_price_eur_per_kg": str(material_price_eur_per_kg),
            "support_density_g_cm3": str(support_density_g_cm3),
            "support_price_eur_per_kg": str(support_price_eur_per_kg),
            "material_display_name": material_display_name,
            "support_material_display_name": support_material_display_name,
        }

        response = requests.post(
            f"{ORCA_WORKER_URL}/slice",
            files=files,
            data=data,
            timeout=ORCA_WORKER_TIMEOUT,
        )

        try:
            payload = response.json()
        except Exception:
            payload = {
                "success": False,
                "error": "INVALID_WORKER_RESPONSE",
                "details": response.text,
                "filename": filename,
            }

        if response.status_code >= 400 or not payload.get("success"):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "SLICE_FAILED",
                    "details": payload.get("details") or payload.get("error") or "Unknown Orca worker error",
                    "filename": filename,
                },
            )

        print_time_hours = float(payload.get("print_time_hours") or 0.0)
        material_cost_eur_total = float(payload.get("material_cost_eur_total") or 0.0)
        filament_weight_g_total = float(payload.get("filament_weight_g_total") or 0.0)
        filament_volume_cm3_total = float(payload.get("filament_volume_cm3_total") or 0.0)
        filament_length_mm_total = float(payload.get("filament_length_mm_total") or 0.0)

        machine_cost_eur = round(print_time_hours * machine_hour_rate_eur, 2)
        subtotal_cost_eur = round(machine_cost_eur + material_cost_eur_total, 2)
        total_price_eur = round(subtotal_cost_eur * margin_factor, 2)

        return {
            "success": True,
            "filename": filename,
            "method": "slice",
            "material_profile": material_profile,
            "support_material_type": worker_support_mode,
            "unit": "mm",
            "machine_hour_rate_eur": machine_hour_rate_eur,
            "margin_factor": margin_factor,
            "print_time_minutes": payload.get("print_time_minutes", 0),
            "print_time_hours": print_time_hours,
            "filament_length_mm_total": round(filament_length_mm_total, 3),
            "filament_volume_cm3_total": round(filament_volume_cm3_total, 3),
            "filament_weight_g_total": round(filament_weight_g_total, 3),
            "material_cost_eur_total": round(material_cost_eur_total, 2),
            "machine_cost_eur": machine_cost_eur,
            "subtotal_cost_eur": subtotal_cost_eur,
            "total_price_eur": total_price_eur,

            "model_filament_length_mm": round(float(payload.get("model_filament_length_mm", 0) or 0), 3),
            "support_filament_length_mm": round(float(payload.get("support_filament_length_mm", 0) or 0), 3),
            "model_filament_volume_cm3": round(float(payload.get("model_filament_volume_cm3", 0) or 0), 3),
            "support_filament_volume_cm3": round(float(payload.get("support_filament_volume_cm3", 0) or 0), 3),
            "model_filament_weight_g": round(float(payload.get("model_filament_weight_g", 0) or 0), 3),
            "support_filament_weight_g": round(float(payload.get("support_filament_weight_g", 0) or 0), 3),
            "model_material_cost_eur": round(float(payload.get("model_material_cost_eur", 0) or 0), 2),
            "support_material_cost_eur": round(float(payload.get("support_material_cost_eur", 0) or 0), 2),

            "applied_slicer_settings": payload.get("applied_slicer_settings", {}),
            "tools": payload.get("tools", []),
            "preview_png_base64": preview_png_base64,
        }

    except SliceInputConversionError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "SLICE_INPUT_CONVERSION_FAILED",
                "details": str(exc),
                "filename": filename,
            },
        )
    except requests.RequestException as exc:
        logger.exception("orca_worker_request_failed filename=%s", filename)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "ORCA_WORKER_UNREACHABLE",
                "details": str(exc),
                "filename": filename,
            },
        )
    except Exception as exc:
        logger.exception("slice_mode_unexpected_error filename=%s", filename)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "INTERNAL_SERVER_ERROR",
                "details": str(exc),
                "filename": filename,
            },
        )


@app.post(
    "/render-preview",
    response_model=None,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    dependencies=[Depends(verify_api_key)],
)
async def render_preview(
    file: UploadFile = File(...),
):
    filename = file.filename or "model.step"

    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "UNSUPPORTED_FILE_FORMAT",
                "details": f"Only {', '.join(ALLOWED_EXTENSIONS)} files are supported",
                "filename": filename,
            },
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "FILE_TOO_LARGE",
                "details": f"Maximum allowed size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB",
                "filename": filename,
            },
        )

    try:
        stl_bytes, stl_filename = convert_upload_to_stl_bytes(file_bytes, filename)
        preview_png_base64 = render_preview_from_converted_stl_bytes(stl_bytes)

        if not preview_png_base64:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "PREVIEW_RENDER_FAILED",
                    "details": "Preview image could not be rendered.",
                    "filename": filename,
                },
            )

        return {
            "success": True,
            "filename": filename,
            "rendered_filename": "preview.png",
            "source_stl_filename": stl_filename,
            "mime_type": "image/png",
            "preview_png_base64": preview_png_base64,
        }

    except SliceInputConversionError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "SLICE_INPUT_CONVERSION_FAILED",
                "details": str(exc),
                "filename": filename,
            },
        )
    except Exception as exc:
        logger.exception("preview_render_unexpected_error filename=%s", filename)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "INTERNAL_SERVER_ERROR",
                "details": str(exc),
                "filename": filename,
            },
        )