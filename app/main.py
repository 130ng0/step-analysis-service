from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Literal

import requests
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES
from app.schemas import ErrorResponse
from app.security import verify_api_key
from app.services.model_analysis import (
    ModelAnalysisError,
    UnsupportedFileFormatError,
    add_extrusion_estimate,
    add_material_estimate,
    add_price_estimate,
    add_runtime_estimate,
    add_support_estimate,
    analyze_model_file_bytes,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("step-analysis-service")

app = FastAPI(
    title="3D Model Analysis Service",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

ORCA_WORKER_URL = os.getenv("ORCA_WORKER_URL", "http://orca-worker:8090")


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
        if r.ok:
            worker_status = "ok"
        else:
            worker_status = f"error:{r.status_code}"
    except Exception:
        worker_status = "unreachable"

    return {
        "status": "ok",
        "orca_worker": worker_status,
    }


@app.post(
    "/analyze-model",
    response_model=None,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    dependencies=[Depends(verify_api_key)],
)
async def analyze_model(
    file: UploadFile = File(...),
    calculation_method: Literal["slice", "fast"] = Form(default="slice"),
    material_profile: Literal["abs", "abs_cf", "abs_esd", "asa", "pc", "pc_cf", "pc_fr", "tpu"] = Form(default="abs"),
    unit: str = Form(default="mm"),
    machine_hour_rate_eur: float | None = Form(default=8.0),
    volumetric_flow_mm3_s: float | None = Form(default=15.0),
    support_angle_deg: float | None = Form(default=45.0),
    support_density_factor: float | None = Form(default=0.22),
    orientation_mode: Literal["auto", "fixed"] = Form(default="auto"),
    infill_percent: float = Form(default=20.0),
    perimeter_count: int = Form(default=5),
    top_layers: int = Form(default=5),
    bottom_layers: int = Form(default=5),
    line_width_mm: float = Form(default=0.45),
    layer_height_mm: float = Form(default=0.20),
    part_material_name: str = Form(default="ABS"),
    part_material_price_per_kg_eur: float | None = Form(default=28.0),
    part_material_density_g_cm3: float | None = Form(default=1.04),
    support_material_type: Literal["none", "breakaway", "hips", "soluble"] = Form(default="breakaway"),
    support_material_price_per_kg_eur: str | None = Form(default=None),
    support_material_density_g_cm3: str | None = Form(default=None),
    margin_factor: float | None = Form(default=1.0),
):
    filename = file.filename or "model.step"

    if unit not in {"mm", "cm", "m"}:
        raise HTTPException(status_code=400, detail="unit must be one of: mm, cm, m")

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

    # ------------------------------------------------------------------
    # SLICE MODE (DEFAULT)
    # ------------------------------------------------------------------
    if calculation_method == "slice":
        try:
            files = {
                "file": (filename, file_bytes, "application/octet-stream"),
            }
            data = {
                "material_profile": material_profile,
            }

            response = requests.post(
                f"{ORCA_WORKER_URL}/slice",
                files=files,
                data=data,
                timeout=300,
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

            machine_cost_eur = round(print_time_hours * float(machine_hour_rate_eur or 0.0), 2)
            subtotal_cost_eur = round(machine_cost_eur + material_cost_eur_total, 2)
            total_price_eur = round(subtotal_cost_eur * float(margin_factor or 1.0), 2)

            return {
                "success": True,
                "filename": filename,
                "method": "slice",
                "material_profile": material_profile,
                "unit": unit,
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
                "tools": payload.get("tools", []),
            }

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

    # ------------------------------------------------------------------
    # FAST MODE (EXISTING LOGIC)
    # ------------------------------------------------------------------
    try:
        result = analyze_model_file_bytes(
            file_bytes=file_bytes,
            filename=filename,
            unit=unit,
        )
        result = add_support_estimate(
            result=result,
            file_bytes=file_bytes,
            filename=filename,
            support_angle_deg=support_angle_deg,
            support_density_factor=support_density_factor,
            support_material_type=support_material_type,
            orientation_mode=orientation_mode,
        )
        result = add_extrusion_estimate(
            result=result,
            infill_percent=infill_percent,
            perimeter_count=perimeter_count,
            top_layers=top_layers,
            bottom_layers=bottom_layers,
            line_width_mm=line_width_mm,
            layer_height_mm=layer_height_mm,
            support_material_type=support_material_type,
        )
        result = add_runtime_estimate(
            result=result,
            machine_hour_rate_eur=machine_hour_rate_eur,
            volumetric_flow_mm3_s=volumetric_flow_mm3_s,
            support_material_type=support_material_type,
        )
        result = add_material_estimate(
            result=result,
            part_material_name=part_material_name,
            part_material_price_per_kg_eur=part_material_price_per_kg_eur,
            part_material_density_g_cm3=part_material_density_g_cm3,
            support_material_type=support_material_type,
            support_material_price_per_kg_eur=support_material_price_per_kg_eur,
            support_material_density_g_cm3=support_material_density_g_cm3,
        )
        result = add_price_estimate(
            result=result,
            margin_factor=margin_factor,
        )
        result["method"] = "fast"
        result["material_profile"] = material_profile
        return result

    except UnsupportedFileFormatError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "UNSUPPORTED_FILE_FORMAT",
                "details": str(exc),
                "filename": filename,
            },
        )
    except ModelAnalysisError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "MODEL_ANALYSIS_FAILED",
                "details": str(exc),
                "filename": filename,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("unexpected_error filename=%s", filename)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "INTERNAL_SERVER_ERROR",
                "details": str(exc),
                "filename": filename,
            },
        )