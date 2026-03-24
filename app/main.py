from __future__ import annotations

import logging
import time
import uuid

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES
from app.schemas import AnalyzeResponse, ErrorResponse
from app.security import verify_api_key
from app.services.model_analysis import (
    ModelAnalysisError,
    UnsupportedFileFormatError,
    add_material_estimate,
    add_price_estimate,
    add_runtime_estimate,
    add_support_estimate,
    analyze_model_file_bytes,
)
from typing import Literal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("step-analysis-service")

app = FastAPI(
    title="3D Model Analysis Service",
    version="1.6.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


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
    return {"status": "ok"}


@app.post(
    "/analyze-model",
    response_model=AnalyzeResponse,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    dependencies=[Depends(verify_api_key)],
)
async def analyze_model(
    file: UploadFile = File(...),
    unit: str = Form(default="mm"),
    machine_hour_rate_eur: float | None = Form(default=8.0),
    volumetric_flow_mm3_s: float | None = Form(default=15.0),
    support_angle_deg: float | None = Form(default=45.0),
    support_density_factor: float | None = Form(default=0.22),
    part_material_name: str = Form(default="ABS"),
    part_material_price_per_kg_eur: float | None = Form(default=28.0),
    part_material_density_g_cm3: float | None = Form(default=1.04),
    support_material_type: Literal["breakaway", "hips", "soluble"] = Form(default="breakaway"),
    support_material_price_per_kg_eur: str | None = Form(default=None),
    support_material_density_g_cm3: str | None = Form(default=None),
    margin_factor: float | None = Form(default=1.0),
):
    file_bytes = await file.read()
    filename = file.filename or "model.step"
    MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "FILE_TOO_LARGE",
                "details": "Maximum allowed size is 100 MB",
                "filename": filename,
            },
        )

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

    try:
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
        )
        result = add_runtime_estimate(
            result=result,
            machine_hour_rate_eur=machine_hour_rate_eur,
            volumetric_flow_mm3_s=volumetric_flow_mm3_s,
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