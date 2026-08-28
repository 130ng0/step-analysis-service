from __future__ import annotations

import base64
import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading

from app.config import ALLOWED_EXTENSIONS
from PIL import Image, ImageChops

_PREVIEW_RENDER_LOCK = threading.Lock()
logger = logging.getLogger("step-analysis-service.preview")
PREVIEW_RENDER_TIMEOUT_SECONDS = max(10, int(os.getenv("PREVIEW_RENDER_TIMEOUT_SECONDS", "120")))


class ModelAnalysisError(Exception):
    pass


class UnsupportedFileFormatError(ModelAnalysisError):
    pass


def _detect_format(filename: str) -> str:
    lower = (filename or "").lower()
    if lower.endswith((".step", ".stp")):
        return "step"
    if lower.endswith(".stl"):
        return "stl"
    raise UnsupportedFileFormatError(f"Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}")


def _crop_png_whitespace(png_bytes: bytes, padding: int = 20) -> bytes:
    try:
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        bg = Image.new("RGB", image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -10)
        bbox = diff.getbbox()
        if not bbox:
            return png_bytes
        left, upper, right, lower = bbox
        left = max(left - padding, 0)
        upper = max(upper - padding, 0)
        right = min(right + padding, image.width)
        lower = min(lower + padding, image.height)
        cropped = image.crop((left, upper, right, lower))
        out = io.BytesIO()
        cropped.save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception:
        logger.exception("preview_crop_failed")
        return png_bytes


def _renderer_command(stl_path: str, png_path: str) -> list[str]:
    base = [sys.executable, "-m", "app.services.preview_renderer_worker", stl_path, png_path]
    xvfb_run = shutil.which("xvfb-run")
    if xvfb_run:
        return [xvfb_run, "-a", "-s", "-screen 0 1024x768x24", *base]
    # Local/dev fallback. The renderer is still isolated in a child process;
    # production Docker installs xvfb-run explicitly.
    return base


def _render_stl_preview_png_base64_isolated(stl_bytes: bytes) -> str | None:
    if not stl_bytes:
        return None

    with tempfile.TemporaryDirectory(prefix="nevo-preview-") as tmp_dir:
        stl_path = os.path.join(tmp_dir, "input.stl")
        png_path = os.path.join(tmp_dir, "preview.png")
        with open(stl_path, "wb") as handle:
            handle.write(stl_bytes)

        command = _renderer_command(stl_path, png_path)
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=PREVIEW_RENDER_TIMEOUT_SECONDS,
                check=False,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            logger.error("preview_renderer_timeout timeout_seconds=%s", PREVIEW_RENDER_TIMEOUT_SECONDS)
            return None
        except Exception:
            logger.exception("preview_renderer_launch_failed")
            return None

        if completed.returncode != 0:
            # A negative return code indicates termination by signal (including
            # a native VTK/OpenGL crash). Crucially, only this child dies.
            logger.error(
                "preview_renderer_failed returncode=%s stderr=%s",
                completed.returncode,
                (completed.stderr or "")[-4000:],
            )
            return None

        if not os.path.exists(png_path) or os.path.getsize(png_path) == 0:
            logger.error("preview_renderer_missing_output")
            return None

        with open(png_path, "rb") as handle:
            png_bytes = handle.read()
        png_bytes = _crop_png_whitespace(png_bytes, padding=20)
        return base64.b64encode(png_bytes).decode("ascii")


def render_stl_preview_png_base64(stl_bytes: bytes) -> str | None:
    """Render one preview in an isolated process.

    VTK/OpenGL is native code and may abort/segfault in a headless environment.
    Keeping it out of the FastAPI process guarantees that a renderer crash can
    never take down port 5050 or the analysis worker pool. Rendering remains
    serialized to keep memory usage predictable while slicing stays parallel.
    """
    with _PREVIEW_RENDER_LOCK:
        return _render_stl_preview_png_base64_isolated(stl_bytes)


def render_preview_from_converted_stl_bytes(stl_bytes: bytes) -> str | None:
    return render_stl_preview_png_base64(stl_bytes)
