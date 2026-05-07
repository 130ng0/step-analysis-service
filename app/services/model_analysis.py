from __future__ import annotations

import base64
import io
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from app.config import ALLOWED_EXTENSIONS
from PIL import Image, ImageChops

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
    raise UnsupportedFileFormatError(
        f"Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
    )


def _load_stl_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, file_type="stl")

    if mesh is None:
        raise ModelAnalysisError("No mesh could be imported from STL file")

    if isinstance(mesh, trimesh.Scene):
        geometries = [g for g in mesh.geometry.values() if g is not None and not g.is_empty]
        if not geometries:
            raise ModelAnalysisError("STL scene does not contain usable mesh geometry")
        mesh = trimesh.util.concatenate(geometries)

    if mesh.is_empty:
        raise ModelAnalysisError("Imported STL mesh is empty")

    if not isinstance(mesh, trimesh.Trimesh):
        raise ModelAnalysisError("Imported STL is not a valid mesh")

    return mesh


def _crop_png_whitespace(png_bytes: bytes, padding: int = 20) -> bytes:
    """
    Entfernt weiße/fast weiße Ränder aus einem PNG und fügt etwas Padding hinzu.
    """
    try:
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")

        bg = Image.new("RGB", image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image, bg)

        # etwas verstärken, damit sehr helle Unterschiede erkannt werden
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
        return png_bytes


def render_stl_preview_png_base64(stl_bytes: bytes) -> str | None:
    """
    Rendert ein einfaches PNG-Preview aus STL-Bytes.
    Rückgabe als base64-String ohne data:-Prefix.
    """
    if not stl_bytes:
        return None

    tmp_path = None
    fig = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
            tmp.write(stl_bytes)
            tmp_path = tmp.name

        mesh = _load_stl_mesh(tmp_path)

        if mesh.is_empty or len(mesh.faces) == 0:
            return None

        mesh = mesh.copy()
        mesh.apply_translation(-mesh.bounding_box.centroid)

        vertices = mesh.vertices
        faces = mesh.faces
        triangles = vertices[faces]

        fig = plt.figure(figsize=(5, 5), dpi=160)
        ax = fig.add_subplot(111, projection="3d")

        poly = Poly3DCollection(
            triangles,
            linewidths=0.05,
            alpha=1.0,
        )
        poly.set_edgecolor((0.2, 0.2, 0.2, 0.15))
        poly.set_facecolor((0.70, 0.70, 0.78, 1.0))
        ax.add_collection3d(poly)

        bounds = mesh.bounds
        mins = bounds[0]
        maxs = bounds[1]
        center = (mins + maxs) / 2.0
        size = float((maxs - mins).max())
        if size <= 0:
            size = 1.0
        half = size / 2.0

        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)

        ax.view_init(elev=22, azim=-55)
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        buf = io.BytesIO()
        plt.savefig(
            buf,
            format="png",
            bbox_inches="tight",
            pad_inches=0.0,
            transparent=False,
        )

        png_bytes = _crop_png_whitespace(buf.getvalue(), padding=18)

        return base64.b64encode(png_bytes).decode("ascii")

    except Exception:
        return None

    finally:
        if fig is not None:
            plt.close(fig)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def render_preview_from_converted_stl_bytes(stl_bytes: bytes) -> str | None:
    """
    Erwartet bereits konvertierte STL-Bytes und rendert daraus ein PNG-Preview.
    """
    return render_stl_preview_png_base64(stl_bytes)