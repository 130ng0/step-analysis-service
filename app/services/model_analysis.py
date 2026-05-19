from __future__ import annotations

import base64
import io
import os
import tempfile
import math
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from app.config import ALLOWED_EXTENSIONS
from PIL import Image, ImageChops
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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


def _view_direction(elev_deg: float = 22.0, azim_deg: float = -55.0):
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)

    return np.array([
        math.cos(elev) * math.cos(azim),
        math.cos(elev) * math.sin(azim),
        math.sin(elev),
    ])


def _extract_feature_edges(mesh: trimesh.Trimesh, angle_threshold_deg: float = 35.0):
    edges = []

    try:
        view_dir = _view_direction()
        face_normals = mesh.face_normals
        threshold = math.cos(math.radians(angle_threshold_deg))

        for edge, faces in zip(mesh.face_adjacency_edges, mesh.face_adjacency):
            n1 = face_normals[faces[0]]
            n2 = face_normals[faces[1]]

            # nur Kanten zeichnen, wenn mindestens eine angrenzende Fläche zur Kamera zeigt
            visible = (np.dot(n1, view_dir) > -0.15) or (np.dot(n2, view_dir) > -0.15)
            if not visible:
                continue

            dot = float(np.dot(n1, n2))

            # nur harte/markante Kanten, kein STL-Dreiecksnetz
            if dot < threshold:
                p1 = mesh.vertices[edge[0]]
                p2 = mesh.vertices[edge[1]]
                edges.append([p1, p2])

    except Exception:
        return []

    return edges


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

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        poly = Poly3DCollection(
            triangles,
            linewidths=0,
            edgecolors="none",
        )

        # komplett deckende Oberfläche
        poly.set_facecolor((0.72, 0.72, 0.78, 1.0))
        poly.set_alpha(1.0)

        ax.add_collection3d(poly)

        feature_edges = _extract_feature_edges(
            mesh,
            angle_threshold_deg=35.0,
        )

        if feature_edges:
            edge_collection = Line3DCollection(
                feature_edges,
                colors=[(0.15, 0.15, 0.18, 0.85)],
                linewidths=0.9,
            )
            ax.add_collection3d(edge_collection)

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
            facecolor="white",
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