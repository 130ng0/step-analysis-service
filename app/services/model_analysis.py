from __future__ import annotations

import math
import os
import tempfile
from typing import Dict, List, Tuple

import cadquery as cq
import numpy as np
import trimesh
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

from app.config import ALLOWED_EXTENSIONS


class ModelAnalysisError(Exception):
    pass


class UnsupportedFileFormatError(ModelAnalysisError):
    pass


DEFAULT_PART_MATERIAL_NAME = "ABS"
DEFAULT_PART_MATERIAL_PRICE_PER_KG_EUR = 28.0
DEFAULT_PART_MATERIAL_DENSITY_G_CM3 = 1.04

DEFAULT_SUPPORT_MATERIAL_TYPE = "breakaway"
DEFAULT_SUPPORT_BREAKAWAY_NAME = "ABS Breakaway"
DEFAULT_SUPPORT_HIPS_NAME = "HIPS"
DEFAULT_SUPPORT_SOLUBLE_NAME = "Soluble Support"

DEFAULT_SUPPORT_BREAKAWAY_PRICE_PER_KG_EUR = 28.0
DEFAULT_SUPPORT_HIPS_PRICE_PER_KG_EUR = 27.0 / 0.75  # 36 €/kg
DEFAULT_SUPPORT_SOLUBLE_PRICE_PER_KG_EUR = 80.0

DEFAULT_SUPPORT_BREAKAWAY_DENSITY_G_CM3 = 1.04
DEFAULT_SUPPORT_HIPS_DENSITY_G_CM3 = 1.03
DEFAULT_SUPPORT_SOLUBLE_DENSITY_G_CM3 = 1.20

def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"", "string", "null", "none"}
    return False

def _to_optional_float(value, field_name: str):
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ModelAnalysisError(f"{field_name} must be a valid number") from exc

def _detect_format(filename: str) -> str:
    lower = (filename or "").lower()
    if lower.endswith((".step", ".stp")):
        return "step"
    if lower.endswith(".stl"):
        return "stl"
    raise UnsupportedFileFormatError(
        f"Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
    )


def _load_step_shape(path: str):
    obj = cq.importers.importStep(path).val()
    if obj is None:
        raise ModelAnalysisError("No shape could be imported from STEP file")
    return obj


def _analyze_step(path: str, filename: str, unit: str) -> Dict:
    obj = _load_step_shape(path)

    volume = float(obj.Volume())
    area = float(obj.Area())
    bbox = obj.BoundingBox()

    if volume <= 0:
        raise ModelAnalysisError("No positive volume found in STEP model")

    return {
        "success": True,
        "filename": filename,
        "source_format": "step",
        "unit": unit,
        "volume_mm3": round(volume, 3),
        "volume_cm3": round(volume / 1000.0, 3),
        "surface_mm2": round(area, 3),
        "bbox_x_mm": round(float(bbox.xlen), 3),
        "bbox_y_mm": round(float(bbox.ylen), 3),
        "bbox_z_mm": round(float(bbox.zlen), 3),
        "solid_count": 1,
        "is_closed_solid": True,
        "warnings": [],
    }


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


def _analyze_stl(path: str, filename: str, unit: str) -> Dict:
    mesh = _load_stl_mesh(path)

    bbox = mesh.bounding_box.extents
    warnings: List[str] = []

    if not mesh.is_watertight:
        warnings.append("Mesh is not watertight. Volume may be unreliable")

    volume = float(mesh.volume)
    if volume < 0:
        warnings.append("Computed volume is negative. Mesh winding may be inverted")
        volume = abs(volume)

    return {
        "success": True,
        "filename": filename,
        "source_format": "stl",
        "unit": unit,
        "volume_mm3": round(volume, 3),
        "volume_cm3": round(volume / 1000.0, 3),
        "surface_mm2": round(float(mesh.area), 3),
        "bbox_x_mm": round(float(bbox[0]), 3),
        "bbox_y_mm": round(float(bbox[1]), 3),
        "bbox_z_mm": round(float(bbox[2]), 3),
        "solid_count": 1,
        "is_closed_solid": bool(mesh.is_watertight),
        "warnings": warnings,
    }


def analyze_model_file_bytes(file_bytes: bytes, filename: str, unit: str) -> Dict:
    if not file_bytes:
        raise ModelAnalysisError("Empty file")

    fmt = _detect_format(filename)
    suffix = os.path.splitext(filename)[1].lower() or ".tmp"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        path = tmp.name

    try:
        if fmt == "step":
            return _analyze_step(path, filename, unit)
        if fmt == "stl":
            return _analyze_stl(path, filename, unit)
        raise UnsupportedFileFormatError(
            f"Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _step_to_trimesh(path: str, linear_deflection: float = 0.15, angular_deflection: float = 0.5) -> trimesh.Trimesh:
    obj = _load_step_shape(path)
    shape = obj.wrapped

    mesher = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    mesher.Perform()

    vertices: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []

    explorer = TopExp_Explorer(shape, TopAbs_FACE)

    while explorer.More():
        face_shape = explorer.Current()
        face = TopoDS.Face_s(face_shape)

        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, loc)

        if triangulation is not None:
            transform = loc.Transformation()
            node_map = {}
            base_index = len(vertices)

            for i in range(1, triangulation.NbNodes() + 1):
                pnt = triangulation.Node(i).Transformed(transform)
                vertices.append((float(pnt.X()), float(pnt.Y()), float(pnt.Z())))
                node_map[i] = base_index + (i - 1)

            orientation_reversed = str(face.Orientation()) == "TopAbs_REVERSED"

            for i in range(1, triangulation.NbTriangles() + 1):
                tri = triangulation.Triangle(i)
                n1, n2, n3 = tri.Get()

                v1 = node_map[n1]
                v2 = node_map[n2]
                v3 = node_map[n3]

                if orientation_reversed:
                    faces.append((v1, v3, v2))
                else:
                    faces.append((v1, v2, v3))

        explorer.Next()

    if not vertices or not faces:
        raise ModelAnalysisError("Could not triangulate STEP model into mesh")

    mesh = trimesh.Trimesh(
        vertices=np.array(vertices, dtype=float),
        faces=np.array(faces, dtype=int),
        process=True,
    )

    if mesh.is_empty:
        raise ModelAnalysisError("Triangulated STEP mesh is empty")

    return mesh


def _load_mesh_for_support(file_bytes: bytes, filename: str) -> trimesh.Trimesh:
    fmt = _detect_format(filename)
    suffix = os.path.splitext(filename)[1].lower() or ".tmp"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        path = tmp.name

    try:
        if fmt == "stl":
            return _load_stl_mesh(path)
        if fmt == "step":
            return _step_to_trimesh(path)
        raise UnsupportedFileFormatError(
            f"Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _support_analysis_from_mesh(
    mesh: trimesh.Trimesh,
    support_angle_deg: float = 45.0,
    support_density_factor: float = 0.22,
) -> Dict:
    triangles = mesh.triangles
    normals = mesh.face_normals

    if len(triangles) == 0:
        raise ModelAnalysisError("Mesh contains no triangles")

    z_up = np.array([0.0, 0.0, 1.0])
    cos_threshold = math.cos(math.radians(90.0 - support_angle_deg))

    critical_projected_area = 0.0
    weighted_height_sum = 0.0
    critical_area_sum = 0.0

    for tri, normal in zip(triangles, normals):
        nz = float(np.dot(normal, z_up))

        if nz >= 0:
            continue

        if abs(nz) < cos_threshold:
            continue

        tri_xy_projected_area = 0.5 * abs(
            (tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1])
            - (tri[2][0] - tri[0][0]) * (tri[1][1] - tri[0][1])
        )

        if tri_xy_projected_area <= 0:
            continue

        avg_z = float((tri[0][2] + tri[1][2] + tri[2][2]) / 3.0)

        angle_weight = min(1.0, abs(nz))
        effective_area = float(tri_xy_projected_area) * angle_weight

        critical_projected_area += effective_area
        weighted_height_sum += effective_area * max(avg_z, 0.0)
        critical_area_sum += effective_area

    if critical_area_sum > 0:
        average_support_height = weighted_height_sum / critical_area_sum
    else:
        average_support_height = 0.0

    support_volume_mm3 = critical_projected_area * average_support_height * support_density_factor

    return {
        "support_angle_deg": round(support_angle_deg, 3),
        "support_density_factor": round(support_density_factor, 3),
        "support_required": support_volume_mm3 > 0,
        "critical_overhang_area_mm2": round(float(critical_projected_area), 3),
        "average_support_height_mm": round(float(average_support_height), 3),
        "support_volume_mm3": round(float(support_volume_mm3), 3),
        "support_volume_cm3": round(float(support_volume_mm3) / 1000.0, 3),
    }


def add_support_estimate(
    result: Dict,
    file_bytes: bytes,
    filename: str,
    support_angle_deg: float | None,
    support_density_factor: float | None,
) -> Dict:
    out = dict(result)

    out["support_angle_deg"] = support_angle_deg
    out["support_density_factor"] = support_density_factor
    out["support_required"] = None
    out["critical_overhang_area_mm2"] = None
    out["average_support_height_mm"] = None
    out["support_volume_mm3"] = None
    out["support_volume_cm3"] = None

    if support_angle_deg is None or support_density_factor is None:
        return out

    if support_angle_deg <= 0 or support_angle_deg >= 90:
        raise ModelAnalysisError("support_angle_deg must be between 0 and 90")

    if support_density_factor < 0 or support_density_factor > 1:
        raise ModelAnalysisError("support_density_factor must be between 0 and 1")

    mesh = _load_mesh_for_support(file_bytes=file_bytes, filename=filename)
    support = _support_analysis_from_mesh(
        mesh=mesh,
        support_angle_deg=support_angle_deg,
        support_density_factor=support_density_factor,
    )
    out.update(support)
    return out


def add_runtime_estimate(
    result: Dict,
    machine_hour_rate_eur: float | None,
    volumetric_flow_mm3_s: float | None,
) -> Dict:
    out = dict(result)

    out["machine_hour_rate_eur"] = machine_hour_rate_eur
    out["volumetric_flow_mm3_s"] = volumetric_flow_mm3_s
    out["estimated_print_seconds"] = None
    out["estimated_print_hours"] = None
    out["estimated_machine_cost_eur"] = None
    out["estimated_total_print_seconds"] = None
    out["estimated_total_print_hours"] = None
    out["estimated_total_machine_cost_eur"] = None

    if machine_hour_rate_eur is None or volumetric_flow_mm3_s is None:
        return out

    if volumetric_flow_mm3_s <= 0:
        raise ModelAnalysisError("volumetric_flow_mm3_s must be greater than 0")

    if machine_hour_rate_eur < 0:
        raise ModelAnalysisError("machine_hour_rate_eur must be >= 0")

    part_volume = float(out["volume_mm3"])
    support_volume = float(out.get("support_volume_mm3") or 0.0)

    part_seconds = part_volume / volumetric_flow_mm3_s
    part_hours = part_seconds / 3600.0
    part_cost = part_hours * machine_hour_rate_eur

    total_volume = part_volume + support_volume
    total_seconds = total_volume / volumetric_flow_mm3_s
    total_hours = total_seconds / 3600.0
    total_cost = total_hours * machine_hour_rate_eur

    out["estimated_print_seconds"] = round(part_seconds, 2)
    out["estimated_print_hours"] = round(part_hours, 4)
    out["estimated_machine_cost_eur"] = round(part_cost, 2)

    out["estimated_total_print_seconds"] = round(total_seconds, 2)
    out["estimated_total_print_hours"] = round(total_hours, 4)
    out["estimated_total_machine_cost_eur"] = round(total_cost, 2)

    return out


def _resolve_support_material(
    support_material_type: str,
    support_material_price_per_kg_eur,
    support_material_density_g_cm3,
    part_material_name: str,
    part_material_price_per_kg_eur: float,
    part_material_density_g_cm3: float,
) -> Dict:
    support_material_type = (support_material_type or DEFAULT_SUPPORT_MATERIAL_TYPE).lower()

    if support_material_type not in {"breakaway", "hips", "soluble"}:
        raise ModelAnalysisError("support_material_type must be one of: breakaway, hips, soluble")

    support_material_price_per_kg_eur = _to_optional_float(
        support_material_price_per_kg_eur, "support_material_price_per_kg_eur"
    )
    support_material_density_g_cm3 = _to_optional_float(
        support_material_density_g_cm3, "support_material_density_g_cm3"
    )

    if support_material_type == "breakaway":
        name = f"{part_material_name} Breakaway"
        price = (
            part_material_price_per_kg_eur
            if support_material_price_per_kg_eur is None
            else support_material_price_per_kg_eur
        )
        density = (
            part_material_density_g_cm3
            if support_material_density_g_cm3 is None
            else support_material_density_g_cm3
        )

    elif support_material_type == "hips":
        name = DEFAULT_SUPPORT_HIPS_NAME
        price = (
            DEFAULT_SUPPORT_HIPS_PRICE_PER_KG_EUR
            if support_material_price_per_kg_eur is None
            else support_material_price_per_kg_eur
        )
        density = (
            DEFAULT_SUPPORT_HIPS_DENSITY_G_CM3
            if support_material_density_g_cm3 is None
            else support_material_density_g_cm3
        )

    else:
        name = DEFAULT_SUPPORT_SOLUBLE_NAME
        price = (
            DEFAULT_SUPPORT_SOLUBLE_PRICE_PER_KG_EUR
            if support_material_price_per_kg_eur is None
            else support_material_price_per_kg_eur
        )
        density = (
            DEFAULT_SUPPORT_SOLUBLE_DENSITY_G_CM3
            if support_material_density_g_cm3 is None
            else support_material_density_g_cm3
        )

    if price < 0:
        raise ModelAnalysisError("support_material_price_per_kg_eur must be >= 0")

    if density <= 0:
        raise ModelAnalysisError("support_material_density_g_cm3 must be > 0")

    return {
        "support_material_type": support_material_type,
        "support_material_name": name,
        "support_material_price_per_kg_eur": float(price),
        "support_material_density_g_cm3": float(density),
    }

def add_material_estimate(
    result: Dict,
    part_material_name: str | None = None,
    part_material_price_per_kg_eur=None,
    part_material_density_g_cm3=None,
    support_material_type: str | None = None,
    support_material_price_per_kg_eur=None,
    support_material_density_g_cm3=None,
) -> Dict:
    out = dict(result)

    part_material_name = part_material_name or DEFAULT_PART_MATERIAL_NAME

    part_material_price_per_kg_eur = _to_optional_float(
        part_material_price_per_kg_eur, "part_material_price_per_kg_eur"
    )
    if part_material_price_per_kg_eur is None:
        part_material_price_per_kg_eur = DEFAULT_PART_MATERIAL_PRICE_PER_KG_EUR

    part_material_density_g_cm3 = _to_optional_float(
        part_material_density_g_cm3, "part_material_density_g_cm3"
    )
    if part_material_density_g_cm3 is None:
        part_material_density_g_cm3 = DEFAULT_PART_MATERIAL_DENSITY_G_CM3

    if part_material_price_per_kg_eur < 0:
        raise ModelAnalysisError("part_material_price_per_kg_eur must be >= 0")

    if part_material_density_g_cm3 <= 0:
        raise ModelAnalysisError("part_material_density_g_cm3 must be > 0")

    support_material = _resolve_support_material(
        support_material_type=support_material_type or DEFAULT_SUPPORT_MATERIAL_TYPE,
        support_material_price_per_kg_eur=support_material_price_per_kg_eur,
        support_material_density_g_cm3=support_material_density_g_cm3,
        part_material_name=part_material_name,
        part_material_price_per_kg_eur=part_material_price_per_kg_eur,
        part_material_density_g_cm3=part_material_density_g_cm3,
    )

    part_volume_cm3 = float(out.get("volume_cm3") or 0.0)
    support_volume_cm3 = float(out.get("support_volume_cm3") or 0.0)

    part_weight_g = part_volume_cm3 * part_material_density_g_cm3
    support_weight_g = support_volume_cm3 * support_material["support_material_density_g_cm3"]

    part_material_cost_eur = (part_weight_g / 1000.0) * part_material_price_per_kg_eur
    support_material_cost_eur = (support_weight_g / 1000.0) * support_material["support_material_price_per_kg_eur"]
    total_material_cost_eur = part_material_cost_eur + support_material_cost_eur

    out["part_material_name"] = part_material_name
    out["part_material_price_per_kg_eur"] = round(part_material_price_per_kg_eur, 2)
    out["part_material_density_g_cm3"] = round(part_material_density_g_cm3, 4)

    out["support_material_type"] = support_material["support_material_type"]
    out["support_material_name"] = support_material["support_material_name"]
    out["support_material_price_per_kg_eur"] = round(support_material["support_material_price_per_kg_eur"], 2)
    out["support_material_density_g_cm3"] = round(support_material["support_material_density_g_cm3"], 4)

    out["part_weight_g"] = round(part_weight_g, 3)
    out["support_weight_g"] = round(support_weight_g, 3)

    out["part_material_cost_eur"] = round(part_material_cost_eur, 2)
    out["support_material_cost_eur"] = round(support_material_cost_eur, 2)
    out["total_material_cost_eur"] = round(total_material_cost_eur, 2)

    return out


def add_price_estimate(
    result: Dict,
    margin_factor: float | None = 1.0,
) -> Dict:
    out = dict(result)

    if margin_factor is None:
        margin_factor = 1.0

    if margin_factor <= 0:
        raise ModelAnalysisError("margin_factor must be greater than 0")

    total_material_cost_eur = float(out.get("total_material_cost_eur") or 0.0)
    total_machine_cost_eur = float(out.get("estimated_total_machine_cost_eur") or 0.0)

    subtotal_cost_eur = total_material_cost_eur + total_machine_cost_eur
    total_price_eur = subtotal_cost_eur * float(margin_factor)

    out["margin_factor"] = round(float(margin_factor), 4)
    out["subtotal_cost_eur"] = round(subtotal_cost_eur, 2)
    out["total_price_eur"] = round(total_price_eur, 2)

    return out