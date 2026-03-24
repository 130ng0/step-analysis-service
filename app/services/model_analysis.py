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
DEFAULT_SUPPORT_HIPS_NAME = "HIPS"
DEFAULT_SUPPORT_SOLUBLE_NAME = "Soluble Support"

DEFAULT_SUPPORT_HIPS_PRICE_PER_KG_EUR = 27.0 / 0.75
DEFAULT_SUPPORT_SOLUBLE_PRICE_PER_KG_EUR = 80.0

DEFAULT_SUPPORT_HIPS_DENSITY_G_CM3 = 1.03
DEFAULT_SUPPORT_SOLUBLE_DENSITY_G_CM3 = 1.20

DEFAULT_LINE_WIDTH_MM = 0.45
DEFAULT_LAYER_HEIGHT_MM = 0.20
DEFAULT_PATH_EFFICIENCY_FACTOR = 1.03
DEFAULT_SUPPORT_SPEED_FACTOR = 0.85

STL_SUPPORT_MAX_FACES = 25000
STL_FULL_SUPPORT_MAX_FILE_MB = 12

STEP_SUPPORT_LINEAR_DEFLECTION = 0.6
STEP_SUPPORT_ANGULAR_DEFLECTION = 1.0
STEP_FULL_SUPPORT_MAX_FILE_MB = 8


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


def _is_large_file(file_bytes: bytes, threshold_mb: int) -> bool:
    return len(file_bytes) > threshold_mb * 1024 * 1024


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


def _simplify_mesh_for_support(mesh: trimesh.Trimesh, max_faces: int = STL_SUPPORT_MAX_FACES) -> trimesh.Trimesh:
    simplified = mesh.copy()

    try:
        simplified.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        simplified.remove_duplicate_faces()
    except Exception:
        pass

    try:
        simplified.remove_degenerate_faces()
    except Exception:
        pass

    try:
        simplified.merge_vertices()
    except Exception:
        pass

    try:
        components = simplified.split(only_watertight=False)
        if components:
            components = sorted(components, key=lambda m: len(m.faces), reverse=True)
            simplified = trimesh.util.concatenate(components[:3])
    except Exception:
        pass

    if len(simplified.faces) > max_faces:
        try:
            simplified = simplified.simplify_quadric_decimation(max_faces)
        except Exception:
            pass

    if simplified.is_empty:
        return mesh

    return simplified


def _step_to_trimesh(
    path: str,
    linear_deflection: float = STEP_SUPPORT_LINEAR_DEFLECTION,
    angular_deflection: float = STEP_SUPPORT_ANGULAR_DEFLECTION,
) -> trimesh.Trimesh:
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
            mesh = _load_stl_mesh(path)
            return _simplify_mesh_for_support(mesh, max_faces=STL_SUPPORT_MAX_FACES)

        if fmt == "step":
            return _step_to_trimesh(
                path,
                linear_deflection=STEP_SUPPORT_LINEAR_DEFLECTION,
                angular_deflection=STEP_SUPPORT_ANGULAR_DEFLECTION,
            )

        raise UnsupportedFileFormatError(
            f"Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _projected_triangle_area_xy(tri) -> float:
    return 0.5 * abs(
        (tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1])
        - (tri[2][0] - tri[0][0]) * (tri[1][1] - tri[0][1])
    )


def _bed_contact_area(mesh: trimesh.Trimesh, layer_height_mm: float = DEFAULT_LAYER_HEIGHT_MM) -> float:
    triangles = mesh.triangles
    normals = mesh.face_normals

    z_min = float(mesh.bounds[0][2])
    threshold = z_min + max(layer_height_mm * 3.0, 0.8)

    contact = 0.0
    for tri, normal in zip(triangles, normals):
        avg_z = float((tri[0][2] + tri[1][2] + tri[2][2]) / 3.0)
        if avg_z > threshold:
            continue
        if float(normal[2]) > -0.95:
            continue
        contact += _projected_triangle_area_xy(tri)
    return float(contact)


def _support_analysis_from_mesh(
    mesh: trimesh.Trimesh,
    support_angle_deg: float = 45.0,
    support_density_factor: float = 0.22,
) -> Dict:
    triangles = mesh.triangles
    normals = mesh.face_normals
    centroids = mesh.triangles_center

    if len(triangles) == 0:
        raise ModelAnalysisError("Mesh contains no triangles")

    z_up = np.array([0.0, 0.0, 1.0])
    cos_threshold = math.cos(math.radians(90.0 - support_angle_deg))

    z_min = float(mesh.bounds[0][2])
    support_volume = 0.0
    critical_projected_area = 0.0
    weighted_height_sum = 0.0
    critical_area_sum = 0.0

    candidate_effective_areas = []
    candidate_origins = []

    for tri, normal, centroid in zip(triangles, normals, centroids):
        nz = float(np.dot(normal, z_up))
        if nz >= 0:
            continue
        if abs(nz) < cos_threshold:
            continue

        proj_area = _projected_triangle_area_xy(tri)
        if proj_area <= 0:
            continue

        angle_weight = min(1.0, abs(nz))
        effective_area = float(proj_area) * angle_weight

        origin = centroid.copy()
        origin[2] -= 0.05

        candidate_effective_areas.append(effective_area)
        candidate_origins.append(origin)

    hit_gaps = {}
    if candidate_origins:
        try:
            origins = np.array(candidate_origins)
            directions = np.tile(np.array([[0.0, 0.0, -1.0]]), (len(origins), 1))
            locations, index_ray, _ = mesh.ray.intersects_location(
                origins,
                directions,
                multiple_hits=False,
            )

            for ray_idx, loc in zip(index_ray, locations):
                gap = float(origins[ray_idx][2] - loc[2])
                if gap > 0:
                    hit_gaps[int(ray_idx)] = gap
        except Exception:
            hit_gaps = {}

    for local_idx, effective_area, origin in zip(range(len(candidate_origins)), candidate_effective_areas, candidate_origins):
        if local_idx in hit_gaps:
            support_height = hit_gaps[local_idx]
        else:
            support_height = max(float(origin[2] - z_min), 0.0)

        critical_projected_area += effective_area
        weighted_height_sum += effective_area * support_height
        critical_area_sum += effective_area
        support_volume += effective_area * support_height * support_density_factor

    if critical_area_sum > 0:
        average_support_height = weighted_height_sum / critical_area_sum
    else:
        average_support_height = 0.0

    return {
        "support_angle_deg": round(support_angle_deg, 3),
        "support_density_factor": round(support_density_factor, 3),
        "support_required": support_volume > 0,
        "critical_overhang_area_mm2": round(float(critical_projected_area), 3),
        "average_support_height_mm": round(float(average_support_height), 3),
        "support_volume_mm3": round(float(support_volume), 3),
        "support_volume_cm3": round(float(support_volume) / 1000.0, 3),
    }


def _orientation_transforms() -> Dict[str, np.ndarray]:
    return {
        "z_plus": np.eye(4),
        "z_minus": trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]),
        "x_plus": trimesh.transformations.rotation_matrix(-math.pi / 2, [0, 1, 0]),
        "x_minus": trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]),
        "y_plus": trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]),
        "y_minus": trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]),
    }


def _evaluate_support_for_orientation(
    mesh: trimesh.Trimesh,
    support_angle_deg: float,
    support_density_factor: float,
    orientation_name: str,
    transform: np.ndarray,
) -> Dict:
    oriented = mesh.copy()
    oriented.apply_transform(transform)

    bounds = oriented.bounds.copy()
    oriented.apply_translation([0.0, 0.0, -float(bounds[0][2])])

    result = _support_analysis_from_mesh(
        oriented,
        support_angle_deg=support_angle_deg,
        support_density_factor=support_density_factor,
    )
    result["selected_orientation"] = orientation_name
    result["build_height_mm"] = float(oriented.extents[2])
    result["bed_contact_area_mm2"] = round(_bed_contact_area(oriented), 3)
    return result


def _best_support_orientation_from_mesh(
    mesh: trimesh.Trimesh,
    support_angle_deg: float,
    support_density_factor: float,
    orientation_mode: str,
) -> Dict:
    transforms = _orientation_transforms()

    if orientation_mode == "fixed":
        best = _evaluate_support_for_orientation(
            mesh=mesh,
            support_angle_deg=support_angle_deg,
            support_density_factor=support_density_factor,
            orientation_name="z_plus",
            transform=transforms["z_plus"],
        )
        best.pop("build_height_mm", None)
        best.pop("bed_contact_area_mm2", None)
        return best

    candidates = []
    for orientation_name, transform in transforms.items():
        candidates.append(
            _evaluate_support_for_orientation(
                mesh=mesh,
                support_angle_deg=support_angle_deg,
                support_density_factor=support_density_factor,
                orientation_name=orientation_name,
                transform=transform,
            )
        )

    min_support = min(c["support_volume_mm3"] for c in candidates)
    near_best = [
        c for c in candidates
        if c["support_volume_mm3"] <= (min_support * 1.05 + 1.0)
    ]

    near_best.sort(
        key=lambda c: (
            c["support_volume_mm3"],
            -c["bed_contact_area_mm2"],
            c["build_height_mm"],
        )
    )
    best = near_best[0]
    best.pop("build_height_mm", None)
    best.pop("bed_contact_area_mm2", None)
    return best


def _fallback_orientation_score(
    base_area: float,
    height: float,
    surface_area: float,
    support_density_factor: float,
) -> float:
    est_critical_area = min(surface_area * 0.008, base_area * 0.07)
    est_support_volume = est_critical_area * (height * 0.45) * support_density_factor
    return est_support_volume


def _fallback_orientation_from_bbox(result: Dict, orientation_mode: str, support_density_factor: float) -> str:
    if orientation_mode == "fixed":
        return "z_plus"

    bx = float(result.get("bbox_x_mm") or 0.0)
    by = float(result.get("bbox_y_mm") or 0.0)
    bz = float(result.get("bbox_z_mm") or 0.0)
    surface_area = float(result.get("surface_mm2") or 0.0)

    candidates = {
        "z_plus": {"base": bx * by, "height": bz},
        "z_minus": {"base": bx * by, "height": bz},
        "x_plus": {"base": by * bz, "height": bx},
        "x_minus": {"base": by * bz, "height": bx},
        "y_plus": {"base": bx * bz, "height": by},
        "y_minus": {"base": bx * bz, "height": by},
    }

    best_key = None
    best_tuple = None
    for key, vals in candidates.items():
        score = _fallback_orientation_score(
            base_area=vals["base"],
            height=vals["height"],
            surface_area=surface_area,
            support_density_factor=support_density_factor,
        )
        ranking = (score, -vals["base"], vals["height"])
        if best_tuple is None or ranking < best_tuple:
            best_tuple = ranking
            best_key = key

    return best_key or "z_plus"


def _fallback_support_estimate_from_part(
    result: Dict,
    support_density_factor: float,
    support_angle_deg: float,
    selected_orientation: str,
) -> Dict:
    surface_mm2 = float(result.get("surface_mm2") or 0.0)

    bx = float(result.get("bbox_x_mm") or 0.0)
    by = float(result.get("bbox_y_mm") or 0.0)
    bz = float(result.get("bbox_z_mm") or 0.0)

    if selected_orientation in {"z_plus", "z_minus"}:
        base_area = bx * by
        height = bz
    elif selected_orientation in {"x_plus", "x_minus"}:
        base_area = by * bz
        height = bx
    else:
        base_area = bx * bz
        height = by

    # Deutlich konservativer Large-File-Fallback:
    # Ziel ist nicht "maximaler theoretischer Support",
    # sondern eine realistische Näherung für Gehäuse / Kisten / technische Teile.
    critical_overhang_area_mm2 = min(surface_mm2 * 0.008, base_area * 0.07)
    average_support_height_mm = height * 0.45
    support_volume_mm3 = critical_overhang_area_mm2 * average_support_height_mm * support_density_factor

    warnings = list(result.get("warnings", []))
    warnings.append("Large file fallback support estimate was used to reduce memory usage")

    return {
        "selected_orientation": selected_orientation,
        "support_angle_deg": round(support_angle_deg, 3),
        "support_density_factor": round(support_density_factor, 3),
        "support_required": support_volume_mm3 > 0,
        "critical_overhang_area_mm2": round(critical_overhang_area_mm2, 3),
        "average_support_height_mm": round(average_support_height_mm, 3),
        "support_volume_mm3": round(support_volume_mm3, 3),
        "support_volume_cm3": round(support_volume_mm3 / 1000.0, 3),
        "warnings": warnings,
    }


def add_support_estimate(
    result: Dict,
    file_bytes: bytes,
    filename: str,
    support_angle_deg: float | None,
    support_density_factor: float | None,
    support_material_type: str | None,
    orientation_mode: str = "auto",
) -> Dict:
    out = dict(result)

    out["selected_orientation"] = "z_plus" if orientation_mode == "fixed" else None
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

    fmt = _detect_format(filename)

    if fmt == "stl" and _is_large_file(file_bytes, STL_FULL_SUPPORT_MAX_FILE_MB):
        selected_orientation = _fallback_orientation_from_bbox(
            out,
            orientation_mode,
            support_density_factor,
        )
        out.update(
            _fallback_support_estimate_from_part(
                result=out,
                support_density_factor=support_density_factor,
                support_angle_deg=support_angle_deg,
                selected_orientation=selected_orientation,
            )
        )
    elif fmt == "step" and _is_large_file(file_bytes, STEP_FULL_SUPPORT_MAX_FILE_MB):
        selected_orientation = _fallback_orientation_from_bbox(
            out,
            orientation_mode,
            support_density_factor,
        )
        out.update(
            _fallback_support_estimate_from_part(
                result=out,
                support_density_factor=support_density_factor,
                support_angle_deg=support_angle_deg,
                selected_orientation=selected_orientation,
            )
        )
    else:
        mesh = _load_mesh_for_support(file_bytes=file_bytes, filename=filename)
        support = _best_support_orientation_from_mesh(
            mesh=mesh,
            support_angle_deg=support_angle_deg,
            support_density_factor=support_density_factor,
            orientation_mode=orientation_mode,
        )
        out.update(support)

    if support_material_type == "none" or (support_density_factor is not None and support_density_factor <= 0):
        out["support_required"] = False
        out["support_volume_mm3"] = 0.0
        out["support_volume_cm3"] = 0.0
        out["average_support_height_mm"] = 0.0
        out["support_density_factor"] = 0.0

    return out


def _oriented_dimensions_from_result(result: Dict) -> Tuple[float, float, float]:
    bx = float(result.get("bbox_x_mm") or 0.0)
    by = float(result.get("bbox_y_mm") or 0.0)
    bz = float(result.get("bbox_z_mm") or 0.0)
    orientation = result.get("selected_orientation") or "z_plus"

    if orientation in {"z_plus", "z_minus"}:
        return bx, by, bz
    if orientation in {"x_plus", "x_minus"}:
        return by, bz, bx
    if orientation in {"y_plus", "y_minus"}:
        return bx, bz, by
    return bx, by, bz


def add_extrusion_estimate(
    result: Dict,
    infill_percent: float = 20.0,
    perimeter_count: int = 5,
    top_layers: int = 5,
    bottom_layers: int = 5,
    line_width_mm: float = DEFAULT_LINE_WIDTH_MM,
    layer_height_mm: float = DEFAULT_LAYER_HEIGHT_MM,
    support_material_type: str = "breakaway",
) -> Dict:
    out = dict(result)

    if infill_percent < 0 or infill_percent > 100:
        raise ModelAnalysisError("infill_percent must be between 0 and 100")

    if perimeter_count < 0 or top_layers < 0 or bottom_layers < 0:
        raise ModelAnalysisError("perimeter_count, top_layers and bottom_layers must be >= 0")

    if line_width_mm <= 0 or layer_height_mm <= 0:
        raise ModelAnalysisError("line_width_mm and layer_height_mm must be > 0")

    part_volume = float(out.get("volume_mm3") or 0.0)
    support_volume = float(out.get("support_volume_mm3") or 0.0)

    dim_x, dim_y, dim_z = _oriented_dimensions_from_result(out)
    bbox_volume = max(dim_x * dim_y * dim_z, 1.0)
    packing_ratio = max(0.08, min(part_volume / bbox_volume, 1.0))

    wall_thickness = perimeter_count * line_width_mm
    top_bottom_total = (top_layers + bottom_layers) * layer_height_mm

    inner_x = max(dim_x - 2.0 * wall_thickness, 0.0)
    inner_y = max(dim_y - 2.0 * wall_thickness, 0.0)
    inner_z = max(dim_z - top_bottom_total, 0.0)

    bbox_shell_volume = max(bbox_volume - (inner_x * inner_y * inner_z), 0.0)

    shell_volume_geom = min(part_volume, bbox_shell_volume * (packing_ratio ** 0.82))
    inner_part_volume = max(part_volume - shell_volume_geom, 0.0)
    infill_fraction = infill_percent / 100.0
    geometry_based_estimate = shell_volume_geom + inner_part_volume * infill_fraction

    minimum_fraction = (
        0.12
        + 0.028 * perimeter_count
        + 0.006 * (top_layers + bottom_layers)
        + 0.38 * infill_fraction
    )
    minimum_fraction = min(max(minimum_fraction, 0.20), 0.48)

    minimum_estimate = part_volume * minimum_fraction

    effective_part_extrusion_volume = max(geometry_based_estimate, minimum_estimate)
    effective_support_extrusion_volume = 0.0 if support_material_type == "none" else support_volume

    out["infill_percent"] = round(infill_percent, 3)
    out["perimeter_count"] = int(perimeter_count)
    out["top_layers"] = int(top_layers)
    out["bottom_layers"] = int(bottom_layers)
    out["line_width_mm"] = round(line_width_mm, 3)
    out["layer_height_mm"] = round(layer_height_mm, 3)

    out["effective_part_extrusion_volume_mm3"] = round(effective_part_extrusion_volume, 3)
    out["effective_support_extrusion_volume_mm3"] = round(effective_support_extrusion_volume, 3)

    return out


def add_runtime_estimate(
    result: Dict,
    machine_hour_rate_eur: float | None,
    volumetric_flow_mm3_s: float | None,
    support_material_type: str | None,
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

    part_extrusion = float(out.get("effective_part_extrusion_volume_mm3") or 0.0)
    support_extrusion = float(out.get("effective_support_extrusion_volume_mm3") or 0.0)

    part_seconds = (part_extrusion / volumetric_flow_mm3_s) * DEFAULT_PATH_EFFICIENCY_FACTOR

    if support_material_type == "none":
        support_seconds = 0.0
    else:
        support_seconds = support_extrusion / (volumetric_flow_mm3_s * DEFAULT_SUPPORT_SPEED_FACTOR)

    part_hours = part_seconds / 3600.0
    total_seconds = part_seconds + support_seconds
    total_hours = total_seconds / 3600.0

    part_cost = part_hours * machine_hour_rate_eur
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

    if support_material_type == "none":
        return {
            "support_material_type": "none",
            "support_material_name": "Kein Support",
            "support_material_price_per_kg_eur": 0.0,
            "support_material_density_g_cm3": 0.0,
        }

    if support_material_type not in {"breakaway", "hips", "soluble"}:
        raise ModelAnalysisError("support_material_type must be one of: none, breakaway, hips, soluble")

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

    part_extrusion_mm3 = float(out.get("effective_part_extrusion_volume_mm3") or 0.0)
    support_extrusion_mm3 = float(out.get("effective_support_extrusion_volume_mm3") or 0.0)

    part_weight_g = (part_extrusion_mm3 / 1000.0) * part_material_density_g_cm3
    support_weight_g = (support_extrusion_mm3 / 1000.0) * support_material["support_material_density_g_cm3"]

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