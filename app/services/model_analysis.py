from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
import threading


from app.config import ALLOWED_EXTENSIONS
from PIL import Image, ImageChops

_PREVIEW_RENDER_LOCK = threading.Lock()
logger = logging.getLogger("step-analysis-service.preview")


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


def _vtk_png_bytes_from_render_window(render_window) -> bytes:
    """Capture the current VTK render window into PNG bytes."""
    import vtk

    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(render_window)
    capture.SetInputBufferTypeToRGB()
    capture.ReadFrontBufferOff()
    capture.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetWriteToMemory(True)
    writer.SetInputConnection(capture.GetOutputPort())
    writer.Write()
    result = writer.GetResult()
    return bytes(memoryview(result))


def _render_stl_preview_png_base64_unlocked(stl_bytes: bytes) -> str | None:
    """Render an opaque CAD preview using VTK's real depth buffer.

    V3.4 deliberately does not overlay Matplotlib 3D lines.  VTK renders the
    solid mesh, feature edges and silhouette through the same depth buffer, so
    back-side engraving/text and hidden geometry cannot bleed through the front
    surface.  A dedicated silhouette pass keeps the complete outer contour
    visible even where adjacent faces have a shallow angle.
    """
    if not stl_bytes:
        return None

    import vtk

    tmp_path = None
    render_window = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
            tmp.write(stl_bytes)
            tmp_path = tmp.name

        reader = vtk.vtkSTLReader()
        reader.SetFileName(tmp_path)
        reader.Update()
        polydata = reader.GetOutput()
        if polydata is None or polydata.GetNumberOfCells() == 0:
            return None

        # Clean/merge STL vertices first. STEP->STL triangulation often stores
        # duplicated points per triangle; feature extraction must operate on the
        # welded topology or real CAD edges can disappear / false boundaries can
        # appear. Surface normals are a separate shading pipeline.
        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(polydata)
        clean.PointMergingOn()
        clean.Update()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(clean.GetOutputPort())
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.SplittingOn()
        normals.SetFeatureAngle(38.0)
        normals.Update()

        surface_mapper = vtk.vtkPolyDataMapper()
        surface_mapper.SetInputConnection(normals.GetOutputPort())
        surface_mapper.ScalarVisibilityOff()

        surface = vtk.vtkActor()
        surface.SetMapper(surface_mapper)
        prop = surface.GetProperty()
        prop.SetColor(0.17, 0.18, 0.20)
        prop.SetOpacity(1.0)
        prop.SetInterpolationToPhong()
        prop.SetAmbient(0.30)
        prop.SetDiffuse(0.78)
        prop.SetSpecular(0.10)
        prop.SetSpecularPower(18.0)

        renderer = vtk.vtkRenderer()
        renderer.SetBackground(1.0, 1.0, 1.0)
        renderer.AddActor(surface)

        render_window = vtk.vtkRenderWindow()
        render_window.SetOffScreenRendering(1)
        render_window.SetSize(700, 560)
        render_window.SetMultiSamples(8)
        render_window.AddRenderer(renderer)

        # Use the same useful isometric family as the old preview, but view the
        # exterior/blank side for this reference part.  Orthographic projection
        # is closer to CAD thumbnails and keeps dimensions visually stable.
        camera = renderer.GetActiveCamera()
        renderer.ResetCamera()
        camera.Azimuth(-52.0)
        camera.Elevation(-24.0)
        camera.OrthogonalizeViewUp()
        camera.ParallelProjectionOn()
        camera.SetParallelScale(camera.GetParallelScale() * 1.08)
        renderer.ResetCameraClippingRange()

        # True structural feature edges. They are rendered as geometry inside
        # VTK, so the depth test hides edges on the far side of the solid.
        feature_edges = vtk.vtkFeatureEdges()
        feature_edges.SetInputConnection(clean.GetOutputPort())
        feature_edges.BoundaryEdgesOn()
        feature_edges.FeatureEdgesOn()
        feature_edges.SetFeatureAngle(18.0)
        feature_edges.ManifoldEdgesOff()
        feature_edges.NonManifoldEdgesOn()

        # Render feature edges as thin 3D tubes instead of OpenGL line
        # primitives. Tubes participate in the regular depth buffer and cannot
        # bleed through an opaque front face. The radius scales with model size.
        bounds = clean.GetOutput().GetBounds()
        dx = bounds[1] - bounds[0]
        dy = bounds[3] - bounds[2]
        dz = bounds[5] - bounds[4]
        diag = max((dx * dx + dy * dy + dz * dz) ** 0.5, 1e-6)

        feature_tubes = vtk.vtkTubeFilter()
        feature_tubes.SetInputConnection(feature_edges.GetOutputPort())
        feature_tubes.SetRadius(diag * 0.00105)
        feature_tubes.SetNumberOfSides(6)
        feature_tubes.CappingOn()

        feature_mapper = vtk.vtkPolyDataMapper()
        feature_mapper.SetInputConnection(feature_tubes.GetOutputPort())
        feature_mapper.ScalarVisibilityOff()
        feature_actor = vtk.vtkActor()
        feature_actor.SetMapper(feature_mapper)
        feature_actor.GetProperty().SetColor(0.018, 0.020, 0.024)
        feature_actor.GetProperty().SetAmbient(1.0)
        feature_actor.GetProperty().SetDiffuse(0.0)
        renderer.AddActor(feature_actor)

        # Independent silhouette extraction fixes missing outside contours such
        # as the upper-left edge reported on the Raspberry case lid.
        silhouette = vtk.vtkPolyDataSilhouette()
        silhouette.SetInputConnection(clean.GetOutputPort())
        silhouette.SetCamera(camera)
        silhouette.SetEnableFeatureAngle(0)
        silhouette.SetBorderEdges(1)
        silhouette.SetPieceInvariant(1)

        silhouette_tubes = vtk.vtkTubeFilter()
        silhouette_tubes.SetInputConnection(silhouette.GetOutputPort())
        silhouette_tubes.SetRadius(diag * 0.00135)
        silhouette_tubes.SetNumberOfSides(6)
        silhouette_tubes.CappingOn()

        silhouette_mapper = vtk.vtkPolyDataMapper()
        silhouette_mapper.SetInputConnection(silhouette_tubes.GetOutputPort())
        silhouette_mapper.ScalarVisibilityOff()
        silhouette_actor = vtk.vtkActor()
        silhouette_actor.SetMapper(silhouette_mapper)
        silhouette_actor.GetProperty().SetColor(0.0, 0.0, 0.0)
        silhouette_actor.GetProperty().SetAmbient(1.0)
        silhouette_actor.GetProperty().SetDiffuse(0.0)
        renderer.AddActor(silhouette_actor)

        # Two soft lights make steps and recesses obvious without blowing out
        # vertical faces. The actor remains fully opaque at all times.
        renderer.RemoveAllLights()
        key = vtk.vtkLight()
        key.SetLightTypeToSceneLight()
        key.SetPosition(-1.0, -1.2, 2.0)
        key.SetFocalPoint(0.0, 0.0, 0.0)
        key.SetIntensity(0.95)
        renderer.AddLight(key)

        fill = vtk.vtkLight()
        fill.SetLightTypeToSceneLight()
        fill.SetPosition(1.2, 0.8, 1.0)
        fill.SetFocalPoint(0.0, 0.0, 0.0)
        fill.SetIntensity(0.38)
        renderer.AddLight(fill)

        render_window.Render()
        png_bytes = _vtk_png_bytes_from_render_window(render_window)
        png_bytes = _crop_png_whitespace(png_bytes, padding=20)
        return base64.b64encode(png_bytes).decode("ascii")

    except Exception:
        logger.exception("preview_render_failed")
        return None

    finally:
        if render_window is not None:
            try:
                render_window.Finalize()
            except Exception:
                pass
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

def render_stl_preview_png_base64(stl_bytes: bytes) -> str | None:
    """Thread-safe wrapper around off-screen VTK preview rendering.

    Preview rendering remains serialized because off-screen graphics contexts are
    not guaranteed to be thread-safe. Conversion and Orca slicing stay parallel.
    """
    with _PREVIEW_RENDER_LOCK:
        return _render_stl_preview_png_base64_unlocked(stl_bytes)


def render_preview_from_converted_stl_bytes(stl_bytes: bytes) -> str | None:
    """
    Erwartet bereits konvertierte STL-Bytes und rendert daraus ein PNG-Preview.
    """
    return render_stl_preview_png_base64(stl_bytes)