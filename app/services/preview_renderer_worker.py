from __future__ import annotations

import os
import sys


def render_stl_to_png(stl_path: str, output_path: str) -> None:
    import vtk

    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetNumberOfCells() == 0:
        raise RuntimeError("STL contains no renderable cells")

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

    camera = renderer.GetActiveCamera()
    renderer.ResetCamera()
    camera.Azimuth(-52.0)
    camera.Elevation(-24.0)
    camera.OrthogonalizeViewUp()
    camera.ParallelProjectionOn()
    camera.SetParallelScale(camera.GetParallelScale() * 1.08)
    renderer.ResetCameraClippingRange()

    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputConnection(clean.GetOutputPort())
    feature_edges.BoundaryEdgesOn()
    feature_edges.FeatureEdgesOn()
    feature_edges.SetFeatureAngle(18.0)
    feature_edges.ManifoldEdgesOff()
    feature_edges.NonManifoldEdgesOn()

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

    try:
        render_window.Render()

        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(render_window)
        capture.SetInputBufferTypeToRGB()
        capture.ReadFrontBufferOff()
        capture.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(output_path)
        writer.SetInputConnection(capture.GetOutputPort())
        writer.Write()
    finally:
        try:
            render_window.Finalize()
        except Exception:
            pass

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("VTK produced no PNG output")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m app.services.preview_renderer_worker INPUT.stl OUTPUT.png", file=sys.stderr)
        return 2
    try:
        render_stl_to_png(sys.argv[1], sys.argv[2])
        return 0
    except Exception as exc:
        print(f"preview renderer failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
