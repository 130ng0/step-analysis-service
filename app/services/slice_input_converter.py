from __future__ import annotations

import os
import tempfile

import cadquery as cq


class SliceInputConversionError(Exception):
    pass


def convert_upload_to_stl_bytes(file_bytes: bytes, filename: str) -> tuple[bytes, str]:
    """
    Nimmt STL direkt.
    Konvertiert STEP/STP in STL für den Orca-Worker.
    """
    lower = (filename or "").lower()

    if lower.endswith(".stl"):
        return file_bytes, filename

    if lower.endswith(".step") or lower.endswith(".stp"):
        return _convert_step_bytes_to_stl_bytes(file_bytes, filename)

    raise SliceInputConversionError(f"Unsupported slice input format: {filename}")


def _convert_step_bytes_to_stl_bytes(file_bytes: bytes, filename: str) -> tuple[bytes, str]:
    step_suffix = os.path.splitext(filename)[1].lower() or ".step"

    with tempfile.NamedTemporaryFile(delete=False, suffix=step_suffix) as step_tmp:
        step_tmp.write(file_bytes)
        step_path = step_tmp.name

    stl_path = step_path + ".stl"

    try:
        shape = cq.importers.importStep(step_path).val()
        if shape is None:
            raise SliceInputConversionError("STEP import returned no shape")

        # Für Slicing reicht ein etwas gröberes STL oft aus und spart Zeit/RAM
        cq.exporters.export(
            shape,
            stl_path,
            tolerance=0.15,
            angularTolerance=0.2,
        )

        if not os.path.exists(stl_path):
            raise SliceInputConversionError("STEP to STL conversion failed: STL not created")

        with open(stl_path, "rb") as f:
            stl_bytes = f.read()

        if not stl_bytes:
            raise SliceInputConversionError("STEP to STL conversion failed: STL is empty")

        base = os.path.splitext(os.path.basename(filename))[0]
        return stl_bytes, f"{base}.stl"

    except SliceInputConversionError:
        raise
    except Exception as exc:
        raise SliceInputConversionError(f"STEP to STL conversion failed: {exc}") from exc
    finally:
        try:
            if os.path.exists(step_path):
                os.unlink(step_path)
        except Exception:
            pass
        try:
            if os.path.exists(stl_path):
                os.unlink(stl_path)
        except Exception:
            pass