"""Pick a conservative orientation for estimating support with one Orca slice."""

from pathlib import Path
import struct

import numpy as np


_STL_TRIANGLE = np.dtype([
    ("normal", "<f4", (3,)),
    ("vertices", "<f4", (3, 3)),
    ("attribute", "<u2"),
])
_MAX_SAMPLES = 32_768
_BLOCK_SIZE = 65_536
_BED_TOLERANCE_MM = 0.05
_MIN_CONTACT_MM2 = 1.0
_MAX_SUPPORT_RATIO = 0.70
_OVERHANG_Z = np.sqrt(0.5)


def _binary_stl(path):
    with path.open("rb") as source:
        header = source.read(84)
    if len(header) != 84:
        return None
    count = struct.unpack_from("<I", header, 80)[0]
    if not count or path.stat().st_size != 84 + count * _STL_TRIANGLE.itemsize:
        return None
    return header, np.memmap(path, mode="r", offset=84, dtype=_STL_TRIANGLE, shape=(count,))


def _score(vertices, bottom, top):
    heights = vertices[:, :, 2]
    centre_z = heights.mean(axis=1)
    cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    projected_area = np.abs(cross[:, 2]) * 0.5
    steep = np.abs(cross[:, 2]) > np.linalg.norm(cross, axis=1) * _OVERHANG_Z
    downward = cross[:, 2] < 0
    bottom_contact = downward & (heights.max(axis=1) - bottom <= _BED_TOLERANCE_MM)
    top_contact = ~downward & (top - heights.min(axis=1) <= _BED_TOLERANCE_MM)

    # This estimates exposed support by projected overhang area and distance
    # from the bed. It does not model occlusion or generate support geometry.
    original = np.sum(projected_area[steep & downward & ~bottom_contact]
                      * (centre_z - bottom)[steep & downward & ~bottom_contact])
    flipped = np.sum(projected_area[steep & ~downward & ~top_contact]
                     * (top - centre_z)[steep & ~downward & ~top_contact])
    return original, flipped, np.sum(projected_area[top_contact])


def prepare_orientation(source: Path, flipped_path: Path) -> tuple[Path, str]:
    """Keep the file orientation unless a sampled estimate clearly favors an X flip."""
    stl = _binary_stl(source)
    if stl is None:
        return source, "original"
    header, triangles = stl
    heights = triangles["vertices"][:, :, 2]
    bottom, top = float(heights.min()), float(heights.max())
    if not np.isfinite([bottom, top]).all() or top - bottom <= _BED_TOLERANCE_MM:
        return source, "original"

    stride = (len(triangles) + _MAX_SAMPLES - 1) // _MAX_SAMPLES
    vertices = triangles["vertices"][::stride].astype(np.float64)
    original, flipped, contact = _score(vertices, bottom, top)
    if not np.isfinite([original, flipped, contact]).all() or (
        original <= 0 or flipped >= original * _MAX_SUPPORT_RATIO or contact < _MIN_CONTACT_MM2
    ):
        return source, "original"

    with flipped_path.open("wb") as target:
        target.write(header)
        for start in range(0, len(triangles), _BLOCK_SIZE):
            block = triangles[start:start + _BLOCK_SIZE].copy()
            block["normal"][:, 1:] *= -1
            block["vertices"][:, :, 1:] *= -1
            block.tofile(target)
    return flipped_path, "flipped_x_180"
