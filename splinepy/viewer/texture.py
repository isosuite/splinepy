"""Turns images into data urls that scenes can embed."""

import base64 as _base64
import pathlib as _pathlib
import struct as _struct
import zlib as _zlib

import numpy as _np

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def _png(arr):
    """Minimal PNG encoder for (h, w) / (h, w, 3) / (h, w, 4) uint8 arrays."""
    h, w = arr.shape[:2]
    channels = 1 if arr.ndim == 2 else arr.shape[2]
    color_type = {1: 0, 3: 2, 4: 6}[channels]
    raw = _np.concatenate(
        [_np.zeros((h, 1), _np.uint8), arr.reshape(h, w * channels)], axis=1
    ).tobytes()

    def chunk(tag, data):
        body = tag + data
        return (
            _struct.pack(">I", len(data)) + body + _struct.pack(">I", _zlib.crc32(body))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", _struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0))
        + chunk(b"IDAT", _zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def to_data_url(texture):
    """Image path, encoded png/jpg bytes or array -> ``data:`` url.

    Arrays are (h, w), (h, w, 3) or (h, w, 4); floats are read as [0, 1],
    integers as [0, 255]. Row 0 is the top of the image.
    """
    if isinstance(texture, str) and texture.startswith("data:"):
        return texture
    if isinstance(texture, (str, _pathlib.Path)):
        path = _pathlib.Path(texture)
        mime = _MIME.get(path.suffix.lower())
        if mime is None:
            raise ValueError(f"Texture must be png or jpg, got {path.suffix}")
        data = path.read_bytes()
    elif isinstance(texture, (bytes, bytearray)):
        data, mime = bytes(texture), "image/png"
    else:
        arr = _np.asarray(texture)
        if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] not in (3, 4)):
            raise ValueError(f"Unsupported texture shape {arr.shape}")
        if arr.dtype.kind == "f":
            arr = arr * 255
        data = _png(_np.ascontiguousarray(_np.clip(arr, 0, 255).astype(_np.uint8)))
        mime = "image/png"
    return f"data:{mime};base64," + _base64.b64encode(data).decode("ascii")
