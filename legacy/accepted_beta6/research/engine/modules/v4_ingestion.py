"""Versioned, lossless numeric ingress. The legacy reader remains unchanged.

TIFF pixels are never decoded by the known-crashing host Pillow in-process.
No resizing, per-image normalization, implicit stack flattening or uint16 clipping.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import struct
import sys

import numpy as np
from PIL import Image, ImageOps
import tifffile

from .io_utils import file_display_name, natural_sort_key

INGRESS_VERSION = "4.0-ingress-1"
ROOT = Path(__file__).resolve().parents[1]
MAX_PIXELS = 64_000_000
MAX_SEQUENCE_PIXELS = 160_000_000


@dataclass
class LoadedFrame:
    name: str
    pixels: np.ndarray
    metadata: dict
    original_index_0based: int = 0
    original_index_1based: int = 1


def array_hash(array):
    a = np.ascontiguousarray(array)
    head = json.dumps({"dtype": a.dtype.str, "shape": a.shape}, sort_keys=True).encode()
    return hashlib.sha256(head + a.tobytes()).hexdigest()


def source_bytes(item):
    source = getattr(item, "source", item)
    return Path(source).read_bytes() if isinstance(source, (str, Path)) else source.getvalue()


def orient(array, orientation):
    operations = {1: lambda a: a, 2: np.fliplr, 3: lambda a: np.rot90(a, 2), 4: np.flipud,
                  5: lambda a: np.swapaxes(a, 0, 1), 6: lambda a: np.rot90(a, -1),
                  7: lambda a: np.flip(np.swapaxes(a, 0, 1), (0, 1)), 8: lambda a: np.rot90(a, 1)}
    if orientation not in operations:
        raise ValueError(f"Unsupported orientation tag {orientation}; specify/correct orientation externally.")
    return np.ascontiguousarray(operations[orientation](array))


def _safe_codec_decode(payload, page_index, orientation):
    """Optional isolated codec worker configured explicitly for this development PC.

    A decoder crash fails just this input, not Streamlit. No hidden downloads.
    Portable installs can supply imagecodecs instead of this optional fallback.
    """
    config_path = ROOT / "v4_codec_runtime.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    executable = config.get("python_executable")
    if not executable or not Path(executable).is_file():
        raise ValueError("TIFF compression needs imagecodecs, or a configured isolated Pillow codec runtime. No pixels were decoded/clipped.")
    if orientation != 1:
        raise ValueError("Compressed TIFF fallback supports orientation=1 only; install imagecodecs for other orientations.")
    command = [executable, "-B", str(ROOT / "tools" / "v4_codec_worker.py"), str(page_index)]
    try:
        result = subprocess.run(command, input=payload, capture_output=True, timeout=45,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Isolated TIFF decoder failed: {exc}") from exc
    if result.returncode:
        raise ValueError(f"Isolated TIFF decoder failed (exit {result.returncode}); no partial frame analysed.")
    array = np.load(io.BytesIO(result.stdout), allow_pickle=False)
    return array, f"isolated Pillow {config.get('pillow_version', 'version-unrecorded')} ({executable})"


def load_frames(item, *, stack_policy="reject", selected_page=1):
    try:
        return _load_frames(item, stack_policy=stack_policy, selected_page=selected_page)
    except (struct.error, IndexError, KeyError, EOFError, OverflowError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Corrupt or unsupported input structure: {exc}") from exc


def _load_frames(item, *, stack_policy="reject", selected_page=1):
    if isinstance(item, LoadedFrame):
        return [item]
    if stack_policy not in ("reject", "sequence", "selected"):
        raise ValueError("Stack policy must be reject, sequence, or selected.")
    payload = source_bytes(item)
    name = file_display_name(item)
    sha = hashlib.sha256(payload).hexdigest()
    is_tiff = payload[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
    frames = []
    if is_tiff:
        with tifffile.TiffFile(io.BytesIO(payload)) as tif:
            pages = len(tif.pages)
            axes = [s.axes for s in tif.series]
            complex_stack = bool(tif.is_ome or any(p.subifds for p in tif.pages)) or any(
                any(axis in a for axis in "CZ") for a in axes)
            if complex_stack and stack_policy != "selected":
                raise ValueError("OME/Z/channel/pyramidal TIFF: select one explicit 2-D page; automatic time interpretation is unsupported.")
            indices = _page_indices(pages, stack_policy, selected_page)
            total = sum(int(np.prod(tif.pages[i].shape)) for i in indices)
            if total > MAX_SEQUENCE_PIXELS:
                raise ValueError("TIFF selection exceeds the development memory guard; select fewer pages.")
            for i in indices:
                page = tif.pages[i]
                if isinstance(page, tifffile.TiffFrame):
                    page = page.aspage()
                if int(np.prod(page.shape)) > MAX_PIXELS:
                    raise ValueError("TIFF page exceeds pixel limit.")
                orientation = int(page.tags[274].value) if 274 in page.tags else 1
                if page.photometric.name not in ("MINISBLACK", "MINISWHITE", "RGB"):
                    raise ValueError(f"TIFF photometric {page.photometric.name} is unsupported; no implicit colour conversion.")
                if page.axes not in ("YX", "YXS"):
                    raise ValueError(f"TIFF page axes {page.axes} are unsupported; export a 2-D grayscale or interleaved RGB page.")
                decoder = f"tifffile {tifffile.__version__}"
                try:
                    raw = page.asarray()
                except (ValueError, KeyError) as exc:
                    if "imagecodecs" not in str(exc).lower():
                        raise ValueError(f"TIFF decode failed: {exc}") from exc
                    raw, decoder = _safe_codec_decode(payload, i, orientation)
                    if raw.dtype != page.dtype or raw.shape != page.shape:
                        raise ValueError("Fallback TIFF decoder changed dtype or dimensions; rejected.")
                conversions = []
                warnings = []
                if page.photometric.name == "MINISWHITE":
                    warnings.append("TIFF MINISWHITE photometric: numeric values are preserved, NOT inverted. Confirm foreground polarity against preview.")
                if complex_stack:
                    warnings.append("Selected physical TIFF page from complex stack; biological/time/z/channel identity requires user verification.")
                arr = orient(raw, orientation)
                if orientation != 1:
                    conversions.append(f"TIFF orientation {orientation} applied, numeric values preserved")
                resolution = {tag: str(page.tags[tag].value) for tag in ("XResolution", "YResolution", "ResolutionUnit") if tag in page.tags}
                frames.append(_frame(name, payload, sha, raw, arr, pages, i, orientation, conversions, warnings,
                                     decoder, "TIFF", stack_policy, {"tiff_axes": axes, "compression": page.compression.name,
                                     "photometric": page.photometric.name, "calibration_metadata": resolution,
                                     "original_bit_depth": page.bitspersample}))
    else:
        with Image.open(io.BytesIO(payload)) as im:
            original_format = im.format
            pages = getattr(im, "n_frames", 1)
            if pages != 1:
                raise ValueError("Only TIFF multipage input is supported; animated input is rejected.")
            if im.width * im.height > MAX_PIXELS:
                raise ValueError("Image exceeds pixel limit.")
            _page_indices(pages, stack_policy, selected_page)
            orientation = int(im.getexif().get(274, 1))
            raw = np.asarray(im)
            conversions, warnings = [], []
            original_mode = im.mode
            if im.mode in ("P", "CMYK", "YCbCr", "LA"):
                im = im.convert("RGBA" if im.mode in ("P", "LA") and "transparency" in im.info else "RGB")
                conversions.append(f"{original_mode} converted to {im.mode}; colour interpretation, not intensity-preserving grayscale")
                warnings.append("Non-native colour mode converted explicitly; not eligible for exact V3 intensity regression.")
            elif im.mode == "1":
                im = im.convert("L")
                conversions.append("1-bit binary expanded to uint8 values 0/255")
            arr = orient(np.asarray(im), orientation)
            if orientation != 1:
                conversions.append(f"EXIF orientation {orientation} applied, numeric values preserved")
                warnings.append(f"EXIF orientation {orientation} applied; inspect image/mask alignment.")
            frames.append(_frame(name, payload, sha, raw, arr, 1, 0, orientation, conversions, warnings,
                                 f"Pillow {Image.__version__}", original_format, stack_policy,
                                 {"source_mode": original_mode, "calibration_metadata": {"dpi": im.info.get("dpi")}}))
    return frames


def _page_indices(pages, policy, selected):
    if pages < 1:
        raise ValueError("Input contains no pages.")
    if pages > 1 and policy == "reject":
        raise ValueError(f"Multi-page TIFF has {pages} pages. Explicitly select a page or interpret pages as a sequence; none were silently discarded.")
    if policy == "selected":
        if isinstance(selected, bool) or int(selected) != selected or not 1 <= selected <= pages:
            raise ValueError(f"Selected page must be a whole number from 1 to {pages}.")
        return [int(selected) - 1]
    return list(range(pages))


def _frame(name, payload, sha, raw, arr, pages, index, orientation, conversions, warnings, decoder, fmt, policy, extra):
    if arr.ndim not in (2, 3) or arr.ndim == 3 and arr.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported image dimensions {arr.shape}; no silent squeeze/flatten.")
    if arr.dtype.kind not in "buif" or not np.isfinite(arr).all():
        raise ValueError("Input must contain finite numeric pixels.")
    if arr.ndim == 3 and arr.shape[2] == 4:
        warnings.append("RGBA input: verify transparency; analysis ignores alpha, it does not composite a background.")
    if arr.dtype.itemsize > 1:
        warnings.append("Native high-bit-depth input preserved. This explicitly differs from legacy V3 RGB/8-bit conversion; historical equivalence is not asserted.")
    meta = {"filename": name if pages == 1 else f"{name}::page{index+1:05d}", "source_filename": name,
            "sha256": sha, "input_sha256": sha, "bytes": len(payload), "format": fmt,
            "original_dtype": str(raw.dtype), "original_bit_depth": 1 if raw.dtype == bool else raw.dtype.itemsize * 8,
            "original_dimensions": list(raw.shape), "original_channels": raw.shape[2] if raw.ndim == 3 else 1,
            "pages": pages, "page_1based": index+1, "stack_interpretation": policy if pages > 1 else "single image",
            "orientation": orientation, "applied_conversion": conversions or ["none; numeric pixel values preserved"],
            "resulting_dtype": str(arr.dtype), "resulting_dimensions": list(arr.shape),
            "width": arr.shape[1], "height": arr.shape[0], "raw_width": raw.shape[1], "raw_height": raw.shape[0],
            "exif_orientation": orientation, "readable": True, "ingress_version": INGRESS_VERSION,
            "decoder": decoder, "warnings": warnings, "decoded_array_sha256": array_hash(arr), **extra}
    return LoadedFrame(meta["filename"], np.ascontiguousarray(arr), meta)


def load_sequence(files, *, stack_policy="reject", selected_page=1):
    names = [file_display_name(f) for f in files]
    if not names:
        raise ValueError("Select at least one image/mask.")
    if len(set(n.casefold() for n in names)) != len(names):
        raise ValueError("Duplicate filenames: rename inputs so provenance and pairing are unambiguous.")
    frames = []
    cumulative_pixels = 0
    for item in sorted(files, key=natural_sort_key):
        selected = load_frames(item, stack_policy=stack_policy, selected_page=selected_page)
        cumulative_pixels += sum(f.pixels.size for f in selected)
        if cumulative_pixels > MAX_SEQUENCE_PIXELS:
            raise ValueError("Selection exceeds development memory guard; split into smaller sequences.")
        frames.extend(selected)
    if len({f.pixels.shape[:2] for f in frames}) != 1:
        raise ValueError("Mixed image dimensions across frames; no resizing or frame deletion was applied.")
    for i, frame in enumerate(frames):
        frame.original_index_0based, frame.original_index_1based = i, i+1
    return frames


def read_uploaded_image(item):
    """App-compatible reader; multi-page input must be pre-expanded explicitly."""
    if isinstance(item, LoadedFrame):
        return item.name, item.pixels
    frames = load_frames(item)
    return frames[0].name, frames[0].pixels


def display_u8(array):
    """Display-only dtype-range scaling; NEVER used as segmentation input."""
    a = np.asarray(array)
    if a.dtype == np.uint8:
        return a
    if a.dtype.kind in "ui":
        info = np.iinfo(a.dtype)
        return np.rint((a.astype(float) - info.min) * 255 / (float(info.max) - info.min)).clip(0, 255).astype(np.uint8)
    a = a.astype(float)
    if a.size and (a.min() < 0 or a.max() > 1):
        a = (a - a.min()) / max(float(np.ptp(a)), 1e-12)
    return np.rint(a * 255).clip(0, 255).astype(np.uint8)
