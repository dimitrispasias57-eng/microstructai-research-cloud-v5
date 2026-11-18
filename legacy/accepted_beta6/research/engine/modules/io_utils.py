"""Input helpers for uploaded images, masks, and labels."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps


SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
LOCAL_IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


@dataclass
class ImageFileRef:
    """Lightweight reference to an uploaded or local image file."""

    source: object
    name: str
    original_index_0based: int
    original_index_1based: int


def read_uploaded_image(uploaded_file) -> tuple[str, np.ndarray]:
    """Read a Streamlit upload, local path, or ImageFileRef into an array."""
    if uploaded_file is None:
        raise ValueError("No file was provided.")

    source = getattr(uploaded_file, "source", uploaded_file)
    name = getattr(uploaded_file, "name", file_display_name(source))

    if isinstance(source, (str, Path)):
        image_handle = Image.open(Path(source))
    else:
        data = source.getvalue()
        image_handle = Image.open(BytesIO(data))

    with image_handle as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("L", "RGB", "RGBA"):
            img = img.convert("RGB")
        arr = np.asarray(img)
    return name, arr


def file_display_name(file_obj) -> str:
    """Return a filename for UploadedFile, Path, or ImageFileRef objects."""
    if isinstance(file_obj, (str, Path)):
        return Path(file_obj).name
    return str(getattr(file_obj, "name", "uploaded_file"))


def natural_sort_key(file_obj) -> list[object]:
    """Natural filename sort key so image2.tif comes before image10.tif."""
    name = file_display_name(file_obj).lower()
    parts = re.split(r"(\d+)", name)
    return [int(part) if part.isdigit() else part for part in parts]


def sort_image_files(files: Iterable) -> list:
    """Return image-like files sorted by natural filename order."""
    return sorted(list(files or []), key=natural_sort_key)


def make_image_file_refs(files: Iterable) -> list[ImageFileRef]:
    """Sort files naturally and attach original loaded-list indices."""
    refs: list[ImageFileRef] = []
    for idx, source in enumerate(sort_image_files(files)):
        refs.append(
            ImageFileRef(
                source=source,
                name=file_display_name(source),
                original_index_0based=idx,
                original_index_1based=idx + 1,
            )
        )
    return refs


def list_local_image_files(folder_path: str) -> tuple[list[Path], list[str]]:
    """List supported local images from a folder using natural sorting."""
    warnings: list[str] = []
    if not folder_path or not folder_path.strip():
        return [], warnings

    expanded = os.path.expandvars(folder_path.strip().strip('"'))
    path = Path(expanded).expanduser()
    if not path.exists():
        return [], [f"Local folder does not exist: {path}"]
    if not path.is_dir():
        return [], [f"Local path is not a folder: {path}"]

    files = [item for item in path.iterdir() if item.is_file() and item.suffix.lower() in LOCAL_IMAGE_EXTENSIONS]
    files = sort_image_files(files)
    if not files:
        warnings.append(f"No supported image files were found in: {path}")
    return files, warnings


def parse_group_labels(label_text: str, expected_count: int) -> tuple[list[str] | None, str | None]:
    """Parse comma/newline labels and validate against the image count."""
    if not label_text or not label_text.strip():
        return None, None
    raw = label_text.replace(",", "\n").splitlines()
    labels = [item.strip() for item in raw if item.strip()]
    if len(labels) != expected_count:
        return None, f"Group labels ignored: expected {expected_count}, received {len(labels)}."
    return labels, None


def file_names(files: Iterable) -> list[str]:
    """Return uploaded file names for display."""
    return [file_display_name(file) for file in files or []]
