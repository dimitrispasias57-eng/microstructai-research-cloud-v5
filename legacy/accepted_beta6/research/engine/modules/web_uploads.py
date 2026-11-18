"""Bounded, in-memory image archives for the hosted V3 workflows."""

from __future__ import annotations

import io
import stat
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

from PIL import Image

from modules.io_utils import SUPPORTED_IMAGE_TYPES, sort_image_files

MAX_ARCHIVE_BYTES = 350 * 1024 * 1024
MAX_EXPANDED_BYTES = 512 * 1024 * 1024
MAX_FILES = 5000
MAX_FILE_BYTES = 30 * 1024 * 1024
MAX_IMAGE_PIXELS = 8_000_000


@dataclass(frozen=True)
class MemoryImage:
    name: str
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)

    def getvalue(self) -> bytes:
        return self.data


def read_sample_archive(payload: bytes) -> dict[str, list[MemoryImage]]:
    """Read sample/image folders without writing files or interpreting local paths.

    Accepts sample/frames, dataset/sample/frames, or sample/images/frames.
    Root-level frames form a sample named 'uploaded_sample'. Unsupported files
    are ignored. Ambiguous duplicate sample or frame names are rejected.
    """
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise ValueError("The ZIP exceeds the 350 MiB upload limit. Split the batch into smaller ZIP files.")
    groups: dict[str, list[MemoryImage]] = {}
    sample_paths: dict[str, str] = {}
    names: set[tuple[str, str]] = set()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        if len(members) > MAX_FILES:
            raise ValueError("The ZIP contains more than 5,000 entries. Split the batch.")
        if sum(item.file_size for item in members) > MAX_EXPANDED_BYTES:
            raise ValueError("The uncompressed ZIP exceeds 512 MiB. Split the batch.")
        for item in members:
            normalized = item.filename.replace("\\", "/")
            path = PurePosixPath(normalized)
            if path.is_absolute() or ".." in path.parts or ":" in normalized or "\x00" in normalized:
                raise ValueError("The ZIP contains an unsafe file path.")
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("ZIP symbolic links are not supported.")
            if item.is_dir() or any(part.startswith(".") or part == "__MACOSX" for part in path.parts):
                continue
            if path.suffix.lower().lstrip(".") not in SUPPORTED_IMAGE_TYPES:
                continue
            if item.flag_bits & 1:
                raise ValueError("Encrypted ZIP files are not supported.")
            if item.file_size > MAX_FILE_BYTES:
                raise ValueError(f"{path.name} exceeds the 30 MiB per-image limit.")
            parent = path.parent.parent if path.parent.name.lower() == "images" else path.parent
            sample = parent.name or "uploaded_sample"
            sample_key = sample.casefold()
            if sample_key in sample_paths and sample_paths[sample_key] != str(parent):
                raise ValueError(f"Two folders have the sample name {sample}. Rename one before uploading.")
            sample_paths[sample_key] = str(parent)
            key = (sample_key, path.name.casefold())
            if key in names:
                raise ValueError(f"Duplicate image filename in sample {sample}: {path.name}")
            names.add(key)
            with archive.open(item) as source:
                data = source.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise ValueError(f"{path.name} exceeds the per-image limit.")
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError(f"{path.name} exceeds 8 million pixels. Use a smaller input image.")
                image.verify()
            groups.setdefault(sample, []).append(MemoryImage(path.name, data))
    if not groups:
        raise ValueError("No supported images were found. Include PNG, JPG, TIFF or BMP frames in sample folders.")
    return {name: sort_image_files(files) for name, files in sorted(groups.items())}
