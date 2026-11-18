"""Explicit mask interpretation and shared ROI validation, before measurement."""
from __future__ import annotations
import numpy as np
from .v4_ingestion import array_hash


def supplied_mask(array, *, interpretation, foreground, threshold=None, image_shape=None):
    a = np.asarray(array)
    if a.size == 0 or a.ndim not in (2, 3):
        raise ValueError("Mask must be a non-empty 2-D image (RGB only with explicit any-channel interpretation).")
    if a.dtype.kind not in "buif" or not np.isfinite(a).all():
        raise ValueError("Mask values must be finite numeric values.")
    if image_shape is not None and tuple(a.shape[:2]) != tuple(image_shape[:2]):
        raise ValueError("Mask/image dimension mismatch AFTER recorded orientation. No resizing is permitted.")
    if foreground not in ("nonzero/high", "zero/low"):
        raise ValueError("Confirm foreground polarity explicitly: nonzero/high or zero/low.")
    warnings = []
    labels = None
    if interpretation == "rgb_nonzero":
        if a.ndim != 3 or a.shape[2] != 3:
            raise ValueError("Any-channel mask interpretation requires RGB without alpha.")
        binary = np.any(a != 0, axis=2)
        warnings.append("All nonzero RGB colours merged into one foreground; colour labels/instances are not retained.")
    else:
        if a.ndim != 2:
            raise ValueError("RGB mask is ambiguous. Choose explicit any-channel interpretation or supply grayscale.")
        values = np.unique(a)
        if interpretation == "binary":
            if len(values) > 2 or (len(values) == 2 and values[0] != 0):
                raise ValueError("Binary masks require background 0 and at most one nonzero value; choose labels or an explicit threshold.")
            binary = a != 0
        elif interpretation == "labels":
            if a.dtype.kind not in "ui" or (a < 0).any():
                raise ValueError("Labelled masks require nonnegative integer labels; zero means background.")
            labels = int(np.count_nonzero(values))
            binary = a != 0
            warnings.append("Nonzero labels merged into a binary union. Touching labels become one connected component; instance counts are not structural component counts.")
        elif interpretation == "threshold":
            if threshold is None or not np.isfinite(threshold):
                raise ValueError("A grayscale mask requires an explicit finite threshold in ORIGINAL numeric units.")
            binary = a > float(threshold)
            warnings.append(f"Explicit grayscale-mask threshold: foreground > {threshold} before any polarity inversion. No automatic threshold chosen.")
        else:
            raise ValueError("Choose binary, labels, threshold, or rgb_nonzero mask interpretation.")
    if foreground == "zero/low":
        binary = ~binary
        warnings.append("Explicit polarity inversion: zero/low values are foreground.")
    if not binary.any():
        warnings.append("Empty foreground. Counts/area can be measured; correlation length is unavailable.")
    if binary.all():
        warnings.append("Full foreground. Counts/area can be measured; no background contrast, correlation length unavailable.")
    return np.ascontiguousarray(binary), {"mask_source": "User supplied", "interpretation": interpretation,
            "foreground_polarity": foreground, "threshold_original_units": threshold if interpretation == "threshold" else None,
            "input_label_count": labels, "mask_input_dtype": str(a.dtype), "mask_input_bit_depth": a.dtype.itemsize * 8 if a.dtype != bool else 1,
            "source_mask_array_sha256": array_hash(a), "interpreted_mask_sha256": array_hash(binary),
            "warnings": warnings, "biological_correctness": "Not established; user review is an internal check only"}


def validate_roi(roi, shape):
    if roi is None:
        return (1, 1, int(shape[1]), int(shape[0]))
    if len(roi) != 4:
        raise ValueError("ROI requires x,y,width,height (1-based pixels).")
    if any(isinstance(x, bool) or not np.isfinite(x) or x < 1 or int(x) != x for x in roi):
        raise ValueError("ROI values must be positive whole numbers.")
    x, y, w, h = map(int, roi)
    if x+w-1 > shape[1] or y+h-1 > shape[0]:
        raise ValueError("ROI crosses image boundaries; no clipping applied.")
    if min(w, h) < 2:
        raise ValueError("ROI dimensions must each be at least 2 pixels.")
    return x, y, w, h


def crop(array, roi):
    x, y, w, h = validate_roi(roi, array.shape)
    return np.ascontiguousarray(array[y-1:y+h-1, x-1:x+w-1])
