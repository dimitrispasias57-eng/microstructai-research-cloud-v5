"""Segmentation helpers for MicroStructAI MVP.

The RBC Gap Mode target is the bright plasma gap network in low-cost
USB-camera aggregation images. A white mask therefore means plasma gaps,
not red blood cells. Object Mode reuses the same low-level machinery for
bright or dark object segmentation in general datasets.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import color, exposure, filters, morphology, transform, util
from skimage.segmentation import clear_border, find_boundaries

from .io_utils import natural_sort_key


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a float grayscale image in the range 0..1."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3:
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        gray = color.rgb2gray(arr)
    else:
        raise ValueError("Expected a 2D grayscale or 3D RGB/RGBA image.")
    gray = util.img_as_float32(gray)
    return normalize_image(gray)


def to_grayscale_float01(image: np.ndarray) -> np.ndarray:
    """Return a grayscale float image in [0, 1] without contrast stretching."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]

    if np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_:
        arr = util.img_as_float32(arr)
    else:
        arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 3:
        gray = color.rgb2gray(arr)
    elif arr.ndim == 2:
        gray = arr
    else:
        raise ValueError("Expected a 2D grayscale or 3D RGB/RGBA image.")

    gray = np.nan_to_num(gray.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.min(gray)) < 0.0 or float(np.max(gray)) > 1.0:
        lo, hi = float(np.min(gray)), float(np.max(gray))
        if hi > lo:
            gray = (gray - lo) / (hi - lo)
        else:
            gray = np.zeros_like(gray, dtype=np.float32)
    return np.clip(gray, 0.0, 1.0).astype(np.float32)


def to_matlab_grayscale_float01(image: np.ndarray) -> np.ndarray:
    """Return a MATLAB ``rgb2gray``-like grayscale image scaled to [0, 1].

    MATLAB's reference pipeline runs ``rgb2gray`` before ``double``. For the
    mostly grayscale RBC image sets this is identical in practice, but using
    MATLAB's RGB coefficients avoids an avoidable source of mismatch for RGB
    inputs. We keep Object Mode on the existing generic grayscale path.
    """
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]

    if np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_:
        arr_float = util.img_as_float32(arr)
    else:
        arr_float = np.asarray(arr, dtype=np.float32)
        if float(np.nanmin(arr_float)) < 0.0 or float(np.nanmax(arr_float)) > 1.0:
            lo, hi = float(np.nanmin(arr_float)), float(np.nanmax(arr_float))
            arr_float = (arr_float - lo) / (hi - lo) if hi > lo else np.zeros_like(arr_float)

    if arr_float.ndim == 3:
        if arr_float.shape[-1] < 3:
            gray = arr_float[..., 0]
        else:
            # MATLAB rgb2gray uses NTSC luminance coefficients.
            gray = 0.29893602 * arr_float[..., 0] + 0.58704307 * arr_float[..., 1] + 0.11402090 * arr_float[..., 2]
    elif arr_float.ndim == 2:
        gray = arr_float
    else:
        raise ValueError("Expected a 2D grayscale or 3D RGB/RGBA image.")

    gray = np.nan_to_num(gray.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(gray, 0.0, 1.0).astype(np.float32)


def normalize_image(gray: np.ndarray) -> np.ndarray:
    """Robustly normalize an image to 0..1 while ignoring non-finite values."""
    arr = np.asarray(gray, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return arr.astype(np.float32)


def robust_display_scale(image: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Scale an image to [0, 1] using robust display percentiles."""
    arr = np.asarray(image, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi <= lo:
        lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def mat2gray_like(image: np.ndarray) -> np.ndarray:
    """Scale an array to [0, 1] like MATLAB ``mat2gray``."""
    arr = np.asarray(image, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def matlab_gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """Approximate MATLAB ``imgaussfilt(image, sigma)`` default support.

    MATLAB's default filter size is ``2*ceil(2*sigma)+1``. SciPy's
    ``gaussian_filter`` defaults to a wider 4-sigma truncation, which can
    change residuals and therefore seed/grow masks. ``mode='nearest'`` matches
    MATLAB's replicate-style boundary handling.
    """
    sigma = float(sigma)
    arr = np.asarray(image, dtype=np.float64)
    if sigma <= 0:
        return arr.copy()
    truncate = float(np.ceil(2.0 * sigma) / sigma)
    return ndi.gaussian_filter(arr, sigma=sigma, mode="nearest", truncate=truncate)


def _matlab_round_positive(value: float) -> int:
    """Round positive values like MATLAB ``round`` for crop-size parity."""
    return int(np.floor(float(value) + 0.5))


def _center_crop_bounds(shape: tuple[int, ...], crop_frac: float) -> dict[str, int | float | bool]:
    """Return Python/MATLAB crop bounds for the centered RBC analysis crop."""
    frac = float(np.clip(crop_frac, 0.05, 1.0))
    h, w = int(shape[0]), int(shape[1])
    if frac >= 0.999:
        crop_h, crop_w = h, w
    else:
        crop_h = max(1, _matlab_round_positive(h * frac))
        crop_w = max(1, _matlab_round_positive(w * frac))
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    return {
        "cropFrac": frac,
        "cropApplied": bool(frac < 0.999),
        "originalHeight": h,
        "originalWidth": w,
        "cropHeight": crop_h,
        "cropWidth": crop_w,
        "cropY0_0based": y0,
        "cropX0_0based": x0,
        "cropY1_0based_exclusive": y0 + crop_h,
        "cropX1_0based_exclusive": x0 + crop_w,
        "cropRowStart_1based": y0 + 1,
        "cropColStart_1based": x0 + 1,
        "cropRowEnd_1based": y0 + crop_h,
        "cropColEnd_1based": x0 + crop_w,
        "resizedBeforeAnalysis": False,
    }


def apply_crop(image: np.ndarray, params: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply Conservative RBC crop settings to a grayscale image."""
    params = dict(params or {})
    if bool(params.get("useMatlabGrayscale", False)):
        gray = to_matlab_grayscale_float01(image)
    else:
        gray = to_grayscale_float01(image)
    use_crop = bool(params.get("useCrop", True))
    if not use_crop:
        bounds = {
            "cropApplied": False,
            "cropMode": "none",
            "cropFrac": 1.0,
            "cropY0_0based": 0,
            "cropY1_0based_exclusive": int(gray.shape[0]),
            "cropX0_0based": 0,
            "cropX1_0based_exclusive": int(gray.shape[1]),
            "cropRowStart_1based": 1,
            "cropRowEnd_1based": int(gray.shape[0]),
            "cropColStart_1based": 1,
            "cropColEnd_1based": int(gray.shape[1]),
        }
        return gray.astype(np.float32), bounds

    mode = str(params.get("cropMode", "center")).lower()
    if mode == "manual":
        rect = params.get("manualCropRect", "")
        try:
            values = [float(part.strip()) for part in str(rect).replace(";", ",").split(",") if part.strip()]
        except Exception:
            values = []
        if len(values) == 4:
            x, y, width, height = values
            image_h, image_w = gray.shape
            x0 = max(0, int(round(x)) - 1)
            y0 = max(0, int(round(y)) - 1)
            x1 = min(image_w, int(round(x + width - 1)))
            y1 = min(image_h, int(round(y + height - 1)))
            if x1 > x0 and y1 > y0:
                bounds = {
                    "cropApplied": True,
                    "cropMode": "manual",
                    "cropFrac": np.nan,
                    "cropY0_0based": y0,
                    "cropY1_0based_exclusive": y1,
                    "cropX0_0based": x0,
                    "cropX1_0based_exclusive": x1,
                    "cropRowStart_1based": y0 + 1,
                    "cropRowEnd_1based": y1,
                    "cropColStart_1based": x0 + 1,
                    "cropColEnd_1based": x1,
                }
                return gray[y0:y1, x0:x1].astype(np.float32), bounds

    bounds = _center_crop_bounds(gray.shape, float(params.get("cropFrac", 0.70)))
    y0 = int(bounds["cropY0_0based"])
    y1 = int(bounds["cropY1_0based_exclusive"])
    x0 = int(bounds["cropX0_0based"])
    x1 = int(bounds["cropX1_0based_exclusive"])
    bounds["cropMode"] = "center"
    return gray[y0:y1, x0:x1].astype(np.float32), dict(bounds)


def safe_spatial_illumination_correction(image: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
    """Conservative spatial flat-field correction from the MATLAB prototype."""
    params = dict(params or {})
    preserve_mean = bool(params.get("preserveSpatialMeanIntensity", True))
    arr = np.asarray(image, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if preserve_mean and arr.size and float(np.nanmin(arr)) >= 0.0 and float(np.nanmax(arr)) <= 1.0:
        # Quantitative v6 path: keep the incoming 0-1 grayscale intensity scale
        # so spatial flat-field correction cannot invent optical mean trends.
        image01 = np.clip(arr, 0.0, 1.0).astype(np.float32)
    else:
        image01 = mat2gray_like(arr)
    if not bool(params.get("applySpatialIlluminationCorrection", True)):
        return image01.astype(np.float32)
    original_mean = float(np.nanmean(image01))
    sigma_ill = max(3.0, float(params.get("spatialIllumSigma", 60.0)))
    method = str(params.get("spatialCorrectionMethod", "divide")).lower()
    strength = float(np.clip(float(params.get("spatialCorrectionStrength", 0.50)), 0.0, 1.0))
    do_clip = bool(params.get("clipSpatialCorrectedImages", True))
    bg = matlab_gaussian_filter(image01, sigma_ill)
    if method == "subtract":
        full = image01 - bg + float(np.mean(bg))
    else:
        bg_norm = bg / (float(np.mean(bg)) + np.finfo(float).eps)
        full = image01 / np.maximum(bg_norm, np.finfo(float).eps)
    if do_clip:
        finite = full[np.isfinite(full)]
        if finite.size:
            lo, hi = np.percentile(finite, [0.5, 99.5])
            full = np.clip(full, lo, hi)
    # Match the MATLAB v6 conservative prototype: clip the correction result
    # but do not mat2gray-rescale the no-POD spatial-corrected image. Per-frame
    # mat2gray is reserved for the POD reconstruction branch and can otherwise
    # amplify early smooth texture into artificial dense masks.
    corrected = (1.0 - strength) * image01 + strength * full
    if preserve_mean:
        corrected_mean = float(np.nanmean(corrected))
        corrected = corrected * original_mean / max(corrected_mean, np.finfo(float).eps)
        return np.clip(corrected, 0.0, 1.0).astype(np.float32)
    return mat2gray_like(corrected)


def estimate_illumination_gain_simple(
    mean_raw: np.ndarray,
    enable_correction: bool = True,
    fixed_log_jump_threshold: float = float(np.log(1.006)),
    adaptive_mad_mult: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate simple multiplicative gain correction from mean-intensity jumps."""
    y = np.asarray(mean_raw, dtype=np.float64).reshape(-1)
    n = y.size
    gain = np.ones(n, dtype=np.float64)
    corrected = y.copy()
    jump_flag = np.zeros(n, dtype=bool)
    if not enable_correction or n < 3:
        return gain, corrected, jump_flag
    work = y.copy()
    work[~np.isfinite(work) | (work <= 0)] = np.nan
    dlog = np.full(n, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlog[1:] = np.diff(np.log(work))
    finite = dlog[np.isfinite(dlog)]
    if finite.size == 0:
        return gain, corrected, jump_flag
    med_d = float(np.median(finite))
    mad_d = float(np.median(np.abs(finite - med_d))) + np.finfo(float).eps
    threshold = max(float(fixed_log_jump_threshold), float(adaptive_mad_mult) * mad_d)
    jump_flag = np.isfinite(dlog) & (np.abs(dlog - med_d) > threshold)
    current_gain = 1.0
    for idx in range(1, n):
        if jump_flag[idx] and np.isfinite(work[idx]) and np.isfinite(work[idx - 1]) and work[idx] > 0:
            ratio = work[idx] / max(work[idx - 1], np.finfo(float).eps)
            if np.isfinite(ratio) and ratio > 0:
                current_gain = current_gain / ratio
        gain[idx] = current_gain
    corrected = y * gain
    return gain, corrected, jump_flag


def matlab_adaptthresh_bright_like(
    image01: np.ndarray,
    block_size: int,
    sensitivity: float = 0.45,
    mean_scale_base: float = 1.60,
) -> np.ndarray:
    """Approximate MATLAB ``adaptthresh`` for bright conservative masks.

    MATLAB's exact ``adaptthresh`` implementation is not available in
    scikit-image. This compatibility path keeps the same user-facing idea as
    ``adaptthresh(Is, sensitivity, ForegroundPolarity="bright")``: local
    thresholding with higher sensitivity accepting more bright foreground. A
    direct export of MATLAB's threshold image from the conservative prototype
    showed sensitivity=0.45 is effectively ``1.15 * localMean(Is)`` with the
    configured neighbourhood, so the default multiplier is
    ``mean_scale_base - sensitivity = 1.60 - 0.45``.
    """
    img = np.asarray(image01, dtype=np.float64)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    if img.size == 0:
        return img
    block = max(3, int(block_size))
    if block % 2 == 0:
        block += 1
    sensitivity = float(np.clip(float(sensitivity), 0.0, 1.0))
    mean_scale_base = max(0.0, float(mean_scale_base))
    local_mean = ndi.uniform_filter(img, size=block, mode="nearest")
    threshold = local_mean * max(0.0, mean_scale_base - sensitivity)
    return np.clip(threshold, 0.0, 1.0)


def make_gap_mask_conservative(image: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
    """Create a conservative bright plasma-gap mask from a corrected no-POD image."""
    params = dict(params or {})
    image01 = mat2gray_like(image)
    method = str(params.get("conservativeSegMethod", "adaptive")).lower()
    min_area = int(params.get("minAreaPx", params.get("minGapAreaPx", 20)))
    sigma = float(params.get("conservativeSmoothSigma", 1.0))
    smooth = filters.gaussian(image01, sigma=sigma, preserve_range=True) if sigma > 0 else image01
    if method == "otsu":
        threshold = filters.threshold_otsu(smooth) if smooth.size else 1.0
        mask = smooth > threshold
    else:
        block_size = max(21, int(2 * np.floor(min(smooth.shape) / 12) + 1))
        if block_size % 2 == 0:
            block_size += 1
        engine = str(params.get("conservativeAdaptiveEngine", "MATLAB adaptthresh-like")).strip().lower()
        if "matlab" in engine or "adaptthresh" in engine:
            local_threshold = matlab_adaptthresh_bright_like(
                smooth,
                block_size=block_size,
                sensitivity=float(params.get("conservativeAdaptSensitivity", 0.45)),
                mean_scale_base=float(params.get("conservativeAdaptMeanScaleBase", 1.60)),
            )
        else:
            # Legacy/current app behaviour. Keep this selectable so older
            # Conservative RBC/Object results can be reproduced exactly.
            offset = float(params.get("conservativeAdaptiveOffset", -0.043))
            local_threshold = filters.threshold_local(smooth, block_size=block_size, offset=offset)
        mask = smooth > local_threshold
    mask = _bwareaopen_matlab(mask.astype(bool), max(0, min_area))
    if np.any(mask):
        mask = morphology.opening(mask, morphology.disk(1))
    return mask.astype(bool)


def simple_gap_metrics(mask: np.ndarray) -> dict[str, Any]:
    """Return simple connected-component gap metrics for a binary mask."""
    bw = np.asarray(mask, dtype=bool)
    labels, n_components = ndi.label(bw, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel())[1:].astype(np.float64) if n_components else np.asarray([], dtype=np.float64)
    if areas.size:
        diams = 2.0 * np.sqrt(areas / np.pi)
        return {
            "gapAreaFrac": float(np.mean(bw)),
            "gapNtotal": int(len(areas)),
            "gapMeanArea": float(np.mean(areas)),
            "gapMeanDiam_px": float(np.mean(diams)),
            "gapMaxDiam_px": float(np.max(diams)),
            "largestGapFrac": float(np.max(areas) / bw.size) if bw.size else np.nan,
        }
    return {
        "gapAreaFrac": float(np.mean(bw)) if bw.size else np.nan,
        "gapNtotal": 0,
        "gapMeanArea": np.nan,
        "gapMeanDiam_px": np.nan,
        "gapMaxDiam_px": np.nan,
        "largestGapFrac": 0.0,
    }


def compute_pod_svd_and_optional_removal(stack: np.ndarray, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run POD/SVD and reconstruct a stack after optional selected-mode removal."""
    params = dict(params or {})
    stack_arr = np.asarray(stack, dtype=np.float64)
    if stack_arr.ndim != 3 or stack_arr.shape[0] == 0:
        return {"pod_removed_stack": stack_arr, "podMode1Energy": np.nan, "first5PODCumulativeEnergy": np.nan, "pod_energy": np.asarray([])}
    remove_modes_requested = max(0, int(params.get("removeModes", 1)))
    compute_zero_mode_branch = bool(params.get("computePODWhenRemoveModesZero", False))
    if remove_modes_requested <= 0 and not compute_zero_mode_branch:
        return {
            "pod_removed_stack": stack_arr.copy().astype(np.float32),
            "podMode1Energy": np.nan,
            "first5PODCumulativeEnergy": np.nan,
            "pod_energy": np.asarray([]),
        }
    n_frames, height, width = stack_arr.shape
    x = stack_arr.reshape(n_frames, height * width).T
    use_mean_subtract = bool(params.get("useMeanSubtract", False))
    mean_vec = np.mean(x, axis=1, keepdims=True)
    xpod = x - mean_vec if use_mean_subtract else x.copy()
    try:
        u, s, vt = np.linalg.svd(xpod, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"pod_removed_stack": stack_arr.copy(), "podMode1Energy": np.nan, "first5PODCumulativeEnergy": np.nan, "pod_energy": np.asarray([])}
    denom = float(np.sum(s**2))
    energy = (s**2 / denom) if denom > 0 else np.full_like(s, np.nan, dtype=float)
    xrec = xpod.copy()
    remove_modes = min(remove_modes_requested, u.shape[1])
    for mode_index in range(remove_modes):
        xrec -= np.outer(u[:, mode_index] * s[mode_index], vt[mode_index, :])
    if use_mean_subtract:
        xrec += mean_vec
    rec = xrec.T.reshape(n_frames, height, width)
    rec_scaled = np.stack([mat2gray_like(rec[idx]) for idx in range(n_frames)], axis=0)
    return {
        "pod_removed_stack": rec_scaled.astype(np.float32),
        "podMode1Energy": float(energy[0]) if energy.size else np.nan,
        "first5PODCumulativeEnergy": float(np.nansum(energy[: min(5, energy.size)])) if energy.size else np.nan,
        "pod_energy": energy,
    }


def compute_mean_intensity_curves(stack: np.ndarray) -> dict[str, np.ndarray]:
    """Compute frame-wise mean/std/CV curves for a stack."""
    arr = np.asarray(stack, dtype=np.float64)
    if arr.ndim != 3:
        return {"mean": np.asarray([]), "std": np.asarray([]), "cv": np.asarray([])}
    mean_values = np.mean(arr, axis=(1, 2))
    std_values = np.std(arr, axis=(1, 2))
    cv_values = std_values / (np.abs(mean_values) + np.finfo(float).eps)
    return {"mean": mean_values, "std": std_values, "cv": cv_values}


def _center_crop(image: np.ndarray, crop_frac: float) -> np.ndarray:
    """Crop the center region used by the RBC MATLAB-style pipeline."""
    bounds = _center_crop_bounds(image.shape, crop_frac)
    y0 = int(bounds["cropY0_0based"])
    y1 = int(bounds["cropY1_0based_exclusive"])
    x0 = int(bounds["cropX0_0based"])
    x1 = int(bounds["cropX1_0based_exclusive"])
    return image[y0:y1, x0:x1]


def _odd_kernel(value: int) -> int:
    """Return a positive odd median-filter kernel size."""
    kernel = max(1, int(value))
    if kernel % 2 == 0:
        kernel += 1
    return kernel


def _remove_small_objects(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than ``min_area`` with version tolerance."""
    if min_area <= 0:
        return np.asarray(mask, dtype=bool)
    try:
        return morphology.remove_small_objects(mask, max_size=max(0, min_area - 1))
    except TypeError:
        return morphology.remove_small_objects(mask, min_size=min_area)


def _bwareaopen_matlab(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove small objects using MATLAB ``bwareaopen`` 2-D defaults.

    MATLAB uses 8-connectivity for 2-D binary images. scikit-image's default
    boolean connectivity is lower, so use explicit 8-connected labeling here
    for the RBC reference mask path.
    """
    bw = np.asarray(mask, dtype=bool)
    threshold = int(min_area)
    if threshold <= 0 or not np.any(bw):
        return bw
    labels, n_labels = ndi.label(bw, structure=np.ones((3, 3), dtype=np.uint8))
    if n_labels == 0:
        return np.zeros_like(bw, dtype=bool)
    areas = np.bincount(labels.ravel())
    keep_labels = np.flatnonzero(areas >= threshold)
    keep_labels = keep_labels[keep_labels != 0]
    return np.isin(labels, keep_labels)


def filter_large_gap_objects(
    mask: np.ndarray,
    enabled: bool = True,
    max_gap_area_frac: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Remove suspiciously large single plasma-gap objects from an RBC mask.

    This is an RBC Gap Mode artefact-control layer. It runs after seed-grow
    segmentation cleanup and before metric extraction. It does not change the
    seed-grow formula itself. When disabled, the original mask is preserved
    while warning diagnostics are still calculated for MATLAB compatibility.
    """
    gap_mask = np.asarray(mask, dtype=bool)
    removed_mask = np.zeros_like(gap_mask, dtype=bool)
    threshold = float(np.clip(max_gap_area_frac, 0.001, 0.35))
    image_area = float(gap_mask.size)
    labels, n_labels = ndi.label(gap_mask, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel())[1:].astype(np.float64) if n_labels else np.asarray([], dtype=np.float64)
    fractions = areas / image_area if image_area > 0 else np.asarray([], dtype=np.float64)
    suspicious_labels = np.flatnonzero(fractions > threshold) + 1
    suspicious_fractions = fractions[fractions > threshold]
    largest_before = float(np.max(fractions)) if fractions.size else 0.0
    largest_suspicious = float(np.max(suspicious_fractions)) if suspicious_fractions.size else 0.0

    filtered_mask = gap_mask.copy()
    if enabled and suspicious_labels.size:
        removed_mask = np.isin(labels, suspicious_labels)
        filtered_mask[removed_mask] = False

    filtered_labels, filtered_count = ndi.label(filtered_mask, structure=np.ones((3, 3), dtype=np.uint8))
    filtered_areas = np.bincount(filtered_labels.ravel())[1:].astype(np.float64) if filtered_count else np.asarray([], dtype=np.float64)
    largest_after = float(np.max(filtered_areas) / image_area) if filtered_areas.size and image_area > 0 else 0.0

    if not suspicious_labels.size:
        artefact_flag = "none"
    elif not enabled:
        artefact_flag = "disabled_warning"
    elif largest_suspicious > 0.20:
        artefact_flag = "high"
    else:
        artefact_flag = "moderate"

    diagnostics: dict[str, Any] = {
        "maxGapAreaFilterEnabled": bool(enabled),
        "maxGapAreaFrac": threshold,
        "nLargeGapObjectsRemoved": int(suspicious_labels.size) if enabled else 0,
        "nLargeGapObjectsDetected": int(suspicious_labels.size),
        "largestRemovedGapFrac": largest_suspicious,
        "largestGapFrac_beforeFilter": largest_before,
        "largestGapFrac_afterFilter": largest_after,
        "largeGapArtefactFlag": artefact_flag,
    }
    return filtered_mask.astype(bool), removed_mask.astype(bool), diagnostics


def filter_large_structure_objects(
    mask: np.ndarray,
    enabled: bool = False,
    max_object_area_frac: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Remove suspiciously large objects from an enhanced generic mask.

    This is an optional Object Mode safety layer. It uses generic object
    terminology deliberately: detected structures are not RBC plasma gaps.
    """
    structure_mask = np.asarray(mask, dtype=bool)
    removed_mask = np.zeros_like(structure_mask, dtype=bool)
    threshold = float(np.clip(max_object_area_frac, 0.001, 0.95))
    image_area = float(structure_mask.size)
    labels, n_labels = ndi.label(structure_mask, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel())[1:].astype(np.float64) if n_labels else np.asarray([], dtype=np.float64)
    fractions = areas / image_area if image_area > 0 else np.asarray([], dtype=np.float64)
    suspicious_labels = np.flatnonzero(fractions > threshold) + 1
    suspicious_fractions = fractions[fractions > threshold]
    largest_before = float(np.max(fractions)) if fractions.size else 0.0
    largest_suspicious = float(np.max(suspicious_fractions)) if suspicious_fractions.size else 0.0

    filtered_mask = structure_mask.copy()
    if enabled and suspicious_labels.size:
        removed_mask = np.isin(labels, suspicious_labels)
        filtered_mask[removed_mask] = False

    filtered_labels, filtered_count = ndi.label(filtered_mask, structure=np.ones((3, 3), dtype=np.uint8))
    filtered_areas = np.bincount(filtered_labels.ravel())[1:].astype(np.float64) if filtered_count else np.asarray([], dtype=np.float64)
    largest_after = float(np.max(filtered_areas) / image_area) if filtered_areas.size and image_area > 0 else 0.0
    if not suspicious_labels.size:
        artefact_flag = "none"
    elif not enabled:
        artefact_flag = "disabled_warning"
    elif largest_suspicious > 0.50:
        artefact_flag = "high"
    else:
        artefact_flag = "moderate"

    diagnostics: dict[str, Any] = {
        "maxObjectAreaFilterEnabled": bool(enabled),
        "maxObjectAreaFrac": threshold,
        "nLargeObjectsRemoved": int(suspicious_labels.size) if enabled else 0,
        "nLargeObjectsDetected": int(suspicious_labels.size),
        "largestRemovedObjectFrac": largest_suspicious,
        "largestObjectFrac_beforeFilter": largest_before,
        "largestObjectFrac_afterFilter": largest_after,
        "largeObjectArtefactFlag": artefact_flag,
    }
    return filtered_mask.astype(bool), removed_mask.astype(bool), diagnostics


def _odd_block_size(value: int, shape: tuple[int, int]) -> int:
    """Choose a legal odd adaptive-threshold block size for the image shape."""
    min_dim = max(3, min(shape))
    block = int(max(3, value))
    if block % 2 == 0:
        block += 1
    if block >= min_dim:
        block = min_dim - 1 if min_dim % 2 == 0 else min_dim
    if block % 2 == 0:
        block -= 1
    return max(3, block)


def _preprocess(gray: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Apply optional denoising, background correction, and contrast enhancement."""
    work = normalize_image(gray)

    sigma = float(params.get("smoothing_sigma", 1.0) or 0.0)
    if sigma > 0:
        work = filters.gaussian(work, sigma=sigma, preserve_range=True)

    if params.get("background_correction", False):
        bg_sigma = float(params.get("background_sigma", max(8, min(work.shape) / 12)))
        background = filters.gaussian(work, sigma=bg_sigma, preserve_range=True)
        work = normalize_image(work - background)

    if params.get("contrast_enhancement", True):
        clip_limit = float(params.get("clahe_clip_limit", 0.02))
        work = exposure.equalize_adapthist(work, clip_limit=clip_limit)

    return normalize_image(work)


def _threshold_image(work: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Threshold a preprocessed image into a binary target mask."""
    method = str(params.get("threshold_method", "otsu")).lower()
    polarity = str(params.get("polarity", "bright")).lower()
    bright_target = polarity != "dark"

    if method == "adaptive":
        block_size = _odd_block_size(int(params.get("adaptive_block_size", 51)), work.shape)
        offset = float(params.get("adaptive_offset", 0.0))
        thresh = filters.threshold_local(work, block_size=block_size, offset=offset)
        mask = work > thresh if bright_target else work < thresh
    elif method == "percentile":
        percentile = float(params.get("percentile", 80.0))
        percentile = min(max(percentile, 1.0), 99.0)
        if bright_target:
            thresh = np.percentile(work, percentile)
            mask = work >= thresh
        else:
            thresh = np.percentile(work, 100.0 - percentile)
            mask = work <= thresh
    else:
        try:
            thresh = filters.threshold_otsu(work)
        except ValueError:
            thresh = float(np.mean(work))
        mask = work > thresh if bright_target else work < thresh

    return mask.astype(bool)


def _postprocess_mask(mask: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Clean a binary mask using size filtering and optional hole filling."""
    min_size = int(params.get("min_object_size", 40) or 0)
    clean = np.asarray(mask, dtype=bool)

    if min_size > 0:
        clean = _remove_small_objects(clean, min_size)

    if params.get("fill_holes", True):
        hole_area = int(params.get("hole_area", max(16, min_size)))
        try:
            clean = morphology.remove_small_holes(clean, max_size=max(0, hole_area - 1))
        except TypeError:
            clean = morphology.remove_small_holes(clean, area_threshold=hole_area)
        clean = ndi.binary_fill_holes(clean)

    if params.get("binary_closing", False):
        radius = int(params.get("closing_radius", 1))
        if radius > 0:
            clean = morphology.closing(clean, morphology.disk(radius))

    return clean.astype(bool)


def segment_rbc_gaps(image: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
    """Segment plasma gaps in RBC aggregation images.

    White/True pixels represent plasma gaps. The default polarity is bright
    because the research pipeline targets bright plasma gaps rather than RBCs.
    """
    return segment_rbc_gaps_seedgrow(image, params)["mask"]


def compute_rbc_ximg01_matlab_like(image: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
    """Return the MATLAB-style continuous Ximg01 field for RBC Xi/LF metrics.

    This preprocessing is separate from the seed/grow plasma-gap segmentation.
    It follows the legacy MATLAB feature field: crop, broad background
    subtraction, optional median filter, percentile normalization, column/global
    mean removal, difference-of-Gaussians, robust-z clipping, then [0, 1] scale.
    """
    params = dict(params or {})
    img = to_matlab_grayscale_float01(image).astype(np.float64)
    img = _center_crop(img, params.get("cropFrac", 0.70)).astype(np.float64)

    bg_sigma = float(params.get("ximg_bgSigma", 40.0))
    bg = matlab_gaussian_filter(img, bg_sigma)
    img_bs = img - bg

    if bool(params.get("useMedian", True)):
        img_bs = ndi.median_filter(img_bs, size=_odd_kernel(int(params.get("medianK", 3))), mode="constant", cval=0.0)

    p_low = float(params.get("ximg_pLow", 1.0))
    p_high = float(params.get("ximg_pHigh", 99.0))
    p1, p99 = np.percentile(img_bs, [p_low, p_high])
    if p99 <= p1:
        return np.zeros_like(img_bs, dtype=np.float32)

    img_n = (img_bs - p1) / (p99 - p1 + np.finfo(float).eps)
    img_n = np.clip(img_n, 0.0, 1.0)
    img_n = img_n - np.mean(img_n, axis=0, keepdims=True)
    img_n = img_n - float(np.mean(img_n))

    sigma_blob = float(params.get("ximg_sigmaBlob", 6.0))
    sigma_bg = float(params.get("ximg_sigmaBG", 25.0))
    lp_blob = matlab_gaussian_filter(img_n, sigma_blob)
    lp_bg = matlab_gaussian_filter(img_n, sigma_bg)
    agg_field = lp_blob - lp_bg
    agg_field = agg_field - float(np.mean(agg_field))

    median_value = float(np.median(agg_field))
    mad_value = float(np.median(np.abs(agg_field - median_value)))
    if not np.isfinite(mad_value) or mad_value <= np.finfo(float).eps:
        mad_value = float(np.std(agg_field))
    if not np.isfinite(mad_value) or mad_value <= np.finfo(float).eps:
        return np.zeros_like(agg_field, dtype=np.float32)

    az = (agg_field - median_value) / (mad_value + np.finfo(float).eps)
    az = np.clip(az, -6.0, 6.0)
    return ((az + 6.0) / 12.0).astype(np.float32)


def compute_rbc_matlab_quality_frame(image: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
    """Return the MATLAB-like normalized frame used for motion quality D(t).

    This mirrors the pre-POD ``imgN`` calculation used before the MATLAB
    quality filter. It is used only for frame inclusion compatibility, not for
    changing segmentation or metric formulas.
    """
    params = dict(params or {})
    img = to_matlab_grayscale_float01(image).astype(np.float64)
    img = _center_crop(img, params.get("cropFrac", 0.70)).astype(np.float64)

    bg_sigma = float(params.get("ximg_bgSigma", 40.0))
    bg = matlab_gaussian_filter(img, bg_sigma)
    img_bs = img - bg

    if bool(params.get("useMedian", True)):
        img_bs = ndi.median_filter(img_bs, size=_odd_kernel(int(params.get("medianK", 3))), mode="constant", cval=0.0)

    p_low = float(params.get("ximg_pLow", 1.0))
    p_high = float(params.get("ximg_pHigh", 99.0))
    p1, p99 = np.percentile(img_bs, [p_low, p_high])
    if p99 <= p1:
        return np.zeros_like(img_bs, dtype=np.float32)

    img_n = (img_bs - p1) / (p99 - p1 + np.finfo(float).eps)
    img_n = np.clip(img_n, 0.0, 1.0)
    img_n = img_n - np.mean(img_n, axis=0, keepdims=True)
    img_n = img_n - float(np.mean(img_n))
    return img_n.astype(np.float32)


def compute_rbc_iraw01_matlab_like(image: np.ndarray, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return MATLAB's processed analysis grayscale used for gap segmentation.

    The legacy MATLAB code names this field ``Iraw01``, but it is not the
    untouched camera image. It is a crop, broad-background-subtracted,
    percentile-normalized, column/global de-meaned image that is finally
    scaled with ``mat2gray``. RBC Gap Mode should threshold this analysis
    grayscale when reproducing MATLAB masks.
    """
    params = dict(params or {})
    raw_gray = to_matlab_grayscale_float01(image).astype(np.float64)
    crop_bounds = _center_crop_bounds(raw_gray.shape, params.get("cropFrac", 0.70))
    y0 = int(crop_bounds["cropY0_0based"])
    y1 = int(crop_bounds["cropY1_0based_exclusive"])
    x0 = int(crop_bounds["cropX0_0based"])
    x1 = int(crop_bounds["cropX1_0based_exclusive"])
    camera_crop_gray = raw_gray[y0:y1, x0:x1].astype(np.float64)

    bg_sigma = float(params.get("ximg_bgSigma", 40.0))
    preprocessing_background = matlab_gaussian_filter(camera_crop_gray, bg_sigma)
    img_bs = camera_crop_gray - preprocessing_background

    median_k = _odd_kernel(int(params.get("medianK", 3)))
    median_applied = bool(params.get("useMedian", True))
    if median_applied:
        img_bs_for_norm = ndi.median_filter(img_bs, size=median_k, mode="constant", cval=0.0)
    else:
        img_bs_for_norm = img_bs

    p_low = float(params.get("ximg_pLow", 1.0))
    p_high = float(params.get("ximg_pHigh", 99.0))
    p1, p99 = np.percentile(img_bs_for_norm, [p_low, p_high])
    if p99 <= p1:
        img_n = np.zeros_like(img_bs_for_norm, dtype=np.float64)
    else:
        img_n = (img_bs_for_norm - p1) / (p99 - p1 + np.finfo(float).eps)
        img_n = np.clip(img_n, 0.0, 1.0)
        img_n = img_n - np.mean(img_n, axis=0, keepdims=True)
        img_n = img_n - float(np.mean(img_n))

    iraw01 = mat2gray_like(img_n)
    return {
        "raw_gray": raw_gray.astype(np.float32),
        "camera_crop_gray": camera_crop_gray.astype(np.float32),
        "analysis_gray": iraw01.astype(np.float32),
        "preprocessing_background": preprocessing_background.astype(np.float32),
        "preprocessing_residual": img_bs.astype(np.float32),
        "preprocessing_residual_for_norm": img_bs_for_norm.astype(np.float32),
        "imgN_pre_mat2gray": img_n.astype(np.float32),
        "crop_bounds": crop_bounds,
        "ximg_bgSigma": bg_sigma,
        "ximg_pLow": p_low,
        "ximg_pHigh": p_high,
        "ximg_p1": float(p1),
        "ximg_p99": float(p99),
        "preprocessingMedianApplied": median_applied,
        "preprocessingMedianK": median_k,
    }


def _mask_component_summary(mask: np.ndarray) -> dict[str, Any]:
    """Return component count and largest fraction for one binary mask."""
    bw = np.asarray(mask, dtype=bool)
    labels, n_components = ndi.label(bw, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel())[1:].astype(np.float64) if n_components else np.asarray([], dtype=np.float64)
    largest_frac = float(np.max(areas) / bw.size) if areas.size and bw.size else 0.0
    return {
        "componentCount": int(n_components),
        "areaFrac": float(np.mean(bw)) if bw.size else np.nan,
        "largestFrac": largest_frac,
    }


def segment_rbc_gaps_seedgrow(image: np.ndarray, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Segment bright plasma gaps using a MATLAB-style seed/grow pipeline.

    White/True pixels are bright plasma gaps. Black/background pixels are
    RBC-rich or aggregate-rich regions. This function is specific to RBC Gap
    Mode and should not be used for generic object segmentation.
    """
    params = dict(params or {})

    input_source = str(params.get("rbcSegmentationInputSource", "MATLAB Iraw01 processed grayscale"))
    include_debug_stages = bool(params.get("include_debug_stages", False))
    if input_source == "Raw grayscale crop (legacy app)":
        raw_gray = to_grayscale_float01(image)
        crop_bounds = _center_crop_bounds(raw_gray.shape, params.get("cropFrac", 0.70))
        y0 = int(crop_bounds["cropY0_0based"])
        y1 = int(crop_bounds["cropY1_0based_exclusive"])
        x0 = int(crop_bounds["cropX0_0based"])
        x1 = int(crop_bounds["cropX1_0based_exclusive"])
        camera_crop_gray = raw_gray[y0:y1, x0:x1].astype(np.float32)
        analysis_gray = camera_crop_gray.astype(np.float64)
        preprocessing_debug: dict[str, Any] = {
            "crop_bounds": crop_bounds,
            "preprocessingMedianApplied": False,
            "preprocessingMedianK": _odd_kernel(int(params.get("medianK", 3))),
            "ximg_bgSigma": np.nan,
            "ximg_pLow": np.nan,
            "ximg_pHigh": np.nan,
            "ximg_p1": np.nan,
            "ximg_p99": np.nan,
        }
        if bool(params.get("useMedian", True)):
            analysis_gray = ndi.median_filter(analysis_gray, size=_odd_kernel(int(params.get("medianK", 3))), mode="nearest")
            preprocessing_debug["preprocessingMedianApplied"] = True
        analysis_source_note = "legacy_app_raw_grayscale_crop"
    else:
        preprocessing_debug = compute_rbc_iraw01_matlab_like(image, params)
        raw_gray = preprocessing_debug["raw_gray"]
        camera_crop_gray = preprocessing_debug["camera_crop_gray"]
        analysis_gray = preprocessing_debug["analysis_gray"].astype(np.float64)
        crop_bounds = preprocessing_debug["crop_bounds"]
        analysis_source_note = "matlab_Iraw01_processed_grayscale"

    # MATLAB calls this panel "Raw", but it is the analysis grayscale field for
    # gap segmentation. Display-scaled Wz below is only for visual validation;
    # seed/grow thresholding uses the unscaled analysis Wz.
    crop_gray = np.asarray(analysis_gray, dtype=np.float32)

    sigma_ill = float(params.get("sigmaIll", 25.0))
    illumination_field = matlab_gaussian_filter(analysis_gray, sigma_ill)
    residual_raw = analysis_gray - illumination_field
    residual = residual_raw - float(np.median(residual_raw))

    mad = float(np.median(np.abs(residual)))
    wz_mad_floor = float(params.get("wzMadFloor", 0.0) or 0.0)
    wz_smooth_sigma = float(params.get("wzSmoothSigma", 0.0) or 0.0)
    wz_denominator = max(mad, wz_mad_floor) + np.finfo(float).eps
    wz = residual / wz_denominator
    if wz_smooth_sigma > 0:
        wz = matlab_gaussian_filter(wz, wz_smooth_sigma)
    display_wz = robust_display_scale(wz)
    display_residual = exposure.equalize_adapthist(robust_display_scale(residual), clip_limit=0.02)

    track_dark = bool(params.get("trackDark", False))
    k_hi = float(params.get("kHi", 2.2))
    k_lo = float(params.get("kLo", 1.1))
    if k_hi < k_lo:
        k_hi, k_lo = k_lo, k_hi

    if track_dark:
        seed = wz < -k_hi
        grow = wz < -k_lo
    else:
        seed = wz > k_hi
        grow = wz > k_lo
    seed = seed & grow

    if np.any(seed) and np.any(grow):
        reconstructed = morphology.reconstruction(
            seed.astype(np.uint8),
            grow.astype(np.uint8),
            method="dilation",
            footprint=np.ones((3, 3), dtype=np.uint8),
        )
        gap_mask = reconstructed > 0
    else:
        reconstructed = np.zeros_like(grow, dtype=np.uint8)
        gap_mask = np.zeros_like(grow, dtype=bool)

    mask_after_reconstruction = gap_mask.astype(bool)
    mask_after_area_open = _bwareaopen_matlab(mask_after_reconstruction, int(params.get("minAreaPx", 20)))
    gap_mask = mask_after_area_open

    close_radius = int(params.get("closeRadius", 2))
    if close_radius > 0 and np.any(gap_mask):
        gap_mask = morphology.closing(gap_mask, morphology.disk(close_radius))
    mask_after_closing = gap_mask.astype(bool)

    if bool(params.get("doFillHoles", True)) and np.any(gap_mask):
        gap_mask = ndi.binary_fill_holes(gap_mask)
    mask_after_fill = gap_mask.astype(bool)

    if bool(params.get("clearBorder", False)) and np.any(gap_mask):
        gap_mask = clear_border(gap_mask)
    mask_after_clear_border = gap_mask.astype(bool)
    mask_before_large_gap_filter = gap_mask.astype(bool)

    gap_mask, removed_gap_mask, large_gap_diagnostics = filter_large_gap_objects(
        gap_mask,
        enabled=bool(params.get("enableMaxGapAreaFilter", True)),
        max_gap_area_frac=float(params.get("maxGapAreaFrac", 0.10)),
    )

    seed_stats = _mask_component_summary(seed)
    grow_stats = _mask_component_summary(grow)
    reconstructed_stats = _mask_component_summary(mask_after_reconstruction)
    final_stats = _mask_component_summary(gap_mask)
    mean_i = float(np.mean(analysis_gray)) if analysis_gray.size else np.nan
    std_i = float(np.std(analysis_gray)) if analysis_gray.size else np.nan
    debug_report: dict[str, Any] = {
        "pipeline": "MicroStructAI RBC Gap Mode seed/grow",
        "reference": "MATLAB segmentGapsAllFrames equivalent",
        "maskWhitePixelsMeaning": "bright plasma gaps",
        "maskBlackPixelsMeaning": "RBC-rich/non-gap regions",
        "analysisInputSource": input_source,
        "analysisSourceNote": analysis_source_note,
        "displayWzNote": "display_Wz is contrast-stretched for viewing; segmentation thresholds use analysis Wz.",
        "matlabRawPanelNote": "MATLAB Raw panel is processed/analysis grayscale (Iraw01), not necessarily true camera raw.",
        "originalImageSize": list(np.asarray(image).shape[:2]),
        "analysisCropSize": [int(crop_gray.shape[0]), int(crop_gray.shape[1])],
        **crop_bounds,
        "meanI": mean_i,
        "stdI": std_i,
        "rawContrast": (std_i / mean_i) if np.isfinite(mean_i) and abs(mean_i) > np.finfo(float).eps else np.nan,
        "sigmaIll": sigma_ill,
        "madW": mad,
        "wzMadFloor": wz_mad_floor,
        "wzDenominator": wz_denominator,
        "wzSmoothSigma": wz_smooth_sigma,
        "kHi": float(params.get("kHi", 2.2)),
        "kLo": float(params.get("kLo", 1.1)),
        "adaptiveEnabled": False,
        "adaptiveTriggered": False,
        "kHiUse": k_hi,
        "kLoUse": k_lo,
        "trackDark": track_dark,
        "minAreaPx": int(params.get("minAreaPx", 20)),
        "closeRadius": close_radius,
        "doFillHoles": bool(params.get("doFillHoles", True)),
        "clearBorder": bool(params.get("clearBorder", False)),
        "maxGapAreaFilterEnabled": bool(params.get("enableMaxGapAreaFilter", True)),
        "maxGapAreaFrac": float(params.get("maxGapAreaFrac", 0.10)),
        "gapAreaFrac": final_stats["areaFrac"],
        "gapNtotal": final_stats["componentCount"],
        "largestGapFrac": final_stats["largestFrac"],
        "numberOfComponents": final_stats["componentCount"],
        "seedAreaFrac": seed_stats["areaFrac"],
        "growAreaFrac": grow_stats["areaFrac"],
        "reconstructedAreaFrac": reconstructed_stats["areaFrac"],
        "reconstructedComponents": reconstructed_stats["componentCount"],
        "maskAfterAreaOpenAreaFrac": _mask_component_summary(mask_after_area_open)["areaFrac"],
        "maskAfterClosingAreaFrac": _mask_component_summary(mask_after_closing)["areaFrac"],
        "maskBeforeLargeGapFilterAreaFrac": _mask_component_summary(mask_before_large_gap_filter)["areaFrac"],
        **{f"largeGap_{key}": value for key, value in large_gap_diagnostics.items()},
        "ximg_bgSigma": preprocessing_debug.get("ximg_bgSigma", np.nan),
        "ximg_pLow": preprocessing_debug.get("ximg_pLow", np.nan),
        "ximg_pHigh": preprocessing_debug.get("ximg_pHigh", np.nan),
        "ximg_p1": preprocessing_debug.get("ximg_p1", np.nan),
        "ximg_p99": preprocessing_debug.get("ximg_p99", np.nan),
        "preprocessingMedianApplied": preprocessing_debug.get("preprocessingMedianApplied", False),
        "preprocessingMedianK": preprocessing_debug.get("preprocessingMedianK", _odd_kernel(int(params.get("medianK", 3)))),
    }

    debug_stages: dict[str, np.ndarray] = {}
    if include_debug_stages:
        debug_stages = {
            "01_analysis_grayscale_crop": crop_gray.astype(np.float32),
            "02_illumination_background_field": illumination_field.astype(np.float32),
            "03_residual_Wraw_I_minus_Ill": residual_raw.astype(np.float32),
            "04_clean_residual_W_minus_median": residual.astype(np.float32),
            "05_Wz_used_for_segmentation": wz.astype(np.float32),
            "06_seed_mask": seed.astype(bool),
            "07_grow_mask": grow.astype(bool),
            "08_reconstructed_mask_before_cleanup": mask_after_reconstruction.astype(bool),
            "09_mask_after_bwareaopen": mask_after_area_open.astype(bool),
            "10_mask_after_closing": mask_after_closing.astype(bool),
            "11_mask_after_fill_clearborder_before_large_filter": mask_before_large_gap_filter.astype(bool),
            "12_final_gap_mask": gap_mask.astype(bool),
            "13_camera_grayscale_crop_for_reference": np.asarray(camera_crop_gray, dtype=np.float32),
        }
        if "preprocessing_background" in preprocessing_debug:
            debug_stages["00_matlab_Iraw_preprocessing_background"] = preprocessing_debug["preprocessing_background"]
            debug_stages["00_matlab_Iraw_preprocessing_residual"] = preprocessing_debug["preprocessing_residual"]

    return {
        "raw_gray": raw_gray,
        "crop_gray": crop_gray,
        "camera_crop_gray": np.asarray(camera_crop_gray, dtype=np.float32),
        "analysis_input_source": input_source,
        "illumination_field": illumination_field.astype(np.float32),
        "Wraw": residual_raw.astype(np.float32),
        "W": residual.astype(np.float32),
        "Wz": wz.astype(np.float32),
        "display_Wz": display_wz,
        "clean_residual": residual.astype(np.float32),
        "display_residual": display_residual.astype(np.float32),
        "seed_mask": seed.astype(bool),
        "grow_mask": grow.astype(bool),
        "reconstructed_mask": mask_after_reconstruction.astype(bool),
        "mask_after_area_open": mask_after_area_open.astype(bool),
        "mask_after_closing": mask_after_closing.astype(bool),
        "mask_before_large_gap_filter": mask_before_large_gap_filter.astype(bool),
        "gap_mask": gap_mask.astype(bool),
        "removed_gap_mask": removed_gap_mask,
        "large_gap_diagnostics": large_gap_diagnostics,
        "debug_report": debug_report,
        "debug_stages": debug_stages,
        # Backward-compatible aliases used by older app code/tests.
        "gray": crop_gray,
        "wz": wz.astype(np.float32),
        "mask": gap_mask.astype(bool),
    }


def _enhanced_structure_candidate(wz: np.ndarray, track_dark: bool, params: dict[str, Any]) -> np.ndarray:
    """Build one bright or dark enhanced generic seed/grow mask candidate."""
    k_hi = float(params.get("objectKHi", 2.2))
    k_lo = float(params.get("objectKLo", 1.1))
    if k_hi < k_lo:
        k_hi, k_lo = k_lo, k_hi
    if track_dark:
        seed = wz < -k_hi
        grow = wz < -k_lo
    else:
        seed = wz > k_hi
        grow = wz > k_lo
    seed = seed & grow

    if np.any(seed) and np.any(grow):
        reconstructed = morphology.reconstruction(seed.astype(np.uint8), grow.astype(np.uint8), method="dilation")
        mask = reconstructed > 0
    else:
        mask = np.zeros_like(grow, dtype=bool)

    mask = _remove_small_objects(mask, int(params.get("objectMinAreaPx", 20)))
    close_radius = int(params.get("objectCloseRadius", 2))
    if close_radius > 0 and np.any(mask):
        mask = morphology.closing(mask, morphology.disk(close_radius))
    if bool(params.get("objectFillHoles", True)) and np.any(mask):
        mask = ndi.binary_fill_holes(mask)
    if bool(params.get("objectClearBorder", False)) and np.any(mask):
        mask = clear_border(mask)
    return np.asarray(mask, dtype=bool)


def _enhanced_structure_candidate_matlab_like(
    wz: np.ndarray,
    track_dark: bool,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Build an enhanced generic candidate using MATLAB/RBC-style cleanup.

    This helper deliberately keeps generic Object Mode language. It borrows
    the robust seed/grow mechanics and 8-connected cleanup used for RBC mask
    parity, but the output remains a generic structure/object/gap mask.
    """
    k_hi = float(params.get("objectKHi", 2.2))
    k_lo = float(params.get("objectKLo", 1.1))
    if k_hi < k_lo:
        k_hi, k_lo = k_lo, k_hi
    if track_dark:
        seed = wz < -k_hi
        grow = wz < -k_lo
    else:
        seed = wz > k_hi
        grow = wz > k_lo
    seed = seed & grow

    if np.any(seed) and np.any(grow):
        reconstructed = morphology.reconstruction(
            seed.astype(np.uint8),
            grow.astype(np.uint8),
            method="dilation",
            footprint=np.ones((3, 3), dtype=np.uint8),
        )
        mask_after_reconstruction = reconstructed > 0
    else:
        reconstructed = np.zeros_like(grow, dtype=np.uint8)
        mask_after_reconstruction = np.zeros_like(grow, dtype=bool)

    mask_after_area_open = _bwareaopen_matlab(mask_after_reconstruction, int(params.get("objectMinAreaPx", 20)))
    mask = mask_after_area_open.astype(bool)
    close_radius = int(params.get("objectCloseRadius", 2))
    if close_radius > 0 and np.any(mask):
        mask = morphology.closing(mask, morphology.disk(close_radius))
    mask_after_closing = mask.astype(bool)

    if bool(params.get("objectFillHoles", True)) and np.any(mask):
        mask = ndi.binary_fill_holes(mask)
    mask_after_fill = mask.astype(bool)

    if bool(params.get("objectClearBorder", False)) and np.any(mask):
        mask = clear_border(mask)
    final_mask = mask.astype(bool)

    stats = {
        "objectKHiUse": k_hi,
        "objectKLoUse": k_lo,
        "objectSeedAreaFrac": float(np.mean(seed)) if seed.size else np.nan,
        "objectGrowAreaFrac": float(np.mean(grow)) if grow.size else np.nan,
        "objectReconstructedAreaFrac": float(np.mean(mask_after_reconstruction)) if mask_after_reconstruction.size else np.nan,
        "objectMaskAfterAreaOpenAreaFrac": float(np.mean(mask_after_area_open)) if mask_after_area_open.size else np.nan,
        "objectMaskAfterClosingAreaFrac": float(np.mean(mask_after_closing)) if mask_after_closing.size else np.nan,
        "objectMaskAfterFillAreaFrac": float(np.mean(mask_after_fill)) if mask_after_fill.size else np.nan,
    }
    stages = {
        "06_seed_mask": seed.astype(bool),
        "07_grow_mask": grow.astype(bool),
        "08_reconstructed_mask_before_cleanup": mask_after_reconstruction.astype(bool),
        "09_mask_after_bwareaopen": mask_after_area_open.astype(bool),
        "10_mask_after_closing": mask_after_closing.astype(bool),
        "11_mask_after_fill_clearborder": final_mask.astype(bool),
    }
    return final_mask, stages, stats


def _enhanced_candidate_score(mask: np.ndarray, wz: np.ndarray) -> float:
    """Return a simple plausibility score for automatic generic polarity."""
    arr = np.asarray(mask, dtype=bool)
    area_fraction = float(np.mean(arr)) if arr.size else 0.0
    if area_fraction <= 1e-5 or area_fraction >= 0.95:
        return -np.inf
    _labels, n_components = ndi.label(arr, structure=np.ones((3, 3), dtype=np.uint8))
    contrast_support = float(np.mean(np.abs(wz[arr]))) if np.any(arr) else 0.0
    extreme_penalty = max(0.0, area_fraction - 0.70) * 10.0
    return contrast_support + min(float(n_components), 100.0) / 100.0 - extreme_penalty


def segment_enhanced_structures_seedgrow(image: np.ndarray, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Segment generic coarse structures with robust seed/grow reconstruction.

    This Object Mode helper adapts the robust preprocessing philosophy used by
    RBC Gap Mode without assigning RBC meaning. White/True pixels represent a
    generic detected structure, gap, pore, object, or coarse pattern.
    """
    params = dict(params or {})
    include_debug_stages = bool(params.get("include_debug_stages", False))
    preprocessing_style = str(params.get("enhancedObjectPreprocessingStyle", "Current generic enhanced mode"))
    matlab_style = preprocessing_style == "MATLAB/RBC-style robust residual mode"
    debug_stages: dict[str, np.ndarray] = {}
    preprocessing_debug: dict[str, Any] = {}

    if matlab_style:
        iraw_params = dict(params)
        iraw_params["cropFrac"] = float(params.get("objectCropFrac", 1.0))
        iraw_params["useMedian"] = bool(params.get("objectUseMedian", True))
        iraw_params["medianK"] = int(params.get("objectMedianK", 3))
        iraw_params["ximg_bgSigma"] = float(params.get("objectPreprocessBgSigma", params.get("ximg_bgSigma", 40.0)))
        preprocessing_debug = compute_rbc_iraw01_matlab_like(image, iraw_params)
        raw_gray = np.asarray(preprocessing_debug["analysis_gray"], dtype=np.float32)
        work = raw_gray.astype(np.float64)
        sigma_ill = float(params.get("objectSigmaIll", 25.0))
        illumination_field = matlab_gaussian_filter(work, sigma_ill)
        residual_raw = work - illumination_field
        residual = residual_raw - float(np.median(residual_raw))
        mad = float(np.median(np.abs(residual)))
        wz_denominator = mad + np.finfo(float).eps
        wz = residual / wz_denominator
        analysis_input_source = "MATLAB/RBC-style processed residual field"
        if include_debug_stages:
            debug_stages.update(
                {
                    "01_analysis_grayscale": raw_gray.astype(np.float32),
                    "02_illumination_background_field": illumination_field.astype(np.float32),
                    "03_residual_Wraw_I_minus_Ill": residual_raw.astype(np.float32),
                    "04_clean_residual_W_minus_median": residual.astype(np.float32),
                    "05_Wz_used_for_segmentation": wz.astype(np.float32),
                    "00_camera_grayscale_crop_for_reference": np.asarray(preprocessing_debug.get("camera_crop_gray", raw_gray), dtype=np.float32),
                }
            )
            if "preprocessing_background" in preprocessing_debug:
                debug_stages["00_processed_input_background"] = preprocessing_debug["preprocessing_background"]
                debug_stages["00_processed_input_residual"] = preprocessing_debug["preprocessing_residual"]
    else:
        raw_gray = to_grayscale_float01(image)
        work = raw_gray.copy()
        if bool(params.get("objectUseMedian", True)):
            work = ndi.median_filter(work, size=_odd_kernel(int(params.get("objectMedianK", 3))))

        sigma_ill = float(params.get("objectSigmaIll", 25.0))
        illumination_field = filters.gaussian(work, sigma=sigma_ill, preserve_range=True)
        residual_raw = work - illumination_field
        residual = residual_raw - float(np.median(residual_raw))
        mad = float(np.median(np.abs(residual)))
        wz_denominator = mad + 1e-6
        wz = residual / wz_denominator
        analysis_input_source = "current generic enhanced grayscale residual"
        if include_debug_stages:
            debug_stages.update(
                {
                    "01_analysis_grayscale": raw_gray.astype(np.float32),
                    "02_illumination_background_field": illumination_field.astype(np.float32),
                    "03_residual_Wraw_I_minus_Ill": residual_raw.astype(np.float32),
                    "04_clean_residual_W_minus_median": residual.astype(np.float32),
                    "05_Wz_used_for_segmentation": wz.astype(np.float32),
                }
            )
    display_wz = robust_display_scale(wz)
    display_residual = exposure.equalize_adapthist(robust_display_scale(residual), clip_limit=0.02)

    requested_target = str(params.get("objectSegmentationTarget", "Bright structures / bright gaps"))
    chosen_stats: dict[str, Any] = {}
    chosen_stages: dict[str, np.ndarray] = {}
    if requested_target == "Dark structures / dark gaps":
        chosen_target = requested_target
        if matlab_style:
            structure_mask, chosen_stages, chosen_stats = _enhanced_structure_candidate_matlab_like(wz, track_dark=True, params=params)
        else:
            structure_mask = _enhanced_structure_candidate(wz, track_dark=True, params=params)
    elif requested_target == "Auto polarity":
        if matlab_style:
            bright_mask, bright_stages, bright_stats = _enhanced_structure_candidate_matlab_like(wz, track_dark=False, params=params)
            dark_mask, dark_stages, dark_stats = _enhanced_structure_candidate_matlab_like(wz, track_dark=True, params=params)
        else:
            bright_mask = _enhanced_structure_candidate(wz, track_dark=False, params=params)
            dark_mask = _enhanced_structure_candidate(wz, track_dark=True, params=params)
            bright_stages, dark_stages = {}, {}
            bright_stats, dark_stats = {}, {}
        if _enhanced_candidate_score(dark_mask, wz) > _enhanced_candidate_score(bright_mask, wz):
            chosen_target = "Dark structures / dark gaps"
            structure_mask = dark_mask
            chosen_stages, chosen_stats = dark_stages, dark_stats
        else:
            chosen_target = "Bright structures / bright gaps"
            structure_mask = bright_mask
            chosen_stages, chosen_stats = bright_stages, bright_stats
    else:
        chosen_target = "Bright structures / bright gaps"
        if matlab_style:
            structure_mask, chosen_stages, chosen_stats = _enhanced_structure_candidate_matlab_like(wz, track_dark=False, params=params)
        else:
            structure_mask = _enhanced_structure_candidate(wz, track_dark=False, params=params)

    if include_debug_stages:
        debug_stages.update(chosen_stages)
        debug_stages["12_final_generic_structure_mask_before_large_filter"] = structure_mask.astype(bool)

    mask_before_large_filter = structure_mask.astype(bool)
    structure_mask, removed_structure_mask, diagnostics = filter_large_structure_objects(
        structure_mask,
        enabled=bool(params.get("enableMaxObjectAreaFilter", False)),
        max_object_area_frac=float(params.get("maxObjectAreaFrac", 0.35)),
    )
    if include_debug_stages:
        debug_stages["13_final_generic_structure_mask"] = structure_mask.astype(bool)
        if np.any(removed_structure_mask):
            debug_stages["14_removed_large_generic_objects"] = removed_structure_mask.astype(bool)

    diagnostics.update(
        {
            "enhancedObjectModeEnabled": True,
            "enhancedObjectPreprocessingStyle": preprocessing_style,
            "objectAnalysisInputSource": analysis_input_source,
            "objectDisplayWzNote": "display_Wz is contrast-stretched for viewing; segmentation thresholds use analysis Wz.",
            "objectMaskMeaning": "generic detected structures/gaps/objects, not RBC plasma gaps",
            "objectSegmentationTarget": requested_target,
            "objectSegmentationTargetUsed": chosen_target,
            "objectSigmaIll": sigma_ill,
            "objectMadW": mad,
            "objectWzDenominator": wz_denominator,
            "objectAnalysisCropFrac": float(params.get("objectCropFrac", 1.0)) if matlab_style else 1.0,
            "objectAnalysisCropSize": f"{int(raw_gray.shape[0])}x{int(raw_gray.shape[1])}",
            "objectMaskBeforeLargeFilterAreaFrac": float(np.mean(mask_before_large_filter)) if mask_before_large_filter.size else np.nan,
        }
    )
    diagnostics.update(chosen_stats)
    if preprocessing_debug:
        diagnostics.update(
            {
                "objectPreprocessBgSigma": preprocessing_debug.get("ximg_bgSigma", np.nan),
                "objectPreprocessPercentileLow": preprocessing_debug.get("ximg_pLow", np.nan),
                "objectPreprocessPercentileHigh": preprocessing_debug.get("ximg_pHigh", np.nan),
                "objectPreprocessP1": preprocessing_debug.get("ximg_p1", np.nan),
                "objectPreprocessP99": preprocessing_debug.get("ximg_p99", np.nan),
                "objectPreprocessingMedianApplied": preprocessing_debug.get("preprocessingMedianApplied", False),
                "objectPreprocessingMedianK": preprocessing_debug.get("preprocessingMedianK", _odd_kernel(int(params.get("objectMedianK", 3)))),
            }
        )
    return {
        "raw_gray": raw_gray,
        "illumination_field": illumination_field.astype(np.float32),
        "Wraw": residual_raw.astype(np.float32),
        "W": residual.astype(np.float32),
        "Wz": wz.astype(np.float32),
        "display_Wz": display_wz,
        "display_residual": display_residual.astype(np.float32),
        "structure_mask": structure_mask.astype(bool),
        "removed_structure_mask": removed_structure_mask.astype(bool),
        "structure_diagnostics": diagnostics,
        "debug_report": diagnostics,
        "debug_stages": debug_stages,
        "mask": structure_mask.astype(bool),
    }


def segment_bright_objects(image: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
    """Segment bright cells, particles, or objects from a general image."""
    params = dict(params or {})
    params["polarity"] = "bright"
    gray = to_grayscale(image)
    work = _preprocess(gray, params)
    mask = _threshold_image(work, params)
    return _postprocess_mask(mask, params)


def segment_dark_objects(image: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
    """Segment dark cells, particles, or objects from a general image."""
    params = dict(params or {})
    params["polarity"] = "dark"
    gray = to_grayscale(image)
    work = _preprocess(gray, params)
    mask = _threshold_image(work, params)
    return _postprocess_mask(mask, params)


def fill_outline_mask(outline_mask: np.ndarray) -> np.ndarray:
    """Fill closed binary outlines into object masks."""
    outline = np.asarray(outline_mask, dtype=bool)
    if outline.size == 0:
        return outline
    closed = morphology.closing(outline, morphology.disk(1))
    filled = ndi.binary_fill_holes(closed)
    return filled.astype(bool)


def load_ground_truth_mask(
    mask_image: np.ndarray,
    mask_type: str = "Outline mask",
    outline_polarity: str = "dark outlines",
    min_object_size: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a filled ground-truth mask from a binary mask or outline image.

    Returns ``(input_mask, filled_mask)``. For outline masks, ``input_mask`` is
    the detected outline and ``filled_mask`` is the hole-filled object mask.
    """
    gray = to_grayscale_float01(mask_image)
    try:
        threshold = filters.threshold_otsu(gray)
    except ValueError:
        threshold = 0.5

    if mask_type == "Outline mask":
        if outline_polarity == "bright outlines":
            input_mask = gray >= threshold
        else:
            input_mask = gray <= threshold
        filled_mask = fill_outline_mask(input_mask)
    else:
        border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        background_is_bright = float(np.median(border_pixels)) >= 0.5
        input_mask = gray <= threshold if background_is_bright else gray >= threshold
        filled_mask = input_mask

    input_mask = np.asarray(input_mask, dtype=bool)
    filled_mask = _remove_small_objects(np.asarray(filled_mask, dtype=bool), int(min_object_size))
    return input_mask, filled_mask.astype(bool)


def _filename_number_key(uploaded_file: Any) -> str | None:
    """Extract a stable number key from an uploaded filename."""
    name = getattr(uploaded_file, "name", str(uploaded_file))
    numbers = re.findall(r"\d+", name)
    return numbers[-1].lstrip("0") or "0" if numbers else None


def match_image_mask_files(image_files: list, mask_files: list, strategy: str = "sorted order") -> tuple[list[tuple[Any, Any]], list[str]]:
    """Match uploaded image files to mask files by sorted order or filename numbers."""
    warnings: list[str] = []
    images = sorted(image_files, key=natural_sort_key)
    masks = sorted(mask_files, key=natural_sort_key)

    if len(images) != len(masks):
        warnings.append(f"Number of images ({len(images)}) and masks ({len(masks)}) is different.")

    if strategy == "match by filename numbers":
        mask_by_number: dict[str, Any] = {}
        for mask_file in masks:
            key = _filename_number_key(mask_file)
            if key is not None and key not in mask_by_number:
                mask_by_number[key] = mask_file

        pairs: list[tuple[Any, Any]] = []
        for image_file in images:
            key = _filename_number_key(image_file)
            if key is None or key not in mask_by_number:
                warnings.append(f"No matching mask found for {getattr(image_file, 'name', 'image')}.")
                continue
            pairs.append((image_file, mask_by_number[key]))
        return pairs, warnings

    pair_count = min(len(images), len(masks))
    return list(zip(images[:pair_count], masks[:pair_count])), warnings


def resize_mask_to_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a mask to an image shape using nearest-neighbor interpolation."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == shape:
        return mask
    resized = transform.resize(
        mask.astype(np.uint8),
        shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return resized >= 0.5


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color_rgb: tuple[float, float, float] = (1.0, 0.1, 0.1),
    alpha: float = 0.4,
) -> np.ndarray:
    """Return an RGB overlay where mask pixels are tinted with ``color_rgb``."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        rgb = np.dstack([normalize_image(arr)] * 3)
    elif arr.ndim == 3:
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        rgb = util.img_as_float32(arr)
        if rgb.max() > 1:
            rgb = rgb / 255.0
    else:
        raise ValueError("Expected a 2D or RGB/RGBA image for overlay.")

    mask = resize_mask_to_shape(mask, rgb.shape[:2])
    overlay = rgb.copy()
    tint = np.asarray(color_rgb, dtype=np.float32)
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * tint
    return np.clip(overlay, 0, 1)


def create_boundary_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    linewidth: int = 1,
) -> np.ndarray:
    """Draw mask boundaries over a grayscale image in MATLAB visboundaries style."""
    gray = to_grayscale_float01(image)
    mask = resize_mask_to_shape(mask, gray.shape)
    boundaries = find_boundaries(mask, mode="outer")

    width = max(1, int(linewidth))
    if width > 1:
        boundaries = morphology.dilation(boundaries, morphology.disk(width - 1))

    rgb = np.dstack([gray, gray, gray]).astype(np.float32)
    color_arr = np.asarray(color, dtype=np.float32)
    if np.max(color_arr) > 1.0:
        color_arr = color_arr / 255.0
    rgb[boundaries] = np.clip(color_arr, 0.0, 1.0)
    return np.clip(rgb, 0.0, 1.0)
