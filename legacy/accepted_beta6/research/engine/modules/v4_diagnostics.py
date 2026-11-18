"""Transparent, observable heuristics. No truth-mask/Dice fitting or hidden score."""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure

DIAGNOSTICS_VERSION = "4.0-observable-1"
# Descriptive guardrails, not estimated biological acceptance limits.
THRESHOLDS = {"sparse_fraction": .01, "dense_fraction": .90, "border_occupancy": .50,
              "dominant_foreground_fraction": .90, "extreme_components_per_megapixel": 10000,
              "tiny_component_fraction": .50, "tiny_area_px2": 4, "saturation_fraction": .05,
              "low_dynamic_fraction": .01, "low_normalized_laplacian_variance": 1e-4}


def display_table(table):
    """Arrow-safe display only; exported diagnostic values remain typed."""
    view = table.copy()
    if "value" in view:
        view["value"] = view["value"].map(lambda value: "Unassessed" if value is None else str(value))
    return view


def row(category, name, value, status, reason):
    return {"category": category, "diagnostic": name, "value": value, "status": status, "reason": reason,
            "diagnostics_version": DIAGNOSTICS_VERSION}


def image_diagnostics(image, metadata=None):
    category = "IMAGE QUALITY"
    if image is None:
        return [row(category, "image_available", None, "CAUTION", "Mask-only input: image quality is unassessed, NOT passed.")]
    a = np.asarray(image)
    if not np.isfinite(a).all():
        return [row(category, "finite_pixels", False, "FAIL", "Nonfinite pixels prevent reliable intensity diagnostics.")]
    gray = a.astype(float)
    if gray.ndim == 3:
        gray = gray[..., :3].mean(axis=2)
    if a.dtype.kind in "ui":
        limits = np.iinfo(a.dtype)
        lo, hi = float(limits.min), float(limits.max)
    elif a.dtype == bool:
        lo, hi = 0., 1.
    else:
        lo, hi = float(gray.min()), float(gray.max())
    span = max(hi-lo, np.finfo(float).eps)
    q1, q99 = np.percentile(gray, [1, 99])
    dynamic = float((q99-q1)/span)
    saturation = float(np.mean((gray == lo) | (gray == hi))) if a.dtype.kind in "bui" else None
    norm = (gray-gray.min())/max(float(np.ptp(gray)), 1e-12)
    focus = float(np.var(ndi.laplace(norm)))
    rows = [row(category, "intensity_min_max", [float(gray.min()), float(gray.max())], "PASS", "Observed values; PASS is not segmentation validation."),
            row(category, "robust_dynamic_fraction", dynamic, "CAUTION" if dynamic < THRESHOLDS["low_dynamic_fraction"] else "PASS", "1st–99th percentile range / dtype range; <0.01 cautions. Low exposure is not proof of bad biology."),
            row(category, "normalized_laplacian_variance", focus, "CAUTION" if focus < THRESHOLDS["low_normalized_laplacian_variance"] else "PASS", "Focus proxy after min/max normalization; <1e-4 cautions. Noise can inflate it; not a calibrated blur detector."),
            row(category, "dtype_endpoint_fraction", saturation, "CAUTION" if saturation is None or saturation > THRESHOLDS["saturation_fraction"] else "PASS", "Pixels at dtype endpoints; >5% cautions, not proof of clipping. Float saturation cannot be inferred without sensor range.")]
    if metadata:
        rows.append(row(category, "ingestion_conversion", metadata.get("applied_conversion"), "PASS" if metadata.get("resulting_dtype") == metadata.get("original_dtype") else "CAUTION", "Conversion is recorded explicitly. No undetected sensor clipping can be ruled out."))
    return rows


def mask_diagnostics(mask):
    a = np.asarray(mask, dtype=bool)
    labels = measure.label(a, connectivity=2)
    sizes = np.bincount(labels.ravel())[1:]
    area = float(a.mean())
    border = np.zeros(a.shape, bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    border_fraction = float(a[border].mean())
    dominant = float(sizes.max()/sizes.sum()) if sizes.size else 0.
    tiny = float((sizes < THRESHOLDS["tiny_area_px2"]).mean()) if sizes.size else 0.
    density = float(len(sizes) * 1e6 / a.size)
    cat = "MASK PLAUSIBILITY"
    rows = [row(cat, "foreground_fraction", area, "CAUTION" if area < .01 or area > .90 else "PASS", "<0.01 or >0.90 occupancy cautions. Empty/full masks retain mathematically defined raw results."),
            row(cat, "ROI_border_occupancy", border_fraction, "CAUTION" if border_fraction > .50 else "PASS", "Fraction of ROI boundary pixels in foreground; >0.50 may reflect cropping/contact, not necessarily segmentation failure."),
            row(cat, "dominant_component_foreground_fraction", dominant, "CAUTION" if dominant > .90 else "PASS", ">0.90 of all foreground in one component. Expected in some network/single-object domains."),
            row(cat, "components_per_megapixel", density, "CAUTION" if density > 10000 else "PASS", ">10000 suggests fragmentation or naturally numerous small objects; not a biological rejection rule."),
            row(cat, "tiny_component_fraction", tiny, "CAUTION" if tiny > .50 else "PASS", ">50% of components have <4 pixels². Scale-dependent; no objects are removed by this diagnostic.")]
    if min(a.shape) < 16 or a.size < 256:
        rows.append(row(cat, "small_ROI", list(a.shape), "CAUTION", "Very small ROI; 16-pixel lacunarity may be unavailable and topology is strongly boundary-sensitive."))
    return rows


def temporal_diagnostics(n, *, confirmed=False, interval=None, reference_available=False, ordering_warnings=()):
    cat = "TEMPORAL / REFERENCE ELIGIBILITY"
    rows = [row(cat, "frame_count", n, "PASS" if n >= 2 else "CAUTION", "At least two consistent frames needed for change descriptors; no temporal conclusion from one mask."),
            row(cat, "sequence_order_confirmed", bool(confirmed), "PASS" if confirmed else "CAUTION", "User confirmation, not independent verification of acquisition order or motion registration."),
            row(cat, "time_interval_s", interval, "PASS" if confirmed and interval is not None else "CAUTION", "Seconds available only with explicit positive interval and timing confirmation. Otherwise frame-pair units only."),
            row(cat, "RBC_reference_eligibility", reference_available, "PASS" if reference_available else "CAUTION", "Generic/supplied masks do not define an RBC score reference. Legacy RBC gates remain separate and unchanged.")]
    rows += [row(cat, "ordering_warning", warning, "CAUTION", "No missing frames synthesized; confirm actual temporal spacing.") for warning in ordering_warnings]
    return rows
