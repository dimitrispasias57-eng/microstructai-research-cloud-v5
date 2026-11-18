"""Read-only input/interpretation safeguards. No segmentation or score changes."""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .io_utils import file_display_name, natural_sort_key
from .qc_plotting import available_metrics, bool_values, metric_view, view_counts
from .rbc_scores import compute_official_mask_scores

DEV_VERSION = "4.0.0-dev.2"
PARENT_COMMIT = "b5cc7b46916f06ded98a8cd19272b4370d2256ab"
LIMITATION = (
    "Exploratory image-structure descriptors, not validated biomarkers or RheoScan-equivalent indices. "
    "Passing computational QC does not establish that a detected bright region is a biological plasma gap. "
    "Image/overlay review is an internal check, not independent validation."
)


def settings_digest(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()


def input_selection_signature(files):
    """Cheap rerun identity; byte-level hashes are captured by the run preflight."""
    items = []
    for item in files:
        source = getattr(item, "source", item)
        name = file_display_name(item)
        if hasattr(source, "metadata") and hasattr(source, "pixels"):
            marker = [source.metadata["input_sha256"], source.metadata["page_1based"]]
        elif isinstance(source, (str, Path)):
            try:
                stat = Path(source).stat()
                marker = [str(Path(source).resolve()), stat.st_size, stat.st_mtime_ns]
            except OSError:
                marker = [str(source), "unavailable"]
        elif getattr(source, "file_id", None) is not None:
            marker = [source.file_id, getattr(source, "size", None)]
        else:
            marker = [hashlib.sha256(source.getvalue()).hexdigest()]
        items.append([name, marker])
    return settings_digest(items)


def manual_roi(params, width, height):
    """Strict 1-based x,y,width,height, matching the unchanged V3 crop engine."""
    if not params.get("useCrop", True) or params.get("cropMode", "center") != "manual":
        return None
    try:
        values = [float(x.strip()) for x in str(params.get("manualCropRect", "")).replace(";", ",").split(",")]
        if len(values) != 4 or not all(math.isfinite(x) and x.is_integer() and x >= 1 for x in values):
            raise ValueError
        x, y, w, h = map(int, values)
        if x + w - 1 > width or y + h - 1 > height:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise ValueError(f"Manual ROI must be four positive whole numbers x,y,width,height inside {width}×{height}; coordinates start at 1. No clipping or fallback crop was applied.") from None
    return x, y, w, h


def inspect_inputs(files, params=None):
    """Decode through V4 lossless ingress and record orientation without changes.

    Multi-page files require explicit expansion before this compatibility check.
    No global upload cache or persistent copy of research images is used.
    """
    params = params or {}
    records, errors, warnings = [], [], []
    names = [file_display_name(f) for f in files]
    if not files:
        errors.append("Select at least one image.")
    if len(set(n.casefold() for n in names)) != len(names):
        errors.append("Duplicate filenames: use unique names so results can be matched to source frames.")
    if names != [file_display_name(f) for f in sorted(files, key=natural_sort_key)]:
        warnings.append("Selected order is not natural filename order. Confirm acquisition order; this check does not reorder files.")
    numbered = [re.match(r"^(.*?)(\d+)$", Path(n).stem) for n in names]
    if len(names) > 1 and all(numbered) and len({m.group(1) for m in numbered}) == 1:
        ids = [int(m.group(2)) for m in numbered]
        if any(b - a != 1 for a, b in zip(ids, ids[1:])):
            warnings.append("Filename frame numbers are non-consecutive or duplicated. The uniform-FPS time axis may not represent elapsed acquisition time; no missing frames were synthesized.")
    elif len(names) > 1:
        warnings.append("Acquisition order/continuity cannot be verified from filenames alone.")
    for position, item in enumerate(files, 1):
        name = file_display_name(item)
        source = getattr(item, "source", item)
        record = {"position": position, "filename": name, "readable": False}
        try:
            from .v4_ingestion import load_frames
            frame = load_frames(item)[0]
            record.update(frame.metadata)
            warnings.extend(f"{name}: {w}" for w in frame.metadata["warnings"])
            rect = manual_roi(params, frame.pixels.shape[1], frame.pixels.shape[0])
            if rect is not None:
                if min(rect[2:]) < 2:
                    raise ValueError("ROI dimensions must be at least 2 pixels.")
                if min(rect[2:]) < 16 or rect[2]*rect[3] < 256:
                    warnings.append(f"{name}: extremely small ROI; 16-pixel lacunarity may be unavailable and boundary effects dominate.")
        except (OSError, ValueError, TypeError, Image.DecompressionBombError) as exc:
            record["error"] = str(exc)
            errors.append(f"{name}: {exc}")
        records.append(record)
    dimensions = {(r["width"], r["height"]) for r in records if r["readable"]}
    if len(dimensions) > 1:
        errors.append("Mixed image dimensions/orientations in this sample: " + ", ".join(f"{w}×{h}" for w, h in sorted(dimensions)) + ". Split into consistent recordings; no resize, rotation or frame deletion was applied.")
    warnings.append("Dimensions cannot detect every rotated field of view. Check physical channel orientation and mask overlays yourself.")
    return {"ok": not errors, "records": records, "errors": errors, "warnings": list(dict.fromkeys(warnings)),
            "selection_signature": input_selection_signature(files)}


def require_inputs(files, params):
    result = inspect_inputs(files, params)
    if not result["ok"]:
        raise ValueError("Input check failed: " + " | ".join(result["errors"]))
    return result


def attach_context(result, preflight, run_settings, acquisition=None, roi_png=None):
    """Store immutable run context outside engine params; scores remain untouched."""
    result["development_context"] = {
        "development_version": DEV_VERSION, "source_v3_commit": PARENT_COMMIT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_settings": json.loads(json.dumps(run_settings, default=str)),
        "input_settings_sha256": settings_digest(run_settings),
        "input_check": preflight, "acquisition": dict(acquisition or {}),
        "interpretation_limit": LIMITATION,
    }
    root = Path(__file__).resolve().parents[1]
    code_paths = [root / "app.py", root / "requirements.txt", *sorted((root / "modules").glob("*.py"))]
    result["development_context"]["executed_source_sha256"] = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths
    }
    if roi_png is not None:
        result["development_roi_png"] = roi_png
    return result


def interpretation_notes(analysis):
    df = analysis.get("metrics", pd.DataFrame())
    ctx = analysis.get("development_context", {})
    acquisition = ctx.get("acquisition", {})
    params = analysis.get("params", {})
    notes = [LIMITATION]
    if len(df) == 1:
        notes.append("SINGLE IMAGE: report raw spatial descriptors only. A sequence-relative score from one image has no temporal aggregation interpretation, even if the legacy engine returns a finite number.")
    elif analysis.get("mode") == "RBC Gap Mode":
        notes.append("SEQUENCE: SASI and AggIndex depend on segmentation, QC, and normalization/reference membership. A value of 50 is not a clinical midpoint; higher is not automatically healthier or more aggregated.")
    fixed = params.get("rbcScoreReferenceStats")
    if fixed:
        notes.append("A fixed reference was supplied for mask SASI/GapScore. This is not external biological validation; AggIndex also includes other sequence-normalized components. Cross-recording comparability is not established.")
    else:
        notes.append("No locked external score reference is configured. Do not rank different recordings/donors by these sequence-normalized composite scores as if they shared an absolute scale.")
    fps = params.get("matlab_fps", 1.0)
    notes.append(f"Stored engine FPS: {fps}. " + ("User confirms uniform acquisition timing for the selected frames; not independently verified." if acquisition.get("timing_confirmed") else "Timing is UNCONFIRMED. Existing seconds axes are provisional; use frame numbers until acquisition timing/selection continuity is verified."))
    scale = acquisition.get("um_per_pixel")
    notes.append(f"User-declared spatial calibration: {scale} µm/pixel ({acquisition.get('scale_source', '')}). Stored metrics remain in native pixel units; no silent conversion." if scale else "No verified spatial scale recorded. Lengths/areas remain pixels/pixels², not micrometres.")
    notes.append("ROI experiment selected: results are development outputs and are not directly interchangeable with the locked PATEND central-crop benchmark." if acquisition.get("experimental_roi") else "No guided ROI override selected. See captured settings for the actual crop (including any manual crop chosen in legacy controls).")
    notes.append("Bright regions are an algorithmic phase, not proof of plasma identity. DIC, stained smears and channel-wall images need separate segmentation suitability assessment.")
    return notes


def metric_availability(analysis):
    df = analysis.get("metrics", pd.DataFrame())
    records = []
    for spec in available_metrics(df):
        view = metric_view(df, spec)
        counts = view_counts(view)
        allowed = view["value"].dropna()
        scope = "Raw descriptor; inspect segmentation/units"
        if spec.family == "score":
            scope = "Sequence-relative experimental score"
            if len(df) < 2:
                allowed = allowed.iloc[:0]
                scope = "NOT INTERPRETABLE from one image"
        records.append({"metric": spec.label, "source": view["source_column"].iloc[0] if len(view) else "unavailable",
                        "unit": spec.unit, "QC accepted": counts["Accepted"], "excluded": counts["Excluded"],
                        "unavailable": counts["Unavailable"], "QC unassessed": counts["QC unavailable"],
                        "reportable finite n": len(allowed), "median": float(allowed.median()) if len(allowed) else np.nan,
                        "interpretation": scope})
    return pd.DataFrame(records)


QC_CATEGORIES = {
    "Image-quality diagnostics": ("opticalWarningFlag", "qcFailSHI_tileCV", "qcFailLowContrastSeparation", "rawGainCorrectionSensitiveFlag"),
    "Mask-plausibility diagnostics": ("hardBadMaskFlag", "warningMaskFlag", "hugeGapFlag", "qcFailImplausibleMask", "sparseMaskMorphologyFiltered", "unstableSegmentationFlag"),
    "Temporal/reference eligibility": ("beforeStartFrame", "beforeStartExcluded", "qcOutsideStableWindow", "usedInSummary", "usedForMainSummary", "trustedForScore"),
}


def qc_category_table(df):
    rows = []
    for category, fields in QC_CATEGORIES.items():
        found = False
        for field in fields:
            if field not in df:
                continue
            found = True
            flag = bool_values(df[field])
            positive = field in ("usedInSummary", "usedForMainSummary", "trustedForScore")
            rows.append({"category": category, "stored field": field, "true frames": int(flag.sum()),
                         "meaning of true": "Eligible under this stored rule" if positive else "Warning/exclusion indicator; inspect exact metric QC", "total": len(df)})
        if not found:
            rows.append({"category": category, "stored field": "Not available", "true frames": None, "meaning of true": "Unassessed, not passed", "total": len(df)})
    return pd.DataFrame(rows)


def reference_diagnostics(analysis):
    """Reconstruct the actual trusted SASI/GapScore reference without updating it.

    Call the frozen helper with the SAME source columns/eligibility as V3's
    add_rbc_qc_aware_score_columns. No new thresholds or normalization constants.
    PTI and S_struct/AggIndex are explicitly outside this diagnostic's scope.
    """
    df = analysis.get("metrics", pd.DataFrame())
    required = ["Xi_QC", "diameterForTrustedScore", "countForTrustedScore", "areaForTrustedScore", "trustedForScore"]
    if df.empty or any(c not in df for c in required):
        return pd.DataFrame()
    fixed = analysis.get("params", {}).get("rbcScoreReferenceStats") or None
    eligible = bool_values(df["trustedForScore"])
    scores = compute_official_mask_scores(df, xi_col=required[0], diameter_col=required[1],
                                         count_col=required[2], area_col=required[3],
                                         reference_mask=eligible, reference_stats=fixed)
    stats = scores["scoreReferenceStats"].iloc[0]
    rows = []
    for key, col in zip(("xi_mask", "gap_mean_diameter", "gap_count", "gap_area_fraction"), required[:4]):
        stat = stats[key]
        source = "Supplied fixed reference" if fixed else "Within-sequence eligible finite rows"
        finite = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).notna()
        if not fixed and not (eligible & finite).any():
            source = "Legacy fallback to all finite rows (no eligible finite rows)"
        caution = []
        if not fixed and stat["n"] < 2:
            caution.append("Fewer than two reference observations: no sequence dispersion")
        if stat["mad"] == 0:
            caution.append("ZERO MAD: legacy denominator uses machine epsilon; saturation/instability possible")
        if not np.isfinite(stat["mad"]):
            caution.append("No finite reference dispersion")
        rows.append({"component": key, "input column": col, "n": stat["n"], "median": stat["median"],
                     "raw MAD": stat["mad"], "reference source": source, "caution": "; ".join(caution) or "No zero-MAD warning; this is not a robustness certificate"})
    return pd.DataFrame(rows)
