"""Unified mask -> measurement -> provenance pipeline, independent of segmentation."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .v4_ingestion import load_sequence, array_hash
from .v4_masks import supplied_mask, validate_roi, crop
from .v4_diagnostics import image_diagnostics, mask_diagnostics, temporal_diagnostics, THRESHOLDS
from .structural_core import measure_mask, measure_temporal, CORE_VERSION, SCHEMA_VERSION, METRIC_NOTES, metric_unit

APP_VERSION = "4.0.0-dev.2"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def sequence_warnings(frames):
    names = [f.name for f in frames]
    if len(names) < 2:
        return []
    if all("::page" in name for name in names):
        return ["TIFF page order is explicit, but pages are not necessarily time points; confirm acquisition interpretation."]
    matched = [re.match(r"^(.*?)(\d+)$", Path(name).stem) for name in names]
    if all(matched) and len({m[1] for m in matched}) == 1:
        ids = [int(m[2]) for m in matched]
        return ["Filename frame numbers are non-consecutive; uniform time spacing may be invalid."] if any(b-a != 1 for a, b in zip(ids, ids[1:])) else []
    return ["Acquisition continuity cannot be established from filenames; order is natural filename order, then TIFF page order."]


def validate_acquisition(acquisition):
    a = deepcopy(acquisition or {})
    for key in ("um_per_pixel", "time_interval_s"):
        value = a.get(key)
        if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be a positive finite number when supplied.")
    if a.get("um_per_pixel") is not None and not str(a.get("scale_source", "")).strip():
        raise ValueError("A physical calibration needs its measurement/source; metadata alone is not verification.")
    return a


def prepare_supplied(mask_files, image_files=None, *, interpretation="binary", foreground=None,
                     threshold=None, stack_policy="reject", selected_page=1, roi=None,
                     pairing_confirmed=False, acquisition=None):
    """Decode + validate + preview only. Does NOT invoke segmentation or metrics."""
    acquisition = validate_acquisition(acquisition)
    masks = load_sequence(mask_files, stack_policy=stack_policy, selected_page=selected_page)
    images = load_sequence(image_files, stack_policy=stack_policy, selected_page=selected_page) if image_files else []
    if images and len(images) != len(masks):
        raise ValueError("Image/mask frame counts differ. Pairing is one-to-one; no missing matches skipped.")
    if images and not pairing_confirmed:
        raise ValueError("Confirm the displayed natural-order image/mask pairing explicitly before analysis.")
    frames, warnings = [], sequence_warnings(masks)
    settings = {"mask_interpretation": interpretation, "foreground": foreground, "threshold": threshold,
                "stack_policy": stack_policy, "selected_page": selected_page, "roi_1based_xywh": roi,
                "pairing": "user-confirmed natural-order one-to-one" if images else "mask-only",
                "acquisition": acquisition, "core_version": CORE_VERSION}
    for i, mf in enumerate(masks):
        image = images[i] if images else None
        binary, provenance = supplied_mask(mf.pixels, interpretation=interpretation, foreground=foreground,
                                           threshold=threshold, image_shape=image.pixels.shape if image else None)
        rect = validate_roi(roi, binary.shape)
        if image and image.metadata["orientation"] != mf.metadata["orientation"]:
            provenance["warnings"].append("Image and mask orientation tags differ; transforms applied separately. Equal dimensions cannot prove geometric correspondence; review overlay.")
        if image and image.metadata.get("calibration_metadata") != mf.metadata.get("calibration_metadata"):
            provenance["warnings"].append("Image/mask calibration metadata differ or are unavailable. No rescaling/alignment attempted; user-declared calibration recorded separately.")
        analysed_mask = crop(binary, rect)
        analysed_image = crop(image.pixels, rect) if image else None
        provenance.update(filename=mf.name, paired_image=image.name if image else None,
                          mask_file_sha256=mf.metadata["input_sha256"], mask_sha256=array_hash(analysed_mask),
                          image_ingestion=deepcopy(image.metadata) if image else None,
                          mask_ingestion=deepcopy(mf.metadata), roi_1based_xywh=list(rect),
                          original_mask_dimensions=list(binary.shape), analysed_dimensions=list(analysed_mask.shape),
                          spatial_calibration=acquisition.get("um_per_pixel"), calibration_source=acquisition.get("scale_source"),
                          time_interval_s=acquisition.get("time_interval_s"), settings_sha256=digest(settings))
        for record in (mf.metadata, image.metadata if image else {}):
            warnings.extend(record.get("warnings", []))
        warnings.extend(provenance["warnings"])
        frames.append({"filename": mf.name, "mask": analysed_mask, "image": analysed_image,
                       "original_image": image.pixels if image else mf.pixels, "original_mask": binary,
                       "provenance": provenance})
    return {"frames": frames, "settings": settings, "warnings": list(dict.fromkeys(warnings)),
            "acquisition": acquisition, "signature": digest([f["provenance"] for f in frames])}


def analyze_supplied(prepared):
    """Supplied-mask route has no import or call to any segmentation function."""
    result = {"mode": "Verified Mask Analysis", "params": deepcopy(prepared["settings"]),
              "warnings": prepared["warnings"].copy(), "previews": prepared["frames"],
              "segmentation_adapter": {"name": "User-supplied mask", "version": "4.0-supplied-1",
                                       "source": "User supplied", "domain": "Generic binary foreground",
                                       "implementation": "Automatic segmentation BYPASSED"}}
    result["v4_context"] = make_context(result, [f["provenance"] for f in prepared["frames"]], prepared["acquisition"])
    ensure_measurements(result)
    result["metrics"] = result["v4_structural_metrics"].copy()
    return result


def make_context(result, frames, acquisition=None):
    root = Path(__file__).resolve().parents[1]
    paths = [root / "app.py", root / "requirements.txt", root / "requirements-v4.txt", root / "v4_codec_runtime.json",
             root / "tools/v4_codec_worker.py", *sorted((root / "modules").glob("*.py"))]
    return {"app_version": APP_VERSION, "analysis_schema": SCHEMA_VERSION, "legacy_analysis_schema": "3.1",
            "structural_core_version": CORE_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
            "segmentation": result["segmentation_adapter"], "mask_source": result["segmentation_adapter"]["source"],
            "settings_sha256": digest(result["params"]), "settings": deepcopy(result["params"]),
            "source_hashes": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()},
            "frames": frames, "acquisition": validate_acquisition(acquisition),
            "diagnostic_thresholds": deepcopy(THRESHOLDS), "warnings": result.get("warnings", []).copy(),
            "units": "Native pixels / pixels² unless explicitly stated. Spatial calibration recorded, not silently applied.",
            "scope": "DEVELOPMENT / REGRESSION. Input checks do not establish biological correctness or independent validation.",
            "rbc_indices": "Experimental RBC-specific derived indices; unchanged and available only through RBC route. Not generic measures."}


def attach_automatic(result, loaded_frames, acquisition=None):
    lookup = {f.name: f for f in loaded_frames}
    provenance = []
    for preview in result.get("previews", []):
        source = lookup.get(preview["filename"])
        mask = np.asarray(preview["mask"], dtype=bool)
        rect = None
        # Conservative engine records exact half-open bounds in previews/metrics.
        candidates = [preview, preview.get("crop_info", {}), preview.get("crop_metadata", {})]
        metrics = result.get("metrics", pd.DataFrame())
        matches = metrics.loc[metrics["filename"].eq(preview["filename"])] if "filename" in metrics else pd.DataFrame()
        if not matches.empty:
            candidates.append(matches.iloc[0].to_dict())
        for c in candidates:
            if all(k in c for k in ("cropX0_0based", "cropY0_0based", "cropX1_0based_exclusive", "cropY1_0based_exclusive")):
                x, y = int(c["cropX0_0based"]), int(c["cropY0_0based"])
                rect = [x+1, y+1, int(c["cropX1_0based_exclusive"])-x, int(c["cropY1_0based_exclusive"])-y]
                break
        if rect is None and source:
            from .segmentation import apply_crop
            if source.pixels.shape[:2] == mask.shape:
                rect = [1, 1, mask.shape[1], mask.shape[0]]
            else:
                _, bounds = apply_crop(source.pixels, result["params"])
                expected = [bounds["cropX0_0based"]+1, bounds["cropY0_0based"]+1,
                            bounds["cropX1_0based_exclusive"]-bounds["cropX0_0based"], bounds["cropY1_0based_exclusive"]-bounds["cropY0_0based"]]
                if tuple(expected[2:][::-1]) == mask.shape:
                    rect = expected
        record = {"filename": preview["filename"], "mask_source": "Automatically segmented",
                  "mask_sha256": array_hash(mask), "image_ingestion": deepcopy(source.metadata) if source else None,
                  "roi_1based_xywh": rect, "analysed_dimensions": list(mask.shape),
                  "original_mask_dimensions": list(source.pixels.shape[:2]) if source else None,
                  "segmentation": result["segmentation_adapter"], "settings_sha256": digest(result["params"]),
                  "biological_correctness": "Not established by computational QC"}
        if source and rect:
            preview["v4_diagnostic_image"] = crop(source.pixels, rect)
            record["image_diagnostics_source"] = "Original decoded numeric image, same ROI; before legacy segmentation preprocessing"
        else:
            preview["v4_diagnostic_image"] = None
            record["image_diagnostics_source"] = "Unavailable; not substituted with an intensity-corrected image"
        record["engine_preprocessing"] = {
            "implementation": result["segmentation_adapter"]["implementation"],
            "version": result["segmentation_adapter"]["version"],
            "conversion": "Unchanged legacy grayscale/float normalization and selected filters after lossless ingress; see exact resolved settings. Display scaling is never an analysis input.",
            "settings_sha256": digest(result["params"]),
        }
        if rect is None:
            result["warnings"].append(f"{preview['filename']}: ROI transform unavailable; this run is not a complete-provenance freeze candidate.")
        preview["provenance"] = record
        provenance.append(record)
    result["v4_context"] = make_context(result, provenance, acquisition)
    result["v4_context"]["selected_inputs"] = [deepcopy(f.metadata) for f in loaded_frames]
    processed = set(lookup) & {p["filename"] for p in provenance}
    result["v4_context"]["selected_not_processed"] = sorted(set(lookup) - processed)
    return result


def update_acquisition(result, acquisition):
    if "v4_context" not in result:
        return
    result["v4_context"]["acquisition"] = validate_acquisition(acquisition)
    for frame in result["v4_context"]["frames"]:
        frame["spatial_calibration"] = acquisition.get("um_per_pixel")
        frame["calibration_source"] = acquisition.get("scale_source")
        frame["time_interval_s"] = acquisition.get("time_interval_s")
    # Metadata controls are kept outside unchanged legacy engine settings.
    result["v4_context"]["run_provenance_sha256"] = digest({"settings": result["v4_context"]["settings"], "acquisition": acquisition})
    result.pop("v4_diagnostics", None)


def ensure_measurements(result):
    """Lazy, per-result cache: switching plot/report sections does not recalculate."""
    if "v4_structural_metrics" not in result:
        rows, distributions, masks = [], [], []
        for preview in result.get("previews", []):
            binary = np.asarray(preview["mask"], dtype=bool)
            values, objects = measure_mask(binary)
            rows.append({"filename": preview["filename"], **values})
            objects.insert(0, "filename", preview["filename"])
            distributions.append(objects)
            masks.append(binary)
        result["v4_structural_metrics"] = pd.DataFrame(rows)
        result["v4_component_distribution"] = pd.concat(distributions, ignore_index=True) if distributions else pd.DataFrame()
        result["v4_temporal"] = measure_temporal(masks) if masks else pd.DataFrame()
        result["v4_metric_availability"] = pd.DataFrame([
            {"metric": c, "units": metric_unit(c), "finite_count": int(np.isfinite(result["v4_structural_metrics"][c]).sum()),
             "total_frames": len(rows), "note": METRIC_NOTES.get(c, "Direct descriptor of the selected binary foreground; no biological identity implied.")}
            for c in result["v4_structural_metrics"] if c != "filename"])
    if "v4_diagnostics" not in result:
        diagnostics = []
        for p in result.get("previews", []):
            raw = p.get("v4_diagnostic_image", p.get("image"))
            record = p.get("provenance", {})
            d = image_diagnostics(raw, record.get("image_ingestion")) + mask_diagnostics(p["mask"])
            diagnostics += [{"filename": p["filename"], **r} for r in d]
        ctx = result.get("v4_context", {})
        a = ctx.get("acquisition", {})
        native_reference_eligible = False
        if result.get("mode") == "RBC Gap Mode":
            from .development_safety import reference_diagnostics
            reference = reference_diagnostics(result)
            if not reference.empty:
                native_reference_eligible = bool((reference["n"] >= 2).all() and
                                                 (reference["raw MAD"] > 0).all() and
                                                 not reference["reference source"].str.contains("fallback", case=False).any())
                for ref in reference.to_dict("records"):
                    diagnostics.append({"filename": "sequence", "category": "TEMPORAL / REFERENCE ELIGIBILITY",
                                        "diagnostic": "Legacy RBC reference: " + str(ref["component"]),
                                        "value": {"n": ref["n"], "median": ref["median"], "raw MAD": ref["raw MAD"]},
                                        "status": "PASS" if str(ref.get("caution", "")).startswith("No zero-MAD warning") else "CAUTION",
                                        "reason": "Unchanged reference reconstruction: " + str(ref.get("reference source", "")) + "; " + str(ref.get("caution", "")),
                                        "diagnostics_version": "4.0-observable-1"})
        diagnostics += [{"filename": "sequence", **r} for r in temporal_diagnostics(
            len(result.get("previews", [])), confirmed=a.get("timing_confirmed", False), interval=a.get("time_interval_s"),
            reference_available=native_reference_eligible, ordering_warnings=[w for w in result.get("warnings", []) if "order" in w.lower() or "consecutive" in w.lower()])]
        result["v4_diagnostics"] = pd.DataFrame(diagnostics)
    return result
