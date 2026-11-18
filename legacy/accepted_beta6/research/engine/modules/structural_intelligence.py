"""MicroStructAI V3 structural-intelligence orchestration and exports.

This layer is deliberately downstream of the selected final mask.  It never
selects a competing trusted gap mask and never writes into existing score
columns.  New outputs carry explicit mask/preprocessing/POD provenance.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from skimage import filters, measure, morphology

from .gap_dynamics import DynamicsConfig, analyze_gap_dynamics
from .morphology_intelligence import (
    STATIC_ALGORITHM_VERSION,
    StaticConfig,
    analyze_static_gap_structure,
    as_binary_mask,
    contour_mode_analysis,
    gap_network_features,
    gap_object_features,
    local_thickness_analysis,
    topology_features,
)


STRUCTURAL_INTELLIGENCE_VERSION = "3.1.0-rbc-structural-intelligence"


def structural_settings_from_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(params.get("enableStructuralIntelligence", True)),
        "frameStride": max(1, int(params.get("structuralFrameStride", 1))),
        "enableCellRichSecondary": bool(params.get("enableCellRichSecondary", True)),
        "enableDynamics": bool(params.get("enableGapDynamics", True)),
        "enableRobustness": bool(params.get("enableStructuralRobustness", True)),
        "robustnessStride": max(1, int(params.get("structuralRobustnessStride", 10))),
        "minObjectAreaPx": max(1, int(params.get("structuralMinObjectAreaPx", params.get("minAreaPx", 20)))),
        "granulometryRadiiPx": tuple(int(v) for v in params.get("structuralGranulometryRadiiPx", [1, 2, 3, 4, 6, 8, 12])),
        "topologyScalesPx": tuple(int(v) for v in params.get("structuralTopologyScalesPx", [-4, -2, -1, 0, 1, 2, 4])),
        "gridSizes": tuple(int(v) for v in params.get("structuralGridSizes", [2, 4, 8])),
        "contourMaxMode": max(2, int(params.get("structuralContourMaxMode", 10))),
        "contourMinAreaPx": max(20, int(params.get("structuralContourMinAreaPx", 100))),
        "contourResamplePoints": max(32, int(params.get("structuralContourResamplePoints", 128))),
        "contourSmoothingSigma": max(0.0, float(params.get("structuralContourSmoothingSigma", 1.0))),
        "darkBlurSigma": max(0.0, float(params.get("cellRichBlurSigma", 1.0))),
        "darkMinAreaPx": max(1, int(params.get("cellRichMinAreaPx", params.get("minAreaPx", 20)))),
        "darkThresholdShiftFraction": float(params.get("cellRichThresholdShiftFraction", 0.05)),
        "registrationMaxError": float(params.get("registrationMaxError", 0.75)),
        "registrationMaxShiftFraction": float(params.get("registrationMaxShiftFraction", 0.10)),
        "trackingMinIoU": float(params.get("trackingMinIoU", 0.05)),
        "trackingMinOverlap": float(params.get("trackingMinOverlapFraction", 0.15)),
        "trackingMaxCentroidDistancePx": float(params.get("trackingMaxCentroidDistancePx", 35.0)),
        "sourceSettings": {
            "selectedAnalysisMode": params.get("rbcAnalysisMethod", params.get("rbcMode", "RBC Gap Mode")),
            "cropFrac": params.get("cropFrac", 0.70),
            "spatialCorrectionMethod": params.get("spatialCorrectionMethod", "divide"),
            "spatialIllumSigma": params.get("spatialIllumSigma", 60.0),
            "spatialCorrectionStrength": params.get("spatialCorrectionStrength", 0.50),
            "applyRawGainCorrectionToImages": params.get("applyRawGainCorrectionToImages", True),
            "enableMeanIlluminationCorrection": params.get("enableMeanIlluminationCorrection", True),
            "enablePODRemoval": params.get("enablePODRemoval", False),
            "removeModes": params.get("removeModes", 0),
        },
        "algorithmVersion": STRUCTURAL_INTELLIGENCE_VERSION,
    }


def _static_config(settings: dict[str, Any]) -> StaticConfig:
    return StaticConfig(
        min_object_area_px=settings["minObjectAreaPx"],
        min_network_object_area_px=settings["minObjectAreaPx"],
        granulometry_radii_px=tuple(settings["granulometryRadiiPx"]),
        topology_scales_px=tuple(settings["topologyScalesPx"]),
        grid_sizes=tuple(settings["gridSizes"]),
        contour_min_area_px=settings["contourMinAreaPx"],
        contour_points=settings["contourResamplePoints"],
        contour_smoothing_sigma=settings["contourSmoothingSigma"],
        contour_max_mode=settings["contourMaxMode"],
    )


def _dynamics_config(settings: dict[str, Any]) -> DynamicsConfig:
    return DynamicsConfig(
        max_registration_error=settings["registrationMaxError"],
        max_registration_fraction=settings["registrationMaxShiftFraction"],
        min_object_area_px=settings["minObjectAreaPx"],
        min_iou=settings["trackingMinIoU"],
        min_overlap_fraction=settings["trackingMinOverlap"],
        max_centroid_distance_px=settings["trackingMaxCentroidDistancePx"],
    )


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a, b = as_binary_mask(mask_a), as_binary_mask(mask_b)
    denominator = float(a.sum() + b.sum())
    return float(2.0 * np.logical_and(a, b).sum() / denominator) if denominator > 0 else 1.0


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a, b = as_binary_mask(mask_a), as_binary_mask(mask_b)
    union = float(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum() / union) if union > 0 else 1.0


def _cv(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    mean = float(np.mean(arr))
    return float(np.std(arr, ddof=1) / abs(mean)) if abs(mean) > 1e-12 else np.nan


def reliability_class(median_dice: float, feature_cv: float) -> str:
    if np.isfinite(median_dice) and median_dice >= 0.90 and (not np.isfinite(feature_cv) or feature_cv <= 0.10):
        return "RELIABLE"
    if np.isfinite(median_dice) and median_dice >= 0.75 and (not np.isfinite(feature_cv) or feature_cv <= 0.25):
        return "MODERATE"
    if np.isfinite(median_dice) and median_dice >= 0.50:
        return "UNSTABLE"
    return "UNRELIABLE"


def gap_boundary_robustness(mask: np.ndarray, min_area_px: int = 20) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Test descriptor sensitivity to bounded one-pixel boundary perturbations.

    This is a downstream boundary-robustness test.  It does not replace or
    retune the official segmentation algorithm.
    """
    official = as_binary_mask(mask)
    disk = morphology.disk(1)
    variants = {
        "official": official,
        "erode_1px": morphology.binary_erosion(official, disk),
        "dilate_1px": morphology.binary_dilation(official, disk),
        "open_1px": morphology.binary_opening(official, disk),
        "close_1px": morphology.binary_closing(official, disk),
    }
    rows = []
    for name, variant in variants.items():
        labels = measure.label(variant, connectivity=2)
        areas = [region.area for region in measure.regionprops(labels) if region.area >= min_area_px]
        thickness, _map, _dist = local_thickness_analysis(variant)
        topology = topology_features(variant)
        network, _skeleton, _edges = gap_network_features(variant, StaticConfig(min_object_area_px=min_area_px, min_network_object_area_px=min_area_px))
        contour_objects = gap_object_features(variant, StaticConfig(min_object_area_px=min_area_px))
        contour_modes, _visuals, _contour_summary = contour_mode_analysis(
            variant,
            contour_objects,
            StaticConfig(min_object_area_px=min_area_px, contour_min_area_px=max(50, min_area_px), max_contours_per_frame=1),
        )
        contour_values = {}
        if not contour_modes.empty:
            largest_object = int(contour_modes.sort_values("objectArea_px2", ascending=False).iloc[0]["gapObjectID"])
            largest_modes = contour_modes.loc[contour_modes["gapObjectID"] == largest_object]
            contour_values = {
                f"contourMode{int(mode_row['mode'])}Amplitude": float(mode_row["normalizedAmplitude"])
                for _, mode_row in largest_modes.iterrows()
                if int(mode_row["mode"]) >= 2
            }
        rows.append(
            {
                "perturbation": name,
                "diceVsOfficial": dice(official, variant),
                "iouVsOfficial": iou(official, variant),
                "gapAreaFraction": float(np.mean(variant)),
                "gapObjectCount": len(areas),
                "medianGapArea_px2": float(np.median(areas)) if areas else np.nan,
                "medianLocalThickness_px": thickness["localThicknessMedian_px"],
                "eulerCharacteristic": topology["topologyEulerCharacteristic"],
                "gapNetworkSkeletonDensity": network["gapNetworkSkeletonDensity"],
                **contour_values,
            }
        )
    table = pd.DataFrame(rows)
    perturbed = table.loc[table["perturbation"] != "official"]
    feature_cvs = {
        column: _cv(table[column].tolist())
        for column in ["gapAreaFraction", "gapObjectCount", "medianGapArea_px2", "medianLocalThickness_px", "eulerCharacteristic", "gapNetworkSkeletonDensity"]
    }
    finite_cvs = [value for value in feature_cvs.values() if np.isfinite(value)]
    median_feature_cv = float(np.median(finite_cvs)) if finite_cvs else np.nan
    median_dice = float(perturbed["diceVsOfficial"].median()) if not perturbed.empty else np.nan
    summary = {
        "gapRobustnessMedianDice": median_dice,
        "gapRobustnessMedianIoU": float(perturbed["iouVsOfficial"].median()) if not perturbed.empty else np.nan,
        "gapRobustnessMedianFeatureCV": median_feature_cv,
        "gapRepresentationStability": reliability_class(median_dice, median_feature_cv),
        "gapRobustnessFeatureCV": feature_cvs,
        "gapRobustnessScope": "one-pixel erosion/dilation/opening/closing of the unchanged official mask",
    }
    return table, summary


def _cellrich_mask(image: np.ndarray, threshold: float, blur_sigma: float, min_area_px: int) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    smooth = ndi.gaussian_filter(arr, sigma=max(0.0, blur_sigma)) if blur_sigma > 0 else arr
    mask = smooth <= float(threshold)
    mask = morphology.remove_small_objects(mask, min_size=min_area_px, connectivity=2)
    mask = morphology.binary_closing(mask, morphology.disk(1))
    return mask.astype(bool)


def analyze_cellrich_representation(image: np.ndarray, settings: dict[str, Any]) -> dict[str, Any]:
    """Analyse dark aggregate-rich domains as a secondary, QC-gated phase."""
    arr = np.asarray(image, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=float(np.nanmedian(arr)) if np.isfinite(arr).any() else 0.0)
    dynamic_range = float(np.percentile(arr, 99) - np.percentile(arr, 1))
    if dynamic_range <= 1e-8:
        return {
            "summary": {"cellRichReliability": "UNRELIABLE", "cellRichFailureReason": "insufficient_intensity_contrast"},
            "mask": np.zeros_like(arr, dtype=bool),
            "objects": pd.DataFrame(),
            "robustness": pd.DataFrame(),
        }
    try:
        # Three-class Otsu separates two plausible dark-domain levels from a
        # bright background; the upper boundary retains both dark classes.
        # Fall back to ordinary Otsu when the intensity support is too small.
        if np.unique(arr).size >= 3:
            threshold = float(filters.threshold_multiotsu(arr, classes=3)[-1])
        else:
            threshold = float(filters.threshold_otsu(arr))
    except ValueError:
        threshold = float(np.median(arr))
    # Otsu thresholds are histogram-bin edges. Add half of one nominal
    # 8-bit-equivalent bin so pixels at the highest dark-class level are not
    # lost solely to quantization.
    threshold += dynamic_range / 512.0
    shift = settings["darkThresholdShiftFraction"] * dynamic_range
    variants = [
        ("threshold_low", threshold - shift, settings["darkBlurSigma"]),
        ("selected", threshold, settings["darkBlurSigma"]),
        ("threshold_high", threshold + shift, settings["darkBlurSigma"]),
        ("blur_low", threshold, max(0.0, settings["darkBlurSigma"] - 0.5)),
        ("blur_high", threshold, settings["darkBlurSigma"] + 0.5),
    ]
    masks = {name: _cellrich_mask(arr, value, blur, settings["darkMinAreaPx"]) for name, value, blur in variants}
    selected = masks["selected"]
    selected_labels = measure.label(selected, connectivity=2)
    object_rows = []
    for region in measure.regionprops(selected_labels):
        if region.area < settings["darkMinAreaPx"]:
            continue
        object_rows.append(
            {
                "cellRichDomainID": int(region.label),
                "area_px2": float(region.area),
                "eccentricity": float(region.eccentricity),
                "orientation_deg": float(np.degrees(region.orientation)),
                "solidity": float(region.solidity),
                "touchesROIBorder": bool((region.coords[:, 0] == 0).any() or (region.coords[:, 1] == 0).any() or (region.coords[:, 0] == selected.shape[0] - 1).any() or (region.coords[:, 1] == selected.shape[1] - 1).any()),
            }
        )
    objects = pd.DataFrame(object_rows)
    robustness_rows = []
    counts, fractions = [], []
    for name, mask in masks.items():
        labels = measure.label(mask, connectivity=2)
        count = int(labels.max())
        area_fraction = float(mask.mean())
        counts.append(float(count))
        fractions.append(area_fraction)
        robustness_rows.append(
            {
                "perturbation": name,
                "diceVsSelected": dice(selected, mask),
                "iouVsSelected": iou(selected, mask),
                "cellRichAreaFraction": area_fraction,
                "cellRichDomainCount": count,
            }
        )
    robustness = pd.DataFrame(robustness_rows)
    perturbed = robustness.loc[robustness["perturbation"] != "selected"]
    feature_cv = float(np.nanmedian([_cv(fractions), _cv(counts)]))
    median_dice = float(perturbed["diceVsSelected"].median())
    reliability = reliability_class(median_dice, feature_cv)
    areas = pd.to_numeric(objects.get("area_px2", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    summary = {
        "cellRichAreaFraction": float(selected.mean()),
        "cellRichConnectedDomainCount": int(len(objects)),
        "cellRichMedianDomainArea_px2": float(np.median(areas)) if areas.size else np.nan,
        "cellRichLargestDomainFraction": float(np.max(areas) / selected.size) if areas.size else 0.0,
        "cellRichMedianEccentricity": float(pd.to_numeric(objects.get("eccentricity", pd.Series(dtype=float)), errors="coerce").median()) if not objects.empty else np.nan,
        "cellRichMedianSolidity": float(pd.to_numeric(objects.get("solidity", pd.Series(dtype=float)), errors="coerce").median()) if not objects.empty else np.nan,
        "cellRichThreshold": threshold,
        "cellRichMedianDice": median_dice,
        "cellRichMedianFeatureCV": feature_cv,
        "cellRichReliability": reliability,
        "cellRichFailureReason": "" if reliability in {"RELIABLE", "MODERATE"} else "bounded threshold/blur perturbations were not reproducible",
        "cellRichInterpretation": "dark/cell-rich connected domains; not individual RBCs",
    }
    return {"summary": summary, "mask": selected, "objects": objects, "robustness": robustness}


def _metadata_for_frame(row: pd.Series, position: int) -> dict[str, Any]:
    def value(column: str, default: Any = None) -> Any:
        result = row.get(column, default)
        if isinstance(result, np.generic):
            result = result.item()
        return result

    return {
        "framePosition": position,
        "filename": value("filename", f"frame_{position}"),
        "sourceFrameNumber": value("sourceFrameNumber", value("originalImageIndex_1based", position)),
        "analysisIndex_1based": value("analysisIndex_1based", position),
        "beforeStartFrame": bool(value("beforeStartFrame", False)),
        "usedForMainSummary": bool(value("usedForMainSummary", value("usedInSummary", True))),
        "trustedForScore": bool(value("trustedForScore", True)),
        "rbcFrameQC": value("rbcFrameQC", "not_available"),
        "selectedAnalysisMode": value("selectedRBCMethod", "RBC Gap Mode"),
        "preprocessingSource": value("finalPreprocessingSource", "selected final preprocessing"),
        "maskSource": value("selectedFinalMaskSource", value("finalMaskSource", "selected final mask")),
        "podStatus": value("podStatus", "not_available"),
        "podProcessingEnabled": bool(value("podProcessingEnabled", False)),
        "existingXi_mask": value("Xi_mask", np.nan),
        "existingSASI_0_100": value("SASI_0_100", np.nan),
        "existingGapScore_0_100": value("GapScore_0_100", np.nan),
        "existingAggIndex_0_100": value("AggIndex_0_100", np.nan),
        "existingGapAreaFraction": value("gapAreaFrac", np.nan),
        "structuralSequenceAlgorithmVersion": STRUCTURAL_INTELLIGENCE_VERSION,
    }


def _add_meta(table: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame()
    out = table.copy()
    for key, value in reversed(list(metadata.items())):
        if key in out.columns:
            out[key] = value
        else:
            out.insert(0, key, value)
    return out


def _combine_frame_tables(frames: list[dict[str, Any]], key: str) -> pd.DataFrame:
    tables = [_add_meta(frame.get(key, pd.DataFrame()), frame["metadata"]) for frame in frames]
    tables = [table for table in tables if not table.empty]
    return pd.concat(tables, ignore_index=True, sort=False) if tables else pd.DataFrame()


def _summary_table(frames: list[dict[str, Any]], key: str = "static") -> pd.DataFrame:
    rows = []
    for frame in frames:
        source = frame.get(key, {})
        summary = source.get("summary", {}) if isinstance(source, dict) else {}
        rows.append({**frame["metadata"], **summary})
    return pd.DataFrame(rows)


def analyze_structural_sequence(previews: list[dict[str, Any]], metrics: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Analyse one sequence after the existing V3 workflow has completed."""
    settings = structural_settings_from_params(params)
    if not settings["enabled"] or not previews or metrics.empty:
        return {"enabled": False, "settings": settings, "frames": [], "tables": {}, "summary": {}}
    count = min(len(previews), len(metrics))
    cfg = _static_config(settings)
    frames: list[dict[str, Any]] = []
    masks: list[np.ndarray] = []
    images: list[np.ndarray] = []
    warnings: list[str] = []
    for position in range(count):
        preview = previews[position]
        metadata = _metadata_for_frame(metrics.iloc[position], position + 1)
        mask = as_binary_mask(preview["mask"])
        image = np.asarray(preview.get("image", preview.get("camera_crop_gray")), dtype=np.float32)
        masks.append(mask)
        images.append(image)
        try:
            static = analyze_static_gap_structure(mask, cfg)
        except Exception as exc:
            static = {"summary": {"advancedGapAnalysisSuppressed": True, "advancedGapAnalysisFailureReason": str(exc)}}
            warnings.append(f"{metadata['filename']}: static structural analysis failed ({exc})")
        if settings["enableCellRichSecondary"]:
            try:
                cellrich = analyze_cellrich_representation(image, settings)
            except Exception as exc:
                cellrich = {"summary": {"cellRichReliability": "UNRELIABLE", "cellRichFailureReason": str(exc)}, "objects": pd.DataFrame(), "robustness": pd.DataFrame()}
                warnings.append(f"{metadata['filename']}: cell-rich analysis failed ({exc})")
        else:
            cellrich = {"summary": {"cellRichReliability": "NOT_ASSESSED", "cellRichFailureReason": "disabled"}, "objects": pd.DataFrame(), "robustness": pd.DataFrame()}
        if settings["enableRobustness"] and position % settings["robustnessStride"] == 0:
            try:
                robustness_table, robustness_summary = gap_boundary_robustness(mask, settings["minObjectAreaPx"])
            except Exception as exc:
                robustness_table, robustness_summary = pd.DataFrame(), {"gapRepresentationStability": "NOT_ASSESSED", "gapRobustnessFailureReason": str(exc)}
        else:
            robustness_table, robustness_summary = pd.DataFrame(), {"gapRepresentationStability": "NOT_ASSESSED"}
        frames.append(
            {
                "metadata": metadata,
                "static": static,
                "cellrich": cellrich,
                "gapRobustness": {"table": robustness_table, "summary": robustness_summary},
            }
        )

    if settings["enableDynamics"] and len(masks) >= 2:
        try:
            dynamics = analyze_gap_dynamics(images, masks, _dynamics_config(settings))
        except Exception as exc:
            dynamics = {"summary": {"trackingSuppressed": True, "trackingSuppressionReason": str(exc)}, "registration": {"qc": pd.DataFrame()}, "tracking": {"tracks": pd.DataFrame(), "events": pd.DataFrame()}, "turnover": pd.DataFrame()}
            warnings.append(f"Sequence dynamics failed ({exc})")
    else:
        dynamics = {"summary": {"trackingSuppressed": True, "trackingSuppressionReason": "disabled or fewer than two frames"}, "registration": {"qc": pd.DataFrame()}, "tracking": {"tracks": pd.DataFrame(), "events": pd.DataFrame()}, "turnover": pd.DataFrame()}

    tables = {
        "gap_frame_summary": _summary_table(frames, "static"),
        "gap_object_features": _combine_frame_tables([{"metadata": f["metadata"], "table": f.get("static", {}).get("objects", pd.DataFrame())} for f in frames], "table"),
        "gap_size_spectrum": _combine_frame_tables([{"metadata": f["metadata"], "table": f.get("static", {}).get("sizeSpectrum", pd.DataFrame())} for f in frames], "table"),
        "gap_granulometry": _combine_frame_tables([{"metadata": f["metadata"], "table": f.get("static", {}).get("granulometry", pd.DataFrame())} for f in frames], "table"),
        "gap_chord_lengths": _combine_frame_tables([{"metadata": f["metadata"], "table": f.get("static", {}).get("chordLengths", pd.DataFrame())} for f in frames], "table"),
        "gap_local_thickness": _summary_table(frames, "static")[[c for c in _summary_table(frames, "static").columns if "Thickness" in c or c in frames[0]["metadata"]]],
        "gap_local_thickness_distribution": _combine_frame_tables([{"metadata": f["metadata"], "table": f.get("static", {}).get("thicknessDistribution", pd.DataFrame())} for f in frames], "table"),
        "gap_spatial_features": _combine_frame_tables([{"metadata": f["metadata"], "table": f.get("static", {}).get("spatialTiles", pd.DataFrame())} for f in frames], "table"),
        "gap_topology_features": _summary_table(frames, "static")[[c for c in _summary_table(frames, "static").columns if "topology" in c.lower() or c in frames[0]["metadata"]]],
        "gap_multiscale_topology": _combine_frame_tables([{"metadata": f["metadata"], "table": f.get("static", {}).get("multiscaleTopology", pd.DataFrame())} for f in frames], "table"),
        "gap_network_features": _summary_table(frames, "static")[[c for c in _summary_table(frames, "static").columns if "gapNetwork" in c or c in frames[0]["metadata"]]],
        "gap_network_degree_distribution": _combine_frame_tables([{"metadata": f["metadata"], "table": f.get("static", {}).get("networkDegreeDistribution", pd.DataFrame())} for f in frames], "table"),
        "gap_contour_modes": _combine_frame_tables([{"metadata": f["metadata"], "table": f.get("static", {}).get("contourModes", pd.DataFrame())} for f in frames], "table"),
        "gap_feature_robustness": _combine_frame_tables([{"metadata": f["metadata"], "table": f.get("gapRobustness", {}).get("table", pd.DataFrame())} for f in frames], "table"),
        "cellrich_frame_summary": _summary_table(frames, "cellrich"),
        "cellrich_object_features": _combine_frame_tables([{"metadata": f["metadata"], "table": f.get("cellrich", {}).get("objects", pd.DataFrame())} for f in frames], "table"),
        "cellrich_feature_robustness": _combine_frame_tables([{"metadata": f["metadata"], "table": f.get("cellrich", {}).get("robustness", pd.DataFrame())} for f in frames], "table"),
        "registration_qc": dynamics.get("registration", {}).get("qc", pd.DataFrame()),
        "gap_tracks": dynamics.get("tracking", {}).get("tracks", pd.DataFrame()),
        "gap_events": dynamics.get("tracking", {}).get("events", pd.DataFrame()),
        "gap_dynamic_summary": pd.DataFrame([dynamics.get("summary", {})]),
        "gap_temporal_turnover": dynamics.get("turnover", pd.DataFrame()),
    }
    dual = dual_representation_qc(frames)
    tables["dual_representation_qc"] = dual["table"]
    passport = structural_passport(tables, dual["summary"], dynamics.get("summary", {}), settings)
    return {
        "enabled": True,
        "settings": settings,
        "frames": frames,
        "dynamics": dynamics,
        "tables": tables,
        "dualRepresentation": dual,
        "passport": passport,
        "warnings": warnings,
        "summary": {**dual["summary"], **dynamics.get("summary", {})},
        "algorithmVersion": STRUCTURAL_INTELLIGENCE_VERSION,
    }


def dual_representation_qc(frames: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for frame in frames:
        gap = frame.get("gapRobustness", {}).get("summary", {})
        dark = frame.get("cellrich", {}).get("summary", {})
        rows.append(
            {
                **frame["metadata"],
                "gapRepresentationStability": gap.get("gapRepresentationStability", "NOT_ASSESSED"),
                "gapMedianDice": gap.get("gapRobustnessMedianDice", np.nan),
                "gapMedianFeatureCV": gap.get("gapRobustnessMedianFeatureCV", np.nan),
                "cellRichRepresentationStability": dark.get("cellRichReliability", "NOT_ASSESSED"),
                "cellRichMedianDice": dark.get("cellRichMedianDice", np.nan),
                "cellRichMedianFeatureCV": dark.get("cellRichMedianFeatureCV", np.nan),
                "primaryRepresentation": "GAP",
            }
        )
    table = pd.DataFrame(rows)
    rank = {"RELIABLE": 4, "MODERATE": 3, "UNSTABLE": 2, "UNRELIABLE": 1, "NOT_ASSESSED": 0}
    gap_labels = table.get("gapRepresentationStability", pd.Series(dtype=str)).astype(str)
    dark_labels = table.get("cellRichRepresentationStability", pd.Series(dtype=str)).astype(str)
    gap_score = float(gap_labels.map(rank).replace(0, np.nan).median()) if not gap_labels.empty else np.nan
    dark_score = float(dark_labels.map(rank).replace(0, np.nan).median()) if not dark_labels.empty else np.nan
    inverse = {value: key for key, value in rank.items()}
    gap_label = inverse.get(int(round(gap_score)), "NOT_ASSESSED") if np.isfinite(gap_score) else "NOT_ASSESSED"
    dark_label = inverse.get(int(round(dark_score)), "NOT_ASSESSED") if np.isfinite(dark_score) else "NOT_ASSESSED"
    gap_usable = float((gap_labels.isin(["RELIABLE", "MODERATE"])).mean()) if not gap_labels.empty else 0.0
    dark_usable = float((dark_labels.isin(["RELIABLE", "MODERATE"])).mean()) if not dark_labels.empty else 0.0
    disagreement = bool(dark_score > gap_score + 0.5) if np.isfinite(gap_score) and np.isfinite(dark_score) else False
    summary = {
        "gapRepresentationStability": gap_label,
        "cellRichRepresentationStability": dark_label,
        "gapUsableFramePercentage": 100.0 * gap_usable,
        "cellRichUsableFramePercentage": 100.0 * dark_usable,
        "gapMedianDice": float(pd.to_numeric(table.get("gapMedianDice"), errors="coerce").median()) if not table.empty else np.nan,
        "cellRichMedianDice": float(pd.to_numeric(table.get("cellRichMedianDice"), errors="coerce").median()) if not table.empty else np.nan,
        "gapMedianFeatureCV": float(pd.to_numeric(table.get("gapMedianFeatureCV"), errors="coerce").median()) if not table.empty else np.nan,
        "cellRichMedianFeatureCV": float(pd.to_numeric(table.get("cellRichMedianFeatureCV"), errors="coerce").median()) if not table.empty else np.nan,
        "primaryRepresentation": "GAP",
        "morphologyRepresentationDisagreement": disagreement,
        "recommendedAction": "Inspect segmentation" if disagreement else ("Ignore secondary representation" if dark_label in {"UNSTABLE", "UNRELIABLE"} else "Use cell-rich results only as separate secondary evidence"),
    }
    return {"table": table, "summary": summary}


def _trusted_rows(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    trusted = pd.Series(True, index=table.index)
    if "usedForMainSummary" in table:
        trusted &= table["usedForMainSummary"].fillna(False).astype(bool)
    if "trustedForScore" in table:
        trusted &= table["trustedForScore"].fillna(False).astype(bool)
    selected = table.loc[trusted].copy()
    return selected if not selected.empty else table.copy()


def _median(table: pd.DataFrame, column: str) -> float:
    if column not in table:
        return np.nan
    values = pd.to_numeric(table[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.median()) if not values.empty else np.nan


def structural_passport(tables: dict[str, pd.DataFrame], dual: dict[str, Any], dynamics: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    frame = _trusted_rows(tables.get("gap_frame_summary", pd.DataFrame()))
    cell = _trusted_rows(tables.get("cellrich_frame_summary", pd.DataFrame()))
    return {
        "title": "RBC Structural Passport",
        "researchUseOnly": True,
        "existingMicroStructAI": {
            "Xi_mask": _median(frame, "existingXi_mask"),
            "SASI_0_100": _median(frame, "existingSASI_0_100"),
            "GapScore_0_100": _median(frame, "existingGapScore_0_100"),
            "AggIndex_0_100": _median(frame, "existingAggIndex_0_100"),
            "gapAreaFraction": _median(frame, "gapAreaFraction"),
        },
        "gapPopulation": {
            "gapObjectCount": _median(frame, "gapObjectCount_all"),
            "medianGapArea_px2": _median(frame, "gapAreaMedian_px2"),
            "P90GapArea_px2": _median(frame, "gapAreaP90_px2"),
            "largestGapFraction": _median(frame, "largestGapFractionOfImage"),
        },
        "gapSpaceScale": {
            "medianLocalThickness_px": _median(frame, "localThicknessMedian_px"),
            "P90LocalThickness_px": _median(frame, "localThicknessP90_px"),
            "dominantGranulometricRadius_px": _median(frame, "dominantGranulometricRadius_px"),
            "horizontalChordMedian_px": _median(frame, "horizontalChordMedian_px"),
            "verticalChordMedian_px": _median(frame, "verticalChordMedian_px"),
        },
        "spatialStructure": {
            "heterogeneityCV": _median(frame, "spatialGapFractionCV"),
            "MoranI": _median(frame, "spatialGapFractionMoranI"),
            "leftRightGradient": _median(frame, "spatialLeftRightGradient"),
            "topBottomGradient": _median(frame, "spatialTopBottomGradient"),
        },
        "topology": {
            "components": _median(frame, "topologyConnectedComponents"),
            "holes": _median(frame, "topologyEnclosedHoles"),
            "EulerCharacteristic": _median(frame, "topologyEulerCharacteristic"),
            "topologyStability": _median(frame, "topologyStability"),
        },
        "voidNetwork": {
            "skeletonDensity": _median(frame, "gapNetworkSkeletonDensity"),
            "branchDensity": _median(frame, "gapNetworkBranchDensity"),
            "endpointDensity": _median(frame, "gapNetworkEndpointDensity"),
            "tortuosityApprox": _median(frame, "gapNetworkTortuosityApprox"),
            "nematicOrder": _median(frame, "gapNetworkNematicOrder"),
        },
        "contourStructure": {
            "computedModeMaximum": _median(frame, "contourHighestComputedMode"),
            "lowOrderEnergy": _median(frame, "contourLowOrderEnergyMedian"),
            "reliability": "requires robustness table; high-order modes are not assumed reliable",
        },
        "dynamics": dynamics,
        "cellRichSecondary": {
            "areaFraction": _median(cell, "cellRichAreaFraction"),
            "connectedDomainCount": _median(cell, "cellRichConnectedDomainCount"),
            "largestDomainFraction": _median(cell, "cellRichLargestDomainFraction"),
            "reliability": dual.get("cellRichRepresentationStability", "NOT_ASSESSED"),
        },
        "reliability": dual,
        "provenance": {
            "selectedMode": settings["sourceSettings"].get("selectedAnalysisMode"),
            "trustedMaskRule": "selected final mask supplied by existing V3 workflow",
            "PODStatus": "primary analysis remains no-POD when source settings disable POD",
            "units": "pixels, pixels^2, frames unless calibration metadata are supplied",
            "algorithmVersion": STRUCTURAL_INTELLIGENCE_VERSION,
        },
    }


def passport_markdown(passport: dict[str, Any]) -> str:
    lines = ["# RBC Structural Passport", "", "Research use only. No clinical or mechanism classification is made.", ""]
    for section, values in passport.items():
        if section in {"title", "researchUseOnly"}:
            continue
        lines.append(f"## {section.replace('_', ' ').title()}")
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, dict):
                    lines.append(f"- {key}: {json.dumps(value, default=str, sort_keys=True)}")
                else:
                    lines.append(f"- {key}: {value}")
        else:
            lines.append(str(values))
        lines.append("")
    return "\n".join(lines)


def structural_export_zip(structural: dict[str, Any]) -> bytes:
    """Create the complete additive export bundle without changing legacy CSVs."""
    buffer = io.BytesIO()
    tables = structural.get("tables", {})
    expected = [
        "gap_frame_summary", "gap_object_features", "gap_size_spectrum", "gap_granulometry", "gap_chord_lengths", "gap_local_thickness", "gap_local_thickness_distribution",
        "gap_spatial_features", "gap_topology_features", "gap_multiscale_topology",
        "gap_network_features", "gap_network_degree_distribution", "gap_contour_modes", "gap_feature_robustness",
        "cellrich_frame_summary", "cellrich_object_features", "cellrich_feature_robustness",
        "dual_representation_qc", "registration_qc", "gap_tracks", "gap_events", "gap_dynamic_summary",
    ]
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in expected:
            table = tables.get(name, pd.DataFrame())
            archive.writestr(f"{name}.csv", table.to_csv(index=False).encode("utf-8-sig"))
        passport = structural.get("passport", {})
        archive.writestr("rbc_structural_passport.json", json.dumps(passport, indent=2, default=str).encode("utf-8"))
        archive.writestr("rbc_structural_passport.md", passport_markdown(passport).encode("utf-8"))
        archive.writestr("structural_configuration.json", json.dumps(structural.get("settings", {}), indent=2, default=str).encode("utf-8"))
        archive.writestr("future_mechanism_lab_interface.json", json.dumps(future_mechanism_lab_schema(), indent=2).encode("utf-8"))
    return buffer.getvalue()


def device_distance_features(mask: np.ndarray, device_mask: np.ndarray, band_edges_px: list[float]) -> pd.DataFrame:
    """Describe gap structure in distance bands from a user-supplied device mask."""
    gap = as_binary_mask(mask)
    device = as_binary_mask(device_mask)
    if device.shape != gap.shape:
        raise ValueError("Device and gap masks must have identical dimensions.")
    if not device.any():
        raise ValueError("Device mask is empty.")
    distance = ndi.distance_transform_edt(~device)
    edges = sorted({max(0.0, float(value)) for value in band_edges_px})
    if not edges or edges[0] != 0.0:
        edges.insert(0, 0.0)
    edges.append(np.inf)
    rows = []
    _thickness_summary, thickness_map, _ = local_thickness_analysis(gap)
    for lower, upper in zip(edges[:-1], edges[1:]):
        band = (distance >= lower) & (distance < upper) & (~device)
        if not band.any():
            continue
        local_gap = gap & band
        labels = measure.label(local_gap, connectivity=2)
        values = thickness_map[local_gap]
        topology = topology_features(local_gap)
        network, _skeleton, _network_edges = gap_network_features(
            local_gap,
            StaticConfig(min_object_area_px=1, min_network_object_area_px=1),
        )
        objects = gap_object_features(local_gap, StaticConfig(min_object_area_px=1))
        orientations = pd.to_numeric(objects.get("orientation_deg", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        weights = pd.to_numeric(objects.get("area_px2", pd.Series(dtype=float)), errors="coerce").fillna(0).to_numpy(dtype=float)
        if orientations.size and weights.size == orientations.size and float(weights.sum()) > 0:
            orientation_order = float(abs(np.average(np.exp(2j * np.radians(orientations)), weights=weights)))
            mean_orientation = float(np.degrees(0.5 * np.angle(np.average(np.exp(2j * np.radians(orientations)), weights=weights))))
        else:
            orientation_order = mean_orientation = np.nan
        tile_values = []
        image_rows, image_cols = gap.shape
        for row_parts in np.array_split(np.arange(image_rows), 4):
            for col_parts in np.array_split(np.arange(image_cols), 4):
                tile_band = band[np.ix_(row_parts, col_parts)]
                if int(tile_band.sum()) < 10:
                    continue
                tile_gap = gap[np.ix_(row_parts, col_parts)]
                tile_values.append(float(tile_gap[tile_band].mean()))
        tile_values_array = np.asarray(tile_values, dtype=float)
        heterogeneity = float(np.std(tile_values_array, ddof=1) / np.mean(tile_values_array)) if tile_values_array.size >= 2 and float(np.mean(tile_values_array)) > 0 else np.nan
        rows.append(
            {
                "distanceLower_px": lower,
                "distanceUpper_px": upper,
                "bandPixelCount": int(band.sum()),
                "gapAreaFractionWithinBand": float(local_gap.sum() / band.sum()),
                "gapObjectCount": int(labels.max()),
                "medianLocalThickness_px": float(np.median(values)) if values.size else np.nan,
                "spatialGapFractionCV_4x4": heterogeneity,
                "topologyConnectedComponents": topology["topologyConnectedComponents"],
                "topologyEnclosedHoles": topology["topologyEnclosedHoles"],
                "topologyEulerCharacteristic": topology["topologyEulerCharacteristic"],
                "gapNetworkSkeletonDensity": network["gapNetworkSkeletonDensity"],
                "gapNetworkBranchDensity": network["gapNetworkBranchDensity"],
                "gapObjectNematicOrder": orientation_order,
                "gapObjectMeanAxialOrientation_deg": mean_orientation,
                "bandBoundaryCaution": "objects/topology may be clipped at artificial distance-band boundaries",
                "units": "pixels; device geometry supplied by researcher",
            }
        )
    return pd.DataFrame(rows)


def estimate_device_influence_profile(
    table: pd.DataFrame,
    metric: str,
    minimum_frames_per_band: int = 3,
) -> dict[str, Any]:
    """Estimate a conservative return-to-far-field distance from repeated frames.

    A distance is reported only when every subsequent band's 95% mean CI overlaps
    the farthest band's CI.  This is an exploratory spatial profile, not a causal
    device-effect test.
    """
    required = {"distanceLower_px", "distanceUpper_px", metric}
    if table.empty or not required.issubset(table.columns):
        return {"supported": False, "reason": "required distance-band columns are missing", "profile": pd.DataFrame()}
    rows = []
    for (lower, upper), group in table.groupby(["distanceLower_px", "distanceUpper_px"], dropna=False, sort=True):
        values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size < minimum_frames_per_band:
            continue
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else np.nan
        half_width = 1.96 * sem if np.isfinite(sem) else np.nan
        rows.append(
            {
                "distanceLower_px": float(lower),
                "distanceUpper_px": float(upper),
                "metric": metric,
                "mean": mean,
                "ci95Lower": mean - half_width,
                "ci95Upper": mean + half_width,
                "nFrames": int(values.size),
            }
        )
    profile = pd.DataFrame(rows).sort_values("distanceLower_px", ignore_index=True) if rows else pd.DataFrame()
    if len(profile) < 3:
        return {"supported": False, "reason": "at least three populated distance bands with repeated frames are required", "profile": profile}
    far = profile.iloc[-1]
    influence_distance = np.nan
    for index in range(len(profile)):
        remaining = profile.iloc[index:]
        overlap = (remaining["ci95Upper"] >= far["ci95Lower"]) & (remaining["ci95Lower"] <= far["ci95Upper"])
        if bool(overlap.all()):
            influence_distance = float(profile.iloc[index]["distanceLower_px"])
            break
    if not np.isfinite(influence_distance):
        return {"supported": False, "reason": "no stable return to the far-field 95% confidence interval was detected", "profile": profile}
    return {
        "supported": True,
        "influenceDistance_px": influence_distance,
        "reason": "first band after which all 95% mean confidence intervals overlap the far-field interval",
        "profile": profile,
        "caution": "exploratory association; device geometry and repeated frames do not establish causality",
    }


def compare_bulk_measurements(structural_summary: pd.DataFrame, bulk: pd.DataFrame, sample_column_structural: str, sample_column_bulk: str) -> dict[str, pd.DataFrame]:
    """Merge optional bulk data and return exploratory rank correlations."""
    merged = structural_summary.merge(bulk, left_on=sample_column_structural, right_on=sample_column_bulk, how="inner", suffixes=("_structural", "_bulk"))
    numeric = merged.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for i, left in enumerate(numeric):
        for right in numeric[i + 1 :]:
            valid = merged[[left, right]].dropna()
            if len(valid) < 3:
                continue
            rho, p_value = spearmanr(valid[left], valid[right])
            rows.append({"metricA": left, "metricB": right, "spearmanRho": float(rho), "pValue": float(p_value), "n": len(valid)})
    return {"matched": merged, "correlations": pd.DataFrame(rows)}


def find_similar_score_different_structure(sample_table: pd.DataFrame, score_columns: list[str], structural_columns: list[str]) -> pd.DataFrame:
    """Rank sample pairs close in existing-score space but far structurally."""
    required = [column for column in score_columns + structural_columns if column in sample_table]
    if len([c for c in score_columns if c in sample_table]) == 0 or len([c for c in structural_columns if c in sample_table]) == 0:
        return pd.DataFrame()
    data = sample_table.copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        median = data[column].median()
        mad = float(np.median(np.abs(data[column].dropna().to_numpy(dtype=float) - median))) if data[column].notna().any() else np.nan
        scale = mad if np.isfinite(mad) and mad > 1e-12 else data[column].std(ddof=0)
        data[f"z__{column}"] = (data[column] - median) / scale if np.isfinite(scale) and scale > 1e-12 else 0.0
    id_col = next((column for column in ["sampleName", "sampleID", "filename"] if column in data), None)
    rows = []
    scores = [f"z__{c}" for c in score_columns if f"z__{c}" in data]
    structures = [f"z__{c}" for c in structural_columns if f"z__{c}" in data]
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            score_delta = data.loc[[i, j], scores].dropna(axis=1)
            structure_delta = data.loc[[i, j], structures].dropna(axis=1)
            if score_delta.shape[1] == 0 or structure_delta.shape[1] == 0:
                continue
            score_distance = float(np.linalg.norm(score_delta.iloc[0] - score_delta.iloc[1]))
            structural_distance = float(np.linalg.norm(structure_delta.iloc[0] - structure_delta.iloc[1]))
            rows.append(
                {
                    "sampleA": data.iloc[i][id_col] if id_col else i,
                    "sampleB": data.iloc[j][id_col] if id_col else j,
                    "existingScoreDistance": score_distance,
                    "structuralDistance": structural_distance,
                    "discoveryRankScore": float(structural_distance / (score_distance + 0.25)),
                    "interpretation": "exploratory pair; no automatic biological significance",
                }
            )
    return pd.DataFrame(rows).sort_values("discoveryRankScore", ascending=False, ignore_index=True) if rows else pd.DataFrame()


def future_mechanism_lab_schema() -> dict[str, Any]:
    return {
        "status": "architecture_only",
        "researchNotice": "Mechanism inference is not validated from current imaging data.",
        "acceptedFutureFields": {
            "modelType": "string",
            "parameters": "object",
            "protocol": "object",
            "predictedTimeCurves": "table",
            "predictedStructuralDescriptors": "table",
            "predictedBreakupRecovery": "table",
            "uncertainty": "object",
        },
        "candidateModelLabels": ["depletion-like", "immobile-bridging-like", "mobile-bridging-like", "mixed"],
        "classificationEnabled": False,
    }
