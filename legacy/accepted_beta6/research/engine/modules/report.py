"""Short report generation for MicroStructAI MVP."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .matlab_comparison import matlab_comparison_report_lines


MODE_NOTES = {
    "RBC Gap Mode": (
        "White mask pixels represent segmented plasma gaps. RBC-rich aggregated regions remain dark/background. "
        "Startup QC avoids interpreting preprocessing-amplified early texture as mature plasma-gap morphology."
    ),
    "Object Mode": "White mask pixels represent segmented cells, particles, or other target objects.",
    "Ground-Truth Mask Mode": "Metrics are extracted from uploaded manual masks or filled outlines; optional Dice/IoU compare automatic segmentation to ground truth.",
}


def _numeric_series_from_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    """Return numeric values for a column as a Series, tolerating scalars."""
    if isinstance(df, pd.DataFrame):
        if column_name not in df.columns:
            return pd.Series(dtype="float64")
        data = df.loc[:, column_name]
    else:
        data = df

    if isinstance(data, pd.DataFrame):
        raw_values = data.to_numpy().ravel()
    elif isinstance(data, pd.Series):
        raw_values = data.to_numpy()
    else:
        raw_values = np.asarray([data])

    return pd.to_numeric(pd.Series(raw_values), errors="coerce").replace([np.inf, -np.inf], np.nan)


def _metric_summary(df: pd.DataFrame, metrics: list[str]) -> list[str]:
    lines: list[str] = []
    for metric in metrics:
        values = _numeric_series_from_column(df, metric).dropna()
        if values.empty:
            continue
        mean_value = safe_mean(df, metric)
        if not np.isfinite(mean_value):
            continue
        lines.append(f"- {metric}: mean {mean_value:.4g}, median {values.median():.4g}, range {values.min():.4g} to {values.max():.4g}")
    return lines


def safe_mean(df: pd.DataFrame, column_name: str) -> float:
    """Return a finite numeric column mean, or NaN when unavailable."""
    values = _numeric_series_from_column(df, column_name)
    if values.empty:
        return np.nan
    finite_values = values[np.isfinite(values.to_numpy(dtype=float))]
    if finite_values.empty:
        return np.nan
    return float(finite_values.mean())


def _rbc_main_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return RBC rows selected for headline summaries."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if "usedInSummary" in df.columns:
        used = df["usedInSummary"].fillna(False).astype(bool)
    elif "usedForMainSummary" in df.columns:
        used = df["usedForMainSummary"].fillna(False).astype(bool)
    elif "includedInMainSummary" in df.columns:
        used = df["includedInMainSummary"].fillna(False).astype(bool)
    else:
        return df
    return df.loc[used].copy()


def compact_frame_ranges(frame_numbers, max_items: int = 30) -> str:
    """Return compact frame ranges such as '1, 5, 7-9' with optional shortening."""
    values: list[int] = []
    iterable = [] if frame_numbers is None else frame_numbers
    for value in iterable:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            values.append(int(round(numeric)))
    values = sorted(set(values))
    if not values:
        return "none"

    total_count = len(values)
    displayed_values = values[:max_items]
    ranges: list[str] = []
    start = displayed_values[0]
    previous = displayed_values[0]
    for value in displayed_values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")

    text = ", ".join(ranges)
    remaining = total_count - len(displayed_values)
    if remaining > 0:
        text += f" ... ({remaining} more frames)"
    return text


def rbc_included_excluded_frame_numbers(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Return original 1-based frame numbers included/excluded from the RBC main summary."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return [], []
    frame_col = "originalImageIndex_1based" if "originalImageIndex_1based" in df.columns else "analysisIndex_1based"
    if frame_col not in df.columns:
        return [], []
    if "usedInSummary" in df.columns:
        included_mask = df["usedInSummary"].fillna(False).astype(bool)
    elif "includedInMainSummary" in df.columns:
        included_mask = df["includedInMainSummary"].fillna(False).astype(bool)
    elif "usedForMainSummary" in df.columns:
        included_mask = df["usedForMainSummary"].fillna(False).astype(bool)
    else:
        included_mask = pd.Series(True, index=df.index)
    frames = pd.to_numeric(df[frame_col], errors="coerce")
    included = frames.loc[included_mask & frames.notna()].round().astype(int).tolist()
    excluded = frames.loc[(~included_mask) & frames.notna()].round().astype(int).tolist()
    return included, excluded


def stable_start_frame_text(qc_summary: dict) -> str:
    """Return display text for the first stable contiguous reliable run."""
    try:
        reliable_used = float(qc_summary.get("reliableFramesUsedForMainSummary", np.nan))
    except (TypeError, ValueError):
        reliable_used = np.nan
    try:
        min_stable = float(qc_summary.get("minConsecutiveReliableFramesForStableWindow", qc_summary.get("stableRunLengthRequired", 5)))
    except (TypeError, ValueError):
        min_stable = 5.0
    stable_value = qc_summary.get("suggestedStableStartFrame", np.nan)
    stable_ok = (
        bool(qc_summary.get("stableReliableWindowDetected", False))
        and np.isfinite(reliable_used)
        and reliable_used >= min_stable
    )
    if not stable_ok:
        return "Not available / no stable reliable window detected"
    try:
        numeric_stable = float(stable_value)
    except (TypeError, ValueError):
        return "Not available / no stable reliable window detected"
    if not np.isfinite(numeric_stable):
        return "Not available / no stable reliable window detected"
    return str(int(round(numeric_stable)))


def _segmentation_lines(mode: str, segmentation_params: dict) -> list[str]:
    """Summarize the active segmentation settings."""
    if mode == "RBC Gap Mode":
        return [
            "- Method: seed/grow morphological reconstruction of plasma gaps",
            f"- Preset used: {segmentation_params.get('cameraPresetName', 'Manual/current controls')}",
            f"- Overlay style: {segmentation_params.get('overlay_style', 'n/a')}",
            f"- Overlay background: {segmentation_params.get('overlay_background', 'n/a')}",
            f"- Boundary line width: {segmentation_params.get('boundaryLineWidth', 'n/a')}",
            f"- LF/HF method: {segmentation_params.get('lf_hf_method', 'n/a')}",
            f"- RBC segmentation input source: {segmentation_params.get('rbcSegmentationInputSource', 'MATLAB Iraw01 processed grayscale')}",
            "- MATLAB Raw/debug panel note: the segmentation input is processed analysis grayscale, not necessarily untouched camera raw.",
            f"- RBC gap safety profile: {segmentation_params.get('rbcGapSafetyMode', 'n/a')}",
            f"- Maximum single-gap artefact filter enabled: {segmentation_params.get('enableMaxGapAreaFilter', 'n/a')}",
            f"- maxGapAreaFrac: {segmentation_params.get('maxGapAreaFrac', 'n/a')}",
            f"- cropFrac: {segmentation_params.get('cropFrac', 'n/a')}",
            f"- preserveSpatialMeanIntensity: {segmentation_params.get('preserveSpatialMeanIntensity', 'n/a')}",
            f"- Iraw01 bgSigma / pLow / pHigh: {segmentation_params.get('ximg_bgSigma', 'n/a')} / {segmentation_params.get('ximg_pLow', 'n/a')} / {segmentation_params.get('ximg_pHigh', 'n/a')}",
            f"- useMedian / medianK: {segmentation_params.get('useMedian', 'n/a')} / {segmentation_params.get('medianK', 'n/a')}",
            f"- sigmaIll: {segmentation_params.get('sigmaIll', 'n/a')}",
            f"- wzMadFloor / wzSmoothSigma: {segmentation_params.get('wzMadFloor', 'n/a')} / {segmentation_params.get('wzSmoothSigma', 'n/a')}",
            f"- trackDark: {segmentation_params.get('trackDark', 'n/a')}",
            f"- kHi / kLo: {segmentation_params.get('kHi', 'n/a')} / {segmentation_params.get('kLo', 'n/a')}",
            f"- minAreaPx: {segmentation_params.get('minAreaPx', 'n/a')}",
            f"- closeRadius: {segmentation_params.get('closeRadius', 'n/a')}",
            f"- doFillHoles: {segmentation_params.get('doFillHoles', 'n/a')}",
            f"- clearBorder: {segmentation_params.get('clearBorder', 'n/a')}",
            f"- hugeGapFracThreshold: {segmentation_params.get('hugeGapFracThreshold', 'n/a')}",
            f"- largestToMedianGapRatioThreshold: {segmentation_params.get('largestToMedianGapRatioThreshold', 'n/a')}",
            f"- maxIlluminationCVForReliableMask: {segmentation_params.get('maxIlluminationCVForReliableMask', 'n/a')}",
            f"- Adaptive anti-merging repair: {segmentation_params.get('adaptiveAntiMergingRepair', 'n/a')}",
            f"- Camera intensity-drop correction: {segmentation_params.get('intensityCorrectionMode', 'n/a')}",
        ]
    if mode == "Ground-Truth Mask Mode":
        return [
            "- Method: morphology extraction from ground-truth masks",
            f"- Matching strategy: {segmentation_params.get('matching_strategy', 'n/a')}",
            f"- Mask type: {segmentation_params.get('mask_type', 'n/a')}",
            f"- Outline polarity: {segmentation_params.get('outline_polarity', 'n/a')}",
            f"- GT min object size: {segmentation_params.get('gt_min_object_size', 'n/a')} px",
            f"- Compare automatic segmentation: {segmentation_params.get('compare_auto', 'n/a')}",
            f"- Automatic segmentation method: {segmentation_params.get('auto_method', 'n/a')}",
        ]
    if mode == "Object Mode" and segmentation_params.get("enhancedObjectModeEnabled", False):
        return [
            "- Method: enhanced generic seed/grow morphological reconstruction",
            "- Mask meaning: generic detected structures; not RBC plasma gaps",
            f"- Overlay style: {segmentation_params.get('overlay_style', 'n/a')}",
            f"- Enhanced preprocessing style: {segmentation_params.get('enhancedObjectPreprocessingStyle', 'Current generic enhanced mode')}",
            f"- Analysis crop fraction: {segmentation_params.get('objectCropFrac', 'n/a')}",
            f"- preserveSpatialMeanIntensity: {segmentation_params.get('preserveSpatialMeanIntensity', 'n/a')}",
            f"- Processed-input background sigma: {segmentation_params.get('objectPreprocessBgSigma', 'n/a')}",
            f"- Segmentation target: {segmentation_params.get('objectSegmentationTarget', 'n/a')}",
            f"- useMedian / medianK: {segmentation_params.get('objectUseMedian', 'n/a')} / {segmentation_params.get('objectMedianK', 'n/a')}",
            f"- Illumination sigma: {segmentation_params.get('objectSigmaIll', 'n/a')}",
            f"- kHi / kLo: {segmentation_params.get('objectKHi', 'n/a')} / {segmentation_params.get('objectKLo', 'n/a')}",
            f"- Min structure area: {segmentation_params.get('objectMinAreaPx', 'n/a')} px",
            f"- Closing radius: {segmentation_params.get('objectCloseRadius', 'n/a')}",
            f"- Fill holes: {segmentation_params.get('objectFillHoles', 'n/a')}",
            f"- Clear border structures: {segmentation_params.get('objectClearBorder', 'n/a')}",
            f"- Maximum single-object artefact filter enabled: {segmentation_params.get('enableMaxObjectAreaFilter', 'n/a')}",
            f"- maxObjectAreaFrac: {segmentation_params.get('maxObjectAreaFrac', 'n/a')}",
            f"- Experimental generic composite scores enabled: {segmentation_params.get('experimentalCompositeScoresEnabled', 'n/a')}",
            f"- Camera intensity-drop correction: {segmentation_params.get('intensityCorrectionMode', 'n/a')}",
        ]
    return [
        f"- Overlay style: {segmentation_params.get('overlay_style', 'n/a')}",
        f"- Threshold method: {segmentation_params.get('threshold_method', 'n/a')}",
        f"- Polarity: {segmentation_params.get('polarity', 'n/a')}",
        f"- Min object size: {segmentation_params.get('min_object_size', 'n/a')} px",
        f"- Smoothing sigma: {segmentation_params.get('smoothing_sigma', 'n/a')}",
        f"- Fill holes: {segmentation_params.get('fill_holes', 'n/a')}",
        f"- preserveSpatialMeanIntensity: {segmentation_params.get('preserveSpatialMeanIntensity', 'n/a')}",
        f"- Camera intensity-drop correction: {segmentation_params.get('intensityCorrectionMode', 'n/a')}",
    ]


def generate_report(
    mode: str,
    metrics_df: pd.DataFrame,
    segmentation_params: dict,
    pca_result=None,
    warnings: list[str] | None = None,
    matlab_comparison_result=None,
) -> str:
    """Generate a compact Markdown report for the current analysis."""
    warnings = warnings or []
    input_method = segmentation_params.get("input_method", "Upload image files")
    files_loaded = segmentation_params.get("files_loaded_before_filtering", len(metrics_df))
    indexing_mode = segmentation_params.get("indexing_mode", "1-based indexing")
    skip_first_n = segmentation_params.get("skip_first_n", 0)
    range_start = segmentation_params.get("range_start_display", "n/a")
    range_end = segmentation_params.get("range_end_display", "n/a")
    lines: list[str] = [
        "# MicroStructAI MVP Report",
        "",
        "## Dataset Summary",
        f"- Analysis mode: {mode}",
        f"- Input method: {input_method}",
        f"- Files loaded: {files_loaded}",
        f"- Image range analysed: {range_start} to {range_end} using {indexing_mode}",
        f"- Images analysed: {len(metrics_df)}",
        f"- Skip first N images: {skip_first_n}" if int(skip_first_n or 0) > 0 else "- Skip first N images: 0",
        f"- Mode note: {MODE_NOTES.get(mode, 'Structural image analysis mode.')}",
        "",
        "## Segmentation Method",
    ]
    if mode == "RBC Gap Mode":
        lines.insert(-2, f"- RBC analysis window: {segmentation_params.get('rbc_analysis_window_mode', 'n/a')}")
    if mode == "RBC Gap Mode" and segmentation_params.get("matlab_window_enabled", False):
        first_time = segmentation_params.get("matlab_first_time_analysed", np.nan)
        last_time = segmentation_params.get("matlab_last_time_analysed", np.nan)
        lines.extend(
            [
                "- MATLAB analysis-window compatibility: enabled",
                f"- MATLAB fps: {segmentation_params.get('matlab_fps', 'n/a')}",
                f"- MATLAB podStart_s: {segmentation_params.get('matlab_podStart_s', 'n/a')}",
                f"- MATLAB analysisStart_s: {segmentation_params.get('matlab_analysisStart_s', 'n/a')}",
                f"- MATLAB-like quality filter: {'enabled' if segmentation_params.get('matlab_quality_filter_enabled', False) else 'disabled'}",
                f"- MATLAB frames after podStart: {segmentation_params.get('matlab_frames_after_podStart', 'n/a')}",
                f"- MATLAB frames after quality filter: {segmentation_params.get('matlab_frames_after_quality_filter', 'n/a')}",
                f"- MATLAB isAna-like frames: {segmentation_params.get('matlab_isAna_true_count', 'n/a')}",
                f"- First matlabTime_s analysed: {first_time:.4g}" if np.isfinite(first_time) else "- First matlabTime_s analysed: n/a",
                f"- Last matlabTime_s analysed: {last_time:.4g}" if np.isfinite(last_time) else "- Last matlabTime_s analysed: n/a",
                "- Manual Start/End image range was overridden by MATLAB compatibility window selection.",
            ]
        )
    elif mode == "RBC Gap Mode":
        lines.append("- MATLAB analysis-window compatibility: disabled")
    lines.extend(_segmentation_lines(mode, segmentation_params))
    if "intensityCorrectionMode" in metrics_df.columns:
        mode_value = str(metrics_df["intensityCorrectionMode"].dropna().iloc[0]) if metrics_df["intensityCorrectionMode"].dropna().size else str(segmentation_params.get("intensityCorrectionMode", "n/a"))
        applied = bool(metrics_df.get("intensityCorrectionApplied", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool).any())
        needed = bool(metrics_df.get("intensityCorrectionNeeded", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool).any())
        jump_values = _numeric_series_from_column(metrics_df, "nIlluminationJumpFrames").dropna()
        max_jump = safe_mean(metrics_df, "maxAbsLogJump")
        temporal_cv = safe_mean(metrics_df, "meanIntensityTemporalCV")
        sensitive_count = int(metrics_df.get("rawGainCorrectionSensitiveFlag", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool).sum())
        lines.extend(
            [
                "",
                "## Camera Intensity-Drop Correction",
                f"- Mode: {mode_value}",
                f"- Correction applied: {applied}",
                f"- Correction needed/detected: {needed}",
                f"- Detected intensity jump frames: {int(jump_values.iloc[0]) if not jump_values.empty and np.isfinite(jump_values.iloc[0]) else 'n/a'}",
                f"- Maximum log jump: {max_jump:.4g}" if np.isfinite(max_jump) else "- Maximum log jump: n/a",
                f"- Mean intensity temporal CV: {temporal_cv:.4g}" if np.isfinite(temporal_cv) else "- Mean intensity temporal CV: n/a",
                f"- Intensity-correction-sensitive mask frames: {sensitive_count}",
            ]
        )
    lines.extend(["", "## Key Metric Trends"])

    if mode == "Ground-Truth Mask Mode":
        priority_metrics = ["GT_nObjects", "GT_meanDiam", "GT_Xi_px", "GT_lacunarity16", "GT_skelDensity", "Dice_auto_vs_GT", "IoU_auto_vs_GT"]
    elif mode == "RBC Gap Mode":
        priority_metrics = [
            "gapAreaFrac",
            "gapNtotal",
            "gapMeanDiam_px",
            "gapLargestFrac",
            "Xi_interp_px",
            "Xi_px",
            "lacunarity16",
            "gapAreaSlope",
            "gapAreaSlope_R2",
            "LF_log10_LFoverHF",
            "LF_log10_LFoverHF_mvp",
            "LF_log10_LFoverHF_matlab",
            "SHI_tileCV",
            "SASI_QC_0_100",
            "AggIndex_QC_0_100",
            "GapScore_QC_0_100",
            "PTI_QC_0_100",
            "aggOccFrac",
            "aggLargestFrac",
            "conservativeGapAreaFrac",
            "podRemovedGapAreaFrac",
            "podAreaDelta",
            "podMode1Energy",
            "rawMeanIntensity",
            "correctedMeanIntensity",
            "rawCV",
            "correctedCV",
        ]
        if (segmentation_params.get("rbc_qc_summary", {}) or {}).get("scoresConfidence") == "low":
            priority_metrics = [metric for metric in priority_metrics if metric not in {"SASI_0_100", "AggIndex_0_100"}]
    elif mode == "Object Mode" and (
        segmentation_params.get("enhancedObjectModeEnabled", False)
        or segmentation_params.get("conservativeObjectModeEnabled", False)
        or segmentation_params.get("experimentalCompositeScoresEnabled", False)
    ):
        priority_metrics = [
            "areaFrac",
            "objectCount",
            "meanDiam_px",
            "medianDiam_px",
            "largestObjectFrac",
            "correlationLength_px",
            "lacunarity16",
            "sizeDistributionSlope",
            "sizeDistributionSlope_R2",
            "textureHeterogeneity_tileCV",
            "skeletonDensity",
            "branchDensity",
            "endpointDensity",
            "gSASI_0_100",
            "gAggIndex_0_100",
            "Object_Xi_mask",
            "Object_Xi_QC",
            "ObjectStructureScore_QC",
            "ObjectAggregationIndex_QC",
            "podObjectAreaDelta",
        ]
    else:
        priority_metrics = ["areaFrac", "nObjects", "meanDiam", "largestObjectFrac", "Xi_px", "lacunarity16", "skelDensity"]
    summary_source_df = _rbc_main_summary_df(metrics_df) if mode == "RBC Gap Mode" else metrics_df
    if mode == "RBC Gap Mode":
        lines.append(f"- RBC key metric summaries use {len(summary_source_df)} frame(s) from the selected main summary window.")
        lines.append("- Raw scores are calculated and exported for transparency. QC-aware scores are used for trusted summaries and interpretation.")
    summary = _metric_summary(summary_source_df, priority_metrics)
    lines.extend(summary if summary else ["- No finite metric values were available."])

    if mode == "Ground-Truth Mask Mode":
        mean_gt_count = safe_mean(metrics_df, "GT_nObjects")
        mean_gt_diam = safe_mean(metrics_df, "GT_meanDiam")
        mean_gt_xi = safe_mean(metrics_df, "GT_Xi_px")
        mean_dice = safe_mean(metrics_df, "Dice_auto_vs_GT")
        mean_iou = safe_mean(metrics_df, "IoU_auto_vs_GT")
        lines.extend(
            [
                "",
                "## Ground-Truth Summary",
                f"- Image/mask pairs analysed: {len(metrics_df)}",
                f"- Mean GT object count: {mean_gt_count:.4g}" if np.isfinite(mean_gt_count) else "- Mean GT object count: n/a",
                f"- Mean GT diameter: {mean_gt_diam:.4g}" if np.isfinite(mean_gt_diam) else "- Mean GT diameter: n/a",
                f"- Mean GT Xi: {mean_gt_xi:.4g}" if np.isfinite(mean_gt_xi) else "- Mean GT Xi: n/a",
            ]
        )
        if "Dice_auto_vs_GT" in metrics_df.columns and "IoU_auto_vs_GT" in metrics_df.columns:
            if np.isfinite(mean_dice) and np.isfinite(mean_iou):
                lines.extend([f"- Mean Dice_auto_vs_GT: {mean_dice:.4g}", f"- Mean IoU_auto_vs_GT: {mean_iou:.4g}"])
                if mean_dice < 0.60 or mean_iou < 0.50:
                    lines.append(
                        "- Low Dice/IoU indicates that automatic segmentation does not match the manual ground truth well. "
                        "In this case, ground-truth masks should be used for morphology extraction, and automatic segmentation should be treated as a future improvement."
                    )
            else:
                lines.append("- Automatic segmentation comparison was requested, but finite Dice/IoU values were not available.")
        else:
            lines.append("- Automatic segmentation was not compared with ground truth in this run.")

    if mode == "Object Mode" and (
        segmentation_params.get("enhancedObjectModeEnabled", False)
        or segmentation_params.get("conservativeObjectModeEnabled", False)
        or segmentation_params.get("experimentalCompositeScoresEnabled", False)
    ):
        low_resolution_values = metrics_df.get("lowResolutionRisk", pd.Series(dtype=object)).astype(str)
        confidence_values = metrics_df.get("segmentationConfidence", pd.Series(dtype=object)).astype(str)
        lines.extend(["", "## Object Mode Summary"])
        if segmentation_params.get("enhancedObjectModeEnabled", False):
            lines.extend(
                [
                    "- Enhanced Object Mode uses robust bright/dark structure segmentation to quantify generic morphology.",
                    "- The detected mask does not represent RBC plasma gaps.",
                    f"- Enhanced preprocessing style: {segmentation_params.get('enhancedObjectPreprocessingStyle', 'Current generic enhanced mode')}",
                    f"- Segmentation target requested: {segmentation_params.get('objectSegmentationTarget', 'n/a')}",
                ]
            )
        if segmentation_params.get("conservativeObjectModeEnabled", False):
            lines.extend(
                [
                    f"- Object analysis method: {segmentation_params.get('selectedObjectMethod', 'Conservative Object Mode')}",
                    "- Conservative/POD Object modes use generic no-POD/POD binary object masks for morphology and QC diagnostics.",
                    "- The selected final object mask drives the metrics table, CSV export, plots, and generic ObjectStructure/ObjectAggregation scores.",
                    "- These object-mode scores are generic mask-based structural/coarsening scores. They are not RBC-specific biomarkers.",
                ]
            )
        lines.extend(
            [
                f"- Low-resolution risks observed: {', '.join(sorted(set(low_resolution_values))) if not low_resolution_values.empty else 'n/a'}",
                f"- Segmentation confidence values observed: {', '.join(sorted(set(confidence_values))) if not confidence_values.empty else 'n/a'}",
                "- Low-resolution warning: results describe coarse structure only. Fine object count, boundary shape, and small-object size distribution may be unreliable.",
                "- BBBC-style datasets can validate generic segmentation/morphology behaviour in Object Mode or Ground-Truth Mask Mode. They do not validate RBC-specific SASI/AggIndex interpretation.",
            ]
        )
        if segmentation_params.get("experimentalCompositeScoresEnabled", False):
            lines.extend(
                [
                    "",
                    "## Experimental Generic Structure Scores",
                    f"- Mean gSASI_0_100: {safe_mean(metrics_df, 'gSASI_0_100'):.4g}"
                    if np.isfinite(safe_mean(metrics_df, "gSASI_0_100"))
                    else "- Mean gSASI_0_100: n/a",
                    f"- Mean gAggIndex_0_100: {safe_mean(metrics_df, 'gAggIndex_0_100'):.4g}"
                    if np.isfinite(safe_mean(metrics_df, "gAggIndex_0_100"))
                    else "- Mean gAggIndex_0_100: n/a",
                    "- Higher gSASI may indicate larger, less fragmented coarse structural organisation in this image set.",
                    "- The direction of interpretation depends on the sample type.",
                    "- For some materials, larger/fewer structures may indicate aggregation or coarsening; for others it may indicate collapse, merging, or segmentation artefact.",
                    "- These optional scores reuse the mathematical structure of the RBC aggregation scores, but they are not validated for this sample type. Interpret them only as exploratory image-structure indices.",
                    "- Do not interpret gSASI/gAggIndex as RBC aggregation scores.",
                ]
            )

    syl_columns = [
        ("meanI_syl", "Optical mean intensity"),
        ("meanI_syl_raw", "Optical mean intensity raw"),
        ("meanI_syl_corrected", "Optical mean intensity corrected"),
        ("cv_syl", "Optical CV"),
        ("cv_syl_raw", "Optical CV raw"),
        ("cv_syl_corrected", "Optical CV corrected"),
        ("maskSASI_syl", "Mask-SASI"),
        ("maskSASI_syl_raw", "Mask-SASI raw"),
        ("maskSASI_syl_QC", "Mask-SASI QC"),
        ("maskGapScore_syl", "Mask-GapScore"),
        ("maskGapScore_syl_raw", "Mask-GapScore raw"),
        ("maskGapScore_syl_QC", "Mask-GapScore QC"),
        ("maskAggIndex_syl", "Mask-AggIndex"),
        ("maskAggIndex_syl_raw", "Mask-AggIndex raw"),
        ("maskAggIndex_syl_QC", "Mask-AggIndex QC"),
        ("objectStructure_syl", "ObjectStructure"),
        ("objectStructure_syl_raw", "ObjectStructure raw"),
        ("objectStructure_syl_QC", "ObjectStructure QC"),
        ("objectAggregation_syl", "ObjectAggregation"),
        ("objectAggregation_syl_raw", "ObjectAggregation raw"),
        ("objectAggregation_syl_QC", "ObjectAggregation QC"),
        ("gSASI_syl", "gSASI"),
    ]
    available_syl = [item for item in syl_columns if f"{item[0]}_amplitude" in metrics_df.columns and np.isfinite(safe_mean(metrics_df, f"{item[0]}_amplitude"))]
    if available_syl:
        lines.extend(
            [
                "",
                "## Normalized Syllectogram Summary",
                "- Optical syllectograms are calculated from corrected grayscale image intensity and CV, not from the binary mask.",
                "- Mask-SASI/Mask-GapScore syllectograms are calculated from final selected binary mask metrics using mask-based Xi.",
                "- Object structure and aggregation syllectograms are generic mask-based structural/coarsening curves calculated from the final selected object mask.",
                "- Raw curves are shown for transparency. QC curves are used for trusted trend summaries and may contain gaps when frames fail reliability checks.",
                "- Raw intensity-based Xi is retained as a diagnostic only. Trusted structure syllectograms use mask-based Xi.",
                "- These normalized curves are validation/research trend descriptors and do not change the source scores or segmentation metrics.",
            ]
        )
        for prefix, label in available_syl:
            lines.extend(
                [
                    f"- {label} amplitude: {safe_mean(metrics_df, f'{prefix}_amplitude'):.4g}",
                    f"- {label} AUC: {safe_mean(metrics_df, f'{prefix}_AUC'):.4g}",
                    f"- {label} T50: {safe_mean(metrics_df, f'{prefix}_T50'):.4g}" if np.isfinite(safe_mean(metrics_df, f"{prefix}_T50")) else f"- {label} T50: n/a",
                    f"- {label} final value: {safe_mean(metrics_df, f'{prefix}_final_value'):.4g}",
                ]
            )

    fallback_summary = segmentation_params.get("startup_transient_fallback_summary", {}) or {}
    if segmentation_params.get("enableStartupTransientFallback", False) or fallback_summary:
        fallback_count = fallback_summary.get("nStartupFallbackFrames", 0)
        lines.extend(
            [
                "",
                "## Startup / Transient Fallback",
                f"- Startup/transient fallback enabled: {segmentation_params.get('enableStartupTransientFallback', False)}",
                f"- Startup/transient fallback frames: {fallback_count}",
                f"- Fraction fallback frames: {fallback_summary.get('fracStartupFallbackFrames', 'n/a')}",
                f"- Median startupRawCV: {fallback_summary.get('medianStartupRawCV', 'n/a')}",
                f"- Median startupGradientEnergy: {fallback_summary.get('medianStartupGradientEnergy', 'n/a')}",
                f"- Median startupFFT_LF_HF_ratio: {fallback_summary.get('medianStartupFFT_LF_HF_ratio', 'n/a')}",
                f"- Median startupStreakIndex: {fallback_summary.get('medianStartupStreakIndex', 'n/a')}",
                "- Startup/transient metrics are early-frame QC texture diagnostics calculated from raw/lightly corrected images.",
                "- They do not replace normal masks and are not used for SASI, AggIndex, normal object counts, or normal morphology scores.",
            ]
        )
        if int(fallback_count or 0) > 0:
            lines.append(
                "- Frames using fallback should be interpreted as startup/transient texture, not true aggregates, plasma gaps, or validated object morphology."
            )

    if mode == "RBC Gap Mode":
        qc_summary = segmentation_params.get("rbc_qc_summary", {}) or {}
        large_gap_summary = segmentation_params.get("large_gap_artefact_summary", {}) or {}
        trend_rhos = qc_summary.get("temporalTrendRhos", {}) or {}
        trend_text = ", ".join(
            f"{metric}: {value:.3g}" for metric, value in trend_rhos.items() if np.isfinite(value)
        )
        sample_qc_for_interpretation = qc_summary.get("sampleQC")
        if sample_qc_for_interpretation == "Reliable aggregation morphology" and qc_summary.get("scoresConfidence") != "low":
            aggregation_interpretation = "Reliable aggregation morphology"
        elif sample_qc_for_interpretation == "No reliable plasma-gap aggregation detected":
            aggregation_interpretation = "No reliable aggregation detected"
        else:
            aggregation_interpretation = "Low-confidence / weak pre-gap signal"
        included_frames, excluded_frames = rbc_included_excluded_frame_numbers(metrics_df)
        stable_start_text = stable_start_frame_text(qc_summary)
        frame_col = "originalImageIndex_1based" if "originalImageIndex_1based" in metrics_df.columns else "analysisIndex_1based"
        if frame_col in metrics_df.columns and "isolatedReliableFrame" in metrics_df.columns:
            isolated_mask = metrics_df["isolatedReliableFrame"].fillna(False).astype(bool)
            isolated_values = pd.to_numeric(metrics_df.loc[isolated_mask, frame_col], errors="coerce").dropna().round().astype(int).tolist()
        else:
            isolated_values = []
        included_col = "usedInSummary" if "usedInSummary" in metrics_df.columns else "includedInMainSummary"
        if frame_col in metrics_df.columns and {included_col, "isStartupReliableMorphology"}.issubset(metrics_df.columns):
            included_mask = metrics_df[included_col].fillna(False).astype(bool)
            startup_mask = metrics_df["isStartupReliableMorphology"].fillna(False).astype(bool)
            startup_excluded_mask = (~included_mask) & (~startup_mask | metrics_df.get("qcOutsideStableWindow", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool))
            startup_excluded_values = pd.to_numeric(metrics_df.loc[startup_excluded_mask, frame_col], errors="coerce").dropna().round().astype(int).tolist()
        else:
            startup_excluded_values = []
        stable_detected_text = "yes" if qc_summary.get("stableReliableWindowDetected", False) else "no"
        stable_rule_text = qc_summary.get("stableWindowRule", segmentation_params.get("stableWindowRule", "n/a"))
        startup_stable_frame = qc_summary.get("startupQCStableStartFrame", np.nan)
        first_reliable_morphology = qc_summary.get("firstReliableMorphologyFrame", np.nan)
        final_start_frame_used = qc_summary.get("finalStartFrameUsed", np.nan)
        startup_analysis_start = qc_summary.get("startupAnalysisStart_s", "Not used")
        auto_start_frame = qc_summary.get("autoStartFrame", qc_summary.get("autoStartFrame_POD", np.nan))
        auto_start_time = qc_summary.get("autoStartTime_s", np.nan)
        auto_start_original = qc_summary.get("autoStart_originalImageIndex", np.nan)
        auto_start_file = qc_summary.get("autoStart_originalFileName", "")
        try:
            peak_slope_value = float(qc_summary.get("peakSlope", np.nan))
        except (TypeError, ValueError):
            peak_slope_value = np.nan
        try:
            max_start_signal_value = float(qc_summary.get("maxStartSignalInSearchWindow", np.nan))
        except (TypeError, ValueError):
            max_start_signal_value = np.nan
        frames_before_start = len(metrics_df) - int(qc_summary.get("framesUsedForMainSummary", 0))
        lines.extend(
            [
                "",
                "## RBC QC / no-aggregation check",
                f"- sampleQC: {qc_summary.get('sampleQC', 'n/a')}",
                f"- Aggregation interpretation: {aggregation_interpretation}",
                f"- Total frames loaded: {qc_summary.get('totalFramesLoaded', files_loaded)}",
                f"- Frames used for QC: {qc_summary.get('framesUsedForQC', len(metrics_df))}",
                f"- Frames used for main summary: {qc_summary.get('framesUsedForMainSummary', 'n/a')}",
                f"- Main summary window: {qc_summary.get('mainSummaryWindow', segmentation_params.get('rbc_analysis_window_mode', 'n/a'))}",
                f"- Startup QC mode: {qc_summary.get('startupQCMode', segmentation_params.get('startupQCMode', 'n/a'))}",
                f"- Start-frame mode: {qc_summary.get('startFrameMode', segmentation_params.get('startFrameMode', 'n/a'))}",
                f"- autoStartMethod: {qc_summary.get('autoStartMethod', 'n/a')}",
                f"- autoStartFrame: {int(round(auto_start_frame)) if np.isfinite(auto_start_frame) else 'Not available'}",
                f"- autoStartTime_s: {auto_start_time:.4g}" if np.isfinite(auto_start_time) else "- autoStartTime_s: Not available",
                f"- autoStart_originalImageIndex: {int(round(auto_start_original)) if np.isfinite(auto_start_original) else 'Not available'}",
                f"- autoStart_originalFileName: {auto_start_file if auto_start_file else 'Not available'}",
                f"- peakSlope: {peak_slope_value:.4g}" if np.isfinite(peak_slope_value) else "- peakSlope: n/a",
                f"- maxStartSignalInSearchWindow: {max_start_signal_value:.4g}" if np.isfinite(max_start_signal_value) else "- maxStartSignalInSearchWindow: n/a",
                f"- no-start safety triggered: {qc_summary.get('noStartSafetyTriggered', 'n/a')}",
                f"- Frames used for main summary: {int(round(auto_start_frame)) if np.isfinite(auto_start_frame) else 'Not available'} to final frame"
                if int(qc_summary.get("framesUsedForMainSummary", 0) or 0) > 0
                else "- Frames used for main summary: none",
                f"- Frames before autoStartFrame excluded from main summary: {max(0, frames_before_start)}",
                f"- Manual lower bound used: {qc_summary.get('startupManualLowerBoundUsed', 'n/a')}",
                f"- analysisStart_s lower bound: {startup_analysis_start}",
                f"- autoStartSmoothWin_s: {qc_summary.get('autoStartSmoothWin_s', segmentation_params.get('autoStartSmoothWin_s', 'n/a'))}",
                f"- autoStartMinDevelopedSignal: {qc_summary.get('autoStartMinDevelopedSignal', segmentation_params.get('autoStartMinDevelopedSignal', 'n/a'))}",
                f"- autoStartSlopeFrac: {qc_summary.get('autoStartSlopeFrac', segmentation_params.get('autoStartSlopeFrac', 'n/a'))}",
                f"- autoStartEarlyScaleFrac: {qc_summary.get('autoStartEarlyScaleFrac', segmentation_params.get('autoStartEarlyScaleFrac', 'n/a'))}",
                f"- autoStartSearchFrac: {qc_summary.get('autoStartSearchFrac', segmentation_params.get('autoStartSearchFrac', 'n/a'))}",
                f"- autoStartMinPeakSlope: {qc_summary.get('autoStartMinPeakSlope', segmentation_params.get('autoStartMinPeakSlope', 'n/a'))}",
                f"- autoStartMinMaxSignal: {qc_summary.get('autoStartMinMaxSignal', segmentation_params.get('autoStartMinMaxSignal', 'n/a'))}",
                f"- Stable-window rule used: {stable_rule_text}",
                f"- Minimum consecutive reliable frames required: {qc_summary.get('minConsecutiveReliableFramesForStableWindow', qc_summary.get('stableRunLengthRequired', 'n/a'))}",
                f"- minSHI_tileCV_forReliable: {qc_summary.get('minSHI_tileCV_forReliable', segmentation_params.get('minSHI_tileCV_forReliable', 'n/a'))}",
                f"- Sparse-mask morphology filter enabled: {qc_summary.get('applySparseMaskMorphologyFilter', segmentation_params.get('applySparseMaskMorphologyFilter', 'n/a'))}",
                f"- minReliableMaskAreaFracForMorphology: {qc_summary.get('minReliableMaskAreaFracForMorphology', segmentation_params.get('minReliableMaskAreaFracForMorphology', 'n/a'))}",
                f"- minReliableObjectsForMorphology: {qc_summary.get('minReliableObjectsForMorphology', segmentation_params.get('minReliableObjectsForMorphology', 'n/a'))}",
                f"- Sparse-mask morphology filtered frames: {qc_summary.get('sparseMaskMorphologyFilteredFrames', qc_summary.get('sparseMaskMorphologyFilteredFrameCount', 'n/a'))}",
                f"- Stable reliable window detected: {stable_detected_text}",
                f"- firstReliableMorphologyFrame: {int(round(first_reliable_morphology)) if np.isfinite(first_reliable_morphology) else 'Not available'}",
                f"- startupQCStableStartFrame: {int(round(startup_stable_frame)) if np.isfinite(startup_stable_frame) else 'Not available'}",
                f"- finalStartFrameUsed: {int(round(final_start_frame_used)) if np.isfinite(final_start_frame_used) else 'Not available'}",
                f"- QC selected frame 1: {qc_summary.get('startupQCSelectedFrame1', 'n/a')}",
                f"- Automatic startup QC moved later than manual analysisStart_s: {qc_summary.get('startupQCMovedLaterThanManualLowerBound', 'n/a')}",
                f"- Stable window start frame: {stable_start_text}",
                f"- Suggested stable start frame: {stable_start_text}",
                f"- Frame numbers used for main summary: {compact_frame_ranges(included_frames)}",
                f"- Frame numbers excluded by QC: {compact_frame_ranges(excluded_frames)}",
                f"- Frame numbers excluded by startup QC: {compact_frame_ranges(startup_excluded_values)}",
                f"- Isolated reliable frames excluded: {compact_frame_ranges(isolated_values)}",
                f"- Exclude pre-start frames: {qc_summary.get('excludePreStartFramesFromTrustedAnalysis', 'n/a')}",
                f"- Exclude bad post-start QC frames: {qc_summary.get('excludeBadPostStartQCFramesFromTrustedAnalysis', 'n/a')}",
                f"- Used in trusted summary: {qc_summary.get('usedInSummaryFrames', qc_summary.get('usedInSummaryFrameCount', 'n/a'))}",
                f"- Low-confidence frames excluded: {qc_summary.get('lowConfidenceFramesExcluded', 'n/a')}",
                f"- Weak/pre-gap frames excluded: {qc_summary.get('weakPreGapFramesExcluded', 'n/a')}",
                f"- Possible artefact frames excluded: {qc_summary.get('possibleArtifactFramesExcluded', 'n/a')}",
                f"- QC diagnostic, SHI below threshold: {qc_summary.get('framesFailedSHI_tileCV', 'n/a')}",
                f"- QC diagnostic, low contrast/separation: {qc_summary.get('framesFailedLowContrastSeparation', 'n/a')}",
                f"- QC diagnostic, implausible mask: {qc_summary.get('framesFailedImplausibleMask', 'n/a')}",
                f"- QC diagnostic, outside stable window: {qc_summary.get('framesOutsideStableWindow', 'n/a')}",
                f"- QC diagnostic, huge-gap frames: {qc_summary.get('hugeGapFrameCount', 'n/a')}",
                f"- QC diagnostic, hardBadMaskFrames: {qc_summary.get('hardBadMaskFrames', qc_summary.get('hardBadMaskFrameCount', 'n/a'))}",
                f"- QC diagnostic, pre-start frames detected: {qc_summary.get('preStartFramesDetected', qc_summary.get('preStartFrameCount', 'n/a'))}",
                f"- QC diagnostic, pre-start frames excluded: {qc_summary.get('beforeStartExcludedFrames', qc_summary.get('beforeStartExcludedFrameCount', 'n/a'))}",
                f"- QC diagnostic, post-start QC warning frames: {qc_summary.get('postStartQCWarningFrames', qc_summary.get('postStartQCWarningFrameCount', 'n/a'))}",
                f"- QC diagnostic, post-start QC-excluded frames: {qc_summary.get('postStartQCExcludedFrames', qc_summary.get('postStartQCExcludedFrameCount', 'n/a'))}",
                f"- QC diagnostic, lowConfidenceFrames: {qc_summary.get('lowConfidenceFrames', qc_summary.get('lowConfidenceFrameCount', 'n/a'))}",
                f"- QC diagnostic, sparse-mask morphology filtered frames: {qc_summary.get('sparseMaskMorphologyFilteredFrames', qc_summary.get('sparseMaskMorphologyFilteredFrameCount', 'n/a'))}",
                f"- QC diagnostic, warning-only trusted frames: {qc_summary.get('visualWarningOnlyFrames', qc_summary.get('visualWarningOnlyFrameCount', 'n/a'))}",
                f"- QC diagnostic, trustedForScore frames: {qc_summary.get('trustedForScoreFrames', qc_summary.get('trustedScoreFrameCount', 'n/a'))}",
                f"- QC diagnostic, usedInSummaryFrames: {qc_summary.get('usedInSummaryFrames', qc_summary.get('usedInSummaryFrameCount', 'n/a'))}",
                f"- QC diagnostic, unstable segmentation frames: {qc_summary.get('unstableSegmentationFrameCount', 'n/a')}",
                f"- Mean maskReliabilityScore: {qc_summary.get('meanMaskReliabilityScore', 'n/a')}",
                f"- Minimum maskReliabilityScore: {qc_summary.get('minMaskReliabilityScore', 'n/a')}",
                f"- Post-start bad-mask QC exclusion enabled: {qc_summary.get('excludeBadPostStartQCFramesFromTrustedAnalysis', qc_summary.get('postStartBadFrameExclusionEnabled', 'n/a'))}",
                f"- maskReliabilityMinForSummary: {qc_summary.get('maskReliabilityMinForSummary', segmentation_params.get('maskReliabilityMinForSummary', 'n/a'))}",
                f"- Frames excluded by mask QC: {qc_summary.get('nFramesExcludedByMaskQC', 'n/a')}",
                f"- Percent frames excluded by mask QC: {qc_summary.get('percentFramesExcludedByMaskQC', 'n/a')}",
                f"- sampleMaskQCRisk: {qc_summary.get('sampleMaskQCRisk', 'n/a')}",
                f"- Adaptive anti-merging repair enabled: {qc_summary.get('adaptiveRepairEnabled', segmentation_params.get('adaptiveAntiMergingRepair', 'n/a'))}",
                f"- Adaptive repairs triggered: {qc_summary.get('adaptiveRepairTriggeredCount', 'n/a')}",
                f"- Adaptive repairs kept: {qc_summary.get('adaptiveRepairKeptCount', 'n/a')}",
                f"- Scores confidence: {qc_summary.get('scoresConfidence', 'n/a')}",
                f"- Reliable frames: {qc_summary.get('reliableFrameCount', 'n/a')}",
                f"- Weak/pre-gap frames: {qc_summary.get('weakPreGapFrameCount', 'n/a')}",
                f"- Possible artefact frames: {qc_summary.get('possibleArtifactFrameCount', 'n/a')}",
                f"- Reliable frame percentage: {qc_summary.get('reliableFramePercent', np.nan):.4g}%"
                if np.isfinite(qc_summary.get("reliableFramePercent", np.nan))
                else "- Reliable frame percentage: n/a",
                f"- Temporal trend rhos: {trend_text if trend_text else 'n/a'}",
                f"- Recommendation: {qc_summary.get('recommendation', 'n/a')}",
                "- Early low-contrast frames may produce apparent gap masks after background subtraction and contrast normalization. "
                "The first-rise detector waits until raw-image brightness/contrast evidence and aggregate-network topology indicate a developed structural signal. "
                "The main summary then uses all frames from the detected start frame onward.",
                "- Startup QC is intended to avoid interpreting preprocessing-amplified early texture as mature plasma-gap morphology.",
                "- Trend plots may show all analysed frames for QC purposes, while main summary values are calculated only from included reliable frames.",
                "- maskReliabilityScore is a 0-100 diagnostic where 80-100 is generally reliable, 50-80 is usable with caution, and below 50 is low confidence.",
            ]
        )
        if qc_summary.get("postStartBadFrameExclusionEnabled", False):
            lines.append(
                "- When post-start QC exclusion is enabled, low-confidence masks are excluded from summary statistics but retained in the exported per-frame table."
            )
        if qc_summary.get("adaptiveRepairEnabled", False):
            lines.append(
                "- Adaptive repair was enabled; suspicious merged gap masks were re-segmented with conservative morphology settings and accepted only when QC improved."
            )
        rbc_method = segmentation_params.get("rbcAnalysisMethod", segmentation_params.get("rbcMode", "Standard RBC Gap Mode"))
        lines.extend(
            [
                "",
                "## RBC Method / Final Mask Source",
                f"- selectedRBCMethod: {metrics_df.get('selectedRBCMethod', pd.Series(['n/a'])).dropna().astype(str).iloc[0] if 'selectedRBCMethod' in metrics_df.columns and not metrics_df.get('selectedRBCMethod', pd.Series(dtype=object)).dropna().empty else rbc_method}",
                f"- finalPreprocessingSource: {metrics_df.get('finalPreprocessingSource', pd.Series(['n/a'])).dropna().astype(str).iloc[0] if 'finalPreprocessingSource' in metrics_df.columns and not metrics_df.get('finalPreprocessingSource', pd.Series(dtype=object)).dropna().empty else 'n/a'}",
                f"- finalMaskSource: {metrics_df.get('finalMaskSource', pd.Series(['n/a'])).dropna().astype(str).iloc[0] if 'finalMaskSource' in metrics_df.columns and not metrics_df.get('finalMaskSource', pd.Series(dtype=object)).dropna().empty else 'n/a'}",
                f"- finalAreaFracSource: {metrics_df.get('finalAreaFracSource', pd.Series(['n/a'])).dropna().astype(str).iloc[0] if 'finalAreaFracSource' in metrics_df.columns and not metrics_df.get('finalAreaFracSource', pd.Series(dtype=object)).dropna().empty else 'n/a'}",
                f"- Mean areaFracFinal: {safe_mean(metrics_df, 'areaFracFinal'):.4g}" if np.isfinite(safe_mean(metrics_df, "areaFracFinal")) else "- Mean areaFracFinal: n/a",
                f"- Mean nObjectsFinal: {safe_mean(metrics_df, 'nObjectsFinal'):.4g}" if np.isfinite(safe_mean(metrics_df, "nObjectsFinal")) else "- Mean nObjectsFinal: n/a",
            ]
        )
        if rbc_method != "Standard RBC Gap Mode":
            conservative_reliable = (
                metrics_df.get("conservativeMaskReliable", pd.Series(False, index=metrics_df.index))
                .fillna(False)
                .astype(bool)
                .sum()
                if not metrics_df.empty
                else 0
            )
            pod_sensitive = (
                metrics_df.get("podSensitiveFlag", pd.Series(False, index=metrics_df.index))
                .fillna(False)
                .astype(bool)
                .sum()
                if "podSensitiveFlag" in metrics_df.columns
                else 0
            )
            pod_active = (
                metrics_df.get("podProcessingEnabled", pd.Series(False, index=metrics_df.index))
                .fillna(False)
                .astype(bool)
                .any()
                if "podProcessingEnabled" in metrics_df.columns
                else False
            )
            lines.extend(
                [
                    "",
                    "## Conservative RBC / POD-QC Summary",
                    f"- RBC analysis method: {rbc_method}",
                    "- Main mask source: corrected no-POD image" if rbc_method != "Experimental POD-removed RBC Mode" or not pod_active else "- Main mask source: experimental POD-removed image",
                    f"- selectedRBCMethod: {metrics_df.get('selectedRBCMethod', pd.Series(['n/a'])).dropna().astype(str).iloc[0] if 'selectedRBCMethod' in metrics_df.columns and not metrics_df.get('selectedRBCMethod', pd.Series(dtype=object)).dropna().empty else rbc_method}",
                    f"- finalPreprocessingSource: {metrics_df.get('finalPreprocessingSource', pd.Series(['n/a'])).dropna().astype(str).iloc[0] if 'finalPreprocessingSource' in metrics_df.columns and not metrics_df.get('finalPreprocessingSource', pd.Series(dtype=object)).dropna().empty else 'n/a'}",
                    f"- finalMaskSource: {metrics_df.get('finalMaskSource', pd.Series(['n/a'])).dropna().astype(str).iloc[0] if 'finalMaskSource' in metrics_df.columns and not metrics_df.get('finalMaskSource', pd.Series(dtype=object)).dropna().empty else 'n/a'}",
                    f"- finalAreaFracSource: {metrics_df.get('finalAreaFracSource', pd.Series(['n/a'])).dropna().astype(str).iloc[0] if 'finalAreaFracSource' in metrics_df.columns and not metrics_df.get('finalAreaFracSource', pd.Series(dtype=object)).dropna().empty else 'n/a'}",
                    f"- POD processing active: {bool(pod_active)}",
                    f"- removeModes: {metrics_df.get('removeModes', pd.Series(['n/a'])).dropna().astype(str).iloc[0] if 'removeModes' in metrics_df.columns and not metrics_df.get('removeModes', pd.Series(dtype=object)).dropna().empty else 'n/a'}",
                    f"- Mean areaFracFinal: {safe_mean(metrics_df, 'areaFracFinal'):.4g}" if np.isfinite(safe_mean(metrics_df, "areaFracFinal")) else "- Mean areaFracFinal: n/a",
                    f"- Mean nObjectsFinal: {safe_mean(metrics_df, 'nObjectsFinal'):.4g}" if np.isfinite(safe_mean(metrics_df, "nObjectsFinal")) else "- Mean nObjectsFinal: n/a",
                    f"- Mean standardAreaFrac: {safe_mean(metrics_df, 'standardAreaFrac'):.4g}" if np.isfinite(safe_mean(metrics_df, "standardAreaFrac")) else "- Mean standardAreaFrac: n/a",
                    f"- Mean conservativeNoPODAreaFrac: {safe_mean(metrics_df, 'conservativeNoPODAreaFrac'):.4g}" if np.isfinite(safe_mean(metrics_df, "conservativeNoPODAreaFrac")) else "- Mean conservativeNoPODAreaFrac: n/a",
                    f"- Conservative reliable frames: {int(conservative_reliable)}",
                    f"- POD-sensitive / uncertain frames: {int(pod_sensitive) if pod_active else 0}",
                    f"- Mean conservativeGapAreaFrac: {safe_mean(metrics_df, 'conservativeGapAreaFrac'):.4g}"
                    if np.isfinite(safe_mean(metrics_df, "conservativeGapAreaFrac"))
                    else "- Mean conservativeGapAreaFrac: n/a",
                ]
            )
            if pod_active:
                lines.extend(
                    [
                        f"- Mean podRemovedGapAreaFrac: {safe_mean(metrics_df, 'podRemovedGapAreaFrac'):.4g}"
                        if np.isfinite(safe_mean(metrics_df, "podRemovedGapAreaFrac"))
                        else "- Mean podRemovedGapAreaFrac: n/a",
                        f"- Mean podAreaDelta: {safe_mean(metrics_df, 'podAreaDelta'):.4g}" if np.isfinite(safe_mean(metrics_df, "podAreaDelta")) else "- Mean podAreaDelta: n/a",
                        f"- Mean podMode1Energy: {safe_mean(metrics_df, 'podMode1Energy'):.4g}" if np.isfinite(safe_mean(metrics_df, "podMode1Energy")) else "- Mean podMode1Energy: n/a",
                        "- POD-removed residuals are diagnostic unless Experimental POD-removed RBC Mode is selected.",
                    ]
                )
            else:
                lines.append("- POD removal disabled; POD-removed masks and POD-sensitive warnings were not calculated.")
            lines.append(f"- Illumination jump frames: {int(metrics_df.get('illuminationJumpFlag', pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool).sum())}")
        lines.extend(
            [
                "",
                "## Maximum single-gap artefact filter",
                f"- RBC gap safety profile: {segmentation_params.get('rbcGapSafetyMode', 'n/a')}",
                f"- enableMaxGapAreaFilter: {large_gap_summary.get('maxGapAreaFilterEnabled', segmentation_params.get('enableMaxGapAreaFilter', 'n/a'))}",
                f"- maxGapAreaFrac: {large_gap_summary.get('maxGapAreaFrac', segmentation_params.get('maxGapAreaFrac', 'n/a'))}",
                f"- Total suspicious large gap objects detected: {large_gap_summary.get('totalLargeGapObjectsDetected', 'n/a')}",
                f"- Total large gap objects removed: {large_gap_summary.get('totalLargeGapObjectsRemoved', 'n/a')}",
                f"- Frames with large-gap removal: {large_gap_summary.get('framesWithLargeGapRemoval', 'n/a')}",
                f"- Percent frames with large-gap removal: {large_gap_summary.get('percentFramesWithLargeGapRemoval', 'n/a')}",
                f"- Maximum suspicious object fraction: {large_gap_summary.get('maxLargestRemovedGapFrac', 'n/a')}",
                f"- largeGapArtefactRisk: {large_gap_summary.get('largeGapArtefactRisk', 'n/a')}",
                f"- MATLAB-compatible mode disabled filter: {large_gap_summary.get('matlabCompatibleModeDisabledFilter', 'n/a')}",
                "- This filter is an artefact-control safety layer. Any removed object should be reviewed in the overlays.",
            ]
        )
        if qc_summary.get("startupQCSelectedFrame1", False):
            lines.append("- Reliable morphology was detected from the beginning of the analysed sequence.")
        if (
            stable_rule_text != "disabled/manual only"
            and stable_start_text == "Not available / no stable reliable window detected"
            and not qc_summary.get("stableReliableWindowDetected", False)
        ):
            lines.append("- No stable reliable analysis window was detected. Check overlays or use a later acquisition window if available.")
            lines.append("- No stable reliable aggregation window was detected. Isolated reliable-looking frames were not used for the main summary.")
        if qc_summary.get("noStartSafetyTriggered", False):
            lines.append("- No reliable plasma-gap aggregation start was detected. SASI/AggIndex are calculated for transparency only and should not be interpreted as mature aggregation scores.")

        rbc_summary_df = _rbc_main_summary_df(metrics_df)
        scores_confidence = qc_summary.get("scoresConfidence")
        if qc_summary.get("scoresInterpretableWithWarnings", False):
            scores_interpretable_text = "Yes, with QC warnings"
        elif qc_summary.get("scoresInterpretable", False):
            scores_interpretable_text = "Yes"
        else:
            scores_interpretable_text = "Low confidence"
        mean_sasi = safe_mean(rbc_summary_df, "SASI_0_100")
        mean_agg_index = safe_mean(rbc_summary_df, "AggIndex_0_100")
        mean_xi_interp = safe_mean(rbc_summary_df, "Xi_interp_px")
        mean_xi_px = safe_mean(rbc_summary_df, "Xi_px")
        xi_source = "Xi_interp_px" if np.isfinite(mean_xi_interp) else "Xi_px"
        mean_xi = mean_xi_interp if xi_source == "Xi_interp_px" else mean_xi_px
        lf_hf_source = segmentation_params.get("lf_hf_method", "MATLAB-compatible continuous-field")
        mean_gap_diam = safe_mean(rbc_summary_df, "gapMeanDiam_px")
        mean_gap_count = safe_mean(rbc_summary_df, "gapNtotal")
        lines.extend(
            [
                "",
                "## RBC Gap Score Summary",
                f"- Images analysed: {len(metrics_df)}",
                f"- Frames used for main summary: {len(rbc_summary_df)}",
                f"- Reliable frames used for main summary: {qc_summary.get('reliableFramesUsedForMainSummary', 'n/a')}",
                f"- Scores confidence: {qc_summary.get('scoresConfidence', 'n/a')}",
                f"- Scores interpretable: {scores_interpretable_text}",
                "- Segmentation target: bright plasma gaps",
                f"- RBC score Xi source: {xi_source}",
                f"- RBC LF/HF source: {lf_hf_source}",
                f"- Mean {xi_source}: {mean_xi:.4g}" if np.isfinite(mean_xi) else f"- Mean {xi_source}: n/a",
                f"- Mean gap diameter: {mean_gap_diam:.4g}" if np.isfinite(mean_gap_diam) else "- Mean gap diameter: n/a",
                f"- Mean gap count: {mean_gap_count:.4g}" if np.isfinite(mean_gap_count) else "- Mean gap count: n/a",
                "- SASI and AggIndex are RBC aggregation-specific research scores and should not be interpreted as clinical diagnostic outputs.",
            ]
        )
        if scores_confidence == "low":
            lines.extend(
                [
                    "",
                    "## Low-confidence calculated scores",
                    f"- Mean SASI_0_100: {mean_sasi:.4g}" if np.isfinite(mean_sasi) else "- Mean SASI_0_100: n/a",
                    f"- Mean AggIndex_0_100: {mean_agg_index:.4g}" if np.isfinite(mean_agg_index) else "- Mean AggIndex_0_100: n/a",
                    "- These values are provided for transparency only and should not be interpreted as mature aggregation scores.",
                ]
            )
        else:
            lines.extend(
                [
                    f"- Mean SASI_0_100: {mean_sasi:.4g}" if np.isfinite(mean_sasi) else "- Mean SASI_0_100: n/a",
                    f"- Mean AggIndex_0_100: {mean_agg_index:.4g}" if np.isfinite(mean_agg_index) else "- Mean AggIndex_0_100: n/a",
                ]
            )
        mean_sasi_qc = safe_mean(metrics_df, "SASI_QC_0_100")
        mean_agg_qc = safe_mean(metrics_df, "AggIndex_QC_0_100")
        interpretable_frames = (
            int(metrics_df.get("rbcScoreInterpretable", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool).sum())
            if not metrics_df.empty
            else 0
        )
        if "SASI_QC_0_100" in metrics_df.columns or "AggIndex_QC_0_100" in metrics_df.columns:
            lines.extend(
                [
                    "",
                    "## QC-Aware Score Interpretation",
                    "- Raw SASI_0_100/AggIndex_0_100 are retained for MATLAB compatibility and transparency.",
                    "- Xi_raw_intensity is retained as the continuous-field diagnostic Xi. Trusted interpretation uses Xi_mask/Xi_QC from the selected final binary gap mask.",
                    "- MaskSASI_like_final and MaskGapScore_like_final are MATLAB-v5-style mask-only diagnostic scores for validating Conservative/POD mask trends.",
                    "- QC-aware score columns are gated raw score values: SASI_QC_0_100, GapScore_QC_0_100, PTI_QC_0_100, and AggIndex_QC_0_100 are set to n/a when trustedForScore is false.",
                    "- When sparse-mask morphology filtering is enabled, frames below the minimum mask area/count keep raw area/count outputs but trusted Xi, gap diameter, SASI, and GapScore become n/a.",
                    "- visualGreyExcluded marks frames greyed in combined plots; warningMaskFlag alone does not grey/exclude a point.",
                    "- visualWarningOnly marks frames with warning-only QC flags that remain trusted and coloured in score plots.",
                    f"- Frames with interpretable RBC scores: {interpretable_frames}",
                    f"- QC excluded from trusted score analysis: {qc_summary.get('QCExcludedFrames', qc_summary.get('QCExcludedFrameCount', 'n/a'))}",
                    f"- QC warning frames: {qc_summary.get('warningMaskFrames', qc_summary.get('warningMaskFrameCount', 'n/a'))}",
                    f"- Mean SASI_QC_0_100: {mean_sasi_qc:.4g}" if np.isfinite(mean_sasi_qc) else "- Mean SASI_QC_0_100: n/a",
                    f"- Mean AggIndex_QC_0_100: {mean_agg_qc:.4g}" if np.isfinite(mean_agg_qc) else "- Mean AggIndex_QC_0_100: n/a",
                ]
            )
        sample_qc = qc_summary.get("sampleQC")
        reliable_used = qc_summary.get("reliableFramesUsedForMainSummary", np.nan)
        try:
            reliable_used_value = float(reliable_used)
        except (TypeError, ValueError):
            reliable_used_value = np.nan
        if scores_confidence == "low":
            lines.append("- Main RBC aggregation scores are low-confidence.")
        if sample_qc != "Reliable aggregation morphology":
            lines.append("- No confident mature aggregation interpretation is provided for this sample.")
        try:
            min_reliable_summary = max(1.0, float(qc_summary.get("minConsecutiveReliableFramesForStableWindow", qc_summary.get("stableRunLengthRequired", 3))))
        except (TypeError, ValueError):
            min_reliable_summary = 3.0
        if stable_rule_text != "disabled/manual only" and np.isfinite(reliable_used_value) and reliable_used_value < min_reliable_summary:
            lines.append("- Too few reliable plasma-gap frames for confident aggregation scoring.")
        if qc_summary.get("lowConfidenceFramesIncluded", 0) not in (0, "0") and not qc_summary.get("lowConfidenceExclusionEnabled", False):
            lines.append(
                "- Low-confidence frames were included in the main summary. SASI/AggIndex may reflect weak pre-gap texture rather than mature aggregation morphology."
            )

        if sample_qc == "Reliable aggregation morphology" and scores_confidence != "low":
            lines.append(
                "- Higher SASI generally indicates stronger structural organisation of plasma-gap morphology, based on larger structural length scale, larger gap size, and lower gap fragmentation."
            )
        elif sample_qc == "Weak/pre-gap structural texture":
            lines.append(
                "- Weak/pre-gap structural texture was detected. This may represent early structural signal or residual texture. Use with caution."
            )
            lines.append(
                "- The detected structures may represent early/weak structural signal or residual texture. A negative/control interpretation may be appropriate if no visible plasma-gap morphology is present."
            )
        elif sample_qc == "No reliable plasma-gap aggregation detected":
            lines.append(
                "- No reliable plasma-gap aggregation morphology detected. SASI/AggIndex are low-confidence and should not be interpreted as aggregation scores."
            )
            lines.append("- SASI/AggIndex are not interpretable for aggregation in this sample.")
        elif sample_qc == "Possible residual/illumination artefact":
            lines.append(
                "- Possible residual/illumination artefact was detected. SASI/AggIndex should be treated as low-confidence until overlays and frame-level QC are reviewed."
            )
        syllectogram_summary = segmentation_params.get("sasi_syllectogram_research_summary", {}) or {}
        if syllectogram_summary.get("enableSASISyllectogramResearchOutput", False):
            ai_like = float(syllectogram_summary.get("SASI_syllectogram_AI_like", np.nan))
            amplitude = float(syllectogram_summary.get("SASI_syllectogram_AMP_like", np.nan))
            t_half = float(syllectogram_summary.get("SASI_syllectogram_T_half_like", np.nan))
            fit_r2 = float(syllectogram_summary.get("SASI_syllectogram_fit_R2", np.nan))
            fit_rmse = float(syllectogram_summary.get("SASI_syllectogram_fit_RMSE", np.nan))
            start_value = float(syllectogram_summary.get("SASI_syllectogram_startFrame_or_startTime", np.nan))
            lines.extend(
                [
                    "",
                    "## Optional SASI Pseudo-Syllectogram Research Output",
                    "- A SASI-based pseudo-syllectogram was fitted with a double-exponential curve and converted to a RheoScan-like area-ratio AI.",
                    "- This SASI-derived AI-like value is an image-derived RheoScan-like validation metric and should not be reported as the official RheoScan AI.",
                    f"- SASI_syllectogram_AI_like: {ai_like:.4g}" if np.isfinite(ai_like) else "- SASI_syllectogram_AI_like: n/a",
                    f"- SASI_syllectogram_AMP_like: {amplitude:.4g}" if np.isfinite(amplitude) else "- SASI_syllectogram_AMP_like: n/a",
                    f"- SASI_syllectogram_T_half_like: {t_half:.4g}" if np.isfinite(t_half) else "- SASI_syllectogram_T_half_like: n/a",
                    f"- SASI_syllectogram_fit_R2: {fit_r2:.4g}" if np.isfinite(fit_r2) else "- SASI_syllectogram_fit_R2: n/a",
                    f"- SASI_syllectogram_fit_RMSE: {fit_rmse:.4g}" if np.isfinite(fit_rmse) else "- SASI_syllectogram_fit_RMSE: n/a",
                    f"- SASI_syllectogram_fit_success: {syllectogram_summary.get('SASI_syllectogram_fit_success', False)}",
                    f"- SASI_syllectogram_fit_method: {syllectogram_summary.get('SASI_syllectogram_fit_method', 'n/a')}",
                    f"- SASI_syllectogram_orientation: {syllectogram_summary.get('SASI_syllectogram_orientation', 'n/a')}",
                    f"- SASI_syllectogram_axis: {syllectogram_summary.get('SASI_syllectogram_axis', 'n/a')}",
                    f"- SASI_syllectogram_startFrame_or_startTime: {start_value:.4g}" if np.isfinite(start_value) else "- SASI_syllectogram_startFrame_or_startTime: n/a",
                    f"- SASI_syllectogram_analysis_points: {syllectogram_summary.get('SASI_syllectogram_analysis_points', 0)}",
                ]
            )
            if syllectogram_summary.get("SASI_syllectogram_warning"):
                lines.append(f"- Warning: {syllectogram_summary['SASI_syllectogram_warning']}")

    if pca_result is not None and getattr(pca_result, "warning", None) is None and not getattr(pca_result, "scores", pd.DataFrame()).empty:
        explained = getattr(pca_result, "explained_variance", pd.Series(dtype=float))
        pc1 = explained.get("PC1", np.nan)
        pc2 = explained.get("PC2", np.nan)
        used_metrics = list(getattr(pca_result, "used_metrics", []))
        pca_xi_source = "Xi_interp_px" if "Xi_interp_px" in used_metrics else ("Xi_px" if "Xi_px" in used_metrics else "not used")
        pca_lines = [
            "",
            "## PCA Summary",
            f"- Metrics used: {', '.join(used_metrics)}",
        ]
        if mode == "RBC Gap Mode":
            pca_lines.append(f"- PCA Xi source: {pca_xi_source}")
            qc_summary = segmentation_params.get("rbc_qc_summary", {}) or {}
            if qc_summary.get("scoresConfidence") == "low":
                pca_lines.append("- PCA is exploratory because the sample did not pass reliable aggregation QC.")
        pca_lines.extend(
            [
                f"- PC1 explained variance: {pc1 * 100:.1f}%" if np.isfinite(pc1) else "- PC1 explained variance: n/a",
                f"- PC2 explained variance: {pc2 * 100:.1f}%" if np.isfinite(pc2) else "- PC2 explained variance: n/a",
            ]
        )
        lines.extend(pca_lines)
    else:
        pca_warning = getattr(pca_result, "warning", None) if pca_result is not None else "PCA was not run."
        lines.extend(["", "## PCA Summary", f"- {pca_warning}"])

    if mode == "RBC Gap Mode":
        lines.extend(matlab_comparison_report_lines(matlab_comparison_result))

    lines.extend(
        [
            "",
            "## Warnings and Limitations",
            "- This MVP is for research-use structural image analysis only.",
            "- It is not intended for clinical diagnosis, treatment decisions, or medical advice.",
            "- RBC-specific SASI/AggIndex are research structural scores calculated only in RBC Gap Mode.",
            "- RBC aggregate-network metrics are inferred from the complement of the gap mask and are not true RBC segmentation.",
            "- Ground-Truth Mask Mode is for validation and morphology extraction; it does not depend on RBC-specific SASI/AggIndex.",
        ]
    )
    if mode == "Object Mode" and (
        segmentation_params.get("enhancedObjectModeEnabled", False)
        or segmentation_params.get("experimentalCompositeScoresEnabled", False)
    ):
        lines.append("- Object Mode masks represent generic detected structures. Optional gSASI/gAggIndex are exploratory structural scores, not validated biological or clinical outputs.")
    existing_text = "\n".join(lines)
    for warning in warnings:
        if str(warning) not in existing_text:
            lines.append(f"- {warning}")
    return "\n".join(lines).strip() + "\n"
