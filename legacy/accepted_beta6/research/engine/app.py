"""MicroStructAI V3 Streamlit application."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import ndimage as ndi
from PIL import Image

from modules.calibration import (
    DEFAULT_CAMERA_PRESETS,
    TUNABLE_PARAMETER_KEYS,
    calibration_confidence,
    calibration_report_markdown,
    camera_preset_json,
    generate_calibration_candidates,
    load_camera_preset_json,
    preset_settings,
    safe_filename_token,
    score_calibration_candidate,
)
from modules.io_utils import ImageFileRef, LOCAL_IMAGE_EXTENSIONS, SUPPORTED_IMAGE_TYPES, list_local_image_files, make_image_file_refs, read_uploaded_image
from modules.web_uploads import read_sample_archive
from modules.runtime_mode import is_web_deployment
from modules.matlab_comparison import AUTO_MATCH, MATCH_METHODS, compare_matlab_metrics
from modules.metrics import (
    CORE_METRICS,
    GENERIC_OBJECT_METRICS,
    NORMALIZED_SYLLECTOGRAM_METRICS,
    RBC_AGG_APPROX_METRICS,
    RBC_GAP_METRICS,
    RBC_SASI_SYLLECTOGRAM_METRICS,
    RBC_SCORE_METRICS,
    compute_core_metrics,
    add_normalized_syllectogram_outputs,
    compute_xi_from_binary_mask,
    compute_xi_interp_matlab_like,
    compute_generic_structure_scores,
    compute_object_mode_metrics,
    compute_rbc_aggregation_scores,
    compute_rbc_gap_metrics,
    compute_sasi_syllectogram_research_output,
    dice_coefficient,
    gap_area_slope_matlab_debug,
    iou_coefficient,
    matlab_robust_z,
    safe_numeric_series,
    sigmoid100,
)
from modules.multi_sample import (
    BAR_METRICS,
    PRIMARY_RBC_METRICS,
    TIME_CURVE_METRICS,
    infer_experimental_labels,
    multi_sample_report,
    plot_multi_sample_bar,
    plot_multi_sample_time_curves,
    prepare_multi_sample_comparison,
)
from modules.pca_analysis import PCAResult, finite_numeric_columns, run_pca
from modules.plotting import (
    group_metrics_by_order_of_magnitude,
    plot_gt_summary,
    plot_matlab_difference,
    plot_matlab_scatter,
    plot_matlab_spearman_matrix_comparison,
    plot_matlab_trend,
    plot_metric_lines,
    plot_pc1_loadings,
    plot_pc1_over_index,
    plot_pca_scatter,
    plot_rbc_metric_lines_qc,
    plot_rbc_spearman_correlation_matrix,
    plot_segmentation_quality,
)
from modules.report import MODE_NOTES, compact_frame_ranges, generate_report, rbc_included_excluded_frame_numbers, stable_start_frame_text
from modules.rbc_scores import (
    RBC_SCORE_FORMULA_VERSION,
    compute_official_mask_scores,
    sigmoid100_preserve_nan,
)
from modules.structural_intelligence import analyze_structural_sequence, passport_markdown
from modules.structural_ui import render_structural_intelligence
from modules.qc_plot_ui import render_qc_plots
from modules.qc_plotting import count_caption, metric_view, qc_details
from modules.ui_cache import cached_pca, figure_png, frame_key, remembered_upload
from modules.development_safety import DEV_VERSION, attach_context, require_inputs, settings_digest, input_selection_signature
from modules.development_ui import input_controls, render_quality, render_sample_report, roi_preview
from modules.v4_ingestion import read_uploaded_image, load_sequence
from modules.structural_core import compute_core_metrics
from modules.segmentation_adapters import run_legacy_adapter
from modules.v4_pipeline import attach_automatic, update_acquisition
from modules.segmentation import (
    apply_crop,
    compute_rbc_matlab_quality_frame,
    compute_rbc_iraw01_matlab_like,
    compute_rbc_ximg01_matlab_like,
    compute_mean_intensity_curves,
    compute_pod_svd_and_optional_removal,
    create_boundary_overlay,
    estimate_illumination_gain_simple,
    load_ground_truth_mask,
    make_gap_mask_conservative,
    match_image_mask_files,
    overlay_mask_on_image,
    resize_mask_to_shape,
    safe_spatial_illumination_correction,
    segment_bright_objects,
    segment_dark_objects,
    segment_enhanced_structures_seedgrow,
    segment_rbc_gaps_seedgrow,
    simple_gap_metrics,
    to_grayscale_float01,
)


WEB_DEPLOYMENT = is_web_deployment()
DEFAULT_CAMERA_PRESETS["PATEND user settings - no POD"] = json.loads(
    (Path(__file__).parent / "presets" / "patend_v3_nopod_settings.json").read_text(encoding="utf-8")
)

MODE_OPTIONS = [
    "RBC Gap Mode",
    "Object Mode",
    "Ground-Truth Mask Mode",
    "Batch Validation",
    "Multi-Sample Comparison",
    "Auto-Tuning / Camera Calibration",
]
APP_VERSION = DEV_VERSION
ANALYSIS_SCHEMA_VERSION = "3.1"
INPUT_METHODS = ["Upload image files"]
if not WEB_DEPLOYMENT:
    INPUT_METHODS.append("Load images from local folder path")
OBJECT_ANALYSIS_METHODS = [
    "Standard Object Mode",
    "Enhanced Object Mode",
    "Conservative Object Mode",
    "Experimental POD-removed Object Mode",
]
BATCH_VALIDATION_ANALYSIS_MODES = [
    "RBC Gap Mode",
    "Standard Object Mode",
    "Enhanced Object Mode",
    "Conservative Object Mode",
    "Experimental POD-removed Object Mode",
    "Ground-Truth Mask Mode",
]
INDEX_COLUMNS = [
    "originalImageIndex_0based",
    "originalImageIndex_1based",
    "analysisIndex_0based",
    "analysisIndex_1based",
]
MATLAB_WINDOW_COLUMNS = ["matlabTime_s", "time_s", "isAna_like_matlab", "motionD_like_matlab", "matlabQualityKeep"]
RBC_QC_COLUMNS = [
    "structuralContrast",
    "maskIntensitySeparationZ",
    "boundarySupport",
    "rbcQCLabel",
    "rbcFrameQC",
    "includedInMainSummary",
    "usedInSummary",
    "excludePreStartFramesFromTrustedAnalysis",
    "excludeBadPostStartQCFramesFromTrustedAnalysis",
    "beforeStartFrame",
    "rbcFrameReliable",
    "rbcLowConfidence",
    "lowConfidenceFlag",
    "visualQCExcluded",
    "visualGreyExcluded",
    "visualWarningOnly",
    "warningMaskFlag",
    "trustedForScore",
    "hardFailure",
    "earlyLowConfidenceFrame",
    "stableReliableWindowFrame",
    "isolatedReliableFrame",
    "usedForMainSummary",
    "mainSummaryWindow",
    "reliableFramePercent",
    "suggestedStableStartFrame",
    "suggestedStableStartAnalysisIndex",
    "framesUsedForMainSummary",
    "reliableFramesUsedForMainSummary",
    "lowConfidenceFramesExcluded",
    "weakPreGapFramesExcluded",
    "possibleArtifactFramesExcluded",
    "isolatedReliableFramesExcluded",
    "scoresConfidence",
    "lowConfidenceExclusionEnabled",
    "stableReliableWindowDetected",
    "stableWindowRule",
    "minConsecutiveReliableFramesForStableWindow",
    "minSHI_tileCV_forReliable",
    "isReliableMorphology",
    "isStartupReliableMorphology",
    "isAnaFinal",
    "isAna_manualOriginal",
    "autoStartMethod",
    "autoStartTime_s",
    "autoStartFrame_POD",
    "autoStart_originalImageIndex",
    "autoStart_originalFileName",
    "autoStartFallbackUsed",
    "autoStartFallbackReason",
    "autoStartMaskFallbackReason",
    "autoStartMaskMinAreaFrac",
    "autoStartMaskMinGapNtotal",
    "autoStartMaskMinMeanDiam_px",
    "autoStartMaskMinConsecutiveFrames",
    "autoStartMaskFallbackRunStart",
    "autoStartMaskFallbackRunEnd",
    "rawBrightFrac_autoStart",
    "rawContrast01_autoStart",
    "bfN_autoStart",
    "ctN_autoStart",
    "skelN_autoStart",
    "branchN_autoStart",
    "startSignal_autoStart",
    "startSignalSlope_autoStart",
    "startupQC_startTime_s",
    "startupQCStableStartFrame",
    "firstReliableMorphologyFrame",
    "finalStartFrameUsed",
    "startFrameMode",
    "startupQCMode",
    "normAggSkelLen",
    "normAggBranchPts",
    "normSHI",
    "startupCriteriaPassed",
    "qcFailSHI_tileCV",
    "qcFailLowContrastSeparation",
    "qcFailImplausibleMask",
    "qcOutsideStableWindow",
    "sampleQC",
    "scoresInterpretable",
    "QCExcludedFrames",
    "hardBadMaskFrames",
    "warningMaskFrames",
    "lowConfidenceFrames",
    "preStartFramesDetected",
    "beforeStartExcludedFrames",
    "postStartQCWarningFrames",
    "postStartQCExcludedFrames",
    "visualWarningOnlyFrames",
    "trustedForScoreFrames",
    "usedInSummaryFrames",
    "hugeGapFlag",
    "illuminationCV",
    "badMaskFlag",
    "hardBadMaskFlag",
    "podSensitiveFlag",
    "spatialCorrectionSensitiveFlag",
    "rawGainCorrectionSensitiveFlag",
    "conservativeWarningFlag",
    "unstableSegmentationFlag",
    "maskReliabilityScore",
    "applySparseMaskMorphologyFilter",
    "maskMorphologyReliable",
    "minReliableMaskAreaFracForMorphology",
    "minReliableObjectsForMorphology",
    "sparseMaskMorphologyFiltered",
    "sparseMaskMorphologyFilteredFrames",
    "raw_gapMeanDiam",
    "trusted_gapMeanDiam",
    "raw_gapMeanArea",
    "trusted_gapMeanArea",
    "raw_largestGapFrac",
    "trusted_largestGapFrac",
    "raw_Xi",
    "trusted_Xi",
    "raw_Xi_px",
    "trusted_Xi_px",
    "raw_Xi_interp_px",
    "trusted_Xi_interp_px",
    "trusted_Xi_mask",
    "raw_SASI_0_100",
    "trusted_SASI_0_100",
    "raw_GapScore_0_100",
    "trusted_GapScore_0_100",
    "trusted_PTI_0_100",
    "trusted_AggIndex_0_100",
    "mask_gapAreaFrac",
    "mask_gapNtotal",
    "mask_gapMeanDiam_px",
    "mask_gapMaxDiam_px",
    "mask_largestGapFrac",
    "Xi_raw_intensity",
    "Xi_mask",
    "Xi_mask_noPOD",
    "Xi_mask_PODremoved",
    "Xi_QC",
    "xiSource",
    "reliableForMaskScore",
    "MaskSASI_like_final",
    "MaskGapScore_like_final",
    "MaskSASI_like_noPOD",
    "MaskSASI_like_PODremoved",
    "MaskGapScore_like_noPOD",
    "MaskGapScore_like_PODremoved",
    "SASI_QC",
    "GapScore_QC",
    "PTI_QC",
    "AggIndex_QC",
    "scoreSource",
    "maskQCFailureReasons",
    "excludedByMaskQC",
    "exclusionReason",
    "nFramesExcludedByMaskQC",
    "percentFramesExcludedByMaskQC",
    "meanMaskReliabilityScore",
    "minMaskReliabilityScore",
    "sampleMaskQCRisk",
    "postStartBadFrameExclusionEnabled",
    "maskReliabilityMinForSummary",
    "adaptiveRepairEnabled",
    "adaptiveRepairTriggered",
    "adaptiveRepairKept",
    "adaptiveRepairReason",
    "largestGapFrac_beforeRepair",
    "largestGapFrac_afterRepair",
    "largestToMedianGapRatio_beforeRepair",
    "largestToMedianGapRatio_afterRepair",
    "maskReliabilityScore_beforeRepair",
    "maskReliabilityScore_afterRepair",
    "hugeGapFrameCount",
    "badMaskFrameCount",
    "unstableSegmentationFrameCount",
    "adaptiveRepairTriggeredCount",
    "adaptiveRepairKeptCount",
]
STARTUP_TRANSIENT_COLUMNS = [
    "enableStartupTransientFallback",
    "startupFallbackUsed",
    "normalMaskReliable",
    "frameClass",
    "startupFallbackReason",
    "startupEarlyFrameFlag",
    "startupRawMean",
    "startupRawStd",
    "startupRawCV",
    "startupGradientEnergy",
    "startupTextureVariance",
    "startupFFT_LF_HF_ratio",
    "startupMotionD",
    "startupStreakIndex",
    "nStartupFallbackFrames",
    "fracStartupFallbackFrames",
    "medianStartupRawCV",
    "medianStartupGradientEnergy",
    "medianStartupFFT_LF_HF_ratio",
    "medianStartupStreakIndex",
    "firstReliableFrame",
    "firstStableFrameTime_s",
]
CONSERVATIVE_RBC_COLUMNS = [
    "rbcMode",
    "selectedRBCMethod",
    "finalPreprocessingSource",
    "finalMaskSource",
    "finalAreaFracSource",
    "areaFracFinal",
    "nObjectsFinal",
    "useCrop",
    "cropMode",
    "cropFrac",
    "applySpatialIlluminationCorrection",
    "spatialCorrectionMethod",
    "spatialIllumSigma",
    "spatialCorrectionStrength",
    "conservativeAdaptiveEngine",
    "conservativeAdaptSensitivity",
    "conservativeAdaptMeanScaleBase",
    "conservativeAdaptiveOffset",
    "podRemovedAdaptiveOffset",
    "applyRawGainCorrectionToImages",
    "enablePODRemoval",
    "removeModes",
    "rawMeanIntensity",
    "rawStdIntensity",
    "rawCV",
    "correctedMeanIntensity",
    "correctedStdIntensity",
    "correctedCV",
    "illuminationGain",
    "illuminationJumpFlag",
    "noPODMeanIntensity",
    "noPODCorrectedMeanIntensity",
    "podRemovedMeanIntensity",
    "podRemovedCorrectedMeanIntensity",
    "noPOD_CV",
    "podRemoved_CV",
    "conservativeNoPODAreaFrac",
    "conservativeGapAreaFrac",
    "conservativeAreaFrac",
    "standardAreaFrac",
    "podRemovedGapAreaFrac",
    "podRemovedAreaFrac",
    "podAreaDelta",
    "podAreaRatio",
    "podConsistencyFlag",
    "podMode1Energy",
    "first5PODCumulativeEnergy",
    "spatialCorrectionChangedMaskTooMuch",
    "rawGainCorrectionChangedMaskTooMuch",
    "conservativeFrameClass",
    "conservativeMaskReliable",
]
RBC_ANALYSIS_WINDOW_OPTIONS = ["Full sequence", "Fixed start frame", "MATLAB-compatible window", "QC-suggested stable window"]
PLOT_X_AXIS_OPTIONS = ["analysisIndex_1based", "originalImageIndex_1based", "filename"]
RBC_OVERLAY_BACKGROUNDS = ["Raw grayscale image", "Illumination-corrected Wz", "Contrast-enhanced residual"]
RBC_ANALYSIS_METHODS = ["Standard RBC Gap Mode", "Conservative RBC Gap Mode", "Experimental POD-removed RBC Mode"]
ENHANCED_OBJECT_PREPROCESSING_STYLES = ["Current generic enhanced mode", "MATLAB/RBC-style robust residual mode"]
GT_METRICS = [
    "GT_areaFrac",
    "GT_nObjects",
    "GT_meanArea",
    "GT_medianArea",
    "GT_meanDiam",
    "GT_medianDiam",
    "GT_largestObjectFrac",
    "GT_Xi_px",
    "GT_lacunarity16",
    "GT_skelDensity",
    "GT_branchDensity",
    "GT_endpointDensity",
]
AUTO_METRICS = ["AUTO_areaFrac", "AUTO_nObjects", "AUTO_meanArea", "AUTO_meanDiam", "Dice_auto_vs_GT", "IoU_auto_vs_GT"]
BATCH_MAIN_METRICS = ["gapMeanDiam_px", "gapNtotal", "gapAreaFrac", "Xi_interp_px", "LF_log10_LFoverHF", "SHI_tileCV"]
BATCH_SCORE_METRICS = ["SASI_0_100", "AggIndex_0_100", "GapScore_0_100", "PTI_0_100"]

INTENSITY_CORRECTION_MODE_OPTIONS = ["Off — do not correct frame-to-frame intensity changes", "On — always apply correction", "Auto — apply only if significant jumps are detected"]
INTENSITY_CORRECTION_MODE_VALUES = {
    INTENSITY_CORRECTION_MODE_OPTIONS[0]: "off",
    INTENSITY_CORRECTION_MODE_OPTIONS[1]: "on",
    INTENSITY_CORRECTION_MODE_OPTIONS[2]: "auto",
}
INTENSITY_CORRECTION_MODE_LABELS = {
    "off": INTENSITY_CORRECTION_MODE_OPTIONS[0],
    "on": INTENSITY_CORRECTION_MODE_OPTIONS[1],
    "auto": INTENSITY_CORRECTION_MODE_OPTIONS[2],
}
BATCH_COMPARISON_METRICS = BATCH_MAIN_METRICS + BATCH_SCORE_METRICS + ["gapAreaSlope"]
PCA_DEFAULT_METRICS = [
    "areaFrac",
    "nObjects",
    "meanArea",
    "meanDiam",
    "largestObjectFrac",
    "Xi_px",
    "lacunarity16",
    "skelDensity",
    "branchDensity",
    "endpointDensity",
    "anisotropy",
    "orientOrderParam",
    "interfaceDensity_px",
    "largestClusterFrac_phase",
    "largestClusterFrac_complement",
]
RBC_PCA_METRICS = [
    "gapAreaFrac",
    "gapNtotal",
    "gapMeanDiam_px",
    "Xi_interp_px",
    "Xi_px",
    "lacunarity16",
    "gapAreaSlope",
    "LF_log10_LFoverHF",
    "SHI_tileCV",
    # Composite scores remain available in the Custom preset. They are not
    # default PCA inputs because their component metrics are already present.
    "anisotropy",
    "orientOrderParam",
    "interfaceDensity_px",
    "largestClusterFrac_phase",
    "largestClusterFrac_complement",
]
PCA_PRESETS = ["V3 primitive metrics", "QC-trusted structural research features", "MATLAB legacy metric set", "Custom"]
STRUCTURAL_PCA_METRICS = [
    "gapAreaMedian_px2",
    "gapAreaP90_px2",
    "largestGapFractionOfImage",
    "localThicknessMedian_px",
    "localThicknessP90_px",
    "dominantGranulometricRadius_px",
    "chordLengthAnisotropy",
    "spatialGapFractionCV",
    "spatialGapFractionMoranI",
    "topologyConnectedComponents",
    "topologyEnclosedHoles",
    "topologyEulerCharacteristic",
    "gapNetworkSkeletonDensity",
    "gapNetworkBranchDensity",
    "gapNetworkEndpointDensity",
    "gapNetworkNematicOrder",
    "contourLowOrderEnergyMedian",
]
STRUCTURAL_PCA_TRUSTED_DEFAULTS = [
    "gapAreaMedian_px2",
    "gapAreaP90_px2",
    "largestGapFractionOfImage",
    "localThicknessMedian_px",
    "localThicknessP90_px",
    "dominantGranulometricRadius_px",
    "spatialGapFractionCV",
    "topologyConnectedComponents",
    "topologyEnclosedHoles",
    "topologyEulerCharacteristic",
]
MATLAB_LEGACY_PCA_METRICS = [
    "AggIndex_0_100",
    "PTI_0_100",
    "SHI_tileCV",
    "gapMeanDiam_px",
    "gapAreaFrac",
    "gapNtotal",
    "gapAreaSlope",
    "LF_log10_LFoverHF",
    "aggLargestFrac",
    "aggSkelLen",
    "aggBranchPts",
    "motionD",
    "SASI_0_100",
]


def collect_input_files(mode: str, input_method: str) -> tuple[list[ImageFileRef], list[ImageFileRef], dict[str, Any], list[str]]:
    """Collect uploaded or local-folder files and attach original indices."""
    warnings: list[str] = []
    input_info: dict[str, Any] = {"input_method": input_method}

    if input_method == "Load images from local folder path":
        if WEB_DEPLOYMENT:
            raise ValueError("Local folder inputs are disabled in the online app.")
        st.sidebar.info("Local folder path mode works only when the app is running locally on your computer. It will not work in cloud deployment.")
        image_folder = st.sidebar.text_input("Image folder path", value="", placeholder=r"C:\path\to\RBC_images")
        image_files_raw, image_warnings = list_local_image_files(image_folder)
        warnings.extend(image_warnings)
        input_info["image_folder_path"] = image_folder

        mask_files_raw: list = []
        if mode == "Ground-Truth Mask Mode":
            mask_folder = st.sidebar.text_input("Mask / outline folder path", value="", placeholder=r"C:\path\to\RBC_masks")
            mask_files_raw, mask_warnings = list_local_image_files(mask_folder)
            warnings.extend(mask_warnings)
            input_info["mask_folder_path"] = mask_folder
    else:
        image_files_raw = st.sidebar.file_uploader("Upload image files", type=SUPPORTED_IMAGE_TYPES, accept_multiple_files=True) or []
        mask_files_raw = []
        if mode == "Ground-Truth Mask Mode":
            mask_files_raw = st.sidebar.file_uploader("Upload ground-truth mask/outline files", type=SUPPORTED_IMAGE_TYPES, accept_multiple_files=True) or []

    image_refs = make_image_file_refs(image_files_raw)
    mask_refs = make_image_file_refs(mask_files_raw)
    input_info["files_loaded_before_filtering"] = len(image_refs)
    input_info["masks_loaded_before_filtering"] = len(mask_refs)
    return image_refs, mask_refs, input_info, warnings


def range_controls(total_images: int, mode: str) -> tuple[dict[str, Any], list[str]]:
    """Render image range controls and return a normalized range selection."""
    warnings: list[str] = []
    st.sidebar.header("Image Range")
    indexing_mode = st.sidebar.selectbox("Indexing mode", ["1-based indexing", "0-based indexing"], index=0)
    base = 1 if indexing_mode == "1-based indexing" else 0
    last_index = total_images if base == 1 else max(0, total_images - 1)
    min_index = base

    skip_first_n = int(st.sidebar.number_input("Skip first N images", min_value=0, max_value=max(0, total_images), value=0, step=1))
    if mode == "RBC Gap Mode":
        st.sidebar.caption("Use this to match MATLAB settings such as analysisStart frame/image.")

    effective_start_default = min(last_index, skip_first_n + base) if total_images else base
    start_image_index = int(
        st.sidebar.number_input(
            "Start image index",
            min_value=min_index,
            max_value=max(last_index, min_index),
            value=max(min_index, effective_start_default),
            step=1,
            disabled=skip_first_n > 0,
            key=f"range_start_{mode}_{indexing_mode}_{skip_first_n}_{total_images}",
        )
    )
    if skip_first_n > 0:
        start_image_index = skip_first_n + base
        st.sidebar.caption("Skip first N images overrides Start image index.")

    end_image_index = int(
        st.sidebar.number_input(
            "End image index",
            min_value=min_index,
            max_value=max(last_index, min_index),
            value=max(last_index, min_index),
            step=1,
            key=f"range_end_{mode}_{indexing_mode}_{total_images}",
        )
    )

    start_0based = max(0, start_image_index - base)
    end_0based = min(max(0, end_image_index - base), max(0, total_images - 1))
    valid_range = total_images > 0 and start_0based <= end_0based and start_0based < total_images
    if total_images > 0 and start_0based > end_0based:
        warnings.append("Start image index is greater than End image index. Adjust the range before running analysis.")
    elif total_images > 0 and start_0based >= total_images:
        warnings.append("Start image index is outside the loaded image list. Adjust the range before running analysis.")

    if valid_range:
        start_display = start_0based + base
        end_display = end_0based + base
        selected_count = end_0based - start_0based + 1
        st.sidebar.success(f"Analysing images {start_display} to {end_display} out of {total_images} loaded images.")
    else:
        start_display = start_image_index
        end_display = end_image_index
        selected_count = 0
        if total_images:
            st.sidebar.warning("No images are selected by the current range.")

    plot_x_axis = st.sidebar.selectbox("Plot x-axis", PLOT_X_AXIS_OPTIONS, index=0)
    return (
        {
            "indexing_mode": indexing_mode,
            "skip_first_n": skip_first_n,
            "range_start_input": start_image_index,
            "range_end_input": end_image_index,
            "range_start_0based": start_0based,
            "range_end_0based": end_0based,
            "range_start_display": start_display,
            "range_end_display": end_display,
            "range_valid": valid_range,
            "selected_image_count": selected_count,
            "plot_x_axis": plot_x_axis,
        },
        warnings,
    )


def apply_range_to_refs(file_refs: list[ImageFileRef], range_info: dict[str, Any]) -> list[ImageFileRef]:
    """Apply a normalized range to naturally sorted file references."""
    if not range_info.get("range_valid", False):
        return []
    start = int(range_info.get("range_start_0based", 0))
    end = int(range_info.get("range_end_0based", -1))
    return file_refs[start : end + 1]


def filter_pairs_by_image_range(pairs: list[tuple[ImageFileRef, ImageFileRef]], range_info: dict[str, Any]) -> list[tuple[ImageFileRef, ImageFileRef]]:
    """Filter matched image/mask pairs by the image file's original index."""
    if not range_info.get("range_valid", False):
        return []
    start = int(range_info.get("range_start_0based", 0))
    end = int(range_info.get("range_end_0based", -1))
    return [(image_ref, mask_ref) for image_ref, mask_ref in pairs if start <= image_ref.original_index_0based <= end]


def image_index_metadata(image_file: Any, analysis_idx: int) -> dict[str, int]:
    """Create original and analysis index columns for one analysed image."""
    original_0 = int(getattr(image_file, "original_index_0based", analysis_idx))
    return {
        "imageIndex": analysis_idx,
        "originalImageIndex_0based": original_0,
        "originalImageIndex_1based": int(getattr(image_file, "original_index_1based", original_0 + 1)),
        "analysisIndex_0based": analysis_idx,
        "analysisIndex_1based": analysis_idx + 1,
    }


def object_segmentation_controls(
    title: str = "Object Segmentation",
    allow_enhanced: bool = True,
    force_enhanced: bool | None = None,
    force_object_method: str | None = None,
    allow_experimental_scores: bool = True,
) -> dict[str, Any]:
    """Render generic object segmentation controls."""
    st.sidebar.header(title)
    if force_object_method in OBJECT_ANALYSIS_METHODS:
        selected_method = force_object_method
    elif force_enhanced is not None:
        selected_method = "Enhanced Object Mode" if bool(force_enhanced) else "Standard Object Mode"
    elif allow_enhanced:
        selected_method = st.sidebar.selectbox("Object analysis method", OBJECT_ANALYSIS_METHODS, index=0)
    else:
        selected_method = "Standard Object Mode"

    if selected_method == "Enhanced Object Mode":
        st.sidebar.info(
            "In Object Mode, masks represent generic detected structures. "
            "They are not interpreted as RBC plasma gaps unless RBC Gap Mode is used."
        )
        params: dict[str, Any] = {
            "selectedObjectMethod": selected_method,
            "objectAnalysisMethod": selected_method,
            "enhancedObjectModeEnabled": True,
            "enhancedObjectPreprocessingStyle": st.sidebar.selectbox(
                "Enhanced preprocessing style",
                ENHANCED_OBJECT_PREPROCESSING_STYLES,
                index=0,
                help=(
                    "The current generic mode preserves existing Enhanced Object Mode behavior. "
                    "The MATLAB/RBC-style option reuses a robust processed residual field for generic structures only."
                ),
            ),
            "objectSegmentationTarget": st.sidebar.selectbox(
                "Segmentation target",
                ["Bright structures / bright gaps", "Dark structures / dark gaps", "Auto polarity"],
                index=0,
            ),
            "experimentalCompositeScoresEnabled": st.sidebar.checkbox("Show experimental composite structure scores", value=False),
        }
        if params["experimentalCompositeScoresEnabled"]:
            st.sidebar.warning(
                "These optional scores reuse the mathematical structure of the RBC aggregation scores, "
                "but they are not validated for this sample type. Interpret them only as exploratory image-structure indices."
            )
        if params["enhancedObjectPreprocessingStyle"] == "MATLAB/RBC-style robust residual mode":
            st.sidebar.caption(
                "This preset borrows the robust residual preprocessing used for RBC/MATLAB mask matching, "
                "but Object Mode masks remain generic structures and are not plasma gaps."
            )
        with st.sidebar.expander("Enhanced robust segmentation settings", expanded=True):
            if params["enhancedObjectPreprocessingStyle"] == "MATLAB/RBC-style robust residual mode":
                params["objectCropFrac"] = float(
                    st.number_input(
                        "Analysis crop fraction",
                        min_value=0.05,
                        max_value=1.0,
                        value=1.0,
                        step=0.05,
                        format="%.2f",
                        help="Use 1.0 for full generic images; set 0.70 only when you intentionally want the RBC/MATLAB-style center crop.",
                    )
                )
                params["objectPreprocessBgSigma"] = float(
                    st.number_input("Processed-input background sigma", min_value=1.0, max_value=120.0, value=40.0, step=1.0)
                )
            params["objectUseMedian"] = st.checkbox("Use median filter", value=True)
            params["objectMedianK"] = int(st.number_input("Median kernel size", min_value=1, max_value=15, value=3, step=2))
            params["objectSigmaIll"] = float(st.number_input("Illumination sigma", min_value=1.0, max_value=100.0, value=25.0, step=1.0))
            params["objectKHi"] = float(st.number_input("Seed threshold kHi", min_value=0.1, max_value=10.0, value=2.2, step=0.1))
            params["objectKLo"] = float(st.number_input("Grow threshold kLo", min_value=0.1, max_value=10.0, value=1.1, step=0.1))
            params["objectMinAreaPx"] = int(st.number_input("Min structure area (px)", min_value=0, max_value=100000, value=20, step=5))
            params["objectCloseRadius"] = int(st.number_input("Closing radius", min_value=0, max_value=20, value=2, step=1))
            params["objectFillHoles"] = st.checkbox("Fill structure holes", value=True)
            params["objectClearBorder"] = st.checkbox("Clear border structures", value=False)
        with st.sidebar.expander("Enhanced object artefact filter", expanded=False):
            params["enableMaxObjectAreaFilter"] = st.checkbox("Enable max object area filter", value=False)
            params["maxObjectAreaFrac"] = float(
                st.number_input("Max single-object area fraction", min_value=0.001, max_value=0.95, value=0.35, step=0.01, format="%.3f")
            )
            params["applySparseMaskMorphologyFilter"] = bool(
                st.checkbox(
                    "Set morphology metrics to NaN when mask support is too low",
                    value=False,
                    help=(
                        "Optional generic Object-mode filter. If the detected mask area/count is too small, "
                        "Xi-like and diameter-based trusted structure scores are set to NaN while raw area/count remain exported."
                    ),
                )
            )
            params["minReliableMaskAreaFracForMorphology"] = float(
                st.number_input("Min reliable mask area fraction for morphology", min_value=0.0, max_value=1.0, value=0.03, step=0.005, format="%.4f")
            )
            params["minReliableObjectsForMorphology"] = float(
                st.number_input("Min reliable object count for morphology", min_value=0.0, max_value=100000.0, value=10.0, step=1.0)
            )
            st.caption("Optional generic safety layer. Removed structures are always reported.")
        params.update(intensity_correction_controls(default_mode="off", key_prefix="enhanced_object", expanded=False))
        params.update(startup_transient_fallback_controls(mode_label="Enhanced Object Mode"))
        return params

    if selected_method in {"Conservative Object Mode", "Experimental POD-removed Object Mode"}:
        st.sidebar.info(
            "Conservative/POD Object modes use generic object masks only. "
            "They do not create RBC plasma-gap or RBC aggregation outputs."
        )
        experimental_pod = selected_method.startswith("Experimental")
        params = {
            "selectedObjectMethod": selected_method,
            "objectAnalysisMethod": selected_method,
            "enhancedObjectModeEnabled": False,
            "conservativeObjectModeEnabled": True,
            "experimentalPODObjectModeEnabled": bool(experimental_pod),
            "experimentalCompositeScoresEnabled": False,
            "enableObjectStructureScores": True,
            "objectSegmentationTarget": st.sidebar.selectbox(
                "Object segmentation target",
                ["Bright structures / bright gaps", "Dark structures / dark gaps", "Auto polarity"],
                index=0,
            ),
        }
        st.sidebar.warning(
            "ObjectStructureScore and ObjectAggregationIndex are generic mask-based structural/coarsening indices. "
            "They are not RBC SASI/AggIndex and are not validated biomarkers."
        )
        params.update(intensity_correction_controls(default_mode="auto", key_prefix="conservative_object", expanded=False))
        with st.sidebar.expander("Conservative Object preprocessing", expanded=True):
            params["useCrop"] = bool(st.checkbox("Use crop", value=True))
            params["cropMode"] = st.selectbox("Crop mode", ["center", "manual"], index=0)
            params["cropFrac"] = float(
                st.number_input("Center crop fraction", min_value=0.05, max_value=1.0, value=0.70, step=0.05, format="%.2f")
            )
            if params["cropMode"] == "manual":
                manual_x = int(st.number_input("Manual crop x", min_value=1, value=1, step=1))
                manual_y = int(st.number_input("Manual crop y", min_value=1, value=1, step=1))
                manual_width = int(st.number_input("Manual crop width", min_value=1, value=512, step=1))
                manual_height = int(st.number_input("Manual crop height", min_value=1, value=512, step=1))
                params["manualCropRect"] = f"{manual_x},{manual_y},{manual_width},{manual_height}"
            params["applySpatialIlluminationCorrection"] = bool(st.checkbox("Apply spatial flat-field correction", value=True))
            params["spatialCorrectionMethod"] = st.selectbox("Spatial correction method", ["divide", "subtract"], index=0)
            params["spatialIllumSigma"] = float(st.number_input("Spatial illumination sigma", min_value=3.0, max_value=300.0, value=60.0, step=5.0))
            params["spatialCorrectionStrength"] = float(
                st.number_input("Spatial correction strength", min_value=0.0, max_value=1.0, value=0.50, step=0.05, format="%.2f")
            )
            params["preserveSpatialMeanIntensity"] = bool(st.checkbox("Preserve frame mean during spatial correction", value=True))
            params["clipSpatialCorrectedImages"] = bool(st.checkbox("Clip spatial-corrected images", value=True))
            st.caption("Global raw-gain correction is controlled by Camera intensity-drop correction above.")
            params["smoothWindowFrames"] = int(st.number_input("Smooth window frames", min_value=1, max_value=101, value=5, step=2))
        with st.sidebar.expander("Conservative Object POD/segmentation", expanded=True):
            params["removeModes"] = int(st.number_input("POD modes to remove", min_value=0, max_value=20, value=1, step=1))
            params["computePODWhenRemoveModesZero"] = bool(
                st.checkbox(
                    "MATLAB-style zero-mode POD branch",
                    value=True,
                    help="When removeModes = 0 and a POD branch is active, still run the MATLAB prototype POD reconstruction/rescale branch.",
                )
            )
            params["enablePODSensitivityDiagnostic"] = bool(
                st.checkbox(
                    "Enable POD sensitivity diagnostic",
                    value=bool(not experimental_pod),
                    help="When enabled, computes the POD-branch mask only as a sensitivity diagnostic in Conservative Object mode. With removeModes = 0, this matches the MATLAB zero-mode reconstruction/rescale diagnostic.",
                )
            )
            pod_flags_ui = conservative_pod_processing_flags(params, selected_method)
            params["enablePODRemoval"] = bool(pod_flags_ui["podFinalEnabled"])
            params["enablePODRemovedMode"] = bool(pod_flags_ui["podFinalEnabled"])
            st.info(str(pod_flags_ui["podStatus"]))
            if params["enablePODSensitivityDiagnostic"] and not params["enablePODRemoval"]:
                st.caption("The Conservative no-POD object mask remains the final mask.")
            else:
                st.caption("Conservative Object QC active.")
            params["useMeanSubtract"] = bool(st.checkbox("Mean subtract before POD/SVD", value=False))
            params["conservativeObjectSegMethod"] = st.selectbox("Conservative object segmentation method", ["adaptive", "otsu"], index=0)
            params["conservativeObjectMinAreaPx"] = int(st.number_input("Conservative object min area px", min_value=0, max_value=100000, value=20, step=5))
            params["conservativeObjectSmoothSigma"] = float(st.number_input("Conservative object smoothing sigma", min_value=0.0, max_value=10.0, value=1.0, step=0.25))
            object_adaptive_engine_options = ["MATLAB adaptthresh-like", "Local offset (current)"]
            params["conservativeAdaptiveEngine"] = st.selectbox(
                "Conservative object adaptive threshold engine",
                object_adaptive_engine_options,
                index=0,
                help="MATLAB adaptthresh-like uses local-mean scaling measured from MATLAB adaptthresh. Local offset preserves the earlier threshold_local behaviour.",
            )
            params["conservativeAdaptSensitivity"] = float(
                st.number_input(
                    "Conservative object adapt sensitivity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.45,
                    step=0.01,
                    format="%.2f",
                    help="Used only by the MATLAB adaptthresh-like engine.",
                )
            )
            params["conservativeAdaptMeanScaleBase"] = float(
                st.number_input(
                    "Conservative object adapt mean-scale base",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.60,
                    step=0.05,
                    format="%.2f",
                    help="Used only by the MATLAB adaptthresh-like engine. Threshold multiplier is meanScaleBase - sensitivity; 1.60 with sensitivity 0.45 gives 1.15*local mean.",
                )
            )
            params["conservativeObjectAdaptiveOffset"] = float(
                st.number_input("Conservative object adaptive offset", min_value=-0.5, max_value=0.5, value=-0.043, step=0.005, format="%.3f")
            )
            params["podRemovedObjectAdaptiveOffset"] = float(
                st.number_input(
                    "POD-branch object adaptive offset",
                    min_value=-0.5,
                    max_value=0.5,
                    value=-0.070,
                    step=0.005,
                    format="%.3f",
                    help="Advanced override. MATLAB v6 uses the same segmentation settings for no-POD and POD-branch masks unless this override is enabled.",
                )
            )
            params["useSeparatePODAdaptiveOffset"] = bool(
                st.checkbox(
                    "Use separate POD-branch adaptive offset",
                    value=False,
                    help="Off matches the MATLAB v6 prototype: no-POD and POD-branch masks use the same adaptive segmentation settings.",
                )
            )
        with st.sidebar.expander("Conservative Object score/QC thresholds", expanded=False):
            params["applySparseMaskMorphologyFilter"] = bool(
                st.checkbox(
                    "Set morphology metrics to NaN when mask support is too low",
                    value=False,
                    help=(
                        "Optional generic Object-mode filter. If the detected mask area/count is too small, "
                        "Xi-like and diameter-based trusted structure scores are set to NaN while raw area/count remain exported."
                    ),
                )
            )
            params["minReliableMaskAreaFracForMorphology"] = float(
                st.number_input("Min reliable mask area fraction for morphology", min_value=0.0, max_value=1.0, value=0.03, step=0.005, format="%.4f")
            )
            params["minReliableObjectsForMorphology"] = float(
                st.number_input("Min reliable object count for morphology", min_value=0.0, max_value=100000.0, value=10.0, step=1.0)
            )
            params["minReliableObjectAreaFracForXi"] = float(
                st.number_input("Min reliable object area fraction for mask Xi", min_value=0.0, max_value=1.0, value=0.03, step=0.005, format="%.3f")
            )
            params["minReliableObjectsForXi"] = int(st.number_input("Min reliable object count for mask Xi", min_value=0, max_value=100000, value=10, step=1))
            params["maxObjectAreaFracMaskQC"] = float(
                st.number_input("Max object area fraction for reliable mask QC", min_value=0.0, max_value=1.0, value=0.85, step=0.05, format="%.2f")
            )
            params["hugeObjectFracThreshold"] = float(
                st.number_input("Huge single-object fraction threshold", min_value=0.001, max_value=1.0, value=0.35, step=0.01, format="%.3f")
            )
            params["maxAllowedPODObjectAreaDelta"] = float(
                st.number_input("Max warning POD/no-POD object area delta", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f")
            )
            params["maxAllowedSpatialObjectAreaChange"] = float(
                st.number_input("Max warning spatial-correction area change", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f")
            )
            params["maxAllowedRawGainObjectAreaChange"] = float(
                st.number_input("Max warning raw-gain area change", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f")
            )
        params.update(startup_transient_fallback_controls(mode_label=selected_method))
        return params

    threshold_method = st.sidebar.selectbox("Threshold method", ["otsu", "adaptive", "percentile"])
    polarity = st.sidebar.selectbox("Polarity", ["bright", "dark"], index=0)
    min_object_size = st.sidebar.number_input("Min object size", min_value=0, max_value=100000, value=40, step=5)
    smoothing_sigma = st.sidebar.slider("Smoothing sigma", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    fill_holes = st.sidebar.checkbox("Fill holes", value=True)
    experimental_scores = st.sidebar.checkbox("Show experimental composite structure scores", value=False) if allow_experimental_scores else False
    if experimental_scores:
        st.sidebar.warning(
            "These optional scores reuse the mathematical idea of RBC aggregation scores, but they are not validated "
            "for this sample type. Interpret them only as exploratory image-structure indices."
        )
    params: dict[str, Any] = {
        "selectedObjectMethod": "Standard Object Mode",
        "objectAnalysisMethod": "Standard Object Mode",
        "enhancedObjectModeEnabled": False,
        "conservativeObjectModeEnabled": False,
        "experimentalPODObjectModeEnabled": False,
        "objectSegmentationTarget": "Bright structures / bright gaps" if polarity == "bright" else "Dark structures / dark gaps",
        "experimentalCompositeScoresEnabled": bool(experimental_scores),
        "threshold_method": threshold_method,
        "polarity": polarity,
        "min_object_size": int(min_object_size),
        "smoothing_sigma": float(smoothing_sigma),
        "fill_holes": bool(fill_holes),
    }
    params.update(intensity_correction_controls(default_mode="off", key_prefix="standard_object", expanded=False))
    if threshold_method == "adaptive":
        params["adaptive_block_size"] = st.sidebar.slider("Adaptive block size", min_value=15, max_value=251, value=51, step=2)
        params["adaptive_offset"] = st.sidebar.slider("Adaptive offset", min_value=-0.2, max_value=0.2, value=0.0, step=0.01)
    elif threshold_method == "percentile":
        params["percentile"] = st.sidebar.slider("Target percentile", min_value=50.0, max_value=99.0, value=80.0, step=1.0)
    with st.sidebar.expander("Advanced preprocessing"):
        params["background_correction"] = st.checkbox("Background correction", value=False)
        params["contrast_enhancement"] = st.checkbox("Contrast enhancement", value=True)
        params["clahe_clip_limit"] = st.slider("Contrast clip limit", min_value=0.005, max_value=0.1, value=0.02, step=0.005)
        params["applySparseMaskMorphologyFilter"] = bool(
            st.checkbox(
                "Set morphology metrics to NaN when mask support is too low",
                value=False,
                help=(
                    "Optional generic Object-mode filter. If the detected mask area/count is too small, "
                    "Xi-like and diameter-based trusted structure scores are set to NaN while raw area/count remain exported."
                ),
            )
        )
        params["minReliableMaskAreaFracForMorphology"] = float(
            st.number_input("Min reliable mask area fraction for morphology", min_value=0.0, max_value=1.0, value=0.03, step=0.005, format="%.4f")
        )
        params["minReliableObjectsForMorphology"] = float(
            st.number_input("Min reliable object count for morphology", min_value=0.0, max_value=100000.0, value=10.0, step=1.0)
        )
    params.update(startup_transient_fallback_controls(mode_label="Object Mode"))
    return params


def control_key(prefix: str, name: str) -> str | None:
    """Return a stable optional Streamlit widget key."""
    return f"{prefix}_{name}" if prefix else None


def normalize_intensity_correction_mode(value: Any, default: str = "off") -> str:
    """Normalize camera intensity-drop correction mode to off/on/auto."""
    text = str(value or default).strip()
    if text in INTENSITY_CORRECTION_MODE_VALUES:
        text = INTENSITY_CORRECTION_MODE_VALUES[text]
    lower = text.lower()
    if lower.startswith("off"):
        return "off"
    if lower.startswith("on"):
        return "on"
    if lower.startswith("auto"):
        return "auto"
    return default if default in {"off", "on", "auto"} else "off"


def intensity_correction_controls(
    defaults: dict[str, Any] | None = None,
    default_mode: str = "off",
    key_prefix: str = "",
    expanded: bool = False,
) -> dict[str, Any]:
    """Render user-selectable global camera intensity-drop correction controls."""
    defaults = dict(defaults or {})
    mode_default = normalize_intensity_correction_mode(defaults.get("intensityCorrectionMode", default_mode), default_mode)
    with st.sidebar.expander("Camera intensity-drop correction", expanded=expanded):
        mode_label = st.selectbox(
            "Camera intensity-drop correction",
            INTENSITY_CORRECTION_MODE_OPTIONS,
            index=INTENSITY_CORRECTION_MODE_OPTIONS.index(INTENSITY_CORRECTION_MODE_LABELS.get(mode_default, INTENSITY_CORRECTION_MODE_LABELS["off"])),
            key=control_key(key_prefix, "intensityCorrectionMode"),
            help=(
                "Compensates for frame-to-frame light/camera brightness jumps by multiplying each frame by a global gain. "
                "Use Auto if you are unsure. Turn Off for stable cameras or when you want the raw optical intensity curve unchanged."
            ),
        )
        mode = INTENSITY_CORRECTION_MODE_VALUES[mode_label]
        st.caption(
            "Off preserves raw frame-to-frame brightness. On always applies estimated multiplicative gain. "
            "Auto applies gain only when jumps or unstable mean intensity are detected."
        )
        fixed_threshold = float(
            st.number_input(
                "fixedLogJumpThreshold",
                min_value=0.0,
                max_value=1.0,
                value=float(defaults.get("fixedLogJumpThreshold", np.log(1.006))),
                step=0.0005,
                format="%.6f",
                key=control_key(key_prefix, "fixedLogJumpThreshold_camera"),
            )
        )
        adaptive_mad = float(
            st.number_input(
                "adaptiveMADmult",
                min_value=0.0,
                max_value=100.0,
                value=float(defaults.get("adaptiveMADmult", 4.0)),
                step=0.5,
                key=control_key(key_prefix, "adaptiveMADmult_camera"),
            )
        )
        max_cv = float(
            st.number_input(
                "maxMeanIntensityTemporalCVForStableCamera",
                min_value=0.0,
                max_value=1.0,
                value=float(defaults.get("maxMeanIntensityTemporalCVForStableCamera", 0.02)),
                step=0.001,
                format="%.3f",
                key=control_key(key_prefix, "maxMeanIntensityTemporalCVForStableCamera"),
            )
        )
        min_jump_frames = int(
            st.number_input(
                "minJumpFramesForAutoCorrection",
                min_value=0,
                max_value=100000,
                value=int(defaults.get("minJumpFramesForAutoCorrection", 1)),
                step=1,
                key=control_key(key_prefix, "minJumpFramesForAutoCorrection"),
            )
        )
        clip_gain = bool(
            st.checkbox(
                "clipGainCorrectedImages",
                value=bool(defaults.get("clipGainCorrectedImages", True)),
                key=control_key(key_prefix, "clipGainCorrectedImages_camera"),
            )
        )
    return {
        "intensityCorrectionMode": mode,
        "applyRawGainCorrectionToImages": mode != "off",
        "enableMeanIlluminationCorrection": mode != "off",
        "fixedLogJumpThreshold": fixed_threshold,
        "adaptiveMADmult": adaptive_mad,
        "maxMeanIntensityTemporalCVForStableCamera": max_cv,
        "minJumpFramesForAutoCorrection": min_jump_frames,
        "clipGainCorrectedImages": clip_gain,
    }


def evaluate_intensity_correction(mean_raw: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    """Decide whether global multiplicative camera gain correction is applied."""
    y = np.asarray(mean_raw, dtype=np.float64).reshape(-1)
    n = y.size
    mode = normalize_intensity_correction_mode(params.get("intensityCorrectionMode", "off"), "off")
    fixed_threshold = float(params.get("fixedLogJumpThreshold", np.log(1.006)))
    adaptive_mad = float(params.get("adaptiveMADmult", 4.0))
    gain_candidate, corrected_candidate, jump_flags = estimate_illumination_gain_simple(
        y,
        enable_correction=True,
        fixed_log_jump_threshold=fixed_threshold,
        adaptive_mad_mult=adaptive_mad,
    )

    work = y.copy()
    work[~np.isfinite(work) | (work <= 0)] = np.nan
    dlog = np.full(n, np.nan, dtype=np.float64)
    if n >= 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            dlog[1:] = np.diff(np.log(work))
    finite_dlog = dlog[np.isfinite(dlog)]
    centered_abs = np.asarray([], dtype=np.float64)
    if finite_dlog.size:
        med_dlog = float(np.nanmedian(finite_dlog))
        centered_abs = np.abs(finite_dlog - med_dlog)
    max_abs_log_jump = float(np.nanmax(centered_abs)) if centered_abs.size else 0.0
    finite_mean = y[np.isfinite(y)]
    mean_intensity_temporal_cv = (
        float(np.nanstd(finite_mean) / (abs(float(np.nanmean(finite_mean))) + np.finfo(float).eps))
        if finite_mean.size
        else np.nan
    )
    n_jump_frames = int(np.count_nonzero(jump_flags))
    min_jump_frames = int(params.get("minJumpFramesForAutoCorrection", 1))
    max_temporal_cv = float(params.get("maxMeanIntensityTemporalCVForStableCamera", 0.02))
    correction_needed = bool(
        (n_jump_frames >= min_jump_frames and min_jump_frames > 0)
        or (np.isfinite(max_abs_log_jump) and max_abs_log_jump > fixed_threshold)
        or (np.isfinite(mean_intensity_temporal_cv) and mean_intensity_temporal_cv > max_temporal_cv)
    )
    correction_applied = bool(mode == "on" or (mode == "auto" and correction_needed))
    if mode == "off":
        correction_applied = False

    gain = np.asarray(gain_candidate if correction_applied else np.ones(n, dtype=np.float64), dtype=np.float64)
    corrected_mean = np.asarray(corrected_candidate if correction_applied else y, dtype=np.float64)
    finite_gain = gain[np.isfinite(gain)]
    gain_range = float(np.nanmax(finite_gain) - np.nanmin(finite_gain)) if finite_gain.size else 0.0
    return {
        "mode": mode,
        "applied": bool(correction_applied),
        "needed": bool(correction_needed),
        "gain": gain,
        "corrected_mean": corrected_mean,
        "jump_flags": np.asarray(jump_flags, dtype=bool),
        "n_jump_frames": n_jump_frames,
        "max_abs_log_jump": max_abs_log_jump,
        "mean_temporal_cv": mean_intensity_temporal_cv,
        "gain_range": gain_range,
        "median_gain": float(np.nanmedian(finite_gain)) if finite_gain.size else 1.0,
        "min_gain": float(np.nanmin(finite_gain)) if finite_gain.size else 1.0,
        "max_gain": float(np.nanmax(finite_gain)) if finite_gain.size else 1.0,
    }


def intensity_correction_frame_metadata(correction: dict[str, Any], idx: int) -> dict[str, Any]:
    """Return per-frame export metadata for camera intensity correction."""
    gain = np.asarray(correction.get("gain", []), dtype=np.float64)
    corrected_mean = np.asarray(correction.get("corrected_mean", []), dtype=np.float64)
    jump_flags = np.asarray(correction.get("jump_flags", []), dtype=bool)
    return {
        "intensityCorrectionMode": correction.get("mode", "off"),
        "intensityCorrectionApplied": bool(correction.get("applied", False)),
        "intensityCorrectionNeeded": bool(correction.get("needed", False)),
        "nIlluminationJumpFrames": int(correction.get("n_jump_frames", 0)),
        "maxAbsLogJump": float(correction.get("max_abs_log_jump", np.nan)),
        "meanIntensityTemporalCV": float(correction.get("mean_temporal_cv", np.nan)),
        "gainRange": float(correction.get("gain_range", np.nan)),
        "medianIlluminationGain": float(correction.get("median_gain", np.nan)),
        "minIlluminationGain": float(correction.get("min_gain", np.nan)),
        "maxIlluminationGain": float(correction.get("max_gain", np.nan)),
        "illuminationGain": float(gain[idx]) if idx < gain.size else 1.0,
        "illuminationJumpFlag": bool(jump_flags[idx]) if idx < jump_flags.size else False,
        "noPODCorrectedMeanIntensity": float(corrected_mean[idx]) if idx < corrected_mean.size else np.nan,
    }


def startup_transient_fallback_controls(
    defaults: dict[str, Any] | None = None,
    key_prefix: str = "",
    mode_label: str = "analysis",
) -> dict[str, Any]:
    """Render optional startup/transient fallback controls."""
    defaults = dict(defaults or {})
    with st.sidebar.expander("Startup/transient fallback", expanded=False):
        enabled = bool(
            st.checkbox(
                "Use startup/transient fallback for QC-flagged early frames",
                value=bool(defaults.get("enableStartupTransientFallback", False)),
                key=control_key(key_prefix, f"enableStartupTransientFallback_{safe_filename_token(mode_label)}"),
                help=(
                    "Optional. For early frames that fail existing QC/mask reliability checks, keep normal metrics "
                    "for transparency but add separate raw-image startup/transient texture diagnostics."
                ),
            )
        )
        st.caption(
            "Fallback metrics are startup/QC texture diagnostics only. They do not replace masks and are not used for "
            "SASI, AggIndex, normal object counts, or normal morphology scores."
        )
        params: dict[str, Any] = {
            "enableStartupTransientFallback": enabled,
            "startupFallbackEarlyFrameLimit": int(defaults.get("startupFallbackEarlyFrameLimit", 20)),
            "startupFallbackTimeLimit_s": float(defaults.get("startupFallbackTimeLimit_s", 20.0)),
            "startupFallbackReliabilityThreshold": float(defaults.get("startupFallbackReliabilityThreshold", defaults.get("maskReliabilityMinForSummary", 60.0))),
        }
        if enabled:
            params["startupFallbackEarlyFrameLimit"] = int(
                st.number_input(
                    "Early frame limit",
                    min_value=0,
                    max_value=100000,
                    value=int(defaults.get("startupFallbackEarlyFrameLimit", 20)),
                    step=1,
                    key=control_key(key_prefix, f"startupFallbackEarlyFrameLimit_{safe_filename_token(mode_label)}"),
                    help="Frames at or below this analysis/original index are considered early for fallback decisions.",
                )
            )
            params["startupFallbackTimeLimit_s"] = float(
                st.number_input(
                    "Early time limit_s",
                    min_value=0.0,
                    max_value=100000.0,
                    value=float(defaults.get("startupFallbackTimeLimit_s", 20.0)),
                    step=1.0,
                    key=control_key(key_prefix, f"startupFallbackTimeLimit_s_{safe_filename_token(mode_label)}"),
                    help="Frames before this time are considered early when time_s/matlabTime_s exists.",
                )
            )
            params["startupFallbackReliabilityThreshold"] = float(
                st.number_input(
                    "Fallback reliability threshold",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(defaults.get("startupFallbackReliabilityThreshold", defaults.get("maskReliabilityMinForSummary", 60.0))),
                    step=5.0,
                    key=control_key(key_prefix, f"startupFallbackReliabilityThreshold_{safe_filename_token(mode_label)}"),
                    help="Existing maskReliabilityScore values below this threshold can trigger fallback when the frame is early.",
                )
            )
    return params


def rbc_camera_preset_controls() -> tuple[dict[str, Any], str]:
    """Render optional built-in/uploaded RBC camera preset selection."""
    st.sidebar.header("RBC Camera Preset")
    calibration_result = st.session_state.get("camera_calibration_result")
    has_auto_tuned_preset = (
        isinstance(calibration_result, dict)
        and isinstance(calibration_result.get("recommendedSettings"), dict)
        and bool(calibration_result.get("recommendedSettings"))
    )
    preset_options = ["Manual/current controls"] + list(DEFAULT_CAMERA_PRESETS)
    if has_auto_tuned_preset:
        preset_options.append("Use last Auto-Tuning recommendation")
    preset_options.append("Load JSON preset")
    selected = st.sidebar.selectbox("Camera/setup preset", preset_options, index=0)
    if selected in DEFAULT_CAMERA_PRESETS:
        st.sidebar.caption("Preset values are setup-specific starting points. Inspect overlays before interpreting RBC scores.")
        return dict(DEFAULT_CAMERA_PRESETS[selected]), selected
    if selected == "Use last Auto-Tuning recommendation":
        preset_name = str(calibration_result.get("recommendedPresetName", "Last Auto-Tuning recommendation"))
        confidence = str(calibration_result.get("calibrationConfidence", "n/a"))
        st.sidebar.success(f"Applied last Auto-Tuning preset: {preset_name}")
        st.sidebar.caption(f"Calibration confidence: {confidence}. Inspect overlays before interpreting RBC scores.")
        return dict(calibration_result["recommendedSettings"]), preset_name
    if selected == "Load JSON preset":
        preset_file = st.sidebar.file_uploader("Load camera preset JSON", type=["json"], key="rbc_camera_preset_json")
        if preset_file is None:
            st.sidebar.info("Choose a MicroStructAI camera preset JSON file.")
            return {}, "Uploaded preset not loaded"
        try:
            preset = load_camera_preset_json(preset_file.getvalue())
        except Exception as exc:
            st.sidebar.error(f"Could not load camera preset: {exc}")
            return {}, "Uploaded preset could not be loaded"
        preset_name = str(preset.get("presetName", "Uploaded camera preset"))
        st.sidebar.success(f"Loaded preset: {preset_name}")
        return preset_settings(preset), preset_name
    return {}, "Manual/current controls"


def rbc_segmentation_controls(defaults: dict[str, Any] | None = None, key_prefix: str = "") -> dict[str, Any]:
    """Render RBC seed-grow segmentation controls."""
    defaults = dict(defaults or {})
    st.sidebar.header("RBC Seed-Grow Segmentation")
    method_default = str(defaults.get("rbcAnalysisMethod", RBC_ANALYSIS_METHODS[0]))
    rbc_method = st.sidebar.selectbox(
        "RBC analysis method",
        RBC_ANALYSIS_METHODS,
        index=RBC_ANALYSIS_METHODS.index(method_default) if method_default in RBC_ANALYSIS_METHODS else 0,
        key=control_key(key_prefix, "rbcAnalysisMethod"),
        help="Standard preserves existing results. Conservative adds a no-POD corrected-mask pipeline. Experimental uses POD-removed images as the main mask source.",
    )
    if rbc_method != "Standard RBC Gap Mode":
        st.sidebar.info(
            "Conservative RBC mode is a separate analysis method. The main Conservative mask uses the corrected no-POD image; "
            "POD-removed masks are QC/diagnostic unless Experimental POD-removed mode is selected."
        )
    safety_profiles = ["MicroStructAI Safe Mode", "MATLAB-compatible mode"]
    safety_default = str(defaults.get("rbcGapSafetyMode", safety_profiles[0]))
    safety_mode = st.sidebar.selectbox(
        "RBC gap safety profile",
        safety_profiles,
        index=safety_profiles.index(safety_default) if safety_default in safety_profiles else 0,
        key=control_key(key_prefix, "rbc_gap_safety_profile"),
        help="Safe Mode removes suspiciously large single gap objects. MATLAB-compatible mode preserves the legacy mask by default.",
    )
    if safety_mode != safety_default:
        max_gap_enabled_default = safety_mode == "MicroStructAI Safe Mode"
    else:
        max_gap_enabled_default = bool(defaults.get("enableMaxGapAreaFilter", safety_mode == "MicroStructAI Safe Mode"))
    enable_max_gap_filter = st.sidebar.checkbox(
        "Enable maximum single-gap artefact filter",
        value=max_gap_enabled_default,
        key=control_key(key_prefix, f"enable_max_gap_area_filter_{safe_filename_token(safety_mode)}"),
        help="Remove any individual plasma-gap object larger than maxGapAreaFrac of the cropped image area.",
    )
    max_gap_area_frac = st.sidebar.number_input(
        "maxGapAreaFrac",
        min_value=0.001,
        max_value=0.35,
        value=float(defaults.get("maxGapAreaFrac", 0.10)),
        step=0.001,
        key=control_key(key_prefix, f"max_gap_area_frac_{safe_filename_token(safety_mode)}"),
        help="Recommended range: 0.05 to 0.20. Advanced testing range: 0.001 to 0.35. Suspicious connected gap objects above this fraction are removed when the filter is enabled.",
    )
    if safety_mode == "MATLAB-compatible mode" and enable_max_gap_filter:
        st.sidebar.caption("The filter is manually enabled. Disable it to preserve legacy MATLAB mask behaviour.")
    overlay_background_default = str(defaults.get("overlay_background", "Illumination-corrected Wz"))
    overlay_background = st.sidebar.selectbox(
        "Overlay background",
        RBC_OVERLAY_BACKGROUNDS,
        index=RBC_OVERLAY_BACKGROUNDS.index(overlay_background_default) if overlay_background_default in RBC_OVERLAY_BACKGROUNDS else 1,
        key=control_key(key_prefix, "overlay_background"),
    )
    lf_options = ["MATLAB-compatible continuous-field", "MVP mask-based"]
    lf_default = str(defaults.get("lf_hf_method", lf_options[0]))
    lf_hf_method = st.sidebar.selectbox(
        "LF/HF method",
        lf_options,
        index=lf_options.index(lf_default) if lf_default in lf_options else 0,
        key=control_key(key_prefix, "lf_hf_method"),
    )
    source_options = ["MATLAB Iraw01 processed grayscale", "Raw grayscale crop (legacy app)"]
    source_default = str(defaults.get("rbcSegmentationInputSource", source_options[0]))
    segmentation_input_source = st.sidebar.selectbox(
        "RBC segmentation input source",
        source_options,
        index=source_options.index(source_default) if source_default in source_options else 0,
        key=control_key(key_prefix, "rbcSegmentationInputSource"),
        help=(
            "MATLAB reference masks are generated from the processed Iraw01 analysis grayscale, not the untouched camera crop. "
            "Use the legacy source only to diagnose old app-vs-MATLAB differences."
        ),
    )
    st.sidebar.caption("RBC seed/grow mode does not use adaptive thresholding; kHi/kLo are applied to analysis Wz.")
    with st.sidebar.expander("MATLAB preprocessing compatibility", expanded=False):
        ximg_bg_sigma = float(st.number_input("Iraw01 bgSigma", min_value=0.0, max_value=500.0, value=float(defaults.get("ximg_bgSigma", 40.0)), step=1.0, key=control_key(key_prefix, "ximg_bgSigma")))
        ximg_p_low = float(st.number_input("Iraw01 pLow", min_value=0.0, max_value=100.0, value=float(defaults.get("ximg_pLow", 1.0)), step=0.5, key=control_key(key_prefix, "ximg_pLow")))
        ximg_p_high = float(st.number_input("Iraw01 pHigh", min_value=0.0, max_value=100.0, value=float(defaults.get("ximg_pHigh", 99.0)), step=0.5, key=control_key(key_prefix, "ximg_pHigh")))
        wz_mad_floor = float(st.number_input("wzMadFloor", min_value=0.0, max_value=1000.0, value=float(defaults.get("wzMadFloor", 0.0)), step=0.001, key=control_key(key_prefix, "wzMadFloor")))
        wz_smooth_sigma = float(st.number_input("wzSmoothSigma", min_value=0.0, max_value=100.0, value=float(defaults.get("wzSmoothSigma", 0.0)), step=0.25, key=control_key(key_prefix, "wzSmoothSigma")))
        st.caption("Default MATLAB compatibility uses bgSigma=40, pLow=1, pHigh=99, wzMadFloor=0, wzSmoothSigma=0.")
    params_out = {
        "rbcAnalysisMethod": rbc_method,
        "rbcMode": rbc_method,
        "rbcGapSafetyMode": safety_mode,
        "enableMaxGapAreaFilter": bool(enable_max_gap_filter),
        "maxGapAreaFrac": float(max_gap_area_frac),
        "overlay_background": overlay_background,
        "lf_hf_method": lf_hf_method,
        "rbcSegmentationInputSource": segmentation_input_source,
        "ximg_bgSigma": ximg_bg_sigma,
        "ximg_pLow": ximg_p_low,
        "ximg_pHigh": ximg_p_high,
        "wzMadFloor": wz_mad_floor,
        "wzSmoothSigma": wz_smooth_sigma,
        "cropFrac": st.sidebar.slider("cropFrac", min_value=0.10, max_value=1.00, value=float(defaults.get("cropFrac", 0.70)), step=0.05, key=control_key(key_prefix, "cropFrac")),
        "useMedian": st.sidebar.checkbox("useMedian", value=bool(defaults.get("useMedian", True)), key=control_key(key_prefix, "useMedian")),
        "medianK": st.sidebar.slider("medianK", min_value=1, max_value=15, value=int(defaults.get("medianK", 3)), step=2, key=control_key(key_prefix, "medianK")),
        "sigmaIll": st.sidebar.slider("sigmaIll", min_value=1.0, max_value=80.0, value=float(defaults.get("sigmaIll", 25.0)), step=1.0, key=control_key(key_prefix, "sigmaIll")),
        "trackDark": st.sidebar.checkbox("trackDark", value=bool(defaults.get("trackDark", False)), key=control_key(key_prefix, "trackDark")),
        "kHi": st.sidebar.number_input("kHi", min_value=0.1, max_value=20.0, value=float(defaults.get("kHi", 2.2)), step=0.1, key=control_key(key_prefix, "kHi")),
        "kLo": st.sidebar.number_input("kLo", min_value=0.1, max_value=20.0, value=float(defaults.get("kLo", 1.1)), step=0.1, key=control_key(key_prefix, "kLo")),
        "minAreaPx": st.sidebar.number_input("minAreaPx", min_value=0, max_value=100000, value=int(defaults.get("minAreaPx", 20)), step=5, key=control_key(key_prefix, "minAreaPx")),
        "closeRadius": st.sidebar.number_input("closeRadius", min_value=0, max_value=25, value=int(defaults.get("closeRadius", 2)), step=1, key=control_key(key_prefix, "closeRadius")),
        "doFillHoles": st.sidebar.checkbox("doFillHoles", value=bool(defaults.get("doFillHoles", True)), key=control_key(key_prefix, "doFillHoles")),
        "clearBorder": st.sidebar.checkbox("clearBorder", value=bool(defaults.get("clearBorder", False)), key=control_key(key_prefix, "clearBorder")),
        "boundaryLineWidth": st.sidebar.slider("boundaryLineWidth", min_value=1, max_value=6, value=int(defaults.get("boundaryLineWidth", 1)), step=1, key=control_key(key_prefix, "boundaryLineWidth")),
    }
    params_out.update(
        intensity_correction_controls(
            defaults,
            default_mode="off" if rbc_method == "Standard RBC Gap Mode" else "auto",
            key_prefix=f"{key_prefix}_rbc" if key_prefix else "rbc",
            expanded=False,
        )
    )
    if rbc_method == "Standard RBC Gap Mode":
        with st.sidebar.expander("Optional Standard RBC intensity corrections", expanded=False):
            params_out["standard_enableRawGainCorrection"] = params_out["intensityCorrectionMode"] != "off"
            params_out["standard_enableSpatialFlatfieldCorrection"] = bool(
                st.checkbox("standard_enableSpatialFlatfieldCorrection", value=bool(defaults.get("standard_enableSpatialFlatfieldCorrection", False)), key=control_key(key_prefix, "standard_enableSpatialFlatfieldCorrection"))
            )
            params_out["standard_enableMeanIntensityCorrectionPlot"] = bool(
                st.checkbox("standard_enableMeanIntensityCorrectionPlot", value=bool(defaults.get("standard_enableMeanIntensityCorrectionPlot", True)), key=control_key(key_prefix, "standard_enableMeanIntensityCorrectionPlot"))
            )
            st.caption("Raw-gain correction is controlled by Camera intensity-drop correction above. Off preserves old Standard RBC results.")
    else:
        with st.sidebar.expander("Conservative RBC / POD-QC settings", expanded=True):
            params_out["useCrop"] = bool(st.checkbox("useCrop", value=bool(defaults.get("useCrop", True)), key=control_key(key_prefix, "conservative_useCrop")))
            params_out["cropMode"] = st.selectbox("cropMode", ["center", "manual"], index=0, key=control_key(key_prefix, "conservative_cropMode"))
            params_out["manualCropRect"] = st.text_input("manualCropRect [x,y,width,height]", value=str(defaults.get("manualCropRect", "")), key=control_key(key_prefix, "conservative_manualCropRect"))
            params_out["applySpatialIlluminationCorrection"] = bool(st.checkbox("applySpatialIlluminationCorrection", value=bool(defaults.get("applySpatialIlluminationCorrection", True)), key=control_key(key_prefix, "applySpatialIlluminationCorrection")))
            params_out["spatialCorrectionMethod"] = st.selectbox("spatialCorrectionMethod", ["divide", "subtract"], index=0, key=control_key(key_prefix, "spatialCorrectionMethod"))
            params_out["spatialIllumSigma"] = float(st.number_input("spatialIllumSigma", min_value=3.0, max_value=300.0, value=float(defaults.get("spatialIllumSigma", 60.0)), step=1.0, key=control_key(key_prefix, "spatialIllumSigma")))
            params_out["spatialCorrectionStrength"] = float(st.number_input("spatialCorrectionStrength", min_value=0.0, max_value=1.0, value=float(defaults.get("spatialCorrectionStrength", 0.50)), step=0.05, key=control_key(key_prefix, "spatialCorrectionStrength")))
            params_out["preserveSpatialMeanIntensity"] = bool(st.checkbox("preserveSpatialMeanIntensity", value=bool(defaults.get("preserveSpatialMeanIntensity", True)), key=control_key(key_prefix, "preserveSpatialMeanIntensity")))
            params_out["clipSpatialCorrectedImages"] = bool(st.checkbox("clipSpatialCorrectedImages", value=bool(defaults.get("clipSpatialCorrectedImages", True)), key=control_key(key_prefix, "clipSpatialCorrectedImages")))
            st.caption("Global raw-gain correction is controlled by Camera intensity-drop correction above.")
            params_out["smoothWindowFrames"] = int(st.number_input("smoothWindowFrames", min_value=1, max_value=101, value=int(defaults.get("smoothWindowFrames", 7)), step=2, key=control_key(key_prefix, "smoothWindowFrames")))
            params_out["removeModes"] = int(st.number_input("removeModes", min_value=0, max_value=20, value=int(defaults.get("removeModes", 1)), step=1, key=control_key(key_prefix, "removeModes")))
            experimental_pod_selected = rbc_method == "Experimental POD-removed RBC Mode"
            params_out["computePODWhenRemoveModesZero"] = bool(
                st.checkbox(
                    "MATLAB-style zero-mode POD branch",
                    value=bool(defaults.get("computePODWhenRemoveModesZero", True)),
                    key=control_key(key_prefix, "computePODWhenRemoveModesZero"),
                    help="When removeModes = 0 and a POD branch is active, still run the MATLAB prototype POD reconstruction/rescale branch.",
                )
            )
            params_out["enablePODSensitivityDiagnostic"] = bool(
                st.checkbox(
                    "enablePODSensitivityDiagnostic",
                    value=bool(defaults.get("enablePODSensitivityDiagnostic", not experimental_pod_selected)),
                    key=control_key(key_prefix, "enablePODSensitivityDiagnostic"),
                    help="When enabled, computes the POD-branch mask only as a sensitivity diagnostic in Conservative mode. With removeModes = 0, this matches the MATLAB zero-mode reconstruction/rescale diagnostic.",
                )
            )
            pod_flags_ui = conservative_pod_processing_flags(params_out, rbc_method)
            params_out["enablePODRemoval"] = bool(pod_flags_ui["podFinalEnabled"])
            params_out["enablePODRemovedMode"] = bool(pod_flags_ui["podFinalEnabled"])
            st.info(str(pod_flags_ui["podStatus"]))
            if params_out["enablePODSensitivityDiagnostic"] and not params_out["enablePODRemoval"]:
                st.caption("The Conservative no-POD mask remains the final mask.")
            else:
                st.caption("Conservative QC active.")
            params_out["useMeanSubtract"] = bool(st.checkbox("useMeanSubtract", value=bool(defaults.get("useMeanSubtract", False)), key=control_key(key_prefix, "useMeanSubtract")))
            params_out["enableConservativeQC"] = bool(st.checkbox("enableConservativeQC", value=bool(defaults.get("enableConservativeQC", True)), key=control_key(key_prefix, "enableConservativeQC")))
            params_out["maxAllowedSpatialGapAreaChange"] = float(st.number_input("maxAllowedSpatialGapAreaChange", min_value=0.0, max_value=1.0, value=float(defaults.get("maxAllowedSpatialGapAreaChange", 0.05)), step=0.01, key=control_key(key_prefix, "maxAllowedSpatialGapAreaChange")))
            params_out["maxAllowedRawGainGapAreaChange"] = float(st.number_input("maxAllowedRawGainGapAreaChange", min_value=0.0, max_value=1.0, value=float(defaults.get("maxAllowedRawGainGapAreaChange", 0.05)), step=0.01, key=control_key(key_prefix, "maxAllowedRawGainGapAreaChange")))
            params_out["maxAllowedPODAreaDelta"] = float(st.number_input("maxAllowedPODAreaDelta", min_value=0.0, max_value=1.0, value=float(defaults.get("maxAllowedPODAreaDelta", 0.05)), step=0.01, key=control_key(key_prefix, "maxAllowedPODAreaDelta")))
            params_out["conservativeSegMethod"] = st.selectbox("conservative segmentation method", ["adaptive", "otsu"], index=0, key=control_key(key_prefix, "conservativeSegMethod"))
            adaptive_engine_options = ["MATLAB adaptthresh-like", "Local offset (current)"]
            adaptive_engine_default = str(defaults.get("conservativeAdaptiveEngine", adaptive_engine_options[0]))
            params_out["conservativeAdaptiveEngine"] = st.selectbox(
                "conservative adaptive threshold engine",
                adaptive_engine_options,
                index=adaptive_engine_options.index(adaptive_engine_default) if adaptive_engine_default in adaptive_engine_options else 0,
                key=control_key(key_prefix, "conservativeAdaptiveEngine"),
                help="MATLAB adaptthresh-like uses local-mean scaling measured from MATLAB adaptthresh. Local offset preserves the earlier app threshold_local behaviour.",
            )
            params_out["conservativeAdaptSensitivity"] = float(
                st.number_input(
                    "conservativeAdaptSensitivity",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(defaults.get("conservativeAdaptSensitivity", 0.45)),
                    step=0.01,
                    format="%.2f",
                    key=control_key(key_prefix, "conservativeAdaptSensitivity"),
                    help="Used only by the MATLAB adaptthresh-like engine. Higher sensitivity accepts more bright foreground; MATLAB prototype uses 0.45.",
                )
            )
            params_out["conservativeAdaptMeanScaleBase"] = float(
                st.number_input(
                    "conservativeAdaptMeanScaleBase",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(defaults.get("conservativeAdaptMeanScaleBase", 1.60)),
                    step=0.05,
                    format="%.2f",
                    key=control_key(key_prefix, "conservativeAdaptMeanScaleBase"),
                    help="Used only by the MATLAB adaptthresh-like engine. Threshold multiplier is meanScaleBase - sensitivity; 1.60 with sensitivity 0.45 gives 1.15*local mean.",
                )
            )
            params_out["conservativeAdaptiveOffset"] = float(
                st.number_input(
                    "conservativeAdaptiveOffset",
                    min_value=-0.50,
                    max_value=0.50,
                    value=float(defaults.get("conservativeAdaptiveOffset", -0.043)),
                    step=0.005,
                    format="%.3f",
                    key=control_key(key_prefix, "conservativeAdaptiveOffset"),
                    help="Used for Conservative no-POD segmentation. More negative values are stricter for bright-gap local thresholding. Default -0.043 matches the MATLAB prototype early-frame no-POD area.",
                )
            )
            params_out["podRemovedAdaptiveOffset"] = float(
                st.number_input(
                    "podBranchAdaptiveOffset",
                    min_value=-0.50,
                    max_value=0.50,
                    value=float(defaults.get("podRemovedAdaptiveOffset", -0.070)),
                    step=0.005,
                    format="%.3f",
                    key=control_key(key_prefix, "podRemovedAdaptiveOffset"),
                    help="Advanced override. MATLAB v6 uses the same segmentation settings for no-POD and POD-branch masks unless the separate-offset override is enabled.",
                )
            )
            params_out["useSeparatePODAdaptiveOffset"] = bool(
                st.checkbox(
                    "useSeparatePODAdaptiveOffset",
                    value=bool(defaults.get("useSeparatePODAdaptiveOffset", False)),
                    key=control_key(key_prefix, "useSeparatePODAdaptiveOffset"),
                    help="Off matches the MATLAB v6 prototype: no-POD and POD-branch masks use the same adaptive segmentation settings.",
                )
            )
            st.caption("POD-removed residuals are diagnostic. They are not called true plasma gaps unless Experimental POD-removed mode is selected.")
    with st.sidebar.expander("Morphology & Dynamics (research)", expanded=False):
        params_out["enableStructuralIntelligence"] = bool(
            st.checkbox(
                "Enable additive structural-intelligence layer",
                value=bool(defaults.get("enableStructuralIntelligence", True)),
                key=control_key(key_prefix, "enableStructuralIntelligence"),
                help="Runs only after the existing final mask and scores are complete. It does not change segmentation or existing metrics.",
            )
        )
        params_out["enableCellRichSecondary"] = bool(
            st.checkbox(
                "Analyse secondary dark/cell-rich domains",
                value=bool(defaults.get("enableCellRichSecondary", True)),
                key=control_key(key_prefix, "enableCellRichSecondary"),
            )
        )
        params_out["enableGapDynamics"] = bool(
            st.checkbox(
                "Enable registration and gap dynamics",
                value=bool(defaults.get("enableGapDynamics", True)),
                key=control_key(key_prefix, "enableGapDynamics"),
            )
        )
        params_out["enableStructuralRobustness"] = bool(
            st.checkbox(
                "Enable bounded robustness checks",
                value=bool(defaults.get("enableStructuralRobustness", True)),
                key=control_key(key_prefix, "enableStructuralRobustness"),
            )
        )
        params_out["structuralRobustnessStride"] = int(
            st.number_input(
                "Robustness check every N frames",
                min_value=1,
                max_value=100,
                value=int(defaults.get("structuralRobustnessStride", 10)),
                step=1,
                key=control_key(key_prefix, "structuralRobustnessStride"),
            )
        )
        params_out["structuralMinObjectAreaPx"] = int(
            st.number_input(
                "Advanced minimum object area (px²)",
                min_value=1,
                max_value=100000,
                value=int(defaults.get("structuralMinObjectAreaPx", defaults.get("minAreaPx", 20))),
                step=1,
                key=control_key(key_prefix, "structuralMinObjectAreaPx"),
            )
        )
        params_out["structuralContourMaxMode"] = int(
            st.number_input(
                "Maximum computed contour mode",
                min_value=2,
                max_value=20,
                value=int(defaults.get("structuralContourMaxMode", 10)),
                step=1,
                key=control_key(key_prefix, "structuralContourMaxMode"),
                help="Computed does not mean validated. The robustness layer determines which low-order modes are supportable.",
            )
        )
        params_out["structuralContourMinAreaPx"] = int(
            st.number_input(
                "Minimum fully observed contour area (px²)",
                min_value=20,
                max_value=100000,
                value=int(defaults.get("structuralContourMinAreaPx", 100)),
                step=10,
                key=control_key(key_prefix, "structuralContourMinAreaPx"),
            )
        )
        st.caption("All geometry uses pixels/pixels² unless spatial calibration is supplied. Gap skeletons describe void-network architecture, not RBC strings.")
    return params_out


def rbc_matlab_window_controls(enabled: bool) -> dict[str, Any]:
    """Render MATLAB analysis-window compatibility controls for RBC mode."""
    params: dict[str, Any] = {
        "matlab_window_enabled": bool(enabled),
        "matlab_fps": 1.0,
        "matlab_podStart_s": 10.0,
        "matlab_analysisStart_s": 20.0,
        "matlab_quality_filter_enabled": False,
        "matlab_goodSigmaMAD": 5.0,
        "matlab_analyse_only_isAna": True,
    }
    if enabled:
        st.sidebar.header("MATLAB Analysis-Window Compatibility")
        st.sidebar.caption("MATLAB-compatible window selection overrides manual Start/End image range for RBC analysis.")
        params["matlab_fps"] = float(st.sidebar.number_input("fps", min_value=0.001, max_value=1000.0, value=1.0, step=1.0))
        params["matlab_podStart_s"] = float(st.sidebar.number_input("podStart_s", min_value=0.0, max_value=100000.0, value=10.0, step=1.0))
        params["matlab_analysisStart_s"] = float(st.sidebar.number_input("analysisStart_s", min_value=0.0, max_value=100000.0, value=20.0, step=1.0))
        params["matlab_quality_filter_enabled"] = bool(st.sidebar.checkbox("Use MATLAB-like quality filter", value=False))
        params["matlab_goodSigmaMAD"] = float(st.sidebar.number_input("goodSigmaMAD", min_value=0.0, max_value=1000.0, value=5.0, step=0.5))
        params["matlab_analyse_only_isAna"] = bool(st.sidebar.checkbox("Analyse only MATLAB analysis window", value=True))
    return params


def rbc_qc_controls(defaults: dict[str, Any] | None = None, key_prefix: str = "") -> dict[str, Any]:
    """Render conservative RBC no-aggregation QC thresholds."""
    defaults = dict(defaults or {})
    st.sidebar.header("RBC QC / No-Aggregation Check")
    enable_qc = bool(st.sidebar.checkbox("Enable RBC QC/no-aggregation check", value=bool(defaults.get("enableRbcQc", True)), key=control_key(key_prefix, "enableRbcQc")))
    params: dict[str, Any] = {"enableRbcQc": enable_qc}
    if not enable_qc:
        st.sidebar.caption("QC labels and stable-window suggestions are disabled; RBC score summaries use the selected sequence.")
        with st.sidebar.expander("Sparse-mask morphology filter", expanded=False):
            params["applySparseMaskMorphologyFilter"] = bool(
                st.checkbox(
                    "Set morphology metrics to NaN when mask support is too low",
                    value=bool(defaults.get("applySparseMaskMorphologyFilter", True)),
                    key=control_key(key_prefix, "applySparseMaskMorphologyFilter"),
                    help=(
                        "If the detected mask area is too small, morphology metrics such as Xi and mean gap diameter "
                        "are not reliable. The app will keep raw area/count values but exclude morphology metrics "
                        "from trusted plots and summaries."
                    ),
                )
            )
            params["minReliableMaskAreaFracForMorphology"] = float(
                st.number_input(
                    "minReliableMaskAreaFracForMorphology",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(defaults.get("minReliableMaskAreaFracForMorphology", 0.03)),
                    step=0.005,
                    format="%.4f",
                    key=control_key(key_prefix, "minReliableMaskAreaFracForMorphology"),
                )
            )
            params["minReliableObjectsForMorphology"] = float(
                st.number_input(
                    "minReliableObjectsForMorphology",
                    min_value=0.0,
                    max_value=100000.0,
                    value=float(defaults.get("minReliableObjectsForMorphology", 10.0)),
                    step=1.0,
                    key=control_key(key_prefix, "minReliableObjectsForMorphology"),
                )
            )
        return params
    start_frame_modes = ["Auto first-rise detector", "Dominant transition (mask)", "Manual start", "MATLAB-compatible window"]
    start_frame_default = str(defaults.get("startFrameMode", "Auto first-rise detector"))
    start_frame_mode = st.sidebar.selectbox(
        "RBC start-frame mode",
        start_frame_modes,
        index=start_frame_modes.index(start_frame_default) if start_frame_default in start_frame_modes else 0,
        key=control_key(key_prefix, "startFrameMode"),
    )
    manual_start_s = 20.0
    if start_frame_mode == "Manual start":
        manual_start_s = float(st.sidebar.number_input("manualStart_s", min_value=0.0, max_value=100000.0, value=float(defaults.get("manualStart_s", 20.0)), step=1.0, key=control_key(key_prefix, "manualStart_s")))
    params.update(
        {
            "startFrameMode": start_frame_mode,
            "startupQCMode": start_frame_mode,
            "useAutoStartFrame": start_frame_mode == "Auto first-rise detector",
            "manualStart_s": manual_start_s,
            "autoStartMinDevelopedSignal": float(st.sidebar.number_input("autoStartMinDevelopedSignal", min_value=0.0, max_value=10.0, value=float(defaults.get("autoStartMinDevelopedSignal", 0.45)), step=0.05, key=control_key(key_prefix, "autoStartMinDevelopedSignal"))),
            "autoStartSlopeFrac": float(st.sidebar.number_input("autoStartSlopeFrac", min_value=0.0, max_value=10.0, value=float(defaults.get("autoStartSlopeFrac", 0.40)), step=0.05, key=control_key(key_prefix, "autoStartSlopeFrac"))),
            "autoStartEarlyScaleFrac": float(st.sidebar.number_input("autoStartEarlyScaleFrac", min_value=0.05, max_value=1.0, value=float(defaults.get("autoStartEarlyScaleFrac", 0.35)), step=0.05, key=control_key(key_prefix, "autoStartEarlyScaleFrac"))),
            "autoStartSearchFrac": float(st.sidebar.number_input("autoStartSearchFrac", min_value=0.05, max_value=1.0, value=float(defaults.get("autoStartSearchFrac", 0.40)), step=0.05, key=control_key(key_prefix, "autoStartSearchFrac"))),
            "autoStartSmoothWin_s": float(st.sidebar.number_input("autoStartSmoothWin_s", min_value=1.0, max_value=101.0, value=float(defaults.get("autoStartSmoothWin_s", 7.0)), step=2.0, key=control_key(key_prefix, "autoStartSmoothWin_s"))),
            "autoStartMinPeakSlope": float(st.sidebar.number_input("autoStartMinPeakSlope", min_value=0.0, max_value=10.0, value=float(defaults.get("autoStartMinPeakSlope", 0.02)), step=0.01, key=control_key(key_prefix, "autoStartMinPeakSlope"))),
            "autoStartMinMaxSignal": float(st.sidebar.number_input("autoStartMinMaxSignal", min_value=0.0, max_value=10.0, value=float(defaults.get("autoStartMinMaxSignal", 0.45)), step=0.05, key=control_key(key_prefix, "autoStartMinMaxSignal"))),
            "stableWindowRule": "Auto first-rise single start",
            "minConsecutiveReliableFramesForStableWindow": 1,
            "minConsecutiveStartupReliableFrames": 1,
            "minSeparationZ": float(st.sidebar.number_input("minSeparationZ", min_value=0.0, max_value=10.0, value=float(defaults.get("minSeparationZ", 1.0)), step=0.1, key=control_key(key_prefix, "minSeparationZ"))),
            "minBoundarySupport": float(st.sidebar.number_input("minBoundarySupport", min_value=0.0, max_value=10.0, value=float(defaults.get("minBoundarySupport", 0.30)), step=0.1, key=control_key(key_prefix, "minBoundarySupport"))),
            "minReliableFramePercent": float(st.sidebar.number_input("minReliableFramePercent", min_value=0.0, max_value=100.0, value=float(defaults.get("minReliableFramePercent", 20.0)), step=5.0, key=control_key(key_prefix, "minReliableFramePercent"))),
            "minTemporalTrendRho": float(st.sidebar.number_input("minTemporalTrendRho", min_value=0.0, max_value=1.0, value=float(defaults.get("minTemporalTrendRho", 0.30)), step=0.05, key=control_key(key_prefix, "minTemporalTrendRho"))),
            "minStructuralContrast": float(st.sidebar.number_input("minStructuralContrast", min_value=0.0, max_value=1.0, value=float(defaults.get("minStructuralContrast", 0.10)), step=0.01, key=control_key(key_prefix, "minStructuralContrast"))),
            "minSHI_tileCV_forReliable": float(st.sidebar.number_input("minSHI_tileCV_forReliable", min_value=0.0, max_value=10.0, value=float(defaults.get("minSHI_tileCV_forReliable", 0.15)), step=0.01, key=control_key(key_prefix, "minSHI_tileCV_forReliable"))),
        }
    )
    with st.sidebar.expander("Mask-quality QC / anti-merging diagnostics", expanded=False):
        params["hugeGapFracThreshold"] = float(
            st.number_input(
                "hugeGapFracThreshold",
                min_value=0.001,
                max_value=0.95,
                value=float(defaults.get("hugeGapFracThreshold", defaults.get("maxGapAreaFrac", 0.10))),
                step=0.001,
                format="%.3f",
                key=control_key(key_prefix, "hugeGapFracThreshold"),
                help="Per-frame QC flag threshold for a single suspiciously large connected plasma-gap object.",
            )
        )
        params["largestToMedianGapRatioThreshold"] = float(
            st.number_input(
                "largestToMedianGapRatioThreshold",
                min_value=1.0,
                max_value=10000.0,
                value=float(defaults.get("largestToMedianGapRatioThreshold", 50.0)),
                step=5.0,
                key=control_key(key_prefix, "largestToMedianGapRatioThreshold"),
            )
        )
        params["maxIlluminationCVForReliableMask"] = float(
            st.number_input(
                "maxIlluminationCVForReliableMask",
                min_value=0.0,
                max_value=10.0,
                value=float(defaults.get("maxIlluminationCVForReliableMask", 0.25)),
                step=0.01,
                key=control_key(key_prefix, "maxIlluminationCVForReliableMask"),
            )
        )
        params["minGapNtotalMaskQC"] = float(
            st.number_input("minGapNtotalMaskQC", min_value=0.0, max_value=100000.0, value=float(defaults.get("minGapNtotalMaskQC", 1.0)), step=1.0, key=control_key(key_prefix, "minGapNtotalMaskQC"))
        )
        params["maxGapNtotalMaskQC"] = float(
            st.number_input("maxGapNtotalMaskQC", min_value=1.0, max_value=1000000.0, value=float(defaults.get("maxGapNtotalMaskQC", 100000.0)), step=100.0, key=control_key(key_prefix, "maxGapNtotalMaskQC"))
        )
        params["adaptiveAntiMergingRepair"] = bool(
            st.checkbox(
                "Adaptive anti-merging repair",
                value=bool(defaults.get("adaptiveAntiMergingRepair", False)),
                key=control_key(key_prefix, "adaptiveAntiMergingRepair"),
                help="OFF by default. Re-runs suspicious frames with closeRadius=0 and doFillHoles=False, keeping the repaired mask only when QC improves.",
            )
        )
        params["adaptiveRepairMinScoreImprovement"] = float(
            st.number_input(
                "adaptiveRepairMinScoreImprovement",
                min_value=0.0,
                max_value=100.0,
                value=float(defaults.get("adaptiveRepairMinScoreImprovement", 5.0)),
                step=1.0,
                key=control_key(key_prefix, "adaptiveRepairMinScoreImprovement"),
            )
        )
        params["excludeBadMaskQCFromMainSummary"] = bool(
            st.checkbox(
                "Exclude bad post-start QC frames from trusted analysis",
                value=bool(
                    defaults.get(
                        "excludeBadPostStartQCFramesFromTrustedAnalysis",
                        defaults.get("excludeBadMaskQCFromMainSummary", False),
                    )
                ),
                key=control_key(key_prefix, "excludeBadMaskQCFromMainSummary"),
                help="When enabled, post-start frames with hard mask/QC failures are excluded from QC-aware scores and trusted summaries. Warning-only frames remain visible.",
            )
        )
        params["excludeBadPostStartQCFramesFromTrustedAnalysis"] = bool(params["excludeBadMaskQCFromMainSummary"])
        params["excludePreStartFramesFromTrustedAnalysis"] = bool(
            st.checkbox(
                "Exclude pre-start frames from trusted analysis",
                value=bool(defaults.get("excludePreStartFramesFromTrustedAnalysis", True)),
                key=control_key(key_prefix, "excludePreStartFramesFromTrustedAnalysis"),
                help=(
                    "When enabled, frames before the detected/manual start frame are shown as grey/open markers and excluded "
                    "from QC-aware scores and trusted summaries. Turn off to include the full sequence."
                ),
            )
        )
        params["maskReliabilityMinForSummary"] = float(
            st.number_input(
                "maskReliabilityMinForSummary",
                min_value=0.0,
                max_value=100.0,
                value=float(defaults.get("maskReliabilityMinForSummary", 60.0)),
                step=5.0,
                key=control_key(key_prefix, "maskReliabilityMinForSummary"),
            )
        )
        params["useMaskXiForRBC"] = bool(
            st.checkbox(
                "Use mask-based Xi for trusted RBC scores",
                value=bool(defaults.get("useMaskXiForRBC", True)),
                key=control_key(key_prefix, "useMaskXiForRBC"),
                help="Recommended ON. Keeps continuous-image Xi as a diagnostic but uses final-mask Xi for QC-gated SASI/GapScore/PTI/AggIndex.",
            )
        )
        params["minReliableGapAreaFracForXi"] = float(
            st.number_input(
                "minReliableGapAreaFracForXi",
                min_value=0.0,
                max_value=1.0,
                value=float(defaults.get("minReliableGapAreaFracForXi", 0.03)),
                step=0.005,
                format="%.4f",
                key=control_key(key_prefix, "minReliableGapAreaFracForXi"),
            )
        )
        params["minReliableObjectsForXi"] = float(
            st.number_input(
                "minReliableObjectsForXi",
                min_value=0.0,
                max_value=100000.0,
                value=float(defaults.get("minReliableObjectsForXi", 10.0)),
                step=1.0,
                key=control_key(key_prefix, "minReliableObjectsForXi"),
            )
        )
        params["rejectUniformXi"] = bool(
            st.checkbox(
                "Reject Xi on near-uniform masks (prevents inflated SASI/AggIndex)",
                value=bool(defaults.get("rejectUniformXi", True)),
                key=control_key(key_prefix, "rejectUniformXi"),
                help=(
                    "When on, Xi_mask is set to NaN for nearly empty, nearly full, "
                    "or too-few-object masks, so structureless frames cannot give an "
                    "artificially long correlation length and inflate SASI/AggIndex."
                ),
            )
        )
        params["maxReliableAreaFracForXi"] = float(
            st.number_input(
                "maxReliableAreaFracForXi (upper area guard)",
                min_value=0.0,
                max_value=1.0,
                value=float(defaults.get("maxReliableAreaFracForXi", 0.97)),
                step=0.005,
                format="%.4f",
                key=control_key(key_prefix, "maxReliableAreaFracForXi"),
                help="Masks more filled than this fraction are treated as near-uniform and Xi is rejected.",
            )
        )
        params["applySparseMaskMorphologyFilter"] = bool(
            st.checkbox(
                "Set morphology metrics to NaN when mask support is too low",
                value=bool(defaults.get("applySparseMaskMorphologyFilter", True)),
                key=control_key(key_prefix, "applySparseMaskMorphologyFilter"),
                help=(
                    "If the detected mask area is too small, morphology metrics such as Xi and mean gap diameter "
                    "are not reliable. The app will keep raw area/count values but exclude morphology metrics "
                    "from trusted plots and summaries."
                ),
            )
        )
        params["minReliableMaskAreaFracForMorphology"] = float(
            st.number_input(
                "minReliableMaskAreaFracForMorphology",
                min_value=0.0,
                max_value=1.0,
                value=float(defaults.get("minReliableMaskAreaFracForMorphology", 0.03)),
                step=0.005,
                format="%.4f",
                key=control_key(key_prefix, "minReliableMaskAreaFracForMorphology"),
            )
        )
        params["minReliableObjectsForMorphology"] = float(
            st.number_input(
                "minReliableObjectsForMorphology",
                min_value=0.0,
                max_value=100000.0,
                value=float(defaults.get("minReliableObjectsForMorphology", 10.0)),
                step=1.0,
                key=control_key(key_prefix, "minReliableObjectsForMorphology"),
            )
        )
        params["maskReliabilityMinForXi"] = float(
            st.number_input(
                "maskReliabilityMinForXi",
                min_value=0.0,
                max_value=100.0,
                value=float(defaults.get("maskReliabilityMinForXi", 60.0)),
                step=5.0,
                key=control_key(key_prefix, "maskReliabilityMinForXi"),
            )
        )
    params.update(startup_transient_fallback_controls(defaults={**defaults, **params}, key_prefix=key_prefix, mode_label="RBC Gap Mode"))
    return params


def rbc_sasi_syllectogram_research_controls(defaults: dict[str, Any] | None = None, key_prefix: str = "") -> dict[str, Any]:
    """Render optional RBC SASI pseudo-syllectogram validation controls."""
    defaults = dict(defaults or {})
    with st.sidebar.expander("Advanced SASI pseudo-syllectogram output", expanded=False):
        enabled = bool(
            st.checkbox(
                "Enable SASI pseudo-syllectogram research output",
                value=bool(defaults.get("enableSASISyllectogramResearchOutput", False)),
                key=control_key(key_prefix, "enableSASISyllectogramResearchOutput"),
            )
        )
        st.caption(
            "Optional research/validation output only. It derives a RheoScan-like area-ratio from fitted SASI(t) "
            "using image/time 20 onward. It is not the official RheoScan AI."
        )
        smooth_window = int(
            st.number_input(
                "SASI pseudo-syllectogram smoothing window (frames)",
                min_value=1,
                max_value=51,
                value=int(defaults.get("sasiSyllectogramSmoothWindow", 5)),
                step=2,
                key=control_key(key_prefix, "sasiSyllectogramSmoothWindow"),
            )
        )
    return {
        "enableSASISyllectogramResearchOutput": enabled,
        "sasiSyllectogramStartThreshold": 20.0,
        "sasiSyllectogramSmoothWindow": smooth_window,
    }


def rbc_analysis_window_controls(qc_enabled: bool, total_images: int) -> dict[str, Any]:
    """Render RBC main-summary window controls."""
    st.sidebar.header("RBC Analysis Window")
    default_index = RBC_ANALYSIS_WINDOW_OPTIONS.index("QC-suggested stable window") if qc_enabled else 0
    mode = st.sidebar.selectbox("Analysis window", RBC_ANALYSIS_WINDOW_OPTIONS, index=default_index)
    default_exclude_low_conf = bool(qc_enabled and mode != "MATLAB-compatible window")
    params: dict[str, Any] = {
        "rbc_analysis_window_mode": mode,
        "rbc_fixed_start_frame_1based": 1,
        "excludeLowConfidenceFromMainSummary": False,
    }
    if qc_enabled:
        params["excludeLowConfidenceFromMainSummary"] = bool(
            st.sidebar.checkbox(
                "Exclude weak/pre-gap and possible artefact frames from main summary",
                value=default_exclude_low_conf,
                key=f"exclude_low_conf_main_{mode}",
            )
        )
    else:
        st.sidebar.caption("Low-confidence frame exclusion is unavailable because RBC QC is disabled.")
    if mode == "QC-suggested stable window":
        st.sidebar.caption("Main sample summaries use the configured stable run of consecutive reliable QC frames. Raw per-frame metrics are still exported.")
    elif mode == "Full sequence":
        st.sidebar.caption("Main sample summaries include all analysed frames, including any early low-confidence frames.")
    elif mode == "Fixed start frame":
        params["rbc_fixed_start_frame_1based"] = int(
            st.sidebar.number_input(
                "Fixed main-summary start frame (1-based)",
                min_value=1,
                max_value=max(1, int(total_images or 1)),
                value=1,
                step=1,
            )
        )
    elif mode == "MATLAB-compatible window":
        st.sidebar.caption("Use MATLAB podStart/time/isAna logic for analysis and keep QC labels as advisory.")
    return params


def web_archive_controls(container, key: str) -> tuple[dict, str]:
    """Keep validated sample archives in the current visitor's session only."""
    upload = container.file_uploader("Upload sample folders as ZIP", type=["zip"], key=f"{key}_zip")
    container.caption("ZIP layout: sample_1/frame_001.png, sample_2/frame_001.png. "
                      "An outer dataset folder and sample/images subfolders are supported. "
                      "Limits: 350 MiB ZIP, 512 MiB expanded, 5,000 entries, 30 MiB and 8 million pixels per image.")
    state_key = f"{key}_archive_data"
    if upload is None:
        st.session_state.pop(state_key, None)
        return {}, ""
    payload = upload.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    cached = st.session_state.get(state_key)
    if not cached or cached["digest"] != digest:
        st.session_state.pop(state_key, None)
        try:
            cached = {"digest": digest, "samples": read_sample_archive(payload)}
        except Exception as exc:
            container.error(f"Could not load the sample ZIP: {exc}")
            return {}, ""
        st.session_state[state_key] = cached
    samples = cached["samples"]
    container.success(f"{len(samples)} samples, {sum(len(files) for files in samples.values())} images loaded.")
    return samples, f"uploaded:{digest}"


def batch_validation_sidebar_controls() -> dict[str, Any]:
    """Render mode-specific batch validation settings."""
    st.sidebar.header("Batch Validation")
    analysis_mode = st.sidebar.selectbox("Batch validation analysis mode", BATCH_VALIDATION_ANALYSIS_MODES, index=0)
    is_rbc = analysis_mode == "RBC Gap Mode"
    batch_input_options = ["Image folders only (no MATLAB CSV)", "Images + optional MATLAB CSV comparison"] if is_rbc else ["Image folders only (no MATLAB CSV)"]
    batch_input_mode = st.sidebar.selectbox("Batch input type", batch_input_options, index=0)
    if WEB_DEPLOYMENT:
        samples, parent_folder = web_archive_controls(st.sidebar, "batch")
        st.session_state["web_batch_samples"] = samples
    else:
        parent_folder = st.sidebar.text_input("Parent sample folder path", value="", placeholder=r"C:\path\to\samples")
    use_current_settings = st.sidebar.checkbox(
        "Use current RBC Gap Mode settings" if is_rbc else "Use shared Object Mode settings",
        value=True,
    )
    st.sidebar.caption("All samples use the shared analysis settings below.")

    params: dict[str, Any] = {
        "batch_validation_mode": analysis_mode,
        "batch_parent_folder": parent_folder,
        "batch_input_mode": batch_input_mode,
        "batch_use_matlab_csv": bool(is_rbc and batch_input_mode != "Image folders only (no MATLAB CSV)"),
        "batch_use_current_rbc_settings": bool(use_current_settings),
        "input_method": "Uploaded sample ZIP" if WEB_DEPLOYMENT else "Local folder path",
        "overlay_style": "Boundary" if is_rbc else "Filled",
        "plot_x_axis": "originalImageIndex_1based",
    }
    if is_rbc:
        preset_defaults, preset_name = rbc_camera_preset_controls()
        params["cameraPresetName"] = preset_name
        params.update(rbc_segmentation_controls(preset_defaults, key_prefix="web_batch"))
        params.update(rbc_qc_controls(preset_defaults, key_prefix="web_batch"))
        params.update(rbc_sasi_syllectogram_research_controls(key_prefix="batch_rbc"))
        if params.get("startFrameMode") == "MATLAB-compatible window":
            params.update(rbc_analysis_window_controls(params.get("enableRbcQc", True), total_images=100000))
            params["rbc_analysis_window_mode"] = "MATLAB-compatible window"
        else:
            params["rbc_analysis_window_mode"] = params.get("startFrameMode", "Auto first-rise detector")
            params["excludeLowConfidenceFromMainSummary"] = False
        params.update(rbc_matlab_window_controls(params.get("rbc_analysis_window_mode") == "MATLAB-compatible window"))
    elif analysis_mode == "Standard Object Mode":
        params.update(
            object_segmentation_controls(
                "Batch Standard Object Segmentation",
                allow_enhanced=False,
                force_object_method="Standard Object Mode",
            )
        )
    elif analysis_mode == "Enhanced Object Mode":
        params.update(
            object_segmentation_controls(
                "Batch Enhanced Object Segmentation",
                allow_enhanced=False,
                force_object_method="Enhanced Object Mode",
            )
        )
    elif analysis_mode == "Conservative Object Mode":
        params.update(
            object_segmentation_controls(
                "Batch Conservative Object Segmentation",
                allow_enhanced=False,
                force_object_method="Conservative Object Mode",
            )
        )
    elif analysis_mode == "Experimental POD-removed Object Mode":
        params.update(
            object_segmentation_controls(
                "Batch Experimental POD-removed Object Segmentation",
                allow_enhanced=False,
                force_object_method="Experimental POD-removed Object Mode",
            )
        )
    else:
        st.sidebar.info("Ground-Truth Mask Mode batch validation is not available yet.")
    return params


CALIBRATION_MODES = [
    "Quick calibration from one positive RBC sequence",
    "Full calibration with positive + negative/control sequences",
    "Validate existing preset",
]


def camera_calibration_sidebar_controls() -> dict[str, Any]:
    """Render setup-specific RBC camera calibration controls."""
    st.sidebar.header("Camera Calibration")
    calibration_mode = st.sidebar.selectbox("Calibration mode", CALIBRATION_MODES, index=0)
    params: dict[str, Any] = {
        "calibrationMode": calibration_mode,
        "input_method": "Upload image files",
        "overlay_style": "Boundary",
        "plot_x_axis": "matlabTime_s",
    }
    if calibration_mode == "Validate existing preset":
        preset_file = st.sidebar.file_uploader("Load camera preset JSON", type=["json"], key="calibration_validate_preset_json")
        params["validatePresetPayload"] = preset_file.getvalue() if preset_file is not None else None
        st.sidebar.caption("Validation runs the loaded setup preset without changing RBC formulas.")
        return params

    calibration_input_method = st.sidebar.selectbox(
        "Calibration image input method",
        INPUT_METHODS,
        index=0,
    )
    params["calibrationInputMethod"] = calibration_input_method
    if calibration_input_method == "Upload image files":
        params["positiveUploads"] = st.sidebar.file_uploader(
            "Upload positive RBC image files",
            type=SUPPORTED_IMAGE_TYPES,
            accept_multiple_files=True,
            key="calibration_positive_image_uploads",
        ) or []
        params["positiveFolder"] = ""
    else:
        st.sidebar.info("Local folder path mode works only when the app is running locally on your computer.")
        params["positiveFolder"] = st.sidebar.text_input("Positive RBC image folder", value="", placeholder=r"C:\...\positive_RBC_sequence")
        params["positiveUploads"] = []
    if calibration_mode == "Full calibration with positive + negative/control sequences":
        if calibration_input_method == "Upload image files":
            params["negativeUploads"] = st.sidebar.file_uploader(
                "Upload negative/control image files",
                type=SUPPORTED_IMAGE_TYPES,
                accept_multiple_files=True,
                key="calibration_negative_image_uploads",
            ) or []
            params["negativeFolder"] = ""
        else:
            params["negativeFolder"] = st.sidebar.text_input("Negative/control image folder", value="", placeholder=r"C:\...\PBS_or_no_gap_control")
            params["negativeUploads"] = []
        params["negativeControlType"] = st.sidebar.selectbox("Negative/control type", ["PBS", "blank", "no blood", "early no-gap", "other"], index=0)
    else:
        params["negativeFolder"] = ""
        params["negativeUploads"] = []
        params["negativeControlType"] = "not provided"
        st.sidebar.warning("Calibration confidence is limited because no negative/no-aggregation control was provided.")

    mark_visual_start = st.sidebar.checkbox("Use user-marked visual start frame", value=False)
    params["userStartFrame"] = (
        int(st.sidebar.number_input("Positive visual start frame (1-based)", min_value=1, max_value=100000, value=1, step=1))
        if mark_visual_start
        else None
    )
    params["basePresetName"] = st.sidebar.selectbox("Base RBC preset", list(DEFAULT_CAMERA_PRESETS), index=0)
    params["maxCalibrationTrials"] = int(st.sidebar.number_input("maxCalibrationTrials", min_value=1, max_value=1000, value=100, step=10))
    with st.sidebar.expander("Calibration plausibility thresholds", expanded=False):
        params["minPlausibleGapAreaFrac"] = float(st.number_input("minPlausibleGapAreaFrac", min_value=0.0, max_value=1.0, value=0.01, step=0.01))
        params["maxPlausibleGapAreaFrac"] = float(st.number_input("maxPlausibleGapAreaFrac", min_value=0.0, max_value=1.0, value=0.60, step=0.05))
        params["minPlausibleGapNtotal"] = int(st.number_input("minPlausibleGapNtotal", min_value=0, max_value=100000, value=20, step=5))
        params["minPlausibleGapMeanDiam_px"] = float(st.number_input("minPlausibleGapMeanDiam_px", min_value=0.0, max_value=1000.0, value=2.0, step=0.5))
        params["maxTinyObjectWarning"] = float(st.number_input("maxTinyObjectWarning", min_value=0.0, max_value=1.0, value=0.30, step=0.05))
        params["maxHugeConnectedRegionWarning"] = float(st.number_input("maxHugeConnectedRegionWarning", min_value=0.0, max_value=1.0, value=0.80, step=0.05))
    params["presetName"] = st.sidebar.text_input("Recommended preset name", value="New camera setup")
    params["cameraSetupNotes"] = st.sidebar.text_area("Camera/setup notes", value="")
    params["presetUserNotes"] = st.sidebar.text_area("Preset notes", value="")
    return params


def build_sidebar() -> tuple[str, list, list, dict[str, Any], bool]:
    """Render sidebar controls and return user selections."""
    st.sidebar.header("Analysis")
    mode = st.sidebar.selectbox("Analysis mode", MODE_OPTIONS)
    if mode == "Batch Validation":
        params = batch_validation_sidebar_controls()
        return mode, [], [], params, False
    if mode == "Multi-Sample Comparison":
        return mode, [], [], {}, False
    if mode == "Auto-Tuning / Camera Calibration":
        params = camera_calibration_sidebar_controls()
        return mode, [], [], params, False

    input_method = st.sidebar.selectbox("Input method", INPUT_METHODS, index=0)
    image_refs, mask_refs, input_info, input_warnings = collect_input_files(mode, input_method)

    if mode == "Ground-Truth Mask Mode":
        st.sidebar.header("Ground Truth Masks")
        params: dict[str, Any] = {
            **input_info,
            "matching_strategy": st.sidebar.selectbox("Matching strategy", ["match by sorted order", "match by filename numbers"], index=0),
            "mask_type": st.sidebar.selectbox("Mask type", ["Filled binary mask", "Outline mask"], index=1),
            "gt_min_object_size": st.sidebar.number_input("GT min object size", min_value=0, max_value=100000, value=20, step=5),
            "compare_auto": st.sidebar.checkbox("Compare automatic segmentation to ground truth", value=False),
        }
        if params["mask_type"] == "Outline mask":
            params["outline_polarity"] = st.sidebar.selectbox("Outline polarity", ["dark outlines", "bright outlines"], index=0)
        else:
            params["outline_polarity"] = "dark outlines"

        if params["compare_auto"]:
            st.sidebar.header("Automatic Segmentation")
            params["auto_method"] = st.sidebar.selectbox("Automatic segmentation method", ["Object Mode segmentation", "RBC Gap Mode segmentation"], index=0)
            if params["auto_method"] == "RBC Gap Mode segmentation":
                params["overlay_style"] = "Boundary"
                params.update(
                    rbc_segmentation_controls(
                        {"rbcGapSafetyMode": "MATLAB-compatible mode", "enableMaxGapAreaFilter": False},
                        key_prefix="gt_auto_rbc",
                    )
                )
            else:
                params.update(object_segmentation_controls("Automatic Object Segmentation", allow_enhanced=False, allow_experimental_scores=False))
        else:
            params["auto_method"] = "None"

        pairs, match_warnings = match_image_mask_files(image_refs, mask_refs, params["matching_strategy"])
        range_info, range_warnings = range_controls(len(image_refs), mode)
        selected_pairs = filter_pairs_by_image_range(pairs, range_info)
        image_files = [pair[0] for pair in selected_pairs]
        mask_files = [pair[1] for pair in selected_pairs]
        params.update(range_info)
        params["matched_pairs_before_filtering"] = len(pairs)
        params["selected_image_count"] = len(image_files)
        params["pre_analysis_warnings"] = input_warnings + match_warnings + range_warnings
    else:
        overlay_styles = ["Filled", "Boundary", "Both"]
        default_overlay_index = 1 if mode == "RBC Gap Mode" else 0
        overlay_style = st.sidebar.selectbox("Overlay style", overlay_styles, index=default_overlay_index)
        if mode == "RBC Gap Mode":
            preset_defaults, preset_name = rbc_camera_preset_controls()
            preset_key = safe_filename_token(preset_name)
            params = {**input_info, "overlay_style": overlay_style, "cameraPresetName": preset_name}
            params.update(rbc_segmentation_controls(preset_defaults, key_prefix=f"single_{preset_key}"))
            params.update(rbc_qc_controls(preset_defaults, key_prefix=f"single_{preset_key}"))
            params.update(rbc_sasi_syllectogram_research_controls(preset_defaults, key_prefix=f"single_{preset_key}"))
            if params.get("startFrameMode") == "MATLAB-compatible window":
                params.update(rbc_analysis_window_controls(params.get("enableRbcQc", True), len(image_refs)))
                params["rbc_analysis_window_mode"] = "MATLAB-compatible window"
            else:
                params["rbc_analysis_window_mode"] = params.get("startFrameMode", "Auto first-rise detector")
                params["excludeLowConfidenceFromMainSummary"] = False
            params.update(rbc_matlab_window_controls(params.get("rbc_analysis_window_mode") == "MATLAB-compatible window"))
        else:
            params = {**input_info, "overlay_style": overlay_style}
            params.update(object_segmentation_controls())
        range_info, range_warnings = range_controls(len(image_refs), mode)
        if mode == "RBC Gap Mode" and params.get("matlab_window_enabled", False):
            image_files = image_refs
            st.sidebar.info("MATLAB compatibility window is enabled and overrides manual Start/End image range during analysis.")
        else:
            image_files = apply_range_to_refs(image_refs, range_info)
        mask_files = []
        params.update(range_info)
        params["selected_image_count"] = len(image_files)
        params["pre_analysis_warnings"] = input_warnings + range_warnings

    for warning in params.get("pre_analysis_warnings", []):
        st.sidebar.warning(warning)

    run_clicked = st.sidebar.button("Run Analysis", type="primary")
    return mode, image_files or [], mask_files or [], params, run_clicked


def segment_object_like(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Run generic bright/dark object segmentation."""
    if params.get("enhancedObjectModeEnabled", False):
        return segment_enhanced_structures_seedgrow(image, params)["structure_mask"]
    if params.get("polarity") == "dark":
        return segment_dark_objects(image, params)
    return segment_bright_objects(image, params)


def segment_object_with_details(image: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    """Run Object Mode segmentation and preserve enhanced preview diagnostics."""
    if params.get("enhancedObjectModeEnabled", False):
        return segment_enhanced_structures_seedgrow(image, params)
    mask = segment_object_like(image, params)
    return {
        "structure_mask": mask,
        "raw_gray": image,
        "removed_structure_mask": np.zeros_like(mask, dtype=bool),
        "structure_diagnostics": {
            "enhancedObjectModeEnabled": False,
            "objectSegmentationTarget": params.get("objectSegmentationTarget", params.get("polarity", "bright")),
            "objectSegmentationTargetUsed": params.get("objectSegmentationTarget", params.get("polarity", "bright")),
            "maxObjectAreaFilterEnabled": False,
            "maxObjectAreaFrac": np.nan,
            "nLargeObjectsRemoved": 0,
            "nLargeObjectsDetected": 0,
            "largestRemovedObjectFrac": 0.0,
            "largestObjectFrac_beforeFilter": np.nan,
            "largestObjectFrac_afterFilter": np.nan,
            "largeObjectArtefactFlag": "none",
        },
    }


def object_mask_component_metrics(mask: np.ndarray) -> dict[str, Any]:
    """Return generic connected-component metrics for a selected object mask."""
    bw = np.asarray(mask, dtype=bool)
    labels, n_components = ndi.label(bw, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel())[1:].astype(np.float64) if n_components else np.asarray([], dtype=np.float64)
    image_area = float(bw.size) if bw.size else np.nan
    diameters = 2.0 * np.sqrt(areas / np.pi) if areas.size else np.asarray([], dtype=float)
    return {
        "objectAreaFrac": float(np.mean(bw)) if bw.size else np.nan,
        "objectNtotal": int(areas.size),
        "objectMeanArea": float(np.mean(areas)) if areas.size else np.nan,
        "objectMedianArea": float(np.median(areas)) if areas.size else np.nan,
        "objectMeanDiam_px": float(np.mean(diameters)) if diameters.size else np.nan,
        "objectMedianDiam_px": float(np.median(diameters)) if diameters.size else np.nan,
        "objectMaxDiam_px": float(np.max(diameters)) if diameters.size else np.nan,
        "objectP90Diam_px": float(np.percentile(diameters, 90)) if diameters.size else np.nan,
        "objectLargestArea": float(np.max(areas)) if areas.size else 0.0,
        "objectLargestFrac": float(np.max(areas) / image_area) if areas.size and image_area else 0.0,
    }


def conservative_object_mask_params(params: dict[str, Any], pod_removed: bool = False) -> dict[str, Any]:
    """Map generic Conservative Object controls onto the shared conservative mask helper."""
    use_separate_pod_offset = bool(params.get("useSeparatePODAdaptiveOffset", False))
    return {
        **params,
        "conservativeSegMethod": params.get("conservativeObjectSegMethod", "adaptive"),
        "minAreaPx": int(params.get("conservativeObjectMinAreaPx", params.get("min_object_size", 20))),
        "conservativeSmoothSigma": float(params.get("conservativeObjectSmoothSigma", 1.0)),
        "conservativeAdaptiveEngine": params.get("conservativeAdaptiveEngine", "MATLAB adaptthresh-like"),
        "conservativeAdaptSensitivity": float(params.get("conservativeAdaptSensitivity", 0.45)),
        "conservativeAdaptMeanScaleBase": float(params.get("conservativeAdaptMeanScaleBase", 1.60)),
        "conservativeAdaptiveOffset": float(
            params.get(
                "podRemovedObjectAdaptiveOffset" if pod_removed and use_separate_pod_offset else "conservativeObjectAdaptiveOffset",
                -0.070 if pod_removed and use_separate_pod_offset else -0.043,
            )
        ),
    }


def conservative_object_target_image(image: np.ndarray, target: str) -> np.ndarray:
    """Return a bright-foreground image for the requested generic object polarity."""
    image01 = to_grayscale_float01(image)
    if str(target).startswith("Dark"):
        return 1.0 - image01
    return image01


def make_conservative_object_mask(image: np.ndarray, params: dict[str, Any], pod_removed: bool = False) -> tuple[np.ndarray, str]:
    """Segment a generic conservative object mask from no-POD or POD-removed imagery."""
    target = str(params.get("objectSegmentationTarget", "Bright structures / bright gaps"))
    mask_params = conservative_object_mask_params(params, pod_removed=pod_removed)
    if target == "Auto polarity":
        bright_mask = make_gap_mask_conservative(conservative_object_target_image(image, "Bright structures / bright gaps"), mask_params)
        dark_mask = make_gap_mask_conservative(conservative_object_target_image(image, "Dark structures / dark gaps"), mask_params)

        def score(mask: np.ndarray) -> float:
            metrics = object_mask_component_metrics(mask)
            area = float(metrics.get("objectAreaFrac", np.nan))
            count = int(metrics.get("objectNtotal", 0) or 0)
            if not np.isfinite(area) or count <= 0:
                return -10.0
            penalty = abs(area - 0.20)
            if area < 0.001 or area > 0.90:
                penalty += 2.0
            return -penalty + min(count, 1000) * 1e-5

        if score(dark_mask) > score(bright_mask):
            return dark_mask.astype(bool), "Dark structures / dark gaps"
        return bright_mask.astype(bool), "Bright structures / bright gaps"

    mask = make_gap_mask_conservative(conservative_object_target_image(image, target), mask_params)
    return mask.astype(bool), target


def object_frame_qc_from_masks(
    final_metrics: dict[str, Any],
    no_pod_metrics: dict[str, Any],
    pod_metrics: dict[str, Any],
    xi_final: float,
    area_delta: float,
    spatial_changed: bool,
    raw_gain_changed: bool,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Create generic Conservative/POD Object QC flags and classes."""
    area = float(final_metrics.get("objectAreaFrac", np.nan))
    count = int(final_metrics.get("objectNtotal", 0) or 0)
    largest = float(final_metrics.get("objectLargestFrac", np.nan))
    min_area = float(params.get("minReliableObjectAreaFracForXi", 0.03))
    min_count = int(params.get("minReliableObjectsForXi", 10))
    max_area = float(params.get("maxObjectAreaFracMaskQC", 0.85))
    huge_threshold = float(params.get("hugeObjectFracThreshold", 0.35))
    max_delta = float(params.get("maxAllowedPODObjectAreaDelta", 0.05))
    pod_diagnostic_active = bool(params.get("podDiagnosticActive", params.get("enablePODSensitivityDiagnostic", False) or params.get("enablePODRemovedMode", False)))

    too_small = (not np.isfinite(area)) or area < min_area or count < min_count
    too_large = np.isfinite(area) and area > max_area
    huge = np.isfinite(largest) and largest > huge_threshold
    low_support = (not np.isfinite(xi_final)) or too_small
    hard_bad = bool(too_small or too_large or huge or low_support)
    pod_sensitive = bool(pod_diagnostic_active and np.isfinite(area_delta) and area_delta > max_delta)
    warning_only = bool(pod_sensitive or spatial_changed or raw_gain_changed)
    reliable = bool(not hard_bad)

    if reliable and not warning_only:
        frame_class = "reliable_object_structure"
        consistency = "POD/no-POD consistent"
    elif reliable and pod_sensitive:
        frame_class = "POD_sensitive_object_result"
        consistency = "POD-sensitive"
    elif reliable and (spatial_changed or raw_gain_changed):
        frame_class = "spatial_correction_sensitive_result"
        consistency = "correction-sensitive"
    elif too_small:
        frame_class = "weak_or_sparse_object_texture"
        consistency = "weak/sparse object mask"
    elif too_large or huge:
        frame_class = "possible_residual_or_illumination_artefact"
        consistency = "implausible large/merged object mask"
    else:
        frame_class = "no_reliable_object_structure"
        consistency = "no reliable object structure"

    return {
        "hardBadObjectMaskFlag": hard_bad,
        "objectMaskReliable": reliable,
        "objectFrameClass": frame_class,
        "objectMaskTooSmallFlag": bool(too_small),
        "objectMaskTooLargeFlag": bool(too_large),
        "objectHugeObjectFlag": bool(huge),
        "objectLowSupportFlag": bool(low_support),
        "podObjectSensitiveFlag": pod_sensitive,
        "spatialCorrectionSensitiveFlag": bool(spatial_changed),
        "rawGainCorrectionSensitiveFlag": bool(raw_gain_changed),
        "conservativeObjectWarningFlag": warning_only,
        "podObjectConsistencyFlag": consistency,
    }


def add_conservative_object_score_columns(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Add optional generic mask-based ObjectStructure/ObjectAggregation scores."""
    out = df.copy()
    enabled = bool(params.get("enableObjectStructureScores", True))
    out["enableObjectStructureScores"] = enabled
    if "objectAreaFrac" not in out.columns:
        out["objectAreaFrac"] = safe_numeric_series(out, "areaFrac")
    if "objectNtotal" not in out.columns:
        out["objectNtotal"] = safe_numeric_series(out, "objectCount").fillna(safe_numeric_series(out, "nObjects"))
    if "objectMeanDiam_px" not in out.columns:
        out["objectMeanDiam_px"] = safe_numeric_series(out, "meanDiam_px").fillna(safe_numeric_series(out, "meanDiam"))
    if "objectLargestFrac" not in out.columns:
        out["objectLargestFrac"] = safe_numeric_series(out, "largestObjectFrac")
    if "Object_Xi_mask" not in out.columns:
        out["Object_Xi_mask"] = safe_numeric_series(out, "correlationLength_px").fillna(safe_numeric_series(out, "Xi_px"))
    if "Object_Xi_QC" not in out.columns:
        confidence = out.get("segmentationConfidence", pd.Series("higher", index=out.index)).fillna("").astype(str).str.lower()
        out["Object_Xi_QC"] = safe_numeric_series(out, "Object_Xi_mask").where(~confidence.eq("low"), np.nan)
    if "objectMaskReliable" not in out.columns:
        out["objectMaskReliable"] = safe_numeric_series(out, "Object_Xi_QC").notna()
    object_sparse_filter_enabled = bool(params.get("applySparseMaskMorphologyFilter", False))
    min_morph_area = float(params.get("minReliableMaskAreaFracForMorphology", params.get("minReliableObjectAreaFracForXi", 0.03)))
    min_morph_objects = float(params.get("minReliableObjectsForMorphology", params.get("minReliableObjectsForXi", 10.0)))
    object_area = safe_numeric_series(out, "objectAreaFrac")
    object_count = safe_numeric_series(out, "objectNtotal")
    object_morphology_reliable = object_area.ge(min_morph_area).fillna(False) & object_count.ge(min_morph_objects).fillna(False)
    object_sparse_filtered = object_sparse_filter_enabled & ~object_morphology_reliable
    out["applySparseMaskMorphologyFilter"] = object_sparse_filter_enabled
    out["minReliableMaskAreaFracForMorphology"] = min_morph_area
    out["minReliableObjectsForMorphology"] = min_morph_objects
    out["maskMorphologyReliable"] = object_morphology_reliable.to_numpy(dtype=bool)
    out["objectMaskMorphologyReliable"] = object_morphology_reliable.to_numpy(dtype=bool)
    out["sparseMaskMorphologyFiltered"] = object_sparse_filtered.to_numpy(dtype=bool)
    out["sparseMaskMorphologyFilteredFrames"] = int(object_sparse_filtered.sum())
    object_trust_mask = ~object_sparse_filtered
    out["raw_objectMeanDiam_px"] = safe_numeric_series(out, "objectMeanDiam_px")
    out["trusted_objectMeanDiam_px"] = safe_numeric_series(out, "objectMeanDiam_px").where(object_trust_mask, np.nan)
    out["raw_Object_Xi_mask"] = safe_numeric_series(out, "Object_Xi_mask")
    out["trusted_Object_Xi_mask"] = safe_numeric_series(out, "Object_Xi_mask").where(object_trust_mask, np.nan)
    out["objectSyllectogramSource_structure"] = "final_selected_object_mask"
    out["objectSyllectogramSource_intensity"] = "corrected_grayscale"
    if not enabled or out.empty:
        for col in [
            "ObjectStructureScore_raw",
            "ObjectStructureScore_raw_0_100",
            "ObjectStructureScore_QC",
            "ObjectStructureScore_QC_0_100",
            "ObjectStructureScore_0_100",
            "ObjectAggregationIndex_raw",
            "ObjectAggregationIndex_raw_0_100",
            "ObjectAggregationIndex_QC",
            "ObjectAggregationIndex_QC_0_100",
            "ObjectAggregationIndex_0_100",
            "ObjectSASI_like_0_100",
            "ObjectAggIndex_like_0_100",
        ]:
            out[col] = np.nan
        out["objectScoreSource"] = "disabled"
        out["Object_Xi_isTrusted"] = False
        out["ObjectStructure_isTrusted"] = False
        out["ObjectAggregation_isTrusted"] = False
        out["Object_Xi_QC_reason"] = "object structure scores disabled"
        out["ObjectStructure_QC_reason"] = "object structure scores disabled"
        out["ObjectAggregation_QC_reason"] = "object structure scores disabled"
        return out

    reliable = out.get("objectMaskReliable", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    xi = safe_numeric_series(out, "Object_Xi_mask")
    diam = safe_numeric_series(out, "objectMeanDiam_px")
    count = safe_numeric_series(out, "objectNtotal")
    area = safe_numeric_series(out, "objectAreaFrac")
    reliable = reliable & xi.notna() & diam.notna() & count.notna() & area.notna()
    min_area = float(params.get("minReliableObjectAreaFracForXi", 0.03))
    min_count = int(params.get("minReliableObjectsForXi", 10))
    reliable = reliable & (area >= min_area) & (count >= min_count)
    if object_sparse_filter_enabled:
        reliable = reliable & object_morphology_reliable

    raw_reference = xi.notna() & diam.notna() & count.notna() & area.notna()
    score_reference = reliable if reliable.any() else raw_reference
    if not raw_reference.any():
        out["ObjectStructureScore_raw"] = np.nan
        out["ObjectStructureScore_raw_0_100"] = np.nan
        out["ObjectStructureScore_0_100"] = np.nan
        out["ObjectStructureScore_QC"] = np.nan
        out["ObjectStructureScore_QC_0_100"] = np.nan
        out["ObjectAggregationIndex_raw"] = np.nan
        out["ObjectAggregationIndex_raw_0_100"] = np.nan
        out["ObjectAggregationIndex_0_100"] = np.nan
        out["ObjectAggregationIndex_QC"] = np.nan
        out["ObjectAggregationIndex_QC_0_100"] = np.nan
        out["ObjectSASI_like_0_100"] = np.nan
        out["ObjectAggIndex_like_0_100"] = np.nan
        out["objectScoreSource"] = "unavailable: no reliable mask-based Object_Xi_mask frames"
        out["Object_Xi_isTrusted"] = False
        out["ObjectStructure_isTrusted"] = False
        out["ObjectAggregation_isTrusted"] = False
        out["Object_Xi_QC_reason"] = "Object_Xi_mask unavailable"
        out["ObjectStructure_QC_reason"] = "Object_Xi_mask, objectMeanDiam_px, objectNtotal, or objectAreaFrac unavailable"
        out["ObjectAggregation_QC_reason"] = out["ObjectStructure_QC_reason"]
        return out

    z_xi = matlab_robust_z(xi, score_reference)
    z_diam = matlab_robust_z(diam, score_reference)
    z_count = matlab_robust_z(count, score_reference)
    z_area = matlab_robust_z(area, score_reference)
    structure_raw = 0.40 * z_xi + 0.40 * z_diam - 0.20 * z_count
    aggregation_raw = 0.30 * z_diam + 0.30 * z_xi + 0.20 * z_area - 0.20 * z_count
    structure_score = sigmoid100(structure_raw)
    aggregation_score = sigmoid100(aggregation_raw)
    out["ObjectStructureScore_raw"] = structure_raw
    out["ObjectStructureScore_0_100"] = structure_score
    out["ObjectStructureScore_raw_0_100"] = structure_score
    out["ObjectStructureScore_QC"] = structure_score.where(reliable, np.nan)
    out["ObjectStructureScore_QC_0_100"] = out["ObjectStructureScore_QC"]
    out["ObjectAggregationIndex_raw"] = aggregation_raw
    out["ObjectAggregationIndex_0_100"] = aggregation_score
    out["ObjectAggregationIndex_raw_0_100"] = aggregation_score
    out["ObjectAggregationIndex_QC"] = aggregation_score.where(reliable, np.nan)
    out["ObjectAggregationIndex_QC_0_100"] = out["ObjectAggregationIndex_QC"]
    out["ObjectSASI_like_0_100"] = out["ObjectStructureScore_QC"]
    out["ObjectAggIndex_like_0_100"] = out["ObjectAggregationIndex_QC"]
    out["objectScoreSource"] = "mask-based Object_Xi_mask + objectMeanDiam_px + objectNtotal + objectAreaFrac"
    out["Object_Xi_isTrusted"] = (reliable & out["Object_Xi_QC"].notna()).to_numpy(dtype=bool)
    out["ObjectStructure_isTrusted"] = reliable.to_numpy(dtype=bool)
    out["ObjectAggregation_isTrusted"] = reliable.to_numpy(dtype=bool)
    object_reason = np.where(reliable.to_numpy(dtype=bool), "trusted", "mask too sparse, too few objects, Object_Xi_mask unavailable, or low object-mask reliability")
    out["Object_Xi_QC_reason"] = np.where(out["Object_Xi_isTrusted"].to_numpy(dtype=bool), "trusted", object_reason)
    out["ObjectStructure_QC_reason"] = object_reason
    out["ObjectAggregation_QC_reason"] = object_reason

    for suffix, xi_col, area_col, diam_col, count_col in [
        ("noPOD", "Object_Xi_mask_noPOD", "conservativeNoPODObjectAreaFrac", "conservativeNoPODObjectMeanDiam_px", "conservativeNoPODObjectNtotal"),
        ("PODremoved", "Object_Xi_mask_PODremoved", "podRemovedObjectAreaFrac", "podRemovedObjectMeanDiam_px", "podRemovedObjectNtotal"),
    ]:
        if all(col in out.columns for col in [xi_col, area_col, diam_col, count_col]):
            xi_s = safe_numeric_series(out, xi_col)
            area_s = safe_numeric_series(out, area_col)
            diam_s = safe_numeric_series(out, diam_col)
            count_s = safe_numeric_series(out, count_col)
            rel_s = xi_s.notna() & area_s.notna() & diam_s.notna() & count_s.notna() & (area_s >= min_area) & (count_s >= min_count)
            if rel_s.any():
                z_xi_s = matlab_robust_z(xi_s, rel_s)
                z_diam_s = matlab_robust_z(diam_s, rel_s)
                z_count_s = matlab_robust_z(count_s, rel_s)
                z_area_s = matlab_robust_z(area_s, rel_s)
                out[f"ObjectStructureScore_{suffix}"] = sigmoid100(0.40 * z_xi_s + 0.40 * z_diam_s - 0.20 * z_count_s).where(rel_s, np.nan)
                out[f"ObjectAggregationIndex_{suffix}"] = sigmoid100(0.30 * z_diam_s + 0.30 * z_xi_s + 0.20 * z_area_s - 0.20 * z_count_s).where(rel_s, np.nan)
            else:
                out[f"ObjectStructureScore_{suffix}"] = np.nan
                out[f"ObjectAggregationIndex_{suffix}"] = np.nan
    return out


def rbc_overlay_background(rbc_result: dict[str, np.ndarray], background_name: str) -> tuple[np.ndarray, str]:
    """Return the selected RBC overlay background and title prefix."""
    if background_name == "Raw grayscale image":
        return rbc_result["crop_gray"], "Raw"
    if background_name == "Contrast-enhanced residual":
        return rbc_result["display_residual"], "Contrast-enhanced residual"
    return rbc_result["display_Wz"], "Illumination-corrected Wz"


def image_array_to_png_bytes(image: np.ndarray, is_mask: bool = False) -> bytes:
    """Encode a debug image or mask as PNG bytes for ZIP export."""
    arr = np.asarray(image)
    if is_mask or arr.dtype == bool:
        out = arr.astype(bool).astype(np.uint8) * 255
    else:
        work = np.asarray(arr, dtype=np.float32)
        work = np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
        lo = float(np.min(work))
        hi = float(np.max(work))
        if hi > lo:
            work = (work - lo) / (hi - lo)
        else:
            work = np.zeros_like(work)
        out = np.clip(work * 255.0, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(out).save(buffer, format="PNG")
    return buffer.getvalue()


def rbc_mask_difference_image(app_mask: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
    """Return RGB difference map: common white, app-only red, reference-only blue."""
    app_bw = np.asarray(app_mask, dtype=bool)
    ref_bw = np.asarray(reference_mask, dtype=bool)
    rgb = np.zeros((*app_bw.shape, 3), dtype=np.uint8)
    rgb[app_bw & ref_bw] = [255, 255, 255]
    rgb[app_bw & ~ref_bw] = [255, 0, 0]
    rgb[~app_bw & ref_bw] = [0, 120, 255]
    return rgb


def compare_rbc_debug_masks(app_mask: np.ndarray, reference_mask: np.ndarray) -> dict[str, Any]:
    """Compare two RBC gap masks without changing either mask."""
    app_bw = np.asarray(app_mask, dtype=bool)
    ref_bw = np.asarray(reference_mask, dtype=bool)
    if app_bw.shape != ref_bw.shape:
        return {
            "maskShapeMatch": False,
            "pixelAgreement": np.nan,
            "Dice": np.nan,
            "IoU": np.nan,
            "appOnlyPixels": np.nan,
            "matlabOnlyPixels": np.nan,
            "likelyMismatchSource": "Mask shapes differ; check input image, crop size, or resizing before analysis.",
        }
    app_only = app_bw & ~ref_bw
    matlab_only = ref_bw & ~app_bw
    intersection = float(np.logical_and(app_bw, ref_bw).sum())
    app_sum = float(app_bw.sum())
    ref_sum = float(ref_bw.sum())
    union = float(np.logical_or(app_bw, ref_bw).sum())
    dice = float(2.0 * intersection / (app_sum + ref_sum)) if (app_sum + ref_sum) > 0 else 1.0
    iou = float(intersection / union) if union > 0 else 1.0
    likely = "Masks are closely matched."
    if dice < 0.95:
        likely = "Mask mismatch likely comes from preprocessing input source, max-gap filtering, Wz MAD/floor/smoothing, or cleanup settings."
    if app_sum == 0 and ref_sum > 0:
        likely = "App mask is empty while MATLAB-reference mask is not; check kHi/kLo, Wz source, and minAreaPx."
    elif ref_sum == 0 and app_sum > 0:
        likely = "MATLAB-reference mask is empty while app mask is not; check legacy raw-crop source or extra post-processing."
    return {
        "maskShapeMatch": True,
        "pixelAgreement": float(np.mean(app_bw == ref_bw)) if app_bw.size else np.nan,
        "Dice": dice,
        "IoU": iou,
        "appOnlyPixels": int(app_only.sum()),
        "matlabOnlyPixels": int(matlab_only.sum()),
        "appAreaFrac": float(app_sum / app_bw.size) if app_bw.size else np.nan,
        "matlabReferenceAreaFrac": float(ref_sum / ref_bw.size) if ref_bw.size else np.nan,
        "likelyMismatchSource": likely,
    }


def build_rbc_mask_debug_bundle(image: np.ndarray, params: dict[str, Any], filename: str) -> tuple[bytes, dict[str, Any]]:
    """Build a ZIP containing app and MATLAB-reference RBC mask intermediates."""
    app_params = dict(params)
    app_params["include_debug_stages"] = True
    app_result = segment_rbc_gaps_seedgrow(image, app_params)

    matlab_params = dict(params)
    matlab_params.update(
        {
            "include_debug_stages": True,
            "rbcSegmentationInputSource": "MATLAB Iraw01 processed grayscale",
            # MATLAB segmentGapsAllFrames has no maximum single-gap safety layer.
            "enableMaxGapAreaFilter": False,
        }
    )
    matlab_result = segment_rbc_gaps_seedgrow(image, matlab_params)
    comparison = compare_rbc_debug_masks(app_result["gap_mask"], matlab_result["gap_mask"])
    report = {
        "filename": filename,
        "app": app_result.get("debug_report", {}),
        "matlab_reference": matlab_result.get("debug_report", {}),
        "comparison": comparison,
        "notes": [
            "White mask pixels are bright plasma gaps.",
            "Black mask pixels are RBC-rich/non-gap regions.",
            "display_Wz is contrast-stretched for viewing; segmentation thresholds use analysis Wz.",
            "MATLAB Raw panel is processed/analysis grayscale Iraw01, not necessarily true camera raw.",
            "MATLAB-reference debug run uses Iraw01 source and disables the MicroStructAI max-gap artefact filter.",
        ],
    }

    safe_base = safe_filename_token(Path(filename).stem or "rbc_frame")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{safe_base}/debug_report.json", json.dumps(report, indent=2, default=str))
        archive.writestr(
            f"{safe_base}/debug_report.csv",
            pd.DataFrame([{"section": "app", **report["app"]}, {"section": "matlab_reference", **report["matlab_reference"]}]).to_csv(index=False),
        )
        archive.writestr(f"{safe_base}/mask_comparison.csv", pd.DataFrame([comparison]).to_csv(index=False))
        for prefix, result in [("app", app_result), ("matlab_reference", matlab_result)]:
            for stage_name, stage_image in result.get("debug_stages", {}).items():
                archive.writestr(
                    f"{safe_base}/{prefix}/{stage_name}.png",
                    image_array_to_png_bytes(stage_image, is_mask=np.asarray(stage_image).dtype == bool),
                )
            overlay_bg = result.get("display_Wz", result.get("crop_gray"))
            overlay = create_boundary_overlay(overlay_bg, result["gap_mask"], color=(0, 1, 0), linewidth=int(params.get("boundaryLineWidth", 1)))
            archive.writestr(f"{safe_base}/{prefix}/14_final_overlay_gap_boundaries.png", image_array_to_png_bytes(overlay))
        app_bw = np.asarray(app_result["gap_mask"], dtype=bool)
        matlab_bw = np.asarray(matlab_result["gap_mask"], dtype=bool)
        if app_bw.shape == matlab_bw.shape:
            archive.writestr(f"{safe_base}/comparison/app_only_mask.png", image_array_to_png_bytes(app_bw & ~matlab_bw, is_mask=True))
            archive.writestr(f"{safe_base}/comparison/matlab_only_mask.png", image_array_to_png_bytes(~app_bw & matlab_bw, is_mask=True))
            archive.writestr(
                f"{safe_base}/comparison/difference_common_white_app_red_matlab_blue.png",
                image_array_to_png_bytes(rbc_mask_difference_image(app_bw, matlab_bw)),
            )
    return buffer.getvalue(), report


def build_enhanced_object_debug_bundle(image: np.ndarray, params: dict[str, Any], filename: str) -> tuple[bytes, dict[str, Any]]:
    """Build a ZIP containing Enhanced Object Mode generic preprocessing stages."""
    object_params = dict(params)
    object_params["include_debug_stages"] = True
    result = segment_enhanced_structures_seedgrow(image, object_params)
    diagnostics = result.get("structure_diagnostics", {})
    report = {
        "filename": filename,
        "pipeline": "MicroStructAI Enhanced Object Mode generic seed/grow",
        "maskWhitePixelsMeaning": "generic detected structures/gaps/objects",
        "notRbcPlasmaGapMask": True,
        "diagnostics": diagnostics,
        "notes": [
            "Object Mode masks are generic detected structures, not RBC plasma gaps.",
            "display_Wz is contrast-stretched for viewing; segmentation thresholds use analysis Wz.",
            "The MATLAB/RBC-style enhanced preset borrows robust residual preprocessing only; it does not apply RBC score interpretation.",
        ],
    }

    safe_base = safe_filename_token(Path(filename).stem or "enhanced_object_frame")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{safe_base}/debug_report.json", json.dumps(report, indent=2, default=str))
        archive.writestr(f"{safe_base}/debug_report.csv", pd.DataFrame([diagnostics]).to_csv(index=False))
        for stage_name, stage_image in result.get("debug_stages", {}).items():
            archive.writestr(
                f"{safe_base}/stages/{stage_name}.png",
                image_array_to_png_bytes(stage_image, is_mask=np.asarray(stage_image).dtype == bool),
            )
        overlay_bg = result.get("display_Wz", result.get("raw_gray"))
        overlay = create_boundary_overlay(overlay_bg, result["structure_mask"], color=(0, 1, 0), linewidth=1)
        archive.writestr(f"{safe_base}/final_generic_structure_overlay.png", image_array_to_png_bytes(overlay))
        archive.writestr(f"{safe_base}/final_generic_structure_mask.png", image_array_to_png_bytes(result["structure_mask"], is_mask=True))
        removed = result.get("removed_structure_mask")
        if removed is not None and np.any(removed):
            archive.writestr(f"{safe_base}/removed_large_generic_objects.png", image_array_to_png_bytes(removed, is_mask=True))
    return buffer.getvalue(), report


def safe_column_mean(df: pd.DataFrame, column: str) -> float:
    """Return a finite mean for UI summaries, or NaN if unavailable."""
    if column not in df.columns:
        return np.nan
    values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    if finite.empty:
        return np.nan
    return float(finite.mean())


def safe_column_median(df: pd.DataFrame, column: str) -> float:
    """Return a finite median for QC summaries, or NaN if unavailable."""
    if column not in df.columns:
        return np.nan
    values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    if finite.empty:
        return np.nan
    return float(finite.median())


def format_metric_value(value: float, decimals: int = 2) -> str:
    """Format a UI metric value with a clear NaN fallback."""
    return f"{value:.{decimals}f}" if np.isfinite(value) else "n/a"


def add_standard_quality_warnings(df: pd.DataFrame, mode: str, warnings: list[str]) -> None:
    """Append non-fatal quality-control warnings for automatic segmentations."""
    if df.empty:
        return

    area_col = "gapAreaFrac" if mode == "RBC Gap Mode" else "areaFrac"
    if area_col in df.columns:
        area = pd.to_numeric(df[area_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not area.empty and ((area < 0.02).any() or (area > 0.85).any()):
            warnings.append("Mask area fraction is extreme; check segmentation parameters.")

    if mode == "RBC Gap Mode":
        if len(df) < 5:
            warnings.append("Fewer than 5 images analysed; scores can be calculated but PCA/time trends are limited.")
        nan_check_excluded = {
            "largeGapArtefactFlag",
            "hugeGapFlag",
            "badMaskFlag",
            "unstableSegmentationFlag",
            "adaptiveRepairTriggered",
            "adaptiveRepairKept",
            "adaptiveRepairReason",
            "largestGapFrac_afterRepair",
            "largestToMedianGapRatio_afterRepair",
            "maskReliabilityScore_afterRepair",
        }
        qc_cols = [col for col in RBC_GAP_METRICS + RBC_SCORE_METRICS if col in df.columns and col not in nan_check_excluded]
        if qc_cols:
            numeric = df[qc_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            nan_fraction = float(numeric.isna().sum().sum() / numeric.size) if numeric.size else 0.0
            if nan_fraction > 0.25:
                warnings.append("Many RBC-specific metric values are NaN; inspect segmentation and image content before interpreting scores.")


def add_object_mode_quality_warnings(df: pd.DataFrame, params: dict[str, Any], warnings: list[str]) -> None:
    """Append generic Object Mode warnings without assigning RBC meaning."""
    if df.empty:
        return
    low_resolution = df.get("lowResolutionRisk", pd.Series(dtype=object)).astype(str)
    if low_resolution.isin(["high", "moderate"]).any():
        warnings.append(
            "Low-resolution warning: results describe coarse structure only. "
            "Fine object count, boundary shape, and small-object size distribution may be unreliable."
        )
    confidence = df.get("segmentationConfidence", pd.Series(dtype=object)).astype(str)
    if (confidence == "low").any():
        warnings.append("Some Object Mode masks have low segmentation confidence; inspect generic structure overlays before interpretation.")
    pod_object_processing_active = bool(df.get("podProcessingEnabled", pd.Series(False, index=df.index)).fillna(False).astype(bool).any())
    if pod_object_processing_active and "podObjectSensitiveFlag" in df.columns and df["podObjectSensitiveFlag"].fillna(False).astype(bool).any():
        warnings.append("Some Conservative/POD Object masks are POD-sensitive; inspect no-POD/POD object overlays before interpretation.")
    if "spatialCorrectionSensitiveFlag" in df.columns and df["spatialCorrectionSensitiveFlag"].fillna(False).astype(bool).any():
        warnings.append("Some Conservative/POD Object masks are sensitive to spatial illumination correction.")
    if "rawGainCorrectionSensitiveFlag" in df.columns and df["rawGainCorrectionSensitiveFlag"].fillna(False).astype(bool).any():
        warnings.append("Some Conservative/POD Object masks are sensitive to global raw-gain correction.")
    if params.get("conservativeObjectModeEnabled", False):
        warnings.append(
            "Conservative/POD Object Mode scores are generic mask-based structural/coarsening indices. "
            "They are not RBC SASI/AggIndex or validated biomarkers."
        )
    detected = pd.to_numeric(df.get("nLargeObjectsDetected", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    removed = pd.to_numeric(df.get("nLargeObjectsRemoved", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    if (removed > 0).any():
        warnings.append("Large generic structure objects were removed by the optional Object Mode artefact filter; inspect overlays.")
    elif (detected > 0).any() and not bool(params.get("enableMaxObjectAreaFilter", False)):
        warnings.append("Suspiciously large generic structure objects were detected while the optional max-object filter was disabled.")
    if params.get("experimentalCompositeScoresEnabled", False):
        warnings.append(
            "Optional gSASI/gAggIndex scores are exploratory image-structure indices only. "
            "They are not validated for this sample type and are not RBC aggregation scores."
        )
        for column in ["gSASI_warning", "gAggIndex_warning"]:
            values = df.get(column, pd.Series(dtype=object)).dropna().astype(str)
            for warning in sorted(set(values) - {""}):
                warnings.append(warning)


def apply_normalized_syllectogram_outputs(metrics_df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Add normalized syllectogram curves/metrics without changing source scores."""
    if metrics_df.empty:
        params["normalized_syllectogram_summary"] = {}
        return metrics_df
    smooth_window = int(params.get("normalizedSyllectogramSmoothWindow", params.get("sasiSyllectogramSmoothWindow", 5)))
    out, summary = add_normalized_syllectogram_outputs(
        metrics_df,
        smooth_window=smooth_window,
        fit_double_exponential=bool(params.get("fitNormalizedSyllectogramDoubleExp", True)),
    )
    params["normalized_syllectogram_summary"] = summary
    return out


def add_large_gap_artefact_summary(metrics_df: pd.DataFrame, params: dict[str, Any], warnings: list[str]) -> pd.DataFrame:
    """Add transparent sample-level summaries for the RBC max-gap safety layer."""
    if metrics_df.empty:
        params["large_gap_artefact_summary"] = {}
        return metrics_df

    out = metrics_df.copy()
    detected = pd.to_numeric(out.get("nLargeGapObjectsDetected", pd.Series(0, index=out.index)), errors="coerce").fillna(0).astype(int)
    removed = pd.to_numeric(out.get("nLargeGapObjectsRemoved", pd.Series(0, index=out.index)), errors="coerce").fillna(0).astype(int)
    largest = pd.to_numeric(out.get("largestRemovedGapFrac", pd.Series(np.nan, index=out.index)), errors="coerce").replace([np.inf, -np.inf], np.nan)
    n_frames = len(out)
    frames_detected = int((detected > 0).sum())
    frames_removed = int((removed > 0).sum())
    percent_removed = 100.0 * frames_removed / n_frames if n_frames else 0.0
    percent_detected = 100.0 * frames_detected / n_frames if n_frames else 0.0
    risk_basis = percent_detected if not bool(params.get("enableMaxGapAreaFilter", True)) else percent_removed
    risk = "low" if risk_basis < 5.0 else ("moderate" if risk_basis <= 20.0 else "high")
    summary = {
        "maxGapAreaFilterEnabled": bool(params.get("enableMaxGapAreaFilter", True)),
        "maxGapAreaFrac": float(params.get("maxGapAreaFrac", 0.10)),
        "totalLargeGapObjectsRemoved": int(removed.sum()),
        "totalLargeGapObjectsDetected": int(detected.sum()),
        "framesWithLargeGapRemoval": frames_removed,
        "framesWithLargeGapDetection": frames_detected,
        "maxLargestRemovedGapFrac": float(largest.max()) if largest.notna().any() else 0.0,
        "percentFramesWithLargeGapRemoval": percent_removed,
        "percentFramesWithLargeGapDetection": percent_detected,
        "largeGapArtefactRisk": risk,
        "matlabCompatibleModeDisabledFilter": (
            str(params.get("rbcGapSafetyMode", "")) == "MATLAB-compatible mode"
            and not bool(params.get("enableMaxGapAreaFilter", True))
        ),
    }
    for key, value in summary.items():
        out[key] = value
    params["large_gap_artefact_summary"] = summary

    if frames_removed:
        warnings.append(
            f"Large-gap artefact filter removed {summary['totalLargeGapObjectsRemoved']} suspicious object(s) "
            f"from {frames_removed} frame(s). Check overlays."
        )
    elif frames_detected and not summary["maxGapAreaFilterEnabled"]:
        warnings.append(
            f"Maximum single-gap artefact filter is disabled; {frames_detected} frame(s) contain suspiciously large gap objects. "
            "Legacy MATLAB mask behaviour is preserved, but overlays should be checked."
        )
    return out


def prefix_metrics(row: dict[str, Any], prefix: str, metric_names: list[str]) -> dict[str, Any]:
    """Prefix selected metric keys from a core-metrics row."""
    return {f"{prefix}_{metric}": row.get(metric, np.nan) for metric in metric_names}


def finite_percentile(values: np.ndarray, q: float) -> float:
    """Return a finite percentile or NaN."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.percentile(arr, q))


def robust_sigma(values: np.ndarray) -> float:
    """Return robust sigma estimate from MAD with std fallback."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 1e-12:
        sigma = float(np.std(arr))
    return sigma if np.isfinite(sigma) and sigma > 1e-12 else np.nan


def truthy_value(value: Any) -> bool:
    """Interpret common scalar truthy values without pandas ambiguity."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "on", "high", "moderate", "disabled_warning"}
    try:
        return bool(value) if pd.notna(value) else False
    except Exception:
        return False


def numeric_row_value(row: dict[str, Any] | pd.Series, column: str, default: float = np.nan) -> float:
    """Return one row value as a finite float or ``default``."""
    try:
        value = row.get(column, default)  # type: ignore[attr-defined]
    except Exception:
        return default
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if np.isfinite(parsed) else default


def compute_illumination_cv(illumination_field: np.ndarray) -> float:
    """Return std/mean of a smooth illumination field as uneven-lighting evidence."""
    arr = np.asarray(illumination_field, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    mean_value = float(np.mean(arr))
    if abs(mean_value) <= np.finfo(float).eps:
        return np.nan
    return float(np.std(arr) / (abs(mean_value) + np.finfo(float).eps))


def bounded_ratio_score(value: float, target: float) -> float:
    """Convert a value/target relation to a 0..1 reliability component."""
    if not np.isfinite(value) or not np.isfinite(target) or target <= 0:
        return 0.0
    return float(np.clip(value / target, 0.0, 1.0))


def _robust_rescale01(values: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Return a robust 0..1 copy for texture diagnostics."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=float)
    lo = float(np.percentile(finite, p_low))
    hi = float(np.percentile(finite, p_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def compute_startup_transient_metrics(raw_img: np.ndarray, prev_raw_img: np.ndarray | None = None, params: dict[str, Any] | None = None) -> dict[str, float]:
    """Compute raw-image startup/transient texture diagnostics.

    These metrics are optional QC/research diagnostics. They deliberately do
    not create or replace normal gap/object masks, and they are not used for
    SASI, AggIndex, object counts, or normal morphology scores.
    """
    params = params or {}
    eps = np.finfo(float).eps
    raw = to_grayscale_float01(raw_img)
    if raw.size == 0:
        return {
            "startupRawMean": np.nan,
            "startupRawStd": np.nan,
            "startupRawCV": np.nan,
            "startupGradientEnergy": np.nan,
            "startupTextureVariance": np.nan,
            "startupFFT_LF_HF_ratio": np.nan,
            "startupMotionD": np.nan,
            "startupStreakIndex": np.nan,
        }

    sigma_bg = float(params.get("startupFallbackBgSigma", 40.0))
    bg = ndi.gaussian_filter(raw, sigma=max(0.0, sigma_bg)) if sigma_bg > 0 else np.zeros_like(raw)
    corrected = _robust_rescale01(raw - bg)
    mean_value = float(np.mean(corrected))
    std_value = float(np.std(corrected))
    gy, gx = np.gradient(corrected)
    gx_energy = float(np.mean(gx**2))
    gy_energy = float(np.mean(gy**2))
    gradient_energy = gx_energy + gy_energy
    streak_index = float(abs(gx_energy - gy_energy) / (gx_energy + gy_energy + eps))

    centered = corrected - float(np.mean(corrected))
    fft_power = np.abs(np.fft.fft2(centered)) ** 2
    fy = np.fft.fftfreq(corrected.shape[0])
    fx = np.fft.fftfreq(corrected.shape[1])
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    rr = np.sqrt(xx**2 + yy**2)
    lf_band = rr <= 0.03
    hf_band = (rr > 0.03) & (rr <= 0.18)
    lf_power = float(np.sum(fft_power[lf_band])) if np.any(lf_band) else np.nan
    hf_power = float(np.sum(fft_power[hf_band])) if np.any(hf_band) else np.nan
    fft_ratio = float(np.log10((lf_power + eps) / (hf_power + eps))) if np.isfinite(lf_power) and np.isfinite(hf_power) else np.nan

    motion_d = np.nan
    if prev_raw_img is not None:
        prev = to_grayscale_float01(prev_raw_img)
        if prev.shape == raw.shape:
            prev_bg = ndi.gaussian_filter(prev, sigma=max(0.0, sigma_bg)) if sigma_bg > 0 else np.zeros_like(prev)
            prev_corrected = _robust_rescale01(prev - prev_bg)
            motion_d = float(np.mean(np.abs(corrected - prev_corrected)))

    return {
        "startupRawMean": mean_value,
        "startupRawStd": std_value,
        "startupRawCV": float(std_value / (abs(mean_value) + eps)),
        "startupGradientEnergy": gradient_energy,
        "startupTextureVariance": float(np.var(corrected)),
        "startupFFT_LF_HF_ratio": fft_ratio,
        "startupMotionD": motion_d,
        "startupStreakIndex": streak_index,
    }


def add_startup_transient_fallback_flags(
    metrics_df: pd.DataFrame,
    params: dict[str, Any],
    mode: str,
    warnings: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mark early unreliable frames and summarize optional fallback diagnostics."""
    if metrics_df.empty or not params.get("enableStartupTransientFallback", False):
        return metrics_df, {}

    out = metrics_df.copy()
    n_frames = len(out)
    early_limit = float(params.get("startupFallbackEarlyFrameLimit", 20))
    time_limit = float(params.get("startupFallbackTimeLimit_s", params.get("manualStart_s", params.get("matlab_analysisStart_s", 20.0))))
    reliability_threshold = float(params.get("startupFallbackReliabilityThreshold", params.get("maskReliabilityMinForSummary", 60.0)))

    if "originalImageIndex_1based" in out.columns:
        index_series = pd.to_numeric(out["originalImageIndex_1based"], errors="coerce")
    else:
        index_series = pd.to_numeric(out.get("analysisIndex_1based", pd.Series(np.arange(1, n_frames + 1), index=out.index)), errors="coerce")
    early_by_index = index_series <= early_limit

    if "matlabTime_s" in out.columns:
        time_series = pd.to_numeric(out["matlabTime_s"], errors="coerce")
    elif "time_s" in out.columns:
        time_series = pd.to_numeric(out["time_s"], errors="coerce")
    else:
        time_series = pd.Series(np.nan, index=out.index, dtype="float64")
    early_by_time = time_series.notna() & (time_series < time_limit)

    early_existing = pd.Series(False, index=out.index)
    for column in ["earlyLowConfidenceFrame", "qcOutsideStableWindow"]:
        if column in out.columns:
            early_existing = early_existing | out[column].fillna(False).astype(bool)
    if "isAnaFinal" in out.columns:
        early_existing = early_existing | (~out["isAnaFinal"].fillna(False).astype(bool))

    early_frame = early_by_index.fillna(False) | early_by_time.fillna(False) | early_existing

    unreliable = pd.Series(False, index=out.index)
    reasons = pd.Series("", index=out.index, dtype=object)
    if "maskReliabilityScore" in out.columns:
        low_score = pd.to_numeric(out["maskReliabilityScore"], errors="coerce") < reliability_threshold
        unreliable = unreliable | low_score.fillna(False)
        reasons = reasons.mask(low_score.fillna(False), reasons + "low maskReliabilityScore; ")
    if "badMaskFlag" in out.columns:
        bad_mask = out["badMaskFlag"].fillna(False).astype(bool)
        unreliable = unreliable | bad_mask
        reasons = reasons.mask(bad_mask, reasons + "badMaskFlag; ")
    if "rbcLowConfidence" in out.columns:
        low_conf = out["rbcLowConfidence"].fillna(False).astype(bool)
        unreliable = unreliable | low_conf
        reasons = reasons.mask(low_conf, reasons + "RBC low-confidence QC; ")
    if "rbcFrameReliable" in out.columns:
        not_reliable = ~out["rbcFrameReliable"].fillna(False).astype(bool)
        unreliable = unreliable | not_reliable
        reasons = reasons.mask(not_reliable, reasons + "frame QC not reliable; ")
    if "segmentationConfidence" in out.columns:
        seg_low = out["segmentationConfidence"].fillna("").astype(str).str.lower().eq("low")
        unreliable = unreliable | seg_low
        reasons = reasons.mask(seg_low, reasons + "low segmentation confidence; ")

    area = pd.to_numeric(out.get("areaFrac", pd.Series(np.nan, index=out.index)), errors="coerce")
    count = pd.to_numeric(out.get("objectCount", out.get("nObjects", pd.Series(np.nan, index=out.index))), errors="coerce")
    implausible_generic = area.notna() & ((area < 0.001) | (area > 0.90))
    empty_generic = count.notna() & (count <= 0)
    unreliable = unreliable | implausible_generic | empty_generic
    reasons = reasons.mask(implausible_generic, reasons + "implausible mask area; ")
    reasons = reasons.mask(empty_generic, reasons + "empty/invalid object count; ")

    fallback_used = early_frame & unreliable
    normal_reliable = ~unreliable
    frame_class = np.where(
        fallback_used,
        "early_unreliable_startup",
        np.where(early_frame & normal_reliable, "early_reliable", np.where(unreliable, "qc_unreliable", "stable")),
    )

    out["enableStartupTransientFallback"] = True
    out["startupFallbackUsed"] = fallback_used.astype(bool)
    out["normalMaskReliable"] = normal_reliable.astype(bool)
    out["frameClass"] = frame_class
    out["startupFallbackReason"] = reasons.str.strip().str.rstrip(";")
    out["startupEarlyFrameFlag"] = early_frame.astype(bool)

    fallback_subset = out.loc[fallback_used]
    summary_source = fallback_subset if not fallback_subset.empty else out.iloc[0:0]
    fallback_count = int(fallback_used.sum())
    frac_fallback = float(fallback_count / n_frames) if n_frames else 0.0
    reliable_positions = np.flatnonzero(normal_reliable.to_numpy(dtype=bool))
    stable_positions = np.flatnonzero(pd.Series(frame_class, index=out.index).eq("stable").to_numpy(dtype=bool))
    first_reliable_frame = frame_identity_at_position(out, int(reliable_positions[0]), "originalImageIndex_1based") if reliable_positions.size else np.nan
    first_stable_time = (
        float(time_series.iloc[int(stable_positions[0])])
        if stable_positions.size and np.isfinite(pd.to_numeric(pd.Series([time_series.iloc[int(stable_positions[0])]]), errors="coerce").iloc[0])
        else np.nan
    )
    summary = {
        "enableStartupTransientFallback": True,
        "nStartupFallbackFrames": fallback_count,
        "fracStartupFallbackFrames": frac_fallback,
        "medianStartupRawCV": metric_median(summary_source, "startupRawCV"),
        "medianStartupGradientEnergy": metric_median(summary_source, "startupGradientEnergy"),
        "medianStartupFFT_LF_HF_ratio": metric_median(summary_source, "startupFFT_LF_HF_ratio"),
        "medianStartupStreakIndex": metric_median(summary_source, "startupStreakIndex"),
        "firstReliableFrame": first_reliable_frame,
        "firstStableFrameTime_s": first_stable_time,
    }
    for key, value in summary.items():
        out[key] = value
    params["startup_transient_fallback_summary"] = summary
    if fallback_count and warnings is not None:
        warnings.append(
            f"Startup/transient fallback used for {fallback_count} early QC-flagged frame(s). "
            "Normal segmentation metrics remain exported but should not be interpreted as stable aggregation/object morphology for those frames."
        )
    return out, summary


def add_rbc_mask_quality_diagnostics(row: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Add per-frame mask-quality diagnostics without changing the mask.

    These diagnostics are intentionally advisory by default. They can be used
    later by the optional post-start exclusion and adaptive repair switches,
    but simply computing them must not alter historical RBC results.
    """
    out = dict(row)
    gap_area = numeric_row_value(out, "gapAreaFrac")
    gap_count = numeric_row_value(out, "gapNtotal")
    gap_mean_diam = numeric_row_value(out, "gapMeanDiam_px")
    largest_before = numeric_row_value(out, "largestGapFrac_beforeFilter", numeric_row_value(out, "gapLargestFrac", 0.0))
    largest_after = numeric_row_value(out, "largestGapFrac_afterFilter", numeric_row_value(out, "gapLargestFrac", 0.0))
    largest_ratio = numeric_row_value(out, "largestToMedianGapRatio")
    illumination_cv = numeric_row_value(out, "illuminationCV")

    max_gap_frac_threshold = float(params.get("hugeGapFracThreshold", params.get("maxGapAreaFrac", 0.10)))
    ratio_threshold = float(params.get("largestToMedianGapRatioThreshold", 50.0))
    max_illum_cv = float(params.get("maxIlluminationCVForReliableMask", 0.25))
    min_area = float(params.get("minGapAreaFracMaskQC", 0.002))
    max_area = float(params.get("maxGapAreaFracMaskQC", 0.85))
    min_count = float(params.get("minGapNtotalMaskQC", 1.0))
    max_count = float(params.get("maxGapNtotalMaskQC", 100000.0))

    large_flag_text = str(out.get("largeGapArtefactFlag", "none"))
    huge_gap = (
        (np.isfinite(largest_before) and largest_before > max_gap_frac_threshold)
        or (np.isfinite(largest_after) and largest_after > max_gap_frac_threshold)
        or (np.isfinite(largest_ratio) and largest_ratio > ratio_threshold)
        or large_flag_text in {"high", "moderate", "disabled_warning"}
    )
    implausible_area = not (np.isfinite(gap_area) and min_area <= gap_area <= max_area)
    count_bad = not (np.isfinite(gap_count) and min_count <= gap_count <= max_count)
    diam_bad = not (np.isfinite(gap_mean_diam) and gap_mean_diam > 0)
    illumination_bad = np.isfinite(illumination_cv) and illumination_cv > max_illum_cv
    qc_implausible = truthy_value(out.get("qcFailImplausibleMask", False))
    qc_support_bad = truthy_value(out.get("qcFailLowContrastSeparation", False))
    qc_shi_bad = truthy_value(out.get("qcFailSHI_tileCV", False))
    hard_bad_mask = bool(huge_gap or implausible_area or count_bad or diam_bad or qc_implausible)
    bad_mask = bool(hard_bad_mask or illumination_bad or qc_support_bad or qc_shi_bad)

    support_components = [
        bounded_ratio_score(numeric_row_value(out, "structuralContrast"), float(params.get("minStructuralContrast", 0.10))),
        bounded_ratio_score(numeric_row_value(out, "maskIntensitySeparationZ"), float(params.get("minSeparationZ", 1.0))),
        bounded_ratio_score(numeric_row_value(out, "boundarySupport"), float(params.get("minBoundarySupport", 0.30))),
        bounded_ratio_score(numeric_row_value(out, "SHI_tileCV"), float(params.get("minSHI_tileCV_forReliable", 0.15))),
    ]
    support_score = float(np.nanmean(support_components)) if support_components else 0.0
    area_score = 1.0 if not implausible_area else 0.0
    count_score = 1.0 if not count_bad and not diam_bad else 0.0
    huge_score = 0.0 if huge_gap else 1.0
    illum_score = 1.0
    if not np.isfinite(illumination_cv):
        illum_score = 0.75
    elif max_illum_cv > 0:
        illum_score = float(np.clip(1.0 - max(0.0, illumination_cv - max_illum_cv) / max_illum_cv, 0.0, 1.0))

    reliability = 100.0 * (
        0.35 * support_score
        + 0.20 * area_score
        + 0.15 * count_score
        + 0.20 * huge_score
        + 0.10 * illum_score
    )
    if qc_implausible:
        reliability -= 15.0
    if qc_support_bad:
        reliability -= 10.0
    if qc_shi_bad:
        reliability -= 5.0
    if bad_mask:
        severe_bad_mask = huge_gap or implausible_area or count_bad or diam_bad or illumination_bad or qc_implausible or qc_support_bad
        reliability = min(reliability, 49.0 if severe_bad_mask else 79.0)
    reliability = float(np.clip(reliability, 0.0, 100.0))

    reasons: list[str] = []
    if huge_gap:
        reasons.append("huge gap")
    if implausible_area:
        reasons.append("implausible area fraction")
    if count_bad:
        reasons.append("gap count outside QC range")
    if diam_bad:
        reasons.append("invalid gap diameter")
    if illumination_bad:
        reasons.append("uneven illumination")
    if qc_implausible:
        reasons.append("existing implausible-mask QC")
    if qc_support_bad:
        reasons.append("low contrast/separation QC")
    if qc_shi_bad:
        reasons.append("low SHI_tileCV QC")

    out.update(
        {
            "hugeGapFlag": bool(huge_gap),
            "illuminationCV": illumination_cv,
            "badMaskFlag": bool(bad_mask),
            "hardBadMaskFlag": bool(hard_bad_mask),
            "unstableSegmentationFlag": False,
            "maskReliabilityScore": reliability,
            "maskQCFailureReasons": "; ".join(reasons),
            "hugeGapFracThreshold": max_gap_frac_threshold,
            "largestToMedianGapRatioThreshold": ratio_threshold,
            "maxIlluminationCVForReliableMask": max_illum_cv,
        }
    )
    return out


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    """Return a finite Spearman rho or NaN."""
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["x"].nunique() <= 1 or frame["y"].nunique() <= 1:
        return np.nan
    value = frame["x"].corr(frame["y"], method="spearman")
    return float(value) if np.isfinite(value) else np.nan


def compute_rbc_frame_qc_metrics(continuous_field: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Compute frame-level support metrics for RBC plasma-gap QC."""
    field = np.asarray(continuous_field, dtype=float)
    mask_bool = np.asarray(mask, dtype=bool)
    if field.shape != mask_bool.shape:
        return {"structuralContrast": np.nan, "maskIntensitySeparationZ": np.nan, "boundarySupport": np.nan}

    structural_contrast = finite_percentile(field, 95) - finite_percentile(field, 5)
    mask_values = field[mask_bool]
    background_values = field[~mask_bool]
    sigma = robust_sigma(field.ravel())
    if mask_values.size and background_values.size and np.isfinite(sigma):
        separation_z = (float(np.median(mask_values)) - float(np.median(background_values))) / sigma
    else:
        separation_z = np.nan

    gy, gx = np.gradient(field)
    grad = np.sqrt(gx * gx + gy * gy)
    boundary = mask_bool & ~ndi.binary_erosion(mask_bool, structure=np.ones((3, 3), dtype=bool), border_value=0)
    grad_sigma = robust_sigma(grad.ravel())
    if boundary.any() and np.isfinite(grad_sigma):
        boundary_support = (float(np.mean(grad[boundary])) - float(np.median(grad))) / grad_sigma
    else:
        boundary_support = np.nan

    return {
        "structuralContrast": float(structural_contrast) if np.isfinite(structural_contrast) else np.nan,
        "maskIntensitySeparationZ": float(separation_z) if np.isfinite(separation_z) else np.nan,
        "boundarySupport": float(boundary_support) if np.isfinite(boundary_support) else np.nan,
    }


def compute_auto_start_raw_evidence(gray_image: np.ndarray, sigma_bg: float = 40.0, min_bright_area: int = 20) -> dict[str, float]:
    """Compute raw-image evidence used by the first-rise automatic start detector."""
    image = np.asarray(gray_image, dtype=float)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)
    if image.size == 0:
        return {"rawBrightFrac_autoStart": np.nan, "rawContrast01_autoStart": np.nan}

    bg = ndi.gaussian_filter(image, sigma=float(sigma_bg))
    corr = image - bg
    lo = float(np.nanmin(corr))
    hi = float(np.nanmax(corr))
    if hi > lo:
        corr01 = np.clip((corr - lo) / (hi - lo), 0.0, 1.0)
    else:
        corr01 = np.zeros_like(corr, dtype=float)

    p1, p99 = np.percentile(corr01, [1, 99])
    raw_contrast = float(p99 - p1)
    med = float(np.median(corr01))
    mad = float(np.median(np.abs(corr01 - med)))
    bright_thr = med + 3.0 * (mad + 1e-12)
    bright = corr01 > bright_thr
    if int(min_bright_area) > 1 and bright.any():
        labels, n_labels = ndi.label(bright)
        if n_labels:
            counts = np.bincount(labels.ravel())
            keep = counts >= int(min_bright_area)
            keep[0] = False
            bright = keep[labels]
    raw_bright_frac = float(np.count_nonzero(bright) / bright.size) if bright.size else np.nan
    return {
        "rawBrightFrac_autoStart": raw_bright_frac if np.isfinite(raw_bright_frac) else np.nan,
        "rawContrast01_autoStart": raw_contrast if np.isfinite(raw_contrast) else np.nan,
    }


def classify_rbc_frame_qc(row: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Classify one RBC frame as reliable, weak/pre-gap, artefact, or no reliable structure."""
    if not params.get("enableRbcQc", True):
        return {
            "rbcFrameQC": "QC disabled",
            "rbcFrameReliable": True,
            "rbcLowConfidence": False,
        }

    min_sep = float(params.get("minSeparationZ", 1.0))
    min_boundary = float(params.get("minBoundarySupport", 0.30))
    min_contrast = float(params.get("minStructuralContrast", 0.10))
    min_shi = float(params.get("minSHI_tileCV_forReliable", 0.15))
    area_frac = float(row.get("gapAreaFrac", np.nan))
    n_total = float(row.get("gapNtotal", np.nan))
    mean_diam = float(row.get("gapMeanDiam_px", np.nan))
    shi_tile_cv = float(row.get("SHI_tileCV", np.nan))
    sep = float(row.get("maskIntensitySeparationZ", np.nan))
    boundary = float(row.get("boundarySupport", np.nan))
    contrast = float(row.get("structuralContrast", np.nan))

    plausible_mask = (
        np.isfinite(area_frac)
        and 0.002 <= area_frac <= 0.85
        and np.isfinite(n_total)
        and n_total > 0
        and np.isfinite(mean_diam)
        and mean_diam > 0
    )
    support_ok = (
        np.isfinite(sep)
        and sep >= min_sep
        and np.isfinite(boundary)
        and boundary >= min_boundary
        and np.isfinite(contrast)
        and contrast >= min_contrast
    )
    shi_ok = np.isfinite(shi_tile_cv) and shi_tile_cv >= min_shi
    reliable = plausible_mask and support_ok and shi_ok
    weak_signal = (
        plausible_mask
        and (
            (np.isfinite(sep) and sep >= 0.5 * min_sep)
            or (np.isfinite(boundary) and boundary >= 0.5 * min_boundary)
            or (np.isfinite(contrast) and contrast >= 0.5 * min_contrast)
            or (np.isfinite(shi_tile_cv) and shi_tile_cv >= 0.5 * min_shi)
        )
    )
    possible_artifact = plausible_mask and (
        (not np.isfinite(sep) or sep < 0.5 * min_sep)
        and (not np.isfinite(boundary) or boundary < 0.5 * min_boundary)
        and np.isfinite(contrast)
        and contrast >= 0.5 * min_contrast
    )

    if reliable:
        label = "Reliable aggregation morphology"
    elif possible_artifact:
        label = "Possible residual/illumination artefact"
    elif weak_signal:
        label = "Weak/pre-gap structural texture"
    else:
        label = "No reliable plasma-gap aggregation detected"
    return {
        "rbcFrameQC": label,
        "rbcFrameReliable": bool(reliable),
        "rbcLowConfidence": not bool(reliable),
        "qcFailImplausibleMask": bool(not plausible_mask),
        "qcFailLowContrastSeparation": bool(plausible_mask and not support_ok),
        "qcFailSHI_tileCV": bool(plausible_mask and support_ok and not shi_ok),
    }


def rbc_repair_columns(
    row: dict[str, Any],
    enabled: bool,
    triggered: bool = False,
    kept: bool = False,
    reason: str = "not enabled",
    before_row: dict[str, Any] | None = None,
    after_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach adaptive anti-merging repair diagnostics to a per-frame row."""
    out = dict(row)
    before = before_row or row
    after = after_row or {}
    out.update(
        {
            "adaptiveRepairEnabled": bool(enabled),
            "adaptiveRepairTriggered": bool(triggered),
            "adaptiveRepairKept": bool(kept),
            "adaptiveRepairReason": reason,
            "largestGapFrac_beforeRepair": numeric_row_value(before, "largestGapFrac_afterFilter", numeric_row_value(before, "gapLargestFrac")),
            "largestGapFrac_afterRepair": numeric_row_value(after, "largestGapFrac_afterFilter", np.nan),
            "largestToMedianGapRatio_beforeRepair": numeric_row_value(before, "largestToMedianGapRatio"),
            "largestToMedianGapRatio_afterRepair": numeric_row_value(after, "largestToMedianGapRatio", np.nan),
            "maskReliabilityScore_beforeRepair": numeric_row_value(before, "maskReliabilityScore"),
            "maskReliabilityScore_afterRepair": numeric_row_value(after, "maskReliabilityScore", np.nan),
        }
    )
    return out


def analyze_rbc_frame_once(image: np.ndarray, filename: str, params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one RBC segmentation pass and compute per-frame metrics/QC."""
    rbc_result = segment_rbc_gaps_seedgrow(image, params)
    mask = rbc_result["gap_mask"]
    ximg01_field = compute_rbc_ximg01_matlab_like(image, params)
    row = compute_rbc_gap_metrics(
        rbc_result["crop_gray"],
        mask,
        filename=filename,
        mode="RBC Gap Mode",
        continuous_field=ximg01_field,
        lf_hf_method=params.get("lf_hf_method", "MATLAB-compatible continuous-field"),
        minAreaFracForXi=float(params.get("minReliableGapAreaFracForXi", 0.03)),
        minObjectsForXi=int(params.get("minReliableObjectsForXi", 10)),
        maxAreaFracForXi=float(params.get("maxReliableAreaFracForXi", 0.97)),
        rejectUniformXi=bool(params.get("rejectUniformXi", True)),
    )
    row.update(rbc_result.get("large_gap_diagnostics", {}))
    row["rbcSegmentationInputSource"] = rbc_result.get("analysis_input_source", params.get("rbcSegmentationInputSource", "MATLAB Iraw01 processed grayscale"))
    row["segmentation_madW"] = rbc_result.get("debug_report", {}).get("madW", np.nan)
    row["segmentation_kHiUse"] = rbc_result.get("debug_report", {}).get("kHiUse", np.nan)
    row["segmentation_kLoUse"] = rbc_result.get("debug_report", {}).get("kLoUse", np.nan)
    row["illuminationCV"] = compute_illumination_cv(rbc_result.get("illumination_field", np.asarray([])))
    row.update(compute_rbc_frame_qc_metrics(ximg01_field, mask))
    row.update(compute_auto_start_raw_evidence(rbc_result["crop_gray"], sigma_bg=40.0, min_bright_area=20))
    row.update(classify_rbc_frame_qc(row, params))
    row = add_rbc_mask_quality_diagnostics(row, params)
    return rbc_result, row


def adaptive_repair_is_suspicious(row: dict[str, Any]) -> bool:
    """Return True when a frame is suspicious enough to try conservative repair."""
    return bool(
        truthy_value(row.get("hugeGapFlag", False))
        or str(row.get("largeGapArtefactFlag", "none")) in {"high", "moderate", "disabled_warning"}
    )


def repaired_mask_is_acceptable(original_row: dict[str, Any], repaired_row: dict[str, Any], params: dict[str, Any]) -> tuple[bool, str]:
    """Decide whether a conservative repaired RBC mask should replace the original."""
    original_score = numeric_row_value(original_row, "maskReliabilityScore")
    repaired_score = numeric_row_value(repaired_row, "maskReliabilityScore")
    original_largest = numeric_row_value(original_row, "largestGapFrac_afterFilter", numeric_row_value(original_row, "gapLargestFrac"))
    repaired_largest = numeric_row_value(repaired_row, "largestGapFrac_afterFilter", numeric_row_value(repaired_row, "gapLargestFrac"))
    original_ratio = numeric_row_value(original_row, "largestToMedianGapRatio")
    repaired_ratio = numeric_row_value(repaired_row, "largestToMedianGapRatio")
    repaired_area = numeric_row_value(repaired_row, "gapAreaFrac")
    repaired_count = numeric_row_value(repaired_row, "gapNtotal")
    min_area = float(params.get("minGapAreaFracMaskQC", 0.002))
    max_area = float(params.get("maxGapAreaFracMaskQC", 0.85))
    min_improvement = float(params.get("adaptiveRepairMinScoreImprovement", 5.0))
    plausible = np.isfinite(repaired_area) and min_area <= repaired_area <= max_area and np.isfinite(repaired_count) and repaired_count > 0
    if not plausible:
        return False, "repair rejected: repaired mask was implausible or empty"

    largest_improved = np.isfinite(original_largest) and np.isfinite(repaired_largest) and repaired_largest < original_largest
    ratio_improved = np.isfinite(original_ratio) and np.isfinite(repaired_ratio) and repaired_ratio < original_ratio
    score_improved = np.isfinite(original_score) and np.isfinite(repaired_score) and repaired_score >= original_score + min_improvement
    score_not_worse = np.isfinite(original_score) and np.isfinite(repaired_score) and repaired_score >= original_score - 2.0
    if score_improved and (largest_improved or ratio_improved):
        return True, "repair kept: reliability score improved and merged-gap evidence decreased"
    if largest_improved and ratio_improved and score_not_worse and not truthy_value(repaired_row.get("badMaskFlag", False)):
        return True, "repair kept: merged-gap evidence decreased without reducing reliability"
    return False, "repair rejected: QC improvement was insufficient"


def analyze_rbc_frame_with_optional_repair(image: np.ndarray, filename: str, params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run RBC segmentation, optionally repairing suspicious merged-gap masks."""
    original_result, original_row = analyze_rbc_frame_once(image, filename, params)
    repair_enabled = bool(params.get("adaptiveAntiMergingRepair", False))
    if not repair_enabled:
        return original_result, rbc_repair_columns(original_row, enabled=False, reason="not enabled")

    if not adaptive_repair_is_suspicious(original_row):
        return original_result, rbc_repair_columns(original_row, enabled=True, reason="not triggered: no huge-gap condition")

    repair_params = dict(params)
    repair_params["closeRadius"] = 0
    repair_params["doFillHoles"] = False
    repaired_result, repaired_row = analyze_rbc_frame_once(image, filename, repair_params)
    keep_repaired, reason = repaired_mask_is_acceptable(original_row, repaired_row, params)
    if keep_repaired:
        final_row = rbc_repair_columns(
            repaired_row,
            enabled=True,
            triggered=True,
            kept=True,
            reason=reason,
            before_row=original_row,
            after_row=repaired_row,
        )
        return repaired_result, final_row
    final_row = rbc_repair_columns(
        original_row,
        enabled=True,
        triggered=True,
        kept=False,
        reason=reason,
        before_row=original_row,
        after_row=repaired_row,
    )
    return original_result, final_row


def apply_post_start_mask_qc_exclusion(
    df: pd.DataFrame,
    used_mask: np.ndarray,
    params: dict[str, Any],
    qc_enabled: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Optionally exclude bad post-start masks from main-summary statistics."""
    used = np.asarray(used_mask, dtype=bool).copy()
    n_frames = len(df)
    if used.size != n_frames:
        used = np.zeros(n_frames, dtype=bool)
    scores = pd.to_numeric(df.get("maskReliabilityScore", pd.Series(np.nan, index=df.index)), errors="coerce")
    bad_flags = df.get("hardBadMaskFlag", df.get("badMaskFlag", pd.Series(False, index=df.index))).fillna(False).astype(bool)
    min_score = float(params.get("maskReliabilityMinForSummary", 60.0))
    bad_by_score = (scores < min_score) & bad_flags
    candidate_bad = (bad_flags | bad_by_score.fillna(False)).to_numpy(dtype=bool)
    exclusion_enabled = bool(
        qc_enabled
        and params.get(
            "excludeBadPostStartQCFramesFromTrustedAnalysis",
            params.get("excludeBadMaskQCFromMainSummary", False),
        )
    )
    excluded = used & candidate_bad if exclusion_enabled else np.zeros(n_frames, dtype=bool)
    if exclusion_enabled:
        used = used & ~candidate_bad

    n_excluded = int(np.count_nonzero(excluded))
    n_bad_candidates = int(np.count_nonzero(candidate_bad & used_mask.astype(bool))) if used_mask.size == n_frames else int(np.count_nonzero(candidate_bad))
    percent_excluded = 100.0 * n_excluded / n_frames if n_frames else 0.0
    finite_scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
    mean_score = float(finite_scores.mean()) if not finite_scores.empty else np.nan
    min_score_observed = float(finite_scores.min()) if not finite_scores.empty else np.nan
    risk_basis = percent_excluded if exclusion_enabled else (100.0 * n_bad_candidates / n_frames if n_frames else 0.0)
    risk = "low" if risk_basis < 5.0 else ("moderate" if risk_basis <= 20.0 else "high")
    summary = {
        "postStartBadFrameExclusionEnabled": bool(exclusion_enabled),
        "maskReliabilityMinForSummary": min_score,
        "nFramesExcludedByMaskQC": n_excluded,
        "nFramesFailingMaskQCInSummaryWindow": n_bad_candidates,
        "percentFramesExcludedByMaskQC": percent_excluded,
        "meanMaskReliabilityScore": mean_score,
        "minMaskReliabilityScore": min_score_observed,
        "sampleMaskQCRisk": risk,
    }
    return used, excluded, summary


def _mask_score_pair(
    df: pd.DataFrame,
    area_col: str,
    count_col: str,
    diam_col: str,
    xi_col: str,
    hard_bad_col: str,
    min_area: float,
    min_count: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return reliable mask-score flag, SASI-like, and GapScore-like series."""
    area = safe_numeric_series(df, area_col)
    count = safe_numeric_series(df, count_col)
    diam = safe_numeric_series(df, diam_col)
    xi = safe_numeric_series(df, xi_col)
    hard_bad = df.get(hard_bad_col, pd.Series(False, index=df.index)).fillna(False).astype(bool)
    reliable = area.ge(min_area).fillna(False) & count.ge(min_count).fillna(False) & xi.notna() & ~hard_bad
    score_input = pd.DataFrame(
        {"area": area, "count": count, "diameter": diam, "xi_mask": xi},
        index=df.index,
    )
    official = compute_official_mask_scores(
        score_input,
        xi_col="xi_mask",
        diameter_col="diameter",
        count_col="count",
        area_col="area",
        reference_mask=reliable.to_numpy(dtype=bool),
    )
    return (
        reliable,
        official["officialSASI_0_100"].where(reliable, np.nan),
        official["officialGapScore_0_100"].where(reliable, np.nan),
    )


def add_sparse_mask_morphology_columns(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Add raw/trusted morphology columns for sparse-mask support checks.

    The selected binary gap mask still controls raw area/count outputs. This
    helper only gates morphology-dependent trusted values when mask support is
    too sparse for stable Xi/diameter/size descriptors.
    """
    out = df.copy()
    if out.empty:
        return out

    enabled = bool(params.get("applySparseMaskMorphologyFilter", True))
    min_area = float(params.get("minReliableMaskAreaFracForMorphology", 0.03))
    min_objects = float(params.get("minReliableObjectsForMorphology", 10.0))
    area = (
        safe_numeric_series(out, "gapAreaFracFinal")
        .fillna(safe_numeric_series(out, "areaFracFinal"))
        .fillna(safe_numeric_series(out, "mask_gapAreaFrac"))
        .fillna(safe_numeric_series(out, "gapAreaFrac"))
        .fillna(safe_numeric_series(out, "areaFrac"))
    )
    count = (
        safe_numeric_series(out, "gapNtotalFinal")
        .fillna(safe_numeric_series(out, "nObjectsFinal"))
        .fillna(safe_numeric_series(out, "mask_gapNtotal"))
        .fillna(safe_numeric_series(out, "gapNtotal"))
        .fillna(safe_numeric_series(out, "nObjects"))
    )
    reliable = area.ge(min_area).fillna(False) & count.ge(min_objects).fillna(False)
    filtered = reliable.eq(False) & bool(enabled)

    out["applySparseMaskMorphologyFilter"] = enabled
    out["minReliableMaskAreaFracForMorphology"] = min_area
    out["minReliableObjectsForMorphology"] = min_objects
    out["maskMorphologyReliable"] = reliable.to_numpy(dtype=bool)
    out["sparseMaskMorphologyFiltered"] = filtered.to_numpy(dtype=bool)
    out["sparseMaskMorphologyFilteredFrames"] = int(filtered.sum())

    raw_diam = (
        safe_numeric_series(out, "gapMeanDiamFinal")
        .fillna(safe_numeric_series(out, "mask_gapMeanDiam_px"))
        .fillna(safe_numeric_series(out, "gapMeanDiam_px"))
        .fillna(safe_numeric_series(out, "meanDiam"))
    )
    raw_area_mean = (
        safe_numeric_series(out, "mask_gapMeanArea")
        .fillna(safe_numeric_series(out, "gapMeanArea"))
        .fillna(safe_numeric_series(out, "meanArea"))
    )
    raw_largest = (
        safe_numeric_series(out, "largestGapFracFinal")
        .fillna(safe_numeric_series(out, "mask_largestGapFrac"))
        .fillna(safe_numeric_series(out, "gapLargestFrac"))
        .fillna(safe_numeric_series(out, "largestObjectFrac"))
    )
    raw_xi_px = safe_numeric_series(out, "Xi_px")
    raw_xi_interp = safe_numeric_series(out, "Xi_interp_px")
    raw_xi_mask = safe_numeric_series(out, "Xi_mask")
    raw_xi = raw_xi_mask.fillna(raw_xi_interp).fillna(raw_xi_px)

    trust_mask = ~filtered
    out["raw_gapMeanDiam"] = raw_diam
    out["trusted_gapMeanDiam"] = raw_diam.where(trust_mask, np.nan)
    out["raw_gapMeanArea"] = raw_area_mean
    out["trusted_gapMeanArea"] = raw_area_mean.where(trust_mask, np.nan)
    out["raw_largestGapFrac"] = raw_largest
    out["trusted_largestGapFrac"] = raw_largest.where(trust_mask, np.nan)
    out["raw_Xi"] = raw_xi
    out["trusted_Xi"] = raw_xi.where(trust_mask, np.nan)
    out["raw_Xi_px"] = raw_xi_px
    out["trusted_Xi_px"] = raw_xi_px.where(trust_mask, np.nan)
    out["raw_Xi_interp_px"] = raw_xi_interp
    out["trusted_Xi_interp_px"] = raw_xi_interp.where(trust_mask, np.nan)
    if "Xi_mask" in out.columns:
        out["trusted_Xi_mask"] = raw_xi_mask.where(trust_mask, np.nan)
    if "SASI_0_100" in out.columns:
        out["raw_SASI_0_100"] = safe_numeric_series(out, "SASI_0_100")
        out["trusted_SASI_0_100"] = safe_numeric_series(out, "SASI_0_100").where(trust_mask, np.nan)
    if "GapScore_0_100" in out.columns:
        out["raw_GapScore_0_100"] = safe_numeric_series(out, "GapScore_0_100")
        out["trusted_GapScore_0_100"] = safe_numeric_series(out, "GapScore_0_100").where(trust_mask, np.nan)
    return out


def warn_sparse_mask_morphology_filter(df: pd.DataFrame, params: dict[str, Any], warnings: list[str]) -> None:
    """Append the user-facing sparse-mask morphology warning once per analysis."""
    if df.empty or not bool(params.get("applySparseMaskMorphologyFilter", True)):
        return
    if "sparseMaskMorphologyFiltered" not in df.columns:
        return
    filtered_count = int(df["sparseMaskMorphologyFiltered"].fillna(False).astype(bool).sum())
    if filtered_count <= 0:
        return
    message = (
        "Some frames had too little detected mask area to estimate morphology reliably. "
        "Xi and mean gap diameter were set to NaN for trusted analysis."
    )
    if message not in warnings:
        warnings.append(message)


def add_rbc_mask_based_score_columns(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Add MATLAB-v5-style mask-only RBC diagnostic scores.

    These scores use final/no-POD/POD binary mask metrics only. They do not use
    continuous intensity Xi, so early uniform frames cannot become trusted high
    score frames because of smooth illumination correlation.
    """
    out = df.copy()
    if out.empty:
        return out

    min_area = float(params.get("minReliableGapAreaFracForXi", 0.03))
    min_count = float(params.get("minReliableObjectsForXi", 10.0))
    if "hardBadMaskFlag" not in out.columns:
        out["hardBadMaskFlag"] = out.get("hugeGapFlag", pd.Series(False, index=out.index)).fillna(False).astype(bool)

    final_area = safe_numeric_series(out, "mask_gapAreaFrac").fillna(safe_numeric_series(out, "gapAreaFrac")).fillna(safe_numeric_series(out, "areaFrac"))
    final_count = safe_numeric_series(out, "mask_gapNtotal").fillna(safe_numeric_series(out, "gapNtotal")).fillna(safe_numeric_series(out, "nObjects"))
    final_diam = safe_numeric_series(out, "mask_gapMeanDiam_px").fillna(safe_numeric_series(out, "gapMeanDiam_px")).fillna(safe_numeric_series(out, "meanDiam"))
    final_largest = safe_numeric_series(out, "mask_largestGapFrac").fillna(safe_numeric_series(out, "gapLargestFrac")).fillna(safe_numeric_series(out, "largestObjectFrac"))
    out["mask_gapAreaFrac"] = final_area
    out["mask_gapNtotal"] = final_count
    out["mask_gapMeanDiam_px"] = final_diam
    out["mask_largestGapFrac"] = final_largest
    out["gapAreaFracFinal"] = final_area
    out["gapNtotalFinal"] = final_count
    out["gapMeanDiamFinal"] = final_diam
    out["largestGapFracFinal"] = final_largest
    out = add_sparse_mask_morphology_columns(out, params)
    sparse_reliable = out["maskMorphologyReliable"].fillna(False).astype(bool) if bool(params.get("applySparseMaskMorphologyFilter", True)) else pd.Series(True, index=out.index)

    reliable, _mask_sasi_qc, _mask_gap_score_qc = _mask_score_pair(
        out,
        "mask_gapAreaFrac",
        "mask_gapNtotal",
        "mask_gapMeanDiam_px",
        "Xi_mask",
        "hardBadMaskFlag",
        min_area,
        min_count,
    )
    reliable = reliable & sparse_reliable
    out["reliableForMaskScore"] = reliable.to_numpy(dtype=bool)
    reference = reliable.to_numpy(dtype=bool)
    fixed_reference = params.get("rbcScoreReferenceStats")
    if not isinstance(fixed_reference, dict) or not fixed_reference:
        fixed_reference = None
    official = compute_official_mask_scores(
        out,
        xi_col="Xi_mask",
        diameter_col="mask_gapMeanDiam_px",
        count_col="mask_gapNtotal",
        area_col="mask_gapAreaFrac",
        reference_mask=reference,
        reference_stats=fixed_reference,
    )
    mask_sasi_raw = official["officialSASIRaw"]
    mask_gap_score_raw = official["officialGapScoreRaw"]
    out["MaskSASI_raw"] = mask_sasi_raw
    out["MaskSASI_raw_0_100"] = official["officialSASI_0_100"]
    out["MaskSASI_QC_0_100"] = out["MaskSASI_raw_0_100"].where(reliable, np.nan)
    out["MaskSASI_0_100"] = out["MaskSASI_QC_0_100"]
    out["MaskSASI_like_final"] = out["MaskSASI_raw_0_100"]
    out["MaskGapScore_raw"] = mask_gap_score_raw
    out["MaskGapScore_raw_0_100"] = official["officialGapScore_0_100"]
    out["MaskGapScore_QC_0_100"] = out["MaskGapScore_raw_0_100"].where(reliable, np.nan)
    out["MaskGapScore_0_100"] = out["MaskGapScore_QC_0_100"]
    out["MaskGapScore_like_final"] = out["MaskGapScore_raw_0_100"]
    out["MaskAggIndex_raw_0_100"] = 0.50 * out["MaskSASI_raw_0_100"] + 0.50 * out["MaskGapScore_raw_0_100"]
    out["MaskAggIndex_QC_0_100"] = out["MaskAggIndex_raw_0_100"].where(reliable, np.nan)
    out["MaskAggIndex_0_100"] = out["MaskAggIndex_QC_0_100"]
    out["MaskSASI_isTrusted"] = reliable.to_numpy(dtype=bool)
    out["MaskGapScore_isTrusted"] = reliable.to_numpy(dtype=bool)
    out["MaskAggIndex_isTrusted"] = reliable.to_numpy(dtype=bool)
    out["OfficialMaskSASI_raw"] = out["MaskSASI_raw"]
    out["OfficialMaskSASI_0_100"] = out["MaskSASI_raw_0_100"]
    out["OfficialMaskGapScore_raw"] = out["MaskGapScore_raw"]
    out["OfficialMaskGapScore_0_100"] = out["MaskGapScore_raw_0_100"]
    out["SASI_raw"] = out["MaskSASI_raw"]
    out["SASI_0_100"] = out["MaskSASI_raw_0_100"]
    out["GapScore_raw"] = out["MaskGapScore_raw"]
    out["GapScore_0_100"] = out["MaskGapScore_raw_0_100"]
    out["AggIndex_0_100"] = (
        0.50 * out["GapScore_0_100"].fillna(50.0)
        + 0.30 * safe_numeric_series(out, "PTI_0_100").fillna(50.0)
        + 0.20 * safe_numeric_series(out, "S_struct_0_100").fillna(50.0)
    )
    out["AggIndex_raw"] = out["AggIndex_0_100"]
    out["scoreFormulaVersion"] = official["scoreFormulaVersion"]
    out["scoreNormalizationScope"] = official["scoreNormalizationScope"]
    out["primarySASISource"] = "final_binary_mask_xi"
    out["primaryGapScoreSource"] = "final_binary_mask_xi"
    mask_reasons = np.where(reliable.to_numpy(dtype=bool), "trusted", "mask too sparse, too few objects, Xi_mask unavailable, or hardBadMaskFlag")
    out["MaskSASI_QC_reason"] = mask_reasons
    out["MaskGapScore_QC_reason"] = mask_reasons
    out["MaskAggIndex_QC_reason"] = mask_reasons
    out["syllectogramSource_SASI"] = "final_selected_mask"
    out["syllectogramSource_intensity"] = "corrected_grayscale"

    if "conservativeNoPODAreaFrac" in out.columns:
        out["mask_gapAreaFrac_noPOD"] = safe_numeric_series(out, "conservativeNoPODAreaFrac")
        out["mask_gapNtotal_noPOD"] = safe_numeric_series(out, "conservativeNoPODNObjects")
        out["mask_gapMeanDiam_px_noPOD"] = safe_numeric_series(out, "conservativeNoPODMeanDiam_px").fillna(final_diam)
        out["mask_largestGapFrac_noPOD"] = safe_numeric_series(out, "conservativeNoPODLargestGapFrac").fillna(safe_numeric_series(out, "conservativeNoPODLargestFrac")).fillna(final_largest)
        if "Xi_mask_noPOD" in out.columns:
            no_pod_xi = safe_numeric_series(out, "Xi_mask_noPOD")
        else:
            no_pod_xi = safe_numeric_series(out, "Xi_mask").where(out["selectedRBCMethod"].astype(str).str.startswith("Conservative"), np.nan)
            out["Xi_mask_noPOD"] = no_pod_xi
        no_pod_hard = pd.Series(False, index=out.index)
        out["hardBadMaskFlag_noPOD"] = no_pod_hard
        rel_np, sasi_np, gap_np = _mask_score_pair(
            out,
            "mask_gapAreaFrac_noPOD",
            "mask_gapNtotal_noPOD",
            "mask_gapMeanDiam_px_noPOD",
            "Xi_mask_noPOD",
            "hardBadMaskFlag_noPOD",
            min_area,
            min_count,
        )
        out["reliableForMaskScore_noPOD"] = rel_np.to_numpy(dtype=bool)
        out["MaskSASI_like_noPOD"] = sasi_np
        out["MaskGapScore_like_noPOD"] = gap_np

    if "podRemovedAreaFrac" in out.columns:
        out["mask_gapAreaFrac_PODremoved"] = safe_numeric_series(out, "podRemovedAreaFrac")
        out["mask_gapNtotal_PODremoved"] = safe_numeric_series(out, "podRemovedNObjects")
        out["mask_gapMeanDiam_px_PODremoved"] = safe_numeric_series(out, "podRemovedMeanDiam_px").fillna(final_diam)
        out["mask_largestGapFrac_PODremoved"] = safe_numeric_series(out, "podRemovedLargestGapFrac").fillna(final_largest)
        if "Xi_mask_PODremoved" not in out.columns:
            out["Xi_mask_PODremoved"] = safe_numeric_series(out, "Xi_mask").where(out["selectedRBCMethod"].astype(str).str.startswith("Experimental"), np.nan)
        out["hardBadMaskFlag_PODremoved"] = False
        rel_pod, sasi_pod, gap_pod = _mask_score_pair(
            out,
            "mask_gapAreaFrac_PODremoved",
            "mask_gapNtotal_PODremoved",
            "mask_gapMeanDiam_px_PODremoved",
            "Xi_mask_PODremoved",
            "hardBadMaskFlag_PODremoved",
            min_area,
            min_count,
        )
        out["reliableForMaskScore_PODremoved"] = rel_pod.to_numpy(dtype=bool)
        out["MaskSASI_like_PODremoved"] = sasi_pod
        out["MaskGapScore_like_PODremoved"] = gap_pod

    return out


def add_rbc_qc_aware_score_columns(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Add RBC score columns that are safe for user-facing interpretation.

    The raw MATLAB-compatible score columns are preserved exactly. These
    additional columns prevent visually uniform/pre-gap frames from appearing
    as high-confidence aggregation merely because the continuous field has a
    long autocorrelation length. Trusted QC scores are recomputed from the
    selected final-mask Xi (Xi_mask -> Xi_QC) and are NaN when the mask/QC
    support is not reliable enough for interpretation.
    """
    out = df.copy()
    if out.empty:
        return out
    out = add_sparse_mask_morphology_columns(out, params)
    exclude_pre_start = bool(params.get("excludePreStartFramesFromTrustedAnalysis", True))
    exclude_bad_post_start = bool(
        params.get(
            "excludeBadPostStartQCFramesFromTrustedAnalysis",
            params.get("excludeBadMaskQCFromMainSummary", False),
        )
    )
    out["excludePreStartFramesFromTrustedAnalysis"] = exclude_pre_start
    out["excludeBadPostStartQCFramesFromTrustedAnalysis"] = exclude_bad_post_start

    if "includedInMainSummary" in out.columns:
        included = out["includedInMainSummary"].fillna(False).astype(bool)
    elif "usedForMainSummary" in out.columns:
        included = out["usedForMainSummary"].fillna(False).astype(bool)
    else:
        included = pd.Series(True, index=out.index)

    if "isAnaFinal" in out.columns:
        before_start_frame = ~out["isAnaFinal"].fillna(False).astype(bool)
    elif "finalStartFrameUsed" in out.columns and "analysisIndex_1based" in out.columns:
        final_start = pd.to_numeric(out["finalStartFrameUsed"], errors="coerce")
        analysis_index = pd.to_numeric(out["analysisIndex_1based"], errors="coerce")
        first_start = final_start.dropna().iloc[0] if final_start.notna().any() else np.nan
        before_start_frame = analysis_index.lt(first_start).fillna(False) if np.isfinite(first_start) else pd.Series(False, index=out.index)
    else:
        before_start_frame = pd.Series(False, index=out.index)
    out["beforeStartFrame"] = before_start_frame.to_numpy(dtype=bool)

    label_columns = ["rbcQCLabel", "rbcFrameQC", "frameClass", "conservativeFrameClass"]
    label_parts = []
    for column in label_columns:
        if column in out.columns:
            label_parts.append(out[column].fillna("").astype(str))
    if label_parts:
        labels = label_parts[0].copy()
        for part in label_parts[1:]:
            labels = labels.str.cat(part, sep=" | ")
    else:
        labels = pd.Series("", index=out.index)
    labels = labels.fillna("").astype(str).str.lower()
    weak_or_unreliable = labels.str.contains("weak|pre-gap|startup|no reliable|not reliable", regex=True)
    possible_artifact = labels.str.contains("artefact|artifact|residual|illumination", regex=True)
    warning_only = labels.str.contains("pod-sensitive|uncertain", regex=True)
    bad_mask = out.get("badMaskFlag", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    hard_bad_mask = out.get("hardBadMaskFlag", bad_mask).fillna(False).astype(bool)
    pod_sensitive = out.get("podSensitiveFlag", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    spatial_sensitive = out.get("spatialCorrectionSensitiveFlag", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    raw_gain_sensitive = out.get("rawGainCorrectionSensitiveFlag", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    conservative_warning = out.get("conservativeWarningFlag", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    unstable_warning = out.get("unstableSegmentationFlag", pd.Series(False, index=out.index)).fillna(False).astype(bool) & ~hard_bad_mask
    huge_gap = out.get("hugeGapFlag", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    low_confidence_flag = out.get("rbcLowConfidence", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    if "scoresConfidence" in out.columns:
        low_confidence_flag = low_confidence_flag | out["scoresConfidence"].fillna("").astype(str).str.lower().eq("low")
    low_confidence_flag = low_confidence_flag | weak_or_unreliable | possible_artifact
    out["lowConfidenceFlag"] = low_confidence_flag.to_numpy(dtype=bool)
    min_score = float(params.get("maskReliabilityMinForXi", params.get("maskReliabilityMinForSummary", 60.0)))
    reliability = pd.to_numeric(out.get("maskReliabilityScore", pd.Series(np.nan, index=out.index)), errors="coerce")
    strict_mask_qc = bool(exclude_bad_post_start or params.get("postStartBadFrameExclusionEnabled", False))
    low_reliability = reliability.notna() & (reliability < min_score)
    moderate_reliability_warning = reliability.notna() & reliability.ge(min_score).fillna(False) & reliability.lt(80.0).fillna(False)
    warning_mask = (
        (bad_mask & ~hard_bad_mask)
        | warning_only
        | pod_sensitive
        | spatial_sensitive
        | raw_gain_sensitive
        | conservative_warning
        | unstable_warning
        | moderate_reliability_warning
    )
    out["warningMaskFlag"] = warning_mask.to_numpy(dtype=bool)

    min_area_for_xi = float(params.get("minReliableGapAreaFracForXi", 0.03))
    min_objects_for_xi = float(params.get("minReliableObjectsForXi", 10.0))
    area_series = safe_numeric_series(out, "areaFracFinal").fillna(safe_numeric_series(out, "gapAreaFrac")).fillna(safe_numeric_series(out, "areaFrac"))
    object_series = safe_numeric_series(out, "nObjectsFinal").fillna(safe_numeric_series(out, "gapNtotal")).fillna(safe_numeric_series(out, "nObjects"))
    low_area = area_series.notna() & (area_series < min_area_for_xi)
    too_few_objects = object_series.notna() & (object_series < min_objects_for_xi)
    sparse_filter_enabled = bool(params.get("applySparseMaskMorphologyFilter", True))
    morphology_reliable = out["maskMorphologyReliable"].fillna(False).astype(bool) if "maskMorphologyReliable" in out.columns else pd.Series(True, index=out.index)
    sparse_morphology_filtered = (
        out["sparseMaskMorphologyFiltered"].fillna(False).astype(bool)
        if "sparseMaskMorphologyFiltered" in out.columns
        else pd.Series(False, index=out.index)
    )

    startup_not_reliable = pd.Series(False, index=out.index)

    use_mask_xi = bool(params.get("useMaskXiForRBC", True))
    if "Xi_raw_intensity" not in out.columns:
        out["Xi_raw_intensity"] = safe_numeric_series(out, "Xi_interp_px")
    xi_mask = safe_numeric_series(out, "Xi_mask")
    xi_raw = safe_numeric_series(out, "Xi_raw_intensity").fillna(safe_numeric_series(out, "Xi_interp_px")).fillna(safe_numeric_series(out, "Xi_px"))
    xi_candidate = xi_mask if use_mask_xi else xi_raw
    if sparse_filter_enabled:
        xi_candidate = xi_candidate.where(morphology_reliable, np.nan)
    out["SASI_raw_0_100"] = safe_numeric_series(out, "SASI_0_100")
    out["GapScore_raw_0_100"] = safe_numeric_series(out, "GapScore_0_100")
    out["PTI_raw_0_100"] = safe_numeric_series(out, "PTI_0_100")
    out["AggIndex_raw_0_100"] = safe_numeric_series(out, "AggIndex_0_100")
    score_raw_available = (
        out[["SASI_raw_0_100", "GapScore_raw_0_100", "PTI_raw_0_100", "AggIndex_raw_0_100"]]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .any(axis=1)
    )

    if "reliableForMaskScore" in out.columns:
        base_reliable = out["reliableForMaskScore"].fillna(False).astype(bool)
    else:
        base_reliable = area_series.ge(min_area_for_xi).fillna(False) & object_series.ge(min_objects_for_xi).fillna(False) & xi_candidate.notna()
    if sparse_filter_enabled:
        base_reliable = base_reliable & morphology_reliable

    diam_for_trusted_scores = safe_numeric_series(out, "trusted_gapMeanDiam")
    if not sparse_filter_enabled:
        diam_for_trusted_scores = diam_for_trusted_scores.fillna(safe_numeric_series(out, "gapMeanDiam_px"))
    required_raw_mask_metrics_finite = area_series.notna() & object_series.notna() & safe_numeric_series(out, "gapMeanDiam_px").notna()
    required_trusted_mask_metrics_finite = area_series.notna() & object_series.notna() & diam_for_trusted_scores.notna()
    hard_failure = hard_bad_mask | huge_gap | ~required_raw_mask_metrics_finite | ~score_raw_available
    strict_untrusted = low_reliability | low_area | too_few_objects | startup_not_reliable
    if sparse_filter_enabled:
        strict_untrusted = strict_untrusted | sparse_morphology_filtered | ~morphology_reliable | ~required_trusted_mask_metrics_finite
    trusted_for_score = ~hard_failure
    if sparse_filter_enabled:
        trusted_for_score = trusted_for_score & morphology_reliable & ~sparse_morphology_filtered
    if exclude_pre_start:
        trusted_for_score = trusted_for_score & ~before_start_frame
    if exclude_bad_post_start:
        trusted_for_score = trusted_for_score & ~(~before_start_frame & strict_untrusted)

    used_in_summary = included.copy()
    if not exclude_pre_start:
        used_in_summary = used_in_summary | before_start_frame
    else:
        used_in_summary = used_in_summary & ~before_start_frame
    if exclude_bad_post_start:
        used_in_summary = used_in_summary & trusted_for_score
    else:
        used_in_summary = used_in_summary & ~hard_failure
    out["usedInSummary"] = used_in_summary.to_numpy(dtype=bool)

    xi_qc_valid = trusted_for_score & xi_candidate.notna()
    out["trustedForScore"] = trusted_for_score.to_numpy(dtype=bool)
    visual_grey_excluded = ~trusted_for_score
    visual_warning_only = warning_mask & trusted_for_score & ~visual_grey_excluded
    out["hardFailure"] = hard_failure.to_numpy(dtype=bool)
    out["visualGreyExcluded"] = visual_grey_excluded.to_numpy(dtype=bool)
    out["visualWarningOnly"] = visual_warning_only.to_numpy(dtype=bool)
    out["Xi_QC"] = xi_candidate.where(xi_qc_valid, np.nan)
    xi_qc_finite = out["Xi_QC"].notna().to_numpy(dtype=bool)
    out["xiSource"] = np.select(
        [xi_qc_finite & bool(use_mask_xi), xi_qc_finite & (not bool(use_mask_xi))],
        ["mask", "raw_intensity_diagnostic"],
        default="not_reliable",
    )

    reasons: list[str] = []
    for idx in out.index:
        frame_reasons: list[str] = []
        if not bool(used_in_summary.loc[idx]):
            frame_reasons.append("not included in main summary")
        if bool(before_start_frame.loc[idx]):
            if exclude_pre_start:
                frame_reasons.append("before stable start")
            else:
                frame_reasons.append("pre-start frame included by user option")
        if bool(low_confidence_flag.loc[idx]) and not bool(trusted_for_score.loc[idx]):
            frame_reasons.append("low-confidence frame")
        if bool(weak_or_unreliable.loc[idx]):
            frame_reasons.append("weak/pre-gap/startup morphology")
        if bool(possible_artifact.loc[idx]):
            frame_reasons.append("possible residual/illumination artefact")
        if bool(warning_only.loc[idx]):
            frame_reasons.append("POD/spatial sensitivity warning")
        if bool(hard_bad_mask.loc[idx]):
            frame_reasons.append("hardBadMaskFlag")
        if bool(warning_mask.loc[idx]):
            frame_reasons.append("warningMaskFlag")
        elif bool(bad_mask.loc[idx]):
            frame_reasons.append("badMaskFlag")
        if bool(huge_gap.loc[idx]):
            frame_reasons.append("hugeGapFlag")
        if bool(hard_failure.loc[idx]):
            frame_reasons.append("hard/catastrophic failure")
        if bool(low_reliability.loc[idx]) and strict_mask_qc and not bool(before_start_frame.loc[idx]):
            frame_reasons.append(f"maskReliabilityScore < {min_score:g}")
        if bool(low_area.loc[idx]) and strict_mask_qc and not bool(before_start_frame.loc[idx]):
            frame_reasons.append(f"areaFrac < {min_area_for_xi:g}")
        if bool(too_few_objects.loc[idx]) and strict_mask_qc and not bool(before_start_frame.loc[idx]):
            frame_reasons.append(f"nObjects < {min_objects_for_xi:g}")
        if bool(sparse_morphology_filtered.loc[idx]):
            frame_reasons.append(
                "sparse mask morphology filtered "
                f"(areaFrac < {float(params.get('minReliableMaskAreaFracForMorphology', 0.03)):g} "
                f"or nObjects < {float(params.get('minReliableObjectsForMorphology', 10.0)):g})"
            )
        if bool(startup_not_reliable.loc[idx]):
            frame_reasons.append("startupReliable false")
        if not bool(trusted_for_score.loc[idx]):
            frame_reasons.append("not trusted for score")
        if not frame_reasons:
            frame_reasons.append("interpretable reliable gap morphology with trusted mask Xi")
        reasons.append("; ".join(frame_reasons))

    score_valid = trusted_for_score
    out["rbcScoreInterpretable"] = score_valid.to_numpy(dtype=bool)
    out["rbcScoreInterpretationFlag"] = np.where(score_valid, "interpretable_trusted_score", "not_reliable_for_trusted_score")
    out["rbcScoreQCReason"] = reasons
    out["scoreSource"] = np.where(
        score_valid,
        "official_mask_score_recomputed_from_qc_inputs",
        "not_reliable",
    )

    reference_mask = score_valid.to_numpy(dtype=bool)
    out["diameterForTrustedScore"] = diam_for_trusted_scores
    out["countForTrustedScore"] = object_series
    out["areaForTrustedScore"] = area_series
    fixed_reference = params.get("rbcScoreReferenceStats")
    if not isinstance(fixed_reference, dict) or not fixed_reference:
        fixed_reference = None
    official_qc = compute_official_mask_scores(
        out,
        xi_col="Xi_QC",
        diameter_col="diameterForTrustedScore",
        count_col="countForTrustedScore",
        area_col="areaForTrustedScore",
        reference_mask=reference_mask,
        reference_stats=fixed_reference,
    )
    z_xi = matlab_robust_z(out["Xi_QC"], reference_mask)
    z_area = matlab_robust_z(area_series, reference_mask)
    z_slope = matlab_robust_z(safe_numeric_series(out, "gapAreaSlope"), reference_mask)

    sasi_qc_raw = official_qc["officialSASIRaw"]
    gap_score_qc_raw = official_qc["officialGapScoreRaw"]
    pti_qc_raw = 0.40 * z_slope + 0.30 * z_xi + 0.30 * z_area

    out["SASI_QC_raw"] = sasi_qc_raw.where(score_valid, np.nan)
    out["GapScore_QC_raw"] = gap_score_qc_raw.where(score_valid, np.nan)
    out["PTI_QC_raw"] = pti_qc_raw.where(score_valid, np.nan)

    out["SASI_raw_0_100"] = safe_numeric_series(out, "SASI_0_100")
    out["GapScore_raw_0_100"] = safe_numeric_series(out, "GapScore_0_100")
    out["PTI_raw_0_100"] = safe_numeric_series(out, "PTI_0_100")
    out["AggIndex_raw_0_100"] = safe_numeric_series(out, "AggIndex_0_100")
    sasi_qc_scaled = official_qc["officialSASI_0_100"].where(score_valid, np.nan)
    gap_qc_scaled = official_qc["officialGapScore_0_100"].where(score_valid, np.nan)
    pti_qc_scaled = sigmoid100_preserve_nan(pti_qc_raw).where(score_valid, np.nan)
    s_struct_qc = safe_numeric_series(out, "S_struct_0_100").where(score_valid, np.nan)
    agg_qc_scaled = (
        0.50 * gap_qc_scaled.fillna(50.0)
        + 0.30 * pti_qc_scaled.fillna(50.0)
        + 0.20 * s_struct_qc.fillna(50.0)
    ).where(score_valid & gap_qc_scaled.notna(), np.nan)

    out["scoreFormulaVersion"] = official_qc["scoreFormulaVersion"]
    out["scoreNormalizationScope"] = official_qc["scoreNormalizationScope"]
    out["primarySASISource"] = "trusted_final_binary_mask_xi"
    out["primaryGapScoreSource"] = "trusted_final_binary_mask_xi"
    out["score_isTrusted"] = score_valid.to_numpy(dtype=bool)
    out["Xi_isTrusted"] = xi_qc_valid.to_numpy(dtype=bool)
    out["SASI_isTrusted"] = (score_valid & sasi_qc_scaled.notna()).to_numpy(dtype=bool)
    out["GapScore_isTrusted"] = (score_valid & gap_qc_scaled.notna()).to_numpy(dtype=bool)
    out["PTI_isTrusted"] = (score_valid & pti_qc_scaled.notna()).to_numpy(dtype=bool)
    out["AggIndex_isTrusted"] = (score_valid & agg_qc_scaled.notna()).to_numpy(dtype=bool)
    out["SASI_QC_0_100"] = sasi_qc_scaled
    out["GapScore_QC_0_100"] = gap_qc_scaled
    out["PTI_QC_0_100"] = pti_qc_scaled
    out["AggIndex_QC_0_100"] = agg_qc_scaled
    out["SASI_QC"] = out["SASI_QC_0_100"]
    out["GapScore_QC"] = out["GapScore_QC_0_100"]
    out["PTI_QC"] = out["PTI_QC_0_100"]
    out["AggIndex_QC"] = out["AggIndex_QC_0_100"]
    out["trusted_SASI_0_100"] = out["SASI_QC_0_100"]
    out["trusted_GapScore_0_100"] = out["GapScore_QC_0_100"]
    out["trusted_PTI_0_100"] = out["PTI_QC_0_100"]
    out["trusted_AggIndex_0_100"] = out["AggIndex_QC_0_100"]
    reason_series = pd.Series(reasons, index=out.index, dtype=object)
    trusted_reason_values = np.where(score_valid.to_numpy(dtype=bool), "trusted", reason_series)
    trusted_reason_values = np.where(
        score_valid.to_numpy(dtype=bool) & visual_warning_only.to_numpy(dtype=bool),
        "trusted with warningMaskFlag",
        trusted_reason_values,
    )
    trusted_reason_values = np.where(
        score_valid.to_numpy(dtype=bool) & before_start_frame.to_numpy(dtype=bool) & (not exclude_pre_start),
        "pre-start frame included by user option",
        trusted_reason_values,
    )
    trusted_reason = pd.Series(trusted_reason_values, index=out.index, dtype=object)
    xi_trusted_reason = pd.Series(np.where(xi_qc_valid.to_numpy(dtype=bool), "trusted", reason_series), index=out.index, dtype=object)
    out["score_QC_reason"] = trusted_reason
    out["Xi_QC_reason"] = xi_trusted_reason
    out["SASI_QC_reason"] = trusted_reason
    out["GapScore_QC_reason"] = trusted_reason
    out["PTI_QC_reason"] = trusted_reason
    out["AggIndex_QC_reason"] = trusted_reason
    reason_not_trusted = trusted_reason.astype(str).str.lower().ne("trusted")
    visual_qc_excluded = visual_grey_excluded
    out["visualQCExcluded"] = visual_qc_excluded.to_numpy(dtype=bool)
    out["QCExcludedFrames"] = int(visual_qc_excluded.sum())
    out["hardBadMaskFrames"] = int(hard_bad_mask.sum())
    out["warningMaskFrames"] = int(warning_mask.sum())
    out["lowConfidenceFrames"] = int(low_confidence_flag.sum())
    pre_start_excluded_mask = before_start_frame & ~trusted_for_score
    post_start_qc_excluded_mask = (~before_start_frame) & visual_qc_excluded
    post_start_qc_warning_mask = (~before_start_frame) & warning_mask
    out["preStartFramesDetected"] = int(before_start_frame.sum())
    out["beforeStartExcludedFrames"] = int(pre_start_excluded_mask.sum())
    out["postStartQCWarningFrames"] = int(post_start_qc_warning_mask.sum())
    out["postStartQCExcludedFrames"] = int(post_start_qc_excluded_mask.sum())
    out["visualWarningOnlyFrames"] = int(visual_warning_only.sum())
    out["trustedForScoreFrames"] = int(score_valid.sum())
    out["usedInSummaryFrames"] = int(used_in_summary.sum())
    out["framesUsedForMainSummary"] = int(used_in_summary.sum())
    trusted_score_frames = int(score_valid.sum())
    qc_excluded_frames = int(visual_qc_excluded.sum())
    hard_bad_frames = int(hard_bad_mask.sum())
    warning_frames = int(warning_mask.sum())
    low_confidence_frames = int(low_confidence_flag.sum())
    sparse_filtered_frames = int(sparse_morphology_filtered.sum())
    pre_start_detected_frames = int(before_start_frame.sum())
    before_start_frames = int(pre_start_excluded_mask.sum())
    post_start_qc_warning_frames = int(post_start_qc_warning_mask.sum())
    post_start_qc_excluded_frames = int(post_start_qc_excluded_mask.sum())
    visual_warning_only_frames = int(visual_warning_only.sum())
    used_in_summary_frames = int(used_in_summary.sum())
    qc_summary = params.setdefault("rbc_qc_summary", {})
    previous_interpretable = bool(qc_summary.get("scoresInterpretable", trusted_score_frames > 0))
    warning_present = bool(warning_frames > 0 or low_confidence_frames > 0 or hard_bad_frames > 0 or int(huge_gap.sum()) > 0)
    scores_interpretable = bool(previous_interpretable and trusted_score_frames > 0)
    qc_summary.update(
        {
            "trustedScoreFrameCount": trusted_score_frames,
            "trustedForScoreFrames": trusted_score_frames,
            "QCExcludedFrames": qc_excluded_frames,
            "QCExcludedFrameCount": qc_excluded_frames,
            "hardBadMaskFrames": hard_bad_frames,
            "hardBadMaskFrameCount": hard_bad_frames,
            "warningMaskFrames": warning_frames,
            "warningMaskFrameCount": warning_frames,
            "visualWarningOnlyFrames": visual_warning_only_frames,
            "visualWarningOnlyFrameCount": visual_warning_only_frames,
            "lowConfidenceFrames": low_confidence_frames,
            "lowConfidenceFrameCount": low_confidence_frames,
            "sparseMaskMorphologyFilteredFrames": sparse_filtered_frames,
            "sparseMaskMorphologyFilteredFrameCount": sparse_filtered_frames,
            "applySparseMaskMorphologyFilter": bool(params.get("applySparseMaskMorphologyFilter", True)),
            "minReliableMaskAreaFracForMorphology": float(params.get("minReliableMaskAreaFracForMorphology", 0.03)),
            "minReliableObjectsForMorphology": float(params.get("minReliableObjectsForMorphology", 10.0)),
            "preStartFramesDetected": pre_start_detected_frames,
            "preStartFrameCount": pre_start_detected_frames,
            "beforeStartExcludedFrames": before_start_frames,
            "beforeStartExcludedFrameCount": before_start_frames,
            "postStartQCWarningFrames": post_start_qc_warning_frames,
            "postStartQCWarningFrameCount": post_start_qc_warning_frames,
            "postStartQCExcludedFrames": post_start_qc_excluded_frames,
            "postStartQCExcludedFrameCount": post_start_qc_excluded_frames,
            "usedInSummaryFrames": used_in_summary_frames,
            "usedInSummaryFrameCount": used_in_summary_frames,
            "framesUsedForMainSummary": used_in_summary_frames,
            "usedInTrustedSummaryFrames": used_in_summary_frames,
            "visualQCExcludedFrameCount": qc_excluded_frames,
            "excludePreStartFramesFromTrustedAnalysis": exclude_pre_start,
            "excludeBadPostStartQCFramesFromTrustedAnalysis": exclude_bad_post_start,
            "scoresInterpretable": scores_interpretable,
            "scoresInterpretableWithWarnings": bool(scores_interpretable and warning_present),
        }
    )
    out["SASI_interpretable_0_100"] = out["SASI_QC"]
    out["GapScore_interpretable_0_100"] = out["GapScore_QC"]
    out["PTI_interpretable_0_100"] = out["PTI_QC"]
    out["AggIndex_interpretable_0_100"] = out["AggIndex_QC"]

    return out


def contiguous_true_runs(values: pd.Series | np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive zero-based start/end positions for contiguous True runs."""
    arr = pd.Series(values).fillna(False).astype(bool).to_numpy()
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for position, value in enumerate(arr):
        if value and start is None:
            start = position
        elif not value and start is not None:
            runs.append((start, position - 1))
            start = None
    if start is not None:
        runs.append((start, len(arr) - 1))
    return runs


def stable_reliable_window_mask(values: pd.Series, min_run: int = 3) -> tuple[np.ndarray, int | None, list[tuple[int, int]]]:
    """Return reliable frames from the first stable run onward and the first stable start."""
    reliable = values.fillna(False).astype(bool).reset_index(drop=True)
    if min_run <= 0:
        return np.zeros(len(reliable), dtype=bool), None, []
    runs = contiguous_true_runs(reliable)
    stable_runs = [(start, end) for start, end in runs if (end - start + 1) >= min_run]
    mask = np.zeros(len(reliable), dtype=bool)
    if not stable_runs:
        return mask, None, []

    first_stable_start = stable_runs[0][0]
    positions = np.arange(len(reliable))
    mask = reliable.to_numpy(dtype=bool) & (positions >= first_stable_start)
    return mask, first_stable_start, stable_runs


def frame_identity_at_position(df: pd.DataFrame, position: int | None, column: str) -> float:
    """Return a frame/index identifier from a zero-based row position."""
    if position is None or column not in df.columns or position < 0 or position >= len(df):
        return np.nan
    value = pd.to_numeric(pd.Series([df.iloc[position].get(column, np.nan)]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else np.nan


def finite_position_for_frame(df: pd.DataFrame, frame_value: float) -> int | None:
    """Return row position for an original 1-based frame value."""
    if not np.isfinite(frame_value):
        return None
    frame_col = "originalImageIndex_1based" if "originalImageIndex_1based" in df.columns else "analysisIndex_1based"
    if frame_col not in df.columns:
        return None
    values = pd.to_numeric(df[frame_col], errors="coerce").to_numpy(dtype=float)
    matches = np.flatnonzero(np.isfinite(values) & (values >= frame_value))
    return int(matches[0]) if matches.size else None


def first_position_at_or_after_time(df: pd.DataFrame, time_s: float) -> int | None:
    """Return first row position with matlabTime_s/time_s at or after the requested time."""
    time_col = "matlabTime_s" if "matlabTime_s" in df.columns else ("time_s" if "time_s" in df.columns else None)
    if time_col is None:
        fps = max(float(df.attrs.get("fps", 1.0) or 1.0), 1e-9)
        frame_col = "originalImageIndex_1based" if "originalImageIndex_1based" in df.columns else None
        if frame_col is None:
            return None
        times = (pd.to_numeric(df[frame_col], errors="coerce").to_numpy(dtype=float) - 1.0) / fps
    else:
        times = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    matches = np.flatnonzero(np.isfinite(times) & (times >= float(time_s)))
    return int(matches[0]) if matches.size else None


def centered_moving_median(series: pd.Series, window: int) -> pd.Series:
    """Return a NaN-safe centered moving median for startup QC metrics."""
    window = max(1, int(window or 1))
    if window % 2 == 0:
        window += 1
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return numeric.rolling(window=window, center=True, min_periods=1).median()


def normalize_by_sample_max(series: pd.Series) -> pd.Series:
    """Normalize a nonnegative metric by its finite sample maximum."""
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    max_value = float(finite.max()) if not finite.empty else np.nan
    if not np.isfinite(max_value) or abs(max_value) <= 1e-12:
        return pd.Series(np.nan, index=values.index, dtype="float64")
    return values / (max_value + 1e-12)


def early_robust_normalize(series: pd.Series, early_end: int) -> pd.Series:
    """Normalize a startup signal using robust early-window percentiles."""
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    early = values.iloc[: max(1, int(early_end))].dropna()
    if early.empty:
        return pd.Series(np.nan, index=values.index, dtype="float64")
    p5, p95 = np.percentile(early.to_numpy(dtype=float), [5, 95])
    denom = p95 - p5
    if not np.isfinite(denom):
        return pd.Series(np.nan, index=values.index, dtype="float64")
    return ((values - p5) / (denom + 1e-12)).clip(lower=0.0, upper=1.0)


def conservative_mask_plateau_start_position(
    df: pd.DataFrame,
    params: dict[str, Any],
    min_run: int = 3,
) -> tuple[int | None, dict[str, Any]]:
    """Find a conservative RBC fallback start from sustained mask development."""
    if df.empty:
        return None, {"autoStartMaskFallbackReason": "empty table"}
    area = pd.to_numeric(df.get("gapAreaFrac", pd.Series(np.nan, index=df.index)), errors="coerce").replace([np.inf, -np.inf], np.nan)
    count = pd.to_numeric(df.get("gapNtotal", pd.Series(np.nan, index=df.index)), errors="coerce").replace([np.inf, -np.inf], np.nan)
    diam = pd.to_numeric(df.get("gapMeanDiam_px", pd.Series(np.nan, index=df.index)), errors="coerce").replace([np.inf, -np.inf], np.nan)
    min_area = float(params.get("autoStartMaskMinAreaFrac", params.get("minReliableGapAreaFracForXi", 0.03)))
    min_count = float(params.get("autoStartMaskMinGapNtotal", params.get("minReliableObjectsForXi", 20)))
    min_diam = float(params.get("autoStartMaskMinMeanDiam_px", 2.0))
    run_len = max(1, int(params.get("autoStartMaskMinConsecutiveFrames", min_run)))
    valid = (
        area.ge(min_area).fillna(False)
        & count.ge(min_count).fillna(False)
        & diam.ge(min_diam).fillna(False)
    ).to_numpy(dtype=bool)
    for start, end in contiguous_true_runs(valid):
        if end - start + 1 >= run_len:
            return int(start), {
                "autoStartMaskFallbackReason": "sustained conservative mask plateau",
                "autoStartMaskMinAreaFrac": min_area,
                "autoStartMaskMinGapNtotal": min_count,
                "autoStartMaskMinMeanDiam_px": min_diam,
                "autoStartMaskMinConsecutiveFrames": run_len,
                "autoStartMaskFallbackRunStart": int(start + 1),
                "autoStartMaskFallbackRunEnd": int(end + 1),
            }
    return None, {
        "autoStartMaskFallbackReason": "no sustained conservative mask plateau",
        "autoStartMaskMinAreaFrac": min_area,
        "autoStartMaskMinGapNtotal": min_count,
        "autoStartMaskMinMeanDiam_px": min_diam,
        "autoStartMaskMinConsecutiveFrames": run_len,
    }


def dominant_mask_signal_for_mode(df: pd.DataFrame, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Pick the mask-derived signal a dominant-transition start should track for the active mode.

    Conservative / POD RBC modes are plasma-gap analyses, so the start tracks the gap
    mask development (gapAreaFrac preferred). Object mode tracks object occupancy. The
    returned signal is purely mask-derived, consistent with what the mode measures.
    """
    selected_method = str(params.get("rbcAnalysisMethod", ""))
    if selected_method in {"Conservative RBC Gap Mode", "Experimental POD-removed RBC Mode"}:
        candidates = ["gapAreaFrac", "GapScore_raw_0_100", "GapScore", "aggOccFrac", "gapNtotal"]
    else:
        candidates = ["aggOccFrac", "objectAreaFraction", "objectOccFrac", "objectAreaFrac", "gapAreaFrac"]
    for col in candidates:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if int(series.notna().sum()) >= 3:
                return series.to_numpy(dtype=float), col
    return np.full(len(df), np.nan, dtype=float), ""


def dominant_transition_start_position(
    values: np.ndarray,
    search_end: int | None = None,
    min_run: int = 3,
    min_rel_change: float = 0.15,
) -> tuple[int | None, dict[str, Any]]:
    """Detect a start frame at the dominant sustained transition (rise OR fall).

    Unlike the first-rise detector (which only sees rises and can lock onto an early
    saturated plateau), this finds the strongest change in the signal magnitude and
    anchors the start just before that transition. Designed for signals that may start
    saturated and then crash (e.g. raw indices) or build up (e.g. gap development).
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    if n == 0:
        return None, {"dominantReason": "empty signal"}
    finite = np.isfinite(v)
    if int(finite.sum()) < 3:
        return None, {"dominantReason": "insufficient finite values"}
    idx = np.arange(n, dtype=float)
    vfill = v.copy()
    if not finite.all():
        vfill[~finite] = np.interp(idx[~finite], idx[finite], v[finite])
    sig_range = float(np.nanmax(vfill) - np.nanmin(vfill))
    if sig_range <= 0:
        return None, {"dominantReason": "flat signal", "dominantSignalRange": sig_range}
    slope = np.gradient(vfill)
    abs_slope = np.abs(slope)
    end = min(n, max(1, int(search_end))) if search_end else n
    region = abs_slope[:end]
    peak = int(np.argmax(region))
    peak_mag = float(region[peak])
    # Require the per-frame transition to be meaningful relative to the signal range.
    min_needed = (float(min_rel_change) * sig_range) / float(max(1, min_run))
    if peak_mag < min_needed:
        return None, {
            "dominantReason": "no dominant transition above threshold",
            "dominantPeakSlope": peak_mag,
            "dominantMinNeededSlope": min_needed,
            "dominantSignalRange": sig_range,
        }
    # Anchor just before the transition: walk back to where the change begins.
    thr = 0.2 * peak_mag
    start = peak
    while start > 0 and np.isfinite(abs_slope[start - 1]) and abs_slope[start - 1] >= thr:
        start -= 1
    return int(start), {
        "dominantReason": "dominant transition detected",
        "dominantPeakSlope": peak_mag,
        "dominantTransitionFrame": int(peak + 1),
        "dominantSignalRange": sig_range,
        "dominantDirection": "fall" if slope[peak] < 0 else "rise",
    }


def add_first_rise_auto_start_qc(metrics_df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray, int | None, list[tuple[int, int]]]:
    """Detect one RBC aggregation start frame and include all later frames."""
    out = metrics_df.copy()
    n_frames = len(out)
    if n_frames == 0:
        return out, {}, np.zeros(0, dtype=bool), None, []

    mode = str(params.get("startFrameMode", "Auto first-rise detector"))
    fps = max(float(params.get("matlab_fps", 1.0) or 1.0), 1e-9)
    smooth_win = max(1, int(round(float(params.get("autoStartSmoothWin_s", 7.0)) * fps)))
    if smooth_win % 2 == 0:
        smooth_win += 1
    time_values = pd.to_numeric(out.get("matlabTime_s", pd.Series(np.arange(n_frames) / fps, index=out.index)), errors="coerce")
    if not time_values.notna().any():
        time_values = pd.Series(np.arange(n_frames) / fps, index=out.index, dtype="float64")

    raw_bf = centered_moving_median(out.get("rawBrightFrac_autoStart", pd.Series(np.nan, index=out.index)), smooth_win)
    raw_ct = centered_moving_median(out.get("rawContrast01_autoStart", pd.Series(np.nan, index=out.index)), smooth_win)
    skel = centered_moving_median(out.get("aggSkelLen", pd.Series(np.nan, index=out.index)), smooth_win)
    branch = centered_moving_median(out.get("aggBranchPts", pd.Series(np.nan, index=out.index)), smooth_win)

    early_end = min(n_frames, max(1, int(round(float(params.get("autoStartEarlyScaleFrac", 0.35)) * n_frames))))
    bf_n = early_robust_normalize(raw_bf, early_end)
    ct_n = early_robust_normalize(raw_ct, early_end)
    skel_n = early_robust_normalize(skel, early_end)
    branch_n = early_robust_normalize(branch, early_end)
    start_signal = 0.25 * bf_n + 0.20 * ct_n + 0.35 * skel_n + 0.20 * branch_n
    start_signal = centered_moving_median(start_signal, smooth_win)
    start_values = start_signal.to_numpy(dtype=float)
    if np.isfinite(start_values).sum() >= 2:
        d_start = np.gradient(np.nan_to_num(start_values, nan=np.nanmedian(start_values[np.isfinite(start_values)])))
    else:
        d_start = np.full(n_frames, np.nan, dtype=float)
    d_start_sm = centered_moving_median(pd.Series(d_start, index=out.index), smooth_win)

    out["bfN_autoStart"] = bf_n
    out["ctN_autoStart"] = ct_n
    out["skelN_autoStart"] = skel_n
    out["branchN_autoStart"] = branch_n
    out["startSignal_autoStart"] = start_signal
    out["startSignalSlope_autoStart"] = d_start_sm
    out["isAna_manualOriginal"] = out["isAna_like_matlab"] if "isAna_like_matlab" in out.columns else np.nan

    search_end = min(n_frames, max(1, int(round(float(params.get("autoStartSearchFrac", 0.40)) * n_frames))))
    search_signal = start_signal.iloc[:search_end].to_numpy(dtype=float)
    search_slope = d_start_sm.iloc[:search_end].to_numpy(dtype=float)
    finite_slope = np.where(np.isfinite(search_slope), search_slope, -np.inf)
    peak_pos = int(np.argmax(finite_slope)) if np.isfinite(finite_slope).any() else None
    peak_slope = float(finite_slope[peak_pos]) if peak_pos is not None and np.isfinite(finite_slope[peak_pos]) else np.nan
    max_signal_search = float(np.nanmax(search_signal)) if np.isfinite(search_signal).any() else np.nan

    min_peak_slope = float(params.get("autoStartMinPeakSlope", 0.02))
    min_max_signal = float(params.get("autoStartMinMaxSignal", 0.45))
    min_developed = float(params.get("autoStartMinDevelopedSignal", 0.45))
    slope_frac = float(params.get("autoStartSlopeFrac", 0.40))
    no_start = False
    start_pos: int | None = None
    method = "auto_first_rise"
    auto_start_fallback_used = False
    auto_start_fallback_meta: dict[str, Any] = {}

    if mode == "Manual start":
        method = "manual"
        manual_start_s = float(params.get("manualStart_s", 20.0) or 0.0)
        matches = np.flatnonzero(pd.to_numeric(time_values, errors="coerce").to_numpy(dtype=float) >= manual_start_s)
        start_pos = int(matches[0]) if matches.size else None
        no_start = start_pos is None
    elif mode == "MATLAB-compatible window":
        method = "matlab_compatible_window"
        if "isAna_like_matlab" in out.columns:
            ana = out["isAna_like_matlab"].fillna(False).astype(bool).to_numpy()
        else:
            ana = np.ones(n_frames, dtype=bool)
        start_positions = np.flatnonzero(ana)
        start_pos = int(start_positions[0]) if start_positions.size else None
        no_start = start_pos is None
    elif mode == "Dominant transition (mask)":
        method = "dominant_transition"
        dom_signal_raw, dom_col = dominant_mask_signal_for_mode(out, params)
        dom_smooth_win = smooth_win
        dom_signal = centered_moving_median(pd.Series(dom_signal_raw, index=out.index), dom_smooth_win).to_numpy(dtype=float)
        dom_min_run = max(1, int(params.get("autoStartMaskMinConsecutiveFrames", 3)))
        dom_min_rel = float(params.get("dominantMinRelChange", 0.15))
        dom_pos, dom_meta = dominant_transition_start_position(
            dom_signal, search_end=search_end, min_run=dom_min_run, min_rel_change=dom_min_rel
        )
        dom_meta["dominantSignalColumn"] = dom_col
        auto_start_fallback_meta = dom_meta
        out["dominantStartSignal_autoStart"] = dom_signal
        if dom_pos is not None:
            start_pos = int(dom_pos)
            no_start = False
        else:
            no_start = True
    else:
        safety_failed = (
            not np.isfinite(peak_slope)
            or not np.isfinite(max_signal_search)
            or peak_slope < min_peak_slope
            or max_signal_search < min_max_signal
        )
        if safety_failed:
            no_start = True
        elif peak_pos is not None:
            abs_slope = np.abs(d_start_sm.to_numpy(dtype=float))
            for pos in range(peak_pos, n_frames):
                if np.isfinite(start_values[pos]) and start_values[pos] >= min_developed and np.isfinite(abs_slope[pos]) and abs_slope[pos] <= slope_frac * peak_slope:
                    start_pos = pos
                    break
            if start_pos is None:
                candidates = np.flatnonzero(np.isfinite(start_values[:search_end]) & (start_values[:search_end] >= min_developed))
                start_pos = int(candidates[0]) if candidates.size else peak_pos
        selected_method = str(params.get("rbcAnalysisMethod", ""))
        if no_start and selected_method in {"Conservative RBC Gap Mode", "Experimental POD-removed RBC Mode"}:
            fallback_pos, fallback_meta = conservative_mask_plateau_start_position(out, params, min_run=3)
            auto_start_fallback_meta = fallback_meta
            if fallback_pos is not None:
                start_pos = int(fallback_pos)
                no_start = False
                method = "auto_mask_plateau_fallback"
                auto_start_fallback_used = True

    is_ana_final = np.zeros(n_frames, dtype=bool)
    if start_pos is not None and not no_start:
        if mode == "MATLAB-compatible window" and "isAna_like_matlab" in out.columns:
            is_ana_final = out["isAna_like_matlab"].fillna(False).astype(bool).to_numpy()
        else:
            is_ana_final[start_pos:] = True

    frame_col = "originalImageIndex_1based" if "originalImageIndex_1based" in out.columns else "analysisIndex_1based"
    start_frame = frame_identity_at_position(out, start_pos, "analysisIndex_1based")
    original_start = frame_identity_at_position(out, start_pos, frame_col)
    start_time = float(time_values.iloc[start_pos]) if start_pos is not None and np.isfinite(time_values.iloc[start_pos]) and not no_start else np.nan
    start_file = str(out.iloc[start_pos].get("filename", "")) if start_pos is not None and not no_start else ""

    startup_reliable = np.zeros(n_frames, dtype=bool)
    if mode == "Auto first-rise detector" and method == "auto_first_rise":
        startup_reliable = (
            (start_signal.to_numpy(dtype=float) >= min_developed)
            & np.isfinite(start_signal.to_numpy(dtype=float))
        )
    elif mode == "Auto first-rise detector" and method == "auto_mask_plateau_fallback":
        startup_reliable = is_ana_final.copy()
    elif is_ana_final.any():
        startup_reliable = is_ana_final.copy()

    out["isAnaFinal"] = is_ana_final
    out["isStartupReliableMorphology"] = startup_reliable
    out["isReliableMorphology"] = is_ana_final
    out["autoStartMethod"] = method
    out["autoStartTime_s"] = start_time
    out["autoStartFrame_POD"] = start_frame
    out["autoStart_originalImageIndex"] = original_start
    out["autoStart_originalFileName"] = start_file
    out["peakSlope"] = peak_slope
    out["maxStartSignalInSearchWindow"] = max_signal_search
    out["noStartSafetyTriggered"] = bool(no_start and mode == "Auto first-rise detector")
    out["autoStartFallbackUsed"] = bool(auto_start_fallback_used)
    out["autoStartFallbackReason"] = auto_start_fallback_meta.get("autoStartMaskFallbackReason", "")
    for key, value in auto_start_fallback_meta.items():
        out[key] = value
    out["autoStartMinDevelopedSignal"] = min_developed
    out["autoStartSlopeFrac"] = slope_frac
    out["startupQCStableStartFrame"] = original_start
    out["firstReliableMorphologyFrame"] = original_start
    out["finalStartFrameUsed"] = original_start
    out["startupQCMode"] = mode
    out["startFrameMode"] = mode
    out["startupQC_startTime_s"] = start_time
    out["startupCriteriaPassed"] = startup_reliable.astype(int)
    out["normAggSkelLen"] = skel_n
    out["normAggBranchPts"] = branch_n
    out["normSHI"] = np.nan

    summary = {
        "startFrameMode": mode,
        "startupQCMode": mode,
        "autoStartMethod": method,
        "autoStartFrame": start_frame,
        "autoStartTime_s": start_time,
        "autoStartFrame_POD": start_frame,
        "autoStart_originalImageIndex": original_start,
        "autoStart_originalFileName": start_file,
        "peakSlope": peak_slope,
        "maxStartSignalInSearchWindow": max_signal_search,
        "noStartSafetyTriggered": bool(no_start and mode == "Auto first-rise detector"),
        "autoStartFallbackUsed": bool(auto_start_fallback_used),
        **auto_start_fallback_meta,
        "startupQCStableStartFrame": original_start,
        "firstReliableMorphologyFrame": original_start,
        "finalStartFrameUsed": original_start,
        "startupQCSelectedFrame1": bool(np.isfinite(original_start) and int(round(original_start)) == 1),
        "startupQCMovedLaterThanManualLowerBound": False,
        "startupReliableFrameCount": int(np.count_nonzero(startup_reliable)),
        "reliableMorphologyFrameCount": int(np.count_nonzero(is_ana_final)),
        "nFramesUsedFromAutoStart": int(np.count_nonzero(is_ana_final)),
        "startupStableRunDetected": bool(is_ana_final.any()),
        "startupMedianWindow": smooth_win,
        "autoStartMinDevelopedSignal": min_developed,
        "autoStartSlopeFrac": slope_frac,
        "autoStartEarlyScaleFrac": float(params.get("autoStartEarlyScaleFrac", 0.35)),
        "autoStartSearchFrac": float(params.get("autoStartSearchFrac", 0.40)),
        "autoStartSmoothWin_s": float(params.get("autoStartSmoothWin_s", 7.0)),
        "autoStartMinPeakSlope": min_peak_slope,
        "autoStartMinMaxSignal": min_max_signal,
    }
    stable_mask = is_ana_final.copy()
    stable_start_pos = start_pos if is_ana_final.any() else None
    runs = contiguous_true_runs(is_ana_final)
    return out, summary, stable_mask, stable_start_pos, runs


def add_startup_morphology_qc(metrics_df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray, int | None, list[tuple[int, int]]]:
    """Add startup morphology reliability columns and return the main-summary startup mask."""
    out = metrics_df.copy()
    n_frames = len(out)
    if n_frames == 0:
        return out, {}, np.zeros(0, dtype=bool), None, []

    startup_mode = str(params.get("startupQCMode", "Auto from first frame"))
    stable_rule = str(params.get("stableWindowRule", "3 consecutive reliable frames"))
    stable_rule_disabled = stable_rule == "disabled/manual only"
    median_window = int(params.get("startupMedianWindow", 5))
    min_run = int(params.get("minConsecutiveStartupReliableFrames", params.get("minConsecutiveReliableFramesForStableWindow", 3)))
    if stable_rule_disabled:
        min_run = 0

    smoothed = {
        "aggSkelLen": centered_moving_median(out.get("aggSkelLen", pd.Series(np.nan, index=out.index)), median_window),
        "aggBranchPts": centered_moving_median(out.get("aggBranchPts", pd.Series(np.nan, index=out.index)), median_window),
        "SHI_tileCV": centered_moving_median(out.get("SHI_tileCV", pd.Series(np.nan, index=out.index)), median_window),
        "gapAreaFrac": centered_moving_median(out.get("gapAreaFrac", pd.Series(np.nan, index=out.index)), median_window),
        "gapNtotal": centered_moving_median(out.get("gapNtotal", pd.Series(np.nan, index=out.index)), median_window),
        "gapMeanDiam_px": centered_moving_median(out.get("gapMeanDiam_px", pd.Series(np.nan, index=out.index)), median_window),
    }
    out["normAggSkelLen"] = normalize_by_sample_max(smoothed["aggSkelLen"])
    out["normAggBranchPts"] = normalize_by_sample_max(smoothed["aggBranchPts"])
    out["normSHI"] = normalize_by_sample_max(smoothed["SHI_tileCV"])

    allowed_search = np.ones(n_frames, dtype=bool)
    manual_lower_used = startup_mode in {"Auto with manual lower bound", "Fixed manual start only"}
    analysis_start_s = float(params.get("startupAnalysisStart_s", params.get("matlab_analysisStart_s", 20.0)) or 0.0)
    lower_bound_pos = first_position_at_or_after_time(out, analysis_start_s) if manual_lower_used else None
    if startup_mode == "Auto with manual lower bound":
        allowed_search = np.zeros(n_frames, dtype=bool)
        if lower_bound_pos is not None:
            allowed_search[lower_bound_pos:] = True

    c_skel = out["normAggSkelLen"] >= float(params.get("minNormAggSkelLen", 0.75))
    c_branch = out["normAggBranchPts"] >= float(params.get("minNormAggBranchPts", 0.70))
    c_shi = out["normSHI"] >= float(params.get("minNormSHI", 0.65))
    criteria_passed = c_skel.fillna(False).astype(int) + c_branch.fillna(False).astype(int) + c_shi.fillna(False).astype(int)
    out["startupCriteriaPassed"] = criteria_passed.astype(int)

    gap_area_ok = smoothed["gapAreaFrac"] >= float(params.get("minGapAreaFracReliable", 0.03))
    gap_n_ok = smoothed["gapNtotal"] >= float(params.get("minGapNtotalReliable", 20.0))
    gap_diam_ok = smoothed["gapMeanDiam_px"] >= float(params.get("minGapMeanDiamReliable_px", 2.0))
    base_startup_reliable = (
        allowed_search
        & (criteria_passed.to_numpy(dtype=int) >= int(params.get("minStartupCriteriaPassed", 2)))
        & gap_area_ok.fillna(False).to_numpy(dtype=bool)
        & gap_n_ok.fillna(False).to_numpy(dtype=bool)
        & gap_diam_ok.fillna(False).to_numpy(dtype=bool)
    )
    frame_reliable = out["rbcFrameReliable"].fillna(False).astype(bool).to_numpy() if "rbcFrameReliable" in out.columns else np.ones(n_frames, dtype=bool)
    is_reliable_morphology = base_startup_reliable & frame_reliable
    out["isStartupReliableMorphology"] = base_startup_reliable
    out["isReliableMorphology"] = is_reliable_morphology
    out["startupQCMode"] = startup_mode
    out["startupQC_startTime_s"] = analysis_start_s if manual_lower_used else np.nan

    first_reliable_pos = int(np.flatnonzero(base_startup_reliable)[0]) if np.any(base_startup_reliable) else None
    first_reliable_frame = frame_identity_at_position(out, first_reliable_pos, "originalImageIndex_1based")

    startup_stable_mask, startup_stable_pos, startup_runs = (
        stable_reliable_window_mask(pd.Series(base_startup_reliable), min_run=min_run)
        if not stable_rule_disabled and startup_mode != "Fixed manual start only"
        else (np.zeros(n_frames, dtype=bool), None, [])
    )
    startup_stable_frame = frame_identity_at_position(out, startup_stable_pos, "originalImageIndex_1based")

    final_start_pos: int | None = None
    final_start_frame = np.nan
    startup_selected_frame_one = False
    moved_later_than_lower_bound = False
    if startup_mode == "Fixed manual start only" or stable_rule_disabled:
        manual_frame = float(params.get("startupManualStartFrame_1based", 1) or 1)
        manual_frame_pos = finite_position_for_frame(out, manual_frame)
        candidate_positions = [pos for pos in [manual_frame_pos, lower_bound_pos if manual_lower_used else None] if pos is not None]
        final_start_pos = max(candidate_positions) if candidate_positions else manual_frame_pos
    elif startup_stable_pos is not None:
        final_start_pos = startup_stable_pos
        if startup_mode == "Auto with manual lower bound" and lower_bound_pos is not None:
            final_start_pos = max(final_start_pos, lower_bound_pos)
            moved_later_than_lower_bound = final_start_pos > lower_bound_pos

    if final_start_pos is not None:
        final_start_frame = frame_identity_at_position(out, final_start_pos, "originalImageIndex_1based")
        startup_selected_frame_one = bool(np.isfinite(final_start_frame) and int(round(final_start_frame)) == 1)

    if final_start_pos is None:
        final_mask = np.zeros(n_frames, dtype=bool)
    else:
        final_mask = is_reliable_morphology & (np.arange(n_frames) >= final_start_pos)

    out["startupQCStableStartFrame"] = startup_stable_frame
    out["firstReliableMorphologyFrame"] = first_reliable_frame
    out["finalStartFrameUsed"] = final_start_frame

    summary = {
        "startupQCMode": startup_mode,
        "startupMedianWindow": median_window,
        "startupManualLowerBoundUsed": bool(manual_lower_used),
        "startupAnalysisStart_s": analysis_start_s if manual_lower_used else "Not used",
        "startupQC_startTime_s": analysis_start_s if manual_lower_used else np.nan,
        "minNormAggSkelLen": float(params.get("minNormAggSkelLen", 0.75)),
        "minNormAggBranchPts": float(params.get("minNormAggBranchPts", 0.70)),
        "minNormSHI": float(params.get("minNormSHI", 0.65)),
        "minGapAreaFracReliable": float(params.get("minGapAreaFracReliable", 0.03)),
        "minGapNtotalReliable": float(params.get("minGapNtotalReliable", 20.0)),
        "minGapMeanDiamReliable_px": float(params.get("minGapMeanDiamReliable_px", 2.0)),
        "minStartupCriteriaPassed": int(params.get("minStartupCriteriaPassed", 2)),
        "minConsecutiveStartupReliableFrames": int(min_run),
        "firstReliableMorphologyFrame": first_reliable_frame,
        "startupQCStableStartFrame": startup_stable_frame,
        "finalStartFrameUsed": final_start_frame,
        "startupQCSelectedFrame1": bool(startup_selected_frame_one),
        "startupQCMovedLaterThanManualLowerBound": bool(moved_later_than_lower_bound),
        "startupReliableFrameCount": int(np.count_nonzero(base_startup_reliable)),
        "reliableMorphologyFrameCount": int(np.count_nonzero(is_reliable_morphology)),
        "startupStableRunDetected": bool(startup_stable_pos is not None),
    }
    return out, summary, final_mask, final_start_pos, startup_runs


def apply_rbc_main_summary_window(
    df: pd.DataFrame,
    params: dict[str, Any],
    warnings: list[str],
    sample_qc: str,
    stable_start_pos: int | None,
    stable_mask: np.ndarray | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mark rows that should contribute to headline RBC summaries."""
    out = df.copy()
    n_frames = len(out)
    qc_enabled = bool(params.get("enableRbcQc", True))
    stable_window_rule = str(params.get("stableWindowRule", "3 consecutive reliable frames"))
    stable_window_disabled = stable_window_rule == "disabled/manual only"
    window_mode = str(params.get("rbc_analysis_window_mode", "QC-suggested stable window" if qc_enabled else "Full sequence"))
    if window_mode not in RBC_ANALYSIS_WINDOW_OPTIONS:
        window_mode = "QC-suggested stable window" if qc_enabled else "Full sequence"

    start_frame_mode = str(params.get("startFrameMode", "Auto first-rise detector"))
    if "isAnaFinal" in out.columns and start_frame_mode in {"Auto first-rise detector", "Dominant transition (mask)", "Manual start", "MATLAB-compatible window"}:
        base_used = out["isAnaFinal"].fillna(False).astype(bool).to_numpy()
        used, excluded_by_mask_qc, mask_qc_summary = apply_post_start_mask_qc_exclusion(out, base_used, params, qc_enabled)
        stable_mask = base_used.copy()
        stable_detected = bool(base_used.any())
        reliable = used.copy()
        low_conf = out["rbcLowConfidence"].fillna(False).astype(bool).to_numpy() if "rbcLowConfidence" in out.columns else np.zeros(n_frames, dtype=bool)
        frame_qc = out["rbcFrameQC"].astype(str) if "rbcFrameQC" in out.columns else pd.Series("", index=out.index)
        weak_frames = (frame_qc == "Weak/pre-gap structural texture").to_numpy()
        artifact_frames = (frame_qc == "Possible residual/illumination artefact").to_numpy()
        isolated_reliable = np.zeros(n_frames, dtype=bool)
        outside_stable_window = (~base_used) & out.get("isStartupReliableMorphology", pd.Series(False, index=out.index)).fillna(False).astype(bool).to_numpy()
        early_low_conf = np.zeros(n_frames, dtype=bool)

        reliable_used = int(np.count_nonzero(used))
        low_conf_included = int(np.count_nonzero(used & low_conf))
        low_conf_excluded = int(np.count_nonzero((~used) & low_conf))
        weak_excluded = int(np.count_nonzero((~used) & weak_frames))
        artifact_excluded = int(np.count_nonzero((~used) & artifact_frames))
        isolated_reliable_excluded = int(np.count_nonzero(outside_stable_window))
        early_low_conf_included = 0
        if not stable_detected and qc_enabled and start_frame_mode == "Auto first-rise detector":
            warnings.append("No reliable automatic start frame detected. SASI/AggIndex are low-confidence and should not be interpreted as mature aggregation scores.")

        out["earlyLowConfidenceFrame"] = early_low_conf
        out["stableReliableWindowFrame"] = stable_mask
        out["isolatedReliableFrame"] = isolated_reliable
        out["qcOutsideStableWindow"] = outside_stable_window
        out["excludedByMaskQC"] = excluded_by_mask_qc
        out["usedForMainSummary"] = used
        out["mainSummaryWindow"] = start_frame_mode
        out["exclusionReason"] = np.where(excluded_by_mask_qc, "bad mask QC", "")

        if used.any():
            first_used_pos = int(np.flatnonzero(used)[0])
            last_used_pos = int(np.flatnonzero(used)[-1])
        else:
            first_used_pos = None
            last_used_pos = None
        summary = {
            "mainSummaryWindow": start_frame_mode,
            "lowConfidenceExclusionEnabled": False,
            "stableReliableWindowDetected": stable_detected,
            "stableWindowRule": "single automatic/manual start",
            "framesUsedForQC": n_frames,
            "framesUsedForMainSummary": int(np.count_nonzero(used)),
            "reliableFramesUsedForMainSummary": reliable_used,
            "lowConfidenceFramesIncluded": low_conf_included,
            "lowConfidenceFramesExcluded": low_conf_excluded,
            "weakPreGapFramesExcluded": weak_excluded,
            "possibleArtifactFramesExcluded": artifact_excluded,
            "isolatedReliableFramesExcluded": isolated_reliable_excluded,
            "framesOutsideStableWindow": int(np.count_nonzero(outside_stable_window)),
            "earlyLowConfidenceFramesIncluded": early_low_conf_included,
            "lowConfidenceExcluded": bool(low_conf_excluded > 0),
            "mainSummaryFirstOriginalIndex": frame_identity_at_position(out, first_used_pos, "originalImageIndex_1based"),
            "mainSummaryLastOriginalIndex": frame_identity_at_position(out, last_used_pos, "originalImageIndex_1based"),
            "mainSummaryFirstAnalysisIndex": frame_identity_at_position(out, first_used_pos, "analysisIndex_1based"),
            "mainSummaryLastAnalysisIndex": frame_identity_at_position(out, last_used_pos, "analysisIndex_1based"),
            **mask_qc_summary,
        }
        if mask_qc_summary.get("nFramesExcludedByMaskQC", 0):
            warnings.append(
                "Post-start mask QC exclusion removed "
                f"{mask_qc_summary['nFramesExcludedByMaskQC']} frame(s) from summary statistics. "
                "The frames remain in the exported per-frame table."
            )
        return out, summary

    stable_mask = np.asarray(stable_mask, dtype=bool) if stable_mask is not None else np.ones(n_frames, dtype=bool)
    if stable_mask.size != n_frames:
        stable_mask = np.zeros(n_frames, dtype=bool)
    stable_detected = bool(stable_mask.any()) if qc_enabled and not stable_window_disabled else (not qc_enabled)

    base_window = np.ones(n_frames, dtype=bool)
    if window_mode == "QC-suggested stable window" and qc_enabled:
        if stable_window_disabled:
            base_window = np.ones(n_frames, dtype=bool)
        elif stable_start_pos is None:
            base_window = np.zeros(n_frames, dtype=bool)
        else:
            base_window = np.arange(n_frames) >= stable_start_pos
    elif window_mode == "Fixed start frame":
        fixed_start = int(params.get("rbc_fixed_start_frame_1based", 1))
        if "originalImageIndex_1based" in out.columns:
            original_index = pd.to_numeric(out["originalImageIndex_1based"], errors="coerce").to_numpy(dtype=float)
            base_window = original_index >= fixed_start
        else:
            base_window = np.arange(n_frames) >= max(0, fixed_start - 1)
    elif window_mode == "MATLAB-compatible window":
        base_window = np.ones(n_frames, dtype=bool)
    else:
        base_window = np.ones(n_frames, dtype=bool)

    if base_window.size and not base_window.any() and window_mode in {"Full sequence", "Fixed start frame", "MATLAB-compatible window"}:
        base_window = np.ones(n_frames, dtype=bool)

    reliable_col = "isReliableMorphology" if "isReliableMorphology" in out.columns else "rbcFrameReliable"
    reliable = out[reliable_col].fillna(False).astype(bool).to_numpy() if reliable_col in out.columns else np.ones(n_frames, dtype=bool)
    exclude_low_conf = bool(qc_enabled and params.get("excludeLowConfidenceFromMainSummary", False))
    if qc_enabled and not stable_window_disabled and window_mode != "MATLAB-compatible window":
        used = base_window & stable_mask
    else:
        used = base_window
    if exclude_low_conf:
        used = used & reliable
    used, excluded_by_mask_qc, mask_qc_summary = apply_post_start_mask_qc_exclusion(out, used, params, qc_enabled)

    low_conf = out["rbcLowConfidence"].fillna(False).astype(bool).to_numpy() if "rbcLowConfidence" in out.columns else np.zeros(n_frames, dtype=bool)
    frame_qc = out["rbcFrameQC"].astype(str) if "rbcFrameQC" in out.columns else pd.Series("", index=out.index)
    weak_frames = (frame_qc == "Weak/pre-gap structural texture").to_numpy()
    artifact_frames = (frame_qc == "Possible residual/illumination artefact").to_numpy()
    outside_stable_window = reliable & ~stable_mask if qc_enabled and not stable_window_disabled else np.zeros(n_frames, dtype=bool)
    isolated_reliable = outside_stable_window
    early_low_conf = np.zeros(n_frames, dtype=bool)
    if stable_start_pos is not None:
        early_low_conf = (np.arange(n_frames) < stable_start_pos) & low_conf

    reliable_used = int(np.count_nonzero(used & reliable))
    low_conf_included = int(np.count_nonzero(used & low_conf))
    low_conf_excluded = int(np.count_nonzero((~used) & low_conf))
    weak_excluded = int(np.count_nonzero((~used) & weak_frames))
    artifact_excluded = int(np.count_nonzero((~used) & artifact_frames))
    isolated_reliable_excluded = int(np.count_nonzero((~used) & isolated_reliable))
    early_low_conf_included = int(np.count_nonzero(used & early_low_conf))
    if low_conf_included > 0 and qc_enabled:
        warnings.append(
            "Low-confidence frames were included in the main summary. SASI/AggIndex may reflect weak pre-gap texture rather than mature aggregation morphology."
        )
    min_required_for_summary = max(1, int(params.get("minConsecutiveReliableFramesForStableWindow", 3)))
    if not stable_window_disabled and reliable_used < min_required_for_summary and qc_enabled:
        warnings.append("Too few reliable plasma-gap frames for confident aggregation scoring.")
    if not stable_detected and qc_enabled:
        warnings.append("No stable reliable aggregation window was detected. Isolated reliable-looking frames were not used for the main summary.")

    out["earlyLowConfidenceFrame"] = early_low_conf
    out["stableReliableWindowFrame"] = stable_mask
    out["isolatedReliableFrame"] = isolated_reliable
    out["qcOutsideStableWindow"] = outside_stable_window
    out["excludedByMaskQC"] = excluded_by_mask_qc
    out["usedForMainSummary"] = used
    out["mainSummaryWindow"] = window_mode
    out["exclusionReason"] = np.where(excluded_by_mask_qc, "bad mask QC", "")

    if used.any():
        first_used_pos = int(np.flatnonzero(used)[0])
        last_used_pos = int(np.flatnonzero(used)[-1])
    else:
        first_used_pos = None
        last_used_pos = None

    summary = {
        "mainSummaryWindow": window_mode,
        "lowConfidenceExclusionEnabled": bool(exclude_low_conf),
        "stableReliableWindowDetected": stable_detected,
        "stableWindowRule": stable_window_rule,
        "framesUsedForQC": n_frames,
        "framesUsedForMainSummary": int(np.count_nonzero(used)),
        "reliableFramesUsedForMainSummary": reliable_used,
        "lowConfidenceFramesIncluded": low_conf_included,
        "lowConfidenceFramesExcluded": low_conf_excluded,
        "weakPreGapFramesExcluded": weak_excluded,
        "possibleArtifactFramesExcluded": artifact_excluded,
        "isolatedReliableFramesExcluded": isolated_reliable_excluded,
        "framesOutsideStableWindow": int(np.count_nonzero(outside_stable_window)),
        "earlyLowConfidenceFramesIncluded": early_low_conf_included,
        "lowConfidenceExcluded": bool(low_conf_excluded > 0),
        "mainSummaryFirstOriginalIndex": frame_identity_at_position(out, first_used_pos, "originalImageIndex_1based"),
        "mainSummaryLastOriginalIndex": frame_identity_at_position(out, last_used_pos, "originalImageIndex_1based"),
        "mainSummaryFirstAnalysisIndex": frame_identity_at_position(out, first_used_pos, "analysisIndex_1based"),
        "mainSummaryLastAnalysisIndex": frame_identity_at_position(out, last_used_pos, "analysisIndex_1based"),
        **mask_qc_summary,
    }
    if mask_qc_summary.get("nFramesExcludedByMaskQC", 0):
        warnings.append(
            "Post-start mask QC exclusion removed "
            f"{mask_qc_summary['nFramesExcludedByMaskQC']} frame(s) from summary statistics. "
            "The frames remain in the exported per-frame table."
        )
    return out, summary


def add_rbc_qc_classification(metrics_df: pd.DataFrame, params: dict[str, Any], warnings: list[str]) -> pd.DataFrame:
    """Add sample-level RBC QC/no-aggregation labels without filtering rows."""
    if metrics_df.empty:
        params["rbc_qc_summary"] = {}
        return metrics_df

    out = metrics_df.copy()
    if "rbcFrameReliable" not in out.columns:
        out["rbcFrameReliable"] = False
    if "rbcFrameQC" not in out.columns:
        out["rbcFrameQC"] = "No reliable plasma-gap aggregation detected"
    if "rbcLowConfidence" not in out.columns:
        out["rbcLowConfidence"] = True
    for qc_col in ["qcFailSHI_tileCV", "qcFailLowContrastSeparation", "qcFailImplausibleMask"]:
        if qc_col not in out.columns:
            out[qc_col] = False

    qc_enabled = bool(params.get("enableRbcQc", True))
    if qc_enabled:
        out, startup_summary, stable_mask, stable_start_pos, stable_runs = add_first_rise_auto_start_qc(out, params)
    else:
        startup_summary = {}
        stable_mask, stable_start_pos, stable_runs = np.ones(len(out), dtype=bool), 0, [(0, len(out) - 1)]

    n_frames = len(out)
    reliable_basis_col = "isReliableMorphology" if "isReliableMorphology" in out.columns else "rbcFrameReliable"
    reliable_count = int(out[reliable_basis_col].fillna(False).astype(bool).sum())
    weak_count = int((out["rbcFrameQC"] == "Weak/pre-gap structural texture").sum())
    artifact_count = int((out["rbcFrameQC"] == "Possible residual/illumination artefact").sum())
    no_reliable_count = int((out["rbcFrameQC"] == "No reliable plasma-gap aggregation detected").sum())
    reliable_percent = 100.0 * reliable_count / n_frames if n_frames else 0.0
    artifact_percent = 100.0 * artifact_count / n_frames if n_frames else 0.0
    min_reliable_percent = float(params.get("minReliableFramePercent", 20.0))
    min_trend = float(params.get("minTemporalTrendRho", 0.30))

    time_col = "matlabTime_s" if "matlabTime_s" in out.columns else ("analysisIndex_1based" if "analysisIndex_1based" in out.columns else None)
    trend_metrics = ["Xi_interp_px", "gapMeanDiam_px", "SASI_0_100", "AggIndex_0_100", "LF_log10_LFoverHF"]
    trend_rhos: dict[str, float] = {}
    if time_col:
        time_values = safe_numeric_series_local(out, time_col)
        for metric in trend_metrics:
            if metric in out.columns:
                trend_rhos[metric] = safe_spearman(time_values, safe_numeric_series_local(out, metric))
    finite_rhos = [value for value in trend_rhos.values() if np.isfinite(value)]
    max_positive_rho = max(finite_rhos) if finite_rhos else np.nan

    median_contrast = safe_column_median(out, "structuralContrast")
    median_sep = safe_column_median(out, "maskIntensitySeparationZ")
    median_boundary = safe_column_median(out, "boundarySupport")
    low_contrast = (
        (not np.isfinite(median_contrast) or median_contrast < float(params.get("minStructuralContrast", 0.10)))
        and (not np.isfinite(median_sep) or median_sep < float(params.get("minSeparationZ", 1.0)))
    )
    low_mask_support = (
        (not np.isfinite(median_sep) or median_sep < float(params.get("minSeparationZ", 1.0)))
        and (not np.isfinite(median_boundary) or median_boundary < float(params.get("minBoundarySupport", 0.30)))
    )
    artifact_dominant = (
        artifact_percent >= 40.0
        and artifact_count > reliable_count
        and low_mask_support
        and np.isfinite(median_contrast)
        and median_contrast >= 0.5 * float(params.get("minStructuralContrast", 0.10))
    )
    weak_trend = not np.isfinite(max_positive_rho) or max_positive_rho < min_trend

    if not qc_enabled:
        sample_qc = "QC disabled"
    elif reliable_percent < min_reliable_percent:
        sample_qc = "No reliable plasma-gap aggregation detected"
    elif artifact_dominant:
        sample_qc = "Possible residual/illumination artefact"
    elif reliable_percent < 60.0:
        sample_qc = "Weak/pre-gap structural texture"
    elif weak_trend and low_contrast:
        sample_qc = "No reliable plasma-gap aggregation detected"
    else:
        sample_qc = "Reliable aggregation morphology"
    if startup_summary.get("noStartSafetyTriggered", False) or (
        startup_summary.get("startFrameMode") == "Auto first-rise detector" and not startup_summary.get("startupStableRunDetected", False)
    ):
        sample_qc = "No reliable plasma-gap aggregation detected"
    elif startup_summary.get("startupStableRunDetected", False) and startup_summary.get("startFrameMode") in {"Auto first-rise detector", "Manual start"}:
        sample_qc = "Reliable aggregation morphology"

    stable_window_rule = str(params.get("stableWindowRule", "3 consecutive reliable frames"))
    stable_window_disabled = stable_window_rule == "disabled/manual only"
    min_stable_run = int(params.get("minConsecutiveStartupReliableFrames", params.get("minConsecutiveReliableFramesForStableWindow", 3)))
    suggested_start = np.nan if stable_window_disabled else frame_identity_at_position(out, stable_start_pos, "originalImageIndex_1based")
    suggested_analysis_start = np.nan if stable_window_disabled else frame_identity_at_position(out, stable_start_pos, "analysisIndex_1based")

    recommendation_map = {
        "No reliable plasma-gap aggregation detected": (
            "No reliable plasma-gap aggregation morphology detected. SASI/AggIndex are low-confidence and should not be interpreted as aggregation scores."
        ),
        "Weak/pre-gap structural texture": (
            "Weak/pre-gap structural texture was detected. This may represent early structural signal or residual texture. Use with caution."
        ),
        "Possible residual/illumination artefact": (
            "Detected mask structure has limited contrast or boundary support and may reflect residual or illumination artefact. "
            "Inspect overlays before interpreting RBC aggregation scores."
        ),
        "Reliable aggregation morphology": "RBC plasma-gap morphology appears sufficiently supported for normal research-use score interpretation.",
        "QC disabled": "RBC QC/no-aggregation screening was disabled; score interpretation depends on manual review of overlays and the selected analysis window.",
    }

    out["sampleQC"] = sample_qc
    out["reliableFramePercent"] = reliable_percent
    out["suggestedAnalysisStartFrame"] = suggested_start
    out["possibleLowConfidenceFrames"] = int(n_frames - reliable_count)
    out, main_window_summary = apply_rbc_main_summary_window(out, params, warnings, sample_qc, stable_start_pos, stable_mask)

    reliable_used_for_summary = int(main_window_summary.get("reliableFramesUsedForMainSummary", 0))
    low_conf_included = int(main_window_summary.get("lowConfidenceFramesIncluded", 0))
    low_conf_sample_classes = {
        "Weak/pre-gap structural texture",
        "No reliable plasma-gap aggregation detected",
        "Possible residual/illumination artefact",
    }
    if not qc_enabled:
        scores_confidence = "moderate"
    elif (not stable_window_disabled and not bool(main_window_summary.get("stableReliableWindowDetected", False))) or sample_qc in low_conf_sample_classes or reliable_used_for_summary < max(1, min_stable_run):
        scores_confidence = "low"
    elif low_conf_included > 0:
        scores_confidence = "moderate"
    else:
        scores_confidence = "high"
    scores_interpretable = scores_confidence in {"high", "moderate"} and sample_qc not in low_conf_sample_classes

    out["scoresInterpretable"] = bool(scores_interpretable)
    out["suggestedStableStartFrame"] = suggested_start
    out["suggestedStableStartAnalysisIndex"] = suggested_analysis_start
    out["framesUsedForMainSummary"] = int(main_window_summary.get("framesUsedForMainSummary", 0))
    out["reliableFramesUsedForMainSummary"] = reliable_used_for_summary
    out["lowConfidenceFramesExcluded"] = int(main_window_summary.get("lowConfidenceFramesExcluded", 0))
    out["weakPreGapFramesExcluded"] = int(main_window_summary.get("weakPreGapFramesExcluded", 0))
    out["possibleArtifactFramesExcluded"] = int(main_window_summary.get("possibleArtifactFramesExcluded", 0))
    out["isolatedReliableFramesExcluded"] = int(main_window_summary.get("isolatedReliableFramesExcluded", 0))
    out["scoresConfidence"] = scores_confidence
    out["lowConfidenceExclusionEnabled"] = bool(main_window_summary.get("lowConfidenceExclusionEnabled", False))
    out["stableReliableWindowDetected"] = bool(main_window_summary.get("stableReliableWindowDetected", False))
    out["stableWindowRule"] = stable_window_rule
    out["minConsecutiveReliableFramesForStableWindow"] = min_stable_run
    out["minSHI_tileCV_forReliable"] = float(params.get("minSHI_tileCV_forReliable", 0.15))
    for mask_qc_key in [
        "postStartBadFrameExclusionEnabled",
        "maskReliabilityMinForSummary",
        "nFramesExcludedByMaskQC",
        "nFramesFailingMaskQCInSummaryWindow",
        "percentFramesExcludedByMaskQC",
        "meanMaskReliabilityScore",
        "minMaskReliabilityScore",
        "sampleMaskQCRisk",
    ]:
        out[mask_qc_key] = main_window_summary.get(mask_qc_key, np.nan)
    out["rbcQCLabel"] = out["rbcFrameQC"]
    out["includedInMainSummary"] = out["usedForMainSummary"]
    shi_fail_count = int(out["qcFailSHI_tileCV"].fillna(False).astype(bool).sum())
    support_fail_count = int(out["qcFailLowContrastSeparation"].fillna(False).astype(bool).sum())
    implausible_fail_count = int(out["qcFailImplausibleMask"].fillna(False).astype(bool).sum())
    huge_gap_count = int(out.get("hugeGapFlag", pd.Series(False, index=out.index)).fillna(False).astype(bool).sum())
    bad_mask_count = int(out.get("badMaskFlag", pd.Series(False, index=out.index)).fillna(False).astype(bool).sum())
    unstable_seg_count = int(out.get("unstableSegmentationFlag", pd.Series(False, index=out.index)).fillna(False).astype(bool).sum())
    repair_triggered_count = int(out.get("adaptiveRepairTriggered", pd.Series(False, index=out.index)).fillna(False).astype(bool).sum())
    repair_kept_count = int(out.get("adaptiveRepairKept", pd.Series(False, index=out.index)).fillna(False).astype(bool).sum())
    out["hugeGapFrameCount"] = huge_gap_count
    out["badMaskFrameCount"] = bad_mask_count
    out["unstableSegmentationFrameCount"] = unstable_seg_count
    out["adaptiveRepairTriggeredCount"] = repair_triggered_count
    out["adaptiveRepairKeptCount"] = repair_kept_count

    qc_summary = {
        "sampleQC": sample_qc,
        "reliableFrameCount": reliable_count,
        "weakPreGapFrameCount": weak_count,
        "possibleArtifactFrameCount": artifact_count,
        "noReliableFrameCount": no_reliable_count,
        "reliableFramePercent": reliable_percent,
        "suggestedAnalysisStartFrame": suggested_start,
        "suggestedStableStartFrame": suggested_start,
        "suggestedStableStartAnalysisIndex": suggested_analysis_start,
        "stableWindowRule": stable_window_rule,
        "stableRunLengthRequired": min_stable_run,
        "minConsecutiveReliableFramesForStableWindow": min_stable_run,
        "minSHI_tileCV_forReliable": float(params.get("minSHI_tileCV_forReliable", 0.15)),
        "stableReliableWindowDetected": bool(main_window_summary.get("stableReliableWindowDetected", False)),
        "stableReliableRuns": stable_runs,
        "framesFailedSHI_tileCV": shi_fail_count,
        "framesFailedLowContrastSeparation": support_fail_count,
        "framesFailedImplausibleMask": implausible_fail_count,
        "hugeGapFrameCount": huge_gap_count,
        "badMaskFrameCount": bad_mask_count,
        "unstableSegmentationFrameCount": unstable_seg_count,
        "adaptiveRepairEnabled": bool(params.get("adaptiveAntiMergingRepair", False)),
        "adaptiveRepairTriggeredCount": repair_triggered_count,
        "adaptiveRepairKeptCount": repair_kept_count,
        "framesOutsideStableWindow": int(main_window_summary.get("framesOutsideStableWindow", 0)),
        "possibleLowConfidenceFrames": int(n_frames - reliable_count),
        "scoresInterpretable": bool(scores_interpretable),
        "scoresConfidence": scores_confidence,
        "temporalTrendRhos": trend_rhos,
        "maxPositiveTrendRho": max_positive_rho,
        "medianStructuralContrast": median_contrast,
        "medianMaskIntensitySeparationZ": median_sep,
        "medianBoundarySupport": median_boundary,
        "recommendation": recommendation_map[sample_qc],
    }
    qc_summary.update(startup_summary)
    qc_summary.update(main_window_summary)
    qc_summary["totalFramesLoaded"] = int(params.get("files_loaded_before_filtering", params.get("matlab_total_files_loaded", n_frames)))
    params["rbc_qc_summary"] = qc_summary

    if sample_qc == "No reliable plasma-gap aggregation detected":
        warnings.append(recommendation_map[sample_qc])
    elif sample_qc in {"Weak/pre-gap structural texture", "Possible residual/illumination artefact"}:
        warnings.append(recommendation_map[sample_qc])
    return out


def safe_numeric_series_local(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series with NaN/inf cleanup for local QC use."""
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def matlab_round_positive(value: float) -> int:
    """Round nonnegative MATLAB-style half-up values for frame indexing."""
    return int(np.floor(float(value) + 0.5))


def matlab_mad_flag1_like(values: np.ndarray) -> float:
    """NaN-safe MAD approximation used for MATLAB quality-window filtering."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan
    center = float(np.median(finite))
    return float(np.median(np.abs(finite - center)))


def prepare_rbc_matlab_window_entries(image_files: list, params: dict[str, Any], warnings: list[str]) -> list[dict[str, Any]]:
    """Apply MATLAB podStart/quality/isAna frame selection without changing segmentation."""
    fps = max(float(params.get("matlab_fps", 1.0) or 1.0), 1e-9)
    if not params.get("matlab_window_enabled", False):
        entries: list[dict[str, Any]] = []
        for image_file in image_files:
            original_1 = int(getattr(image_file, "original_index_1based", 1))
            matlab_time = float((original_1 - 1) / fps)
            entries.append(
                {
                    "file": image_file,
                    "metadata": {
                        "matlabTime_s": matlab_time,
                        "time_s": matlab_time,
                    },
                }
            )
        return entries

    total_files = len(image_files)
    pod_start_s = float(params.get("matlab_podStart_s", 10.0) or 0.0)
    analysis_start_s = float(params.get("matlab_analysisStart_s", 20.0) or 0.0)
    start_pos = max(0, min(total_files, matlab_round_positive(pod_start_s * fps)))
    t_all = np.arange(total_files, dtype=float) / fps

    pod_files = image_files[start_pos:]
    pod_times = t_all[start_pos:]
    d_values = np.full(len(pod_files), np.nan, dtype=float)
    quality_keep = np.ones(len(pod_files), dtype=bool)
    quality_threshold = np.nan

    if params.get("matlab_quality_filter_enabled", False) and pod_files:
        prev_frame = None
        for idx, image_file in enumerate(pod_files):
            try:
                _filename, image = read_uploaded_image(image_file)
                frame = compute_rbc_matlab_quality_frame(image, params)
                if prev_frame is not None and frame.shape == prev_frame.shape:
                    d_values[idx] = float(np.mean(np.abs(frame - prev_frame)))
                prev_frame = frame
            except Exception as exc:
                warnings.append(f"{getattr(image_file, 'name', f'image_{idx}')}: MATLAB quality D calculation failed ({exc}).")

        med_d = float(np.nanmedian(d_values)) if np.isfinite(d_values).any() else np.nan
        mad_d = matlab_mad_flag1_like(d_values)
        if np.isfinite(med_d) and np.isfinite(mad_d):
            quality_threshold = med_d + float(params.get("matlab_goodSigmaMAD", 5.0)) * mad_d
            quality_keep = d_values < quality_threshold
            quality_keep[np.isnan(d_values)] = True
        else:
            quality_keep = np.ones(len(pod_files), dtype=bool)
            warnings.append("MATLAB-like quality filter could not calculate finite D statistics; all podStart frames were kept.")

    quality_files = [file for file, keep in zip(pod_files, quality_keep) if keep]
    quality_times = np.asarray([time for time, keep in zip(pod_times, quality_keep) if keep], dtype=float)
    quality_d = np.asarray([value for value, keep in zip(d_values, quality_keep) if keep], dtype=float)
    is_ana = quality_times >= analysis_start_s

    if params.get("matlab_analyse_only_isAna", True):
        selected = [
            (file, time, d_value, bool(is_ana_value))
            for file, time, d_value, is_ana_value in zip(quality_files, quality_times, quality_d, is_ana)
            if is_ana_value
        ]
    else:
        selected = [
            (file, time, d_value, bool(is_ana_value))
            for file, time, d_value, is_ana_value in zip(quality_files, quality_times, quality_d, is_ana)
        ]

    selected_times = [item[1] for item in selected]
    params.update(
        {
            "matlab_total_files_loaded": total_files,
            "matlab_podStart_position_0based": start_pos,
            "matlab_frames_after_podStart": len(pod_files),
            "matlab_frames_after_quality_filter": len(quality_files),
            "matlab_isAna_true_count": int(np.count_nonzero(is_ana)),
            "matlab_selected_frame_count": len(selected),
            "matlab_quality_threshold": quality_threshold,
            "matlab_first_time_analysed": float(min(selected_times)) if selected_times else np.nan,
            "matlab_last_time_analysed": float(max(selected_times)) if selected_times else np.nan,
        }
    )

    if not selected:
        warnings.append("MATLAB compatibility window selected no frames for analysis.")

    entries: list[dict[str, Any]] = []
    for file, time, d_value, is_ana_value in selected:
        entries.append(
            {
                "file": file,
                "metadata": {
                    "matlabTime_s": float(time),
                    "time_s": float(time),
                    "motionD_like_matlab": float(d_value) if np.isfinite(d_value) else np.nan,
                    "isAna_like_matlab": bool(is_ana_value),
                    "matlabQualityKeep": True,
                },
            }
        )
    return entries


def conservative_pod_processing_flags(params: dict[str, Any], selected_method: str) -> dict[str, Any]:
    """Separate conservative no-POD analysis, experimental POD mode, and optional POD diagnostics."""
    remove_modes = max(0, int(params.get("removeModes", 0) or 0))
    experimental_selected = str(selected_method).startswith("Experimental")
    explicit_pod_final = bool(params.get("enablePODRemovedMode", False))
    zero_mode_branch_enabled = bool(params.get("computePODWhenRemoveModesZero", True))
    pod_branch_available = bool(remove_modes > 0 or zero_mode_branch_enabled)
    pod_final_enabled = bool(pod_branch_available and (experimental_selected or explicit_pod_final))
    pod_diagnostic_active = bool(pod_branch_available and (pod_final_enabled or bool(params.get("enablePODSensitivityDiagnostic", False))))
    pod_processing_enabled = bool(pod_final_enabled or pod_diagnostic_active)
    zero_mode_reconstruction = bool(pod_processing_enabled and remove_modes <= 0)
    if not pod_processing_enabled:
        pod_status = "POD sensitivity diagnostic disabled"
    elif zero_mode_reconstruction and pod_final_enabled:
        pod_status = "Experimental POD branch active with removeModes = 0 (MATLAB-style zero-mode reconstruction/rescale)"
    elif zero_mode_reconstruction:
        pod_status = "POD sensitivity diagnostic active with removeModes = 0 (MATLAB-style zero-mode reconstruction/rescale)"
    elif pod_final_enabled:
        pod_status = "Experimental POD-removed final mask active"
    elif pod_diagnostic_active:
        pod_status = "POD sensitivity diagnostic active"
    else:
        pod_status = "POD sensitivity diagnostic disabled"
    return {
        "removeModes": remove_modes,
        "experimentalPODSelected": bool(experimental_selected),
        "podFinalEnabled": pod_final_enabled,
        "podDiagnosticActive": pod_diagnostic_active,
        "podProcessingEnabled": pod_processing_enabled,
        "podRemovalDisabled": bool(not pod_processing_enabled),
        "zeroModePODReconstruction": zero_mode_reconstruction,
        "podStatus": pod_status,
    }


def conservative_rbc_frame_class(
    area_no_pod: float,
    area_pod: float,
    pod_delta: float,
    spatial_changed: bool,
    raw_gain_changed: bool,
    largest_frac: float,
    params: dict[str, Any],
) -> tuple[str, bool, str]:
    """Classify Conservative RBC frame reliability from no-POD/POD consistency."""
    pod_diagnostic_active = bool(params.get("podDiagnosticActive", params.get("enablePODSensitivityDiagnostic", False) or params.get("enablePODRemovedMode", False)))
    max_delta = float(params.get("maxAllowedPODAreaDelta", 0.05))
    max_largest = float(params.get("hugeGapFracThreshold", params.get("maxGapAreaFrac", 0.10)))
    min_area = float(params.get("minGapAreaFracMaskQC", 0.002))
    max_area = float(params.get("maxGapAreaFracMaskQC", 0.85))
    if np.isfinite(area_no_pod) and area_no_pod < 0.03 and (not pod_diagnostic_active or not np.isfinite(area_pod) or area_pod < 0.05):
        return "weak/pre-gap or startup-like", False, "weak/pre-gap"
    reasons: list[str] = []
    if not np.isfinite(area_no_pod) or area_no_pod < min_area or area_no_pod > max_area:
        reasons.append("area outside QC bounds")
    if pod_diagnostic_active and np.isfinite(pod_delta) and pod_delta > max_delta:
        reasons.append("POD-sensitive")
    if spatial_changed:
        reasons.append("spatial correction changed mask")
    if raw_gain_changed:
        reasons.append("raw gain correction changed mask")
    if np.isfinite(largest_frac) and largest_frac > max_largest:
        reasons.append("largest gap too high")
    if reasons:
        label = "POD-sensitive / uncertain" if any(reason == "POD-sensitive" for reason in reasons) else "conservative QC uncertain"
        return label, False, "; ".join(reasons)
    if np.isfinite(area_no_pod) and area_no_pod >= 0.03:
        return "conservative reliable morphology", True, "POD/no-POD consistent"
    return "weak/pre-gap or startup-like", False, "weak/pre-gap"


def analyze_conservative_rbc_dataset(analysis_entries: list[dict[str, Any]], params: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    """Run the Conservative/Experimental RBC stack-level POD-QC pipeline."""
    records: list[dict[str, Any]] = []
    previews: list[dict[str, Any]] = []
    loaded: list[dict[str, Any]] = []
    for idx, entry in enumerate(analysis_entries):
        image_file = entry["file"]
        fallback_name = getattr(image_file, "name", f"image_{idx}")
        try:
            filename, image = read_uploaded_image(image_file)
            raw_crop, crop_bounds = apply_crop(image, {**params, "useMatlabGrayscale": True})
            loaded.append({"entry": entry, "file": image_file, "filename": filename, "image": image, "raw_crop": raw_crop, "crop_bounds": crop_bounds})
        except Exception as exc:
            warnings.append(f"{fallback_name}: conservative RBC loading failed ({exc}).")

    if not loaded:
        return {"metrics": pd.DataFrame(), "previews": [], "warnings": warnings, "params": params, "mode": "RBC Gap Mode"}

    raw_stack = np.stack([item["raw_crop"] for item in loaded], axis=0).astype(np.float32)
    spatial_stack = np.stack([safe_spatial_illumination_correction(frame, params) for frame in raw_stack], axis=0).astype(np.float32)
    raw_curves = compute_mean_intensity_curves(raw_stack)
    spatial_curves = compute_mean_intensity_curves(spatial_stack)
    intensity_correction = evaluate_intensity_correction(raw_curves["mean"], params)
    gain = np.asarray(intensity_correction["gain"], dtype=np.float64)
    mean_gain_corrected = np.asarray(intensity_correction["corrected_mean"], dtype=np.float64)
    stack_used = spatial_stack.copy()
    if bool(intensity_correction.get("applied", False)):
        stack_used = stack_used * gain[:, None, None]
        if bool(params.get("clipGainCorrectedImages", True)):
            stack_used = np.clip(stack_used, 0.0, 1.0)
    selected_rbc_method = str(params.get("rbcAnalysisMethod", "Conservative RBC Gap Mode"))
    pod_flags = conservative_pod_processing_flags(params, selected_rbc_method)
    remove_modes = int(pod_flags["removeModes"])
    experimental_pod_main = bool(pod_flags["podFinalEnabled"])
    pod_diagnostic_active = bool(pod_flags["podDiagnosticActive"])
    pod_processing_enabled = bool(pod_flags["podProcessingEnabled"])
    pod_status = str(pod_flags["podStatus"])
    zero_mode_pod = bool(pod_flags.get("zeroModePODReconstruction", False))
    no_pod_curves = compute_mean_intensity_curves(stack_used)
    if pod_processing_enabled:
        pod_result = compute_pod_svd_and_optional_removal(stack_used, {**params, "removeModes": remove_modes})
        pod_stack = np.asarray(pod_result.get("pod_removed_stack", stack_used), dtype=np.float32)
        pod_curves = compute_mean_intensity_curves(pod_stack)
        pod_mode1_energy = float(pod_result.get("podMode1Energy", np.nan))
        first5_pod_energy = float(pod_result.get("first5PODCumulativeEnergy", np.nan))
    else:
        pod_result = {"pod_removed_stack": stack_used.copy(), "podMode1Energy": np.nan, "first5PODCumulativeEnergy": np.nan, "pod_energy": np.asarray([])}
        pod_stack = stack_used.copy()
        pod_curves = {"mean": np.full(len(loaded), np.nan), "std": np.full(len(loaded), np.nan), "cv": np.full(len(loaded), np.nan)}
        pod_mode1_energy = np.nan
        first5_pod_energy = np.nan
    max_spatial_delta = float(params.get("maxAllowedSpatialGapAreaChange", 0.05))
    max_raw_delta = float(params.get("maxAllowedRawGainGapAreaChange", 0.05))

    prev_startup_raw: np.ndarray | None = None
    for idx, item in enumerate(loaded):
        try:
            raw_frame = raw_stack[idx]
            spatial_frame = spatial_stack[idx]
            no_pod_frame = stack_used[idx]
            pod_frame = pod_stack[idx]
            no_pod_mask_params = dict(params)
            pod_offset_key = "podRemovedAdaptiveOffset" if bool(params.get("useSeparatePODAdaptiveOffset", False)) else "conservativeAdaptiveOffset"
            pod_mask_params = {
                **params,
                "conservativeAdaptiveOffset": float(
                    params.get(pod_offset_key, params.get("conservativeAdaptiveOffset", -0.043))
                ),
            }
            raw_mask = make_gap_mask_conservative(raw_frame, params)
            spatial_mask = make_gap_mask_conservative(spatial_frame, params)
            no_pod_mask = make_gap_mask_conservative(no_pod_frame, no_pod_mask_params)
            pod_mask = make_gap_mask_conservative(pod_frame, pod_mask_params) if pod_processing_enabled else np.zeros_like(no_pod_mask, dtype=bool)
            main_image = pod_frame if experimental_pod_main else no_pod_frame
            main_mask = pod_mask if experimental_pod_main else no_pod_mask
            no_pod_simple = simple_gap_metrics(no_pod_mask)
            pod_simple = simple_gap_metrics(pod_mask) if pod_processing_enabled else {
                "gapAreaFrac": np.nan,
                "gapNtotal": 0,
                "gapMeanArea": np.nan,
                "gapMeanDiam_px": np.nan,
                "gapMaxDiam_px": np.nan,
                "largestGapFrac": np.nan,
            }
            main_simple = pod_simple if experimental_pod_main else no_pod_simple
            xi_no_pod = compute_xi_from_binary_mask(
                no_pod_mask,
                minAreaFracForXi=float(params.get("minReliableGapAreaFracForXi", 0.03)),
                minObjectsForXi=int(params.get("minReliableObjectsForXi", 10)),
            )
            xi_pod = compute_xi_from_binary_mask(
                pod_mask,
                minAreaFracForXi=float(params.get("minReliableGapAreaFracForXi", 0.03)),
                minObjectsForXi=int(params.get("minReliableObjectsForXi", 10)),
            ) if pod_processing_enabled else np.nan
            raw_area = float(np.mean(raw_mask))
            spatial_area = float(np.mean(spatial_mask))
            area_no_pod = float(no_pod_simple.get("gapAreaFrac", np.nan))
            area_pod = float(pod_simple.get("gapAreaFrac", np.nan))
            try:
                standard_result = segment_rbc_gaps_seedgrow(item["image"], {**params, "rbcAnalysisMethod": "Standard RBC Gap Mode"})
                standard_area = float(np.mean(standard_result["gap_mask"]))
                standard_simple = simple_gap_metrics(standard_result["gap_mask"])
                standard_n = int(standard_simple.get("gapNtotal", 0) or 0)
            except Exception:
                standard_area = np.nan
                standard_n = 0
            pod_delta = float(abs(area_pod - area_no_pod)) if pod_processing_enabled and np.isfinite(area_pod) and np.isfinite(area_no_pod) else np.nan
            pod_ratio = float(area_pod / (area_no_pod + np.finfo(float).eps)) if pod_processing_enabled and np.isfinite(area_pod) and np.isfinite(area_no_pod) else np.nan
            spatial_changed = bool(abs(spatial_area - raw_area) > max_spatial_delta) if np.isfinite(spatial_area) and np.isfinite(raw_area) else False
            raw_gain_changed = bool(abs(area_no_pod - spatial_area) > max_raw_delta) if np.isfinite(area_no_pod) and np.isfinite(spatial_area) else False
            raw_gain_delta = float(abs(area_no_pod - spatial_area)) if np.isfinite(area_no_pod) and np.isfinite(spatial_area) else np.nan
            raw_gain_delta = float(abs(area_no_pod - spatial_area)) if np.isfinite(area_no_pod) and np.isfinite(spatial_area) else np.nan
            frame_class, mask_reliable, consistency_flag = conservative_rbc_frame_class(
                area_no_pod,
                area_pod,
                pod_delta,
                spatial_changed,
                raw_gain_changed,
                float(no_pod_simple.get("largestGapFrac", np.nan)),
                {**params, "podDiagnosticActive": pod_diagnostic_active, "enablePODRemovedMode": experimental_pod_main},
            )
            row = compute_rbc_gap_metrics(
                main_image,
                main_mask,
                filename=item["filename"],
                mode="RBC Gap Mode",
                continuous_field=main_image,
                lf_hf_method=params.get("lf_hf_method", "MATLAB-compatible continuous-field"),
                minAreaFracForXi=float(params.get("minReliableGapAreaFracForXi", 0.03)),
                minObjectsForXi=int(params.get("minReliableObjectsForXi", 10)),
                maxAreaFracForXi=float(params.get("maxReliableAreaFracForXi", 0.97)),
                rejectUniformXi=bool(params.get("rejectUniformXi", True)),
            )
            final_mask_source = "POD-removed conservative mask" if experimental_pod_main else "Conservative no-POD mask"
            final_preprocessing_source = "POD-removed conservative image" if experimental_pod_main else "Corrected no-POD conservative image"
            final_area_source = "podRemovedAreaFrac" if experimental_pod_main else "conservativeNoPODAreaFrac"
            final_area_frac = float(main_simple.get("gapAreaFrac", np.nan))
            final_n_objects = int(main_simple.get("gapNtotal", 0) or 0)
            pod_sensitive = bool(pod_diagnostic_active and np.isfinite(pod_delta) and pod_delta > float(params.get("maxAllowedPODAreaDelta", 0.05)))
            conservative_warning = bool(pod_sensitive or spatial_changed or raw_gain_changed)
            row.update(
                {
                    "areaFrac": final_area_frac,
                    "nObjects": final_n_objects,
                    "meanArea": float(main_simple.get("gapMeanArea", row.get("meanArea", np.nan))),
                    "meanDiam": float(main_simple.get("gapMeanDiam_px", row.get("meanDiam", np.nan))),
                    "largestObjectFrac": float(main_simple.get("largestGapFrac", row.get("largestObjectFrac", np.nan))),
                    "gapAreaFrac": final_area_frac,
                    "gapNtotal": final_n_objects,
                    "gapMeanDiam_px": float(main_simple.get("gapMeanDiam_px", row.get("gapMeanDiam_px", np.nan))),
                    "gapLargestFrac": float(main_simple.get("largestGapFrac", row.get("gapLargestFrac", np.nan))),
                    "mask_gapAreaFrac": final_area_frac,
                    "mask_gapNtotal": final_n_objects,
                    "mask_gapMeanArea": float(main_simple.get("gapMeanArea", row.get("mask_gapMeanArea", np.nan))),
                    "mask_gapMeanDiam_px": float(main_simple.get("gapMeanDiam_px", row.get("mask_gapMeanDiam_px", np.nan))),
                    "mask_gapMaxDiam_px": float(main_simple.get("gapMaxDiam_px", row.get("mask_gapMaxDiam_px", np.nan))),
                    "mask_largestGapFrac": float(main_simple.get("largestGapFrac", row.get("mask_largestGapFrac", np.nan))),
                    "Xi_mask": xi_pod if experimental_pod_main else xi_no_pod,
                    "Xi_mask_noPOD": xi_no_pod,
                    "Xi_mask_PODremoved": xi_pod,
                    "rbcMode": selected_rbc_method,
                    "selectedRBCMethod": selected_rbc_method,
                    "finalPreprocessingSource": final_preprocessing_source,
                    "finalMaskSource": final_mask_source,
                    "finalAreaFracSource": final_area_source,
                    "areaFracFinal": final_area_frac,
                    "nObjectsFinal": final_n_objects,
                    "useCrop": bool(params.get("useCrop", True)),
                    "conservativeGrayscaleConversion": "MATLAB rgb2gray",
                    "cropMode": item.get("crop_bounds", {}).get("cropMode", params.get("cropMode", "center")),
                    "cropFrac": params.get("cropFrac", 0.70),
                    "applySpatialIlluminationCorrection": bool(params.get("applySpatialIlluminationCorrection", True)),
                    "spatialCorrectionMethod": params.get("spatialCorrectionMethod", "divide"),
                    "spatialIllumSigma": float(params.get("spatialIllumSigma", 60.0)),
                    "spatialCorrectionStrength": float(params.get("spatialCorrectionStrength", 0.50)),
                    "preserveSpatialMeanIntensity": bool(params.get("preserveSpatialMeanIntensity", True)),
                    "conservativeAdaptiveEngine": params.get("conservativeAdaptiveEngine", "MATLAB adaptthresh-like"),
                    "conservativeAdaptSensitivity": float(params.get("conservativeAdaptSensitivity", 0.45)),
                    "conservativeAdaptMeanScaleBase": float(params.get("conservativeAdaptMeanScaleBase", 1.60)),
                    "conservativeAdaptiveOffset": float(params.get("conservativeAdaptiveOffset", -0.043)),
                    "podRemovedAdaptiveOffset": float(params.get("podRemovedAdaptiveOffset", -0.070)),
                    "useSeparatePODAdaptiveOffset": bool(params.get("useSeparatePODAdaptiveOffset", False)),
                    "computePODWhenRemoveModesZero": bool(params.get("computePODWhenRemoveModesZero", True)),
                    "applyRawGainCorrectionToImages": bool(intensity_correction.get("applied", False)),
                    "enablePODRemoval": bool(experimental_pod_main),
                    "enablePODRemovedMode": bool(experimental_pod_main),
                    "enablePODSensitivityDiagnostic": bool(pod_diagnostic_active),
                    "podDiagnosticActive": bool(pod_diagnostic_active),
                    "podProcessingEnabled": bool(pod_processing_enabled),
                    "podRemovalDisabled": bool(not pod_processing_enabled),
                    "zeroModePODReconstruction": bool(zero_mode_pod),
                    "podStatus": pod_status,
                    "selectedFinalMaskSource": final_mask_source,
                    "removeModes": remove_modes,
                    "rawMeanIntensity": float(raw_curves["mean"][idx]),
                    "rawStdIntensity": float(raw_curves["std"][idx]),
                    "rawCV": float(raw_curves["cv"][idx]),
                    "correctedMeanIntensity": float(mean_gain_corrected[idx]) if idx < len(mean_gain_corrected) else float(raw_curves["mean"][idx]),
                    "correctedStdIntensity": float(no_pod_curves["std"][idx]) if intensity_correction.get("applied", False) else float(raw_curves["std"][idx]),
                    "correctedCV": float(no_pod_curves["cv"][idx]) if intensity_correction.get("applied", False) else float(raw_curves["cv"][idx]),
                    "noPODMeanIntensity": float(no_pod_curves["mean"][idx]),
                    "noPODCorrectedMeanIntensity": float(mean_gain_corrected[idx]) if idx < len(mean_gain_corrected) else np.nan,
                    "podRemovedMeanIntensity": float(pod_curves["mean"][idx]),
                    "podRemovedCorrectedMeanIntensity": float(pod_curves["mean"][idx]),
                    "noPOD_CV": float(no_pod_curves["cv"][idx]),
                    "podRemoved_CV": float(pod_curves["cv"][idx]),
                    "conservativeNoPODAreaFrac": area_no_pod,
                    "conservativeNoPODNObjects": int(no_pod_simple.get("gapNtotal", 0) or 0),
                    "conservativeNoPODMeanArea": float(no_pod_simple.get("gapMeanArea", np.nan)),
                    "conservativeNoPODMeanDiam_px": float(no_pod_simple.get("gapMeanDiam_px", np.nan)),
                    "conservativeNoPODMaxDiam_px": float(no_pod_simple.get("gapMaxDiam_px", np.nan)),
                    "conservativeNoPODLargestGapFrac": float(no_pod_simple.get("largestGapFrac", np.nan)),
                    "conservativeGapAreaFrac": area_no_pod,
                    "conservativeAreaFrac": area_no_pod,
                    "standardAreaFrac": standard_area,
                    "standardNObjects": standard_n,
                    "podRemovedGapAreaFrac": area_pod,
                    "podRemovedAreaFrac": area_pod,
                    "podRemovedNObjects": int(pod_simple.get("gapNtotal", 0) or 0),
                    "podRemovedMeanArea": float(pod_simple.get("gapMeanArea", np.nan)),
                    "podRemovedMeanDiam_px": float(pod_simple.get("gapMeanDiam_px", np.nan)),
                    "podRemovedMaxDiam_px": float(pod_simple.get("gapMaxDiam_px", np.nan)),
                    "podRemovedLargestGapFrac": float(pod_simple.get("largestGapFrac", np.nan)),
                    "podAreaDelta": pod_delta,
                    "podAreaRatio": pod_ratio,
                    "podConsistencyFlag": consistency_flag,
                    "podSensitiveFlag": pod_sensitive,
                    "podMode1Energy": pod_mode1_energy,
                    "first5PODCumulativeEnergy": first5_pod_energy,
                    "spatialCorrectionChangedMaskTooMuch": spatial_changed,
                    "spatialCorrectionSensitiveFlag": spatial_changed,
                    "rawGainCorrectionChangedMaskTooMuch": raw_gain_changed,
                    "rawGainCorrectionSensitiveFlag": raw_gain_changed,
                    "rawGainAreaBeforeCorrection": spatial_area,
                    "rawGainAreaAfterCorrection": area_no_pod,
                    "rawGainAreaDelta": raw_gain_delta,
                    "conservativeWarningFlag": conservative_warning,
                    "conservativeFrameClass": frame_class,
                    "conservativeMaskReliable": bool(mask_reliable),
                    "rbcFrameQC": "Reliable aggregation morphology" if mask_reliable else ("Weak/pre-gap structural texture" if "weak" in frame_class else frame_class),
                    "rbcFrameReliable": bool(mask_reliable),
                    "rbcLowConfidence": not bool(mask_reliable),
                    "analysis_input_source": "conservative_noPOD" if not experimental_pod_main else "experimental_POD_removed",
                }
            )
            row.update(intensity_correction_frame_metadata(intensity_correction, idx))
            row["illuminationCV"] = float(no_pod_curves["cv"][idx])
            row.update(compute_rbc_frame_qc_metrics(main_image, main_mask))
            row.update(compute_auto_start_raw_evidence(raw_frame, sigma_bg=40.0, min_bright_area=20))
            row = add_rbc_mask_quality_diagnostics(row, params)
            if params.get("enableStartupTransientFallback", False):
                row.update(compute_startup_transient_metrics(raw_frame, prev_startup_raw, params))
            row.update(image_index_metadata(item["file"], idx))
            row.update(item["entry"].get("metadata", {}))
            records.append(row)
            boundary_width = int(params.get("boundaryLineWidth", 1))
            previews.append(
                {
                    "filename": item["filename"],
                    "source_file": item["file"],
                    "image": main_image,
                    "camera_crop_gray": raw_frame,
                    "wz": _robust_rescale01(main_image),
                    "mask": main_mask,
                    "conservative_no_pod_image": no_pod_frame,
                    "conservative_no_pod_mask": no_pod_mask,
                    "pod_removed_image": pod_frame,
                    "pod_removed_mask": pod_mask,
                    "filled_overlay": overlay_mask_on_image(main_image, main_mask),
                    "boundary_overlay": create_boundary_overlay(main_image, main_mask, color=(0, 1, 0), linewidth=boundary_width),
                    "pod_boundary_overlay": create_boundary_overlay(pod_frame, pod_mask, color=(1, 0.75, 0), linewidth=boundary_width),
                    "no_pod_boundary_overlay": create_boundary_overlay(no_pod_frame, no_pod_mask, color=(0, 1, 0), linewidth=boundary_width),
                    "filled_title": f"{final_preprocessing_source} + filled gap mask",
                    "boundary_title": f"{final_preprocessing_source} + gap boundaries",
                    "mask_label": final_mask_source,
                    "conservative_rbc_mode": True,
                    "conservative_info": {
                        "selectedRBCMethod": selected_rbc_method,
                        "finalPreprocessingSource": final_preprocessing_source,
                        "finalMaskSource": final_mask_source,
                        "finalAreaFracSource": final_area_source,
                        "conservativeGrayscaleConversion": "MATLAB rgb2gray",
                        "areaFracFinal": final_area_frac,
                        "nObjectsFinal": final_n_objects,
                        "area_noPOD": area_no_pod,
                        "conservativeNoPODAreaFrac": area_no_pod,
                        "conservativeAreaFrac": area_no_pod,
                        "standardAreaFrac": standard_area,
                        "area_PODremoved": area_pod,
                        "podRemovedAreaFrac": area_pod,
                        "podAreaDelta": pod_delta,
                        "podConsistencyFlag": consistency_flag,
                        "podMode1Energy": pod_mode1_energy,
                        "removeModes": remove_modes,
                        "conservativeAdaptiveEngine": params.get("conservativeAdaptiveEngine", "MATLAB adaptthresh-like"),
                        "conservativeAdaptSensitivity": float(params.get("conservativeAdaptSensitivity", 0.45)),
                        "conservativeAdaptMeanScaleBase": float(params.get("conservativeAdaptMeanScaleBase", 1.60)),
                        "useSeparatePODAdaptiveOffset": bool(params.get("useSeparatePODAdaptiveOffset", False)),
                        "computePODWhenRemoveModesZero": bool(params.get("computePODWhenRemoveModesZero", True)),
                        "enablePODRemoval": bool(experimental_pod_main),
                        "enablePODRemovedMode": bool(experimental_pod_main),
                        "enablePODSensitivityDiagnostic": bool(pod_diagnostic_active),
                        "podDiagnosticActive": bool(pod_diagnostic_active),
                        "podProcessingEnabled": bool(pod_processing_enabled),
                        "podRemovalDisabled": bool(not pod_processing_enabled),
                        "zeroModePODReconstruction": bool(zero_mode_pod),
                        "podStatus": pod_status,
                        "conservativeFrameClass": frame_class,
                        "conservativeMaskReliable": bool(mask_reliable),
                    },
                }
            )
            prev_startup_raw = raw_frame
        except Exception as exc:
            warnings.append(f"{item['filename']}: conservative RBC analysis failed ({exc}).")

    metrics_df = pd.DataFrame(records)
    if not metrics_df.empty and bool(metrics_df.get("rawGainCorrectionSensitiveFlag", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool).any()):
        warnings.append("Camera intensity correction changed the mask in some RBC frames. These frames are marked intensity-correction sensitive.")
    return {"metrics": metrics_df, "previews": previews, "warnings": warnings, "params": params, "mode": "RBC Gap Mode"}


def analyze_conservative_object_dataset(analysis_entries: list[dict[str, Any]], params: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    """Run Conservative/POD Object Mode using generic object terminology."""
    records: list[dict[str, Any]] = []
    previews: list[dict[str, Any]] = []
    loaded: list[dict[str, Any]] = []
    for idx, entry in enumerate(analysis_entries):
        image_file = entry["file"]
        fallback_name = getattr(image_file, "name", f"image_{idx}")
        try:
            filename, image = read_uploaded_image(image_file)
            raw_crop, crop_bounds = apply_crop(image, params)
            loaded.append({"entry": entry, "file": image_file, "filename": filename, "image": image, "raw_crop": raw_crop, "crop_bounds": crop_bounds})
        except Exception as exc:
            warnings.append(f"{fallback_name}: conservative object loading failed ({exc}).")

    if not loaded:
        return {"metrics": pd.DataFrame(), "previews": [], "warnings": warnings, "params": params, "mode": "Object Mode"}

    raw_stack = np.stack([item["raw_crop"] for item in loaded], axis=0).astype(np.float32)
    spatial_stack = np.stack([safe_spatial_illumination_correction(frame, params) for frame in raw_stack], axis=0).astype(np.float32)
    raw_curves = compute_mean_intensity_curves(raw_stack)
    spatial_curves = compute_mean_intensity_curves(spatial_stack)
    intensity_correction = evaluate_intensity_correction(raw_curves["mean"], params)
    gain = np.asarray(intensity_correction["gain"], dtype=np.float64)
    stack_used = spatial_stack.copy()
    if bool(intensity_correction.get("applied", False)):
        stack_used = stack_used * gain[:, None, None]
        if bool(params.get("clipGainCorrectedImages", True)):
            stack_used = np.clip(stack_used, 0.0, 1.0)
    selected_method = str(params.get("selectedObjectMethod", params.get("objectAnalysisMethod", "Conservative Object Mode")))
    pod_flags = conservative_pod_processing_flags(params, selected_method)
    remove_modes = int(pod_flags["removeModes"])
    experimental_pod_main = bool(pod_flags["podFinalEnabled"])
    pod_diagnostic_active = bool(pod_flags["podDiagnosticActive"])
    pod_processing_enabled = bool(pod_flags["podProcessingEnabled"])
    pod_status = str(pod_flags["podStatus"])
    zero_mode_pod = bool(pod_flags.get("zeroModePODReconstruction", False))
    no_pod_curves = compute_mean_intensity_curves(stack_used)
    if pod_processing_enabled:
        pod_result = compute_pod_svd_and_optional_removal(stack_used, {**params, "removeModes": remove_modes})
        pod_stack = np.asarray(pod_result.get("pod_removed_stack", stack_used), dtype=np.float32)
        pod_curves = compute_mean_intensity_curves(pod_stack)
        pod_mode1_energy = float(pod_result.get("podMode1Energy", np.nan))
        first5_pod_energy = float(pod_result.get("first5PODCumulativeEnergy", np.nan))
    else:
        pod_result = {"pod_removed_stack": stack_used.copy(), "podMode1Energy": np.nan, "first5PODCumulativeEnergy": np.nan, "pod_energy": np.asarray([])}
        pod_stack = stack_used.copy()
        pod_curves = {"mean": np.full(len(loaded), np.nan), "std": np.full(len(loaded), np.nan), "cv": np.full(len(loaded), np.nan)}
        pod_mode1_energy = np.nan
        first5_pod_energy = np.nan
    max_spatial_delta = float(params.get("maxAllowedSpatialObjectAreaChange", 0.05))
    max_raw_delta = float(params.get("maxAllowedRawGainObjectAreaChange", 0.05))
    min_area_for_xi = float(params.get("minReliableObjectAreaFracForXi", 0.03))
    min_objects_for_xi = int(params.get("minReliableObjectsForXi", 10))

    prev_startup_raw: np.ndarray | None = None
    for idx, item in enumerate(loaded):
        try:
            raw_frame = raw_stack[idx]
            spatial_frame = spatial_stack[idx]
            no_pod_frame = stack_used[idx]
            pod_frame = pod_stack[idx]
            raw_mask, raw_target = make_conservative_object_mask(raw_frame, params, pod_removed=False)
            spatial_mask, spatial_target = make_conservative_object_mask(spatial_frame, params, pod_removed=False)
            no_pod_mask, no_pod_target = make_conservative_object_mask(no_pod_frame, params, pod_removed=False)
            if pod_processing_enabled:
                pod_mask, pod_target = make_conservative_object_mask(pod_frame, params, pod_removed=True)
            else:
                pod_mask, pod_target = np.zeros_like(no_pod_mask, dtype=bool), no_pod_target

            main_image = pod_frame if experimental_pod_main else no_pod_frame
            main_mask = pod_mask if experimental_pod_main else no_pod_mask
            main_target = pod_target if experimental_pod_main else no_pod_target
            no_pod_metrics = object_mask_component_metrics(no_pod_mask)
            pod_metrics = object_mask_component_metrics(pod_mask) if pod_processing_enabled else {
                "objectAreaFrac": np.nan,
                "objectNtotal": 0,
                "objectMeanArea": np.nan,
                "objectMedianArea": np.nan,
                "objectMeanDiam_px": np.nan,
                "objectMedianDiam_px": np.nan,
                "objectLargestFrac": np.nan,
                "objectLargestArea": np.nan,
            }
            main_metrics = pod_metrics if experimental_pod_main else no_pod_metrics
            raw_area = float(np.mean(raw_mask))
            spatial_area = float(np.mean(spatial_mask))
            area_no_pod = float(no_pod_metrics.get("objectAreaFrac", np.nan))
            area_pod = float(pod_metrics.get("objectAreaFrac", np.nan))
            pod_delta = float(abs(area_pod - area_no_pod)) if pod_processing_enabled and np.isfinite(area_pod) and np.isfinite(area_no_pod) else np.nan
            pod_ratio = float(area_pod / (area_no_pod + np.finfo(float).eps)) if pod_processing_enabled and np.isfinite(area_pod) and np.isfinite(area_no_pod) else np.nan
            spatial_changed = bool(abs(spatial_area - raw_area) > max_spatial_delta) if np.isfinite(spatial_area) and np.isfinite(raw_area) else False
            raw_gain_delta = float(abs(area_no_pod - spatial_area)) if np.isfinite(area_no_pod) and np.isfinite(spatial_area) else np.nan
            raw_gain_changed = bool(abs(area_no_pod - spatial_area) > max_raw_delta) if np.isfinite(area_no_pod) and np.isfinite(spatial_area) else False
            xi_no_pod = compute_xi_from_binary_mask(no_pod_mask, minAreaFracForXi=min_area_for_xi, minObjectsForXi=min_objects_for_xi)
            xi_pod = compute_xi_from_binary_mask(pod_mask, minAreaFracForXi=min_area_for_xi, minObjectsForXi=min_objects_for_xi) if pod_processing_enabled else np.nan
            xi_final = xi_pod if experimental_pod_main else xi_no_pod
            xi_raw_intensity = compute_xi_interp_matlab_like(main_image)
            qc = object_frame_qc_from_masks(
                main_metrics,
                no_pod_metrics,
                pod_metrics,
                xi_final,
                pod_delta,
                spatial_changed,
                raw_gain_changed,
                {**params, "podDiagnosticActive": pod_diagnostic_active, "enablePODRemovedMode": experimental_pod_main},
            )

            row = compute_object_mode_metrics(main_image, main_mask, filename=item["filename"], mode="Object Mode", enhanced=False)
            final_mask_source = "POD-removed object mask" if experimental_pod_main else "Conservative no-POD object mask"
            final_preprocessing_source = "POD-removed corrected object image" if experimental_pod_main else "Corrected no-POD conservative object image"
            final_area_source = "podRemovedObjectAreaFrac" if experimental_pod_main else "conservativeNoPODObjectAreaFrac"
            row.update(
                {
                    "mode": "Object Mode",
                    "selectedObjectMethod": selected_method,
                    "objectAnalysisMethod": selected_method,
                    "enhancedObjectModeEnabled": False,
                    "conservativeObjectModeEnabled": True,
                    "experimentalPODObjectModeEnabled": bool(experimental_pod_main),
                    "enablePODRemoval": bool(experimental_pod_main),
                    "enablePODRemovedMode": bool(experimental_pod_main),
                    "enablePODSensitivityDiagnostic": bool(pod_diagnostic_active),
                    "podDiagnosticActive": bool(pod_diagnostic_active),
                    "podProcessingEnabled": bool(pod_processing_enabled),
                    "podRemovalDisabled": bool(not pod_processing_enabled),
                    "zeroModePODReconstruction": bool(zero_mode_pod),
                    "podStatus": pod_status,
                    "selectedFinalMaskSource": final_mask_source,
                    "removeModes": remove_modes,
                    "objectSegmentationTarget": params.get("objectSegmentationTarget", "Bright structures / bright gaps"),
                    "objectSegmentationTargetUsed": main_target,
                    "experimentalCompositeScoresEnabled": False,
                    "finalObjectMaskSource": final_mask_source,
                    "finalObjectPreprocessingSource": final_preprocessing_source,
                    "finalObjectAreaFracSource": final_area_source,
                    "areaFrac": float(main_metrics.get("objectAreaFrac", np.nan)),
                    "nObjects": int(main_metrics.get("objectNtotal", 0) or 0),
                    "objectCount": int(main_metrics.get("objectNtotal", 0) or 0),
                    "meanArea": float(main_metrics.get("objectMeanArea", np.nan)),
                    "medianArea": float(main_metrics.get("objectMedianArea", np.nan)),
                    "meanDiam": float(main_metrics.get("objectMeanDiam_px", np.nan)),
                    "medianDiam": float(main_metrics.get("objectMedianDiam_px", np.nan)),
                    "meanDiam_px": float(main_metrics.get("objectMeanDiam_px", np.nan)),
                    "medianDiam_px": float(main_metrics.get("objectMedianDiam_px", np.nan)),
                    "largestObjectFrac": float(main_metrics.get("objectLargestFrac", np.nan)),
                    "objectAreaFrac": float(main_metrics.get("objectAreaFrac", np.nan)),
                    "objectNtotal": int(main_metrics.get("objectNtotal", 0) or 0),
                    "objectMeanArea": float(main_metrics.get("objectMeanArea", np.nan)),
                    "objectMeanDiam_px": float(main_metrics.get("objectMeanDiam_px", np.nan)),
                    "objectLargestFrac": float(main_metrics.get("objectLargestFrac", np.nan)),
                    "objectAreaFracFinal": float(main_metrics.get("objectAreaFrac", np.nan)),
                    "objectNtotalFinal": int(main_metrics.get("objectNtotal", 0) or 0),
                    "objectMeanAreaFinal": float(main_metrics.get("objectMeanArea", np.nan)),
                    "objectMeanDiamFinal": float(main_metrics.get("objectMeanDiam_px", np.nan)),
                    "objectLargestFracFinal": float(main_metrics.get("objectLargestFrac", np.nan)),
                    "objectLargestAreaFinal": float(main_metrics.get("objectLargestArea", np.nan)),
                    "conservativeNoPODObjectAreaFrac": area_no_pod,
                    "conservativeNoPODObjectNtotal": int(no_pod_metrics.get("objectNtotal", 0) or 0),
                    "conservativeNoPODObjectMeanArea": float(no_pod_metrics.get("objectMeanArea", np.nan)),
                    "conservativeNoPODObjectMeanDiam_px": float(no_pod_metrics.get("objectMeanDiam_px", np.nan)),
                    "conservativeNoPODObjectLargestFrac": float(no_pod_metrics.get("objectLargestFrac", np.nan)),
                    "podRemovedObjectAreaFrac": area_pod,
                    "podRemovedObjectNtotal": int(pod_metrics.get("objectNtotal", 0) or 0),
                    "podRemovedObjectMeanArea": float(pod_metrics.get("objectMeanArea", np.nan)),
                    "podRemovedObjectMeanDiam_px": float(pod_metrics.get("objectMeanDiam_px", np.nan)),
                    "podRemovedObjectLargestFrac": float(pod_metrics.get("objectLargestFrac", np.nan)),
                    "podObjectAreaDelta": pod_delta,
                    "podObjectAreaRatio": pod_ratio,
                    "Object_Xi_raw_intensity": xi_raw_intensity,
                    "Object_Xi_mask": xi_final,
                    "Object_Xi_QC": xi_final if qc.get("objectMaskReliable", False) else np.nan,
                    "Object_Xi_mask_noPOD": xi_no_pod,
                    "Object_Xi_mask_PODremoved": xi_pod,
                    "objectXiSource": "selected final binary object mask",
                    "correlationLength_px": xi_final,
                    "Xi_px": xi_final,
                    "objectRawMeanIntensity": float(raw_curves["mean"][idx]),
                    "objectCorrectedMeanIntensity": float(np.asarray(intensity_correction.get("corrected_mean", raw_curves["mean"]), dtype=np.float64)[idx]),
                    "objectRawCV": float(raw_curves["cv"][idx]),
                    "objectCorrectedCV": float(no_pod_curves["cv"][idx]) if intensity_correction.get("applied", False) else float(raw_curves["cv"][idx]),
                    "podMode1Energy": pod_mode1_energy,
                    "first5PODCumulativeEnergy": first5_pod_energy,
                    "rawObjectAreaFrac": raw_area,
                    "spatialObjectAreaFrac": spatial_area,
                    "noPODObjectMeanIntensity": float(no_pod_curves["mean"][idx]),
                    "podRemovedObjectMeanIntensity": float(pod_curves["mean"][idx]),
                    "noPODObjectCV": float(no_pod_curves["cv"][idx]),
                    "podRemovedObjectCV": float(pod_curves["cv"][idx]),
                    "rawGainAreaBeforeCorrection": spatial_area,
                    "rawGainAreaAfterCorrection": area_no_pod,
                    "rawGainAreaDelta": raw_gain_delta,
                    "objectUseCrop": bool(params.get("useCrop", True)),
                    "objectCropMode": item.get("crop_bounds", {}).get("cropMode", params.get("cropMode", "center")),
                    "objectCropFrac": params.get("cropFrac", 0.70),
                    "preserveSpatialMeanIntensity": bool(params.get("preserveSpatialMeanIntensity", True)),
                    "conservativeObjectSegMethod": params.get("conservativeObjectSegMethod", "adaptive"),
                    "conservativeObjectMinAreaPx": int(params.get("conservativeObjectMinAreaPx", 20)),
                    "conservativeAdaptiveEngine": params.get("conservativeAdaptiveEngine", "MATLAB adaptthresh-like"),
                    "conservativeAdaptSensitivity": float(params.get("conservativeAdaptSensitivity", 0.45)),
                    "conservativeAdaptMeanScaleBase": float(params.get("conservativeAdaptMeanScaleBase", 1.60)),
                    "conservativeObjectAdaptiveOffset": float(params.get("conservativeObjectAdaptiveOffset", -0.043)),
                    "podRemovedObjectAdaptiveOffset": float(params.get("podRemovedObjectAdaptiveOffset", -0.070)),
                    "useSeparatePODAdaptiveOffset": bool(params.get("useSeparatePODAdaptiveOffset", False)),
                    "computePODWhenRemoveModesZero": bool(params.get("computePODWhenRemoveModesZero", True)),
                    "analysis_input_source": "conservative_object_noPOD" if not experimental_pod_main else "experimental_object_POD_removed",
                }
            )
            row.update(intensity_correction_frame_metadata(intensity_correction, idx))
            row["objectIlluminationGain"] = row["illuminationGain"]
            row["objectIlluminationJumpFlag"] = row["illuminationJumpFlag"]
            row.update(qc)
            row.update(image_index_metadata(item["file"], idx))
            row.update(item["entry"].get("metadata", {}))
            records.append(row)

            boundary_width = int(params.get("boundaryLineWidth", 1))
            previews.append(
                {
                    "filename": item["filename"],
                    "source_file": item["file"],
                    "image": main_image,
                    "camera_crop_gray": raw_frame,
                    "wz": _robust_rescale01(main_image),
                    "mask": main_mask,
                    "conservative_no_pod_image": no_pod_frame,
                    "conservative_no_pod_mask": no_pod_mask,
                    "pod_removed_image": pod_frame,
                    "pod_removed_mask": pod_mask,
                    "filled_overlay": overlay_mask_on_image(main_image, main_mask),
                    "boundary_overlay": create_boundary_overlay(main_image, main_mask, color=(0, 1, 0), linewidth=boundary_width),
                    "pod_boundary_overlay": create_boundary_overlay(pod_frame, pod_mask, color=(1, 0.75, 0), linewidth=boundary_width),
                    "no_pod_boundary_overlay": create_boundary_overlay(no_pod_frame, no_pod_mask, color=(0, 1, 0), linewidth=boundary_width),
                    "filled_title": f"{final_preprocessing_source} + filled object mask",
                    "boundary_title": f"{final_preprocessing_source} + object boundaries",
                    "mask_label": final_mask_source,
                    "conservative_object_mode": True,
                    "conservative_object_info": {
                        "selectedObjectMethod": selected_method,
                        "finalObjectPreprocessingSource": final_preprocessing_source,
                        "finalObjectMaskSource": final_mask_source,
                        "finalObjectAreaFracSource": final_area_source,
                        "objectAreaFracFinal": float(main_metrics.get("objectAreaFrac", np.nan)),
                        "objectNtotalFinal": int(main_metrics.get("objectNtotal", 0) or 0),
                        "conservativeNoPODObjectAreaFrac": area_no_pod,
                        "podRemovedObjectAreaFrac": area_pod,
                        "podObjectAreaDelta": pod_delta,
                        "podObjectConsistencyFlag": qc.get("podObjectConsistencyFlag", "n/a"),
                        "removeModes": remove_modes,
                        "conservativeAdaptiveEngine": params.get("conservativeAdaptiveEngine", "MATLAB adaptthresh-like"),
                        "conservativeAdaptSensitivity": float(params.get("conservativeAdaptSensitivity", 0.45)),
                        "conservativeAdaptMeanScaleBase": float(params.get("conservativeAdaptMeanScaleBase", 1.60)),
                        "useSeparatePODAdaptiveOffset": bool(params.get("useSeparatePODAdaptiveOffset", False)),
                        "computePODWhenRemoveModesZero": bool(params.get("computePODWhenRemoveModesZero", True)),
                        "enablePODRemoval": bool(experimental_pod_main),
                        "enablePODRemovedMode": bool(experimental_pod_main),
                        "enablePODSensitivityDiagnostic": bool(pod_diagnostic_active),
                        "podDiagnosticActive": bool(pod_diagnostic_active),
                        "podProcessingEnabled": bool(pod_processing_enabled),
                        "podRemovalDisabled": bool(not pod_processing_enabled),
                        "zeroModePODReconstruction": bool(zero_mode_pod),
                        "podStatus": pod_status,
                        "Object_Xi_mask": xi_final,
                        "Object_Xi_QC": xi_final if qc.get("objectMaskReliable", False) else np.nan,
                        "objectFrameClass": qc.get("objectFrameClass", "n/a"),
                        "objectMaskReliable": qc.get("objectMaskReliable", False),
                        "objectSegmentationTargetUsed": main_target,
                    },
                }
            )
            prev_startup_raw = raw_frame
        except Exception as exc:
            warnings.append(f"{item['filename']}: conservative object analysis failed ({exc}).")

    metrics_df = pd.DataFrame(records)
    if not metrics_df.empty:
        if bool(metrics_df.get("rawGainCorrectionSensitiveFlag", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool).any()):
            warnings.append("Camera intensity correction changed the generic object mask in some frames. These frames are marked intensity-correction sensitive.")
        metrics_df = add_conservative_object_score_columns(metrics_df, params)
    return {"metrics": metrics_df, "previews": previews, "warnings": warnings, "params": params, "mode": "Object Mode"}


def prepare_standard_rbc_optional_corrections(
    analysis_entries: list[dict[str, Any]],
    params: dict[str, Any],
    warnings: list[str],
) -> dict[int, dict[str, Any]]:
    """Prepare optional Standard RBC raw-gain/spatial corrections without changing defaults."""
    if not (
        params.get("standard_enableRawGainCorrection", False)
        or params.get("standard_enableSpatialFlatfieldCorrection", False)
        or params.get("standard_enableMeanIntensityCorrectionPlot", True)
    ):
        return {}
    loaded: list[dict[str, Any]] = []
    for idx, entry in enumerate(analysis_entries):
        try:
            filename, image = read_uploaded_image(entry["file"])
            raw = to_grayscale_float01(image)
            loaded.append({"entry": entry, "filename": filename, "image": image, "raw": raw})
        except Exception as exc:
            warnings.append(f"{getattr(entry['file'], 'name', f'image_{idx}')}: could not prepare Standard RBC correction diagnostics ({exc}).")
    if not loaded:
        return {}
    raw_stack = np.stack([item["raw"] for item in loaded], axis=0)
    raw_curves = compute_mean_intensity_curves(raw_stack)
    intensity_correction = evaluate_intensity_correction(raw_curves["mean"], params)
    gain = np.asarray(intensity_correction["gain"], dtype=np.float64)
    out: dict[int, dict[str, Any]] = {}
    for idx, item in enumerate(loaded):
        corrected = item["raw"].copy()
        if params.get("standard_enableSpatialFlatfieldCorrection", False):
            corrected = safe_spatial_illumination_correction(
                corrected,
                {
                    "applySpatialIlluminationCorrection": True,
                    "spatialCorrectionMethod": params.get("spatialCorrectionMethod", "divide"),
                    "spatialIllumSigma": params.get("spatialIllumSigma", 60.0),
                    "spatialCorrectionStrength": params.get("spatialCorrectionStrength", 0.50),
                    "clipSpatialCorrectedImages": True,
                },
            )
        if intensity_correction.get("applied", False):
            corrected = corrected * gain[idx]
            corrected = np.clip(corrected, 0.0, 1.0)
        corrected_std = float(np.std(corrected))
        corrected_mean = float(np.mean(corrected))
        correction_metadata = intensity_correction_frame_metadata(intensity_correction, idx)
        payload = {
            "analysis_image": corrected if (intensity_correction.get("applied", False) or params.get("standard_enableSpatialFlatfieldCorrection", False)) else item["image"],
            "rawMeanIntensity": float(raw_curves["mean"][idx]),
            "rawStdIntensity": float(raw_curves["std"][idx]),
            "rawCV": float(raw_curves["cv"][idx]),
            "correctedMeanIntensity": corrected_mean if intensity_correction.get("applied", False) else float(raw_curves["mean"][idx]),
            "correctedStdIntensity": corrected_std if intensity_correction.get("applied", False) else float(raw_curves["std"][idx]),
            "correctedCV": float(corrected_std / (abs(corrected_mean) + np.finfo(float).eps)) if intensity_correction.get("applied", False) else float(raw_curves["cv"][idx]),
            "standard_enableRawGainCorrection": bool(intensity_correction.get("applied", False)),
            "standard_enableSpatialFlatfieldCorrection": bool(params.get("standard_enableSpatialFlatfieldCorrection", False)),
            "standard_enableMeanIntensityCorrectionPlot": bool(params.get("standard_enableMeanIntensityCorrectionPlot", True)),
        }
        payload.update(correction_metadata)
        out[id(item["entry"]["file"])] = payload
    return out


def prepare_standard_object_optional_corrections(
    analysis_entries: list[dict[str, Any]],
    params: dict[str, Any],
    warnings: list[str],
) -> dict[int, dict[str, Any]]:
    """Prepare Off/On/Auto camera intensity correction for Standard/Enhanced Object modes."""
    loaded: list[dict[str, Any]] = []
    for idx, entry in enumerate(analysis_entries):
        try:
            filename, image = read_uploaded_image(entry["file"])
            raw = to_grayscale_float01(image)
            loaded.append({"entry": entry, "filename": filename, "image": image, "raw": raw})
        except Exception as exc:
            warnings.append(f"{getattr(entry['file'], 'name', f'image_{idx}')}: could not prepare Object Mode intensity diagnostics ({exc}).")
    if not loaded:
        return {}

    raw_stack = np.stack([item["raw"] for item in loaded], axis=0)
    raw_curves = compute_mean_intensity_curves(raw_stack)
    intensity_correction = evaluate_intensity_correction(raw_curves["mean"], params)
    gain = np.asarray(intensity_correction["gain"], dtype=np.float64)
    out: dict[int, dict[str, Any]] = {}
    for idx, item in enumerate(loaded):
        corrected = item["raw"].copy()
        if intensity_correction.get("applied", False):
            corrected = corrected * gain[idx]
            if bool(params.get("clipGainCorrectedImages", True)):
                corrected = np.clip(corrected, 0.0, 1.0)
        corrected_std = float(np.std(corrected))
        corrected_mean = float(np.mean(corrected))
        metadata = {
            "analysis_image": corrected if intensity_correction.get("applied", False) else item["image"],
            "rawMeanIntensity": float(raw_curves["mean"][idx]),
            "rawStdIntensity": float(raw_curves["std"][idx]),
            "rawCV": float(raw_curves["cv"][idx]),
            "correctedMeanIntensity": corrected_mean if intensity_correction.get("applied", False) else float(raw_curves["mean"][idx]),
            "correctedStdIntensity": corrected_std if intensity_correction.get("applied", False) else float(raw_curves["std"][idx]),
            "correctedCV": float(corrected_std / (abs(corrected_mean) + np.finfo(float).eps)) if intensity_correction.get("applied", False) else float(raw_curves["cv"][idx]),
            "objectRawMeanIntensity": float(raw_curves["mean"][idx]),
            "objectRawCV": float(raw_curves["cv"][idx]),
            "objectCorrectedMeanIntensity": corrected_mean if intensity_correction.get("applied", False) else float(raw_curves["mean"][idx]),
            "objectCorrectedCV": float(corrected_std / (abs(corrected_mean) + np.finfo(float).eps)) if intensity_correction.get("applied", False) else float(raw_curves["cv"][idx]),
        }
        metadata.update(intensity_correction_frame_metadata(intensity_correction, idx))
        metadata["objectIlluminationGain"] = metadata["illuminationGain"]
        metadata["objectIlluminationJumpFlag"] = metadata["illuminationJumpFlag"]
        out[id(item["entry"]["file"])] = metadata
    return out


def _analyze_standard_dataset_v3(image_files: list, mode: str, params: dict[str, Any]) -> dict[str, Any]:
    """Analyze RBC Gap Mode or Object Mode image uploads."""
    records: list[dict[str, Any]] = []
    previews: list[dict[str, Any]] = []
    warnings: list[str] = list(params.get("pre_analysis_warnings", []))
    prev_startup_raw: np.ndarray | None = None

    if mode == "RBC Gap Mode":
        analysis_entries = prepare_rbc_matlab_window_entries(image_files, params, warnings)
        if params.get("rbcAnalysisMethod", "Standard RBC Gap Mode") != "Standard RBC Gap Mode":
            analysis = analyze_conservative_rbc_dataset(analysis_entries, params, warnings)
            metrics_df = analysis["metrics"]
            if not metrics_df.empty:
                metrics_df = add_large_gap_artefact_summary(metrics_df, params, warnings)
                metrics_df = compute_rbc_aggregation_scores(metrics_df)
                metrics_df = add_rbc_qc_classification(metrics_df, params, warnings)
                metrics_df, _startup_fallback_summary = add_startup_transient_fallback_flags(metrics_df, params, mode, warnings)
                metrics_df = add_rbc_mask_based_score_columns(metrics_df, params)
                metrics_df = add_rbc_qc_aware_score_columns(metrics_df, params)
                metrics_df, sasi_syllectogram_summary = compute_sasi_syllectogram_research_output(
                    metrics_df,
                    enabled=bool(params.get("enableSASISyllectogramResearchOutput", False)),
                    start_threshold=float(params.get("sasiSyllectogramStartThreshold", 20.0)),
                    smooth_window=int(params.get("sasiSyllectogramSmoothWindow", 5)),
                )
                params["sasi_syllectogram_research_summary"] = sasi_syllectogram_summary
                if sasi_syllectogram_summary.get("SASI_syllectogram_warning"):
                    warnings.append(str(sasi_syllectogram_summary["SASI_syllectogram_warning"]))
                warn_sparse_mask_morphology_filter(metrics_df, params, warnings)
                metrics_df = apply_normalized_syllectogram_outputs(metrics_df, params)
                analysis["metrics"] = metrics_df
                analysis["warnings"] = warnings
                analysis["params"] = params
            add_standard_quality_warnings(analysis["metrics"], mode, warnings)
            return analysis
        standard_corrections = prepare_standard_rbc_optional_corrections(analysis_entries, params, warnings)
    else:
        analysis_entries = [{"file": image_file, "metadata": {}} for image_file in image_files]
        standard_corrections = {}
        object_corrections = {}
        if params.get("selectedObjectMethod", "Standard Object Mode") in {"Conservative Object Mode", "Experimental POD-removed Object Mode"}:
            analysis = analyze_conservative_object_dataset(analysis_entries, params, warnings)
            metrics_df = analysis["metrics"]
            if not metrics_df.empty:
                metrics_df, _startup_fallback_summary = add_startup_transient_fallback_flags(metrics_df, params, mode, warnings)
                metrics_df = apply_normalized_syllectogram_outputs(metrics_df, params)
                add_object_mode_quality_warnings(metrics_df, params, warnings)
                analysis["metrics"] = metrics_df
                analysis["warnings"] = warnings
                analysis["params"] = params
            add_standard_quality_warnings(analysis["metrics"], mode, warnings)
            return analysis
        object_corrections = prepare_standard_object_optional_corrections(analysis_entries, params, warnings)

    for idx, entry in enumerate(analysis_entries):
        image_file = entry["file"]
        fallback_name = getattr(image_file, "name", f"image_{idx}")
        try:
            filename, image = read_uploaded_image(image_file)
            if mode == "RBC Gap Mode":
                correction_payload = standard_corrections.get(id(image_file), {})
                analysis_image = correction_payload.get("analysis_image", image)
                rbc_result, row = analyze_rbc_frame_with_optional_repair(analysis_image, filename, params)
                selected_rbc_method = params.get("rbcAnalysisMethod", "Standard RBC Gap Mode")
                row["rbcMode"] = selected_rbc_method
                row["selectedRBCMethod"] = selected_rbc_method
                row["finalPreprocessingSource"] = str(
                    rbc_result.get("segmentation_input_source", params.get("rbcSegmentationInputSource", "Standard RBC seed-grow preprocessing"))
                )
                row["finalMaskSource"] = "Standard RBC seed-grow mask"
                row["finalAreaFracSource"] = "areaFrac"
                row["areaFracFinal"] = float(row.get("areaFrac", np.nan))
                row["nObjectsFinal"] = int(row.get("nObjects", 0) or 0)
                row["standardAreaFrac"] = float(row.get("areaFrac", np.nan))
                row["standardNObjects"] = int(row.get("nObjects", 0) or 0)
                row["conservativeNoPODAreaFrac"] = np.nan
                row["conservativeAreaFrac"] = np.nan
                row["podRemovedAreaFrac"] = np.nan
                row.update({key: value for key, value in correction_payload.items() if key != "analysis_image"})
                mask = rbc_result["gap_mask"]
                if correction_payload:
                    try:
                        raw_standard_result = segment_rbc_gaps_seedgrow(image, {**params, "rbcAnalysisMethod": "Standard RBC Gap Mode"})
                        raw_area_before = float(np.mean(raw_standard_result["gap_mask"]))
                    except Exception:
                        raw_area_before = np.nan
                    raw_area_after = float(np.mean(mask)) if mask is not None else np.nan
                    raw_gain_delta = float(abs(raw_area_after - raw_area_before)) if np.isfinite(raw_area_after) and np.isfinite(raw_area_before) else np.nan
                    raw_gain_sensitive = bool(
                        correction_payload.get("intensityCorrectionApplied", False)
                        and np.isfinite(raw_gain_delta)
                        and raw_gain_delta > float(params.get("maxAllowedRawGainGapAreaChange", 0.05))
                    )
                    row.update(
                        {
                            "rawGainAreaBeforeCorrection": raw_area_before,
                            "rawGainAreaAfterCorrection": raw_area_after,
                            "rawGainAreaDelta": raw_gain_delta,
                            "rawGainCorrectionChangedMaskTooMuch": raw_gain_sensitive,
                            "rawGainCorrectionSensitiveFlag": raw_gain_sensitive,
                        }
                    )
                startup_raw = rbc_result.get("camera_crop_gray", rbc_result.get("crop_gray", image))
                background, background_title = rbc_overlay_background(rbc_result, params.get("overlay_background", "Illumination-corrected Wz"))
                boundary_width = int(params.get("boundaryLineWidth", 1))
                preview = {
                    "filename": filename,
                    "source_file": image_file,
                    "image": rbc_result["crop_gray"],
                    "camera_crop_gray": rbc_result.get("camera_crop_gray", rbc_result["crop_gray"]),
                    "wz": rbc_result["display_Wz"],
                    "mask": mask,
                    "removed_gap_mask": rbc_result.get("removed_gap_mask", np.zeros_like(mask, dtype=bool)),
                    "large_gap_diagnostics": rbc_result.get("large_gap_diagnostics", {}),
                    "debug_report": rbc_result.get("debug_report", {}),
                    "filled_overlay": overlay_mask_on_image(background, mask),
                    "boundary_overlay": create_boundary_overlay(background, mask, color=(0, 1, 0), linewidth=boundary_width),
                    "filled_title": f"{background_title} + filled gap mask",
                    "boundary_title": f"{background_title} + gap boundaries",
                    "mask_label": "Plasma gap mask",
                }
            else:
                correction_payload = object_corrections.get(id(image_file), {})
                object_analysis_image = correction_payload.get("analysis_image", image)
                if params.get("enhancedObjectModeEnabled", False):
                    object_result = segment_object_with_details(object_analysis_image, params)
                    mask = object_result["structure_mask"]
                    startup_raw = object_result.get("raw_gray", object_analysis_image)
                    object_metrics_image = (
                        object_result.get("raw_gray", object_analysis_image)
                        if params.get("enhancedObjectPreprocessingStyle") == "MATLAB/RBC-style robust residual mode"
                        else object_analysis_image
                    )
                    row = compute_object_mode_metrics(object_metrics_image, mask, filename=filename, mode=mode, enhanced=True)
                    row.update(object_result.get("structure_diagnostics", {}))
                    row["experimentalCompositeScoresEnabled"] = bool(params.get("experimentalCompositeScoresEnabled", False))
                    object_overlay_background = object_result.get("display_Wz", object_result.get("raw_gray", object_analysis_image))
                    preview = {
                        "filename": filename,
                        "source_file": image_file,
                        "image": object_result.get("raw_gray", object_analysis_image),
                        "mask": mask,
                        "filled_overlay": overlay_mask_on_image(object_overlay_background, mask),
                        "boundary_overlay": create_boundary_overlay(object_overlay_background, mask, color=(0, 255, 0), linewidth=1),
                        "mask_label": "Generic structure mask",
                        "enhanced_object_mode": True,
                        "object_target": row.get("objectSegmentationTargetUsed", row.get("objectSegmentationTarget", "generic structures")),
                        "enhanced_preprocessing_style": row.get("enhancedObjectPreprocessingStyle", params.get("enhancedObjectPreprocessingStyle", "Current generic enhanced mode")),
                        "object_analysis_input_source": row.get("objectAnalysisInputSource", "generic enhanced residual"),
                        "removed_structure_mask": object_result.get("removed_structure_mask", np.zeros_like(mask, dtype=bool)),
                        "structure_diagnostics": object_result.get("structure_diagnostics", {}),
                    }
                    preview["wz"] = object_result["display_Wz"]
                else:
                    mask = segment_object_like(object_analysis_image, params)
                    startup_raw = object_analysis_image
                    if params.get("experimentalCompositeScoresEnabled", False) or params.get("batch_generic_metrics_enabled", False):
                        row = compute_object_mode_metrics(object_analysis_image, mask, filename=filename, mode=mode, enhanced=False)
                    else:
                        row = compute_core_metrics(object_analysis_image, mask, filename=filename, mode=mode)
                    preview = {
                        "filename": filename,
                        "image": object_analysis_image,
                        "mask": mask,
                        "filled_overlay": overlay_mask_on_image(object_analysis_image, mask),
                        "boundary_overlay": create_boundary_overlay(object_analysis_image, mask, color=(0, 255, 0), linewidth=1),
                        "mask_label": "Object mask",
                    }
                if correction_payload:
                    if params.get("enhancedObjectModeEnabled", False):
                        raw_mask_for_gain = segment_object_with_details(image, params)["structure_mask"]
                    else:
                        raw_mask_for_gain = segment_object_like(image, params)
                    raw_gain_area_before = float(np.mean(raw_mask_for_gain))
                    raw_gain_area_after = float(np.mean(mask))
                    raw_gain_delta = float(abs(raw_gain_area_after - raw_gain_area_before))
                    sensitive = bool(
                        correction_payload.get("intensityCorrectionApplied", False)
                        and raw_gain_delta > float(params.get("maxAllowedRawGainObjectAreaChange", 0.05))
                    )
                    row.update({key: value for key, value in correction_payload.items() if key != "analysis_image"})
                    row.update(
                        {
                            "rawGainAreaBeforeCorrection": raw_gain_area_before,
                            "rawGainAreaAfterCorrection": raw_gain_area_after,
                            "rawGainAreaDelta": raw_gain_delta,
                            "rawGainCorrectionChangedMaskTooMuch": sensitive,
                            "rawGainCorrectionSensitiveFlag": sensitive,
                        }
                    )
            if params.get("enableStartupTransientFallback", False):
                row.update(compute_startup_transient_metrics(startup_raw, prev_startup_raw, params))
            prev_startup_raw = np.asarray(startup_raw)
            row.update(image_index_metadata(image_file, idx))
            row.update(entry.get("metadata", {}))
            records.append(row)
            previews.append(preview)
        except Exception as exc:
            warnings.append(f"{fallback_name}: analysis failed ({exc}).")

    metrics_df = pd.DataFrame(records)
    if mode == "RBC Gap Mode" and not metrics_df.empty:
        if bool(metrics_df.get("rawGainCorrectionSensitiveFlag", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool).any()):
            warnings.append("Camera intensity correction changed the mask in some RBC frames. These frames are marked intensity-correction sensitive.")
        metrics_df = add_large_gap_artefact_summary(metrics_df, params, warnings)
        metrics_df = compute_rbc_aggregation_scores(metrics_df)
        metrics_df = add_rbc_qc_classification(metrics_df, params, warnings)
        metrics_df, _startup_fallback_summary = add_startup_transient_fallback_flags(metrics_df, params, mode, warnings)
        metrics_df = add_rbc_mask_based_score_columns(metrics_df, params)
        metrics_df = add_rbc_qc_aware_score_columns(metrics_df, params)
        metrics_df, sasi_syllectogram_summary = compute_sasi_syllectogram_research_output(
            metrics_df,
            enabled=bool(params.get("enableSASISyllectogramResearchOutput", False)),
            start_threshold=float(params.get("sasiSyllectogramStartThreshold", 20.0)),
            smooth_window=int(params.get("sasiSyllectogramSmoothWindow", 5)),
        )
        params["sasi_syllectogram_research_summary"] = sasi_syllectogram_summary
        if sasi_syllectogram_summary.get("SASI_syllectogram_warning"):
            warnings.append(str(sasi_syllectogram_summary["SASI_syllectogram_warning"]))
        warn_sparse_mask_morphology_filter(metrics_df, params, warnings)
        metrics_df = apply_normalized_syllectogram_outputs(metrics_df, params)
    elif (
        mode == "Object Mode"
        and (
            params.get("enhancedObjectModeEnabled", False)
            or params.get("experimentalCompositeScoresEnabled", False)
            or params.get("batch_generic_metrics_enabled", False)
        )
        and not metrics_df.empty
    ):
        metrics_df = compute_generic_structure_scores(metrics_df, enabled=bool(params.get("experimentalCompositeScoresEnabled", False)))
        metrics_df = add_conservative_object_score_columns(metrics_df, params)
        metrics_df, _startup_fallback_summary = add_startup_transient_fallback_flags(metrics_df, params, mode, warnings)
        metrics_df = apply_normalized_syllectogram_outputs(metrics_df, params)
        add_object_mode_quality_warnings(metrics_df, params, warnings)
        if bool(metrics_df.get("rawGainCorrectionSensitiveFlag", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool).any()):
            warnings.append("Camera intensity correction changed the generic object mask in some frames. These frames are marked intensity-correction sensitive.")
    elif mode == "Object Mode" and not metrics_df.empty:
        metrics_df = add_conservative_object_score_columns(metrics_df, params)
        metrics_df, _startup_fallback_summary = add_startup_transient_fallback_flags(metrics_df, params, mode, warnings)
        metrics_df = apply_normalized_syllectogram_outputs(metrics_df, params)
        if bool(metrics_df.get("rawGainCorrectionSensitiveFlag", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool).any()):
            warnings.append("Camera intensity correction changed the generic object mask in some frames. These frames are marked intensity-correction sensitive.")
    add_standard_quality_warnings(metrics_df, mode, warnings)
    return {"metrics": metrics_df, "previews": previews, "warnings": warnings, "params": params, "mode": mode}


def auto_segment_for_gt(image: np.ndarray, params: dict[str, Any], gt_shape: tuple[int, int]) -> np.ndarray:
    """Run selected automatic segmentation for GT comparison and resize to GT shape."""
    if params.get("auto_method") == "RBC Gap Mode segmentation":
        auto_mask = segment_rbc_gaps_seedgrow(image, {"enableMaxGapAreaFilter": False, **params})["gap_mask"]
    else:
        auto_mask = segment_object_like(image, params)
    return resize_mask_to_shape(auto_mask, gt_shape)


def analyze_ground_truth_dataset(image_files: list, mask_files: list, params: dict[str, Any]) -> dict[str, Any]:
    """Analyze paired images and ground-truth masks/outlines."""
    records: list[dict[str, Any]] = []
    previews: list[dict[str, Any]] = []
    pairs, match_warnings = match_image_mask_files(image_files, mask_files, params.get("matching_strategy", "match by sorted order"))
    warnings: list[str] = list(params.get("pre_analysis_warnings", []))
    warnings.extend(match_warnings)

    if not mask_files:
        warnings.append("Ground-Truth Mask Mode needs mask or outline files.")

    for idx, (image_file, mask_file) in enumerate(pairs):
        image_name = getattr(image_file, "name", f"image_{idx}")
        try:
            filename, image = read_uploaded_image(image_file)
            mask_filename, mask_image = read_uploaded_image(mask_file)
            gt_input_mask, gt_filled_mask = load_ground_truth_mask(
                mask_image,
                mask_type=params.get("mask_type", "Outline mask"),
                outline_polarity=params.get("outline_polarity", "dark outlines"),
                min_object_size=int(params.get("gt_min_object_size", 20)),
            )
            gt_filled_mask = resize_mask_to_shape(gt_filled_mask, image.shape[:2])
            gt_input_mask = resize_mask_to_shape(gt_input_mask, image.shape[:2])

            gt_core = compute_core_metrics(image, gt_filled_mask, filename=filename, mode="Ground-Truth Mask Mode")
            row: dict[str, Any] = {
                "filename": filename,
                "maskFile": mask_filename,
                "mode": "Ground-Truth Mask Mode",
            }
            row.update(image_index_metadata(image_file, idx))
            row.update(prefix_metrics(gt_core, "GT", ["areaFrac", "nObjects", "meanArea", "medianArea", "meanDiam", "medianDiam", "largestObjectFrac", "Xi_px", "lacunarity16", "skelDensity", "branchDensity", "endpointDensity"]))

            preview: dict[str, Any] = {
                "filename": filename,
                "image": image,
                "gt_input_mask": gt_input_mask,
                "gt_filled_mask": gt_filled_mask,
                "mask_label": "Ground-truth filled mask",
            }

            if params.get("compare_auto", False):
                auto_mask = auto_segment_for_gt(image, params, gt_filled_mask.shape)
                auto_core = compute_core_metrics(image, auto_mask, filename=filename, mode="Automatic segmentation")
                row.update(prefix_metrics(auto_core, "AUTO", ["areaFrac", "nObjects", "meanArea", "meanDiam"]))
                row["Dice_auto_vs_GT"] = dice_coefficient(auto_mask, gt_filled_mask)
                row["IoU_auto_vs_GT"] = iou_coefficient(auto_mask, gt_filled_mask)
                preview["auto_mask"] = auto_mask
                preview["auto_overlay"] = overlay_mask_on_image(image, auto_mask, color_rgb=(0.0, 0.55, 1.0))
                preview["dice"] = row["Dice_auto_vs_GT"]
                preview["iou"] = row["IoU_auto_vs_GT"]

            records.append(row)
            previews.append(preview)
        except Exception as exc:
            warnings.append(f"{image_name}: ground-truth analysis failed ({exc}).")

    return {"metrics": pd.DataFrame(records), "previews": previews, "warnings": warnings, "params": params, "mode": "Ground-Truth Mask Mode"}


def add_analysis_identity(df: pd.DataFrame, mode: str, params: dict[str, Any]) -> pd.DataFrame:
    """Attach reproducibility metadata to every exported V3 analysis row."""
    out = df.copy()
    if out.empty:
        return out
    settings_json = json.dumps(params, sort_keys=True, default=str, ensure_ascii=True)
    settings_hash = hashlib.sha256(settings_json.encode("utf-8")).hexdigest()
    out["microstructAIVersion"] = APP_VERSION
    out["analysisSchemaVersion"] = ANALYSIS_SCHEMA_VERSION
    out["analysisModeIdentity"] = mode
    out["analysisSettingsSHA256"] = settings_hash
    out["analysisIdentity"] = f"MicroStructAI-{APP_VERSION}:{mode}:{settings_hash[:12]}"
    if "scoreFormulaVersion" not in out.columns:
        out["scoreFormulaVersion"] = "not_applicable"
    if "scoreNormalizationScope" not in out.columns:
        out["scoreNormalizationScope"] = "not_applicable"
    params["microstructAIVersion"] = APP_VERSION
    params["analysisSchemaVersion"] = ANALYSIS_SCHEMA_VERSION
    params["analysisSettingsSHA256"] = settings_hash
    return out


def analyze_standard_dataset(image_files: list, mode: str, params: dict[str, Any]) -> dict[str, Any]:
    """Versioned ingress/adapter around the preserved V3 calculation body."""
    run_settings = deepcopy(params)
    loaded = load_sequence(image_files, stack_policy=params.get("v4StackPolicy", "reject"),
                           selected_page=params.get("v4SelectedPage", 1))
    checked = require_inputs(loaded, params)
    checked["selection_signature"] = input_selection_signature(image_files)
    result = run_legacy_adapter(_analyze_standard_dataset_v3, loaded, mode, params)
    result["warnings"].extend(checked["warnings"])
    attach_context(result, checked, run_settings)
    attach_automatic(result, loaded)
    from modules.v4_batch_provenance import collect_sample
    collect_sample(result)
    return result


def analyze_dataset(image_files: list, mask_files: list, mode: str, params: dict[str, Any]) -> dict[str, Any]:
    """Dispatch analysis by mode and attach a reproducible V3 identity."""
    if mode == "Ground-Truth Mask Mode":
        raise ValueError("Legacy Ground-Truth Mask Mode could implicitly repair/resize masks. In V4 use the explicit Verified Mask Analysis workspace; legacy source is preserved for reproducibility only.")
    else:
        result = analyze_standard_dataset(image_files, mode, params)
    result["metrics"] = add_analysis_identity(result.get("metrics", pd.DataFrame()), mode, params)
    if mode == "RBC Gap Mode" and bool(params.get("enableStructuralIntelligence", True)) and not result["metrics"].empty:
        try:
            result["structural_intelligence"] = analyze_structural_sequence(
                result.get("previews", []),
                result["metrics"],
                params,
            )
            result["warnings"].extend(result["structural_intelligence"].get("warnings", []))
            structural_frames = result["structural_intelligence"].get("tables", {}).get("gap_frame_summary", pd.DataFrame())
            if len(structural_frames) == len(result["metrics"]):
                trusted = result["metrics"].get("trustedForScore", pd.Series(True, index=result["metrics"].index)).fillna(False).astype(bool)
                trusted &= result["metrics"].get("usedForMainSummary", result["metrics"].get("usedInSummary", pd.Series(True, index=result["metrics"].index))).fillna(False).astype(bool)
                for column in STRUCTURAL_PCA_METRICS:
                    if column not in structural_frames:
                        continue
                    values = pd.to_numeric(structural_frames[column], errors="coerce").reset_index(drop=True)
                    result["metrics"][f"struct_{column}"] = values.to_numpy()
                    result["metrics"][f"struct_trusted_{column}"] = values.where(trusted.reset_index(drop=True), np.nan).to_numpy()
        except Exception as exc:
            result["structural_intelligence"] = {
                "enabled": False,
                "warnings": [f"Structural-intelligence layer failed without affecting existing results: {exc}"],
            }
            result["warnings"].append(
                f"Structural-intelligence layer failed without affecting existing masks, metrics or exports ({exc})."
            )
    return result


def visible_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return user-facing columns for the active mode."""
    if df.empty:
        return df
    if "GT_areaFrac" in df.columns:
        preferred = ["filename"] + INDEX_COLUMNS + ["imageIndex", "maskFile", "mode"] + GT_METRICS + [m for m in AUTO_METRICS if m in df.columns]
    else:
        preferred = ["filename"] + INDEX_COLUMNS + ["imageIndex", "mode"] + CORE_METRICS
        if "gapAreaFrac" in df.columns:
            preferred += (
                MATLAB_WINDOW_COLUMNS
                + RBC_QC_COLUMNS
                + STARTUP_TRANSIENT_COLUMNS
                + CONSERVATIVE_RBC_COLUMNS
                + RBC_GAP_METRICS
                + RBC_SCORE_METRICS
                + RBC_SASI_SYLLECTOGRAM_METRICS
                + NORMALIZED_SYLLECTOGRAM_METRICS
                + RBC_AGG_APPROX_METRICS
            )
        elif "objectCount" in df.columns:
            preferred += GENERIC_OBJECT_METRICS + NORMALIZED_SYLLECTOGRAM_METRICS + STARTUP_TRANSIENT_COLUMNS
        else:
            preferred += STARTUP_TRANSIENT_COLUMNS
    preferred = list(dict.fromkeys(preferred))
    ordered = [col for col in preferred if col in df.columns]
    ordered += [col for col in df.columns if col not in ordered]
    return df[ordered]


def rbc_main_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows selected for headline RBC summary statistics."""
    if df.empty:
        return df
    if "usedInSummary" in df.columns:
        used = df["usedInSummary"].fillna(False).astype(bool)
    elif "usedForMainSummary" in df.columns:
        used = df["usedForMainSummary"].fillna(False).astype(bool)
    else:
        return df
    return df.loc[used].copy()


def parse_batch_sample_labels(sample_name: str) -> dict[str, str]:
    """Infer donor, flow group, and stent condition from a sample folder name."""
    labels = infer_experimental_labels(sample_name)
    donor = "" if labels["donor"] == "unknown" else labels["donor"]
    flow_group = "" if labels["flowHistory"] == "unknown" else labels["flowHistory"]
    stent_condition = "" if labels["stentStatus"] == "unknown" else labels["stentStatus"]
    return {"donor": donor, "flowGroup": flow_group, "stentCondition": stent_condition}


def discover_batch_samples(parent_folder: str, detect_matlab_csv: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Discover RBC sample folders, image folders, and optional MATLAB CSV files."""
    warnings: list[str] = []
    if not parent_folder or not parent_folder.strip():
        return pd.DataFrame(), warnings

    parent = Path(os.path.expandvars(parent_folder.strip().strip('"'))).expanduser()
    if not parent.exists():
        return pd.DataFrame(), [f"Batch parent folder does not exist: {parent}"]
    if not parent.is_dir():
        return pd.DataFrame(), [f"Batch parent path is not a folder: {parent}"]

    candidates = [item for item in sorted(parent.iterdir(), key=lambda path: path.name.lower()) if item.is_dir()]
    parent_images, _parent_warnings = list_local_image_files(str(parent))
    if parent_images:
        candidates = [parent] + candidates

    rows: list[dict[str, Any]] = []
    for sample_dir in candidates:
        image_folder = sample_dir / "images" if (sample_dir / "images").is_dir() else sample_dir
        image_files, _warnings = list_local_image_files(str(image_folder))
        if not image_files:
            continue
        matlab_csv = None
        if detect_matlab_csv:
            csv_candidates = sorted(sample_dir.glob("agg_features_*_GAPS.csv"))
            if not csv_candidates:
                csv_candidates = sorted(sample_dir.rglob("agg_features_*_GAPS.csv"))
            matlab_csv = csv_candidates[0] if csv_candidates else None
        labels = parse_batch_sample_labels(sample_dir.name)
        rows.append(
            {
                "include": True,
                "sampleName": sample_dir.name,
                "imageFolder": str(image_folder),
                "nImages": len(image_files),
                "matlabCSVFound": matlab_csv is not None,
                "matlabCSVPath": str(matlab_csv) if matlab_csv else "",
                **labels,
            }
        )
    if not rows:
        warnings.append(f"No sample subfolders with supported image files were found in: {parent}")
    return pd.DataFrame(rows), warnings


def numeric_values(df: pd.DataFrame, column: str) -> pd.Series:
    """Return finite numeric values from a dataframe column."""
    if column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def metric_mean(df: pd.DataFrame, column: str) -> float:
    values = numeric_values(df, column)
    return float(values.mean()) if not values.empty else np.nan


def metric_median(df: pd.DataFrame, column: str) -> float:
    values = numeric_values(df, column)
    return float(values.median()) if not values.empty else np.nan


def metric_min(df: pd.DataFrame, column: str) -> float:
    values = numeric_values(df, column)
    return float(values.min()) if not values.empty else np.nan


def metric_max(df: pd.DataFrame, column: str) -> float:
    values = numeric_values(df, column)
    return float(values.max()) if not values.empty else np.nan


def metric_final(df: pd.DataFrame, column: str) -> float:
    values = numeric_values(df, column)
    return float(values.iloc[-1]) if not values.empty else np.nan


def first_nonempty_value(df: pd.DataFrame, column: str) -> Any:
    """Return the first non-empty dataframe value for repeated sample-level columns."""
    if column not in df.columns:
        return np.nan
    values = df[column].dropna()
    if values.empty:
        return np.nan
    return values.iloc[0]


def matlab_isana_filtered_df(matlab_df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Filter MATLAB rows to isAna == 1 when an isAna column is present."""
    is_ana_col = next((col for col in matlab_df.columns if str(col).lower() == "isana"), None)
    if is_ana_col is None:
        return matlab_df, False
    raw_isana = matlab_df[is_ana_col]
    numeric_isana = pd.to_numeric(raw_isana, errors="coerce")
    if numeric_isana.notna().any():
        mask = numeric_isana == 1
    else:
        mask = raw_isana.astype(str).str.strip().str.lower().isin(["true", "yes", "1"])
    return matlab_df.loc[mask].reset_index(drop=True), True


MATLAB_SAMPLE_KEY_COLUMNS = ["sampleKey", "sampleName", "sample_name", "sampleID", "sample_id"]


def normalize_matlab_sample_key(value: Any) -> str:
    """Normalize sample identifiers while preserving condition-defining tokens."""
    if pd.isna(value):
        return ""
    normalized = re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())
    return re.sub(r"^h0+(\d+)", r"h\1", normalized)


def matlab_reference_sample_info(matlab_df: pd.DataFrame) -> dict[str, Any]:
    """Describe whether a MATLAB CSV contains reliable per-sample identity values."""
    lower_columns = {str(column).lower(): column for column in matlab_df.columns}
    for candidate in MATLAB_SAMPLE_KEY_COLUMNS:
        column = lower_columns.get(candidate.lower())
        if column is None:
            continue
        values = matlab_df[column].dropna().astype(str).str.strip()
        values = values.loc[values != ""]
        if values.empty:
            continue
        unique_values = values.drop_duplicates().tolist()
        return {
            "sampleKeyColumn": str(column),
            "sampleKeys": unique_values,
            "nSampleKeys": len(unique_values),
        }
    return {"sampleKeyColumn": "", "sampleKeys": [], "nSampleKeys": 0}


def filter_matlab_reference_for_sample(matlab_df: pd.DataFrame, sample_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filter a combined MATLAB master CSV to one batch sample before frame matching."""
    info = matlab_reference_sample_info(matlab_df)
    source_rows = int(len(matlab_df))
    info.update(
        {
            "matlabSourceRows": source_rows,
            "matlabSampleRows": source_rows,
            "matlabSampleFilterApplied": False,
            "matlabSampleFilterValue": "",
        }
    )
    sample_key_column = str(info.get("sampleKeyColumn", ""))
    if not sample_key_column:
        return matlab_df.reset_index(drop=True), info

    target = normalize_matlab_sample_key(sample_name)
    normalized_values = matlab_df[sample_key_column].map(normalize_matlab_sample_key)
    matches = normalized_values == target
    if not bool(matches.any()):
        available = ", ".join(map(str, info.get("sampleKeys", [])[:8]))
        if int(info.get("nSampleKeys", 0)) > 8:
            available += ", ..."
        raise ValueError(
            f"MATLAB CSV contains sample identities in {sample_key_column}, but no rows matched batch sample "
            f"{sample_name!r}. Available sample keys include: {available or 'none'}."
        )

    filtered = matlab_df.loc[matches].reset_index(drop=True)
    matched_key = str(matlab_df.loc[matches, sample_key_column].iloc[0])
    info.update(
        {
            "matlabSampleRows": int(len(filtered)),
            "matlabSampleFilterApplied": True,
            "matlabSampleFilterValue": matched_key,
        }
    )
    return filtered, info


def uploaded_matlab_master_info(csv_payload: bytes) -> dict[str, Any]:
    """Inspect an uploaded CSV and report whether it contains multiple MATLAB samples."""
    try:
        matlab_df = pd.read_csv(io.BytesIO(csv_payload))
    except Exception as exc:
        return {"error": str(exc), "sampleKeyColumn": "", "sampleKeys": [], "nSampleKeys": 0}
    info = matlab_reference_sample_info(matlab_df)
    info["rows"] = int(len(matlab_df))
    return info


def short_metric_name(metric: str) -> str:
    """Return compact suffix used in batch comparison columns."""
    return {
        "gapMeanDiam_px": "gapMeanDiam",
        "gapNtotal": "gapNtotal",
        "gapAreaFrac": "gapAreaFrac",
        "Xi_interp_px": "Xi_interp",
        "LF_log10_LFoverHF": "LF",
        "SASI_0_100": "SASI",
        "AggIndex_0_100": "AggIndex",
        "GapScore_0_100": "GapScore",
        "PTI_0_100": "PTI",
        "gapAreaSlope": "gapAreaSlope",
        "SHI_tileCV": "SHI_tileCV",
    }.get(metric, metric)


def comparison_agreement_counts(summary_df: pd.DataFrame) -> dict[str, int]:
    """Count agreement categories for one MATLAB comparison summary."""
    counts = {"strongAgreementCount": 0, "moderateAgreementCount": 0, "sensitiveCautionCount": 0, "needsCheckCount": 0}
    if summary_df.empty:
        return counts
    for _, row in summary_df.iterrows():
        metric = str(row.get("metric", ""))
        pearson = row.get("Pearson_r", np.nan)
        if metric == "gapAreaSlope":
            counts["sensitiveCautionCount"] += 1
        elif np.isfinite(pearson) and pearson >= 0.90:
            counts["strongAgreementCount"] += 1
        elif np.isfinite(pearson) and pearson >= 0.75:
            counts["moderateAgreementCount"] += 1
        else:
            counts["needsCheckCount"] += 1
    return counts


def batch_matlab_summary_columns(comparison_result: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten MATLAB comparison statistics into sample-level batch columns."""
    output: dict[str, Any] = {"matlabMatchedRows": 0, "matlabComparisonStatus": "not run"}
    for metric in BATCH_COMPARISON_METRICS:
        suffix = short_metric_name(metric)
        output[f"r_{suffix}"] = np.nan
        output[f"bias_{suffix}"] = np.nan

    if not comparison_result:
        output.update(comparison_agreement_counts(pd.DataFrame()))
        return output

    summary = comparison_result.get("summary", pd.DataFrame())
    output["matlabMatchedRows"] = int(comparison_result.get("matchedRows", 0))
    output["matlabComparisonStatus"] = "ok" if not summary.empty else "no shared finite metrics"
    for _, row in summary.iterrows():
        metric = str(row.get("metric", ""))
        suffix = short_metric_name(metric)
        if f"r_{suffix}" in output:
            output[f"r_{suffix}"] = row.get("Pearson_r", np.nan)
            output[f"bias_{suffix}"] = row.get("bias", np.nan)

    main_r = [output.get(f"r_{short_metric_name(metric)}", np.nan) for metric in BATCH_MAIN_METRICS]
    score_r = [output.get(f"r_{short_metric_name(metric)}", np.nan) for metric in BATCH_SCORE_METRICS]
    output["meanPearsonMainMetrics"] = float(np.nanmean(main_r)) if np.isfinite(main_r).any() else np.nan
    output["meanPearsonScores"] = float(np.nanmean(score_r)) if np.isfinite(score_r).any() else np.nan
    output.update(comparison_agreement_counts(summary))
    return output


def batch_sample_summary_row(
    sample: pd.Series,
    analysis: dict[str, Any],
    comparison_result: dict[str, Any] | None,
    status: str = "ok",
    error_message: str = "",
) -> dict[str, Any]:
    """Build one row for the batch sample summary table."""
    metrics_df = analysis.get("metrics", pd.DataFrame()) if analysis else pd.DataFrame()
    summary_df = rbc_main_summary_df(metrics_df)
    qc_summary = analysis.get("params", {}).get("rbc_qc_summary", {}) if analysis else {}
    large_gap_summary = analysis.get("params", {}).get("large_gap_artefact_summary", {}) if analysis else {}
    syllectogram_summary = analysis.get("params", {}).get("sasi_syllectogram_research_summary", {}) if analysis else {}
    fallback_summary = analysis.get("params", {}).get("startup_transient_fallback_summary", {}) if analysis else {}
    conservative_reliable = metrics_df.get("conservativeMaskReliable", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool) if not metrics_df.empty else pd.Series(dtype=bool)
    conservative_class = metrics_df.get("conservativeFrameClass", pd.Series(dtype=str)).fillna("").astype(str) if not metrics_df.empty else pd.Series(dtype=str)
    pod_processing_active = metrics_df.get("podProcessingEnabled", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool) if not metrics_df.empty else pd.Series(dtype=bool)
    pod_sensitive = metrics_df.get("podSensitiveFlag", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool) if not metrics_df.empty else pd.Series(dtype=bool)
    frame_time_col = "matlabTime_s" if "matlabTime_s" in metrics_df.columns else "time_s"
    first_cons_pos = int(np.flatnonzero(conservative_reliable.to_numpy(dtype=bool))[0]) if not conservative_reliable.empty and conservative_reliable.any() else None
    included_frames, excluded_frames = rbc_included_excluded_frame_numbers(metrics_df)
    frames_used_for_summary = int(qc_summary.get("framesUsedForMainSummary", len(summary_df) if not summary_df.empty else 0))

    row: dict[str, Any] = {
        "sampleName": sample.get("sampleName", ""),
        "donor": sample.get("donor", ""),
        "flowGroup": sample.get("flowGroup", ""),
        "stentCondition": sample.get("stentCondition", ""),
        "status": status,
        "errorMessage": error_message,
        "nImagesLoaded": int(sample.get("nImages", 0) or 0),
        "nFramesAnalysed": len(metrics_df),
        "nFramesUsedForSummary": frames_used_for_summary,
        "reliableFramePercent": qc_summary.get("reliableFramePercent", np.nan),
        "sampleQC": qc_summary.get("sampleQC", "n/a"),
        "scoresConfidence": qc_summary.get("scoresConfidence", "n/a"),
        "suggestedStableStartFrame": stable_start_frame_text(qc_summary) if qc_summary else "n/a",
        "stableWindowDetected": bool(qc_summary.get("stableReliableWindowDetected", False)),
        "includedFrameRanges": compact_frame_ranges(included_frames),
        "excludedFrameRanges": compact_frame_ranges(excluded_frames),
        "autoStartFrame": qc_summary.get("autoStartFrame", qc_summary.get("autoStartFrame_POD", np.nan)),
        "autoStartTime_s": qc_summary.get("autoStartTime_s", np.nan),
        "autoStartMethod": qc_summary.get("autoStartMethod", "n/a"),
        "autoStart_originalImageIndex": qc_summary.get("autoStart_originalImageIndex", np.nan),
        "autoStart_originalFileName": qc_summary.get("autoStart_originalFileName", ""),
        "peakSlope": qc_summary.get("peakSlope", np.nan),
        "maxStartSignalInSearchWindow": qc_summary.get("maxStartSignalInSearchWindow", np.nan),
        "noStartSafetyTriggered": qc_summary.get("noStartSafetyTriggered", False),
        "nFramesUsedFromAutoStart": qc_summary.get("nFramesUsedFromAutoStart", frames_used_for_summary),
        "maxGapAreaFilterEnabled": large_gap_summary.get("maxGapAreaFilterEnabled", np.nan),
        "maxGapAreaFrac": large_gap_summary.get("maxGapAreaFrac", np.nan),
        "totalLargeGapObjectsRemoved": large_gap_summary.get("totalLargeGapObjectsRemoved", 0),
        "totalLargeGapObjectsDetected": large_gap_summary.get("totalLargeGapObjectsDetected", 0),
        "maxLargestRemovedGapFrac": large_gap_summary.get("maxLargestRemovedGapFrac", np.nan),
        "percentFramesWithLargeGapRemoval": large_gap_summary.get("percentFramesWithLargeGapRemoval", np.nan),
        "percentFramesWithLargeGapDetection": large_gap_summary.get("percentFramesWithLargeGapDetection", np.nan),
        "largeGapArtefactRisk": large_gap_summary.get("largeGapArtefactRisk", "n/a"),
        "postStartBadFrameExclusionEnabled": qc_summary.get("postStartBadFrameExclusionEnabled", False),
        "maskReliabilityMinForSummary": qc_summary.get("maskReliabilityMinForSummary", np.nan),
        "nFramesExcludedByMaskQC": qc_summary.get("nFramesExcludedByMaskQC", np.nan),
        "percentFramesExcludedByMaskQC": qc_summary.get("percentFramesExcludedByMaskQC", np.nan),
        "meanMaskReliabilityScore": qc_summary.get("meanMaskReliabilityScore", np.nan),
        "minMaskReliabilityScore": qc_summary.get("minMaskReliabilityScore", np.nan),
        "sampleMaskQCRisk": qc_summary.get("sampleMaskQCRisk", "n/a"),
        "hugeGapFrameCount": qc_summary.get("hugeGapFrameCount", np.nan),
        "badMaskFrameCount": qc_summary.get("badMaskFrameCount", np.nan),
        "hardBadMaskFrames": qc_summary.get("hardBadMaskFrames", qc_summary.get("hardBadMaskFrameCount", np.nan)),
        "warningMaskFrames": qc_summary.get("warningMaskFrames", qc_summary.get("warningMaskFrameCount", np.nan)),
        "lowConfidenceFrames": qc_summary.get("lowConfidenceFrames", qc_summary.get("lowConfidenceFrameCount", np.nan)),
        "preStartFramesDetected": qc_summary.get("preStartFramesDetected", qc_summary.get("preStartFrameCount", np.nan)),
        "beforeStartExcludedFrames": qc_summary.get("beforeStartExcludedFrames", qc_summary.get("beforeStartExcludedFrameCount", np.nan)),
        "postStartQCWarningFrames": qc_summary.get("postStartQCWarningFrames", qc_summary.get("postStartQCWarningFrameCount", np.nan)),
        "postStartQCExcludedFrames": qc_summary.get("postStartQCExcludedFrames", qc_summary.get("postStartQCExcludedFrameCount", np.nan)),
        "applySparseMaskMorphologyFilter": qc_summary.get("applySparseMaskMorphologyFilter", analysis.get("params", {}).get("applySparseMaskMorphologyFilter", np.nan)),
        "minReliableMaskAreaFracForMorphology": qc_summary.get("minReliableMaskAreaFracForMorphology", analysis.get("params", {}).get("minReliableMaskAreaFracForMorphology", np.nan)),
        "minReliableObjectsForMorphology": qc_summary.get("minReliableObjectsForMorphology", analysis.get("params", {}).get("minReliableObjectsForMorphology", np.nan)),
        "sparseMaskMorphologyFilteredFrames": qc_summary.get("sparseMaskMorphologyFilteredFrames", qc_summary.get("sparseMaskMorphologyFilteredFrameCount", np.nan)),
        "QCExcludedFrames": qc_summary.get("QCExcludedFrames", qc_summary.get("QCExcludedFrameCount", np.nan)),
        "visualWarningOnlyFrames": qc_summary.get("visualWarningOnlyFrames", qc_summary.get("visualWarningOnlyFrameCount", np.nan)),
        "usedInSummaryFrames": qc_summary.get("usedInSummaryFrames", qc_summary.get("usedInSummaryFrameCount", np.nan)),
        "trustedScoreFrameCount": qc_summary.get("trustedForScoreFrames", qc_summary.get("trustedScoreFrameCount", np.nan)),
        "excludePreStartFramesFromTrustedAnalysis": qc_summary.get("excludePreStartFramesFromTrustedAnalysis", np.nan),
        "excludeBadPostStartQCFramesFromTrustedAnalysis": qc_summary.get("excludeBadPostStartQCFramesFromTrustedAnalysis", np.nan),
        "scoresInterpretableWithWarnings": qc_summary.get("scoresInterpretableWithWarnings", False),
        "unstableSegmentationFrameCount": qc_summary.get("unstableSegmentationFrameCount", np.nan),
        "adaptiveRepairEnabled": qc_summary.get("adaptiveRepairEnabled", False),
        "adaptiveRepairTriggeredCount": qc_summary.get("adaptiveRepairTriggeredCount", np.nan),
        "adaptiveRepairKeptCount": qc_summary.get("adaptiveRepairKeptCount", np.nan),
        "enableSASISyllectogramResearchOutput": bool(syllectogram_summary.get("enableSASISyllectogramResearchOutput", False)),
        "enableStartupTransientFallback": bool(fallback_summary.get("enableStartupTransientFallback", False)),
        "nStartupFallbackFrames": fallback_summary.get("nStartupFallbackFrames", 0),
        "fracStartupFallbackFrames": fallback_summary.get("fracStartupFallbackFrames", np.nan),
        "medianStartupRawCV": fallback_summary.get("medianStartupRawCV", np.nan),
        "medianStartupGradientEnergy": fallback_summary.get("medianStartupGradientEnergy", np.nan),
        "medianStartupFFT_LF_HF_ratio": fallback_summary.get("medianStartupFFT_LF_HF_ratio", np.nan),
        "medianStartupStreakIndex": fallback_summary.get("medianStartupStreakIndex", np.nan),
        "firstReliableFrame": fallback_summary.get("firstReliableFrame", np.nan),
        "firstStableFrameTime_s": fallback_summary.get("firstStableFrameTime_s", np.nan),
        "rbcMode": metrics_df.get("rbcMode", pd.Series([analysis.get("params", {}).get("rbcAnalysisMethod", "Standard RBC Gap Mode")])).dropna().astype(str).iloc[0] if not metrics_df.empty else analysis.get("params", {}).get("rbcAnalysisMethod", "Standard RBC Gap Mode"),
        "podProcessingActive": bool(pod_processing_active.any()) if not pod_processing_active.empty else False,
        "removeModes": first_nonempty_value(metrics_df, "removeModes") if not metrics_df.empty else np.nan,
        "nConservativeReliableFrames": int(conservative_reliable.sum()) if not conservative_reliable.empty else 0,
        "fracConservativeReliableFrames": float(conservative_reliable.mean()) if not conservative_reliable.empty else np.nan,
        "nWeakPreGapFrames": int(conservative_class.str.contains("weak/pre-gap", case=False).sum()) if not conservative_class.empty else 0,
        "fracWeakPreGapFrames": float(conservative_class.str.contains("weak/pre-gap", case=False).mean()) if not conservative_class.empty else np.nan,
        "nPODSensitivityFlags": int(pod_sensitive.sum()) if not pod_sensitive.empty and bool(pod_processing_active.any()) else 0,
        "medianPodAreaDelta": metric_median(metrics_df, "podAreaDelta") if not pod_processing_active.empty and bool(pod_processing_active.any()) else np.nan,
        "medianPodMode1Energy": metric_median(metrics_df, "podMode1Energy") if not pod_processing_active.empty and bool(pod_processing_active.any()) else np.nan,
        "medianRawCV": metric_median(metrics_df, "rawCV"),
        "medianCorrectedCV": metric_median(metrics_df, "correctedCV"),
        "intensityCorrectionMode": first_nonempty_value(metrics_df, "intensityCorrectionMode"),
        "intensityCorrectionApplied": bool(metrics_df.get("intensityCorrectionApplied", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool).any()) if not metrics_df.empty else False,
        "intensityCorrectionNeeded": bool(metrics_df.get("intensityCorrectionNeeded", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool).any()) if not metrics_df.empty else False,
        "nIlluminationJumpFrames": first_nonempty_value(metrics_df, "nIlluminationJumpFrames") if "nIlluminationJumpFrames" in metrics_df.columns else (int(metrics_df.get("illuminationJumpFlag", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool).sum()) if not metrics_df.empty else 0),
        "maxAbsLogJump": first_nonempty_value(metrics_df, "maxAbsLogJump"),
        "meanIntensityTemporalCV": first_nonempty_value(metrics_df, "meanIntensityTemporalCV"),
        "medianIlluminationGain": metric_median(metrics_df, "illuminationGain"),
        "maxIlluminationGain": metric_max(metrics_df, "illuminationGain"),
        "minIlluminationGain": metric_min(metrics_df, "illuminationGain"),
        "nRawGainSensitiveFrames": int(metrics_df.get("rawGainCorrectionSensitiveFlag", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool).sum()) if not metrics_df.empty else 0,
        "fracRawGainSensitiveFrames": float(metrics_df.get("rawGainCorrectionSensitiveFlag", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool).mean()) if not metrics_df.empty else np.nan,
        "firstConservativeReliableFrame": frame_identity_at_position(metrics_df, first_cons_pos, "originalImageIndex_1based") if first_cons_pos is not None else np.nan,
        "firstConservativeReliableTime_s": frame_identity_at_position(metrics_df, first_cons_pos, frame_time_col) if first_cons_pos is not None and frame_time_col in metrics_df.columns else np.nan,
    }
    for column in RBC_SASI_SYLLECTOGRAM_METRICS:
        if column == "enableSASISyllectogramResearchOutput":
            continue
        row[column] = syllectogram_summary.get(column, np.nan)
    for column in NORMALIZED_SYLLECTOGRAM_METRICS:
        if column.endswith(("_norm01", "_norm_minRise", "_normMinRise", "_0_1")) or column in {"MeanI_norm_minRise", "MeanI_norm_0_1", "CV_norm_0_1"}:
            continue
        row[column] = first_nonempty_value(metrics_df, column)
    for metric in ["SASI_0_100", "AggIndex_0_100"]:
        qc_metric = metric.replace("_0_100", "_QC_0_100")
        row[f"raw_median_{metric}"] = metric_median(summary_df, metric)
        row[f"raw_mean_{metric}"] = metric_mean(summary_df, metric)
        row[f"raw_final_{metric}"] = metric_final(summary_df, metric)
        summary_metric = qc_metric if qc_metric in summary_df.columns else metric
        row[f"median_{metric}"] = metric_median(summary_df, summary_metric)
        row[f"mean_{metric}"] = metric_mean(summary_df, summary_metric)
        row[f"final_{metric}"] = metric_final(summary_df, summary_metric)
        if qc_metric in summary_df.columns:
            row[f"median_{qc_metric}"] = metric_median(summary_df, qc_metric)
            row[f"mean_{qc_metric}"] = metric_mean(summary_df, qc_metric)
            row[f"final_{qc_metric}"] = metric_final(summary_df, qc_metric)
    for metric in [
        "GapScore_0_100",
        "PTI_0_100",
        "Xi_interp_px",
        "gapMeanDiam_px",
        "gapNtotal",
        "gapAreaFrac",
        "LF_log10_LFoverHF",
        "SHI_tileCV",
        "gapAreaSlope",
    ]:
        qc_metric = metric.replace("_0_100", "_QC_0_100") if metric in {"GapScore_0_100", "PTI_0_100"} else metric
        row[f"raw_median_{metric}"] = metric_median(summary_df, metric)
        row[f"median_{metric}"] = metric_median(summary_df, qc_metric if qc_metric in summary_df.columns else metric)
        if qc_metric != metric and qc_metric in summary_df.columns:
            row[f"median_{qc_metric}"] = metric_median(summary_df, qc_metric)
    # Extra structural-architecture biomarkers: per-sample median + final.
    for metric in [
        "anisotropy",
        "orientOrderParam",
        "interfaceDensity_px",
        "largestClusterFrac_phase",
        "largestClusterFrac_complement",
    ]:
        row[f"median_{metric}"] = metric_median(summary_df, metric)
        row[f"mean_{metric}"] = metric_mean(summary_df, metric)
        row[f"final_{metric}"] = metric_final(summary_df, metric)
    row.update(batch_matlab_summary_columns(comparison_result))
    return row


def batch_failure_summary_row(sample: pd.Series, error_message: str) -> dict[str, Any]:
    """Build a failed sample summary row with expected columns."""
    return batch_sample_summary_row(sample, {"metrics": pd.DataFrame(), "params": {}}, None, status="failed", error_message=error_message)


def batch_object_sample_summary_row(
    sample: pd.Series,
    analysis: dict[str, Any],
    batch_mode: str,
    status: str = "ok",
    error_message: str = "",
) -> dict[str, Any]:
    """Build one generic Object Mode batch summary row."""
    metrics_df = analysis.get("metrics", pd.DataFrame()) if analysis else pd.DataFrame()
    fallback_summary = analysis.get("params", {}).get("startup_transient_fallback_summary", {}) if analysis else {}
    row: dict[str, Any] = {
        "sampleName": sample.get("sampleName", ""),
        "donor": sample.get("donor", ""),
        "flowGroup": sample.get("flowGroup", ""),
        "stentCondition": sample.get("stentCondition", ""),
        "batchValidationMode": batch_mode,
        "status": status,
        "errorMessage": error_message,
        "nImagesLoaded": int(sample.get("nImages", 0) or 0),
        "nFramesAnalysed": len(metrics_df),
        "nFramesUsedForSummary": len(metrics_df),
        "selectedObjectMethod": batch_mode if batch_mode in OBJECT_ANALYSIS_METHODS else "Object Mode",
        "enhancedObjectModeEnabled": bool(batch_mode == "Enhanced Object Mode"),
        "conservativeObjectModeEnabled": bool(batch_mode in {"Conservative Object Mode", "Experimental POD-removed Object Mode"}),
        "experimentalPODObjectModeEnabled": bool(batch_mode == "Experimental POD-removed Object Mode"),
        "experimentalCompositeScoresEnabled": bool(
            metrics_df.get("experimentalCompositeScoresEnabled", pd.Series(dtype=bool)).fillna(False).astype(bool).any()
        ),
        "enableStartupTransientFallback": bool(fallback_summary.get("enableStartupTransientFallback", False)),
        "nStartupFallbackFrames": fallback_summary.get("nStartupFallbackFrames", 0),
        "fracStartupFallbackFrames": fallback_summary.get("fracStartupFallbackFrames", np.nan),
        "medianStartupRawCV": fallback_summary.get("medianStartupRawCV", np.nan),
        "medianStartupGradientEnergy": fallback_summary.get("medianStartupGradientEnergy", np.nan),
        "medianStartupFFT_LF_HF_ratio": fallback_summary.get("medianStartupFFT_LF_HF_ratio", np.nan),
        "medianStartupStreakIndex": fallback_summary.get("medianStartupStreakIndex", np.nan),
        "firstReliableFrame": fallback_summary.get("firstReliableFrame", np.nan),
        "firstStableFrameTime_s": fallback_summary.get("firstStableFrameTime_s", np.nan),
        "intensityCorrectionMode": first_nonempty_value(metrics_df, "intensityCorrectionMode"),
        "intensityCorrectionApplied": bool(metrics_df.get("intensityCorrectionApplied", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool).any()) if not metrics_df.empty else False,
        "intensityCorrectionNeeded": bool(metrics_df.get("intensityCorrectionNeeded", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool).any()) if not metrics_df.empty else False,
        "nIlluminationJumpFrames": first_nonempty_value(metrics_df, "nIlluminationJumpFrames"),
        "maxAbsLogJump": first_nonempty_value(metrics_df, "maxAbsLogJump"),
        "meanIntensityTemporalCV": first_nonempty_value(metrics_df, "meanIntensityTemporalCV"),
        "medianIlluminationGain": metric_median(metrics_df, "illuminationGain"),
        "maxIlluminationGain": metric_max(metrics_df, "illuminationGain"),
        "minIlluminationGain": metric_min(metrics_df, "illuminationGain"),
        "nRawGainSensitiveFrames": int(metrics_df.get("rawGainCorrectionSensitiveFlag", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool).sum()) if not metrics_df.empty else 0,
        "fracRawGainSensitiveFrames": float(metrics_df.get("rawGainCorrectionSensitiveFlag", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool).mean()) if not metrics_df.empty else np.nan,
    }
    if not metrics_df.empty and "objectMaskReliable" in metrics_df.columns:
        reliable = metrics_df["objectMaskReliable"].fillna(False).astype(bool)
        weak = metrics_df.get("objectFrameClass", pd.Series("", index=metrics_df.index)).astype(str).str.contains("weak|sparse|no_reliable", case=False, regex=True)
        pod_sensitive = metrics_df.get("podObjectSensitiveFlag", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool)
        row["nReliableObjectFrames"] = int(reliable.sum())
        row["fracReliableObjectFrames"] = float(reliable.mean()) if len(reliable) else np.nan
        row["nWeakObjectFrames"] = int(weak.sum())
        row["fracWeakObjectFrames"] = float(weak.mean()) if len(weak) else np.nan
        row["nPODObjectSensitiveFrames"] = int(pod_sensitive.sum())
        row["medianPodObjectAreaDelta"] = metric_median(metrics_df, "podObjectAreaDelta")
        row["medianObjectXiMask"] = metric_median(metrics_df, "Object_Xi_mask")
        row["medianObjectStructureScore"] = metric_median(metrics_df, "ObjectStructureScore_QC")
        row["medianObjectAggregationIndex"] = metric_median(metrics_df, "ObjectAggregationIndex_QC")
        reliable_positions = np.flatnonzero(reliable.to_numpy(dtype=bool))
        if reliable_positions.size:
            first_pos = int(reliable_positions[0])
            row["firstReliableObjectFrame"] = frame_identity_at_position(metrics_df, first_pos, "originalImageIndex_1based")
            row["firstReliableObjectTime_s"] = frame_identity_at_position(metrics_df, first_pos, "matlabTime_s")
        else:
            row["firstReliableObjectFrame"] = np.nan
            row["firstReliableObjectTime_s"] = np.nan
        row["nObjectIlluminationJumpFrames"] = int(
            metrics_df.get("objectIlluminationJumpFlag", pd.Series(False, index=metrics_df.index)).fillna(False).astype(bool).sum()
        )
    else:
        row.update(
            {
                "nReliableObjectFrames": np.nan,
                "fracReliableObjectFrames": np.nan,
                "nWeakObjectFrames": np.nan,
                "fracWeakObjectFrames": np.nan,
                "nPODObjectSensitiveFrames": np.nan,
                "medianPodObjectAreaDelta": np.nan,
                "medianObjectXiMask": np.nan,
                "medianObjectStructureScore": np.nan,
                "medianObjectAggregationIndex": np.nan,
                "firstReliableObjectFrame": np.nan,
                "firstReliableObjectTime_s": np.nan,
                "nObjectIlluminationJumpFrames": np.nan,
            }
        )
    for metric in [
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
        "skelDensity",
        "branchDensity",
        "endpointDensity",
        "gSASI_0_100",
        "gAggIndex_0_100",
        "objectAreaFracFinal",
        "objectNtotalFinal",
        "objectMeanDiamFinal",
        "Object_Xi_mask",
        "Object_Xi_QC",
        "ObjectStructureScore_QC",
        "ObjectAggregationIndex_QC",
        "conservativeNoPODObjectAreaFrac",
        "podRemovedObjectAreaFrac",
        "podObjectAreaDelta",
    ]:
        row[f"mean_{metric}"] = metric_mean(metrics_df, metric)
        row[f"median_{metric}"] = metric_median(metrics_df, metric)
        row[f"final_{metric}"] = metric_final(metrics_df, metric)
    for column in ["lowResolutionRisk", "segmentationConfidence", "objectSegmentationTarget", "objectSegmentationTargetUsed", "objectFrameClass"]:
        values = metrics_df.get(column, pd.Series(dtype=object)).dropna().astype(str)
        row[column] = ", ".join(sorted(set(values))) if not values.empty else "n/a"
    row["gSASI_available"] = bool(metrics_df.get("gSASI_available", pd.Series(dtype=bool)).fillna(False).astype(bool).any())
    row["gAggIndex_available"] = bool(metrics_df.get("gAggIndex_available", pd.Series(dtype=bool)).fillna(False).astype(bool).any())
    row["gSASI_warning"] = " | ".join(sorted(set(metrics_df.get("gSASI_warning", pd.Series(dtype=object)).dropna().astype(str)) - {""}))
    row["gAggIndex_warning"] = " | ".join(sorted(set(metrics_df.get("gAggIndex_warning", pd.Series(dtype=object)).dropna().astype(str)) - {""}))
    for column in NORMALIZED_SYLLECTOGRAM_METRICS:
        if column.endswith(("_norm01", "_norm_minRise", "_normMinRise", "_0_1")) or column in {"MeanI_norm_minRise", "MeanI_norm_0_1", "CV_norm_0_1"}:
            continue
        row[column] = first_nonempty_value(metrics_df, column)
    row["matlabComparisonStatus"] = "not applicable for Object Mode"
    return row


def batch_object_failure_summary_row(sample: pd.Series, batch_mode: str, error_message: str) -> dict[str, Any]:
    """Build a failed generic Object Mode batch summary row."""
    return batch_object_sample_summary_row(sample, {"metrics": pd.DataFrame()}, batch_mode, status="failed", error_message=error_message)


def empty_batch_figure(message: str) -> plt.Figure:
    """Return a simple placeholder matplotlib figure."""
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    return fig


def plot_batch_pearson_heatmap(matlab_summary_df: pd.DataFrame) -> plt.Figure:
    """Plot samples by metric Pearson correlations from batch MATLAB comparisons."""
    if matlab_summary_df.empty or not {"sampleName", "metric", "Pearson_r"}.issubset(matlab_summary_df.columns):
        return empty_batch_figure("No MATLAB comparison correlations are available.")

    plot_df = matlab_summary_df.loc[matlab_summary_df["metric"].isin(BATCH_COMPARISON_METRICS)].copy()
    plot_df["Pearson_r"] = pd.to_numeric(plot_df["Pearson_r"], errors="coerce")
    pivot = plot_df.pivot_table(index="sampleName", columns="metric", values="Pearson_r", aggfunc="mean")
    metric_cols = [metric for metric in BATCH_COMPARISON_METRICS if metric in pivot.columns]
    pivot = pivot.reindex(columns=metric_cols)
    if pivot.empty:
        return empty_batch_figure("No finite MATLAB comparison correlations are available.")

    matrix = pivot.to_numpy(dtype=float)
    fig_width = max(8, 0.75 * len(metric_cols))
    fig_height = max(3.5, 0.4 * len(pivot.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("MATLAB comparison Pearson r by sample")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Sample")
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    return fig


def plot_batch_bar(summary_df: pd.DataFrame, column: str, title: str, ylabel: str) -> plt.Figure:
    """Plot a sample-level bar chart for a numeric batch summary column."""
    if summary_df.empty or column not in summary_df.columns:
        return empty_batch_figure(f"No values available for {column}.")
    values = pd.to_numeric(summary_df[column], errors="coerce")
    names = summary_df.get("sampleName", pd.Series(np.arange(len(summary_df)), index=summary_df.index)).astype(str)
    finite = values.notna() & np.isfinite(values)
    if not finite.any():
        return empty_batch_figure(f"No finite values available for {column}.")

    plot_values = values[finite]
    plot_names = names[finite]
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(plot_names)), 4))
    ax.bar(np.arange(len(plot_values)), plot_values.to_numpy(dtype=float), color="#2f6f8f")
    ax.set_xticks(np.arange(len(plot_values)))
    ax.set_xticklabels(plot_names, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_batch_qc_counts(summary_df: pd.DataFrame) -> plt.Figure:
    """Plot counts of batch sample QC classes."""
    if summary_df.empty or "sampleQC" not in summary_df.columns:
        return empty_batch_figure("No QC classes are available.")
    counts = summary_df["sampleQC"].fillna("n/a").astype(str).value_counts()
    if counts.empty:
        return empty_batch_figure("No QC classes are available.")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(counts)), counts.to_numpy(dtype=float), color="#6a8f3a")
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(counts.index, rotation=30, ha="right")
    ax.set_title("Batch QC class counts")
    ax.set_ylabel("Samples")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_batch_condition_summary(summary_df: pd.DataFrame) -> plt.Figure:
    """Plot median SASI/AggIndex by flow and stent labels when available."""
    required = {"flowGroup", "stentCondition", "median_SASI_0_100", "median_AggIndex_0_100"}
    if summary_df.empty or not required.issubset(summary_df.columns):
        return empty_batch_figure("Condition labels or summary scores are not available.")

    plot_df = summary_df.copy()
    plot_df["flowGroup"] = plot_df["flowGroup"].fillna("").astype(str)
    plot_df["stentCondition"] = plot_df["stentCondition"].fillna("").astype(str)
    plot_df = plot_df[(plot_df["flowGroup"] != "") | (plot_df["stentCondition"] != "")]
    if plot_df.empty:
        return empty_batch_figure("No condition labels are available.")

    plot_df["condition"] = (plot_df["flowGroup"] + " " + plot_df["stentCondition"]).str.strip()
    grouped = plot_df.groupby("condition", sort=False)[["median_SASI_0_100", "median_AggIndex_0_100"]].median(numeric_only=True)
    if grouped.empty:
        return empty_batch_figure("No finite condition summary scores are available.")

    x = np.arange(len(grouped.index))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(grouped.index)), 4))
    ax.bar(x - width / 2, grouped["median_SASI_0_100"], width=width, label="Median SASI", color="#2f6f8f")
    ax.bar(x + width / 2, grouped["median_AggIndex_0_100"], width=width, label="Median AggIndex", color="#b07d2c")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Condition summary scores")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


START_FRAME_DIAGNOSTIC_SIGNAL_GROUPS: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "Normalized optical intensity trend",
        "intensity",
        [
            ("MeanI_norm_minRise", "I/min(I)-1"),
            ("opticalMean_raw_normMinRise", "raw I/min(I)-1"),
            ("opticalMean_corrected_normMinRise", "corrected I/min(I)-1"),
            ("MeanI_norm_0_1", "mean I 0-1"),
            ("opticalMean_raw_norm01", "raw mean 0-1"),
            ("opticalMean_corrected_norm01", "corrected mean 0-1"),
        ],
    ),
    (
        "Optical CV / contrast trend",
        "CV / contrast",
        [
            ("CV_norm_0_1", "CV 0-1"),
            ("opticalCV_raw_norm01", "raw CV 0-1"),
            ("opticalCV_corrected_norm01", "corrected CV 0-1"),
            ("rawContrast01_autoStart", "raw contrast"),
            ("ctN_autoStart", "contrast N"),
            ("opticalCV_raw", "raw CV"),
            ("opticalCV_corrected", "corrected CV"),
        ],
    ),
    (
        "Motion / frame-difference signal",
        "motion",
        [
            ("motionD", "motionD"),
            ("motionD_like_matlab", "MATLAB-like D"),
            ("frameDifference", "frame difference"),
            ("rawFrameDiff_autoStart", "raw frame diff"),
            ("frameDiff_autoStart", "frame diff"),
        ],
    ),
    (
        "Structure-growth signal",
        "structure",
        [
            ("skelN_autoStart", "skelN"),
            ("branchN_autoStart", "branchN"),
            ("bfN_autoStart", "bright frac N"),
            ("rawBrightFrac_autoStart", "raw bright frac"),
            ("aggSkelLen", "aggSkelLen"),
            ("aggBranchPts", "aggBranchPts"),
            ("ObjectStructureScore_raw_0_100", "Object structure"),
            ("ObjectAggregationIndex_raw_0_100", "Object aggregation"),
        ],
    ),
    (
        "Combined start-score signal",
        "start score",
        [
            ("startSignal_autoStart", "startSignal"),
        ],
    ),
    (
        "Start-score slope",
        "slope",
        [
            ("startSignalSlope_autoStart", "dStartSm"),
        ],
    ),
]


def start_frame_diagnostic_columns(df: pd.DataFrame, min_finite: int = 2) -> list[str]:
    """Return available start-frame diagnostic columns with enough numeric values."""
    valid: list[str] = []
    for _title, _ylabel, signals in START_FRAME_DIAGNOSTIC_SIGNAL_GROUPS:
        for column, _label in signals:
            if column in valid or column not in df.columns:
                continue
            values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if int(values.notna().sum()) >= int(min_finite):
                valid.append(column)
    return valid


def has_start_frame_diagnostic_signals(df: pd.DataFrame) -> bool:
    """Return True when frame-wise start detector signals are available."""
    return bool(start_frame_diagnostic_columns(df, min_finite=2))


def _first_finite_from_columns(df: pd.DataFrame, columns: list[str]) -> float:
    """Return the first finite numeric value found in the requested columns."""
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not values.empty:
            return float(values.iloc[0])
    return np.nan


def _start_diagnostic_x_axis(df: pd.DataFrame) -> tuple[pd.Series, str, str]:
    """Choose the most interpretable x-axis for start-frame diagnostic plots."""
    for column, label in [
        ("matlabTime_s", "time_s"),
        ("time_s", "time_s"),
        ("analysisIndex_1based", "analysisIndex_1based"),
        ("originalImageIndex_1based", "originalImageIndex_1based"),
    ]:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if int(values.notna().sum()) >= 2:
            return values, label, column
    return pd.Series(np.arange(1, len(df) + 1, dtype=float), index=df.index), "frame index", "__row_position_1based__"


def _marker_from_frame_column(df: pd.DataFrame, x_values: pd.Series, frame_column: str, frame_value: float) -> float:
    """Map a stored frame number onto the selected diagnostic x-axis."""
    if not np.isfinite(frame_value) or frame_column not in df.columns:
        return np.nan
    frames = pd.to_numeric(df[frame_column], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    matches = np.flatnonzero(np.isfinite(frames) & np.isclose(frames, frame_value, atol=1e-6))
    if matches.size:
        candidate = pd.to_numeric(x_values.iloc[int(matches[0])], errors="coerce")
        return float(candidate) if np.isfinite(candidate) else np.nan
    return np.nan


def _selected_start_marker(df: pd.DataFrame, x_values: pd.Series, x_source: str) -> tuple[float, str]:
    """Return the selected start marker position and a user-facing label."""
    method = str(first_nonempty_value(df, "autoStartMethod")).strip().lower()
    mode_text = str(first_nonempty_value(df, "startFrameMode") if "startFrameMode" in df.columns else first_nonempty_value(df, "startupQCMode")).strip().lower()
    if "manual" in method or "manual" in mode_text:
        label = "Manual start frame"
    elif "matlab" in method or "matlab" in mode_text:
        label = "MATLAB analysis start"
    elif "auto" in method or "auto" in mode_text or "first-rise" in mode_text:
        label = "Auto start frame"
    else:
        label = "Selected start frame"

    if x_source in {"matlabTime_s", "time_s"}:
        marker = _first_finite_from_columns(df, ["autoStartTime_s", "startupQC_startTime_s"])
        if np.isfinite(marker):
            return marker, label
    elif x_source == "analysisIndex_1based":
        marker = _first_finite_from_columns(df, ["autoStartFrame_POD", "suggestedStableStartAnalysisIndex"])
        if np.isfinite(marker):
            return marker, label
    elif x_source == "originalImageIndex_1based":
        marker = _first_finite_from_columns(df, ["autoStart_originalImageIndex", "finalStartFrameUsed", "startupQCStableStartFrame"])
        if np.isfinite(marker):
            return marker, label

    for frame_column, frame_value in [
        ("analysisIndex_1based", _first_finite_from_columns(df, ["autoStartFrame_POD", "suggestedStableStartAnalysisIndex"])),
        ("originalImageIndex_1based", _first_finite_from_columns(df, ["autoStart_originalImageIndex", "finalStartFrameUsed", "startupQCStableStartFrame"])),
    ]:
        marker = _marker_from_frame_column(df, x_values, frame_column, frame_value)
        if np.isfinite(marker):
            return marker, label
    return np.nan, label


def plot_auto_start_diagnostics(df: pd.DataFrame) -> plt.Figure:
    """Plot available first-rise/start-frame diagnostic signals."""
    if df.empty:
        return empty_batch_figure("No auto-start diagnostic data are available.")
    x_values, x_label, x_source = _start_diagnostic_x_axis(df)
    marker_x, marker_label = _selected_start_marker(df, x_values, x_source)
    peak_slope = _first_finite_from_columns(df, ["peakSlope"])
    slope_frac = _first_finite_from_columns(df, ["autoStartSlopeFrac"])
    if not np.isfinite(slope_frac):
        slope_frac = 0.40
    signal_threshold = _first_finite_from_columns(df, ["autoStartMinDevelopedSignal"])
    if not np.isfinite(signal_threshold):
        signal_threshold = 0.45

    panel_data: list[tuple[str, str, list[tuple[pd.Series, str, str]]]] = []
    for title, ylabel, signals in START_FRAME_DIAGNOSTIC_SIGNAL_GROUPS:
        lines: list[tuple[pd.Series, str, str]] = []
        for column, label in signals:
            if column not in df.columns:
                continue
            values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if int(values.notna().sum()) >= 2:
                lines.append((values, label, column))
        if lines:
            panel_data.append((title, ylabel, lines))

    if not panel_data:
        return empty_batch_figure("Start-frame diagnostic signals were not available for this analysis mode or dataset.")

    fig_height = min(15.0, max(4.2, 2.35 * len(panel_data)))
    fig, axes = plt.subplots(len(panel_data), 1, figsize=(9.5, fig_height), sharex=True)
    if len(panel_data) == 1:
        axes = [axes]
    marker_drawn_labels: set[str] = set()
    for ax, (title, ylabel, lines) in zip(axes, panel_data):
        for values, label, column in lines:
            ax.plot(x_values, values, marker="o", linewidth=1.4, markersize=3.6, label=label)
            if column == "startSignal_autoStart":
                ax.axhline(signal_threshold, color="#7A7A7A", linestyle="--", linewidth=1.1, label="developed signal threshold")
            if column == "startSignalSlope_autoStart" and np.isfinite(peak_slope):
                ax.axhline(slope_frac * peak_slope, color="#7A7A7A", linestyle="--", linewidth=1.1, label="slope threshold")
        if np.isfinite(marker_x):
            label = marker_label if marker_label not in marker_drawn_labels else None
            ax.axvline(marker_x, color="#B33A3A", linestyle="-.", linewidth=1.35, alpha=0.9, label=label)
            marker_drawn_labels.add(marker_label)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel(x_label)
    fig.suptitle("First-rise automatic start-frame detection")
    fig.tight_layout()
    return fig


def calibration_folder_refs(folder_path: str) -> tuple[list[ImageFileRef], str, list[str]]:
    """Load a local calibration sequence, accepting either a sample or images folder."""
    folder_text = str(folder_path).strip().strip('"')
    if not folder_text:
        return [], "", []
    folder = Path(os.path.expandvars(folder_text)).expanduser()
    image_folder = folder / "images" if (folder / "images").is_dir() else folder
    image_paths, warnings = list_local_image_files(str(image_folder))
    return make_image_file_refs(image_paths), str(image_folder), warnings


def calibration_input_refs(params: dict[str, Any], sequence: str) -> tuple[list[ImageFileRef], str, list[str]]:
    """Collect one calibration sequence from uploads or a local folder."""
    input_method = str(params.get("calibrationInputMethod", "Load images from local folder path"))
    if input_method == "Upload image files":
        uploads = params.get(f"{sequence}Uploads", []) or []
        refs = make_image_file_refs(uploads)
        if not refs:
            return [], f"Uploaded {sequence} image files: none", []
        signature_parts = []
        for ref in refs:
            source_size = getattr(ref.source, "size", None)
            if source_size is None:
                source_size = len(ref.source.getvalue())
            signature_parts.append(f"{ref.name}:{source_size}")
        digest = hashlib.sha1("|".join(signature_parts).encode("utf-8")).hexdigest()[:10]
        preview_names = ", ".join(ref.name for ref in refs[:3])
        more_text = f", ... (+{len(refs) - 3} more)" if len(refs) > 3 else ""
        return refs, f"Uploaded {sequence} image files ({len(refs)}, set {digest}): {preview_names}{more_text}", []
    return calibration_folder_refs(str(params.get(f"{sequence}Folder", "")))


def calibration_analysis_params(settings: dict[str, Any], input_source: str, n_images: int, preset_name: str) -> dict[str, Any]:
    """Build RBC-only analysis parameters for one calibration trial."""
    params = dict(DEFAULT_CAMERA_PRESETS["USB camera RBC default"])
    params.update(settings)
    params.update(
        {
            "cameraPresetName": preset_name,
            "input_method": "Upload image files" if input_source.startswith("Uploaded ") else "Local folder path",
            "image_folder_path": input_source,
            "files_loaded_before_filtering": n_images,
            "selected_image_count": n_images,
            "overlay_style": "Boundary",
            "enableRbcQc": True,
            "startFrameMode": "Auto first-rise detector",
            "startupQCMode": "Auto first-rise detector",
            "useAutoStartFrame": True,
            "rbc_analysis_window_mode": "Auto first-rise detector",
            "excludeLowConfidenceFromMainSummary": False,
            "matlab_window_enabled": False,
            "matlab_fps": float(settings.get("matlab_fps", 1.0) or 1.0),
            "pre_analysis_warnings": [],
        }
    )
    return params


def calibration_plausibility_params(params: dict[str, Any]) -> dict[str, Any]:
    """Return editable mask plausibility thresholds for calibration scoring."""
    return {
        "minPlausibleGapAreaFrac": float(params.get("minPlausibleGapAreaFrac", 0.01)),
        "maxPlausibleGapAreaFrac": float(params.get("maxPlausibleGapAreaFrac", 0.60)),
        "minPlausibleGapNtotal": int(params.get("minPlausibleGapNtotal", 20)),
        "minPlausibleGapMeanDiam_px": float(params.get("minPlausibleGapMeanDiam_px", 2.0)),
        "maxTinyObjectWarning": float(params.get("maxTinyObjectWarning", 0.30)),
        "maxHugeConnectedRegionWarning": float(params.get("maxHugeConnectedRegionWarning", 0.80)),
    }


def run_camera_calibration(params: dict[str, Any]) -> dict[str, Any]:
    """Run a separate RBC camera/setup calibration experiment."""
    positive_refs, positive_source, positive_warnings = calibration_input_refs(params, "positive")
    if not positive_refs:
        raise ValueError("No supported positive RBC images were found. Choose a folder or upload positive RBC image files.")

    negative_refs: list[ImageFileRef] = []
    negative_source = ""
    negative_warnings: list[str] = []
    if params.get("negativeUploads") or str(params.get("negativeFolder", "")).strip():
        negative_refs, negative_source, negative_warnings = calibration_input_refs(params, "negative")
        if not negative_refs:
            raise ValueError("The negative/control sequence was provided, but no supported images were found.")

    base_name = str(params.get("basePresetName", "USB camera RBC default"))
    base_settings = dict(DEFAULT_CAMERA_PRESETS.get(base_name, DEFAULT_CAMERA_PRESETS["USB camera RBC default"]))
    candidates = generate_calibration_candidates(base_settings, int(params.get("maxCalibrationTrials", 100)))
    plausibility = calibration_plausibility_params(params)
    user_start = params.get("userStartFrame")
    status_box = st.empty()
    progress = st.progress(0)
    rows: list[dict[str, Any]] = []
    candidate_settings: dict[int, dict[str, Any]] = {}
    best_score = -np.inf
    best_analyses: tuple[dict[str, Any], dict[str, Any] | None] | None = None
    trial_warnings = list(positive_warnings) + list(negative_warnings)

    for index, settings in enumerate(candidates, start=1):
        status_box.info(f"Testing calibration candidate {index} of {len(candidates)}")
        candidate_settings[index] = dict(settings)
        try:
            positive_trial_params = calibration_analysis_params(settings, positive_source, len(positive_refs), f"Calibration candidate {index}")
            positive_analysis = analyze_standard_dataset(positive_refs, "RBC Gap Mode", positive_trial_params)
            negative_analysis = None
            if negative_refs:
                negative_trial_params = calibration_analysis_params(settings, negative_source, len(negative_refs), f"Calibration candidate {index}")
                negative_analysis = analyze_standard_dataset(negative_refs, "RBC Gap Mode", negative_trial_params)
            score_row = score_calibration_candidate(index, settings, positive_analysis, negative_analysis, plausibility, user_start)
            rows.append(score_row)
            score = float(score_row.get("calibrationScore", -np.inf))
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_analyses = (positive_analysis, negative_analysis)
        except Exception as exc:
            rows.append({"candidateID": index, **{key: settings.get(key) for key in TUNABLE_PARAMETER_KEYS}, "calibrationScore": -np.inf, "error": str(exc)})
            trial_warnings.append(f"Candidate {index} failed ({exc}).")
        finally:
            progress.progress(index / len(candidates))

    status_box.success(f"Calibration finished: {len(candidates)} candidates tested.")
    results_df = pd.DataFrame(rows)
    finite_scores = pd.to_numeric(results_df.get("calibrationScore", pd.Series(dtype=float)), errors="coerce")
    valid = results_df.loc[np.isfinite(finite_scores)].copy()
    if valid.empty:
        raise ValueError("Calibration did not produce any valid candidate settings.")
    valid = valid.sort_values("calibrationScore", ascending=False).reset_index(drop=True)
    best_row = valid.iloc[0]
    best_id = int(best_row["candidateID"])
    recommended_settings = dict(candidate_settings[best_id])
    if best_analyses is None:
        raise ValueError("Calibration did not retain a valid recommended analysis.")
    positive_analysis, negative_analysis = best_analyses
    negative_used = bool(negative_refs)
    confidence = calibration_confidence(best_row, negative_used)
    negative_false_positive = negative_used and str(best_row.get("negativeSampleQC", "")) == "Reliable aggregation morphology"
    preset_name = str(params.get("presetName", "New camera setup")).strip() or "New camera setup"
    preset_json = camera_preset_json(
        preset_name,
        recommended_settings,
        confidence,
        negative_used,
        str(params.get("cameraSetupNotes", "")),
        str(params.get("presetUserNotes", "")),
    )
    result = {
        "calibrationMode": params.get("calibrationMode", ""),
        "calibrationInputMethod": params.get("calibrationInputMethod", "Load images from local folder path"),
        "positiveFolder": str(params.get("positiveFolder", "")),
        "negativeFolder": str(params.get("negativeFolder", "")),
        "positiveSource": positive_source,
        "negativeSource": negative_source,
        "negativeControlType": params.get("negativeControlType", "not provided"),
        "negativeControlUsed": negative_used,
        "negativeFalsePositiveDetected": negative_false_positive,
        "calibrationConfidence": confidence,
        "calibrationResults": valid,
        "recommendedSettings": recommended_settings,
        "recommendedPresetJson": preset_json,
        "recommendedPresetName": preset_name,
        "bestRow": best_row.to_dict(),
        "positiveAnalysis": positive_analysis,
        "negativeAnalysis": negative_analysis,
        "warnings": trial_warnings,
    }
    result["reportMd"] = calibration_report_markdown(result)
    return result


def representative_preview_positions(analysis: dict[str, Any] | None, negative_control: bool = False) -> list[tuple[str, int]]:
    """Choose early/start/late or first/middle/suspicious previews for calibration."""
    if not analysis:
        return []
    metrics = analysis.get("metrics", pd.DataFrame()).reset_index(drop=True)
    if metrics.empty:
        return []
    n_frames = len(metrics)
    if negative_control:
        signal = pd.to_numeric(metrics.get("startSignal_autoStart", pd.Series(np.nan, index=metrics.index)), errors="coerce")
        suspicious = int(signal.fillna(-np.inf).to_numpy().argmax()) if signal.notna().any() else n_frames // 2
        requested = [("First frame", 0), ("Middle frame", n_frames // 2), ("Most suspicious frame", suspicious)]
    else:
        included = metrics.get("usedForMainSummary", pd.Series(False, index=metrics.index)).fillna(False).astype(bool).to_numpy()
        start_positions = np.flatnonzero(included)
        start = int(start_positions[0]) if start_positions.size else 0
        requested = [("Early frame before start", max(0, start - 1)), ("Detected autoStartFrame", start), ("Late frame", n_frames - 1)]
    positions: list[tuple[str, int]] = []
    seen: set[int] = set()
    for label, position in requested:
        position = int(np.clip(position, 0, n_frames - 1))
        if position not in seen:
            positions.append((label, position))
            seen.add(position)
    return positions


def render_calibration_previews(title: str, analysis: dict[str, Any] | None, negative_control: bool = False) -> None:
    """Render representative raw/Wz/mask/boundary panels for a calibration sequence."""
    st.subheader(title)
    if not analysis or analysis.get("metrics", pd.DataFrame()).empty:
        st.info("No preview frames are available.")
        return
    metrics = analysis["metrics"].reset_index(drop=True)
    previews = analysis.get("previews", [])
    for label, position in representative_preview_positions(analysis, negative_control=negative_control):
        if position >= len(previews):
            continue
        row = metrics.iloc[position]
        preview = previews[position]
        st.markdown(f"**{label}: {row.get('filename', f'frame {position + 1}')}**")
        columns = st.columns(4)
        columns[0].image(preview["image"], caption="Raw grayscale crop", use_container_width=True)
        columns[1].image(preview.get("wz", preview["image"]), caption="Illumination-corrected Wz", use_container_width=True)
        columns[2].image(preview["mask"].astype(np.uint8) * 255, caption="Plasma gap mask", use_container_width=True)
        columns[3].image(preview["boundary_overlay"], caption="Gap boundary overlay", use_container_width=True)
        st.caption(
            f"Frame QC: {row.get('rbcQCLabel', row.get('rbcFrameQC', 'n/a'))} | "
            f"Included in trusted summary: {bool(row.get('usedInSummary', row.get('includedInMainSummary', row.get('usedForMainSummary', False))))}"
        )


def plot_calibration_scores(results_df: pd.DataFrame) -> plt.Figure:
    """Plot calibration score by candidate."""
    if results_df.empty:
        return empty_batch_figure("No calibration candidates are available.")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(pd.to_numeric(results_df["candidateID"], errors="coerce"), pd.to_numeric(results_df["calibrationScore"], errors="coerce"), marker="o", linewidth=1.2, color="#2F6F8F")
    ax.set_title("Calibration score by candidate parameter set")
    ax.set_xlabel("Candidate ID")
    ax.set_ylabel("Calibration score")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_calibration_tradeoff(results_df: pd.DataFrame) -> plt.Figure:
    """Plot positive detection reward against control false-positive penalty."""
    if results_df.empty:
        return empty_batch_figure("No calibration trade-off data are available.")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        pd.to_numeric(results_df["negativeFalsePositivePenalty"], errors="coerce"),
        pd.to_numeric(results_df["positiveDetectionScore"], errors="coerce"),
        color="#2F6F8F",
        alpha=0.75,
    )
    ax.set_title("Positive detection vs negative false-positive penalty")
    ax.set_xlabel("Negative false-positive penalty")
    ax.set_ylabel("Positive detection score")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_calibration_top_starts(results_df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Plot detected positive start frames for top calibration candidates."""
    if results_df.empty:
        return empty_batch_figure("No detected calibration starts are available.")
    top = results_df.sort_values("calibrationScore", ascending=False).head(top_n).copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(top["candidateID"].astype(str), pd.to_numeric(top["positiveAutoStartFrame"], errors="coerce"), color="#B07D2C")
    ax.set_title("Auto start frame for top candidates")
    ax.set_xlabel("Candidate ID")
    ax.set_ylabel("Positive autoStartFrame")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_calibration_user_start(results_df: pd.DataFrame) -> plt.Figure:
    """Plot detected candidate starts against an optional user-marked visual start."""
    if results_df.empty or "userStartFrame" not in results_df.columns:
        return empty_batch_figure("No user-marked visual start frame was supplied.")
    user = pd.to_numeric(results_df["userStartFrame"], errors="coerce")
    detected = pd.to_numeric(results_df["positiveAutoStartFrame"], errors="coerce")
    finite = user.notna() & detected.notna()
    if not finite.any():
        return empty_batch_figure("No user-marked visual start comparison is available.")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(user[finite], detected[finite], color="#2F6F8F", alpha=0.75)
    low = float(min(user[finite].min(), detected[finite].min()))
    high = float(max(user[finite].max(), detected[finite].max()))
    ax.plot([low, high], [low, high], color="#777777", linestyle="--")
    ax.set_title("Detected start vs user-marked visual start")
    ax.set_xlabel("User-marked start frame")
    ax.set_ylabel("Detected start frame")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def parse_validation_folders(folder_text: str) -> list[str]:
    """Parse newline-separated additional sample folders."""
    return [line.strip().strip('"') for line in str(folder_text).splitlines() if line.strip()]


def run_preset_validation(sample_folders: list[str], settings: dict[str, Any], detect_matlab_csv: bool = True, image_uploads: dict | None = None) -> dict[str, Any]:
    """Validate the same camera preset on uploaded or desktop RBC samples."""
    rows: list[dict[str, Any]] = []
    analyses: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    for folder_text in sample_folders:
        if WEB_DEPLOYMENT:
            refs = make_image_file_refs((image_uploads or {}).get(folder_text, []))
            image_folder = f"Uploaded {folder_text}"
            warnings = []
            sample_name = folder_text
            detect_matlab_csv = False
        else:
            refs, image_folder, warnings = calibration_folder_refs(folder_text)
            folder_path = Path(image_folder) if image_folder else Path(folder_text)
            sample_folder = folder_path.parent if folder_path.name.lower() == "images" else folder_path
            sample_name = sample_folder.name
        notes.extend([f"{sample_name}: {warning}" for warning in warnings])
        if not refs:
            rows.append({"sampleName": sample_name, "status": "failed", "errorMessage": "No supported images found."})
            continue
        try:
            analysis_params = calibration_analysis_params(settings, image_folder, len(refs), "Preset validation")
            analysis = analyze_standard_dataset(refs, "RBC Gap Mode", analysis_params)
            analyses[sample_name] = analysis
            metrics = analysis["metrics"]
            summary_rows = rbc_main_summary_df(metrics)
            qc_summary = analysis.get("params", {}).get("rbc_qc_summary", {})
            row: dict[str, Any] = {
                "sampleName": sample_name,
                "status": "ok",
                "nImages": len(refs),
                "autoStartFrame": qc_summary.get("autoStartFrame", np.nan),
                "sampleQC": qc_summary.get("sampleQC", "n/a"),
                "scoresConfidence": qc_summary.get("scoresConfidence", "n/a"),
                "reliableFramePercent": qc_summary.get("reliableFramePercent", np.nan),
                "median_SASI_0_100": safe_column_median(summary_rows, "SASI_0_100"),
                "median_AggIndex_0_100": safe_column_median(summary_rows, "AggIndex_0_100"),
                "matlabComparisonStatus": "not requested",
                "matlabMatchedRows": 0,
            }
            if detect_matlab_csv:
                csv_candidates = sorted(sample_folder.glob("agg_features_*_GAPS.csv"))
                if csv_candidates:
                    matlab_df = pd.read_csv(csv_candidates[0])
                    matlab_df, _sample_info = filter_matlab_reference_for_sample(matlab_df, sample_name)
                    matlab_df, _isana_used = matlab_isana_filtered_df(matlab_df)
                    comparison = compare_matlab_metrics(visible_metrics_df(metrics), matlab_df, AUTO_MATCH)
                    row["matlabComparisonStatus"] = "ok"
                    row["matlabMatchedRows"] = comparison.get("matchedRows", 0)
                    row["meanPearsonMainMetrics"] = batch_matlab_summary_columns(comparison).get("meanPearsonMainMetrics", np.nan)
                else:
                    row["matlabComparisonStatus"] = "no MATLAB CSV"
            rows.append(row)
        except Exception as exc:
            rows.append({"sampleName": sample_name, "status": "failed", "errorMessage": str(exc)})
            notes.append(f"{sample_name}: preset validation failed ({exc}).")
    return {"summary": pd.DataFrame(rows), "analyses": analyses, "warnings": notes}


def render_preset_validation_panel(settings: dict[str, Any], key_prefix: str) -> None:
    """Render optional additional-sample preset validation."""
    st.subheader("Validate preset on additional samples")
    uploaded_samples = None
    detect_matlab = False
    if WEB_DEPLOYMENT:
        uploaded_samples, _archive_id = web_archive_controls(st, f"{key_prefix}_validation")
        folders = list(uploaded_samples)
    else:
        folder_text = st.text_area("Additional sample folders, one per line", value="", key=f"{key_prefix}_validation_folders")
        folders = parse_validation_folders(folder_text)
        detect_matlab = st.checkbox("Compare with local MATLAB CSV when available", value=True, key=f"{key_prefix}_validation_matlab")
    if st.button("Run Preset Validation", key=f"{key_prefix}_run_validation"):
        if not folders:
            st.warning("Upload a ZIP containing at least one sample." if WEB_DEPLOYMENT else "Add at least one local sample folder.")
        else:
            with st.spinner("Validating preset on additional samples..."):
                st.session_state[f"{key_prefix}_preset_validation"] = run_preset_validation(folders, settings, detect_matlab_csv=detect_matlab, image_uploads=uploaded_samples)
    validation = st.session_state.get(f"{key_prefix}_preset_validation")
    if not validation:
        return
    summary_df = validation.get("summary", pd.DataFrame())
    validation_export_key = f"{key_prefix}_preset_validation_summary_df"
    summary_df = store_export_dataframe(validation_export_key, summary_df)
    if summary_df.empty:
        st.warning("Preset validation did not produce summary rows.")
        render_csv_download(
            "Download preset validation summary CSV",
            validation_export_key,
            "MicroStructAI_preset_validation_summary.csv",
            f"{key_prefix}_download_preset_validation_summary_empty",
        )
        return
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    render_csv_download(
        "Download preset validation summary CSV",
        validation_export_key,
        "MicroStructAI_preset_validation_summary.csv",
        f"{key_prefix}_download_preset_validation_summary",
    )
    analyses = validation.get("analyses", {})
    if analyses:
        first_name = next(iter(analyses))
        with st.expander(f"Overlay preview: {first_name}", expanded=False):
            render_calibration_previews(first_name, analyses[first_name])


def render_camera_calibration(params: dict[str, Any]) -> None:
    """Render the separate RBC camera/setup calibration workflow."""
    st.header("Auto-Tuning / Camera Calibration")
    st.write(
        "Recommend a setup-specific RBC Gap Mode preset for a camera, microscope, lighting condition, magnification, focus setting, "
        "bit depth, and sample-preparation workflow. Calibration recommendations should always be checked visually using overlays."
    )
    st.warning("Auto-tuning recommends a research-use preset. It does not establish absolute segmentation truth.")
    calibration_mode = str(params.get("calibrationMode", CALIBRATION_MODES[0]))
    if calibration_mode == "Validate existing preset":
        payload = params.get("validatePresetPayload")
        if not payload:
            st.info("Load a camera preset JSON in the sidebar, then validate it on additional samples.")
            return
        try:
            preset = load_camera_preset_json(payload)
        except Exception as exc:
            st.error(f"Could not load camera preset: {exc}")
            return
        st.success(f"Loaded preset: {preset.get('presetName', 'Uploaded camera preset')}")
        st.json(preset)
        render_preset_validation_panel(preset_settings(preset), "loaded_camera_preset")
        return

    positive_refs, positive_source, positive_warnings = calibration_input_refs(params, "positive")
    negative_refs, negative_source, negative_warnings = calibration_input_refs(params, "negative")
    for warning in positive_warnings + negative_warnings:
        st.warning(warning)
    columns = st.columns(2)
    columns[0].metric("Positive images detected", len(positive_refs))
    columns[1].metric("Negative/control images detected", len(negative_refs))
    if calibration_mode == "Quick calibration from one positive RBC sequence":
        st.warning("Calibration confidence is limited because no negative/no-aggregation control was provided.")

    if st.button("Run Auto-Tuning Calibration", type="primary", disabled=not bool(positive_refs)):
        with st.spinner("Testing camera/setup calibration candidates..."):
            try:
                st.session_state["camera_calibration_result"] = run_camera_calibration(params)
            except Exception as exc:
                st.error(f"Calibration failed: {exc}")

    result = st.session_state.get("camera_calibration_result")
    if not result:
        return
    if result.get("positiveSource") != positive_source or result.get("negativeSource", "") != negative_source:
        st.warning("Calibration image inputs changed after the last run. Run Auto-Tuning Calibration again to refresh recommendations.")

    results_df = result.get("calibrationResults", pd.DataFrame())
    results_df = store_export_dataframe("calibration_results_df", results_df)
    settings = result.get("recommendedSettings", {})
    best = result.get("bestRow", {})
    tabs = st.tabs(["Recommended Preset", "Candidate Results", "Overlay Previews", "Diagnostics", "Exports", "Validate Preset"])
    with tabs[0]:
        st.subheader("Recommended Settings")
        confidence = str(result.get("calibrationConfidence", "low"))
        st.metric("Calibration confidence", confidence)
        score_columns = st.columns(5)
        score_columns[0].metric("Calibration score", format_metric_value(float(best.get("calibrationScore", np.nan))))
        score_columns[1].metric("Positive score", format_metric_value(float(best.get("positiveDetectionScore", np.nan))))
        score_columns[2].metric("Negative penalty", format_metric_value(float(best.get("negativeFalsePositivePenalty", np.nan))))
        score_columns[3].metric("Instability penalty", format_metric_value(float(best.get("instabilityPenalty", np.nan))))
        score_columns[4].metric("Start error penalty", format_metric_value(float(best.get("startFrameErrorPenalty", np.nan))))
        recommended_display_keys = TUNABLE_PARAMETER_KEYS + ["enableMaxGapAreaFilter", "maxGapAreaFrac"]
        st.dataframe(pd.DataFrame([{key: settings.get(key) for key in recommended_display_keys}]), use_container_width=True, hide_index=True)
        if result.get("negativeFalsePositiveDetected", False):
            st.error("This preset is not safe for this setup because it produced a false-positive aggregation result on the negative/control sequence.")
        elif not result.get("negativeControlUsed", False):
            st.warning("Calibration confidence is limited because no negative/no-aggregation control was provided.")
        st.info("The recommended preset should be checked visually using overlays before interpreting SASI/AggIndex.")
    with tabs[1]:
        st.subheader("Calibration Candidate Results")
        st.dataframe(results_df, use_container_width=True, hide_index=True)
    with tabs[2]:
        render_calibration_previews("Positive RBC sequence", result.get("positiveAnalysis"))
        if result.get("negativeAnalysis"):
            render_calibration_previews("Negative/control sequence", result.get("negativeAnalysis"), negative_control=True)
    with tabs[3]:
        st.pyplot(plot_calibration_scores(results_df), clear_figure=True)
        st.pyplot(plot_calibration_tradeoff(results_df), clear_figure=True)
        st.pyplot(plot_calibration_top_starts(results_df), clear_figure=True)
        positive_metrics = result.get("positiveAnalysis", {}).get("metrics", pd.DataFrame())
        st.pyplot(plot_auto_start_diagnostics(positive_metrics), clear_figure=True)
        if pd.to_numeric(results_df.get("userStartFrame", pd.Series(dtype=float)), errors="coerce").notna().any():
            st.pyplot(plot_calibration_user_start(results_df), clear_figure=True)
    with tabs[4]:
        st.subheader("Calibration Exports")
        render_csv_download("Download calibration results CSV", "calibration_results_df", "MicroStructAI_calibration_results.csv", "download_calibration_results_csv")
        render_text_download(
            "Download recommended camera preset JSON",
            str(result.get("recommendedPresetJson", "")),
            "MicroStructAI_recommended_camera_preset.json",
            "application/json",
            "download_recommended_camera_preset_json",
        )
        render_text_download(
            "Download named camera preset JSON",
            str(result.get("recommendedPresetJson", "")),
            f"MicroStructAI_camera_preset_{safe_filename_token(str(result.get('recommendedPresetName', 'camera_setup')))}.json",
            "application/json",
            "download_named_camera_preset_json",
        )
        render_text_download(
            "Download calibration report markdown",
            str(result.get("reportMd", "")),
            "MicroStructAI_calibration_report.md",
            "text/markdown",
            "download_calibration_report_md",
        )
        render_export_diagnostics(
            {
                "calibration_results": results_df,
                "recommended_camera_preset": str(result.get("recommendedPresetJson", "")),
                "calibration_report": str(result.get("reportMd", "")),
            }
        )
    with tabs[5]:
        render_preset_validation_panel(settings, "recommended_camera_preset")


BATCH_REPORT_UI_NOTE_LIMIT = 50


def batch_report_markdown(
    summary_df: pd.DataFrame,
    frame_metrics_df: pd.DataFrame,
    matlab_summary_df: pd.DataFrame,
    warnings: list[str],
    batch_input_mode: str = "Images + optional MATLAB CSV comparison",
    batch_validation_mode: str = "RBC Gap Mode",
    processing_note_limit: int | None = None,
) -> str:
    """Generate a batch validation report, optionally shortened for UI preview only."""
    lines: list[str] = [f"# MicroStructAI Batch Validation Report - {batch_validation_mode}", ""]
    n_samples = len(summary_df)
    total_images = int(pd.to_numeric(summary_df.get("nImagesLoaded", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not summary_df.empty else 0
    samples_with_matlab = int((summary_df.get("matlabCSVFound", pd.Series(dtype=bool)) == True).sum()) if "matlabCSVFound" in summary_df.columns else 0
    lines += [
        "## Batch Dataset Summary",
        f"- Batch validation analysis mode: {batch_validation_mode}",
        f"- Batch input type: {batch_input_mode}",
        f"- Samples analysed: {n_samples}",
        f"- Total images discovered: {total_images}",
        f"- Samples with MATLAB CSV: {samples_with_matlab}",
        f"- Samples without MATLAB CSV: {max(0, n_samples - samples_with_matlab)}",
        f"- Combined per-frame rows: {len(frame_metrics_df)}",
        "",
    ]
    if "intensityCorrectionMode" in summary_df.columns:
        correction_applied = int(summary_df.get("intensityCorrectionApplied", pd.Series(False, index=summary_df.index)).fillna(False).astype(bool).sum())
        correction_needed = int(summary_df.get("intensityCorrectionNeeded", pd.Series(False, index=summary_df.index)).fillna(False).astype(bool).sum())
        stable_illumination = int((~summary_df.get("intensityCorrectionNeeded", pd.Series(False, index=summary_df.index)).fillna(False).astype(bool)).sum())
        sensitive_samples = int((pd.to_numeric(summary_df.get("nRawGainSensitiveFrames", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum())
        lines += [
            "## Camera Intensity-Drop Correction Summary",
            f"- Samples where correction was applied: {correction_applied}",
            f"- Samples where correction was needed/detected: {correction_needed}",
            f"- Samples with stable illumination/no correction needed: {stable_illumination}",
            f"- Samples with intensity-correction-sensitive masks: {sensitive_samples}",
            f"- Median mean-intensity temporal CV: {format_metric_value(metric_median(summary_df, 'meanIntensityTemporalCV'), 4)}",
            f"- Median illumination gain: {format_metric_value(metric_median(summary_df, 'medianIlluminationGain'), 4)}",
            "",
        ]

    if batch_validation_mode == "RBC Gap Mode":
        qc_counts = summary_df.get("sampleQC", pd.Series(dtype=str)).fillna("n/a").astype(str).value_counts()
        reliable_count = int(qc_counts.get("Reliable aggregation morphology", 0))
        weak_count = int(qc_counts.get("Weak/pre-gap structural texture", 0))
        no_reliable_count = int(qc_counts.get("No reliable plasma-gap aggregation detected", 0))
        median_reliable = metric_median(summary_df, "reliableFramePercent")
        low_conf_count = int((summary_df.get("scoresConfidence", pd.Series(dtype=str)).astype(str).str.lower() == "low").sum()) if "scoresConfidence" in summary_df.columns else 0
        fallback_samples = int((pd.to_numeric(summary_df.get("nStartupFallbackFrames", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum())
        fallback_frames = int(pd.to_numeric(summary_df.get("nStartupFallbackFrames", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        conservative_samples = int((pd.to_numeric(summary_df.get("nConservativeReliableFrames", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum())
        pod_processing_samples = int(summary_df.get("podProcessingActive", pd.Series(False, index=summary_df.index)).fillna(False).astype(bool).sum()) if "podProcessingActive" in summary_df.columns else 0
        pod_sensitive_frames = int(pd.to_numeric(summary_df.get("nPODSensitivityFlags", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        lines += [
            "## Batch QC Summary",
            f"- Reliable samples: {reliable_count}",
            f"- Weak/pre-gap samples: {weak_count}",
            f"- No-reliable-aggregation samples: {no_reliable_count}",
            f"- Median reliable frame percentage: {format_metric_value(median_reliable, 1)}%",
            f"- Samples with low confidence scores: {low_conf_count}",
            f"- Samples with startup/transient fallback frames: {fallback_samples}",
            f"- Startup/transient fallback frames: {fallback_frames}",
            f"- Samples with conservative reliable frames: {conservative_samples}",
            f"- Samples with active POD processing/diagnostics: {pod_processing_samples}",
            f"- POD-sensitive/uncertain frames: {pod_sensitive_frames}",
            "",
        ]
        if conservative_samples or pod_processing_samples:
            lines += [
                "## Conservative RBC / POD-QC Batch Summary",
                f"- Median conservative reliable frame fraction: {format_metric_value(metric_median(summary_df, 'fracConservativeReliableFrames'), 3)}",
                f"- Median raw CV: {format_metric_value(metric_median(summary_df, 'medianRawCV'), 4)}",
                f"- Median corrected CV: {format_metric_value(metric_median(summary_df, 'medianCorrectedCV'), 4)}",
                "- In Conservative RBC Gap Mode, the corrected no-POD mask is the main plasma-gap mask.",
                "- POD-removed masks are only calculated when POD processing/diagnostics are active.",
                "",
            ]
            if pod_processing_samples:
                lines += [
                    f"- Median podAreaDelta: {format_metric_value(metric_median(summary_df, 'medianPodAreaDelta'), 4)}",
                    f"- Median podMode1Energy: {format_metric_value(metric_median(summary_df, 'medianPodMode1Energy'), 4)}",
                    "",
                ]
        if fallback_frames:
            lines += [
                "## Startup/Transient Fallback",
                "- Startup/transient fallback metrics are early-frame QC texture diagnostics calculated from raw/lightly corrected images.",
                "- They do not replace plasma-gap masks and are not used for SASI, AggIndex, gapMeanDiam, gapNtotal, or normal RBC morphology scores.",
                f"- Median fallback raw CV: {format_metric_value(metric_median(summary_df, 'medianStartupRawCV'), 4)}",
                f"- Median fallback gradient energy: {format_metric_value(metric_median(summary_df, 'medianStartupGradientEnergy'), 6)}",
                "",
            ]
        syllectogram_enabled = (
            summary_df.get("enableSASISyllectogramResearchOutput", pd.Series(False, index=summary_df.index))
            .fillna(False)
            .astype(bool)
        )
        if syllectogram_enabled.any():
            enabled_summary = summary_df.loc[syllectogram_enabled].copy()
            lines += [
                "## Optional SASI Pseudo-Syllectogram Research Output",
                "- A SASI-based pseudo-syllectogram was fitted with a double-exponential curve and converted to a RheoScan-like area-ratio AI.",
                "- This SASI-derived AI-like value is an image-derived RheoScan-like validation metric and should not be reported as the official RheoScan AI.",
                f"- Samples with optional research output enabled: {len(enabled_summary)}",
                f"- Median SASI_syllectogram_AI_like: {format_metric_value(metric_median(enabled_summary, 'SASI_syllectogram_AI_like'), 3)}",
                f"- Median SASI_syllectogram_AMP_like: {format_metric_value(metric_median(enabled_summary, 'SASI_syllectogram_AMP_like'), 3)}",
                f"- Median SASI_syllectogram_T_half_like: {format_metric_value(metric_median(enabled_summary, 'SASI_syllectogram_T_half_like'), 3)}",
                "",
            ]
    else:
        low_resolution_high = int(summary_df.get("lowResolutionRisk", pd.Series(dtype=str)).fillna("").astype(str).str.contains("high", case=False).sum())
        low_confidence = int(summary_df.get("segmentationConfidence", pd.Series(dtype=str)).fillna("").astype(str).str.contains("low", case=False).sum())
        fallback_samples = int((pd.to_numeric(summary_df.get("nStartupFallbackFrames", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum())
        fallback_frames = int(pd.to_numeric(summary_df.get("nStartupFallbackFrames", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        lines += [
            "## Batch Generic Structure Summary",
            f"- Samples with high low-resolution risk: {low_resolution_high}",
            f"- Samples with low segmentation confidence: {low_confidence}",
            f"- Samples with startup/transient fallback frames: {fallback_samples}",
            f"- Startup/transient fallback frames: {fallback_frames}",
            f"- Median areaFrac: {format_metric_value(metric_median(summary_df, 'median_areaFrac'), 4)}",
            f"- Median objectCount: {format_metric_value(metric_median(summary_df, 'median_objectCount'), 3)}",
            f"- Median meanDiam_px: {format_metric_value(metric_median(summary_df, 'median_meanDiam_px'), 3)}",
            f"- Median correlationLength_px: {format_metric_value(metric_median(summary_df, 'median_correlationLength_px'), 3)}",
            "",
        ]
        if fallback_frames:
            lines += [
                "## Startup/Transient Fallback",
                "- Startup/transient fallback metrics are early-frame QC texture diagnostics calculated from raw/lightly corrected images.",
                "- They do not replace Object Mode masks and are not used for normal object count, size, or exploratory gSASI/gAggIndex scores.",
                f"- Median fallback raw CV: {format_metric_value(metric_median(summary_df, 'medianStartupRawCV'), 4)}",
                f"- Median fallback gradient energy: {format_metric_value(metric_median(summary_df, 'medianStartupGradientEnergy'), 6)}",
                "",
            ]

    syl_summary_lines: list[str] = []
    for prefix, label in [
        ("meanI_syl", "Optical mean intensity"),
        ("cv_syl", "Optical CV"),
        ("maskSASI_syl", "Mask-SASI"),
        ("maskGapScore_syl", "Mask-GapScore"),
        ("maskAggIndex_syl", "Mask-AggIndex"),
        ("objectStructure_syl", "ObjectStructure"),
        ("objectAggregation_syl", "ObjectAggregation"),
        ("gSASI_syl", "gSASI"),
    ]:
        amp_col = f"{prefix}_amplitude"
        auc_col = f"{prefix}_AUC"
        t50_col = f"{prefix}_T50"
        if amp_col in summary_df.columns and np.isfinite(metric_median(summary_df, amp_col)):
            syl_summary_lines.extend(
                [
                    f"- Median {label} syllectogram amplitude: {format_metric_value(metric_median(summary_df, amp_col), 4)}",
                    f"- Median {label} syllectogram AUC: {format_metric_value(metric_median(summary_df, auc_col), 4)}",
                    f"- Median {label} syllectogram T50: {format_metric_value(metric_median(summary_df, t50_col), 4)}",
                ]
            )
    if syl_summary_lines:
        lines += [
            "## Normalized Syllectogram Summary",
            "- Optical mean-intensity syllectograms use raw/cropped grayscale and the global gain-corrected grayscale mean, not spatial flat-field/no-POD mean drift.",
            "- Optical CV syllectograms use grayscale frame heterogeneity, not mask area.",
            "- Mask-SASI/Mask-GapScore/Mask-AggIndex syllectograms use final selected binary mask metrics and mask-based Xi.",
            "- Object structure/aggregation syllectograms use final selected object mask metrics and are not RBC biomarkers.",
            *syl_summary_lines,
            "",
        ]

    if batch_validation_mode == "RBC Gap Mode" and not matlab_summary_df.empty:
        mean_main = metric_mean(summary_df, "meanPearsonMainMetrics")
        mean_scores = metric_mean(summary_df, "meanPearsonScores")
        best = summary_df.sort_values("meanPearsonMainMetrics", ascending=False, na_position="last").head(3)
        weakest = summary_df.sort_values("meanPearsonMainMetrics", ascending=True, na_position="last").head(3)
        lines += [
            "## MATLAB Validation Summary",
            f"- Mean Pearson across main morphology metrics: {format_metric_value(mean_main, 3)}",
            f"- Mean Pearson across score metrics: {format_metric_value(mean_scores, 3)}",
            "- Best matching samples: " + (", ".join(best["sampleName"].astype(str).tolist()) if not best.empty else "n/a"),
            "- Weakest matching samples: " + (", ".join(weakest["sampleName"].astype(str).tolist()) if not weakest.empty else "n/a"),
            "- gapAreaSlope is treated as a sensitive/caution metric because small component splitting and log-bin changes can reduce frame-level correlation.",
            "- PTI partly depends on gapAreaSlope; PTI absolute differences may partly reflect gapAreaSlope sensitivity.",
            "",
        ]
    elif batch_validation_mode == "RBC Gap Mode":
        lines += ["## MATLAB Validation Summary", "- No MATLAB comparison CSVs were available or no comparisons produced finite paired metrics.", ""]

    if batch_validation_mode == "RBC Gap Mode":
        lines += [
            "## Main Findings",
            "- The app provides per-sample RBC morphology summaries using QC-included frames and preserves all raw per-frame outputs for transparency.",
            "- When MATLAB CSVs are available, batch validation reports per-sample trend agreement for morphology and score metrics.",
            "- Component-count and size-distribution metrics can be more sensitive in weak or fragmented samples.",
            "",
            "## Warnings",
            "- RBC scores are research-use structural indices only and are not clinical diagnostic outputs.",
            "- QC labels and overlays should be reviewed before interpreting RBC aggregation scores.",
            "- Low-confidence samples should not be interpreted as mature aggregation morphology.",
        ]
    else:
        lines += [
            "## Main Findings",
            "- Object Mode metrics are generic image-structure descriptors. Optional gSASI/gAggIndex are exploratory composite scores and are not RBC aggregation scores.",
            "- Generic component-count and size-distribution metrics can be sensitive to segmentation settings and image resolution.",
            *(
                [
                    "- Enhanced Object Mode uses robust bright/dark structure segmentation to quantify generic morphology. "
                    "The detected mask does not represent RBC plasma gaps."
                ]
                if batch_validation_mode == "Enhanced Object Mode"
                else []
            ),
            *(
                [
                    "- Conservative/POD Object modes use no-POD/POD generic object masks for structural morphology and QC diagnostics. "
                    "The selected final mask drives the batch metrics and exports."
                ]
                if batch_validation_mode in {"Conservative Object Mode", "Experimental POD-removed Object Mode"}
                else []
            ),
            "",
            "## Warnings",
            "- Optional gSASI/gAggIndex are exploratory image-structure indices, not validated biological or clinical outputs.",
            "- Do not compare RBC SASI/AggIndex directly with Object Mode gSASI/gAggIndex. They are different score types.",
            "- Compare samples only when imaging conditions and segmentation settings are equivalent.",
        ]
    if warnings:
        lines += ["", "## Processing Notes"]
        notes_to_show = warnings if processing_note_limit is None else warnings[: max(0, int(processing_note_limit))]
        lines += [f"- {warning}" for warning in notes_to_show]
        if processing_note_limit is not None and len(notes_to_show) < len(warnings):
            lines.append(f"- ... ({len(warnings) - len(notes_to_show)} more notes shown only in UI preview; TXT/MD exports include all processing notes.)")
    return "\n".join(lines)


def run_batch_validation(sample_table: pd.DataFrame, params: dict[str, Any], image_uploads: dict | None = None) -> dict[str, Any]:
    """Run the shared pipeline on uploaded samples or desktop folders."""
    selected = sample_table.loc[sample_table["include"].fillna(False).astype(bool)].copy() if "include" in sample_table.columns else sample_table.copy()
    batch_mode = str(params.get("batch_validation_mode", "RBC Gap Mode"))
    is_rbc_batch = batch_mode == "RBC Gap Mode"
    if batch_mode == "Ground-Truth Mask Mode":
        raise ValueError("Ground-Truth Mask Mode batch validation is not available yet.")
    use_matlab_csv = bool(is_rbc_batch and params.get("batch_use_matlab_csv", True))
    uploaded_matlab_csv_bytes = params.get("batch_uploaded_matlab_csv_bytes", {}) or {}
    summary_rows: list[dict[str, Any]] = []
    frame_tables: list[pd.DataFrame] = []
    matlab_summary_tables: list[pd.DataFrame] = []
    paired_tables: list[pd.DataFrame] = []
    notes: list[str] = []

    progress = st.progress(0)
    status_box = st.empty()
    total = len(selected)
    success_count = 0
    failure_count = 0

    for position, (_, sample) in enumerate(selected.iterrows(), start=1):
        sample_name = str(sample.get("sampleName", f"sample_{position}"))
        status_box.info(f"Processing {position} of {total}: {sample_name}")
        try:
            image_folder = str(sample.get("imageFolder", "")).strip()
            if WEB_DEPLOYMENT:
                image_paths = (image_uploads or {}).get(image_folder, [])
                image_warnings = []
            else:
                image_paths, image_warnings = list_local_image_files(image_folder)
            if image_warnings:
                notes.extend([f"{sample_name}: {warning}" for warning in image_warnings])
            if not image_paths:
                raise ValueError("No supported image files were found.")

            refs = make_image_file_refs(image_paths)
            sample_for_summary = sample.copy()
            sample_for_summary["nImages"] = len(refs)
            sample_params = dict(params)
            sample_params.update(
                {
                    "input_method": "Uploaded sample ZIP" if WEB_DEPLOYMENT else "Local folder path",
                    "image_folder_path": image_folder,
                    "files_loaded_before_filtering": len(refs),
                    "selected_image_count": len(refs),
                    "pre_analysis_warnings": [],
                    "batch_generic_metrics_enabled": not is_rbc_batch,
                }
            )
            analysis_mode = "RBC Gap Mode" if is_rbc_batch else "Object Mode"
            analysis = analyze_standard_dataset(refs, analysis_mode, sample_params)
            metrics_df = visible_metrics_df(analysis["metrics"]).copy()
            metadata = {
                "sampleName": sample_name,
                "donor": sample.get("donor", ""),
                "flowGroup": sample.get("flowGroup", ""),
                "stentCondition": sample.get("stentCondition", ""),
                "batchValidationMode": batch_mode,
            }
            for col, value in reversed(list(metadata.items())):
                metrics_df.insert(0, col, value)
            if analysis.get("warnings"):
                notes.extend([f"{sample_name}: {warning}" for warning in analysis["warnings"]])

            comparison_result: dict[str, Any] | None = None
            matlab_comparison_error = ""
            matlab_csv_path = str(sample.get("matlabCSVPath", "")).strip().strip('"')
            matlab_csv_selection = str(sample.get("matlabCSVSelection", "Auto-detected local CSV")).strip()
            uploaded_csv_name = matlab_csv_selection.removeprefix("Uploaded: ").strip() if matlab_csv_selection.startswith("Uploaded: ") else ""
            use_uploaded_csv = bool(uploaded_csv_name)
            skip_matlab_comparison = matlab_csv_selection == "Skip MATLAB comparison"
            matlab_found = use_matlab_csv and not skip_matlab_comparison and (use_uploaded_csv or bool(matlab_csv_path))
            if is_rbc_batch and matlab_found:
                try:
                    if use_uploaded_csv:
                        csv_payload = uploaded_matlab_csv_bytes.get(uploaded_csv_name)
                        if not csv_payload:
                            raise FileNotFoundError(f"Uploaded MATLAB CSV is no longer available in the app: {uploaded_csv_name}")
                        matlab_csv_path = f"Uploaded in app: {uploaded_csv_name}"
                        matlab_df = pd.read_csv(io.BytesIO(csv_payload))
                    else:
                        if WEB_DEPLOYMENT:
                            raise ValueError("Choose an uploaded MATLAB CSV for online comparison.")
                        selected_matlab_csv = Path(os.path.expandvars(matlab_csv_path)).expanduser()
                        if not selected_matlab_csv.exists() or not selected_matlab_csv.is_file():
                            raise FileNotFoundError(f"Selected MATLAB CSV does not exist: {selected_matlab_csv}")
                        if selected_matlab_csv.suffix.lower() != ".csv":
                            raise ValueError(f"Selected MATLAB reference file is not a CSV: {selected_matlab_csv}")
                        matlab_csv_path = str(selected_matlab_csv)
                        matlab_df = pd.read_csv(matlab_csv_path)
                    matlab_total_rows = int(len(matlab_df))
                    matlab_df, matlab_sample_info = filter_matlab_reference_for_sample(matlab_df, sample_name)
                    matlab_sample_rows = int(len(matlab_df))
                    if matlab_sample_info.get("matlabSampleFilterApplied", False):
                        notes.append(
                            f"{sample_name}: MATLAB reference filtered by "
                            f"{matlab_sample_info.get('sampleKeyColumn')}="
                            f"{matlab_sample_info.get('matlabSampleFilterValue')} "
                            f"({matlab_sample_rows} of {matlab_total_rows} source rows)."
                        )
                    matlab_for_compare, isana_used = matlab_isana_filtered_df(matlab_df)
                    comparison_result = compare_matlab_metrics(metrics_df, matlab_for_compare, AUTO_MATCH)
                    comparison_result.update(
                        {
                            "matlabTotalRows": matlab_total_rows,
                            "matlabSampleRows": matlab_sample_rows,
                            "matlabSampleKeyColumn": matlab_sample_info.get("sampleKeyColumn", ""),
                            "matlabSampleFilterApplied": bool(matlab_sample_info.get("matlabSampleFilterApplied", False)),
                            "matlabSampleFilterValue": matlab_sample_info.get("matlabSampleFilterValue", ""),
                            "matlabRowsCompared": int(len(matlab_for_compare)),
                            "isAnaFilterUsed": bool(isana_used),
                            "appAnalysedRows": int(len(metrics_df)),
                        }
                    )
                    comparison_summary = comparison_result.get("summary", pd.DataFrame()).copy()
                    if not comparison_summary.empty:
                        comparison_summary.insert(0, "sampleName", sample_name)
                        comparison_summary.insert(1, "donor", sample.get("donor", ""))
                        comparison_summary.insert(2, "flowGroup", sample.get("flowGroup", ""))
                        comparison_summary.insert(3, "stentCondition", sample.get("stentCondition", ""))
                        comparison_summary.insert(4, "matlabCSVPath", matlab_csv_path)
                        comparison_summary.insert(5, "matlabCSVSelection", matlab_csv_selection)
                        comparison_summary.insert(6, "matlabSampleFilterValue", matlab_sample_info.get("matlabSampleFilterValue", ""))
                        comparison_summary.insert(7, "matlabSampleRows", matlab_sample_rows)
                        comparison_summary["matchMethodUsed"] = comparison_result.get("matchMethodUsed", "")
                        comparison_summary["matchedRows"] = comparison_result.get("matchedRows", 0)
                        matlab_summary_tables.append(comparison_summary)

                    paired_df = paired_frame_export_df(comparison_result.get("long", pd.DataFrame()))
                    if not paired_df.empty:
                        for col, value in reversed(list(metadata.items())):
                            paired_df.insert(0, col, value)
                        paired_df["matlabCSVPath"] = matlab_csv_path
                        paired_df["matlabCSVSelection"] = matlab_csv_selection
                        paired_df["matlabSampleFilterValue"] = matlab_sample_info.get("matlabSampleFilterValue", "")
                        paired_df["matchMethodUsed"] = comparison_result.get("matchMethodUsed", "")
                        paired_tables.append(paired_df)
                    for warning in comparison_result.get("warnings", []):
                        notes.append(f"{sample_name}: {warning}")
                except Exception as exc:
                    matlab_comparison_error = str(exc)
                    notes.append(f"{sample_name}: MATLAB comparison failed ({exc}).")
                    comparison_result = None

            analysis_for_summary = dict(analysis)
            analysis_for_summary["metrics"] = metrics_df
            summary_row = (
                batch_sample_summary_row(sample_for_summary, analysis_for_summary, comparison_result)
                if is_rbc_batch
                else batch_object_sample_summary_row(sample_for_summary, analysis_for_summary, batch_mode)
            )
            summary_row["batchValidationMode"] = batch_mode
            if matlab_comparison_error:
                summary_row["matlabComparisonStatus"] = f"failed: {matlab_comparison_error}"
            elif not matlab_found:
                summary_row["matlabComparisonStatus"] = "image-only mode" if not use_matlab_csv else "no MATLAB CSV"
            sample_folder_path = Path(image_folder).parent if Path(image_folder).name.lower() == "images" else Path(image_folder)
            summary_row.update(
                {
                    "matlabCSVFound": bool(matlab_found),
                    "matlabCSVPath": matlab_csv_path,
                    "matlabCSVSelection": matlab_csv_selection,
                    "matlabSampleFilterApplied": bool(comparison_result.get("matlabSampleFilterApplied", False)) if comparison_result else False,
                    "matlabSampleFilterValue": comparison_result.get("matlabSampleFilterValue", "") if comparison_result else "",
                    "matlabSampleRows": comparison_result.get("matlabSampleRows", np.nan) if comparison_result else np.nan,
                    "sampleFolder": str(sample_folder_path),
                    "imageFolder": image_folder,
                }
            )
            summary_rows.append(summary_row)
            frame_tables.append(metrics_df)
            success_count += 1
        except Exception as exc:
            failure_count += 1
            notes.append(f"{sample_name}: batch sample failed ({exc}).")
            failed_row = (
                batch_failure_summary_row(sample, str(exc))
                if is_rbc_batch
                else batch_object_failure_summary_row(sample, batch_mode, str(exc))
            )
            failed_image_folder = str(sample.get("imageFolder", ""))
            failed_sample_folder = Path(failed_image_folder).parent if Path(failed_image_folder).name.lower() == "images" else Path(failed_image_folder)
            failed_row.update(
                {
                    "matlabCSVFound": bool(sample.get("matlabCSVFound", False)),
                    "matlabCSVPath": sample.get("matlabCSVPath", ""),
                    "matlabCSVSelection": sample.get("matlabCSVSelection", ""),
                    "sampleFolder": str(failed_sample_folder),
                    "imageFolder": failed_image_folder,
                }
            )
            summary_rows.append(failed_row)
        finally:
            progress.progress(position / total if total else 1.0)

    status_box.success(f"Batch finished: {success_count} succeeded, {failure_count} failed.")
    summary_df = pd.DataFrame(summary_rows)
    frame_metrics_df = pd.concat(frame_tables, ignore_index=True, sort=False) if frame_tables else pd.DataFrame()
    matlab_summary_df = pd.concat(matlab_summary_tables, ignore_index=True, sort=False) if matlab_summary_tables else pd.DataFrame()
    paired_df = pd.concat(paired_tables, ignore_index=True, sort=False) if paired_tables else pd.DataFrame()
    report_md = batch_report_markdown(
        summary_df,
        frame_metrics_df,
        matlab_summary_df,
        notes,
        batch_input_mode=str(params.get("batch_input_mode", "Images + optional MATLAB CSV comparison")),
        batch_validation_mode=batch_mode,
    )
    report_preview_md = batch_report_markdown(
        summary_df,
        frame_metrics_df,
        matlab_summary_df,
        notes,
        batch_input_mode=str(params.get("batch_input_mode", "Images + optional MATLAB CSV comparison")),
        batch_validation_mode=batch_mode,
        processing_note_limit=BATCH_REPORT_UI_NOTE_LIMIT,
    )
    return {
        "sample_summary": summary_df,
        "frame_metrics": frame_metrics_df,
        "matlab_summary": matlab_summary_df,
        "paired_comparison": paired_df,
        "report_md": report_md,
        "report_preview_md": report_preview_md,
        "processing_notes_full": list(notes),
        "batch_input_mode": str(params.get("batch_input_mode", "Images + optional MATLAB CSV comparison")),
        "batch_validation_mode": batch_mode,
        "warnings": notes,
        "success_count": success_count,
        "failure_count": failure_count,
    }


def matlab_legacy_pca_columns(df: pd.DataFrame) -> list[str]:
    """Return MATLAB legacy PCA columns, using Xi_interp_px when available."""
    xi_col = "Xi_interp_px" if "Xi_interp_px" in df.columns else "Xi_px"
    metrics = MATLAB_LEGACY_PCA_METRICS.copy()
    insert_at = metrics.index("aggLargestFrac")
    metrics.insert(insert_at, xi_col)
    return list(dict.fromkeys(metrics))


def build_pca_result(df: pd.DataFrame) -> PCAResult:
    """Render PCA metric controls and return the PCA result."""
    finite_candidates = finite_numeric_columns(df, min_finite=5)
    if "GT_areaFrac" in df.columns:
        pca_pool = GT_METRICS + [m for m in AUTO_METRICS if m in df.columns]
        default_pool = ["GT_areaFrac", "GT_nObjects", "GT_meanDiam", "GT_Xi_px", "GT_lacunarity16", "GT_skelDensity"]
    elif "gapAreaFrac" in df.columns:
        pca_pool = RBC_PCA_METRICS + [
            "gapMedianDiam_px",
            "gapLargestFrac",
            "skelDensity",
            "branchDensity",
            "endpointDensity",
            "aggOccFrac",
            "aggLargestFrac",
            "SASI_0_100",
            "GapScore_0_100",
            "PTI_0_100",
            "AggIndex_0_100",
        ] + [f"struct_{metric}" for metric in STRUCTURAL_PCA_METRICS] + [f"struct_trusted_{metric}" for metric in STRUCTURAL_PCA_METRICS]
        default_xi = "Xi_mask" if "Xi_mask" in finite_candidates else ("Xi_interp_px" if "Xi_interp_px" in finite_candidates else "Xi_px")
        default_pool = ["gapAreaFrac", "gapNtotal", "gapMeanDiam_px", default_xi, "lacunarity16", "LF_log10_LFoverHF", "SHI_tileCV"]
    elif "objectCount" in df.columns:
        pca_pool = CORE_METRICS + GENERIC_OBJECT_METRICS
        default_pool = ["areaFrac", "objectCount", "meanDiam_px", "correlationLength_px", "lacunarity16", "textureHeterogeneity_tileCV"]
        if "gSASI_0_100" in df.columns:
            default_pool.append("gSASI_0_100")
        if "gAggIndex_0_100" in df.columns:
            default_pool.append("gAggIndex_0_100")
    else:
        pca_pool = CORE_METRICS
        default_pool = PCA_DEFAULT_METRICS

    saved_options = st.session_state.setdefault("_pca_view_options", {})
    previous_preset = saved_options.get("preset", PCA_PRESETS[0])
    preset = st.selectbox("PCA preset", PCA_PRESETS,
                          index=PCA_PRESETS.index(previous_preset) if previous_preset in PCA_PRESETS else 0,
                          key="pca_preset_selection")
    saved_options["preset"] = preset
    if preset == "MATLAB legacy metric set":
        requested_metrics = matlab_legacy_pca_columns(df)
        available_requested = [metric for metric in requested_metrics if metric in df.columns]
        selected_metrics = [metric for metric in available_requested if metric in finite_candidates]
        missing_columns = [metric for metric in requested_metrics if metric not in df.columns]
        unavailable_finite = [metric for metric in available_requested if metric not in finite_candidates]
        if missing_columns:
            st.warning("Missing MATLAB legacy PCA columns: " + ", ".join(missing_columns))
        if unavailable_finite:
            st.info("Legacy PCA columns present but not usable for PCA yet: " + ", ".join(unavailable_finite))
        st.caption("MATLAB legacy PCA metrics used: " + (", ".join(selected_metrics) if selected_metrics else "none available"))
    elif preset == "QC-trusted structural research features":
        requested_metrics = [f"struct_trusted_{metric}" for metric in STRUCTURAL_PCA_TRUSTED_DEFAULTS]
        selected_metrics = [metric for metric in requested_metrics if metric in finite_candidates]
        unavailable = [metric for metric in requested_metrics if metric not in finite_candidates]
        if unavailable:
            st.info("Structural features not usable with the current trusted-frame support: " + ", ".join(unavailable))
        st.caption(
            "This preset uses QC-trusted mask-level morphology, local thickness, spatial heterogeneity and low-order topology. "
            "Review Robustness & QC before interpreting PCA; unstable network/contour features remain available only under Custom."
        )
    else:
        candidate_metrics = [c for c in finite_candidates if c in pca_pool]
        if preset == "Custom":
            default_metrics = candidate_metrics[: min(6, len(candidate_metrics))]
        else:
            default_metrics = [m for m in default_pool if m in candidate_metrics]
            if len(default_metrics) < 2:
                default_metrics = candidate_metrics[: min(6, len(candidate_metrics))]
        if preset in saved_options:
            default_metrics = [metric for metric in saved_options[preset] if metric in candidate_metrics]
        selected_metrics = st.multiselect("PCA metrics", candidate_metrics, default=default_metrics,
                                         key=f"pca_metrics_{preset}")
        saved_options[preset] = selected_metrics
        if preset == "V3 primitive metrics" and "gapAreaFrac" in df.columns:
            st.caption("V3 PCA defaults use primitive measurements only; composite scores remain available under Custom to avoid double-weighting their components.")
    return cached_pca(df, selected_metrics, run_pca)


def render_overview(mode: str, image_files: list, mask_files: list, params: dict[str, Any], analysis: dict[str, Any] | None) -> None:
    """Render dataset overview."""
    st.subheader(mode)
    st.write(MODE_NOTES.get(mode, "Structural image analysis mode."))
    files_loaded = int(params.get("files_loaded_before_filtering", len(image_files)))
    st.metric("Files loaded", files_loaded)
    st.metric("Images selected for analysis", len(image_files))
    if mode == "Ground-Truth Mask Mode":
        st.metric("Masks loaded", int(params.get("masks_loaded_before_filtering", len(mask_files))))
    if params.get("range_valid", False) and files_loaded:
        st.caption(
            f"Analysing images {params.get('range_start_display')} to {params.get('range_end_display')} "
            f"out of {files_loaded} loaded images."
        )

    if image_files:
        try:
            name, image = read_uploaded_image(image_files[0])
            st.image(image, caption=f"Example image: {name}", use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not preview first image: {exc}")

    if analysis and analysis["warnings"]:
        st.warning("\n".join(analysis["warnings"]))

    if mode == "RBC Gap Mode" and analysis and not analysis["metrics"].empty:
        df = analysis["metrics"]
        summary_df = rbc_main_summary_df(df)
        xi_summary_col = "Xi_interp_px" if np.isfinite(safe_column_mean(summary_df, "Xi_interp_px")) else "Xi_px"
        qc_summary = analysis.get("params", {}).get("rbc_qc_summary", {})
        low_conf_scores = qc_summary.get("scoresConfidence") == "low"
        st.info(
            "Early low-contrast frames may produce apparent gap masks after background subtraction and contrast normalization. "
            "Startup QC waits until the structural signal is stable enough before calculating the main aggregation summary. "
            "In Auto from first frame mode, the app can start at frame 1 if morphology is reliable from the beginning. "
            "In Auto with manual lower bound mode, the user-defined analysisStart_s acts as the earliest allowed start."
        )
        st.subheader("RBC Gap Mode Summary")
        card_values = [
            ("Images analysed", str(len(df))),
            ("Frames used for main summary", str(len(summary_df))),
            (("Low-conf SASI_0_100" if low_conf_scores else "Mean SASI_0_100"), format_metric_value(safe_column_mean(summary_df, "SASI_0_100"), 1)),
            (("Low-conf AggIndex_0_100" if low_conf_scores else "Mean AggIndex_0_100"), format_metric_value(safe_column_mean(summary_df, "AggIndex_0_100"), 1)),
            (f"Mean {xi_summary_col}", format_metric_value(safe_column_mean(summary_df, xi_summary_col), 2)),
            ("Mean gapMeanDiam_px", format_metric_value(safe_column_mean(summary_df, "gapMeanDiam_px"), 2)),
            ("Mean gapNtotal", format_metric_value(safe_column_mean(summary_df, "gapNtotal"), 1)),
        ]
        columns = st.columns(3)
        for idx, (label, value) in enumerate(card_values):
            columns[idx % 3].metric(label, value)
        if low_conf_scores:
            st.warning("Low-confidence calculated scores are shown for transparency only and should not be interpreted as mature aggregation scores.")

        st.subheader("RBC QC / No-Aggregation Check")
        stable_start_display = stable_start_frame_text(qc_summary)
        startup_stable_value = float(qc_summary.get("startupQCStableStartFrame", np.nan))
        startup_stable_display = str(int(round(startup_stable_value))) if np.isfinite(startup_stable_value) else "Not available"
        included_frames, excluded_frames = rbc_included_excluded_frame_numbers(df)
        isolated_frame_col = "originalImageIndex_1based" if "originalImageIndex_1based" in df.columns else "analysisIndex_1based"
        if isolated_frame_col in df.columns and "isolatedReliableFrame" in df.columns:
            isolated_frames = pd.to_numeric(df.loc[df["isolatedReliableFrame"].fillna(False).astype(bool), isolated_frame_col], errors="coerce").dropna().round().astype(int).tolist()
        else:
            isolated_frames = []
        if qc_summary.get("scoresInterpretableWithWarnings", False):
            scores_interpretable_text = "Yes, with QC warnings"
        elif qc_summary.get("scoresInterpretable", False):
            scores_interpretable_text = "Yes"
        else:
            scores_interpretable_text = "Low confidence"
        qc_cards = [
            ("Sample QC class", str(qc_summary.get("sampleQC", "n/a"))),
            ("RBC analysis method", str(analysis.get("params", {}).get("rbcAnalysisMethod", "Standard RBC Gap Mode"))),
            ("Reliable frame percentage", format_metric_value(float(qc_summary.get("reliableFramePercent", np.nan)), 1) + "%"),
            ("Auto/manual start frame", format_metric_value(float(qc_summary.get("autoStart_originalImageIndex", qc_summary.get("finalStartFrameUsed", np.nan))), 0)),
            ("Start detected", "Yes" if qc_summary.get("stableReliableWindowDetected", False) else "No"),
            ("Low-confidence frames", str(qc_summary.get("possibleLowConfidenceFrames", "n/a"))),
            ("QC low-confidence frames", str(qc_summary.get("lowConfidenceFrames", qc_summary.get("lowConfidenceFrameCount", "n/a")))),
            ("Low-confidence excluded", str(qc_summary.get("lowConfidenceFramesExcluded", "n/a"))),
            ("Reliable frames in summary", str(qc_summary.get("reliableFramesUsedForMainSummary", "n/a"))),
            ("Start-frame mode", str(qc_summary.get("startFrameMode", qc_summary.get("startupQCMode", "n/a")))),
            ("Auto-start method", str(qc_summary.get("autoStartMethod", "n/a"))),
            ("Start time_s", format_metric_value(float(qc_summary.get("autoStartTime_s", np.nan)), 2)),
            ("Final start frame", format_metric_value(float(qc_summary.get("finalStartFrameUsed", np.nan)), 0)),
            ("Min SHI_tileCV", format_metric_value(float(qc_summary.get("minSHI_tileCV_forReliable", np.nan)), 2)),
            ("Sparse morphology filter", str(qc_summary.get("applySparseMaskMorphologyFilter", analysis.get("params", {}).get("applySparseMaskMorphologyFilter", "n/a")))),
            ("Sparse-filtered frames", str(qc_summary.get("sparseMaskMorphologyFilteredFrames", qc_summary.get("sparseMaskMorphologyFilteredFrameCount", "n/a")))),
            ("Exclude pre-start frames", str(qc_summary.get("excludePreStartFramesFromTrustedAnalysis", "n/a"))),
            ("Exclude bad post-start QC", str(qc_summary.get("excludeBadPostStartQCFramesFromTrustedAnalysis", "n/a"))),
            ("Mean mask reliability", format_metric_value(float(qc_summary.get("meanMaskReliabilityScore", np.nan)), 1)),
            ("Trusted score frames", str(qc_summary.get("trustedForScoreFrames", qc_summary.get("trustedScoreFrameCount", "n/a")))),
            ("Pre-start frames detected", str(qc_summary.get("preStartFramesDetected", qc_summary.get("preStartFrameCount", "n/a")))),
            ("Pre-start frames excluded", str(qc_summary.get("beforeStartExcludedFrames", qc_summary.get("beforeStartExcludedFrameCount", "n/a")))),
            ("Post-start QC warning frames", str(qc_summary.get("postStartQCWarningFrames", qc_summary.get("postStartQCWarningFrameCount", qc_summary.get("warningMaskFrameCount", "n/a"))))),
            ("Post-start QC-excluded frames", str(qc_summary.get("postStartQCExcludedFrames", qc_summary.get("postStartQCExcludedFrameCount", "n/a")))),
            ("Hard bad-mask frames", str(qc_summary.get("hardBadMaskFrames", qc_summary.get("hardBadMaskFrameCount", "n/a")))),
            ("Huge-gap frames", str(qc_summary.get("hugeGapFrameCount", "n/a"))),
            ("Mask QC risk", str(qc_summary.get("sampleMaskQCRisk", "n/a"))),
            ("Adaptive repairs kept", f"{qc_summary.get('adaptiveRepairKeptCount', 'n/a')}/{qc_summary.get('adaptiveRepairTriggeredCount', 'n/a')}"),
            ("Scores confidence", str(qc_summary.get("scoresConfidence", "n/a"))),
            ("Scores interpretable", scores_interpretable_text),
        ]
        fallback_summary = analysis.get("params", {}).get("startup_transient_fallback_summary", {})
        if analysis.get("params", {}).get("enableStartupTransientFallback", False) or fallback_summary:
            qc_cards.extend(
                [
                    ("Startup fallback frames", str(fallback_summary.get("nStartupFallbackFrames", 0))),
                    ("Startup fallback fraction", format_metric_value(float(fallback_summary.get("fracStartupFallbackFrames", np.nan)), 3)),
                    ("Median startup raw CV", format_metric_value(float(fallback_summary.get("medianStartupRawCV", np.nan)), 3)),
                ]
            )
        qc_columns = st.columns(3)
        for idx, (label, value) in enumerate(qc_cards):
            qc_columns[idx % 3].metric(label, value)
        st.write(f"Included frames: {compact_frame_ranges(included_frames)}")
        st.write(f"Excluded frames: {compact_frame_ranges(excluded_frames)}")
        st.write(f"Isolated reliable frames excluded: {compact_frame_ranges(isolated_frames)}")
        st.caption(
            "QC diagnostics: "
            f"SHI below threshold = {qc_summary.get('framesFailedSHI_tileCV', 'n/a')}, "
            f"low contrast/separation = {qc_summary.get('framesFailedLowContrastSeparation', 'n/a')}, "
            f"implausible mask = {qc_summary.get('framesFailedImplausibleMask', 'n/a')}, "
            f"sparse mask morphology filtered = {qc_summary.get('sparseMaskMorphologyFilteredFrames', qc_summary.get('sparseMaskMorphologyFilteredFrameCount', 'n/a'))}, "
            f"outside stable window = {qc_summary.get('framesOutsideStableWindow', 'n/a')}, "
            f"trusted score frames = {qc_summary.get('trustedForScoreFrames', qc_summary.get('trustedScoreFrameCount', 'n/a'))}, "
            f"pre-start detected = {qc_summary.get('preStartFramesDetected', qc_summary.get('preStartFrameCount', 'n/a'))}, "
            f"pre-start excluded = {qc_summary.get('beforeStartExcludedFrames', qc_summary.get('beforeStartExcludedFrameCount', 'n/a'))}, "
            f"post-start QC warning frames = {qc_summary.get('postStartQCWarningFrames', qc_summary.get('postStartQCWarningFrameCount', 'n/a'))}, "
            f"post-start QC-excluded frames = {qc_summary.get('postStartQCExcludedFrames', qc_summary.get('postStartQCExcludedFrameCount', 'n/a'))}, "
            f"hard bad-mask frames = {qc_summary.get('hardBadMaskFrames', qc_summary.get('hardBadMaskFrameCount', 'n/a'))}, "
            f"huge-gap frames = {qc_summary.get('hugeGapFrameCount', 'n/a')}."
        )
        large_gap_summary = analysis.get("params", {}).get("large_gap_artefact_summary", {})
        if analysis.get("params", {}).get("rbcAnalysisMethod", "Standard RBC Gap Mode") != "Standard RBC Gap Mode":
            st.subheader("Conservative RBC / POD-QC Summary")
            pod_active = bool(df.get("podProcessingEnabled", pd.Series(False, index=df.index)).fillna(False).astype(bool).any())
            conservative_cards = [
                ("Reliable conservative frames", str(int(df.get("conservativeMaskReliable", pd.Series(False, index=df.index)).fillna(False).astype(bool).sum()))),
                ("Median no-POD area", format_metric_value(safe_column_median(df, "conservativeGapAreaFrac"), 3)),
                ("Illumination jumps", str(int(df.get("illuminationJumpFlag", pd.Series(False, index=df.index)).fillna(False).astype(bool).sum()))),
            ]
            if pod_active:
                conservative_cards.extend(
                    [
                        ("Median POD area", format_metric_value(safe_column_median(df, "podRemovedGapAreaFrac"), 3)),
                        ("Median POD area delta", format_metric_value(safe_column_median(df, "podAreaDelta"), 3)),
                        ("POD mode 1 energy", format_metric_value(safe_column_median(df, "podMode1Energy"), 3)),
                    ]
                )
            conservative_columns = st.columns(3)
            for idx, (label, value) in enumerate(conservative_cards):
                conservative_columns[idx % 3].metric(label, value)
            if pod_active:
                st.caption("Conservative no-POD masks are the main masks unless Experimental POD-removed RBC Mode is selected. POD diagnostic values were computed for this run.")
            else:
                st.caption("POD removal disabled. Conservative no-POD masks are the final masks and no POD-sensitive warnings were calculated.")
        with st.expander("Maximum single-gap artefact filter diagnostics", expanded=False):
            st.write(f"RBC gap safety profile: {analysis.get('params', {}).get('rbcGapSafetyMode', 'n/a')}")
            st.write(f"enableMaxGapAreaFilter: {large_gap_summary.get('maxGapAreaFilterEnabled', 'n/a')}")
            st.write(f"maxGapAreaFrac: {large_gap_summary.get('maxGapAreaFrac', 'n/a')}")
            st.write(f"Frames affected: {large_gap_summary.get('framesWithLargeGapRemoval', 'n/a')}")
            st.write(f"Frames with suspicious large gaps detected: {large_gap_summary.get('framesWithLargeGapDetection', 'n/a')}")
            st.write(f"Maximum suspicious object fraction: {format_metric_value(float(large_gap_summary.get('maxLargestRemovedGapFrac', np.nan)), 3)}")
            st.write(f"Large-gap artefact risk: {large_gap_summary.get('largeGapArtefactRisk', 'n/a')}")
            st.write(f"MATLAB-compatible mode disabled filter: {large_gap_summary.get('matlabCompatibleModeDisabledFilter', 'n/a')}")
        if stable_start_display == "Not available / no stable reliable window detected" and not qc_summary.get("stableReliableWindowDetected", False):
            st.warning("No reliable automatic start frame detected. Check overlays or use manual start/thresholds.")
        with st.expander("First-rise start-frame visual check", expanded=False):
            startup_frame = qc_summary.get("autoStart_originalImageIndex", qc_summary.get("startupQCStableStartFrame", np.nan))
            if not np.isfinite(startup_frame):
                st.warning("No reliable automatic start frame detected.")
                target_pos = 0
            else:
                original_indices = pd.to_numeric(df.get("originalImageIndex_1based", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
                finite_original = np.isfinite(original_indices)
                rounded_original = np.full_like(original_indices, fill_value=np.nan, dtype=float)
                rounded_original[finite_original] = np.round(original_indices[finite_original])
                matches = np.flatnonzero(finite_original & (rounded_original == int(round(float(startup_frame)))))
                target_pos = int(matches[0]) if matches.size else 0
            previews = analysis.get("previews", [])
            if previews and target_pos < len(previews):
                preview = previews[target_pos]
                st.caption(
                    f"Displayed frame: {preview.get('filename', 'n/a')} | "
                    f"autoStartFrame: {format_metric_value(float(qc_summary.get('autoStart_originalImageIndex', np.nan)), 0)} | "
                    f"autoStartTime_s: {format_metric_value(float(qc_summary.get('autoStartTime_s', np.nan)), 2)} | "
                    f"finalStartFrameUsed: {format_metric_value(float(qc_summary.get('finalStartFrameUsed', np.nan)), 0)}"
                )
                cols = st.columns(3)
                cols[0].image(preview["image"], caption="Analysis grayscale crop", use_container_width=True)
                cols[1].image((preview["mask"].astype(np.uint8) * 255), caption="Plasma gap mask", use_container_width=True)
                cols[2].image(preview.get("boundary_overlay", preview.get("filled_overlay")), caption="Startup QC overlay", use_container_width=True)
        recommendation = qc_summary.get("recommendation")
        sample_qc = qc_summary.get("sampleQC")
        if recommendation:
            if sample_qc == "Reliable aggregation morphology" and qc_summary.get("scoresConfidence") != "low":
                st.info(recommendation)
            else:
                st.warning(recommendation)
    elif (
        mode == "Object Mode"
        and analysis
        and (
            analysis.get("params", {}).get("enhancedObjectModeEnabled", False)
            or analysis.get("params", {}).get("conservativeObjectModeEnabled", False)
            or analysis.get("params", {}).get("experimentalCompositeScoresEnabled", False)
        )
        and not analysis["metrics"].empty
    ):
        df = analysis["metrics"]
        st.subheader("Object Mode Summary")
        if analysis.get("params", {}).get("enhancedObjectModeEnabled", False):
            preprocessing_style = analysis.get("params", {}).get("enhancedObjectPreprocessingStyle", "Current generic enhanced mode")
            st.info(
                "Enhanced Object Mode uses robust bright/dark structure segmentation to quantify generic morphology. "
                "The detected mask does not represent RBC plasma gaps."
            )
            st.caption(f"Enhanced preprocessing style: {preprocessing_style}")
        if analysis.get("params", {}).get("conservativeObjectModeEnabled", False):
            pod_active = bool(df.get("podProcessingEnabled", pd.Series(False, index=df.index)).fillna(False).astype(bool).any())
            st.info(
                f"{analysis.get('params', {}).get('selectedObjectMethod', 'Conservative Object Mode')} uses corrected no-POD generic object masks"
                + (" with an active POD diagnostic/final POD branch. " if pod_active else ". POD removal is disabled for this run. ")
                + "The selected final object mask drives the metrics table, plots, exports, and generic scores."
            )
        cards = [
            ("Images analysed", str(len(df))),
            ("Mean areaFrac", format_metric_value(safe_column_mean(df, "areaFrac"), 3)),
            ("Mean objectCount", format_metric_value(safe_column_mean(df, "objectCount"), 1)),
            ("Mean diameter px", format_metric_value(safe_column_mean(df, "meanDiam_px"), 2)),
            ("Mean correlation length px", format_metric_value(safe_column_mean(df, "correlationLength_px"), 2)),
            ("Low-resolution risk", ", ".join(sorted(set(df.get("lowResolutionRisk", pd.Series(["n/a"])).astype(str))))),
        ]
        if "ObjectStructureScore_QC" in df.columns:
            cards.append(("Mean ObjectStructureScore QC", format_metric_value(safe_column_mean(df, "ObjectStructureScore_QC"), 1)))
        if "ObjectAggregationIndex_QC" in df.columns:
            cards.append(("Mean ObjectAggregationIndex QC", format_metric_value(safe_column_mean(df, "ObjectAggregationIndex_QC"), 1)))
        if "gSASI_0_100" in df.columns:
            cards.append(("Mean experimental gSASI", format_metric_value(safe_column_mean(df, "gSASI_0_100"), 1)))
        if "gAggIndex_0_100" in df.columns:
            cards.append(("Mean experimental gAggIndex", format_metric_value(safe_column_mean(df, "gAggIndex_0_100"), 1)))
        columns = st.columns(3)
        for idx, (label, value) in enumerate(cards):
            columns[idx % 3].metric(label, value)
        if bool(analysis.get("params", {}).get("experimentalCompositeScoresEnabled", False)):
            st.warning(
                "These optional scores reuse the mathematical structure of the RBC aggregation scores, but they are not validated "
                "for this sample type. Interpret them only as exploratory image-structure indices."
            )
        fallback_summary = analysis.get("params", {}).get("startup_transient_fallback_summary", {})
        if analysis.get("params", {}).get("enableStartupTransientFallback", False) or fallback_summary:
            st.info(
                "Startup/transient fallback is an optional QC texture diagnostic. "
                f"Fallback frames: {fallback_summary.get('nStartupFallbackFrames', 0)}. "
                "It does not replace Object Mode masks or normal object metrics."
            )

    st.info("Research-use structural image analysis only. This app is not intended for clinical diagnosis, treatment decisions, or medical advice.")


def render_segmentation_preview(analysis: dict[str, Any] | None) -> None:
    """Render mode-specific segmentation previews."""
    if not analysis or not analysis["previews"]:
        st.info("Run analysis to see segmentation previews.")
        return

    previews = analysis["previews"]
    names = [item["filename"] for item in previews]
    choice = st.selectbox("Image", names)
    preview = previews[names.index(choice)]
    metrics_df_all = analysis.get("metrics", pd.DataFrame())
    selected_metrics_row = pd.Series(dtype=object)
    if isinstance(metrics_df_all, pd.DataFrame) and not metrics_df_all.empty and "filename" in metrics_df_all.columns:
        matching_rows = metrics_df_all.loc[metrics_df_all["filename"].astype(str) == str(choice)]
        if not matching_rows.empty:
            selected_metrics_row = matching_rows.iloc[0]

    if "gt_filled_mask" in preview:
        has_auto = "auto_overlay" in preview
        columns = st.columns(4 if has_auto else 3)
        columns[0].image(preview["image"], caption="Original image", use_container_width=True)
        columns[1].image((preview["gt_input_mask"].astype(np.uint8) * 255), caption="Ground-truth input mask/outline", use_container_width=True)
        columns[2].image((preview["gt_filled_mask"].astype(np.uint8) * 255), caption="Ground-truth filled mask", use_container_width=True)
        if has_auto:
            title = f"Auto segmentation overlay | Dice {preview['dice']:.3f}, IoU {preview['iou']:.3f}"
            columns[3].image(preview["auto_overlay"], caption=title, use_container_width=True)
        return

    overlay_style = analysis.get("params", {}).get("overlay_style", "Boundary" if "wz" in preview else "Filled")
    if preview.get("enhanced_object_mode", False):
        diagnostics = preview.get("structure_diagnostics", {})
        preprocessing_style = preview.get("enhanced_preprocessing_style", diagnostics.get("enhancedObjectPreprocessingStyle", "Current generic enhanced mode"))
        analysis_source = preview.get("object_analysis_input_source", diagnostics.get("objectAnalysisInputSource", "generic enhanced residual"))
        st.caption(
            f"Enhanced low-resolution structure mode | target used: {preview.get('object_target', 'n/a')} | "
            f"preprocessing: {preprocessing_style} | white mask pixels are generic detected structures."
        )
        if not selected_metrics_row.empty:
            st.caption(
                f"startupFallbackUsed: {truthy_value(selected_metrics_row.get('startupFallbackUsed', False))} | "
                f"normalMaskReliable: {truthy_value(selected_metrics_row.get('normalMaskReliable', True))} | "
                f"frameClass: {selected_metrics_row.get('frameClass', 'n/a')}"
            )
            if truthy_value(selected_metrics_row.get("startupFallbackUsed", False)):
                st.warning(
                    "This frame was flagged as early/startup transient. Normal generic mask metrics are not reliable. "
                    "Startup/transient texture metrics were calculated separately."
                )
        if preprocessing_style == "MATLAB/RBC-style robust residual mode":
            st.info(
                "This preview uses a MATLAB/RBC-style robust residual field as a generic preprocessing preset. "
                "The mask is still a generic structure mask, not an RBC plasma-gap mask."
            )
        columns = st.columns(4)
        columns[0].image(preview["image"], caption="Analysis grayscale image", use_container_width=True)
        columns[1].image(preview["wz"], caption="Illumination-corrected robust residual", use_container_width=True)
        columns[2].image((preview["mask"].astype(np.uint8) * 255), caption="Generic structure mask", use_container_width=True)
        columns[3].image(preview["boundary_overlay"], caption="Generic structure mask overlay", use_container_width=True)
        st.caption(
            "Green contours show the final generic structure mask used for metric calculation. "
            f"Analysis input source: {analysis_source}."
        )
        removed_count = int(diagnostics.get("nLargeObjectsRemoved", 0) or 0)
        if removed_count > 0:
            st.warning(
                "Large generic structure removed as a possible artefact. Check overlay. "
                f"Removed objects: {removed_count}; largest suspicious object fraction: "
                f"{format_metric_value(float(diagnostics.get('largestRemovedObjectFrac', np.nan)), 3)}."
            )
        elif int(diagnostics.get("nLargeObjectsDetected", 0) or 0) > 0:
            st.warning("Suspiciously large generic structure detected while the optional max-object filter is disabled. Check overlay.")
        with st.expander("Enhanced Object Mode preprocessing diagnostics/export", expanded=False):
            st.caption("This debug export uses generic Object Mode wording and does not create RBC plasma-gap outputs.")
            diagnostic_cols = [
                "enhancedObjectPreprocessingStyle",
                "objectAnalysisInputSource",
                "objectAnalysisCropFrac",
                "objectAnalysisCropSize",
                "objectSigmaIll",
                "objectMadW",
                "objectKHiUse",
                "objectKLoUse",
                "objectSeedAreaFrac",
                "objectGrowAreaFrac",
                "objectReconstructedAreaFrac",
                "objectMaskBeforeLargeFilterAreaFrac",
                "largestObjectFrac_afterFilter",
                "largeObjectArtefactFlag",
            ]
            if diagnostics:
                st.dataframe(
                    pd.DataFrame([{col: diagnostics.get(col, "n/a") for col in diagnostic_cols}]),
                    use_container_width=True,
                    hide_index=True,
                )
            debug_key = f"enhanced_object_debug_{safe_filename_token(str(choice))}"
            if st.button("Build Enhanced Object debug export for selected frame", key=f"build_{debug_key}"):
                source_file = preview.get("source_file")
                if source_file is None:
                    st.warning("This preview does not have a source file reference available for debug export. Re-run analysis and try again.")
                else:
                    try:
                        source_name, source_image = read_uploaded_image(source_file)
                        zip_bytes, debug_report = build_enhanced_object_debug_bundle(source_image, analysis.get("params", {}), source_name)
                        st.session_state[debug_key] = {"bytes": zip_bytes, "report": debug_report, "filename": source_name}
                    except Exception as exc:
                        st.error(f"Could not build Enhanced Object debug export: {exc}")
            if debug_key in st.session_state:
                payload = st.session_state[debug_key]
                st.metric("Debug ZIP byte length", len(payload.get("bytes", b"")))
                st.dataframe(pd.DataFrame([payload.get("report", {}).get("diagnostics", {})]), use_container_width=True, hide_index=True)
                if len(payload.get("bytes", b"")) > 0:
                    st.download_button(
                        "Download Enhanced Object preprocessing debug ZIP",
                        data=payload["bytes"],
                        file_name=f"MicroStructAI_Enhanced_Object_debug_{safe_filename_token(Path(str(payload.get('filename', choice))).stem)}.zip",
                        mime="application/zip",
                        key=f"download_{debug_key}",
                    )
        return

    if preview.get("conservative_object_mode", False):
        info = preview.get("conservative_object_info", {})
        pod_processing = truthy_value(info.get("podProcessingEnabled", False))
        zero_mode_pod = truthy_value(info.get("zeroModePODReconstruction", False))
        pod_status = str(info.get("podStatus", "POD removal disabled" if not pod_processing else "POD diagnostic active"))
        if not selected_metrics_row.empty:
            st.caption(
                f"selectedObjectMethod: {selected_metrics_row.get('selectedObjectMethod', info.get('selectedObjectMethod', 'n/a'))} | "
                f"finalObjectPreprocessingSource: {selected_metrics_row.get('finalObjectPreprocessingSource', info.get('finalObjectPreprocessingSource', 'n/a'))} | "
                f"finalObjectMaskSource: {selected_metrics_row.get('finalObjectMaskSource', info.get('finalObjectMaskSource', 'n/a'))} | "
                f"objectAreaFracFinal: {format_metric_value(float(pd.to_numeric(pd.Series([selected_metrics_row.get('objectAreaFracFinal', np.nan)]), errors='coerce').iloc[0]), 4)} | "
                f"objectNtotalFinal: {format_metric_value(float(pd.to_numeric(pd.Series([selected_metrics_row.get('objectNtotalFinal', np.nan)]), errors='coerce').iloc[0]), 0)} | "
                f"Object_Xi_mask: {format_metric_value(float(pd.to_numeric(pd.Series([selected_metrics_row.get('Object_Xi_mask', np.nan)]), errors='coerce').iloc[0]), 2)} | "
                f"Object_Xi_QC: {format_metric_value(float(pd.to_numeric(pd.Series([selected_metrics_row.get('Object_Xi_QC', np.nan)]), errors='coerce').iloc[0]), 2)} | "
                f"objectFrameClass: {selected_metrics_row.get('objectFrameClass', 'n/a')} | "
                f"objectMaskReliable: {truthy_value(selected_metrics_row.get('objectMaskReliable', False))}"
            )
        pod_caption = (
            f"podRemovedObjectAreaFrac: {format_metric_value(float(info.get('podRemovedObjectAreaFrac', np.nan)), 4)} | "
            f"podObjectAreaDelta: {format_metric_value(float(info.get('podObjectAreaDelta', np.nan)), 4)} | "
            f"podObjectConsistencyFlag: {info.get('podObjectConsistencyFlag', 'n/a')} | "
        ) if pod_processing else f"{pod_status} | "
        st.caption(
            "Conservative Object POD-QC | "
            f"selectedObjectMethod: {info.get('selectedObjectMethod', 'n/a')} | "
            f"finalObjectPreprocessingSource: {info.get('finalObjectPreprocessingSource', 'n/a')} | "
            f"finalObjectMaskSource: {info.get('finalObjectMaskSource', 'n/a')} | "
            f"finalObjectAreaFracSource: {info.get('finalObjectAreaFracSource', 'n/a')} | "
            f"objectAreaFracFinal: {format_metric_value(float(info.get('objectAreaFracFinal', np.nan)), 4)} | "
            f"objectNtotalFinal: {format_metric_value(float(info.get('objectNtotalFinal', np.nan)), 0)} | "
            f"conservativeNoPODObjectAreaFrac: {format_metric_value(float(info.get('conservativeNoPODObjectAreaFrac', np.nan)), 4)} | "
            f"{pod_caption}"
            f"objectSegmentationTargetUsed: {info.get('objectSegmentationTargetUsed', 'n/a')}"
        )
        columns = st.columns(5 if pod_processing else 4)
        columns[0].image(preview.get("camera_crop_gray", preview["image"]), caption="Raw grayscale crop", use_container_width=True)
        columns[1].image(preview["image"], caption=f"{info.get('finalObjectPreprocessingSource', 'Selected object image')}", use_container_width=True)
        columns[2].image((preview["mask"].astype(np.uint8) * 255), caption=f"{info.get('finalObjectMaskSource', 'Selected object mask')}", use_container_width=True)
        columns[3].image(preview["boundary_overlay"], caption="Selected final object overlay", use_container_width=True)
        pod_branch_caption = "POD-branch QC overlay (0 modes removed)" if zero_mode_pod else "POD-removed QC overlay"
        if pod_processing and str(info.get("selectedObjectMethod", "")).startswith("Experimental"):
            columns[4].image(preview.get("no_pod_boundary_overlay", preview["boundary_overlay"]), caption="No-POD comparison/QC overlay", use_container_width=True)
        elif pod_processing:
            columns[4].image(preview.get("pod_boundary_overlay", preview["boundary_overlay"]), caption=pod_branch_caption, use_container_width=True)
        else:
            st.info("POD removal disabled (removeModes = 0 or POD diagnostic disabled). Conservative Object QC active.")
        if not truthy_value(info.get("objectMaskReliable", False)):
            st.warning("This frame is not considered a trusted generic object-structure frame. QC-aware object scores are NaN for this frame.")
        if pod_processing and str(info.get("podObjectConsistencyFlag", "")).lower().startswith("pod-sensitive"):
            st.warning("POD removal strongly changes the detected generic object mask. Treat the POD comparison as diagnostic.")
        st.caption(
            "Green contours show the selected final generic object mask used for Object Mode metrics. "
            "These masks are not RBC plasma gaps and the scores are generic structural/coarsening indices."
        )
        return

    if "wz" in preview:
        metrics_df = analysis.get("metrics", pd.DataFrame())
        qc_row = pd.Series(dtype=object)
        if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty and "filename" in metrics_df.columns:
            matching_rows = metrics_df.loc[metrics_df["filename"].astype(str) == str(choice)]
            if not matching_rows.empty:
                qc_row = matching_rows.iloc[0]
        rbc_qc_label = qc_row.get("rbcQCLabel", qc_row.get("rbcFrameQC", "n/a"))
        included_in_summary = bool(qc_row.get("usedInSummary", qc_row.get("includedInMainSummary", qc_row.get("usedForMainSummary", False))))
        startup_reliable = bool(qc_row.get("isStartupReliableMorphology", False))
        startup_stable_frame = qc_row.get("startupQCStableStartFrame", np.nan)
        final_start_frame = qc_row.get("finalStartFrameUsed", np.nan)
        auto_start_frame = qc_row.get("autoStart_originalImageIndex", startup_stable_frame)
        auto_start_time = qc_row.get("autoStartTime_s", np.nan)
        startup_stable_numeric = pd.to_numeric(pd.Series([startup_stable_frame]), errors="coerce").iloc[0]
        final_start_numeric = pd.to_numeric(pd.Series([final_start_frame]), errors="coerce").iloc[0]
        auto_start_numeric = pd.to_numeric(pd.Series([auto_start_frame]), errors="coerce").iloc[0]
        auto_time_numeric = pd.to_numeric(pd.Series([auto_start_time]), errors="coerce").iloc[0]
        sample_qc = qc_row.get("sampleQC", analysis.get("params", {}).get("rbc_qc_summary", {}).get("sampleQC", "n/a"))
        scores_confidence = qc_row.get("scoresConfidence", analysis.get("params", {}).get("rbc_qc_summary", {}).get("scoresConfidence", "n/a"))
        isolated_reliable = bool(qc_row.get("isolatedReliableFrame", False))
        mask_reliability = float(pd.to_numeric(pd.Series([qc_row.get("maskReliabilityScore", np.nan)]), errors="coerce").iloc[0])
        bad_mask_flag = truthy_value(qc_row.get("badMaskFlag", False))
        huge_gap_flag = truthy_value(qc_row.get("hugeGapFlag", False))
        repair_triggered = truthy_value(qc_row.get("adaptiveRepairTriggered", False))
        repair_kept = truthy_value(qc_row.get("adaptiveRepairKept", False))
        startup_fallback_used = truthy_value(qc_row.get("startupFallbackUsed", False))
        normal_mask_reliable = truthy_value(qc_row.get("normalMaskReliable", True))
        frame_class = qc_row.get("frameClass", "n/a")
        score_interpretable = truthy_value(qc_row.get("rbcScoreInterpretable", False))
        score_flag = qc_row.get("rbcScoreInterpretationFlag", "n/a")
        score_reason = qc_row.get("rbcScoreQCReason", "n/a")
        sasi_qc = float(pd.to_numeric(pd.Series([qc_row.get("SASI_QC_0_100", np.nan)]), errors="coerce").iloc[0])
        agg_qc = float(pd.to_numeric(pd.Series([qc_row.get("AggIndex_QC_0_100", np.nan)]), errors="coerce").iloc[0])
        xi_raw_intensity = float(pd.to_numeric(pd.Series([qc_row.get("Xi_raw_intensity", qc_row.get("Xi_interp_px", np.nan))]), errors="coerce").iloc[0])
        xi_mask_value = float(pd.to_numeric(pd.Series([qc_row.get("Xi_mask", np.nan)]), errors="coerce").iloc[0])
        xi_qc_value = float(pd.to_numeric(pd.Series([qc_row.get("Xi_QC", np.nan)]), errors="coerce").iloc[0])
        xi_source = qc_row.get("xiSource", "n/a")
        score_source = qc_row.get("scoreSource", "n/a")
        st.caption(
            f"selectedRBCMethod: {qc_row.get('selectedRBCMethod', analysis.get('params', {}).get('rbcAnalysisMethod', 'n/a'))} | "
            f"finalPreprocessingSource: {qc_row.get('finalPreprocessingSource', 'n/a')} | "
            f"finalMaskSource: {qc_row.get('finalMaskSource', 'n/a')} | "
            f"adaptiveEngine: {qc_row.get('conservativeAdaptiveEngine', 'n/a')} | "
            f"adaptSensitivity: {format_metric_value(float(pd.to_numeric(pd.Series([qc_row.get('conservativeAdaptSensitivity', np.nan)]), errors='coerce').iloc[0]), 2)} | "
            f"adaptMeanScaleBase: {format_metric_value(float(pd.to_numeric(pd.Series([qc_row.get('conservativeAdaptMeanScaleBase', np.nan)]), errors='coerce').iloc[0]), 2)} | "
            f"areaFracFinal: {format_metric_value(float(pd.to_numeric(pd.Series([qc_row.get('areaFracFinal', np.nan)]), errors='coerce').iloc[0]), 4)} | "
            f"nObjectsFinal: {format_metric_value(float(pd.to_numeric(pd.Series([qc_row.get('nObjectsFinal', np.nan)]), errors='coerce').iloc[0]), 0)} | "
            f"startupReliable: {startup_reliable} | "
            f"autoStartFrame: {format_metric_value(float(auto_start_numeric), 0)} | "
            f"autoStartTime_s: {format_metric_value(float(auto_time_numeric), 2)} | "
            f"finalStartFrameUsed: {format_metric_value(float(final_start_numeric), 0)} | "
            f"usedInSummary: {included_in_summary} | rbcQCLabel: {rbc_qc_label} | "
            f"sampleQC: {sample_qc} | scoresConfidence: {scores_confidence} | "
            f"maskReliabilityScore: {format_metric_value(mask_reliability, 1)} | badMaskFlag: {bad_mask_flag} | hugeGapFlag: {huge_gap_flag} | "
            f"startupFallbackUsed: {startup_fallback_used} | normalMaskReliable: {normal_mask_reliable} | frameClass: {frame_class}"
        )
        st.caption(
            f"QC-aware scores | rbcScoreInterpretable: {score_interpretable} | "
            f"rbcScoreInterpretationFlag: {score_flag} | SASI_QC_0_100: {format_metric_value(sasi_qc, 1)} | "
            f"AggIndex_QC_0_100: {format_metric_value(agg_qc, 1)} | reason: {score_reason}"
        )
        st.caption(
            f"Trusted Xi | Xi_raw_intensity: {format_metric_value(xi_raw_intensity, 2)} | "
            f"Xi_mask: {format_metric_value(xi_mask_value, 2)} | Xi_QC: {format_metric_value(xi_qc_value, 2)} | "
            f"xiSource: {xi_source} | scoreSource: {score_source}"
        )
        if np.isfinite(xi_raw_intensity) and xi_raw_intensity >= 10.0 and not np.isfinite(xi_qc_value):
            st.warning(
                "Raw intensity Xi is high, but the frame failed mask/QC support. "
                "Trusted mask-based Xi and aggregation scores are not reported for this frame."
        )
        if preview.get("conservative_rbc_mode", False):
            info = preview.get("conservative_info", {})
            pod_processing = truthy_value(info.get("podProcessingEnabled", False))
            zero_mode_pod = truthy_value(info.get("zeroModePODReconstruction", False))
            pod_status = str(info.get("podStatus", "POD removal disabled" if not pod_processing else "POD diagnostic active"))
            pod_caption = (
                f"podRemovedAreaFrac: {format_metric_value(float(info.get('podRemovedAreaFrac', np.nan)), 4)} | "
                f"area_PODremoved: {format_metric_value(float(info.get('area_PODremoved', np.nan)), 3)} | "
                f"podAreaDelta: {format_metric_value(float(info.get('podAreaDelta', np.nan)), 3)} | "
                f"podConsistencyFlag: {info.get('podConsistencyFlag', 'n/a')} | "
                f"podMode1Energy: {format_metric_value(float(info.get('podMode1Energy', np.nan)), 3)} | "
            ) if pod_processing else f"{pod_status} | "
            st.caption(
                "Conservative RBC POD-QC | "
                f"selectedRBCMethod: {info.get('selectedRBCMethod', 'n/a')} | "
                f"finalPreprocessingSource: {info.get('finalPreprocessingSource', 'n/a')} | "
                f"finalMaskSource: {info.get('finalMaskSource', 'n/a')} | "
                f"finalAreaFracSource: {info.get('finalAreaFracSource', 'n/a')} | "
                f"areaFracFinal: {format_metric_value(float(info.get('areaFracFinal', np.nan)), 4)} | "
                f"nObjectsFinal: {format_metric_value(float(info.get('nObjectsFinal', np.nan)), 0)} | "
                f"conservativeNoPODAreaFrac: {format_metric_value(float(info.get('conservativeNoPODAreaFrac', np.nan)), 4)} | "
                f"conservativeAreaFrac: {format_metric_value(float(info.get('conservativeAreaFrac', np.nan)), 4)} | "
                f"standardAreaFrac: {format_metric_value(float(info.get('standardAreaFrac', np.nan)), 4)} | "
                f"area_noPOD: {format_metric_value(float(info.get('area_noPOD', np.nan)), 3)} | "
                f"{pod_caption}"
                f"conservativeFrameClass: {info.get('conservativeFrameClass', 'n/a')}"
            )
            columns = st.columns(5 if pod_processing else 4)
            columns[0].image(preview.get("camera_crop_gray", preview["image"]), caption="Raw grayscale crop", use_container_width=True)
            columns[1].image(preview["image"], caption=f"{info.get('finalPreprocessingSource', 'Selected conservative image')}", use_container_width=True)
            columns[2].image((preview["mask"].astype(np.uint8) * 255), caption=f"{info.get('finalMaskSource', 'Selected plasma-gap mask')}", use_container_width=True)
            columns[3].image(preview["boundary_overlay"], caption="Selected final mask overlay", use_container_width=True)
            pod_branch_caption = "POD-branch QC overlay (0 modes removed)" if zero_mode_pod else "POD-removed QC overlay"
            if pod_processing and str(info.get("selectedRBCMethod", "")).startswith("Experimental"):
                columns[4].image(preview.get("no_pod_boundary_overlay", preview["boundary_overlay"]), caption="No-POD comparison/QC overlay", use_container_width=True)
            elif pod_processing:
                columns[4].image(preview.get("pod_boundary_overlay", preview["boundary_overlay"]), caption=pod_branch_caption, use_container_width=True)
            else:
                st.info("POD removal disabled (removeModes = 0 or POD diagnostic disabled). Conservative QC active.")
            if str(info.get("conservativeFrameClass", "")).startswith("weak"):
                st.warning("This frame shows weak/pre-gap morphology. Conservative RBC mode did not detect strong reliable plasma-gap structure.")
            elif info.get("conservativeMaskReliable", False):
                st.info("Conservative RBC mode detected reliable plasma-gap morphology.")
            if pod_processing and str(info.get("podConsistencyFlag", "")).lower().startswith("pod-sensitive"):
                st.warning("POD removal strongly changes the detected mask. Treat POD-removed result as diagnostic only.")
            return
        show_both = overlay_style == "Both"
        columns = st.columns(6 if show_both else 5)
        columns[0].image(preview.get("camera_crop_gray", preview["image"]), caption="Raw grayscale crop (visual only)", use_container_width=True)
        columns[1].image(preview["image"], caption="Analysis grayscale crop", use_container_width=True)
        columns[2].image(preview["wz"], caption="Illumination-corrected Wz", use_container_width=True)
        columns[3].image((preview["mask"].astype(np.uint8) * 255), caption=preview["mask_label"], use_container_width=True)
        if overlay_style == "Filled":
            columns[4].image(preview["filled_overlay"], caption=preview["filled_title"], use_container_width=True)
        elif overlay_style == "Both":
            columns[4].image(preview["filled_overlay"], caption=preview["filled_title"], use_container_width=True)
            columns[5].image(preview["boundary_overlay"], caption=preview["boundary_title"], use_container_width=True)
        else:
            columns[4].image(preview["boundary_overlay"], caption=preview["boundary_title"], use_container_width=True)
        st.caption(
            "Green contours show the final plasma-gap mask used for metric calculation. "
            "The analysis grayscale crop matches MATLAB Iraw01 by default; display Wz is contrast-stretched only for viewing."
        )
        removed_count = int(pd.to_numeric(pd.Series([qc_row.get("nLargeGapObjectsRemoved", 0)]), errors="coerce").fillna(0).iloc[0])
        detected_count = int(pd.to_numeric(pd.Series([qc_row.get("nLargeGapObjectsDetected", 0)]), errors="coerce").fillna(0).iloc[0])
        largest_removed = float(pd.to_numeric(pd.Series([qc_row.get("largestRemovedGapFrac", np.nan)]), errors="coerce").iloc[0])
        if removed_count > 0:
            st.warning(
                "Large bright object removed as possible artefact. Check overlay. "
                f"Removed objects: {removed_count}; largest suspicious gap fraction: {format_metric_value(largest_removed, 3)}."
            )
        elif detected_count > 0 and not bool(qc_row.get("maxGapAreaFilterEnabled", False)):
            st.warning(
                "Suspiciously large bright gap object detected while the maximum single-gap filter is disabled. "
                f"Check overlay. Largest suspicious gap fraction: {format_metric_value(largest_removed, 3)}."
            )
        if not included_in_summary:
            if isolated_reliable:
                st.warning("This frame was excluded from the main summary because it is reliable-looking but outside a stable contiguous reliable window.")
            else:
                st.warning("This frame was excluded from the main summary because it was labelled weak/pre-gap or possible artefact.")
        if bad_mask_flag:
            st.warning(f"This frame is marked badMaskFlag=True. Reasons: {qc_row.get('maskQCFailureReasons', 'n/a')}")
        if startup_fallback_used:
            st.warning(
                "This frame was flagged as early/startup transient. Normal segmentation metrics are not reliable. "
                "Startup/transient texture metrics were calculated separately."
            )
        if repair_triggered:
            status = "kept" if repair_kept else "not kept"
            st.info(f"Adaptive anti-merging repair was triggered for this frame and was {status}: {qc_row.get('adaptiveRepairReason', 'n/a')}")
        with st.expander("RBC MATLAB mask pipeline debug/export", expanded=False):
            st.caption(
                "Builds a one-frame debug ZIP with the app pipeline and a MATLAB-reference rerun. "
                "The MATLAB-reference rerun uses the processed Iraw01 analysis grayscale and disables the MicroStructAI max-gap artefact filter."
            )
            current_debug = preview.get("debug_report", {})
            if current_debug:
                debug_preview_cols = [
                    "analysisInputSource",
                    "analysisCropSize",
                    "cropRowStart_1based",
                    "cropRowEnd_1based",
                    "cropColStart_1based",
                    "cropColEnd_1based",
                    "madW",
                    "wzMadFloor",
                    "wzSmoothSigma",
                    "kHiUse",
                    "kLoUse",
                    "seedAreaFrac",
                    "growAreaFrac",
                    "reconstructedAreaFrac",
                    "gapAreaFrac",
                    "gapNtotal",
                    "largestGapFrac",
                ]
                st.dataframe(
                    pd.DataFrame([{col: current_debug.get(col, "n/a") for col in debug_preview_cols}]),
                    use_container_width=True,
                    hide_index=True,
                )
            debug_key = f"rbc_mask_debug_{safe_filename_token(str(choice))}"
            if st.button("Build debug export for selected frame", key=f"build_{debug_key}"):
                source_file = preview.get("source_file")
                if source_file is None:
                    st.warning("This preview does not have a source file reference available for debug export. Re-run analysis and try again.")
                else:
                    try:
                        source_name, source_image = read_uploaded_image(source_file)
                        zip_bytes, debug_report = build_rbc_mask_debug_bundle(source_image, analysis.get("params", {}), source_name)
                        st.session_state[debug_key] = {"bytes": zip_bytes, "report": debug_report, "filename": source_name}
                    except Exception as exc:
                        st.error(f"Could not build RBC mask debug export: {exc}")
            if debug_key in st.session_state:
                payload = st.session_state[debug_key]
                report = payload.get("report", {})
                comparison = report.get("comparison", {})
                st.metric("Debug ZIP byte length", len(payload.get("bytes", b"")))
                st.dataframe(pd.DataFrame([comparison]), use_container_width=True, hide_index=True)
                st.caption(f"Likely mismatch source: {comparison.get('likelyMismatchSource', 'n/a')}")
                if len(payload.get("bytes", b"")) > 0:
                    st.download_button(
                        "Download RBC mask pipeline debug ZIP",
                        data=payload["bytes"],
                        file_name=f"MicroStructAI_RBC_mask_pipeline_debug_{safe_filename_token(Path(str(payload.get('filename', choice))).stem)}.zip",
                        mime="application/zip",
                        key=f"download_{debug_key}",
                    )
    else:
        if not selected_metrics_row.empty:
            st.caption(
                f"startupFallbackUsed: {truthy_value(selected_metrics_row.get('startupFallbackUsed', False))} | "
                f"normalMaskReliable: {truthy_value(selected_metrics_row.get('normalMaskReliable', True))} | "
                f"frameClass: {selected_metrics_row.get('frameClass', 'n/a')}"
            )
            if truthy_value(selected_metrics_row.get("startupFallbackUsed", False)):
                st.warning(
                    "This frame was flagged as early/startup transient. Normal Object Mode metrics are not reliable. "
                    "Startup/transient texture metrics were calculated separately."
                )
        show_both = overlay_style == "Both"
        columns = st.columns(4 if show_both else 3)
        columns[0].image(preview["image"], caption="Original", use_container_width=True)
        columns[1].image((preview["mask"].astype(np.uint8) * 255), caption=preview["mask_label"], use_container_width=True)
        if overlay_style == "Boundary":
            columns[2].image(preview["boundary_overlay"], caption="Boundary overlay", use_container_width=True)
        elif overlay_style == "Both":
            columns[2].image(preview["filled_overlay"], caption="Filled mask overlay", use_container_width=True)
            columns[3].image(preview["boundary_overlay"], caption="Boundary overlay", use_container_width=True)
        else:
            columns[2].image(preview["filled_overlay"], caption="Filled mask overlay", use_container_width=True)


def render_metrics_table(df: pd.DataFrame) -> None:
    """Render metrics and CSV export."""
    state_key = "single_sample_metrics_df"
    df = store_export_dataframe(state_key, df)
    if df.empty:
        st.info("No metrics available yet.")
        render_csv_download(
            "Download single-sample metrics CSV",
            state_key,
            "MicroStructAI_single_sample_metrics.csv",
            "download_single_sample_metrics_csv_empty",
        )
        render_export_diagnostics({"single_sample_metrics": df})
        return
    if "GT_areaFrac" in df.columns:
        gt_columns = [col for col in ["filename"] + INDEX_COLUMNS + ["maskFile"] + GT_METRICS if col in df.columns]
        st.caption("Ground-truth metric columns are shown first below. Use horizontal scrolling in the full table for every exported column.")
        st.dataframe(df[gt_columns], use_container_width=True, hide_index=True)
        st.divider()
        st.caption("Full metrics table")
    elif "gapAreaFrac" in df.columns:
        st.caption("RBC-specific columns are calculated only in RBC Gap Mode, where the mask represents bright plasma gaps.")
    st.dataframe(df, use_container_width=True, hide_index=True)
    render_csv_download(
        "Download single-sample metrics CSV",
        state_key,
        "MicroStructAI_single_sample_metrics.csv",
        "download_single_sample_metrics_csv",
    )
    render_export_diagnostics({"single_sample_metrics": df})


def plot_rbc_mask_metrics_matlab_v5(df: pd.DataFrame, x_axis: str = "analysisIndex_1based"):
    """Plot mode-appropriate binary-mask RBC diagnostics."""
    plot_df = df.copy()
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No mask metrics available", ha="center", va="center")
        ax.axis("off")
        return fig

    if x_axis in plot_df.columns:
        x = pd.to_numeric(plot_df[x_axis], errors="coerce")
        if not x.notna().any():
            x = pd.Series(np.arange(1, len(plot_df) + 1), index=plot_df.index, dtype="float64")
    else:
        x = pd.Series(np.arange(1, len(plot_df) + 1), index=plot_df.index, dtype="float64")
    x_values = x.to_numpy(dtype=float)
    x_label = x_axis if x_axis in plot_df.columns else "Frame"

    def y(col: str, fallback: str | None = None) -> pd.Series:
        if col in plot_df.columns:
            return pd.to_numeric(plot_df[col], errors="coerce")
        if fallback and fallback in plot_df.columns:
            return pd.to_numeric(plot_df[fallback], errors="coerce")
        return pd.Series(np.nan, index=plot_df.index, dtype="float64")

    method = str(plot_df.get("selectedRBCMethod", pd.Series(["Standard RBC Gap Mode"])).dropna().astype(str).iloc[0]) if "selectedRBCMethod" in plot_df.columns and plot_df["selectedRBCMethod"].notna().any() else "Standard RBC Gap Mode"
    pod_available = any(
        col in plot_df.columns and pd.to_numeric(plot_df[col], errors="coerce").notna().any()
        for col in ["mask_gapAreaFrac_PODremoved", "podRemovedAreaFrac", "MaskSASI_like_PODremoved"]
    )
    conservative_like = method != "Standard RBC Gap Mode" or pod_available
    primary_label = "No POD removal" if conservative_like else "Final selected mask"
    secondary_label = "POD-removed"

    series = [
        ("Gap area fraction from masks", "Area fraction", y("mask_gapAreaFrac_noPOD", "mask_gapAreaFrac"), y("mask_gapAreaFrac_PODremoved"), 0.03),
        ("Gap object count from masks", "Gap count", y("mask_gapNtotal_noPOD", "mask_gapNtotal"), y("mask_gapNtotal_PODremoved"), 10.0),
        ("Mean equivalent gap diameter from masks", "Mean diameter (px)", y("mask_gapMeanDiam_px_noPOD", "mask_gapMeanDiam_px"), y("mask_gapMeanDiam_px_PODremoved"), np.nan),
        ("Largest gap fraction from masks", "Largest gap / image area", y("mask_largestGapFrac_noPOD", "mask_largestGapFrac"), y("mask_largestGapFrac_PODremoved"), np.nan),
        ("Mask-based correlation length, Xi_mask", "Xi_mask (px)", y("Xi_mask_noPOD", "Xi_mask"), y("Xi_mask_PODremoved"), np.nan),
        ("POD/no-POD mask area difference", "|POD - noPOD| area", y("podAreaDelta"), pd.Series(np.nan, index=plot_df.index), 0.05),
        ("Simple SASI-like score from mask metrics only", "Mask SASI-like score", y("MaskSASI_like_noPOD", "MaskSASI_raw_0_100"), y("MaskSASI_like_PODremoved"), np.nan),
        ("Simple GapScore-like score from mask metrics only", "Mask GapScore-like score", y("MaskGapScore_like_noPOD", "MaskGapScore_raw_0_100"), y("MaskGapScore_like_PODremoved"), np.nan),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(13.5, 12.5), sharex=True)
    for ax, (title, ylabel, no_pod, pod, threshold) in zip(axes.ravel(), series):
        no_pod_values = no_pod.to_numpy(dtype=float)
        pod_values = pod.to_numpy(dtype=float)
        plotted = False
        if np.isfinite(no_pod_values).any():
            ax.plot(x_values, no_pod_values, linewidth=1.9, label=primary_label)
            plotted = True
        if np.isfinite(pod_values).any():
            ax.plot(x_values, pod_values, linewidth=1.9, label=secondary_label)
            plotted = True
        if np.isfinite(threshold):
            ax.axhline(threshold, color="#666666", linestyle="--", linewidth=1.1, alpha=0.75, label="Reliability threshold")
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "No finite values available", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.28)
        if plotted:
            ax.legend(loc="best", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel(x_label)
    fig.suptitle("Mask-based RBC metrics over time", fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return fig


def plot_object_mask_metrics_pod(df: pd.DataFrame, x_axis: str = "analysisIndex_1based"):
    """Plot mode-appropriate generic Object mask diagnostics."""
    plot_df = df.copy()
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No object mask metrics available", ha="center", va="center")
        ax.axis("off")
        return fig

    if x_axis in plot_df.columns:
        x = pd.to_numeric(plot_df[x_axis], errors="coerce")
        if not x.notna().any():
            x = pd.Series(np.arange(1, len(plot_df) + 1), index=plot_df.index, dtype="float64")
    else:
        x = pd.Series(np.arange(1, len(plot_df) + 1), index=plot_df.index, dtype="float64")
    x_values = x.to_numpy(dtype=float)
    x_label = x_axis if x_axis in plot_df.columns else "Frame"

    def y(col: str, fallback: str | None = None) -> pd.Series:
        if col in plot_df.columns:
            return pd.to_numeric(plot_df[col], errors="coerce")
        if fallback and fallback in plot_df.columns:
            return pd.to_numeric(plot_df[fallback], errors="coerce")
        return pd.Series(np.nan, index=plot_df.index, dtype="float64")

    method = str(plot_df.get("selectedObjectMethod", pd.Series(["Standard Object Mode"])).dropna().astype(str).iloc[0]) if "selectedObjectMethod" in plot_df.columns and plot_df["selectedObjectMethod"].notna().any() else "Standard Object Mode"
    pod_available = any(
        col in plot_df.columns and pd.to_numeric(plot_df[col], errors="coerce").notna().any()
        for col in ["podRemovedObjectAreaFrac", "ObjectStructureScore_PODremoved", "Object_Xi_mask_PODremoved"]
    )
    conservative_like = method in {"Conservative Object Mode", "Experimental POD-removed Object Mode"} or pod_available
    primary_label = "No POD removal" if conservative_like else "Final selected object mask"
    secondary_label = "POD-removed"

    series = [
        ("Object area fraction from masks", "Area fraction", y("conservativeNoPODObjectAreaFrac", "objectAreaFracFinal"), y("podRemovedObjectAreaFrac"), 0.03),
        ("Object count from masks", "Object count", y("conservativeNoPODObjectNtotal", "objectNtotalFinal"), y("podRemovedObjectNtotal"), 10.0),
        ("Mean equivalent object diameter from masks", "Mean diameter (px)", y("conservativeNoPODObjectMeanDiam_px", "objectMeanDiamFinal"), y("podRemovedObjectMeanDiam_px"), np.nan),
        ("Largest object fraction from masks", "Largest object / image area", y("conservativeNoPODObjectLargestFrac", "objectLargestFracFinal"), y("podRemovedObjectLargestFrac"), np.nan),
        ("Mask-based object correlation length", "Object_Xi_mask (px)", y("Object_Xi_mask_noPOD", "Object_Xi_mask"), y("Object_Xi_mask_PODremoved"), np.nan),
        ("POD/no-POD object area difference", "|POD - noPOD| area", y("podObjectAreaDelta"), pd.Series(np.nan, index=plot_df.index), 0.05),
        ("ObjectStructureScore from mask metrics", "Score (0-100)", y("ObjectStructureScore_noPOD", "ObjectStructureScore_raw_0_100"), y("ObjectStructureScore_PODremoved"), np.nan),
        ("ObjectAggregationIndex from mask metrics", "Score (0-100)", y("ObjectAggregationIndex_noPOD", "ObjectAggregationIndex_raw_0_100"), y("ObjectAggregationIndex_PODremoved"), np.nan),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(13.5, 12.5), sharex=True)
    for ax, (title, ylabel, no_pod, pod, threshold) in zip(axes.ravel(), series):
        no_pod_values = no_pod.to_numpy(dtype=float)
        pod_values = pod.to_numpy(dtype=float)
        plotted = False
        if np.isfinite(no_pod_values).any():
            ax.plot(x_values, no_pod_values, linewidth=1.9, label=primary_label)
            plotted = True
        if np.isfinite(pod_values).any():
            ax.plot(x_values, pod_values, linewidth=1.9, label=secondary_label)
            plotted = True
        if np.isfinite(threshold):
            ax.axhline(threshold, color="#666666", linestyle="--", linewidth=1.1, alpha=0.75, label="Reliability threshold")
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "No finite values available", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.28)
        if plotted:
            ax.legend(loc="best", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel(x_label)
    fig.suptitle("Mask-based Object metrics over time", fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return fig


def plot_normalized_syllectograms(df: pd.DataFrame, x_axis: str = "analysisIndex_1based") -> plt.Figure:
    """Plot source-separated optical and mask-based syllectogram curves."""
    plot_df = df.copy()
    if x_axis in plot_df.columns:
        x = pd.to_numeric(plot_df[x_axis], errors="coerce")
        if not x.notna().any():
            x = pd.Series(np.arange(1, len(plot_df) + 1), index=plot_df.index, dtype="float64")
    else:
        x = pd.Series(np.arange(1, len(plot_df) + 1), index=plot_df.index, dtype="float64")
    x_values = x.to_numpy(dtype=float)
    x_label = x_axis if x_axis in plot_df.columns else "Frame"
    correction_applied = bool(
        plot_df.get("intensityCorrectionApplied", pd.Series(False, index=plot_df.index)).fillna(False).astype(bool).any()
    )
    intensity_source_label = "gain-corrected raw/cropped grayscale" if correction_applied else "raw/cropped grayscale, no gain correction applied"

    def plot_columns(ax, specs: list[tuple[str, str]], title: str, ylabel: str) -> bool:
        plotted_any = False
        plotted_labels: set[str] = set()
        for column, label in specs:
            if label in plotted_labels:
                continue
            if column not in plot_df.columns:
                continue
            values = pd.to_numeric(plot_df[column], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(values).any():
                ax.plot(x_values, values, marker="o", markersize=2.8, linewidth=1.45, label=label)
                plotted_labels.add(label)
                plotted_any = True
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if plotted_any:
            ax.legend(loc="best", fontsize=7)
        else:
            ax.text(0.5, 0.5, "Not available", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        return plotted_any

    has_object_syl = any(
        column in plot_df.columns and pd.to_numeric(plot_df[column], errors="coerce").notna().any()
        for column in ["ObjectStructureSyl_raw_0_1", "ObjectAggregationSyl_raw_0_1", "ObjectStructureSyl_0_1", "ObjectAggregationSyl_0_1", "ObjectStructure_noPOD_syl_norm01", "ObjectStructure_PODremoved_syl_norm01"]
    )
    last_panel_specs = (
        [
            ("ObjectAggregationSyl_raw_0_1", "ObjectAggregation raw"),
            ("ObjectAggregationSyl_QC_0_1", "ObjectAggregation QC"),
            ("ObjectStructureSyl_raw_0_1", "ObjectStructure raw"),
            ("ObjectStructureSyl_QC_0_1", "ObjectStructure QC"),
            ("ObjectAggregationSyl_0_1", "ObjectAggregation raw"),
            ("ObjectStructureSyl_0_1", "ObjectStructure raw"),
            ("ObjectStructure_noPOD_syl_norm01", "ObjectStructure no-POD"),
            ("ObjectStructure_PODremoved_syl_norm01", "ObjectStructure POD-removed"),
            ("gSASI_syl_norm01", "gSASI exploratory"),
        ],
        "Object aggregation syllectogram: final selected object mask",
        "Object score 0-1",
    ) if has_object_syl else (
        [
            ("MaskGapScore_syl_raw_0_1", "Mask-GapScore raw"),
            ("MaskGapScore_syl_QC_0_1", "Mask-GapScore QC"),
            ("MaskAggIndex_syl_raw_0_1", "Mask-AggIndex raw"),
            ("MaskAggIndex_syl_QC_0_1", "Mask-AggIndex QC"),
            ("MaskGapScore_syl_0_1", "Mask-GapScore raw"),
            ("MaskAggIndex_syl_0_1", "Mask-AggIndex raw"),
        ],
        "Mask-GapScore / Mask-AggIndex syllectogram: final selected binary mask",
        "Mask score 0-1",
    )

    panels = [
        (
            [
                ("opticalMean_raw", "Raw mean"),
                ("opticalMean_corrected", "Gain-corrected mean"),
                ("opticalMeanIntensity_raw", "Raw mean"),
                ("opticalMeanIntensity_corrected", "Gain-corrected mean"),
            ],
            "Optical mean-intensity syllectogram: grayscale intensity",
            "Mean intensity",
        ),
        (
            [
                ("opticalMean_raw_normMinRise", "Raw I/min(I)-1"),
                ("opticalMean_corrected_normMinRise", "Corrected I/min(I)-1"),
                ("opticalMean_raw_norm01", "Raw 0-1"),
                ("opticalMean_corrected_norm01", "Corrected 0-1"),
                ("MeanI_norm_minRise", "Corrected I/min(I)-1"),
                ("MeanI_norm_0_1", "Corrected 0-1"),
            ],
            f"Optical intensity normalized curves: {intensity_source_label}",
            "Normalized intensity",
        ),
        (
            [
                ("opticalCV_raw", "Raw CV"),
                ("opticalCV_corrected", "Corrected/used CV"),
            ],
            "Optical CV syllectogram: grayscale intensity",
            "CV",
        ),
        (
            [
                ("opticalCV_raw_norm01", "Raw CV 0-1"),
                ("opticalCV_corrected_norm01", "Corrected/used CV 0-1"),
                ("CV_norm_0_1", "Corrected/used CV 0-1"),
            ],
            "Normalized CV syllectogram: grayscale intensity",
            "Normalized CV",
        ),
        (
            [
                ("MaskSASI_syl_raw_0_1", "Mask-SASI raw"),
                ("MaskSASI_syl_QC_0_1", "Mask-SASI QC"),
                ("MaskSASI_syl_0_1", "Mask-SASI raw"),
                ("MaskSASI_noPOD_syl_norm01", "Mask-SASI no-POD"),
                ("MaskSASI_PODremoved_syl_norm01", "Mask-SASI POD-removed"),
            ],
            "Mask-SASI syllectogram: final selected binary mask",
            "Mask score 0-1",
        ),
        last_panel_specs,
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10.5), sharex=True)
    plotted = False
    for ax, (specs, title, ylabel) in zip(axes.ravel(), panels):
        plotted = plot_columns(ax, specs, title, ylabel) or plotted
    if not plotted:
        axes.ravel()[0].text(0.5, 0.85, "No syllectogram curves available", ha="center", va="center", transform=axes.ravel()[0].transAxes)
    for ax in axes[-1, :]:
        ax.set_xlabel(x_label)
    fig.suptitle("Optical and mask-based syllectograms", fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return fig


OBJECT_CORRELATION_METRICS: list[tuple[str, str]] = [
    ("ObjectAggregationIndex_0_100", "ObjectAggIndex"),
    ("ObjectAggregationIndex_raw_0_100", "ObjectAggRaw"),
    ("ObjectAggregationIndex_QC_0_100", "ObjectAggQC"),
    ("ObjectAggregationIndex_QC", "ObjectAggQC"),
    ("ObjectStructureScore_0_100", "ObjectStruct"),
    ("ObjectStructureScore_raw_0_100", "ObjectStructRaw"),
    ("ObjectStructureScore_QC_0_100", "ObjectStructQC"),
    ("ObjectStructureScore_QC", "ObjectStructQC"),
    ("gAggIndex_0_100", "gAggIndex"),
    ("gSASI_0_100", "gSASI"),
    ("objectAreaFracFinal", "ObjAreaFrac"),
    ("objectAreaFrac", "ObjAreaFrac"),
    ("areaFrac", "AreaFrac"),
    ("objectNtotalFinal", "ObjNtotal"),
    ("objectNtotal", "ObjNtotal"),
    ("objectCount", "ObjCount"),
    ("nObjects", "ObjCount"),
    ("objectMeanDiamFinal", "ObjMeanDiam"),
    ("objectMeanDiam_px", "ObjMeanDiam"),
    ("meanDiam_px", "MeanDiam"),
    ("meanDiam", "MeanDiam"),
    ("objectMeanArea", "ObjMeanArea"),
    ("meanArea", "MeanArea"),
    ("objectLargestFracFinal", "ObjLargestFrac"),
    ("objectLargestFrac", "ObjLargestFrac"),
    ("largestObjectFrac", "LargestFrac"),
    ("objectLargestArea", "ObjLargestArea"),
    ("objectMaxDiam_px", "ObjMaxDiam"),
    ("objectP90Diam_px", "ObjP90Diam"),
    ("Object_Xi_QC", "ObjXiQC"),
    ("Object_Xi_mask", "ObjXi"),
    ("Object_Xi_raw_intensity", "ObjXiRaw"),
    ("correlationLength_px", "CorrLen"),
    ("Xi_px", "Xi"),
    ("objectLacunarity", "ObjLacunarity"),
    ("lacunarity16", "Lacunarity16"),
    ("objectFractalDimension", "ObjFracDim"),
    ("objectSkeletonLength", "ObjSkelLen"),
    ("aggSkelLen", "SkelLen"),
    ("objectBranchPts", "ObjBranchPts"),
    ("aggBranchPts", "BranchPts"),
    ("objectEndpointDensity", "EndPtDensity"),
    ("objectJunctionDensity", "JunctionDensity"),
    ("objectSkelDensity", "SkelDensity"),
    ("objectBranchDensity", "BranchDensity"),
    ("objectTortuosity", "Tortuosity"),
    ("opticalMean_raw", "MeanIraw"),
    ("opticalMean_corrected", "MeanIcorr"),
    ("opticalMean_raw_normMinRise", "MeanIrawRise"),
    ("opticalMean_corrected_normMinRise", "MeanIcorrRise"),
    ("opticalCV_raw", "CVraw"),
    ("opticalCV_corrected", "CVcorr"),
    ("opticalCV_raw_norm01", "CVraw01"),
    ("opticalCV_corrected_norm01", "CVcorr01"),
    ("SHI_tileCV", "SHI"),
    ("textureHeterogeneity_tileCV", "TileCV"),
    ("LF_log10_LFoverHF", "LF/HF"),
    ("LF_ratio", "LFratio"),
    ("textureEntropy", "Entropy"),
    ("textureContrast", "TextureContrast"),
    ("textureHomogeneity", "Homogeneity"),
    ("intensityCV", "IntensityCV"),
    ("correctedCV", "CorrectedCV"),
    ("sizeDistributionSlope", "SizeSlope"),
    ("sizeDistributionSlope_R2", "SlopeR2"),
    ("ObjectStructureSyl_raw_0_1", "ObjStructSyl"),
    ("ObjectStructureSyl_QC_0_1", "ObjStructSylQC"),
    ("ObjectAggregationSyl_raw_0_1", "ObjAggSyl"),
    ("ObjectAggregationSyl_QC_0_1", "ObjAggSylQC"),
]


def has_plot_data(df: pd.DataFrame, y_cols: list[str], min_finite: int = 2) -> list[str]:
    """Return columns that exist and have enough finite numeric values to plot."""
    valid: list[str] = []
    for column in y_cols:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if int(values.notna().sum()) >= int(min_finite):
            valid.append(column)
    return valid


def render_start_frame_diagnostics_section(
    df: pd.DataFrame,
    mode_label: str,
    show_unavailable_message: bool = True,
) -> None:
    """Render start-frame diagnostics independently from correlation/average plots."""
    st.subheader("How the analysis start was selected")
    if not has_start_frame_diagnostic_signals(df):
        if show_unavailable_message:
            st.info("Start-frame diagnostic signals were not available for this analysis mode or dataset.")
        return
    start_method = str(first_nonempty_value(df, "autoStartMethod") or first_nonempty_value(df, "startFrameMode") or "n/a")
    start_time = _first_finite_from_columns(df, ["autoStartTime_s", "startupQC_startTime_s"])
    start_frame = _first_finite_from_columns(df, ["autoStart_originalImageIndex", "finalStartFrameUsed", "autoStartFrame_POD"])
    st.caption(
        f"{mode_label} start-frame diagnostic plots are independent from correlation matrices and average-behaviour plots. "
        f"Selected method: {start_method}; "
        f"start frame: {format_metric_value(start_frame, 0) if np.isfinite(start_frame) else 'Not available'}; "
        f"start time_s: {format_metric_value(start_time, 2) if np.isfinite(start_time) else 'Not available'}."
    )
    st.caption("A vertical marker shows the stored start when its position is available. For an automatic start, "
               "the stored signal and slope curves help explain the detector's decision; for a manual start, "
               "they provide context only. All stored diagnostic frames are shown, including earlier frames when "
               "available. This view does not change the start.")
    st.image(figure_png(("start", frame_key(df)), lambda: plot_auto_start_diagnostics(df)), width="stretch")


def _unique_metric_specs(metric_specs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen_columns: set[str] = set()
    seen_labels: dict[str, int] = {}
    unique: list[tuple[str, str]] = []
    for column, label in metric_specs:
        if column in seen_columns:
            continue
        seen_columns.add(column)
        count = seen_labels.get(label, 0)
        seen_labels[label] = count + 1
        display_label = label if count == 0 else f"{label}{count + 1}"
        unique.append((column, display_label))
    return unique


def _first_available_metric(df: pd.DataFrame, display_name: str, candidates: list[str]) -> dict[str, str] | None:
    """Resolve one display metric to the first available dataframe column."""
    for column in candidates:
        if column in df.columns:
            return {"column": column, "label": display_name}
    return None


def get_compact_correlation_metrics(mode: str, df: pd.DataFrame) -> list[dict[str, str]]:
    """Return compact scientific metrics for correlation analysis by mode."""
    mode_text = str(mode).lower()
    if "rbc" in mode_text or "gap" in mode_text or "gapAreaFrac" in df.columns:
        specs = [
            ("Xi", ["Xi_QC", "Xi_mask", "Xi_px"]),
            ("Xi_interp", ["Xi_interp_px"]),
            ("gapMeanDiam", ["gapMeanDiamFinal", "gapMeanDiam_px", "meanDiam"]),
            ("gapNtotal", ["gapNtotalFinal", "nObjectsFinal", "gapNtotal", "nObjects"]),
            ("gapAreaFrac", ["gapAreaFracFinal", "areaFracFinal", "gapAreaFrac", "areaFrac"]),
            ("gapAreaSlope", ["gapAreaSlope"]),
            ("LF", ["LF_log10_LFoverHF", "LF_ratio"]),
            ("SHI", ["SHI_tileCV"]),
            ("PTI", ["PTI_QC_0_100", "PTI_raw_0_100", "PTI_0_100"]),
            ("AggIndex", ["AggIndex_QC_0_100", "AggIndex_raw_0_100", "AggIndex_0_100"]),
            ("Official mask SASI", ["SASI_QC_0_100", "SASI_raw_0_100", "SASI_0_100"]),
            ("Official mask GapScore", ["GapScore_QC_0_100", "GapScore_raw_0_100", "GapScore_0_100"]),
            ("maskReliability", ["maskReliabilityScore"]),
            ("imageFieldAnisotropy", ["anisotropy"]),
            ("imageFieldOrientOrder", ["orientOrderParam"]),
            ("maskInterfaceDensity", ["interfaceDensity_px"]),
            ("clustFracGap", ["largestClusterFrac_phase"]),
            ("clustFracRBC", ["largestClusterFrac_complement"]),
            ("MeanIcorr", ["opticalMean_corrected", "opticalMeanIntensity_corrected", "MeanI_corrected"]),
            ("opticalCV", ["opticalCV_corrected", "correctedCV", "intensityCV"]),
        ]
    else:
        specs = [
            ("objectCount", ["objectNtotalFinal", "objectNtotal", "objectCount", "nObjects"]),
            ("objectMeanArea", ["objectMeanArea", "meanArea"]),
            ("objectMeanDiam", ["objectMeanDiamFinal", "objectMeanDiam_px", "meanDiam_px", "meanDiam"]),
            ("objectAreaFrac", ["objectAreaFracFinal", "objectAreaFrac", "areaFrac"]),
            ("objectDensity", ["objectDensity", "objectCountDensity"]),
            ("ObjectStructure", ["ObjectStructureScore_QC_0_100", "ObjectStructureScore_QC", "ObjectStructureScore_raw_0_100", "ObjectStructureScore_0_100"]),
            ("ObjectAggScore", ["ObjectAggregationIndex_QC_0_100", "ObjectAggregationIndex_QC", "ObjectAggregationIndex_raw_0_100", "ObjectAggregationIndex_0_100"]),
            ("objectMaskReliability", ["objectMaskReliabilityScore", "maskReliabilityScore"]),
            ("textureContrast", ["objectTextureContrast", "textureContrast"]),
            ("textureEntropy", ["objectTextureEntropy", "textureEntropy"]),
            ("objectHeterogeneity", ["textureHeterogeneity_tileCV", "objectTileCV", "SHI_tileCV"]),
            ("imageFieldAnisotropy", ["anisotropy"]),
            ("imageFieldOrientOrder", ["orientOrderParam"]),
            ("maskInterfaceDensity", ["interfaceDensity_px"]),
            ("clustFrac", ["largestClusterFrac_phase"]),
            ("clustFracComp", ["largestClusterFrac_complement"]),
            ("MeanIcorr", ["opticalMean_corrected", "opticalMeanIntensity_corrected", "MeanI_corrected"]),
            ("opticalCV", ["opticalCV_corrected", "correctedCV", "intensityCV"]),
        ]
    resolved: list[dict[str, str]] = []
    seen_columns: set[str] = set()
    for display_name, candidates in specs:
        item = _first_available_metric(df, display_name, candidates)
        if item and item["column"] not in seen_columns:
            seen_columns.add(item["column"])
            resolved.append(item)
    return resolved


def get_full_numeric_metric_columns(df: pd.DataFrame) -> list[dict[str, str]]:
    """Return numeric metric-like columns while excluding IDs, labels, flags, and indices."""
    excluded_tokens = [
        "filename",
        "filepath",
        "folder",
        "sample",
        "donor",
        "group",
        "condition",
        "status",
        "mode",
        "method",
        "source",
        "label",
        "reason",
        "warning",
        "flag",
        "path",
        "csv",
        "name",
        "key",
        "notes",
        "preset",
    ]
    excluded_exact = {
        "index",
        "frameindex",
        "analysisindex_0based",
        "analysisindex_1based",
        "originalimageindex_0based",
        "originalimageindex_1based",
        "matlabtime_s",
        "time_s",
        "autostartframe",
        "autostartframe_pod",
        "autostarttime_s",
        "finalstartframeused",
        "nframesanalysed",
        "nframesusedforsummary",
        "nimagesloaded",
        "filesloaded",
    }
    results: list[dict[str, str]] = []
    for column in df.columns:
        lower = str(column).lower()
        compact_lower = re.sub(r"[^a-z0-9_]+", "", lower)
        if compact_lower in excluded_exact:
            continue
        if any(token in lower for token in excluded_tokens):
            continue
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            continue
        values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.notna().any():
            results.append({"column": str(column), "label": str(column)})
    return results


def _first_bool_from_column(df: pd.DataFrame, column: str, default: bool) -> bool:
    if column not in df.columns:
        return default
    values = df[column].dropna()
    if values.empty:
        return default
    value = values.iloc[0]
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "on"}
    return bool(value)


def apply_trusted_frame_filter_for_correlation(
    df: pd.DataFrame,
    mode: str,
    qc_mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply raw/trusted/included frame filtering for correlation analysis."""
    metadata: dict[str, Any] = {
        "qcMode": qc_mode,
        "nFramesTotal": int(len(df)),
        "excludePreStartFramesFromTrustedAnalysis": _first_bool_from_column(df, "excludePreStartFramesFromTrustedAnalysis", True),
        "excludeBadPostStartQCFramesFromTrustedAnalysis": _first_bool_from_column(df, "excludeBadPostStartQCFramesFromTrustedAnalysis", False),
    }
    if qc_mode == "All analysed frames":
        filtered = df.copy()
    elif qc_mode == "Included summary frames":
        summary_col = next((col for col in ["objectUsedInSummary", "usedInSummary", "usedForMainSummary", "includedInMainSummary"] if col in df.columns), None)
        filtered = df.loc[df[summary_col].fillna(False).astype(bool)].copy() if summary_col else pd.DataFrame(columns=df.columns)
    else:
        mode_text = str(mode).lower()
        if "rbc" in mode_text or "gap" in mode_text or "gapAreaFrac" in df.columns:
            trusted_col = next((col for col in ["trustedForScore", "score_isTrusted", "rbcFrameReliable"] if col in df.columns), None)
            if trusted_col:
                mask = df[trusted_col].fillna(False).astype(bool)
            else:
                hard = df.get("hardBadMaskFlag", pd.Series(False, index=df.index)).fillna(False).astype(bool)
                huge = df.get("hugeGapFlag", pd.Series(False, index=df.index)).fillna(False).astype(bool)
                before = df.get("beforeStartFrame", pd.Series(False, index=df.index)).fillna(False).astype(bool)
                mask = ~(hard | huge)
                if metadata["excludePreStartFramesFromTrustedAnalysis"]:
                    mask = mask & ~before
            filtered = df.loc[mask].copy()
        else:
            hard = df.get("hardBadObjectMaskFlag", pd.Series(False, index=df.index)).fillna(False).astype(bool)
            huge = df.get("objectHugeObjectFlag", df.get("hugeObjectFlag", pd.Series(False, index=df.index))).fillna(False).astype(bool)
            mask = ~(hard | huge)
            trust_cols = [col for col in ["ObjectStructure_isTrusted", "ObjectAggregation_isTrusted", "Object_Xi_isTrusted", "objectMaskReliable"] if col in df.columns]
            if trust_cols:
                mask = mask & df[trust_cols].fillna(False).astype(bool).any(axis=1)
            filtered = df.loc[mask].copy()
    metadata["nFramesUsed"] = int(len(filtered))
    return filtered, metadata


def clean_metric_table_for_correlation(
    df: pd.DataFrame,
    metric_specs: list[dict[str, str]],
    min_observations: int = 3,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Drop metrics with all NaN, too few observations, or zero variance."""
    columns: dict[str, pd.Series] = {}
    labels: list[str] = []
    label_by_column: dict[str, str] = {}
    seen_labels: dict[str, int] = {}
    for spec in metric_specs:
        column = spec["column"]
        if column not in df.columns or column in columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = values.dropna()
        if len(finite) < int(min_observations) or finite.nunique(dropna=True) < 2:
            continue
        label = str(spec.get("label", spec.get("display_name", column)))
        count = seen_labels.get(label, 0)
        seen_labels[label] = count + 1
        if count:
            label = f"{label}{count + 1}"
        columns[column] = values
        labels.append(label)
        label_by_column[column] = label
    table = pd.DataFrame(columns) if columns else pd.DataFrame(index=df.index)
    return table, labels, label_by_column


def compute_correlation_matrix(
    numeric_table: pd.DataFrame,
    method: str = "spearman",
    min_pairwise_n: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise correlation and valid-pair count matrices."""
    columns = list(numeric_table.columns)
    corr = pd.DataFrame(np.nan, index=columns, columns=columns, dtype="float64")
    n_pairs = pd.DataFrame(0, index=columns, columns=columns, dtype="int64")
    for col_a in columns:
        for col_b in columns:
            pair = numeric_table[[col_a, col_b]].dropna()
            n = int(len(pair))
            n_pairs.loc[col_a, col_b] = n
            if n >= int(min_pairwise_n):
                if col_a == col_b:
                    corr.loc[col_a, col_b] = 1.0
                else:
                    corr.loc[col_a, col_b] = pair[col_a].corr(pair[col_b], method=method)
    return corr, n_pairs


def plot_correlation_matrix(
    corr: pd.DataFrame,
    title: str,
    compact: bool = True,
    show_values: bool = True,
) -> plt.Figure:
    """Plot a clean -1 to +1 correlation heatmap."""
    fig_size = min(18.0, max(7.0, 0.54 * len(corr.columns) + 2.5))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.82))
    if corr.empty or len(corr.columns) < 2:
        ax.text(0.5, 0.5, "Not enough metrics for correlation matrix", ha="center", va="center", wrap=True)
        ax.axis("off")
        return fig
    matrix = corr.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad("#F2F2F2")
    image = ax.imshow(masked, vmin=-1, vmax=1, cmap=cmap, aspect="equal")
    labels = list(corr.columns)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    label_size = 8 if compact else 6
    if len(labels) <= 80:
        ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=label_size)
        ax.set_yticklabels(labels, fontsize=label_size)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.set_title(title, fontweight="bold", fontsize=10 if compact else 9)
    n_labels = len(labels)
    if show_values and n_labels <= 30:
        value_fontsize = 7.0 if n_labels <= 18 else max(4.5, 7.0 - 0.18 * (n_labels - 18))
        for row in range(n_labels):
            for col in range(n_labels):
                value = matrix[row, col]
                if not np.isfinite(value):
                    continue
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white" if abs(value) > 0.65 else "black", fontsize=value_fontsize)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _correlation_export_df(corr: pd.DataFrame, metadata: dict[str, Any], metric_list: list[str]) -> pd.DataFrame:
    """Flatten matrix export enough to carry metadata in the same CSV."""
    if corr.empty:
        return pd.DataFrame()
    out = corr.reset_index().rename(columns={"index": "metric"})
    metadata = dict(metadata)
    metadata["metricListUsed"] = "; ".join(metric_list)
    for key, value in reversed(list(metadata.items())):
        out.insert(0, key, value)
    return out


def render_framewise_correlation_section(
    df: pd.DataFrame,
    mode_label: str,
    sample_name: str,
    key_prefix: str,
    default_qc_mode: str = "All analysed frames",
) -> None:
    """Render compact/default and advanced/full frame-wise correlation matrices."""
    st.subheader("Frame-wise correlation matrix")
    st.caption("Each row is one analysed image/frame. Correlations describe how metrics co-vary over time within this image set.")
    qc_mode = st.selectbox(
        "Correlation matrix frames",
        ["All analysed frames", "Trusted/QC frames", "Included summary frames"],
        index=["All analysed frames", "Trusted/QC frames", "Included summary frames"].index(default_qc_mode),
        key=f"{key_prefix}_framewise_qc_mode",
    )
    with st.expander("Advanced correlation settings", expanded=False):
        method_label = st.selectbox("Correlation method", ["Spearman", "Pearson"], index=0, key=f"{key_prefix}_framewise_method")
        min_obs = int(st.number_input("Minimum valid frames per metric", min_value=3, max_value=max(3, len(df)), value=3, step=1, key=f"{key_prefix}_framewise_min_obs"))
        min_pairwise_n = int(st.number_input("Minimum paired observations", min_value=3, max_value=max(3, len(df)), value=3, step=1, key=f"{key_prefix}_framewise_min_pairwise"))
    method = method_label.lower()
    filtered, filter_meta = apply_trusted_frame_filter_for_correlation(df, mode_label, qc_mode)
    if len(filtered) < 3:
        st.info("Frame-wise correlation requires at least 3 valid frames after the selected filtering.")
        store_export_dataframe(f"{key_prefix}_framewise_correlation_compact_df", pd.DataFrame())
        store_export_dataframe(f"{key_prefix}_framewise_correlation_full_df", pd.DataFrame())
        return

    metadata = {
        "mode": mode_label,
        "sampleName": sample_name,
        "correlationType": "Frame-wise correlation matrix",
        "method": method_label,
        **filter_meta,
    }
    compact_specs = get_compact_correlation_metrics(mode_label, filtered)
    compact_table, compact_labels, compact_label_by_col = clean_metric_table_for_correlation(filtered, compact_specs, min_observations=min_obs)
    if compact_table.shape[1] < 2:
        st.info("Frame-wise correlation matrix skipped because fewer than 2 compact metrics had enough finite values.")
        compact_corr = pd.DataFrame()
    else:
        compact_corr, _compact_n = compute_correlation_matrix(compact_table, method=method, min_pairwise_n=min_pairwise_n)
        compact_corr = compact_corr.rename(index=compact_label_by_col, columns=compact_label_by_col)
        title = f"{mode_label} - Frame-wise {method_label} correlation, n = {len(filtered)} frames"
        st.pyplot(plot_correlation_matrix(compact_corr, title, compact=True, show_values=True), clear_figure=True)
    compact_export = _correlation_export_df(compact_corr, metadata, compact_labels) if "compact_corr" in locals() else pd.DataFrame()
    compact_key = f"{key_prefix}_framewise_correlation_compact_df"
    store_export_dataframe(compact_key, compact_export)
    render_csv_download(
        "Download compact frame-wise correlation matrix CSV",
        compact_key,
        f"framewise_correlation_compact_{safe_filename_token(mode_label)}_{safe_filename_token(sample_name)}.csv",
        f"{key_prefix}_download_framewise_correlation_compact",
    )

    with st.expander("Advanced/debug: full metric correlation matrix", expanded=False):
        full_specs = get_full_numeric_metric_columns(filtered)
        full_table, full_labels, full_label_by_col = clean_metric_table_for_correlation(filtered, full_specs, min_observations=min_obs)
        if full_table.shape[1] < 2:
            st.info("Full frame-wise correlation matrix skipped because fewer than 2 numeric metrics had enough finite values.")
            full_corr = pd.DataFrame()
        else:
            full_corr, _full_n = compute_correlation_matrix(full_table, method=method, min_pairwise_n=min_pairwise_n)
            full_corr = full_corr.rename(index=full_label_by_col, columns=full_label_by_col)
            full_title = f"{mode_label} - Full frame-wise {method_label} correlation, n = {len(filtered)} frames"
            if full_corr.shape[1] > 80:
                st.info(
                    f"Full correlation matrix has {full_corr.shape[1]} metrics, so the heatmap is skipped to avoid an oversized image. "
                    "Use the CSV export below for the full matrix."
                )
            else:
                st.pyplot(plot_correlation_matrix(full_corr, full_title, compact=False, show_values=full_corr.shape[1] <= 30), clear_figure=True)
        full_export = _correlation_export_df(full_corr, metadata, full_labels) if "full_corr" in locals() else pd.DataFrame()
        full_key = f"{key_prefix}_framewise_correlation_full_df"
        store_export_dataframe(full_key, full_export)
        render_csv_download(
            "Download full frame-wise correlation matrix CSV",
            full_key,
            f"framewise_correlation_full_{safe_filename_token(mode_label)}_{safe_filename_token(sample_name)}.csv",
            f"{key_prefix}_download_framewise_correlation_full",
        )


def _object_metric_specs_for_order(ordering: str, df: pd.DataFrame) -> list[tuple[str, str]]:
    specs = _unique_metric_specs(OBJECT_CORRELATION_METRICS)
    if ordering == "Original dataframe order":
        label_by_column = {column: label for column, label in specs}
        return [(column, label_by_column[column]) for column in df.columns if column in label_by_column]
    if ordering == "ObjectStructure-first":
        priority = ["ObjectStructureScore_0_100", "ObjectStructureScore_raw_0_100", "ObjectStructureScore_QC_0_100", "ObjectStructureScore_QC"]
    else:
        priority = ["ObjectAggregationIndex_0_100", "ObjectAggregationIndex_raw_0_100", "ObjectAggregationIndex_QC_0_100", "ObjectAggregationIndex_QC"]
    priority_set = set(priority)
    priority_specs = [spec for spec in specs if spec[0] in priority_set]
    remaining_specs = [spec for spec in specs if spec[0] not in priority_set]
    return priority_specs + remaining_specs


def filter_object_correlation_rows(df: pd.DataFrame, frame_filter: str) -> tuple[pd.DataFrame, str | None]:
    """Return object-mode rows for the selected correlation frame filter."""
    if frame_filter == "All analysed frames":
        return df.copy(), None
    if frame_filter == "Included summary frames":
        summary_col = next((col for col in ["objectUsedInSummary", "usedInSummary", "usedForMainSummary", "includedInMainSummary"] if col in df.columns), None)
        if summary_col is None:
            return pd.DataFrame(columns=df.columns), "Object correlation matrix skipped because no summary-inclusion column is available."
        mask = df[summary_col].fillna(False).astype(bool)
        return df.loc[mask].copy(), None

    hard_mask = df.get("hardBadObjectMaskFlag", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    huge_mask = df.get("objectHugeObjectFlag", df.get("hugeObjectFlag", pd.Series(False, index=df.index))).fillna(False).astype(bool)
    trusted = ~(hard_mask | huge_mask)
    trust_flags: list[pd.Series] = []
    for column in ["ObjectStructure_isTrusted", "ObjectAggregation_isTrusted", "Object_Xi_isTrusted", "objectMaskReliable"]:
        if column in df.columns:
            trust_flags.append(df[column].fillna(False).astype(bool))
    if trust_flags:
        any_trusted = trust_flags[0].copy()
        for flag in trust_flags[1:]:
            any_trusted = any_trusted | flag
        trusted = trusted & any_trusted
    elif "Object_Xi_QC" in df.columns:
        trusted = trusted & pd.to_numeric(df["Object_Xi_QC"], errors="coerce").notna()
    return df.loc[trusted].copy(), None


def build_object_correlation_outputs(
    df: pd.DataFrame,
    frame_filter: str,
    ordering: str = "ObjectAggIndex-first",
    min_finite: int = 5,
    min_metrics: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None]:
    """Build filtered rows, Spearman matrix, and pair table for Object Mode metrics."""
    filtered, filter_message = filter_object_correlation_rows(df, frame_filter)
    if filter_message:
        return filtered, pd.DataFrame(), pd.DataFrame(), filter_message
    if filtered.empty:
        return filtered, pd.DataFrame(), pd.DataFrame(), "Object correlation matrix skipped because no frames matched the selected frame filter."

    specs = _object_metric_specs_for_order(ordering, filtered)
    valid_specs: list[tuple[str, str]] = []
    numeric_columns: dict[str, pd.Series] = {}
    label_by_column: dict[str, str] = {}
    for column, label in specs:
        if column not in filtered.columns:
            continue
        values = pd.to_numeric(filtered[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if int(values.notna().sum()) < int(min_finite):
            continue
        finite_values = values.dropna()
        if finite_values.nunique(dropna=True) < 2 or float(finite_values.std(ddof=0)) <= 1e-12:
            continue
        valid_specs.append((column, label))
        numeric_columns[column] = values
        label_by_column[column] = label

    if len(valid_specs) < int(min_metrics):
        return (
            filtered,
            pd.DataFrame(),
            pd.DataFrame(),
            f"Object correlation matrix skipped because fewer than {min_metrics} metrics had enough finite values.",
        )

    numeric = pd.DataFrame(numeric_columns)
    corr = numeric.corr(method="spearman", min_periods=int(min_finite))
    ordered_columns = [column for column, _label in valid_specs]
    corr = corr.loc[ordered_columns, ordered_columns]
    corr_display = corr.rename(index=label_by_column, columns=label_by_column)

    pair_rows: list[dict[str, Any]] = []
    for i, metric_1 in enumerate(ordered_columns):
        for metric_2 in ordered_columns[i + 1 :]:
            pair_frame = numeric[[metric_1, metric_2]].dropna()
            n_pairs = int(len(pair_frame))
            rho = pair_frame[metric_1].corr(pair_frame[metric_2], method="spearman") if n_pairs >= int(min_finite) else np.nan
            pair_rows.append(
                {
                    "metric_1": label_by_column[metric_1],
                    "metric_2": label_by_column[metric_2],
                    "metric_1_column": metric_1,
                    "metric_2_column": metric_2,
                    "spearman_rho": rho,
                    "n_pairs": n_pairs,
                    "frame_filter": frame_filter,
                }
            )
    pairs = pd.DataFrame(pair_rows)
    return filtered, corr_display, pairs, None


def plot_object_spearman_correlation_matrix(corr: pd.DataFrame, frame_filter: str) -> plt.Figure:
    """Plot an ObjectAggIndex-first Spearman matrix for Object Mode metrics."""
    fig_size = max(7.5, 0.62 * len(corr.columns) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.82))
    if corr.empty or len(corr.columns) < 2:
        ax.text(0.5, 0.5, "Not enough object metrics for correlation matrix", ha="center", va="center")
        ax.axis("off")
        return fig
    matrix = corr.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#E6E6E6")
    image = ax.imshow(masked, vmin=-1, vmax=1, cmap=cmap, aspect="equal")
    labels = list(corr.columns)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=50, ha="right")
    ax.set_yticklabels(labels)
    suffix = "" if frame_filter == "All analysed frames" else f", {frame_filter.lower()}"
    ax.set_title(f"Spearman correlation matrix - Object metrics{suffix}", fontweight="bold")
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = matrix[row, col]
            if not np.isfinite(value):
                continue
            text_color = "white" if value < -0.35 or value > 0.72 else "black"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=7.5)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def render_object_correlation_matrix_section(df: pd.DataFrame) -> None:
    """Render Object Mode Spearman correlation matrix and export tables."""
    render_framewise_correlation_section(
        df,
        str(first_nonempty_value(df, "selectedObjectMethod") or "Object Mode"),
        str(first_nonempty_value(df, "sampleName") or first_nonempty_value(df, "filename") or "image_set"),
        "single_object",
    )
    st.caption(
        "Object-mode correlations are generic object/particle structural relationships. "
        "They are not RBC aggregation biomarkers."
    )


AVERAGE_STAT_OPTIONS = ["Mean +/- SD", "Mean +/- SEM", "Median + IQR"]
AVERAGE_SCALE_OPTIONS = ["Raw values", "0-1 normalized within sample", "Robust z-score"]
AVERAGE_FRAME_FILTER_OPTIONS = ["All analysed frames", "QC/trusted frames", "Included summary frames"]


def _metric_spec(display_name: str, candidates: list[str], category: str = "key") -> dict[str, Any]:
    return {"display_name": display_name, "candidates": candidates, "category": category}


def average_profile_specs(kind: str, frame_filter: str = "All analysed frames") -> list[dict[str, Any]]:
    """Return compact key metrics for average-profile plots by analysis kind."""
    trusted_scores = frame_filter in {"QC/trusted frames", "Included summary frames"}
    if kind == "rbc":
        score_prefixes = [
            ("Official mask SASI", "SASI_QC_0_100" if trusted_scores else "SASI_raw_0_100", "SASI_0_100"),
            ("AggIndex", "AggIndex_QC_0_100" if trusted_scores else "AggIndex_raw_0_100", "AggIndex_0_100"),
            ("Official mask GapScore", "GapScore_QC_0_100" if trusted_scores else "GapScore_raw_0_100", "GapScore_0_100"),
            ("PTI", "PTI_QC_0_100" if trusted_scores else "PTI_raw_0_100", "PTI_0_100"),
        ]
        specs = [_metric_spec(label, [primary, fallback], "score") for label, primary, fallback in score_prefixes]
        xi_candidates = ["Xi_QC", "Xi_mask", "Xi_interp_px", "Xi_px"] if trusted_scores else ["Xi_mask", "Xi_interp_px", "Xi_px", "Xi_QC"]
        specs.extend(
            [
                _metric_spec("Gap area fraction", ["gapAreaFracFinal", "areaFracFinal", "gapAreaFrac", "areaFrac"], "morphology"),
                _metric_spec("Gap count", ["gapNtotalFinal", "nObjectsFinal", "gapNtotal", "nObjects"], "morphology"),
                _metric_spec("Gap mean diameter", ["gapMeanDiamFinal", "gapMeanDiam_px", "meanDiam"], "morphology"),
                _metric_spec("Largest gap fraction", ["largestGapFracFinal", "gapLargestFrac", "largestObjectFrac"], "morphology"),
                _metric_spec("Xi", xi_candidates, "morphology"),
                _metric_spec("Xi interp", ["Xi_interp_px", "Xi_px"], "morphology"),
                _metric_spec("SHI tile CV", ["SHI_tileCV"], "heterogeneity"),
                _metric_spec("LF/HF", ["LF_log10_LFoverHF", "LF_ratio"], "heterogeneity"),
                _metric_spec("Optical CV", ["opticalCV_corrected", "correctedCV", "intensityCV"], "heterogeneity"),
                _metric_spec("Optical mean", ["opticalMean_corrected", "opticalMean_raw"], "heterogeneity"),
            ]
        )
        return specs
    if kind == "object":
        score_specs = [
            _metric_spec("Object structure score", ["ObjectStructureScore_QC_0_100", "ObjectStructureScore_QC", "ObjectStructureScore_raw_0_100", "ObjectStructureScore_0_100"], "score"),
            _metric_spec("Object aggregation/coarsening score", ["ObjectAggregationIndex_QC_0_100", "ObjectAggregationIndex_QC", "ObjectAggregationIndex_raw_0_100", "ObjectAggregationIndex_0_100"], "score"),
            _metric_spec("gSASI", ["gSASI_0_100"], "score"),
            _metric_spec("gAggIndex", ["gAggIndex_0_100"], "score"),
        ]
        return score_specs + [
            _metric_spec("Object area fraction", ["objectAreaFracFinal", "objectAreaFrac", "areaFrac"], "morphology"),
            _metric_spec("Object count", ["objectNtotalFinal", "objectNtotal", "objectCount", "nObjects"], "morphology"),
            _metric_spec("Object mean diameter", ["objectMeanDiamFinal", "objectMeanDiam_px", "meanDiam_px", "meanDiam"], "morphology"),
            _metric_spec("Largest object fraction", ["objectLargestFracFinal", "objectLargestFrac", "largestObjectFrac"], "morphology"),
            _metric_spec("Object Xi", ["Object_Xi_QC", "Object_Xi_mask", "correlationLength_px", "Xi_px"], "morphology"),
            _metric_spec("Object lacunarity", ["objectLacunarity", "lacunarity16"], "network"),
            _metric_spec("Fractal dimension", ["objectFractalDimension"], "network"),
            _metric_spec("Skeleton length", ["objectSkeletonLength", "aggSkelLen"], "network"),
            _metric_spec("Branch points", ["objectBranchPts", "aggBranchPts"], "network"),
            _metric_spec("Endpoint density", ["objectEndpointDensity"], "network"),
            _metric_spec("Junction density", ["objectJunctionDensity"], "network"),
            _metric_spec("Optical CV", ["opticalCV_corrected", "correctedCV", "intensityCV"], "heterogeneity"),
            _metric_spec("Optical mean", ["opticalMean_corrected", "opticalMean_raw"], "heterogeneity"),
            _metric_spec("Texture entropy", ["textureEntropy"], "heterogeneity"),
            _metric_spec("Texture contrast", ["textureContrast"], "heterogeneity"),
            _metric_spec("Texture homogeneity", ["textureHomogeneity"], "heterogeneity"),
        ]
    return [
        _metric_spec("GT area fraction", ["GT_areaFrac", "areaFrac"], "morphology"),
        _metric_spec("GT object count", ["GT_nObjects", "nObjects"], "morphology"),
        _metric_spec("GT mean diameter", ["GT_meanDiam", "meanDiam"], "morphology"),
        _metric_spec("GT Xi", ["GT_Xi_px", "Xi_px"], "morphology"),
        _metric_spec("GT lacunarity", ["GT_lacunarity16", "lacunarity16"], "heterogeneity"),
        _metric_spec("Dice", ["Dice_auto_vs_GT"], "validation"),
        _metric_spec("IoU", ["IoU_auto_vs_GT"], "validation"),
    ]


def resolve_metric_specs(df: pd.DataFrame, specs: list[dict[str, Any]], categories: set[str] | None = None) -> list[dict[str, Any]]:
    """Resolve candidate columns to one real numeric column per display metric."""
    resolved: list[dict[str, Any]] = []
    used_columns: set[str] = set()
    for spec in specs:
        category = str(spec.get("category", "key"))
        if categories and category not in categories:
            continue
        column = next((candidate for candidate in spec["candidates"] if candidate in df.columns), None)
        if column is None or column in used_columns:
            continue
        used_columns.add(column)
        resolved.append({**spec, "column": column})
    return resolved


def filter_average_profile_rows(df: pd.DataFrame, kind: str, frame_filter: str) -> pd.DataFrame:
    """Filter rows for single-sample average profiles without changing raw data."""
    if frame_filter == "All analysed frames":
        return df.copy()
    if frame_filter == "Included summary frames":
        summary_col = next((col for col in ["objectUsedInSummary", "usedInSummary", "usedForMainSummary", "includedInMainSummary"] if col in df.columns), None)
        if summary_col:
            return df.loc[df[summary_col].fillna(False).astype(bool)].copy()
        return pd.DataFrame(columns=df.columns)
    if kind == "rbc":
        trusted_col = next((col for col in ["trustedForScore", "score_isTrusted", "rbcFrameReliable"] if col in df.columns), None)
        if trusted_col:
            return df.loc[df[trusted_col].fillna(False).astype(bool)].copy()
        hard = df.get("hardBadMaskFlag", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        huge = df.get("hugeGapFlag", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        return df.loc[~(hard | huge)].copy()
    hard = df.get("hardBadObjectMaskFlag", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    huge = df.get("objectHugeObjectFlag", df.get("hugeObjectFlag", pd.Series(False, index=df.index))).fillna(False).astype(bool)
    trusted = ~(hard | huge)
    trust_cols = [col for col in ["ObjectStructure_isTrusted", "ObjectAggregation_isTrusted", "Object_Xi_isTrusted", "objectMaskReliable"] if col in df.columns]
    if trust_cols:
        any_trusted = df[trust_cols].fillna(False).astype(bool).any(axis=1)
        trusted = trusted & any_trusted
    return df.loc[trusted].copy()


def normalize_profile_values(values: pd.Series, normalization_type: str) -> pd.Series:
    """Normalize one metric series for cross-metric profile plots."""
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if normalization_type == "0-1 normalized within sample":
        finite = x.dropna()
        if finite.empty:
            return x
        lo = float(finite.min())
        hi = float(finite.max())
        if abs(hi - lo) <= 1e-12:
            return pd.Series(0.5, index=x.index, dtype="float64").where(x.notna(), np.nan)
        return (x - lo) / (hi - lo)
    if normalization_type == "Robust z-score":
        finite = x.dropna()
        if finite.empty:
            return x
        med = float(np.nanmedian(finite))
        mad = float(np.nanmedian(np.abs(finite - med)))
        if mad > 1e-12:
            return (x - med) / (1.4826 * mad + 1e-12)
        sd = float(np.nanstd(finite, ddof=0))
        if sd > 1e-12:
            return (x - float(np.nanmean(finite))) / sd
        return pd.Series(0.0, index=x.index, dtype="float64").where(x.notna(), np.nan)
    return x


def build_average_metric_profile_df(
    df: pd.DataFrame,
    specs: list[dict[str, Any]],
    mode_label: str,
    frame_filter: str,
    statistic: str,
    normalization_type: str,
) -> pd.DataFrame:
    """Return average metric profile rows with central tendency and variability."""
    rows: list[dict[str, Any]] = []
    for spec in specs:
        column = spec["column"]
        raw_values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if raw_values.empty:
            continue
        values = normalize_profile_values(raw_values, normalization_type)
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        n = int(values.shape[0])
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if n >= 2 else np.nan
        sem = float(sd / np.sqrt(n)) if n >= 2 and np.isfinite(sd) else np.nan
        median = float(values.median())
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = float(q3 - q1)
        raw_mean = float(raw_values.mean())
        cv = float(sd / abs(mean)) if n >= 2 and np.isfinite(sd) and abs(mean) > 1e-12 else np.nan
        if statistic == "Median + IQR":
            plot_value = median
            err_lower = max(0.0, median - q1)
            err_upper = max(0.0, q3 - median)
        elif statistic == "Mean +/- SEM":
            plot_value = mean
            err_lower = sem
            err_upper = sem
        else:
            plot_value = mean
            err_lower = sd
            err_upper = sd
        rows.append(
            {
                "metric_name": column,
                "display_name": str(spec["display_name"]),
                "category": str(spec.get("category", "key")),
                "mode": mode_label,
                "frame_filter": frame_filter,
                "n": n,
                "mean": mean,
                "sd": sd,
                "sem": sem,
                "median": median,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "cv": cv,
                "raw_mean_before_normalization": raw_mean,
                "plot_value": plot_value,
                "error_lower": err_lower,
                "error_upper": err_upper,
                "normalization_type": normalization_type,
                "statistic": statistic,
                "insufficient_variability": n < 2,
            }
        )
    return pd.DataFrame(rows)


def plot_average_metric_profile(profile_df: pd.DataFrame, title: str, x_label: str) -> plt.Figure:
    """Plot average metric profile with horizontal error bars."""
    if profile_df.empty:
        return empty_batch_figure("Average metric profile skipped because fewer than two metrics had finite values.")
    plot_df = profile_df.copy().reset_index(drop=True)
    y = np.arange(len(plot_df))
    x = pd.to_numeric(plot_df["plot_value"], errors="coerce").to_numpy(dtype=float)
    err_low = pd.to_numeric(plot_df["error_lower"], errors="coerce").to_numpy(dtype=float)
    err_high = pd.to_numeric(plot_df["error_upper"], errors="coerce").to_numpy(dtype=float)
    err_low = np.where(np.isfinite(err_low), err_low, 0.0)
    err_high = np.where(np.isfinite(err_high), err_high, 0.0)
    fig_height = max(4.2, 0.42 * len(plot_df) + 1.4)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    ax.barh(y, x, xerr=np.vstack([err_low, err_high]), color="#2f6f8f", alpha=0.86, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["display_name"].astype(str).tolist())
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def average_profile_caption(statistic: str, frame_filter: str, n_rows: int, normalization_type: str, unit: str = "frames") -> str:
    stat_text = {"Mean +/- SD": "mean +/- SD", "Mean +/- SEM": "mean +/- SEM", "Median + IQR": "median + IQR"}.get(statistic, statistic)
    if frame_filter == "QC/trusted frames":
        frame_text = "Only frames passing selected QC/trusted-frame settings are used."
    elif frame_filter == "Included summary frames":
        frame_text = "Only frames included in the summary statistics are used."
    else:
        frame_text = "All calculable analysed frames are used."
    norm_text = " Normalized values are used only for visual comparison between metrics with different units." if normalization_type != "Raw values" else ""
    return f"Values show {stat_text} across {n_rows} {unit}. {frame_text}{norm_text}"


def render_average_metric_profile_section(
    df: pd.DataFrame,
    kind: str,
    mode_label: str,
    key_prefix: str,
    default_frame_filter: str | None = None,
) -> pd.DataFrame:
    """Render single image-set average metric profile and export table."""
    st.subheader("Average key metrics with variability")
    if default_frame_filter is None:
        default_frame_filter = "Included summary frames" if any(col in df.columns for col in ["usedInSummary", "usedForMainSummary", "includedInMainSummary"]) else "All analysed frames"
    frame_filter = st.selectbox(
        "Average metric frames",
        AVERAGE_FRAME_FILTER_OPTIONS,
        index=AVERAGE_FRAME_FILTER_OPTIONS.index(default_frame_filter) if default_frame_filter in AVERAGE_FRAME_FILTER_OPTIONS else 0,
        key=f"{key_prefix}_average_metric_frames",
    )
    metric_set = st.selectbox(
        "Average profile metric set",
        ["Compact key metrics", "Scores only", "Morphology/heterogeneity only", "Advanced custom"],
        index=0,
        key=f"{key_prefix}_average_metric_set",
    )
    statistic = st.selectbox("Average metric summary statistic", AVERAGE_STAT_OPTIONS, index=0, key=f"{key_prefix}_average_metric_statistic")
    specs_all = average_profile_specs(kind, frame_filter)
    categories: set[str] | None = None
    if metric_set == "Scores only":
        categories = {"score", "validation"}
    elif metric_set == "Morphology/heterogeneity only":
        categories = {"morphology", "heterogeneity", "network"}
    resolved = resolve_metric_specs(df, specs_all, categories=categories)
    if metric_set == "Advanced custom":
        available_labels = [f"{item['display_name']} ({item['column']})" for item in resolved]
        selected_labels = st.multiselect(
            "Advanced metrics",
            available_labels,
            default=available_labels[: min(12, len(available_labels))],
            key=f"{key_prefix}_average_metric_custom",
        )
        selected_set = set(selected_labels)
        resolved = [item for item in resolved if f"{item['display_name']} ({item['column']})" in selected_set]
    all_scores = bool(resolved) and all(str(item.get("category")) in {"score", "validation"} for item in resolved)
    default_scale = "Raw values" if all_scores else "0-1 normalized within sample"
    scale = st.selectbox(
        "Average metric plot scale",
        AVERAGE_SCALE_OPTIONS,
        index=AVERAGE_SCALE_OPTIONS.index(default_scale),
        key=f"{key_prefix}_average_metric_scale",
    )
    filtered = filter_average_profile_rows(df, kind, frame_filter)
    if filtered.empty:
        st.info("Average metric profile skipped because no frames matched the selected frame filter.")
        profile_df = pd.DataFrame()
    else:
        profile_df = build_average_metric_profile_df(filtered, resolved, mode_label, frame_filter, statistic, scale)
        if len(profile_df) < 2:
            st.info("Average metric profile skipped because fewer than two metrics had finite values.")
        else:
            title_kind = "RBC" if kind == "rbc" else "object" if kind == "object" else "ground-truth"
            title = f"Average {title_kind} key metrics with variability - {frame_filter.lower()}"
            st.pyplot(plot_average_metric_profile(profile_df, title, scale), clear_figure=True)
            st.caption(average_profile_caption(statistic, frame_filter, len(filtered), scale))
            if kind == "object":
                st.caption("These are generic object structural metrics and scores. They are not RBC-specific biomarkers.")
    state_key = f"{key_prefix}_average_metric_profile_df"
    store_export_dataframe(state_key, profile_df)
    render_csv_download(
        "Download average metric profile CSV",
        state_key,
        "average_metric_profile.csv",
        f"{key_prefix}_download_average_metric_profile_csv",
    )
    return profile_df


def summary_profile_specs(kind: str, basis: str = "mean") -> list[dict[str, Any]]:
    """Return sample-summary metric candidates for batch/multi-sample average behaviour."""
    prefix_order = [basis, "mean" if basis == "median" else "median", "final"]

    def candidates(metric: str, extras: list[str] | None = None) -> list[str]:
        cols = [f"{prefix}_{metric}" for prefix in prefix_order]
        if extras:
            cols.extend(extras)
        return cols

    if kind == "rbc":
        return [
            _metric_spec("SASI", candidates("SASI_0_100"), "score"),
            _metric_spec("AggIndex", candidates("AggIndex_0_100"), "score"),
            _metric_spec("GapScore", candidates("GapScore_0_100"), "score"),
            _metric_spec("PTI", candidates("PTI_0_100"), "score"),
            _metric_spec("Xi interp", candidates("Xi_interp_px"), "morphology"),
            _metric_spec("Gap mean diameter", candidates("gapMeanDiam_px"), "morphology"),
            _metric_spec("Gap count", candidates("gapNtotal"), "morphology"),
            _metric_spec("Gap area fraction", candidates("gapAreaFrac"), "morphology"),
            _metric_spec("LF/HF", candidates("LF_log10_LFoverHF"), "heterogeneity"),
            _metric_spec("SHI tile CV", candidates("SHI_tileCV"), "heterogeneity"),
            _metric_spec("Gap area slope", candidates("gapAreaSlope"), "heterogeneity"),
        ]
    return [
        _metric_spec("Object structure score", candidates("ObjectStructureScore", ["medianObjectStructureScore", "meanObjectStructureScore"]), "score"),
        _metric_spec("Object aggregation/coarsening score", candidates("ObjectAggregationIndex", ["medianObjectAggregationIndex", "meanObjectAggregationIndex"]), "score"),
        _metric_spec("gSASI", candidates("gSASI_0_100"), "score"),
        _metric_spec("gAggIndex", candidates("gAggIndex_0_100"), "score"),
        _metric_spec("Object area fraction", candidates("areaFrac"), "morphology"),
        _metric_spec("Object count", candidates("objectCount"), "morphology"),
        _metric_spec("Object mean diameter", candidates("meanDiam_px"), "morphology"),
        _metric_spec("Correlation length", candidates("correlationLength_px"), "morphology"),
        _metric_spec("Texture heterogeneity", candidates("textureHeterogeneity_tileCV"), "heterogeneity"),
        _metric_spec("Lacunarity", candidates("lacunarity16"), "heterogeneity"),
    ]


def infer_summary_kind(summary_df: pd.DataFrame) -> str:
    """Infer whether a sample-summary table is RBC or generic Object-like."""
    columns = set(summary_df.columns)
    if any(col in columns for col in ["mean_SASI_0_100", "median_SASI_0_100", "mean_AggIndex_0_100", "median_AggIndex_0_100"]):
        return "rbc"
    return "object"


def build_sample_mean_correlation_outputs(
    summary_df: pd.DataFrame,
    kind: str,
    basis: str = "mean",
    min_samples: int = 3,
    min_finite: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None]:
    """Build sample-mean/median Spearman matrix and pair table."""
    if len(summary_df) < int(min_samples):
        return summary_df.copy(), pd.DataFrame(), pd.DataFrame(), f"Sample-mean correlation requires at least {min_samples} samples."
    specs = resolve_metric_specs(summary_df, summary_profile_specs(kind, basis))
    numeric_columns: dict[str, pd.Series] = {}
    label_by_column: dict[str, str] = {}
    for spec in specs:
        column = spec["column"]
        values = pd.to_numeric(summary_df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = values.dropna()
        if len(finite) < int(min_finite) or finite.nunique(dropna=True) < 2:
            continue
        numeric_columns[column] = values
        label_by_column[column] = str(spec["display_name"])
    if len(numeric_columns) < 3:
        return summary_df.copy(), pd.DataFrame(), pd.DataFrame(), "Sample-mean correlation skipped because fewer than 3 metrics had enough finite sample values."
    numeric = pd.DataFrame(numeric_columns)
    corr = numeric.corr(method="spearman", min_periods=int(min_finite))
    corr_display = corr.rename(index=label_by_column, columns=label_by_column)
    ordered_columns = list(numeric_columns.keys())
    pair_rows: list[dict[str, Any]] = []
    for i, metric_1 in enumerate(ordered_columns):
        for metric_2 in ordered_columns[i + 1 :]:
            pair_frame = numeric[[metric_1, metric_2]].dropna()
            n_pairs = int(len(pair_frame))
            rho = pair_frame[metric_1].corr(pair_frame[metric_2], method="spearman") if n_pairs >= int(min_finite) else np.nan
            pair_rows.append(
                {
                    "metric_1": label_by_column[metric_1],
                    "metric_2": label_by_column[metric_2],
                    "metric_1_column": metric_1,
                    "metric_2_column": metric_2,
                    "spearman_rho": rho,
                    "n_pairs": n_pairs,
                    "summary_basis": basis,
                }
            )
    return summary_df.copy(), corr_display, pd.DataFrame(pair_rows), None


def plot_sample_mean_correlation_matrix(corr: pd.DataFrame, basis: str) -> plt.Figure:
    """Plot sample-level Spearman matrix."""
    fig_size = max(7.5, 0.62 * len(corr.columns) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.82))
    if corr.empty or len(corr.columns) < 2:
        ax.text(0.5, 0.5, "Not enough samples/metrics for sample-mean correlation", ha="center", va="center")
        ax.axis("off")
        return fig
    matrix = corr.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#E6E6E6")
    image = ax.imshow(masked, vmin=-1, vmax=1, cmap=cmap, aspect="equal")
    labels = list(corr.columns)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=50, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(f"Sample-{basis} Spearman correlation matrix", fontweight="bold")
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = matrix[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white" if value < -0.35 or value > 0.72 else "black", fontsize=7.5)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def render_sample_average_behaviour_section(
    summary_df: pd.DataFrame,
    key_prefix: str,
    kind: str | None = None,
    title_prefix: str = "Batch",
) -> None:
    """Render sample-level average profiles, group profiles, and sample-mean correlations."""
    if summary_df.empty:
        st.info("Average behaviour plots need a non-empty sample summary table.")
        return
    kind = kind or infer_summary_kind(summary_df)
    st.subheader("Average key metrics with variability")
    basis = st.selectbox("Sample metric basis", ["mean", "median"], index=0, key=f"{key_prefix}_sample_profile_basis")
    statistic = st.selectbox("Average metric summary statistic", AVERAGE_STAT_OPTIONS, index=0, key=f"{key_prefix}_sample_profile_stat")
    scale = st.selectbox("Average metric plot scale", AVERAGE_SCALE_OPTIONS, index=1, key=f"{key_prefix}_sample_profile_scale")
    specs = resolve_metric_specs(summary_df, summary_profile_specs(kind, basis))
    profile_df = build_average_metric_profile_df(summary_df, specs, title_prefix, "Sample rows", statistic, scale)
    profile_key = f"{key_prefix}_sample_mean_metric_profile_df"
    store_export_dataframe(profile_key, profile_df)
    if len(profile_df) < 2:
        st.info("Average metric profile skipped because fewer than two sample metrics had finite values.")
    else:
        st.pyplot(plot_average_metric_profile(profile_df, f"{title_prefix} sample-average key metrics with variability", scale), clear_figure=True)
        st.caption(f"Values show {statistic.lower().replace('+/-', '+/-')} across {len(summary_df)} samples. Error bars are across samples.")
        st.caption("Normalized values are used only for visual comparison between metrics with different units." if scale != "Raw values" else "Raw units are shown.")
    render_csv_download(
        "Download sample mean metric profile CSV",
        profile_key,
        "sample_mean_metric_profile.csv",
        f"{key_prefix}_download_sample_mean_metric_profile_csv",
    )

    group_candidates = [col for col in ["flowGroup", "flowHistory", "stentCondition", "stentStatus", "condition"] if col in summary_df.columns]
    if {"flowGroup", "stentCondition"}.issubset(summary_df.columns):
        tmp = summary_df.copy()
        tmp["flowGroup x stentCondition"] = (tmp["flowGroup"].fillna("").astype(str) + " " + tmp["stentCondition"].fillna("").astype(str)).str.strip()
        group_source_df = tmp
        group_candidates.append("flowGroup x stentCondition")
    else:
        group_source_df = summary_df
    group_profile_all = pd.DataFrame()
    if group_candidates:
        group_col = st.selectbox("Group average metric profile", group_candidates, index=0, key=f"{key_prefix}_group_profile_column")
        group_values = [value for value in group_source_df[group_col].fillna("unknown").astype(str).unique().tolist() if value]
        selected_group = st.selectbox("Group shown", group_values, index=0, key=f"{key_prefix}_group_profile_value") if group_values else ""
        rows: list[pd.DataFrame] = []
        for group_value, group_df in group_source_df.groupby(group_col, dropna=False, sort=False):
            group_label = str(group_value) if str(group_value).strip() else "unknown"
            group_profile = build_average_metric_profile_df(group_df, specs, f"{title_prefix} group: {group_label}", f"{group_col}={group_label}", statistic, scale)
            if not group_profile.empty:
                group_profile["group_column"] = group_col
                group_profile["group_value"] = group_label
                rows.append(group_profile)
        group_profile_all = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        group_key = f"{key_prefix}_group_average_metric_profile_df"
        store_export_dataframe(group_key, group_profile_all)
        if selected_group and not group_profile_all.empty:
            selected_profile = group_profile_all.loc[group_profile_all["group_value"].astype(str) == str(selected_group)].copy()
            if len(selected_profile) >= 2:
                st.pyplot(plot_average_metric_profile(selected_profile, f"Group-average key metrics - {selected_group}", scale), clear_figure=True)
                st.caption("Group error bars are across samples in the selected group, not across frames.")
            else:
                st.info("Group average metric profile skipped because the selected group has fewer than two finite metrics.")
        render_csv_download(
            "Download group average metric profile CSV",
            group_key,
            "group_average_metric_profile.csv",
            f"{key_prefix}_download_group_average_metric_profile_csv",
        )
    else:
        st.caption("Group average metric profile skipped because no group labels were available.")

    st.subheader("Sample-mean correlation matrix")
    st.caption("Each row is one sample/image set, using the selected mean or median metric value for that sample. This describes average-behaviour relationships across samples.")
    with st.expander("Advanced sample-mean correlation settings", expanded=False):
        method_label = st.selectbox("Correlation method", ["Spearman", "Pearson"], index=0, key=f"{key_prefix}_sample_mean_corr_method")
        min_sample_obs = int(
            st.number_input(
                "Minimum valid samples per metric",
                min_value=3,
                max_value=max(3, len(summary_df)),
                value=3,
                step=1,
                key=f"{key_prefix}_sample_mean_min_obs",
            )
        )
    method = method_label.lower()
    mode_label = "RBC Gap Mode" if kind == "rbc" else "Object Mode"
    compact_sample_specs = resolve_metric_specs(summary_df, summary_profile_specs(kind, basis))
    sample_metric_cols = [spec["column"] for spec in compact_sample_specs]
    sample_metric_table = summary_df[[col for col in ["sampleName", "sampleKey", "donor", "flowGroup", "flowHistory", "stentCondition", "stentStatus"] if col in summary_df.columns] + sample_metric_cols].copy() if sample_metric_cols else pd.DataFrame()
    if not sample_metric_table.empty:
        sample_metric_table.insert(0, "summaryBasis", basis)
        sample_metric_table.insert(0, "mode", mode_label)
    metric_table_key = f"{key_prefix}_sample_mean_metric_table_df"
    store_export_dataframe(metric_table_key, sample_metric_table)
    render_csv_download(
        "Download sample-mean metric table CSV",
        metric_table_key,
        f"sample_mean_metric_table_{safe_filename_token(title_prefix)}.csv",
        f"{key_prefix}_download_sample_mean_metric_table_csv",
    )

    metadata = {
        "mode": mode_label,
        "batchName": title_prefix,
        "correlationType": "Sample-mean correlation matrix",
        "method": method_label,
        "summaryBasis": basis,
        "nSamplesTotal": int(len(summary_df)),
    }
    compact_table, compact_labels, compact_label_by_col = clean_metric_table_for_correlation(summary_df, compact_sample_specs, min_observations=min_sample_obs)
    n_compact_samples = int(compact_table.notna().any(axis=1).sum()) if not compact_table.empty else 0
    metadata["nSamplesUsed"] = n_compact_samples
    compact_corr = pd.DataFrame()
    if len(summary_df) < 3 or n_compact_samples < 3:
        st.info("Sample-mean correlation requires at least 3 valid image sets/samples.")
    elif compact_table.shape[1] < 2:
        st.info("Sample-mean correlation skipped because fewer than 2 compact metrics had enough finite sample values.")
    else:
        compact_corr, _compact_n = compute_correlation_matrix(compact_table, method=method, min_pairwise_n=3)
        compact_corr = compact_corr.rename(index=compact_label_by_col, columns=compact_label_by_col)
        st.pyplot(
            plot_correlation_matrix(
                compact_corr,
                f"{mode_label} - Sample-{basis} {method_label} correlation, n = {n_compact_samples} samples",
                compact=True,
                show_values=True,
            ),
            clear_figure=True,
        )
    compact_key = f"{key_prefix}_sample_mean_correlation_compact_df"
    store_export_dataframe(compact_key, _correlation_export_df(compact_corr, metadata, compact_labels))
    render_csv_download(
        "Download compact sample-mean correlation matrix CSV",
        compact_key,
        f"sample_mean_correlation_compact_{safe_filename_token(title_prefix)}.csv",
        f"{key_prefix}_download_sample_mean_correlation_compact_csv",
    )

    with st.expander("Advanced/debug: full sample-mean metric correlation matrix", expanded=False):
        full_specs = get_full_numeric_metric_columns(summary_df)
        full_table, full_labels, full_label_by_col = clean_metric_table_for_correlation(summary_df, full_specs, min_observations=min_sample_obs)
        n_full_samples = int(full_table.notna().any(axis=1).sum()) if not full_table.empty else 0
        full_metadata = {**metadata, "nSamplesUsed": n_full_samples, "metricScope": "full numeric metrics"}
        full_corr = pd.DataFrame()
        if len(summary_df) < 3 or n_full_samples < 3:
            st.info("Sample-mean correlation requires at least 3 valid image sets/samples.")
        elif full_table.shape[1] < 2:
            st.info("Full sample-mean correlation skipped because fewer than 2 numeric metrics had enough finite sample values.")
        else:
            full_corr, _full_n = compute_correlation_matrix(full_table, method=method, min_pairwise_n=3)
            full_corr = full_corr.rename(index=full_label_by_col, columns=full_label_by_col)
            if full_corr.shape[1] > 80:
                st.info(
                    f"Full sample-mean correlation matrix has {full_corr.shape[1]} metrics, so the heatmap is skipped to avoid an oversized image. "
                    "Use the CSV export below for the full matrix."
                )
            else:
                st.pyplot(
                    plot_correlation_matrix(
                        full_corr,
                        f"{mode_label} - Full sample-{basis} {method_label} correlation, n = {n_full_samples} samples",
                        compact=False,
                        show_values=False,
                    ),
                    clear_figure=True,
                )
        full_key = f"{key_prefix}_sample_mean_correlation_full_df"
        store_export_dataframe(full_key, _correlation_export_df(full_corr, full_metadata, full_labels))
        render_csv_download(
            "Download full sample-mean correlation matrix CSV",
            full_key,
            f"sample_mean_correlation_full_{safe_filename_token(title_prefix)}.csv",
            f"{key_prefix}_download_sample_mean_correlation_full_csv",
        )


def store_default_average_behaviour_exports(summary_df: pd.DataFrame, key_prefix: str, kind: str, title_prefix: str) -> None:
    """Store default average-behaviour export tables for batch/multi-sample export tabs."""
    if summary_df.empty:
        for suffix in [
            "sample_mean_metric_profile_df",
            "sample_mean_metric_table_df",
            "group_average_metric_profile_df",
            "sample_mean_correlation_compact_df",
            "sample_mean_correlation_full_df",
        ]:
            store_export_dataframe(f"{key_prefix}_{suffix}", pd.DataFrame())
        return
    basis = "mean"
    statistic = "Mean +/- SD"
    scale = "0-1 normalized within sample"
    specs = resolve_metric_specs(summary_df, summary_profile_specs(kind, basis))
    profile_df = build_average_metric_profile_df(summary_df, specs, title_prefix, "Sample rows", statistic, scale)
    store_export_dataframe(f"{key_prefix}_sample_mean_metric_profile_df", profile_df)
    sample_metric_cols = [spec["column"] for spec in specs]
    id_cols = [col for col in ["sampleName", "sampleKey", "donor", "flowGroup", "flowHistory", "stentCondition", "stentStatus"] if col in summary_df.columns]
    sample_metric_table = summary_df[id_cols + sample_metric_cols].copy() if sample_metric_cols else pd.DataFrame()
    if not sample_metric_table.empty:
        sample_metric_table.insert(0, "summaryBasis", basis)
        sample_metric_table.insert(0, "mode", "RBC Gap Mode" if kind == "rbc" else "Object Mode")
    store_export_dataframe(f"{key_prefix}_sample_mean_metric_table_df", sample_metric_table)
    metadata = {
        "mode": "RBC Gap Mode" if kind == "rbc" else "Object Mode",
        "batchName": title_prefix,
        "correlationType": "Sample-mean correlation matrix",
        "method": "Spearman",
        "summaryBasis": basis,
        "nSamplesTotal": int(len(summary_df)),
    }
    compact_table, compact_labels, compact_label_by_col = clean_metric_table_for_correlation(summary_df, specs, min_observations=3)
    n_compact = int(compact_table.notna().any(axis=1).sum()) if not compact_table.empty else 0
    compact_corr = pd.DataFrame()
    if len(summary_df) >= 3 and n_compact >= 3 and compact_table.shape[1] >= 2:
        compact_corr, _compact_n = compute_correlation_matrix(compact_table, method="spearman", min_pairwise_n=3)
        compact_corr = compact_corr.rename(index=compact_label_by_col, columns=compact_label_by_col)
    store_export_dataframe(f"{key_prefix}_sample_mean_correlation_compact_df", _correlation_export_df(compact_corr, {**metadata, "nSamplesUsed": n_compact}, compact_labels))
    full_specs = get_full_numeric_metric_columns(summary_df)
    full_table, full_labels, full_label_by_col = clean_metric_table_for_correlation(summary_df, full_specs, min_observations=3)
    n_full = int(full_table.notna().any(axis=1).sum()) if not full_table.empty else 0
    full_corr = pd.DataFrame()
    if len(summary_df) >= 3 and n_full >= 3 and full_table.shape[1] >= 2:
        full_corr, _full_n = compute_correlation_matrix(full_table, method="spearman", min_pairwise_n=3)
        full_corr = full_corr.rename(index=full_label_by_col, columns=full_label_by_col)
    store_export_dataframe(f"{key_prefix}_sample_mean_correlation_full_df", _correlation_export_df(full_corr, {**metadata, "nSamplesUsed": n_full, "metricScope": "full numeric metrics"}, full_labels))
    group_candidates = [col for col in ["flowGroup", "flowHistory", "stentCondition", "stentStatus", "condition"] if col in summary_df.columns]
    group_source_df = summary_df.copy()
    if {"flowGroup", "stentCondition"}.issubset(group_source_df.columns):
        group_source_df["flowGroup x stentCondition"] = (
            group_source_df["flowGroup"].fillna("").astype(str) + " " + group_source_df["stentCondition"].fillna("").astype(str)
        ).str.strip()
        group_candidates.append("flowGroup x stentCondition")
    rows: list[pd.DataFrame] = []
    if group_candidates:
        group_col = group_candidates[0]
        for group_value, group_df in group_source_df.groupby(group_col, dropna=False, sort=False):
            group_label = str(group_value) if str(group_value).strip() else "unknown"
            group_profile = build_average_metric_profile_df(group_df, specs, f"{title_prefix} group: {group_label}", f"{group_col}={group_label}", statistic, scale)
            if not group_profile.empty:
                group_profile["group_column"] = group_col
                group_profile["group_value"] = group_label
                rows.append(group_profile)
    group_profile_all = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    store_export_dataframe(f"{key_prefix}_group_average_metric_profile_df", group_profile_all)


def render_structural_architecture_section(df: pd.DataFrame, plot_x_axis: str) -> None:
    """Frame-wise panel for the extra structural-architecture biomarkers:
    grayscale-field directionality, mask interface density, and mask-phase
    connectivity / percolation."""
    arch_cols = [
        "anisotropy",
        "orientOrderParam",
        "interfaceDensity_px",
        "largestClusterFrac_phase",
        "largestClusterFrac_complement",
        "orientation_deg",
    ]
    if not has_plot_data(df, arch_cols, min_finite=2):
        return
    st.subheader("Structural architecture biomarkers")
    st.caption(
        "Beyond bulk aggregometry: anisotropy and orientational order come from "
        "the grayscale/DoG image field; interface density and connectivity come "
        "from the binary mask. These raw metrics are not folded into an index."
    )

    # (b) percolation phase selector: foreground structure vs complement (plasma/background)
    perc_phase = st.radio(
        "Percolation / largest-cluster phase",
        ["Foreground (structure)", "Complement (background/plasma)"],
        index=0,
        horizontal=True,
        help="Which phase the spanning / largest-cluster connectivity is measured on.",
    )
    if perc_phase.startswith("Complement"):
        cluster_col, perc_flag_col = "largestClusterFrac_complement", "percolates_complement"
    else:
        cluster_col, perc_flag_col = "largestClusterFrac_phase", "percolates"

    if has_plot_data(df, ["anisotropy", "orientOrderParam", cluster_col], min_finite=2):
        st.pyplot(
            plot_metric_lines(
                df,
                ["anisotropy", "orientOrderParam", cluster_col],
                x_axis=plot_x_axis,
                title="Image-field directionality and mask connectivity (0-1 scale)",
            ),
            clear_figure=True,
        )
    if has_plot_data(df, ["interfaceDensity_px"], min_finite=2):
        st.pyplot(
            plot_metric_lines(
                df,
                ["interfaceDensity_px"],
                x_axis=plot_x_axis,
                title="Binary-mask interface (boundary) density (px^-1)",
            ),
            clear_figure=True,
        )
    if has_plot_data(df, ["orientation_deg"], min_finite=2):
        st.pyplot(
            plot_metric_lines(
                df,
                ["orientation_deg"],
                x_axis=plot_x_axis,
                title="Dominant structural orientation (deg)",
            ),
            clear_figure=True,
        )
    if perc_flag_col in df.columns:
        flag_series = df[perc_flag_col]
        try:
            spanning_frac = float(pd.Series(flag_series).astype(bool).mean())
            st.caption(
                f"Spanning fraction ({perc_phase.lower()}): "
                f"{spanning_frac:.2f} of analysed frames percolate."
            )
        except Exception:
            pass


def render_plots(df: pd.DataFrame, plot_x_axis: str) -> None:
    """Default to the consistent QC view; preserve legacy/non-RBC diagnostics."""
    if df.empty or "gapAreaFrac" not in df.columns or "GT_areaFrac" in df.columns:
        render_legacy_plots(df, plot_x_axis)
        return
    render_qc_plots(df, plot_x_axis)
    render_start_frame_diagnostics_section(
        df, str(first_nonempty_value(df, "selectedRBCMethod") or "RBC Gap Mode"), show_unavailable_message=True,
    )
    with st.expander("Additional analyses: average profiles and correlations", expanded=False):
        st.caption("These analyses have their own clearly labelled frame filters. The grey-point overlay above "
                   "is display-only and never changes these calculations. Frame-wise correlations are exploratory, "
                   "not independent validation.")
        mode = str(first_nonempty_value(df, "selectedRBCMethod") or "RBC Gap Mode")
        if st.checkbox("Load average profiles and correlations", value=False, key="load_additional_statistics"):
            render_average_metric_profile_section(df, "rbc", mode, "single_rbc", default_frame_filter="Included summary frames")
            render_framewise_correlation_section(
                df, mode, str(first_nonempty_value(df, "sampleName") or "image_set"),
                "single_rbc", default_qc_mode="Included summary frames",
            )
    with st.expander("Advanced: raw mask and normalized-curve diagnostics", expanded=False):
        st.caption("Separate troubleshooting views. These can contain excluded values and use their own stored "
                   "normalizations; they are not the accepted-results plots above.")
        if st.checkbox("Show raw diagnostic charts (may include excluded frames)", value=False, key="qc_raw_diagnostics"):
            st.warning("DIAGNOSTIC ONLY — raw curves can include QC-excluded frames. Do not use these curves "
                       "as trusted results. Raw and QC-normalized curves need not share the same reference scale.")
            for fig in [plot_rbc_mask_metrics_matlab_v5(df, plot_x_axis), plot_normalized_syllectograms(df, plot_x_axis)]:
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)


def render_legacy_plots(df: pd.DataFrame, plot_x_axis: str) -> None:
    """Render mode-specific plots."""
    if df.empty:
        st.info("No metrics available yet.")
        return
    render_structural_architecture_section(df, plot_x_axis)
    if "GT_areaFrac" in df.columns:
        gt_metrics = ["GT_nObjects", "GT_meanArea", "GT_meanDiam", "GT_Xi_px", "GT_lacunarity16", "GT_skelDensity"]
        render_scale_grouped_metric_plots(df, gt_metrics, plot_x_axis, "Ground-truth morphology trends")
        if {"Dice_auto_vs_GT", "IoU_auto_vs_GT"}.issubset(df.columns):
            st.pyplot(plot_segmentation_quality(df, x_axis=plot_x_axis), clear_figure=True)
        render_average_metric_profile_section(df, "gt", "Ground-Truth Mask Mode", "single_gt")
        st.info("Sample-mean correlation requires at least 3 independent samples. For this single image set, use frame-wise plots and the average metric profile.")
    elif "gapAreaFrac" in df.columns:
        plot_frames = st.selectbox(
            "Plot frames",
            ["All analysed frames", "Only frames used for main summary", "Both, with excluded frames greyed out"],
            index=2,
        )
        if plot_frames == "Only frames used for main summary":
            st.info("Plots show only frames used for the main summary.")
        else:
            st.info("Plots show all analysed frames. Main summary statistics use only QC-included frames.")
        render_average_metric_profile_section(df, "rbc", str(first_nonempty_value(df, "selectedRBCMethod") or "RBC Gap Mode"), "single_rbc")
        st.info("Sample-mean correlation requires at least 3 independent samples. For this single image set, use the frame-wise correlation matrix and average metric profile.")
        render_start_frame_diagnostics_section(
            df,
            str(first_nonempty_value(df, "selectedRBCMethod") or "RBC Gap Mode"),
            show_unavailable_message=True,
        )
        mask_plot_cols = [
            "mask_gapAreaFrac",
            "mask_gapAreaFrac_noPOD",
            "mask_gapAreaFrac_PODremoved",
            "mask_gapNtotal",
            "mask_gapNtotal_noPOD",
            "mask_gapNtotal_PODremoved",
            "mask_gapMeanDiam_px",
            "mask_gapMeanDiam_px_noPOD",
            "mask_gapMeanDiam_px_PODremoved",
            "mask_largestGapFrac",
            "mask_largestGapFrac_noPOD",
            "mask_largestGapFrac_PODremoved",
            "Xi_mask",
            "Xi_mask_noPOD",
            "Xi_mask_PODremoved",
            "podAreaDelta",
            "MaskSASI_like_final",
            "MaskSASI_like_noPOD",
            "MaskSASI_like_PODremoved",
            "MaskSASI_raw_0_100",
            "MaskGapScore_like_noPOD",
            "MaskGapScore_like_PODremoved",
            "MaskGapScore_raw_0_100",
        ]
        if has_plot_data(df, mask_plot_cols, min_finite=2):
            st.subheader("Mask-based RBC metrics over time")
            raw_count = int(
                pd.DataFrame(
                    {
                        col: pd.to_numeric(df[col], errors="coerce")
                        for col in ["MaskSASI_raw_0_100", "MaskGapScore_raw_0_100", "MaskAggIndex_raw_0_100"]
                        if col in df.columns
                    }
                ).replace([np.inf, -np.inf], np.nan).notna().any(axis=1).sum()
            ) if any(col in df.columns for col in ["MaskSASI_raw_0_100", "MaskGapScore_raw_0_100", "MaskAggIndex_raw_0_100"]) else 0
            qc_count = int(
                pd.DataFrame(
                    {
                        col: pd.to_numeric(df[col], errors="coerce")
                        for col in ["MaskSASI_QC_0_100", "MaskGapScore_QC_0_100", "MaskAggIndex_QC_0_100"]
                        if col in df.columns
                    }
                ).replace([np.inf, -np.inf], np.nan).notna().any(axis=1).sum()
            ) if any(col in df.columns for col in ["MaskSASI_QC_0_100", "MaskGapScore_QC_0_100", "MaskAggIndex_QC_0_100"]) else 0
            trusted_count = int(df.get("score_isTrusted", pd.Series(False, index=df.index)).fillna(False).astype(bool).sum()) if "score_isTrusted" in df.columns else qc_count
            if "usedInSummary" in df.columns:
                included_count = int(df["usedInSummary"].fillna(False).astype(bool).sum())
            elif "usedForMainSummary" in df.columns:
                included_count = int(df["usedForMainSummary"].fillna(False).astype(bool).sum())
            else:
                included_count = trusted_count
            visual_grey_count = (
                int(df.get("visualGreyExcluded", df.get("visualQCExcluded", pd.Series(False, index=df.index))).fillna(False).astype(bool).sum())
                if "visualGreyExcluded" in df.columns or "visualQCExcluded" in df.columns
                else max(0, len(df) - trusted_count)
            )
            warning_only_count = (
                int(df.get("visualWarningOnly", pd.Series(False, index=df.index)).fillna(False).astype(bool).sum())
                if "visualWarningOnly" in df.columns
                else 0
            )
            st.caption(
                f"selectedRBCMethod: {first_nonempty_value(df, 'selectedRBCMethod') or 'Standard RBC Gap Mode'} | "
                f"finalMaskSource: {first_nonempty_value(df, 'finalMaskSource') or 'final selected mask'} | "
                f"finite raw score frames: {raw_count} | finite QC score frames: {qc_count} | "
                f"trusted frames: {trusted_count} | usedInSummary frames: {included_count} | "
                f"grey-excluded frames: {visual_grey_count} | warning-only trusted frames: {warning_only_count}"
            )
            st.caption(
                f"Official V3 mask diagnostics ({RBC_SCORE_FORMULA_VERSION}) are calculated from binary gap masks only. "
                "LegacyIntensitySASI and LegacyIntensityGapScore remain exported for MATLAB/history comparisons."
            )
            st.pyplot(plot_rbc_mask_metrics_matlab_v5(df, plot_x_axis), clear_figure=True)
        elif any(col in df.columns for col in mask_plot_cols):
            st.caption("Skipped RBC mask-based metrics over time: no finite mask/POD diagnostic values are available.")
        rbc_syllectogram_cols = [
            "MeanI_norm_minRise",
            "MeanI_norm_0_1",
            "CV_norm_0_1",
            "opticalMean_raw",
            "opticalMean_corrected",
            "opticalMean_raw_normMinRise",
            "opticalMean_corrected_normMinRise",
            "opticalCV_raw",
            "opticalCV_corrected",
            "opticalCV_raw_norm01",
            "opticalCV_corrected_norm01",
            "MaskSASI_syl_0_1",
            "MaskSASI_syl_raw_0_1",
            "MaskSASI_syl_QC_0_1",
            "MaskGapScore_syl_0_1",
            "MaskGapScore_syl_raw_0_1",
            "MaskGapScore_syl_QC_0_1",
            "MaskAggIndex_syl_0_1",
            "MaskAggIndex_syl_raw_0_1",
            "MaskAggIndex_syl_QC_0_1",
        ]
        if has_plot_data(df, rbc_syllectogram_cols, min_finite=2):
            st.subheader("Syllectogram trends")
            st.caption(
                "Optical mean-intensity comes from raw/cropped grayscale and the global gain-corrected grayscale mean, not the spatial flat-field/no-POD mean. "
                "Optical CV comes from grayscale heterogeneity. Mask-SASI/Mask-GapScore/Mask-AggIndex come from the final selected binary gap mask."
            )
            st.pyplot(plot_normalized_syllectograms(df, plot_x_axis), clear_figure=True)
        elif any(col in df.columns for col in rbc_syllectogram_cols):
            st.caption("Skipped RBC syllectogram trends: no finite syllectogram values are available.")
        score_metrics = ["SASI_raw_0_100", "AggIndex_raw_0_100", "PTI_raw_0_100", "GapScore_raw_0_100"]
        score_metrics = [metric if metric in df.columns else metric.replace("_raw_0_100", "_0_100") for metric in score_metrics]
        qc_score_metrics = ["SASI_QC_0_100", "AggIndex_QC_0_100", "PTI_QC_0_100", "GapScore_QC_0_100"]
        structure_metrics = ["Xi_interp_px", "Xi_px", "gapMeanDiam_px", "gapNtotal", "S_struct_0_100"]
        heterogeneity_metrics = ["gapAreaFrac", "gapAreaSlope", "LF_log10_LFoverHF", "SHI_tileCV", "lacunarity16"]
        score_x_axis = "analysisIndex_1based" if "analysisIndex_1based" in df.columns else plot_x_axis
        st.subheader("RBC score trends")
        st.caption(
            "Official mask-derived raw values are shown for transparency and may include low-reliability frames. "
            "Use the QC series for interpretation; normalization is within-sequence unless a locked reference was supplied."
        )
        render_scale_grouped_metric_plots(
            df,
            score_metrics,
            score_x_axis,
            "Raw values",
            plot_frames=None,
            show_subheader=False,
            skip_message="Skipped raw RBC score trends: no finite raw score values are available.",
        )
        available_qc_scores = [metric for metric in qc_score_metrics if metric in df.columns]
        if available_qc_scores:
            qc_finite_count = int(
                pd.DataFrame({metric: pd.to_numeric(df[metric], errors="coerce") for metric in available_qc_scores})
                .replace([np.inf, -np.inf], np.nan)
                .notna()
                .any(axis=1)
                .sum()
            )
            st.caption(
                "QC-aware score trends use only trusted mask-based Xi and mask/QC-supported frames. "
                "Frames without reliable support appear as gaps/NaN."
            )
            if qc_finite_count < 3:
                st.warning(
                    "Fewer than 3 QC-trusted values are available. Raw values are shown for transparency, "
                    "but trusted trend metrics are not reported."
                )
            render_scale_grouped_metric_plots(
                df,
                available_qc_scores,
                score_x_axis,
                "QC-aware values",
                plot_frames=None,
                show_subheader=False,
                skip_message="Skipped QC-aware RBC score trends: no finite QC values are available under the selected exclusion settings.",
            )
        st.caption("Grey open points are excluded from trusted score analysis; yellow triangles are warning-only frames that remain trusted and coloured.")
        render_scale_grouped_metric_plots(
            df,
            score_metrics,
            score_x_axis,
            "Raw values with QC markers",
            plot_frames=plot_frames,
            show_subheader=False,
            skip_message="Skipped combined RBC score trends: no finite raw score values are available.",
        )
        render_scale_grouped_metric_plots(
            df,
            structure_metrics,
            plot_x_axis,
            "RBC morphology trends",
            plot_frames=plot_frames,
            skip_message="Skipped RBC morphology trends: no finite morphology values are available.",
        )
        render_scale_grouped_metric_plots(
            df,
            heterogeneity_metrics,
            plot_x_axis,
            "RBC heterogeneity trends",
            plot_frames=plot_frames,
            skip_message="Skipped RBC heterogeneity trends: no finite heterogeneity values are available.",
        )
        render_framewise_correlation_section(
            df,
            str(first_nonempty_value(df, "selectedRBCMethod") or "RBC Gap Mode"),
            str(first_nonempty_value(df, "sampleName") or first_nonempty_value(df, "filename") or "image_set"),
            "single_rbc",
        )
    else:
        if "objectCount" in df.columns:
            # TODO: add Object-mode equivalents for excludePreStartFramesFromTrustedAnalysis,
            # excludeBadPostStartQCFramesFromTrustedAnalysis, and combined raw/QC grey-warning plots.
            if has_start_frame_diagnostic_signals(df) or any(
                column in df.columns
                for column in ["autoStartMethod", "startFrameMode", "autoStartTime_s", "autoStartFrame_POD", "finalStartFrameUsed"]
            ):
                render_start_frame_diagnostics_section(
                    df,
                    str(first_nonempty_value(df, "selectedObjectMethod") or "Object Mode"),
                    show_unavailable_message=True,
                )
            object_mask_plot_cols = [
                "conservativeNoPODObjectAreaFrac",
                "podRemovedObjectAreaFrac",
                "conservativeNoPODObjectNtotal",
                "podRemovedObjectNtotal",
                "conservativeNoPODObjectMeanDiam_px",
                "podRemovedObjectMeanDiam_px",
                "conservativeNoPODObjectLargestFrac",
                "podRemovedObjectLargestFrac",
                "Object_Xi_mask",
                "Object_Xi_mask_noPOD",
                "Object_Xi_mask_PODremoved",
                "podObjectAreaDelta",
                "ObjectStructureScore_QC",
                "ObjectStructureScore_raw_0_100",
                "ObjectStructureScore_noPOD",
                "ObjectStructureScore_PODremoved",
                "ObjectAggregationIndex_QC",
                "ObjectAggregationIndex_raw_0_100",
                "ObjectAggregationIndex_noPOD",
                "ObjectAggregationIndex_PODremoved",
            ]
            if has_plot_data(df, object_mask_plot_cols, min_finite=2):
                st.subheader("Mask-based Object metrics over time")
                raw_count = int(
                    pd.DataFrame(
                        {
                            col: pd.to_numeric(df[col], errors="coerce")
                            for col in ["ObjectStructureScore_raw_0_100", "ObjectAggregationIndex_raw_0_100"]
                            if col in df.columns
                        }
                    ).replace([np.inf, -np.inf], np.nan).notna().any(axis=1).sum()
                ) if any(col in df.columns for col in ["ObjectStructureScore_raw_0_100", "ObjectAggregationIndex_raw_0_100"]) else 0
                qc_count = int(
                    pd.DataFrame(
                        {
                            col: pd.to_numeric(df[col], errors="coerce")
                            for col in ["ObjectStructureScore_QC_0_100", "ObjectAggregationIndex_QC_0_100"]
                            if col in df.columns
                        }
                    ).replace([np.inf, -np.inf], np.nan).notna().any(axis=1).sum()
                ) if any(col in df.columns for col in ["ObjectStructureScore_QC_0_100", "ObjectAggregationIndex_QC_0_100"]) else 0
                trusted_count = int(df.get("ObjectStructure_isTrusted", pd.Series(False, index=df.index)).fillna(False).astype(bool).sum()) if "ObjectStructure_isTrusted" in df.columns else qc_count
                st.caption(
                    f"selectedObjectMethod: {first_nonempty_value(df, 'selectedObjectMethod') or 'Standard Object Mode'} | "
                    f"finalObjectMaskSource: {first_nonempty_value(df, 'finalObjectMaskSource') or 'final selected object mask'} | "
                    f"finite raw score frames: {raw_count} | finite QC score frames: {qc_count} | "
                    f"trusted frames: {trusted_count} | QC-excluded frames: {max(0, len(df) - trusted_count)}"
                )
                st.caption(
                    "These diagnostics are calculated from generic binary object masks only. "
                    "They are not RBC plasma-gap or RBC aggregation metrics."
                )
                st.pyplot(plot_object_mask_metrics_pod(df, plot_x_axis), clear_figure=True)
            elif any(col in df.columns for col in object_mask_plot_cols):
                st.caption("Skipped Object mask-based metrics over time: no finite conservative/POD object values are available.")
            object_syllectogram_cols = [
                "MeanI_norm_minRise",
                "MeanI_norm_0_1",
                "CV_norm_0_1",
                "opticalMean_raw",
                "opticalMean_corrected",
                "opticalMean_raw_normMinRise",
                "opticalMean_corrected_normMinRise",
                "opticalCV_raw",
                "opticalCV_corrected",
                "opticalCV_raw_norm01",
                "opticalCV_corrected_norm01",
                "ObjectStructureSyl_0_1",
                "ObjectStructureSyl_raw_0_1",
                "ObjectStructureSyl_QC_0_1",
                "ObjectAggregationSyl_0_1",
                "ObjectAggregationSyl_raw_0_1",
                "ObjectAggregationSyl_QC_0_1",
                "gSASI_syl_norm01",
            ]
            if has_plot_data(df, object_syllectogram_cols, min_finite=2):
                st.subheader("Syllectogram trends")
                st.caption(
                    "Optical mean-intensity comes from raw/cropped grayscale and the global gain-corrected grayscale mean, not spatial flat-field mean drift. "
                    "Optical CV comes from grayscale heterogeneity. Object structure/aggregation syllectograms come from the final selected generic object mask and are not RBC biomarkers."
                )
                st.pyplot(plot_normalized_syllectograms(df, plot_x_axis), clear_figure=True)
            elif any(col in df.columns for col in object_syllectogram_cols):
                st.caption("Skipped Object syllectogram trends: no finite object syllectogram values are available.")
            object_score_metrics = [
                metric
                for metric in [
                    "gSASI_0_100",
                    "gAggIndex_0_100",
                    "ObjectStructureScore_raw_0_100",
                    "ObjectStructureScore_QC_0_100",
                    "ObjectAggregationIndex_raw_0_100",
                    "ObjectAggregationIndex_QC_0_100",
                    "ObjectStructureScore_QC",
                    "ObjectAggregationIndex_QC",
                ]
                if metric in df.columns
            ]
            if object_score_metrics:
                st.warning("Object Mode composite scores are exploratory generic image-structure indices. They are not RBC aggregation scores.")
                render_scale_grouped_metric_plots(
                    df,
                    object_score_metrics,
                    plot_x_axis,
                    "Object-mode score trends",
                    skip_message="Skipped Object-mode score trends: no finite object score values are available.",
                )
            object_morphology_metrics = [
                "areaFrac",
                "objectCount",
                "meanDiam_px",
                "medianDiam_px",
                "largestObjectFrac",
                "correlationLength_px",
                "Xi_px",
            ]
            object_heterogeneity_metrics = [
                "lacunarity16",
                "textureHeterogeneity_tileCV",
                "sizeDistributionSlope",
                "sizeDistributionSlope_R2",
                "objectTileCV",
            ]
            render_scale_grouped_metric_plots(
                df,
                object_morphology_metrics,
                plot_x_axis,
                "Object morphology trends",
                skip_message="Skipped Object morphology trends: no finite object morphology values are available.",
            )
            render_scale_grouped_metric_plots(
                df,
                object_heterogeneity_metrics,
                plot_x_axis,
                "Object heterogeneity trends",
                skip_message="Skipped Object heterogeneity trends: no finite object heterogeneity values are available.",
            )
            exploratory_scores = [
                metric
                for metric in ["gSASI_0_100", "gAggIndex_0_100", "ObjectStructureScore_QC", "ObjectAggregationIndex_QC"]
                if metric in df.columns
                and metric not in object_score_metrics
            ]
            if exploratory_scores:
                st.warning("Object Mode composite scores are exploratory generic image-structure indices. They are not RBC aggregation scores.")
                render_scale_grouped_metric_plots(
                    df,
                    exploratory_scores,
                    plot_x_axis,
                    "Object-mode score trends",
                    skip_message="Skipped Object-mode score trends: no finite exploratory score values are available.",
                )
            render_average_metric_profile_section(df, "object", str(first_nonempty_value(df, "selectedObjectMethod") or "Object Mode"), "single_object")
            st.info("Sample-mean correlation requires at least 3 independent samples. For this single image set, use the frame-wise correlation matrix and average metric profile.")
            render_object_correlation_matrix_section(df)
        else:
            if has_start_frame_diagnostic_signals(df) or any(
                column in df.columns
                for column in ["autoStartMethod", "startFrameMode", "autoStartTime_s", "autoStartFrame_POD", "finalStartFrameUsed"]
            ):
                render_start_frame_diagnostics_section(df, "Object Mode", show_unavailable_message=True)
            render_scale_grouped_metric_plots(
                df,
                ["areaFrac", "nObjects", "meanDiam", "Xi_px"],
                plot_x_axis,
                "Object morphology trends",
                skip_message="Skipped Object morphology trends: no finite object morphology values are available.",
            )
            render_scale_grouped_metric_plots(
                df,
                ["lacunarity16"],
                plot_x_axis,
                "Object heterogeneity trends",
                skip_message="Skipped Object heterogeneity trends: no finite object heterogeneity values are available.",
            )
            render_average_metric_profile_section(df, "object", str(first_nonempty_value(df, "selectedObjectMethod") or "Object Mode"), "single_object_fallback")
            st.info("Sample-mean correlation requires at least 3 independent samples. For this single image set, use the frame-wise correlation matrix and average metric profile.")
            render_object_correlation_matrix_section(df)


def render_scale_grouped_metric_plots(
    df: pd.DataFrame,
    metrics: list[str],
    plot_x_axis: str,
    section_title: str,
    plot_frames: str | None = None,
    show_subheader: bool = True,
    skip_message: str | None = None,
) -> None:
    """Render separate trend figures for metrics whose typical magnitudes differ."""
    available_metrics = has_plot_data(df, metrics, min_finite=2)
    groups = group_metrics_by_order_of_magnitude(df, available_metrics)
    if not groups:
        if skip_message:
            st.caption(skip_message)
        return

    if show_subheader:
        st.subheader(section_title)
    else:
        st.caption(section_title)
    if len(groups) > 1:
        st.caption("Metrics are split into separate charts by typical order of magnitude so smaller trends remain visible.")
    for scale_label, grouped_metrics in groups:
        chart_title = f"{section_title} | {scale_label}"
        st.caption(f"{scale_label}: {', '.join(grouped_metrics)}")
        if plot_frames is None:
            figure = plot_metric_lines(df, grouped_metrics, x_axis=plot_x_axis, title=chart_title)
        else:
            figure = plot_rbc_metric_lines_qc(
                df,
                grouped_metrics,
                x_axis=plot_x_axis,
                plot_frames=plot_frames,
                title=chart_title,
            )
        st.pyplot(figure, clear_figure=True)


def render_pca(df: pd.DataFrame, plot_x_axis: str) -> PCAResult | None:
    """Render PCA scatter, PC1 trace, and PC1 loadings."""
    if df.empty:
        st.info("No metrics available yet.")
        return None
    if len(df) < 5:
        st.info("PCA needs at least 5 images.")
        return None
    if "gapAreaFrac" in df.columns and "scoresConfidence" in df.columns:
        confidence_values = df["scoresConfidence"].astype(str).str.lower()
        if (confidence_values == "low").any():
            st.warning("PCA is exploratory because the sample did not pass reliable aggregation QC.")

    pca_result = build_pca_result(df)
    if pca_result.warning:
        st.warning(pca_result.warning)
        return pca_result

    pc1 = pca_result.explained_variance.get("PC1", np.nan)
    pc2 = pca_result.explained_variance.get("PC2", np.nan)
    columns = st.columns(2)
    columns[0].metric("PC1 explained variance", f"{pc1 * 100:.1f}%" if np.isfinite(pc1) else "n/a")
    columns[1].metric("PC2 explained variance", f"{pc2 * 100:.1f}%" if np.isfinite(pc2) else "n/a")
    pca_key = (frame_key(pca_result.scores), frame_key(pca_result.loadings), tuple(pca_result.explained_variance))
    for name, factory in [
        ("scatter", lambda: plot_pca_scatter(pca_result.scores, pca_result.explained_variance)),
        ("trace", lambda: plot_pc1_over_index(pca_result.scores, x_axis=plot_x_axis)),
        ("loadings", lambda: plot_pc1_loadings(pca_result.loadings)),
    ]:
        st.image(figure_png(("pca", name, pca_key, plot_x_axis), factory), width="stretch")
    return pca_result


def has_exportable_rows(df: pd.DataFrame | None) -> bool:
    """Return True when a dataframe has rows and columns to export."""
    return isinstance(df, pd.DataFrame) and not df.empty and len(df.columns) > 0


EXPORT_EMPTY_WARNING = "No rows available to export. Run the analysis/comparison first."


def csv_bytes_for_excel(df: pd.DataFrame) -> bytes:
    """Return CSV bytes with UTF-8 BOM so Excel opens text columns correctly."""
    return df.to_csv(index=False).encode("utf-8-sig")


def store_export_dataframe(state_key: str, df: pd.DataFrame | None) -> pd.DataFrame:
    """Persist one raw final dataframe as the authoritative CSV export object."""
    stored_df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    st.session_state[state_key] = stored_df
    return stored_df


def csv_export_diagnostics(state_key: str) -> tuple[pd.DataFrame, bytes, dict[str, Any]]:
    """Inspect and encode one authoritative session-state CSV dataframe."""
    exists = state_key in st.session_state
    value = st.session_state.get(state_key)
    is_dataframe = isinstance(value, pd.DataFrame)
    df = value if is_dataframe else pd.DataFrame()
    rows, columns = df.shape
    payload = csv_bytes_for_excel(df) if rows > 0 and columns > 0 else b""
    diagnostics = {
        "state key": state_key,
        "exists": exists,
        "type": type(value).__name__ if exists else "missing",
        "shape": f"{rows} rows x {columns} columns",
        "rows": rows,
        "columns": columns,
        "CSV byte length": len(payload),
    }
    return df, payload, diagnostics


def render_csv_export_diagnostics(diagnostics: dict[str, Any], container: Any = None) -> None:
    """Show compact diagnostics beside a CSV download button."""
    outlet = container or st
    outlet.caption(
        "Export diagnostics: "
        f"exists={diagnostics['exists']} | "
        f"type={diagnostics['type']} | "
        f"shape={diagnostics['shape']} | "
        f"rows={diagnostics['rows']} | "
        f"columns={diagnostics['columns']} | "
        f"CSV byte length={diagnostics['CSV byte length']}"
    )


def render_csv_download(
    label: str,
    state_key: str,
    file_name: str,
    key: str,
    empty_warning: str = EXPORT_EMPTY_WARNING,
    container: Any = None,
) -> bool:
    """Render a state-backed CSV download button only for populated raw dataframes."""
    outlet = container or st
    df, payload, diagnostics = csv_export_diagnostics(state_key)
    render_csv_export_diagnostics(diagnostics, container=outlet)
    if not diagnostics["exists"] or not isinstance(st.session_state.get(state_key), pd.DataFrame):
        outlet.warning(empty_warning)
        outlet.download_button(label, data=b"", file_name=file_name, mime="text/csv", key=key, disabled=True)
        return False
    if len(df) == 0 or len(df.columns) == 0 or len(payload) == 0:
        outlet.warning(empty_warning)
        outlet.download_button(label, data=b"", file_name=file_name, mime="text/csv", key=key, disabled=True)
        return False

    outlet.download_button(
        label,
        data=payload,
        file_name=file_name,
        mime="text/csv",
        key=key,
        on_click="ignore",
    )
    outlet.caption(f"Rows available for export: {len(df)} | Columns available for export: {len(df.columns)} | Download size: {len(payload)} bytes")
    return True


def render_text_download(
    label: str,
    text: str | None,
    file_name: str,
    mime: str,
    key: str,
    empty_warning: str = EXPORT_EMPTY_WARNING,
    container: Any = None,
) -> bool:
    """Render a state-backed text download button with transparent diagnostics."""
    outlet = container or st
    state_key = f"{key}_text"
    st.session_state[state_key] = text if isinstance(text, str) else ""
    stored_text = st.session_state[state_key]
    payload = stored_text.encode("utf-8-sig") if stored_text.strip() else b""
    lines = stored_text.splitlines()
    outlet.caption(
        "Export diagnostics: "
        f"exists={state_key in st.session_state} | "
        f"type={type(stored_text).__name__} | "
        f"shape={len(lines)} rows x 1 columns | "
        f"rows={len(lines)} | columns=1 | byte length={len(payload)}"
    )
    if not stored_text.strip() or len(payload) == 0:
        outlet.warning(empty_warning)
        outlet.download_button(label, data=b"", file_name=file_name, mime=mime, key=key, disabled=True)
        return False

    outlet.download_button(
        label,
        data=payload,
        file_name=file_name,
        mime=mime,
        key=key,
        on_click="ignore",
    )
    outlet.caption(f"Rows available for export: {len(lines)} | Columns available for export: 1 | Download size: {len(payload)} bytes")
    return True


def export_diagnostic_row(name: str, export_object: Any) -> dict[str, Any]:
    """Describe one dataframe or text export object for transparent debugging."""
    if isinstance(export_object, pd.DataFrame):
        payload = csv_bytes_for_excel(export_object) if has_exportable_rows(export_object) else b""
        return {
            "dataframe name": name,
            "type": type(export_object).__name__,
            "shape": f"{len(export_object)} rows x {len(export_object.columns)} columns",
            "first 3 column names": ", ".join(map(str, export_object.columns[:3])) or "none",
            "byte length": len(payload),
            "download enabled": has_exportable_rows(export_object),
        }
    if isinstance(export_object, str):
        lines = export_object.splitlines()
        return {
            "dataframe name": name,
            "type": type(export_object).__name__,
            "shape": f"{len(lines)} lines x 1 text document",
            "first 3 column names": "markdown/text",
            "byte length": len(export_object.encode("utf-8-sig")) if export_object.strip() else 0,
            "download enabled": bool(export_object.strip()),
        }
    return {
        "dataframe name": name,
        "type": type(export_object).__name__,
        "shape": "0 rows x 0 columns",
        "first 3 column names": "none",
        "byte length": 0,
        "download enabled": False,
    }


def render_export_diagnostics(export_objects: dict[str, Any]) -> None:
    """Show export shapes and enabled states without mutating export data."""
    with st.expander("Export diagnostics", expanded=False):
        diagnostics_df = pd.DataFrame([export_diagnostic_row(name, value) for name, value in export_objects.items()])
        st.dataframe(diagnostics_df, use_container_width=True, hide_index=True)


def count_batch_report_processing_notes(report_text: str) -> int:
    """Count complete processing-note rows in a rendered batch report."""
    in_processing_notes = False
    count = 0
    for line in report_text.splitlines():
        if line.strip() == "## Processing Notes":
            in_processing_notes = True
            continue
        if in_processing_notes and line.startswith("## "):
            break
        if in_processing_notes and line.startswith("- ") and not line.startswith("- ... ("):
            count += 1
    return count


def batch_report_strings_for_storage(result: dict[str, Any]) -> tuple[str, str, int, int, int]:
    """Return canonical full export text and a separate optional UI preview."""
    notes_value = result.get("processing_notes_full", result.get("warnings"))
    notes = list(notes_value) if isinstance(notes_value, list) else None
    if notes is not None:
        summary_df = result.get("sample_summary", pd.DataFrame())
        frame_metrics_df = result.get("frame_metrics", pd.DataFrame())
        matlab_summary_df = result.get("matlab_summary", pd.DataFrame())
        input_mode = str(result.get("batch_input_mode", "Images + optional MATLAB CSV comparison"))
        validation_mode = str(result.get("batch_validation_mode", "RBC Gap Mode"))
        full_report = batch_report_markdown(
            summary_df,
            frame_metrics_df,
            matlab_summary_df,
            notes,
            batch_input_mode=input_mode,
            batch_validation_mode=validation_mode,
        )
        preview_report = batch_report_markdown(
            summary_df,
            frame_metrics_df,
            matlab_summary_df,
            notes,
            batch_input_mode=input_mode,
            batch_validation_mode=validation_mode,
            processing_note_limit=BATCH_REPORT_UI_NOTE_LIMIT,
        )
        return full_report, preview_report, len(notes), min(len(notes), BATCH_REPORT_UI_NOTE_LIMIT), count_batch_report_processing_notes(full_report)

    full_report = result.get("report_md", "")
    full_report = full_report if isinstance(full_report, str) else ""
    preview_report = result.get("report_preview_md", full_report)
    preview_report = preview_report if isinstance(preview_report, str) else full_report
    total_count = count_batch_report_processing_notes(full_report)
    preview_count = count_batch_report_processing_notes(preview_report)
    return full_report, preview_report, total_count, preview_count, total_count


def store_batch_outputs_in_session_state(result: dict[str, Any]) -> None:
    """Persist final combined batch outputs as the authoritative export objects."""
    store_export_dataframe("batch_sample_summary_df", result.get("sample_summary", pd.DataFrame()))
    store_export_dataframe("batch_frame_metrics_df", result.get("frame_metrics", pd.DataFrame()))
    store_export_dataframe("batch_matlab_comparison_df", result.get("matlab_summary", pd.DataFrame()))
    store_export_dataframe("batch_paired_comparison_df", result.get("paired_comparison", pd.DataFrame()))
    st.session_state["batch_validation_mode"] = str(result.get("batch_validation_mode", "RBC Gap Mode"))
    report_text, report_preview, total_notes, preview_notes, exported_notes = batch_report_strings_for_storage(result)
    if isinstance(report_text, str) and report_text.strip():
        st.session_state["batch_report_md"] = report_text
        st.session_state["batch_validation_report_md"] = report_text
        st.session_state["batch_validation_report_txt"] = report_text
        st.session_state["batch_report_preview_md"] = report_preview
        st.session_state["batch_processing_notes_total_count"] = total_notes
        st.session_state["batch_processing_notes_preview_count"] = preview_notes
        st.session_state["batch_processing_notes_export_count"] = exported_notes


def render_batch_dataframe_download(
    state_key: str,
    label: str,
    file_name: str,
    key: str,
) -> bool:
    """Preview and download one authoritative combined batch dataframe from session state."""
    df, csv_bytes, diagnostics = csv_export_diagnostics(state_key)
    rows, columns = df.shape

    st.markdown(f"**{state_key}**")
    render_csv_export_diagnostics(diagnostics)
    st.caption("First 5 columns: " + (", ".join(map(str, df.columns[:5])) if columns else "none"))
    if rows and columns:
        st.dataframe(df.head(3), use_container_width=True, hide_index=True)
    else:
        st.info("No preview rows are available.")
    st.caption(f"Rows available for export: {rows} | Columns available for export: {columns}")

    if not diagnostics["exists"] or diagnostics["type"] != "DataFrame" or rows == 0 or columns == 0 or len(csv_bytes) == 0:
        st.warning("No rows available to export. Run batch analysis first.")
        st.download_button(label, data=b"", file_name=file_name, mime="text/csv", key=key, disabled=True)
        return False

    st.download_button(
        label,
        data=csv_bytes,
        file_name=file_name,
        mime="text/csv",
        key=key,
        on_click="ignore",
    )
    st.caption(f"Download size: {len(csv_bytes)} bytes")
    return True


def render_batch_markdown_download(
    state_key: str,
    label: str,
    file_name: str,
    key: str,
) -> bool:
    """Preview and download the authoritative batch markdown report from session state."""
    value = st.session_state.get(state_key)
    report_text = value if isinstance(value, str) else ""
    exists = state_key in st.session_state and isinstance(value, str)
    lines = report_text.splitlines()
    md_bytes = report_text.encode("utf-8-sig") if report_text.strip() else b""

    st.markdown(f"**{state_key}**")
    st.caption(
        "Export diagnostics: "
        f"exists={state_key in st.session_state} | "
        f"type={type(value).__name__ if state_key in st.session_state else 'missing'} | "
        f"shape={len(lines)} rows x 1 columns | "
        f"rows={len(lines)} | columns=1 | byte length={len(md_bytes)}"
    )
    st.caption(f"Report text length: {len(report_text)} characters")
    st.caption(f"Rows available for export: {len(lines)} | Columns available for export: 1")
    if lines:
        st.code("\n".join(lines[:3]), language="markdown")
    else:
        st.info("No report preview lines are available.")

    if not exists or not report_text.strip() or len(md_bytes) == 0:
        st.warning("No rows available to export. Run batch analysis first.")
        st.download_button(label, data=b"", file_name=file_name, mime="text/markdown", key=key, disabled=True)
        return False

    st.download_button(
        label,
        data=md_bytes,
        file_name=file_name,
        mime="text/markdown",
        key=key,
        on_click="ignore",
    )
    st.caption(f"Download size: {len(md_bytes)} bytes")
    return True


def render_batch_text_download(
    state_key: str,
    label: str,
    file_name: str,
    key: str,
) -> bool:
    """Download the final displayed batch report as UTF-8 plain text."""
    value = st.session_state.get(state_key)
    report_txt = value if isinstance(value, str) else ""
    exists = state_key in st.session_state
    report_bytes = report_txt.encode("utf-8") if report_txt.strip() else b""
    st.markdown(f"**{state_key}**")
    st.caption(
        "Export diagnostics: "
        f"report exists={'yes' if exists else 'no'} | "
        f"type={type(value).__name__ if exists else 'missing'} | "
        f"character length={len(report_txt)} | "
        f"byte length={len(report_bytes)}"
    )
    st.caption("First 200 characters preview: " + (report_txt[:200] if report_txt else "none"))
    if not isinstance(value, str) or not report_txt.strip() or len(report_bytes) == 0:
        st.warning("No batch validation report available for export yet. Run batch validation first.")
        return False

    st.download_button(
        label=label,
        data=report_bytes,
        file_name=file_name,
        mime="text/plain",
        key=key,
        on_click="ignore",
    )
    st.caption(f"Download size: {len(report_bytes)} bytes")
    return True


def render_batch_processing_note_diagnostics() -> None:
    """Show whether the UI preview and both exported reports contain the expected notes."""
    total_notes = int(st.session_state.get("batch_processing_notes_total_count", 0) or 0)
    preview_text = st.session_state.get("batch_report_preview_md", "")
    md_text = st.session_state.get("batch_report_md", "")
    txt_text = st.session_state.get("batch_validation_report_txt", "")
    preview_notes = count_batch_report_processing_notes(preview_text) if isinstance(preview_text, str) else 0
    md_notes = count_batch_report_processing_notes(md_text) if isinstance(md_text, str) else 0
    txt_notes = count_batch_report_processing_notes(txt_text) if isinstance(txt_text, str) else 0
    st.caption(
        "Processing-note diagnostics: "
        f"total collected={total_notes} | "
        f"UI preview included={preview_notes} | "
        f"TXT export included={txt_notes} | "
        f"MD export included={md_notes}"
    )
    if txt_notes < total_notes or md_notes < total_notes:
        st.warning("Exported report processing-note count is lower than the collected note count. Run Batch Analysis again before downloading.")


def render_batch_export_diagnostics() -> None:
    """Show whether authoritative batch export objects exist and contain data."""
    state_keys = [
        "batch_sample_summary_df",
        "batch_frame_metrics_df",
        "batch_matlab_comparison_df",
        "batch_paired_comparison_df",
        "batch_average_behaviour_sample_mean_metric_profile_df",
        "batch_average_behaviour_group_average_metric_profile_df",
        "batch_average_behaviour_sample_mean_metric_table_df",
        "batch_average_behaviour_sample_mean_correlation_compact_df",
        "batch_average_behaviour_sample_mean_correlation_full_df",
        "batch_report_md",
        "batch_report_preview_md",
        "batch_validation_report_txt",
    ]
    selected_mode = str(st.session_state.get("batch_validation_mode", "not available"))
    rows: list[dict[str, Any]] = []
    for state_key in state_keys:
        exists = state_key in st.session_state
        value = st.session_state.get(state_key)
        if isinstance(value, pd.DataFrame):
            csv_bytes = csv_bytes_for_excel(value) if has_exportable_rows(value) else b""
            rows.append(
                {
                    "session_state object": state_key,
                    "exists": exists,
                    "type": type(value).__name__,
                    "shape": f"{len(value)} rows x {len(value.columns)} columns",
                    "rows": len(value),
                    "columns": len(value.columns),
                    "byte length": len(csv_bytes),
                    "first 5 columns": ", ".join(map(str, value.columns[:5])) or "none",
                    "report text length": "n/a",
                }
            )
        elif isinstance(value, str):
            rows.append(
                {
                    "session_state object": state_key,
                    "exists": exists,
                    "type": type(value).__name__,
                    "shape": "text document",
                    "rows": len(value.splitlines()),
                    "columns": 1,
                    "byte length": len(value.encode("utf-8-sig")) if value.strip() else 0,
                    "first 5 columns": "n/a",
                    "report text length": str(len(value)),
                }
            )
        else:
            rows.append(
                {
                    "session_state object": state_key,
                    "exists": exists,
                    "type": type(value).__name__ if exists else "missing",
                    "shape": "not available",
                    "rows": 0,
                    "columns": 0,
                    "byte length": 0,
                    "first 5 columns": "none",
                    "report text length": "n/a",
                }
            )
    with st.expander("Export Diagnostics", expanded=False):
        st.caption(f"Selected batch validation mode: {selected_mode}")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def paired_frame_export_df(comparison_long: pd.DataFrame | None) -> pd.DataFrame:
    """Convert long MATLAB comparison rows into one wide row per frame/index."""
    if not has_exportable_rows(comparison_long):
        return pd.DataFrame()
    required = {"metric", "frameIndex", "appValue", "matlabValue", "difference"}
    if not required.issubset(comparison_long.columns):
        return pd.DataFrame()

    group_cols = ["frameIndex"]
    if "filename" in comparison_long.columns:
        group_cols.append("filename")

    rows: list[dict[str, Any]] = []
    for keys, group in comparison_long.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        for _, metric_row in group.iterrows():
            metric = str(metric_row["metric"])
            row[f"{metric}_app"] = metric_row.get("appValue", np.nan)
            row[f"{metric}_MATLAB"] = metric_row.get("matlabValue", np.nan)
            row[f"{metric}_diff_app_minus_MATLAB"] = metric_row.get("difference", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def render_matlab_comparison(analysis: dict[str, Any] | None) -> dict[str, Any] | None:
    """Render MATLAB CSV comparison tools for RBC Gap Mode."""
    if not analysis or analysis.get("mode") != "RBC Gap Mode" or analysis["metrics"].empty:
        st.info("Run RBC Gap Mode analysis before comparing with MATLAB.")
        st.session_state["matlab_comparison_result"] = None
        return None

    st.write("Upload a MATLAB per-frame CSV to compare shared RBC Gap Mode metrics against the current MicroStructAI results.")
    qc_summary = analysis.get("params", {}).get("rbc_qc_summary", {})
    if qc_summary:
        qc_text = f"RBC QC class: {qc_summary.get('sampleQC', 'n/a')} | Reliable frames: {qc_summary.get('reliableFramePercent', np.nan):.1f}%"
        if qc_summary.get("scoresInterpretable", False):
            st.info(qc_text)
        else:
            st.warning(qc_text + ". MATLAB comparison still uses the analysed frames for reproducibility.")
    csv_file = remembered_upload("Upload MATLAB per-frame CSV", ["csv"], "matlab_comparison_csv")
    match_method = st.selectbox("Matching method", MATCH_METHODS, index=0)
    if csv_file is None:
        st.session_state["matlab_comparison_result"] = None
        st.info("Upload a MATLAB CSV such as agg_features_H006_35_NoStent_GAPS.csv.")
        return None

    try:
        matlab_df = pd.read_csv(io.BytesIO(csv_file.getvalue()))
    except Exception as exc:
        st.session_state["matlab_comparison_result"] = None
        st.error(f"Could not read MATLAB CSV: {exc}")
        return None

    app_df = visible_metrics_df(analysis["metrics"])
    exclude_qc_flagged = False
    if "rbcFrameReliable" in app_df.columns:
        exclude_qc_flagged = st.checkbox("Exclude QC-flagged app frames from MATLAB comparison", value=False)
        if exclude_qc_flagged:
            before_qc_filter = len(app_df)
            app_df = app_df.loc[app_df["rbcFrameReliable"].fillna(False).astype(bool)].reset_index(drop=True)
            st.info(f"QC frame exclusion is enabled for this comparison: {len(app_df)} of {before_qc_filter} app rows kept.")
        else:
            st.caption("QC labels are shown for context; MATLAB comparison uses all app rows unless QC exclusion is selected.")
    matlab_total_rows = int(len(matlab_df))
    is_ana_col = next((col for col in matlab_df.columns if str(col).lower() == "isana"), None)
    matlab_analysis_rows: int | None = None
    compare_only_isana = False
    matlab_for_compare = matlab_df
    st.caption(f"MATLAB CSV rows: {matlab_total_rows}")
    if is_ana_col is not None:
        raw_isana = matlab_df[is_ana_col]
        numeric_isana = pd.to_numeric(raw_isana, errors="coerce")
        if numeric_isana.notna().any():
            is_ana_mask = numeric_isana == 1
        else:
            is_ana_mask = raw_isana.astype(str).str.strip().str.lower().isin(["true", "yes", "1"])
        matlab_analysis_rows = int(is_ana_mask.sum())
        st.caption(f"MATLAB analysis rows where isAna == 1: {matlab_analysis_rows}")
        compare_only_isana = st.checkbox("Compare only MATLAB analysis rows (isAna == 1)", value=True)
        if compare_only_isana:
            matlab_for_compare = matlab_df.loc[is_ana_mask].reset_index(drop=True)
            st.info(f"Comparing against {len(matlab_for_compare)} MATLAB analysis rows after isAna filtering.")
            if len(app_df) != len(matlab_for_compare):
                st.warning(
                    f"App analysed rows ({len(app_df)}) differ from MATLAB isAna rows ({len(matlab_for_compare)}). "
                    "If row-order matching is used, comparison will use the minimum length."
                )

    result = compare_matlab_metrics(app_df, matlab_for_compare, match_method)
    result.update(
        {
            "matlabTotalRows": matlab_total_rows,
            "matlabAnalysisRows": matlab_analysis_rows if matlab_analysis_rows is not None else "n/a",
            "matlabRowsCompared": int(len(matlab_for_compare)),
            "appAnalysedRows": int(len(app_df)),
            "isAnaFilterUsed": bool(compare_only_isana),
            "qcFrameExclusionUsed": bool(exclude_qc_flagged),
        }
    )
    st.session_state["matlab_comparison_result"] = result

    st.caption(f"Match method used: {result['matchMethodUsed']} | Matched rows: {result['matchedRows']}")
    for warning in result.get("warnings", []):
        st.warning(warning)
    missing = result.get("missingMetrics", [])
    if missing:
        with st.expander("Missing comparison metrics"):
            for item in missing:
                st.write(f"- {item}")

    comparison_df = store_export_dataframe("matlab_comparison_table_df", result.get("summary", pd.DataFrame()))
    paired_export_df = store_export_dataframe(
        "matlab_paired_frame_data_df",
        paired_frame_export_df(result.get("long", pd.DataFrame())),
    )
    app_df = store_export_dataframe("matlab_full_app_metrics_df", app_df)
    if not has_exportable_rows(comparison_df):
        render_csv_download(
            "Download MATLAB comparison table CSV",
            "matlab_comparison_table_df",
            "MicroStructAI_MATLAB_comparison_table.csv",
            "download_matlab_comparison_table_csv_empty",
        )
        render_export_diagnostics(
            {
                "MATLAB_comparison_table": comparison_df,
                "MATLAB_paired_frame_data": paired_frame_export_df(result.get("long", pd.DataFrame())),
                "full_app_metrics": app_df,
            }
        )
        return result

    st.subheader("Comparison Table")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    comparison_long = result.get("long", pd.DataFrame())
    if has_exportable_rows(paired_export_df):
        with st.expander("Paired per-frame comparison data", expanded=False):
            st.dataframe(paired_export_df, use_container_width=True, hide_index=True)
        st.subheader("Matched-frame correlation structure")
        st.caption(
            "The first two matrices show Spearman correlations across the same matched frames for MicroStructAI and MATLAB. "
            "The third matrix shows the correlation difference: app minus MATLAB."
        )
        st.pyplot(plot_matlab_spearman_matrix_comparison(paired_export_df), clear_figure=True)

    st.subheader("Exports")
    render_csv_download(
        "Download MATLAB comparison table CSV",
        "matlab_comparison_table_df",
        "MicroStructAI_MATLAB_comparison_table.csv",
        "download_matlab_comparison_table_csv",
    )

    render_csv_download(
        "Download paired per-frame comparison data CSV",
        "matlab_paired_frame_data_df",
        "MicroStructAI_MATLAB_paired_frame_data.csv",
        "download_matlab_paired_frame_data_csv",
    )

    render_csv_download(
        "Download full app metrics CSV",
        "matlab_full_app_metrics_df",
        "MicroStructAI_app_metrics.csv",
        "download_full_app_metrics_csv",
    )
    render_export_diagnostics(
        {
            "MATLAB_comparison_table": comparison_df,
            "MATLAB_paired_frame_data": paired_export_df,
            "full_app_metrics": app_df,
        }
    )

    metrics_to_plot = st.multiselect(
        "Metrics to plot",
        comparison_df["metric"].astype(str).tolist(),
        default=comparison_df["metric"].astype(str).tolist(),
    )
    for metric in metrics_to_plot:
        with st.expander(metric, expanded=False):
            columns = st.columns(3)
            columns[0].pyplot(plot_matlab_scatter(comparison_long, metric), clear_figure=True)
            columns[1].pyplot(plot_matlab_difference(comparison_long, metric), clear_figure=True)
            columns[2].pyplot(plot_matlab_trend(comparison_long, metric), clear_figure=True)

    gap_previews = [item for item in analysis.get("previews", []) if "mask" in item]
    if gap_previews:
        with st.expander("gapAreaSlope MATLAB debug for one app frame", expanded=False):
            debug_names = [item["filename"] for item in gap_previews]
            debug_choice = st.selectbox("Debug frame", debug_names, key="gap_area_slope_debug_frame")
            debug_preview = gap_previews[debug_names.index(debug_choice)]
            debug = gap_area_slope_matlab_debug(debug_preview["mask"])
            debug_cols = st.columns(6)
            debug_cols[0].metric("Components", debug["nComponents"])
            debug_cols[1].metric("Min area", format_metric_value(debug["minComponentArea"], 2))
            debug_cols[2].metric("Max area", format_metric_value(debug["maxComponentArea"], 2))
            debug_cols[3].metric("Slope", format_metric_value(debug["gapAreaSlope"], 4))
            debug_cols[4].metric("R2", format_metric_value(debug["gapAreaSlope_R2"], 4))
            debug_cols[5].metric("Bins used", debug["gapAreaSlope_nBins"])

            edges = debug.get("binEdges", [])
            counts = debug.get("binCounts", [])
            if len(edges) >= 2 and counts:
                centers = np.sqrt(np.asarray(edges[:-1], dtype=float) * np.asarray(edges[1:], dtype=float))
                bin_df = pd.DataFrame(
                    {
                        "bin": np.arange(1, len(counts) + 1),
                        "leftEdge": edges[:-1],
                        "rightEdge": edges[1:],
                        "count": counts,
                        "center": centers,
                        "kept": np.asarray(counts) > 0,
                    }
                )
                st.caption("MATLAB-style log-spaced bins and histcounts output")
                st.dataframe(bin_df, use_container_width=True, hide_index=True)

            x_values = debug.get("log10BinCenters", [])
            y_values = debug.get("log10Counts", [])
            centers_used = debug.get("nonzeroBinCenters", [])
            if x_values and y_values:
                fit_df = pd.DataFrame(
                    {
                        "nonzeroBinCenter": centers_used,
                        "log10BinCenter": x_values,
                        "log10Count": y_values,
                    }
                )
                st.caption("Regression inputs: log10(count) vs log10(area-bin centre)")
                st.dataframe(fit_df, use_container_width=True, hide_index=True)
    return result


MULTI_SAMPLE_CURRENT_SOURCE = "Current batch analysis results stored in session_state"
MULTI_SAMPLE_UPLOAD_SOURCE = "Upload sample_summary.csv and frame_metrics.csv"


def read_dashboard_csv(uploaded_file: Any, label: str) -> pd.DataFrame | None:
    """Read one uploaded dashboard CSV and show a non-fatal error when invalid."""
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    except Exception as exc:
        st.error(f"Could not read {label}: {exc}")
        return None


def render_multi_sample_report_download(
    label: str,
    state_key: str,
    file_name: str,
    mime: str,
    key: str,
) -> bool:
    """Render a non-empty UTF-8 multi-sample report download with diagnostics."""
    value = st.session_state.get(state_key)
    report_text = value if isinstance(value, str) else ""
    payload = report_text.encode("utf-8") if report_text.strip() else b""
    st.caption(
        "Export diagnostics: "
        f"exists={state_key in st.session_state} | "
        f"type={type(value).__name__ if state_key in st.session_state else 'missing'} | "
        f"character length={len(report_text)} | "
        f"byte length={len(payload)}"
    )
    if not report_text.strip() or not payload:
        st.warning("No multi-sample comparison report available to export yet.")
        return False
    st.download_button(label=label, data=payload, file_name=file_name, mime=mime, key=key, on_click="ignore")
    st.caption(f"Download size: {len(payload)} bytes")
    return True


def multi_sample_summary_display_columns(summary_df: pd.DataFrame) -> list[str]:
    """Return a stable, scan-friendly comparison table column order."""
    preferred = [
        "sampleName",
        "donor",
        "flowHistory",
        "stentStatus",
        "nImagesLoaded",
        "nFramesAnalysed",
        "nFramesUsedForSummary",
        "sampleQC",
        "scoresConfidence",
        "autoStartFrame",
        "reliableFramePercent",
        "largeGapArtefactRisk",
        "percentFramesWithLargeGapRemoval",
        "maxLargestRemovedGapFrac",
    ]
    for metric in ["SASI_0_100", "AggIndex_0_100", "Xi_interp_px", "gapMeanDiam_px", "gapNtotal"]:
        preferred.extend([f"mean_{metric}", f"median_{metric}", f"final_{metric}"])
    for metric in ["gSASI_0_100", "gAggIndex_0_100", "correlationLength_px", "meanDiam_px", "objectCount"]:
        preferred.extend([f"mean_{metric}", f"median_{metric}", f"final_{metric}"])
    preferred.extend(["median_gapAreaFrac", "median_LF_log10_LFoverHF", "median_SHI_tileCV"])
    return [column for column in preferred if column in summary_df.columns]


def render_multi_sample_comparison_dashboard(key_prefix: str = "multi_sample") -> None:
    """Render the multi-sample dashboard without rerunning image analysis."""
    st.header("Multi-Sample Comparison")
    st.write("Compare existing RBC Gap Mode or Object Mode batch outputs visually and numerically without rerunning segmentation.")
    has_current_batch = isinstance(st.session_state.get("batch_sample_summary_df"), pd.DataFrame) and isinstance(
        st.session_state.get("batch_frame_metrics_df"), pd.DataFrame
    )
    source_options = [MULTI_SAMPLE_CURRENT_SOURCE, MULTI_SAMPLE_UPLOAD_SOURCE]
    source = st.radio(
        "Comparison input source",
        source_options,
        index=0 if has_current_batch else 1,
        key=f"{key_prefix}_source",
    )

    if source == MULTI_SAMPLE_CURRENT_SOURCE:
        sample_summary_df = st.session_state.get("batch_sample_summary_df", pd.DataFrame())
        frame_metrics_df = st.session_state.get("batch_frame_metrics_df", pd.DataFrame())
        matlab_comparison_df = st.session_state.get("batch_matlab_comparison_df", pd.DataFrame())
        if not isinstance(sample_summary_df, pd.DataFrame) or sample_summary_df.empty:
            st.warning("Current batch sample summary is unavailable. Run Batch Validation or upload exported CSV files.")
            return
        if not isinstance(frame_metrics_df, pd.DataFrame):
            frame_metrics_df = pd.DataFrame()
        if not isinstance(matlab_comparison_df, pd.DataFrame):
            matlab_comparison_df = pd.DataFrame()
        st.caption("Using the final current batch dataframes stored in session state. Image analysis was not rerun.")
        if isinstance(st.session_state.get("batch_validation_report_txt"), str):
            st.caption("The existing batch validation TXT report is also available in session state for reference.")
    else:
        st.caption("Upload the exported batch tables. MATLAB comparison CSV is optional.")
        sample_upload = st.file_uploader("Upload sample_summary.csv", type=["csv"], key=f"{key_prefix}_sample_summary_upload")
        frame_upload = st.file_uploader("Upload frame_metrics.csv", type=["csv"], key=f"{key_prefix}_frame_metrics_upload")
        matlab_upload = st.file_uploader("Optional: upload matlab_comparison.csv", type=["csv"], key=f"{key_prefix}_matlab_upload")
        if sample_upload is None or frame_upload is None:
            st.info("Upload both sample_summary.csv and frame_metrics.csv to build the dashboard.")
            return
        sample_summary_df = read_dashboard_csv(sample_upload, "sample_summary.csv")
        frame_metrics_df = read_dashboard_csv(frame_upload, "frame_metrics.csv")
        matlab_comparison_df = read_dashboard_csv(matlab_upload, "matlab_comparison.csv") if matlab_upload is not None else pd.DataFrame()
        if sample_summary_df is None or frame_metrics_df is None:
            return
        matlab_comparison_df = matlab_comparison_df if isinstance(matlab_comparison_df, pd.DataFrame) else pd.DataFrame()

    prepared = prepare_multi_sample_comparison(sample_summary_df, frame_metrics_df, matlab_comparison_df)
    for error in prepared.get("errors", []):
        st.error(error)
    if prepared.get("errors"):
        return

    summary_df = prepared.get("summary", pd.DataFrame())
    frames_df = prepared.get("frames", pd.DataFrame())
    group_df = prepared.get("groups", pd.DataFrame())
    matlab_df = prepared.get("matlab", pd.DataFrame())
    warnings = prepared.get("warnings", [])
    score_type = str(prepared.get("scoreType", "rbc"))
    if summary_df.empty:
        st.warning("No comparable sample rows were available.")
        return

    report_text = multi_sample_report(summary_df, frames_df, group_df, matlab_df, warnings)
    summary_df = store_export_dataframe("multi_sample_comparison_summary_df", summary_df)
    group_df = store_export_dataframe("multi_sample_group_summary_df", group_df)
    store_export_dataframe("multi_sample_matlab_comparison_df", matlab_df)
    average_behaviour_prefix = f"{key_prefix}_average_behaviour"
    store_default_average_behaviour_exports(
        summary_df,
        average_behaviour_prefix,
        "object" if score_type == "generic" else "rbc",
        "Multi-sample",
    )
    st.session_state["multi_sample_comparison_report_txt"] = report_text
    st.session_state["multi_sample_comparison_report_md"] = report_text

    for warning in warnings:
        st.warning(warning)

    overview = st.columns(4)
    overview[0].metric("Samples", len(summary_df))
    overview[1].metric("Frame rows", len(frames_df))
    overview[2].metric("Available comparison metrics", len(prepared.get("metricsAvailable", [])))
    overview[3].metric("Group summary rows", len(group_df))

    tabs = st.tabs(["Summary", "Plots", "Group Summary", "MATLAB", "Report", "Exports"])
    with tabs[0]:
        st.subheader("Sample Comparison Summary")
        display_columns = multi_sample_summary_display_columns(summary_df)
        st.dataframe(summary_df[display_columns] if display_columns else summary_df, use_container_width=True, hide_index=True)
        if score_type == "generic":
            st.info(
                "Higher gSASI may indicate larger, less fragmented coarse structural organisation. "
                "The direction of interpretation depends on sample type. gSASI/gAggIndex are exploratory generic indices, not RBC aggregation scores."
            )
            st.warning("Do not compare RBC SASI/AggIndex directly with Object Mode gSASI/gAggIndex. They are different score types.")
        else:
            st.info(
                "Higher SASI generally indicates stronger structural organisation of plasma-gap morphology. "
                "Higher Xi_interp_px suggests a larger characteristic structural length scale. "
                "Lower gapNtotal together with larger gapMeanDiam_px may indicate more consolidated plasma-gap morphology."
            )
        st.caption("These are research-use structural indices, not clinical diagnostic outputs. Compare only equivalent imaging settings and presets.")

    with tabs[1]:
        render_sample_average_behaviour_section(
            summary_df,
            key_prefix=average_behaviour_prefix,
            kind="generic" if score_type == "generic" else "rbc",
            title_prefix="Multi-sample",
        )
        if any(column in summary_df.columns for column in ["autoStartFrame", "autoStartTime_s", "firstReliableObjectFrame", "firstReliableObjectTime_s"]):
            st.subheader("Start-frame detection summary")
            st.caption("These per-sample start summaries are separate from sample-mean correlations and average metric profiles.")
            if "autoStartFrame" in summary_df.columns:
                st.pyplot(plot_batch_bar(summary_df, "autoStartFrame", "Auto start frame by sample", "Frame index"), clear_figure=True)
            if "autoStartTime_s" in summary_df.columns:
                st.pyplot(plot_batch_bar(summary_df, "autoStartTime_s", "Auto start time by sample", "time_s"), clear_figure=True)
            if "firstReliableObjectFrame" in summary_df.columns:
                st.pyplot(plot_batch_bar(summary_df, "firstReliableObjectFrame", "First reliable object frame by sample", "Frame index"), clear_figure=True)
            if "firstReliableObjectTime_s" in summary_df.columns:
                st.pyplot(plot_batch_bar(summary_df, "firstReliableObjectTime_s", "First reliable object time by sample", "time_s"), clear_figure=True)
        st.subheader("Sample-Level Metric Bars")
        st.caption("Stored sample-summary statistics from the batch/export. The time-curve display controls below "
                   "do not recalculate these bars. Check the original summary's frame-selection settings.")
        statistic = st.selectbox("Bar plot statistic", ["median", "mean", "final"], index=0, key=f"{key_prefix}_bar_statistic")
        group_by = st.selectbox(
            "Bar plot grouping",
            ["Sample", "flowHistory", "stentStatus", "flowHistory x stentStatus"],
            index=0,
            key=f"{key_prefix}_bar_grouping",
        )
        plot_columns = st.columns(2)
        for index, metric in enumerate(prepared.get("barMetrics", BAR_METRICS)):
            with plot_columns[index % 2]:
                st.pyplot(plot_multi_sample_bar(summary_df, metric, statistic=statistic, group_by=group_by), clear_figure=True)

        st.subheader("Selected Sample Time Curves")
        available_time_metrics = [metric for metric in prepared.get("timeCurveMetrics", TIME_CURVE_METRICS) if metric in frames_df.columns]
        if not available_time_metrics:
            st.warning("Time-curve metrics are missing from frame_metrics.csv.")
        else:
            curve_metric = st.selectbox("Time-curve metric", available_time_metrics, index=0, key=f"{key_prefix}_curve_metric")
            sample_names = summary_df["sampleName"].astype(str).tolist()
            selected_samples = st.multiselect(
                "Samples shown in time curves",
                sample_names,
                default=sample_names[: min(6, len(sample_names))],
                key=f"{key_prefix}_curve_samples",
            )
            time_axis = st.selectbox(
                "Time curve x-axis",
                ["time_s if available", "matlabTime_s", "originalImageIndex_1based", "analysisIndex_1based"],
                index=0,
                key=f"{key_prefix}_curve_axis",
            )
            show_curve_excluded = st.checkbox("Show excluded values in grey", value=False, key=f"{key_prefix}_curve_excluded")
            st.caption("QC-aware values are used where available. Lines stop at excluded/missing frames. "
                       "Yellow triangles are accepted warnings; grey circles are optional raw diagnostics. "
                       "Grey points never enter accepted statistics. Scores may use different within-sequence normalization across samples.")
            figure, curve_warnings = plot_multi_sample_time_curves(
                frames_df, curve_metric, selected_samples, time_axis, show_excluded=show_curve_excluded,
            )
            for warning in curve_warnings:
                st.warning(warning)
            st.pyplot(figure, clear_figure=True)
            plt.close(figure)
            for selected_sample in selected_samples:
                sample_frames = frames_df.loc[frames_df["sampleName"].astype(str) == str(selected_sample)]
                st.caption(f"{selected_sample}: {count_caption(metric_view(sample_frames, curve_metric))}")
            with st.expander("Time-curve values and exclusion reasons", expanded=False):
                shown_frames = frames_df.loc[frames_df["sampleName"].astype(str).isin([str(name) for name in selected_samples])]
                detail = qc_details(shown_frames, curve_metric)
                st.dataframe(detail, use_container_width=True, hide_index=True)
                st.download_button("Download time-curve QC table", detail.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="time_curves_QC.csv", mime="text/csv", key=f"{key_prefix}_curve_qc_download")

    with tabs[2]:
        st.subheader("Inferred Experimental Group Summary")
        if group_df.empty:
            st.info("No flowHistory or stentStatus groups could be inferred from the sample names.")
        else:
            st.dataframe(group_df, use_container_width=True, hide_index=True)
            if (group_df.get("nSamples", pd.Series(dtype=int)) < 2).any():
                st.warning("Some groups contain fewer than 2 samples. Interpret these group summaries cautiously.")

    with tabs[3]:
        st.subheader("Optional MATLAB Comparison Data")
        if matlab_df.empty:
            st.info("No MATLAB comparison CSV was supplied or stored for this dashboard.")
        else:
            st.dataframe(matlab_df, use_container_width=True, hide_index=True)

    with tabs[4]:
        st.subheader("Multi-Sample Comparison Report")
        st.text_area("Comparison report", report_text, height=460, key=f"{key_prefix}_report_preview")
        render_multi_sample_report_download(
            "Download multi-sample comparison report (.txt)",
            "multi_sample_comparison_report_txt",
            "MicroStructAI_multi_sample_comparison_report.txt",
            "text/plain",
            f"{key_prefix}_download_report_txt",
        )
        render_multi_sample_report_download(
            "Download multi-sample comparison report (.md)",
            "multi_sample_comparison_report_md",
            "MicroStructAI_multi_sample_comparison_report.md",
            "text/markdown",
            f"{key_prefix}_download_report_md",
        )

    with tabs[5]:
        st.subheader("Multi-Sample Comparison Exports")
        render_csv_download(
            "Download multi-sample comparison summary CSV",
            "multi_sample_comparison_summary_df",
            "MicroStructAI_multi_sample_comparison_summary.csv",
            f"{key_prefix}_download_summary_csv",
        )
        render_csv_download(
            "Download multi-sample group summary CSV",
            "multi_sample_group_summary_df",
            "MicroStructAI_multi_sample_group_summary.csv",
            f"{key_prefix}_download_group_csv",
        )
        render_csv_download(
            "Download sample mean metric profile CSV",
            f"{average_behaviour_prefix}_sample_mean_metric_profile_df",
            "sample_mean_metric_profile.csv",
            f"{key_prefix}_download_sample_mean_metric_profile_csv",
        )
        render_csv_download(
            "Download group average metric profile CSV",
            f"{average_behaviour_prefix}_group_average_metric_profile_df",
            "group_average_metric_profile.csv",
            f"{key_prefix}_download_group_average_metric_profile_csv",
        )
        render_csv_download(
            "Download sample-mean metric table CSV",
            f"{average_behaviour_prefix}_sample_mean_metric_table_df",
            f"sample_mean_metric_table_{safe_filename_token('multi_sample')}.csv",
            f"{key_prefix}_download_sample_mean_metric_table_csv",
        )
        render_csv_download(
            "Download sample-mean correlation matrix CSV",
            f"{average_behaviour_prefix}_sample_mean_correlation_compact_df",
            "sample_mean_correlation_compact_multi_sample.csv",
            f"{key_prefix}_download_sample_mean_correlation_compact_csv",
        )
        render_csv_download(
            "Download full sample-mean correlation matrix CSV",
            f"{average_behaviour_prefix}_sample_mean_correlation_full_df",
            "sample_mean_correlation_full_multi_sample.csv",
            f"{key_prefix}_download_sample_mean_correlation_full_csv",
        )
        render_multi_sample_report_download(
            "Download multi-sample comparison report (.txt)",
            "multi_sample_comparison_report_txt",
            "MicroStructAI_multi_sample_comparison_report.txt",
            "text/plain",
            f"{key_prefix}_download_report_txt_exports",
        )
        render_multi_sample_report_download(
            "Download multi-sample comparison report (.md)",
            "multi_sample_comparison_report_md",
            "MicroStructAI_multi_sample_comparison_report.md",
            "text/markdown",
            f"{key_prefix}_download_report_md_exports",
        )
        render_export_diagnostics(
            {
                "multi_sample_comparison_summary": summary_df,
                "multi_sample_group_summary": group_df,
                "sample_mean_metric_profile": st.session_state.get(f"{average_behaviour_prefix}_sample_mean_metric_profile_df", pd.DataFrame()),
                "group_average_metric_profile": st.session_state.get(f"{average_behaviour_prefix}_group_average_metric_profile_df", pd.DataFrame()),
                "sample_mean_metric_table": st.session_state.get(f"{average_behaviour_prefix}_sample_mean_metric_table_df", pd.DataFrame()),
                "sample_mean_correlation_compact": st.session_state.get(f"{average_behaviour_prefix}_sample_mean_correlation_compact_df", pd.DataFrame()),
                "sample_mean_correlation_full": st.session_state.get(f"{average_behaviour_prefix}_sample_mean_correlation_full_df", pd.DataFrame()),
                "multi_sample_comparison_report_txt": report_text,
                "multi_sample_comparison_report_md": report_text,
            }
        )


def render_batch_validation(params: dict[str, Any]) -> None:
    """Render batch validation discovery, processing, plots, reports, and exports."""
    st.header("Batch Validation")
    batch_validation_mode = str(params.get("batch_validation_mode", "RBC Gap Mode"))
    is_rbc_batch = batch_validation_mode == "RBC Gap Mode"
    use_matlab_csv = bool(params.get("batch_use_matlab_csv", True))
    batch_input_mode = str(params.get("batch_input_mode", "Images + optional MATLAB CSV comparison"))
    source_label = "sample folders in your uploaded ZIP" if WEB_DEPLOYMENT else "local sample folders"
    st.write(f"Analyse {source_label} using {batch_validation_mode}.")
    st.caption(f"Selected batch validation analysis mode: {batch_validation_mode}")
    if batch_validation_mode == "Ground-Truth Mask Mode":
        st.warning("Ground-Truth Mask Mode batch validation is not available yet.")
        return
    if use_matlab_csv:
        st.caption("Current batch input type: images with optional MATLAB CSV comparison.")
    else:
        st.success("Current batch input type: image folders only. MATLAB CSV files are not required and will be ignored.")
    stored_report_txt = st.session_state.get("batch_validation_report_txt")
    if not isinstance(stored_report_txt, str) or not stored_report_txt.strip():
        st.warning("No batch validation report available for export yet. Run batch validation first.")

    parent_folder = str(params.get("batch_parent_folder", "")).strip()
    if not parent_folder:
        st.info("Upload a ZIP of your sample folders in the sidebar." if WEB_DEPLOYMENT else "Enter the parent sample folder path in the sidebar.")
        return

    image_uploads = None
    if WEB_DEPLOYMENT:
        image_uploads = st.session_state.get("web_batch_samples", {})
        discovered_df = pd.DataFrame([
            {"include": True, "sampleName": name, "imageFolder": name, "nImages": len(files),
             "matlabCSVFound": False, "matlabCSVPath": "", **parse_batch_sample_labels(name)}
            for name, files in image_uploads.items()
        ])
        discovery_warnings = []
    else:
        discovered_df, discovery_warnings = discover_batch_samples(parent_folder, detect_matlab_csv=use_matlab_csv)
    for warning in discovery_warnings:
        st.warning(warning)
    if discovered_df.empty:
        st.info("No batch samples are available yet.")
        return

    uploaded_matlab_csvs = []
    uploaded_matlab_csv_bytes: dict[str, bytes] = {}
    uploaded_matlab_master_summaries: dict[str, dict[str, Any]] = {}
    selected_uploaded_master = ""
    matlab_csv_selection_options = ["Skip MATLAB comparison"]
    if not WEB_DEPLOYMENT:
        matlab_csv_selection_options.insert(0, "Auto-detected local CSV")
    if use_matlab_csv:
        st.subheader("Choose MATLAB CSV Files")
        st.caption(
            "Select MATLAB CSV files with the file picker, then choose the CSV for each sample in the table below. "
            "Apply one combined master CSV to every sample, or skip comparison for a sample."
        )
        uploaded_matlab_csvs = st.file_uploader(
            "Select MATLAB CSV files",
            type=["csv"],
            accept_multiple_files=True,
            key="batch_matlab_csv_file_picker",
        ) or []
        uploaded_matlab_csv_bytes = {csv_file.name: csv_file.getvalue() for csv_file in uploaded_matlab_csvs}
        matlab_csv_selection_options += [f"Uploaded: {name}" for name in uploaded_matlab_csv_bytes]
        if uploaded_matlab_csv_bytes:
            st.success(f"MATLAB CSV files selected in this window: {len(uploaded_matlab_csv_bytes)}")
            uploaded_matlab_master_summaries = {
                name: uploaded_matlab_master_info(payload) for name, payload in uploaded_matlab_csv_bytes.items()
            }
            master_candidates = [
                name for name, info in uploaded_matlab_master_summaries.items() if int(info.get("nSampleKeys", 0)) > 1
            ]
            if master_candidates:
                master_options = ["Do not assign an uploaded master automatically"] + master_candidates
                selected_uploaded_master = st.selectbox(
                    "Combined MATLAB master CSV for all samples",
                    master_options,
                    index=1 if len(master_candidates) == 1 else 0,
                    help="The selected master CSV is filtered by sampleKey before isAna and time_s matching. Individual rows can still be overridden below.",
                )
                if selected_uploaded_master == "Do not assign an uploaded master automatically":
                    selected_uploaded_master = ""
                else:
                    master_info = uploaded_matlab_master_summaries[selected_uploaded_master]
                    st.info(
                        f"Using {selected_uploaded_master} as a filtered master reference: "
                        f"{master_info.get('rows', 0)} rows across {master_info.get('nSampleKeys', 0)} samples. "
                        f"Each batch sample will use only its matching {master_info.get('sampleKeyColumn', 'sampleKey')} rows."
                    )

    st.subheader("Detected Samples")
    if use_matlab_csv:
        st.caption(
            "For each sample, choose a MATLAB CSV from the dropdown. Combined master CSVs are filtered by sampleKey before comparison. "
            "Choose an uploaded file or skip comparison for a sample."
        )
        display_df = discovered_df.copy()
        if selected_uploaded_master:
            display_df["matlabCSVSelection"] = f"Uploaded: {selected_uploaded_master}"
        else:
            display_df["matlabCSVSelection"] = np.where(
                display_df["matlabCSVPath"].fillna("").astype(str).str.strip() != "",
                "Auto-detected local CSV",
                "Skip MATLAB comparison",
            )
        disabled_columns = ["imageFolder", "nImages", "matlabCSVFound"]
        if WEB_DEPLOYMENT:
            disabled_columns.append("matlabCSVPath")
        column_config = {
            "include": st.column_config.CheckboxColumn("Include", default=True),
            "nImages": st.column_config.NumberColumn("Images", format="%d"),
            "matlabCSVFound": st.column_config.CheckboxColumn("Auto-detected MATLAB CSV"),
            "matlabCSVSelection": st.column_config.SelectboxColumn(
                "Choose MATLAB CSV",
                options=matlab_csv_selection_options,
                required=True,
                help="Use an uploaded CSV selected in this window, or skip MATLAB comparison for this sample.",
            ),
            "matlabCSVPath": None if WEB_DEPLOYMENT else st.column_config.TextColumn("Local MATLAB CSV path"),
        }
    else:
        st.caption("Image-only mode: edit sample names and condition labels. MATLAB CSV detection and comparison are skipped.")
        display_df = discovered_df.drop(columns=["matlabCSVFound", "matlabCSVPath"], errors="ignore")
        disabled_columns = ["imageFolder", "nImages"]
        column_config = {
            "include": st.column_config.CheckboxColumn("Include", default=True),
            "nImages": st.column_config.NumberColumn("Images", format="%d"),
        }
    edited_df = st.data_editor(
        display_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=disabled_columns,
        column_config=column_config,
        key=f"batch_detected_samples_editor_{parent_folder}_{'matlab' if use_matlab_csv else 'image_only'}_{selected_uploaded_master or 'per_sample'}",
    )
    if not use_matlab_csv:
        edited_df["matlabCSVFound"] = False
        edited_df["matlabCSVPath"] = ""
    selected_count = int(edited_df["include"].fillna(False).astype(bool).sum()) if "include" in edited_df.columns else len(edited_df)
    st.caption(f"Samples selected for batch analysis: {selected_count}")

    if st.button("Run Batch Analysis", type="primary", disabled=selected_count == 0):
        with st.spinner("Running batch validation..."):
            run_params = dict(params)
            run_params["batch_uploaded_matlab_csv_bytes"] = uploaded_matlab_csv_bytes
            batch_result = run_batch_validation(edited_df, run_params, image_uploads=image_uploads)
            st.session_state["batch_validation_result"] = batch_result
            store_batch_outputs_in_session_state(batch_result)
            st.session_state["batch_validation_parent"] = parent_folder
            st.session_state["batch_validation_input_mode"] = batch_input_mode
            st.session_state["batch_validation_mode"] = batch_validation_mode

    result = st.session_state.get("batch_validation_result")
    if not result:
        st.warning("No batch validation report available for export yet. Run batch validation first.")
        return
    required_batch_state_keys = {
        "batch_sample_summary_df",
        "batch_frame_metrics_df",
        "batch_matlab_comparison_df",
        "batch_paired_comparison_df",
        "batch_report_md",
        "batch_report_preview_md",
        "batch_validation_report_txt",
        "batch_processing_notes_total_count",
        "batch_processing_notes_preview_count",
        "batch_processing_notes_export_count",
    }
    if not all(state_key in st.session_state for state_key in required_batch_state_keys):
        store_batch_outputs_in_session_state(result)
    if st.session_state.get("batch_validation_parent") != parent_folder:
        st.warning("The sample source changed after the last batch run. Run Batch Analysis again to refresh results.")
    if st.session_state.get("batch_validation_input_mode") != batch_input_mode:
        st.warning("The batch input type changed after the last batch run. Run Batch Analysis again to refresh results.")
    if st.session_state.get("batch_validation_mode") != batch_validation_mode:
        st.warning("The batch validation analysis mode changed after the last batch run. Run Batch Analysis again to refresh results.")

    summary_df = st.session_state["batch_sample_summary_df"]
    frame_metrics_df = st.session_state["batch_frame_metrics_df"]
    matlab_summary_df = st.session_state["batch_matlab_comparison_df"]
    paired_df = st.session_state["batch_paired_comparison_df"]
    report_md = st.session_state["batch_report_md"]
    report_preview_md = st.session_state["batch_report_preview_md"]
    store_default_average_behaviour_exports(
        summary_df,
        "batch_average_behaviour",
        "rbc" if is_rbc_batch else "object",
        f"Batch {batch_validation_mode}",
    )

    tabs = st.tabs(["Sample Summary", "Frame Metrics", "MATLAB Comparison", "Plots", "Multi-Sample Comparison", "Report", "Exports"])
    with tabs[0]:
        st.subheader("Batch Sample Summary")
        if summary_df.empty:
            st.info("No sample summary rows were generated.")
        else:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("Combined Per-Frame Metrics")
        if frame_metrics_df.empty:
            st.info("No per-frame metrics were generated.")
        else:
            st.dataframe(frame_metrics_df, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("Batch MATLAB Comparison Summary")
        if not is_rbc_batch:
            st.info("MATLAB RBC CSV comparison is available only for RBC Gap Mode batches.")
        elif not use_matlab_csv:
            st.info("Image-only batch mode is enabled, so MATLAB CSV comparison was skipped.")
        elif matlab_summary_df.empty:
            st.info("No MATLAB comparison summaries are available. Add per-sample agg_features_*_GAPS.csv files to enable validation.")
        else:
            st.caption(
                "The exact MATLAB reference file selected for each sample is listed in the Sample Summary table under matlabCSVPath. "
                "When a combined master CSV is used, matlabSampleFilterValue shows the sampleKey subset used for comparison."
            )
            st.dataframe(matlab_summary_df, use_container_width=True, hide_index=True)
        if not paired_df.empty:
            with st.expander("Combined paired frame data", expanded=False):
                st.dataframe(paired_df, use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("Batch Validation Plots")
        render_sample_average_behaviour_section(
            summary_df,
            key_prefix="batch_average_behaviour",
            kind="rbc" if is_rbc_batch else "object",
            title_prefix=f"Batch {batch_validation_mode}",
        )
        if is_rbc_batch and use_matlab_csv:
            st.pyplot(plot_batch_pearson_heatmap(matlab_summary_df), clear_figure=True)
            st.pyplot(plot_batch_bar(summary_df, "meanPearsonMainMetrics", "Mean Pearson r across main metrics", "Mean Pearson r"), clear_figure=True)
            st.pyplot(plot_batch_bar(summary_df, "meanPearsonScores", "Mean Pearson r across score metrics", "Mean Pearson r"), clear_figure=True)
        elif is_rbc_batch:
            st.info("Image-only mode: MATLAB correlation plots are skipped. QC and condition summary plots remain available.")
        if is_rbc_batch:
            st.pyplot(plot_batch_qc_counts(summary_df), clear_figure=True)
            st.subheader("Start-frame detection summary")
            st.caption("Per-sample start-frame summaries are separate from batch correlation and average-behaviour plots.")
            if "autoStartFrame" in summary_df.columns:
                st.pyplot(plot_batch_bar(summary_df, "autoStartFrame", "Auto start frame by sample", "Frame index"), clear_figure=True)
            if "autoStartTime_s" in summary_df.columns:
                st.pyplot(plot_batch_bar(summary_df, "autoStartTime_s", "Auto start time by sample", "time_s"), clear_figure=True)
            if "autoStartFrame" not in summary_df.columns and "autoStartTime_s" not in summary_df.columns:
                st.info("Start-frame summary columns were not available for this batch run.")
            if "noStartSafetyTriggered" in summary_df.columns:
                no_start_count = int(summary_df["noStartSafetyTriggered"].fillna(False).astype(bool).sum())
                st.metric("No-start safety count", no_start_count)
            st.pyplot(plot_batch_condition_summary(summary_df), clear_figure=True)
        else:
            st.info("Object Mode batch plots summarize generic morphology. They do not use RBC-specific interpretation.")
            for column, title, ylabel in [
                ("median_areaFrac", "Median area fraction by sample", "areaFrac"),
                ("median_objectCount", "Median object count by sample", "objectCount"),
                ("median_meanDiam_px", "Median object diameter by sample", "meanDiam_px"),
                ("median_correlationLength_px", "Median correlation length by sample", "correlationLength_px"),
                ("median_gSASI_0_100", "Median exploratory gSASI by sample", "gSASI_0_100"),
                ("median_gAggIndex_0_100", "Median exploratory gAggIndex by sample", "gAggIndex_0_100"),
            ]:
                if column in summary_df.columns and pd.to_numeric(summary_df[column], errors="coerce").notna().any():
                    st.pyplot(plot_batch_bar(summary_df, column, title, ylabel), clear_figure=True)

    with tabs[4]:
        render_multi_sample_comparison_dashboard(key_prefix="batch_multi_sample")

    with tabs[5]:
        st.subheader("Batch Validation Report")
        st.text_area("Report preview", report_preview_md, height=460)
        render_batch_processing_note_diagnostics()
        render_batch_markdown_download(
            "batch_report_md",
            "Download batch markdown report",
            f"MicroStructAI_batch_validation_{safe_filename_token(batch_validation_mode)}_report.md",
            "download_batch_validation_report_md",
        )
        render_batch_text_download(
            "batch_validation_report_txt",
            "Download batch validation report (.txt)",
            f"MicroStructAI_batch_validation_{safe_filename_token(batch_validation_mode)}_report.txt",
            "download_batch_validation_report_txt",
        )

    with tabs[6]:
        st.subheader("Batch Exports")
        st.caption("Downloads below use the final combined batch dataframes stored in session state immediately after analysis.")
        render_batch_dataframe_download(
            "batch_sample_summary_df",
            "Download batch sample summary CSV",
            "MicroStructAI_batch_sample_summary.csv",
            "download_batch_sample_summary_csv",
        )
        render_batch_dataframe_download(
            "batch_frame_metrics_df",
            "Download combined per-frame metrics CSV",
            "MicroStructAI_batch_frame_metrics.csv",
            "download_batch_frame_metrics_csv",
        )
        render_batch_dataframe_download(
            "batch_average_behaviour_sample_mean_metric_profile_df",
            "Download sample mean metric profile CSV",
            "sample_mean_metric_profile.csv",
            "download_batch_sample_mean_metric_profile_csv",
        )
        render_batch_dataframe_download(
            "batch_average_behaviour_group_average_metric_profile_df",
            "Download group average metric profile CSV",
            "group_average_metric_profile.csv",
            "download_batch_group_average_metric_profile_csv",
        )
        render_batch_dataframe_download(
            "batch_average_behaviour_sample_mean_metric_table_df",
            "Download sample-mean metric table CSV",
            "sample_mean_metric_table_batch.csv",
            "download_batch_sample_mean_metric_table_csv",
        )
        render_batch_dataframe_download(
            "batch_average_behaviour_sample_mean_correlation_compact_df",
            "Download compact sample-mean correlation matrix CSV",
            "sample_mean_correlation_compact_batch.csv",
            "download_batch_sample_mean_correlation_compact_csv",
        )
        render_batch_dataframe_download(
            "batch_average_behaviour_sample_mean_correlation_full_df",
            "Download full sample-mean correlation matrix CSV",
            "sample_mean_correlation_full_batch.csv",
            "download_batch_sample_mean_correlation_full_csv",
        )
        if is_rbc_batch and use_matlab_csv:
            render_batch_dataframe_download(
                "batch_matlab_comparison_df",
                "Download batch MATLAB comparison summary CSV",
                "MicroStructAI_batch_MATLAB_comparison_summary.csv",
                "download_batch_matlab_summary_csv",
            )
            render_batch_dataframe_download(
                "batch_paired_comparison_df",
                "Download combined paired MATLAB comparison data CSV",
                "MicroStructAI_batch_paired_comparison_data.csv",
                "download_batch_paired_comparison_data_csv",
            )
        else:
            st.caption("Image-only mode: MATLAB comparison CSV exports are not shown.")
        render_batch_markdown_download(
            "batch_report_md",
            "Download batch markdown report",
            f"MicroStructAI_batch_validation_{safe_filename_token(batch_validation_mode)}_report.md",
            "download_batch_validation_report_md_exports",
        )
        render_batch_processing_note_diagnostics()
        render_batch_text_download(
            "batch_validation_report_txt",
            "Download batch validation report (.txt)",
            f"MicroStructAI_batch_validation_{safe_filename_token(batch_validation_mode)}_report.txt",
            "download_batch_validation_report_txt_exports",
        )
        render_batch_export_diagnostics()


def render_report(analysis: dict[str, Any] | None, pca_result: PCAResult | None, matlab_comparison_result: dict[str, Any] | None = None) -> None:
    """Render and export a simple text report."""
    if not analysis or analysis["metrics"].empty:
        st.info("Run analysis to generate a report.")
        return
    render_sample_report(analysis)
    report_text = generate_report(
        analysis["mode"],
        visible_metrics_df(analysis["metrics"]),
        analysis["params"],
        pca_result=pca_result,
        warnings=analysis["warnings"],
        matlab_comparison_result=matlab_comparison_result,
    )
    structural = analysis.get("structural_intelligence", {})
    if structural.get("enabled") and structural.get("passport"):
        report_text += "\n\n" + passport_markdown(structural["passport"])
    if analysis.get("v4_context"):
        from modules.v4_pipeline import ensure_measurements
        from modules.v4_report import json_data
        ensure_measurements(analysis)
        report_text += "\n\nV4 DEVELOPMENT PROVENANCE — complete per-input records\n"
        report_text += json_data(analysis["v4_context"]).decode("utf-8")
        report_text += "\n\nDIRECT DESCRIPTOR AVAILABILITY / UNITS\n" + analysis["v4_metric_availability"].to_csv(index=False)
        for category in ("IMAGE QUALITY", "MASK PLAUSIBILITY", "TEMPORAL / REFERENCE ELIGIBILITY"):
            diagnostics = analysis["v4_diagnostics"]
            report_text += "\n\n" + category + "\n" + diagnostics.loc[diagnostics.category.eq(category)].to_csv(index=False)
    st.text_area("Report", report_text, height=360)
    render_text_download(
        "Download report (.txt)",
        report_text,
        "microstructai_report.txt",
        "text/plain",
        "download_single_sample_report_txt",
    )
    render_export_diagnostics({"single_sample_report": report_text})


@st.fragment
def render_results_workspace(mode, image_files, mask_files, params, analysis, metrics_df) -> None:
    """Render only the requested section; result widgets do not rebuild the sidebar."""
    if st.session_state.get("_current_display_analysis") is not analysis:
        st.session_state["_current_display_analysis"] = analysis
        st.session_state.pop("_v3_display_cache", None)
        st.session_state.pop("_last_pca_result", None)
        st.session_state.pop("matlab_comparison_result", None)
        st.session_state.pop("_structural_export_owner", None)
        st.session_state.pop("_structural_export_bytes", None)
    names = ["Overview", "Segmentation Preview"]
    if mode == "RBC Gap Mode":
        names.append("Morphology & Dynamics")
    names += ["Quality & Interpretation", "Metrics Table", "Plots", "PCA"]
    if mode == "RBC Gap Mode":
        names.append("MATLAB Comparison")
    names.append("Report")
    if st.session_state.get("results_section", names[0]) not in names:
        st.session_state["results_section"] = names[0]
    section = st.radio("Results section", names, horizontal=True, key="results_section")
    st.caption("Only the selected section loads. Switching views does not rerun image analysis.")
    if analysis:
        if len(analysis.get("metrics", [])) == 1:
            st.warning("Single image: interpret raw spatial descriptors only. Any legacy sequence-relative score shown in detailed views is not interpretable as an aggregation level or a temporal result.")
        if not analysis.get("development_context", {}).get("acquisition", {}).get("timing_confirmed"):
            st.caption("Timing unconfirmed: any seconds axis is provisional. Use frame numbers until acquisition timing is checked.")
    axis = params.get("plot_x_axis", "analysisIndex_1based")
    if section == "Overview":
        render_overview(mode, image_files, mask_files, analysis.get("params", params) if analysis else params, analysis)
    elif section == "Segmentation Preview":
        render_segmentation_preview(analysis)
    elif section == "Morphology & Dynamics":
        render_structural_intelligence(analysis.get("structural_intelligence") if analysis else None)
    elif section == "Quality & Interpretation":
        render_quality(analysis)
    elif section == "Metrics Table":
        render_metrics_table(metrics_df)
    elif section == "Plots":
        render_plots(metrics_df, axis)
    elif section == "PCA":
        st.session_state["_last_pca_result"] = render_pca(metrics_df, axis)
    elif section == "MATLAB Comparison":
        render_matlab_comparison(analysis)
    elif section == "Report":
        pca_result = st.session_state.get("_last_pca_result")
        if pca_result is None:
            st.info("PCA has not been viewed for this analysis. Open PCA first if you want its results included in the report.")
        else:
            st.caption("PCA uses your most recently viewed selection for this analysis: " + ", ".join(pca_result.used_metrics))
        render_report(analysis, pca_result, st.session_state.get("matlab_comparison_result"))


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(page_title="MicroStructAI Development", layout="wide")
    st.title("MicroStructAI — Development copy")
    st.caption("Interpretable structural characterisation for image datasets · research use only")
    st.warning("Separate development copy. Your working V3 and online app are unchanged. V3 segmentation and score formulas are preserved; ROI experiments are optional and not externally validated.")
    st.caption(f"Interface {APP_VERSION} · V3 formula engine · no publishing shortcut in this copy")
    workspace = st.sidebar.radio("V4 workspace", ["Automatic analysis (V3-compatible algorithms)", "Verified Mask Analysis"], key="v4_workspace")
    if workspace == "Verified Mask Analysis":
        from modules.v4_ui import verified_mask_workspace
        verified_mask_workspace()
        return
    if WEB_DEPLOYMENT:
        st.info(
            "Upload images for single-sample analysis, a ZIP of sample folders for batch analysis, "
            "or exported CSV files for multi-sample comparison. Camera calibration and presets are available in Analysis mode."
        )
        st.warning(
            "Do not upload patient-identifiable, confidential, or clinically sensitive material. "
            "MicroStructAI V3 is not a medical device and must not be used for diagnosis or treatment decisions."
        )

    mode, image_files, mask_files, params, run_clicked = build_sidebar()
    if mode == "Ground-Truth Mask Mode":
        st.warning("Use Verified Mask Analysis in the V4 workspace selector. It supports image+mask or mask-only input, explicit polarity, strict alignment checks and no automatic segmentation.")
        return
    if mode == "RBC Gap Mode":
        st.caption(
            f"Experimental mask-derived SASI ({RBC_SCORE_FORMULA_VERSION}). "
            "Default normalization is within-sequence, not an absolute cross-sample aggregation scale. See Quality & Interpretation."
        )

    if mode == "Batch Validation":
        st.info("Legacy batch numbers are preserved. V4 appends per-sample source/mask/ROI provenance and separate diagnostics to new batch reports; old CSV layouts remain compatibility exports.")
        render_batch_validation(params)
        batch_result = st.session_state.get("batch_validation_result", {})
        if batch_result.get("v4_batch_provenance"):
            from modules.v4_report import json_data
            st.download_button("Download complete V4 batch provenance (JSON)", json_data(batch_result["v4_batch_provenance"]), "MicroStructAI_V4_batch_provenance.json", "application/json", key="v4_batch_provenance_download", on_click="ignore")
        return
    if mode == "Multi-Sample Comparison":
        st.info("Compatibility viewer for previously exported tables. Missing source/mask provenance cannot be reconstructed from those tables; this is not a new V4 measurement report.")
        render_multi_sample_comparison_dashboard(key_prefix="standalone_multi_sample")
        return
    if mode == "Auto-Tuning / Camera Calibration":
        render_camera_calibration(params)
        return

    params, acquisition = input_controls(image_files, params, mode)
    if run_clicked:
        if not image_files:
            st.error("No images are selected for analysis. Check the input method, folder/uploads, and image range.")
        elif mode == "Ground-Truth Mask Mode" and not mask_files:
            st.error("No ground-truth mask/outline files are selected for analysis.")
        else:
            with st.spinner("Running analysis..."):
                try:
                    # Freeze chosen settings/metadata for the result, not later sidebar edits.
                    analysis = analyze_dataset(image_files, mask_files, mode, deepcopy(params))
                    analysis["development_context"]["acquisition"] = deepcopy(acquisition)
                    update_acquisition(analysis, acquisition)
                    try:
                        analysis["development_roi_png"] = roi_preview(image_files, params, mode)
                    except (ValueError, OSError) as exc:
                        analysis["warnings"].append(f"Crop preview unavailable: {exc}")
                    st.session_state["analysis"] = analysis
                    st.session_state.pop("_dev_run_failed", None)
                except (ValueError, OSError) as exc:
                    st.session_state["_dev_run_failed"] = str(exc)

    if st.session_state.get("_dev_run_failed"):
        st.error(st.session_state["_dev_run_failed"])
        if st.session_state.get("analysis"):
            st.warning("The requested run did not complete. Any results below belong to the PREVIOUS successful run.")

    analysis = st.session_state.get("analysis")
    if analysis and analysis.get("mode") != mode:
        st.info("Sidebar mode changed. Run analysis again to refresh results for the selected mode.")
    if analysis and analysis.get("development_context"):
        saved = analysis["development_context"]
        if (saved.get("input_settings_sha256") != settings_digest(params)
                or saved.get("acquisition", {}) != acquisition
                or saved.get("input_check", {}).get("selection_signature") != input_selection_signature(image_files)):
            st.warning("Inputs, controls or metadata differ from the displayed run. Results and downloads retain their captured settings. Run analysis again to apply changes.")

    if mode == "Object Mode" and analysis and analysis.get("v4_context"):
        from modules.v4_ui import render_v4_results
        render_v4_results(analysis)
        return
    metrics_df = visible_metrics_df(analysis["metrics"]) if analysis else pd.DataFrame()
    metrics_df = store_export_dataframe("single_sample_metrics_df", metrics_df)
    if not metrics_df.empty:
        render_csv_download(
            "Export legacy native metrics CSV (full provenance in Report ZIP)",
            "single_sample_metrics_df",
            "MicroStructAI_single_sample_metrics.csv",
            "download_sidebar_single_sample_metrics_csv",
            container=st.sidebar,
        )

    render_results_workspace(mode, image_files, mask_files, params, analysis, metrics_df)


# Reporting-only registration. The original batch calculation/report bodies
# remain byte/AST-preserved and are still checked against the V3 source.
from modules.v4_batch_provenance import with_batch_provenance, with_batch_report_provenance
run_batch_validation = with_batch_provenance(run_batch_validation)
batch_report_strings_for_storage = with_batch_report_provenance(batch_report_strings_for_storage)

if __name__ == "__main__":
    main()
