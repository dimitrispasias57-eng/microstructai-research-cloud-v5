"""Structural metric extraction for binary masks and source images."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import signal
from scipy.optimize import least_squares
from skimage import measure, morphology, transform

from .rbc_scores import compute_official_mask_scores
from .segmentation import resize_mask_to_shape, to_grayscale


CORE_METRICS = [
    "areaFrac",
    "nObjects",
    "meanArea",
    "medianArea",
    "meanDiam",
    "medianDiam",
    "largestObjectFrac",
    "Xi_px",
    "lacunarity16",
    "skelDensity",
    "branchDensity",
    "endpointDensity",
    "anisotropy",
    "orientOrderParam",
    "orientation_deg",
    "interfaceDensity_px",
    "percolates",
    "largestClusterFrac_phase",
    "percolates_complement",
    "largestClusterFrac_complement",
    "intensityMean",
    "intensityStd",
]

GENERIC_OBJECT_METRICS = [
    "selectedObjectMethod",
    "objectAnalysisMethod",
    "conservativeObjectModeEnabled",
    "experimentalPODObjectModeEnabled",
    "enableObjectStructureScores",
    "finalObjectMaskSource",
    "finalObjectPreprocessingSource",
    "finalObjectAreaFracSource",
    "objectAreaFracFinal",
    "objectNtotalFinal",
    "objectMeanAreaFinal",
    "objectMeanDiamFinal",
    "objectLargestFracFinal",
    "objectLargestAreaFinal",
    "conservativeNoPODObjectAreaFrac",
    "conservativeNoPODObjectNtotal",
    "conservativeNoPODObjectMeanDiam_px",
    "conservativeNoPODObjectLargestFrac",
    "podRemovedObjectAreaFrac",
    "podRemovedObjectNtotal",
    "podRemovedObjectMeanDiam_px",
    "podRemovedObjectLargestFrac",
    "podObjectAreaDelta",
    "podObjectAreaRatio",
    "podObjectConsistencyFlag",
    "Object_Xi_raw_intensity",
    "Object_Xi_mask",
    "Object_Xi_QC",
    "Object_Xi_isTrusted",
    "Object_Xi_QC_reason",
    "Object_Xi_mask_noPOD",
    "Object_Xi_mask_PODremoved",
    "objectXiSource",
    "ObjectStructureScore_raw",
    "ObjectStructureScore_raw_0_100",
    "ObjectStructureScore_QC",
    "ObjectStructureScore_QC_0_100",
    "ObjectStructureScore_0_100",
    "ObjectStructure_isTrusted",
    "ObjectStructure_QC_reason",
    "ObjectStructureScore_noPOD",
    "ObjectStructureScore_PODremoved",
    "ObjectAggregationIndex_raw",
    "ObjectAggregationIndex_raw_0_100",
    "ObjectAggregationIndex_QC",
    "ObjectAggregationIndex_QC_0_100",
    "ObjectAggregationIndex_0_100",
    "ObjectAggregation_isTrusted",
    "ObjectAggregation_QC_reason",
    "ObjectAggregationIndex_noPOD",
    "ObjectAggregationIndex_PODremoved",
    "ObjectSASI_like_0_100",
    "ObjectAggIndex_like_0_100",
    "objectScoreSource",
    "hardBadObjectMaskFlag",
    "objectMaskReliable",
    "objectFrameClass",
    "objectMaskTooSmallFlag",
    "objectMaskTooLargeFlag",
    "objectHugeObjectFlag",
    "objectLowSupportFlag",
    "podObjectSensitiveFlag",
    "spatialCorrectionSensitiveFlag",
    "rawGainCorrectionSensitiveFlag",
    "conservativeObjectWarningFlag",
    "objectRawMeanIntensity",
    "objectCorrectedMeanIntensity",
    "objectRawCV",
    "objectCorrectedCV",
    "objectIlluminationGain",
    "objectIlluminationJumpFlag",
    "enhancedObjectModeEnabled",
    "objectSegmentationTarget",
    "objectSegmentationTargetUsed",
    "experimentalCompositeScoresEnabled",
    "objectCount",
    "meanDiam_px",
    "medianDiam_px",
    "correlationLength_px",
    "sizeDistributionSlope",
    "sizeDistributionSlope_R2",
    "sizeDistributionSlope_nBins",
    "textureHeterogeneity_tileCV",
    "LF_log10_LFoverHF_generic",
    "skeletonDensity",
    "lowResolutionRisk",
    "lowResolutionWarning",
    "segmentationConfidence",
    "maxObjectAreaFilterEnabled",
    "maxObjectAreaFrac",
    "nLargeObjectsRemoved",
    "nLargeObjectsDetected",
    "largestRemovedObjectFrac",
    "largestObjectFrac_beforeFilter",
    "largestObjectFrac_afterFilter",
    "largeObjectArtefactFlag",
    "gSASI_raw",
    "gSASI_0_100",
    "gSASI_available",
    "gSASI_warning",
    "gAggIndex_0_100",
    "gAggIndex_available",
    "gAggIndex_warning",
]

RBC_GAP_METRICS = [
    "gapAreaFrac",
    "gapNtotal",
    "gapMeanDiam_px",
    "gapMedianDiam_px",
    "gapLargestArea_px",
    "gapLargestFrac",
    "largestToMedianGapRatio",
    "Xi_raw_intensity",
    "Xi_mask",
    "Xi_QC",
    "Xi_mask_noPOD",
    "Xi_mask_PODremoved",
    "xiSource",
    "mask_gapAreaFrac",
    "mask_gapNtotal",
    "mask_gapMeanArea",
    "mask_gapMeanDiam_px",
    "mask_gapMaxDiam_px",
    "mask_largestGapFrac",
    "mask_gapAreaFrac_noPOD",
    "mask_gapNtotal_noPOD",
    "mask_gapMeanDiam_px_noPOD",
    "mask_largestGapFrac_noPOD",
    "mask_gapAreaFrac_PODremoved",
    "mask_gapNtotal_PODremoved",
    "mask_gapMeanDiam_px_PODremoved",
    "mask_largestGapFrac_PODremoved",
    "Xi_interp_px",
    "gapAreaSlope",
    "gapAreaSlope_mvp",
    "gapAreaSlope_R2",
    "gapAreaSlope_nBins",
    "LF_log10_LFoverHF",
    "LF_log10_LFoverHF_mvp",
    "LF_log10_LFoverHF_matlab",
    "SHI_tileCV",
    "maxGapAreaFilterEnabled",
    "maxGapAreaFrac",
    "nLargeGapObjectsRemoved",
    "nLargeGapObjectsDetected",
    "largestRemovedGapFrac",
    "largestGapFrac_beforeFilter",
    "largestGapFrac_afterFilter",
    "largeGapArtefactFlag",
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
    "adaptiveRepairTriggered",
    "adaptiveRepairKept",
    "adaptiveRepairReason",
    "largestGapFrac_beforeRepair",
    "largestGapFrac_afterRepair",
    "largestToMedianGapRatio_beforeRepair",
    "largestToMedianGapRatio_afterRepair",
    "maskReliabilityScore_beforeRepair",
    "maskReliabilityScore_afterRepair",
]

RBC_SCORE_METRICS = [
    "SASI_raw",
    "SASI_0_100",
    "GapScore_raw",
    "GapScore_0_100",
    "PTI_raw",
    "PTI_0_100",
    "AggIndex_raw",
    "SASI_raw_0_100",
    "GapScore_raw_0_100",
    "PTI_raw_0_100",
    "AggIndex_raw_0_100",
    "S_struct_0_100",
    "AggIndex_0_100",
    "score_isTrusted",
    "visualQCExcluded",
    "visualGreyExcluded",
    "visualWarningOnly",
    "warningMaskFlag",
    "trustedForScore",
    "hardFailure",
    "excludePreStartFramesFromTrustedAnalysis",
    "excludeBadPostStartQCFramesFromTrustedAnalysis",
    "lowConfidenceFlag",
    "beforeStartFrame",
    "usedInSummary",
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
    "score_QC_reason",
    "Xi_isTrusted",
    "Xi_QC_reason",
    "SASI_isTrusted",
    "SASI_QC_reason",
    "GapScore_isTrusted",
    "GapScore_QC_reason",
    "PTI_isTrusted",
    "PTI_QC_reason",
    "AggIndex_isTrusted",
    "AggIndex_QC_reason",
    "MaskSASI_like_noPOD",
    "MaskSASI_like_PODremoved",
    "MaskGapScore_like_noPOD",
    "MaskGapScore_like_PODremoved",
    "MaskSASI_like_final",
    "MaskGapScore_like_final",
    "MaskSASI_raw_0_100",
    "MaskSASI_QC_0_100",
    "MaskSASI_isTrusted",
    "MaskSASI_QC_reason",
    "MaskGapScore_raw_0_100",
    "MaskGapScore_QC_0_100",
    "MaskGapScore_isTrusted",
    "MaskGapScore_QC_reason",
    "MaskAggIndex_raw_0_100",
    "MaskAggIndex_QC_0_100",
    "MaskAggIndex_isTrusted",
    "MaskAggIndex_QC_reason",
    "SASI_QC",
    "GapScore_QC",
    "PTI_QC",
    "AggIndex_QC",
    "SASI_QC_0_100",
    "GapScore_QC_0_100",
    "PTI_QC_0_100",
    "AggIndex_QC_0_100",
    "SASI_interpretable_0_100",
    "GapScore_interpretable_0_100",
    "PTI_interpretable_0_100",
    "AggIndex_interpretable_0_100",
    "rbcScoreInterpretable",
    "rbcScoreInterpretationFlag",
    "rbcScoreQCReason",
    "scoreSource",
]

RBC_SASI_SYLLECTOGRAM_METRICS = [
    "enableSASISyllectogramResearchOutput",
    "SASI_syllectogram_AI_like",
    "SASI_syllectogram_AMP_like",
    "SASI_syllectogram_T_half_like",
    "SASI_syllectogram_fit_R2",
    "SASI_syllectogram_fit_RMSE",
    "SASI_syllectogram_fit_success",
    "SASI_syllectogram_orientation",
    "SASI_syllectogram_startFrame_or_startTime",
    "SASI_syllectogram_axis",
    "SASI_syllectogram_T_half_units",
    "SASI_syllectogram_fit_method",
    "SASI_syllectogram_analysis_points",
    "SASI_syllectogram_warning",
]

NORMALIZED_SYLLECTOGRAM_METRICS = [
    "syllectogram_axis",
    "syllectogram_axis_units",
    "syllectogramSource_intensity",
    "syllectogramSource_SASI",
    "syllectogramSource_SASI_QC",
    "syllectogramSource_MaskGapScore",
    "syllectogramSource_MaskGapScore_QC",
    "syllectogramSource_MaskAggIndex",
    "syllectogramSource_MaskAggIndex_QC",
    "objectSyllectogramSource_intensity",
    "objectSyllectogramSource",
    "objectSyllectogramSource_structure",
    "objectSyllectogramSource_structure_QC",
    "objectSyllectogramSource_aggregation",
    "objectSyllectogramSource_aggregation_QC",
    "opticalMean_raw",
    "opticalMean_corrected",
    "opticalMean_raw_normMinRise",
    "opticalMean_corrected_normMinRise",
    "opticalMean_normMinRise",
    "opticalMean_raw_norm01",
    "opticalMean_corrected_norm01",
    "opticalMean_norm01",
    "opticalMeanIntensity_raw",
    "opticalMeanIntensity_corrected",
    "opticalMeanIntensity_norm_minRise",
    "opticalMeanIntensity_norm_0_1",
    "opticalCV_raw_norm01",
    "opticalCV_corrected_norm01",
    "opticalCV_raw",
    "opticalCV_corrected",
    "opticalCV_norm_0_1",
    "opticalIlluminationGain",
    "opticalIlluminationJumpFlag",
    "objectOpticalMeanIntensity_raw",
    "objectOpticalMeanIntensity_corrected",
    "objectOpticalMeanIntensity_norm_minRise",
    "objectOpticalMeanIntensity_norm_0_1",
    "objectOpticalCV_raw",
    "objectOpticalCV_corrected",
    "objectOpticalCV_norm_0_1",
    "objectOpticalIlluminationGain",
    "objectOpticalIlluminationJumpFlag",
    "MeanI_norm_minRise",
    "MeanI_norm_0_1",
    "CV_norm_0_1",
    "MaskSASI_raw",
    "MaskSASI_0_100",
    "MaskSASI_raw_0_100",
    "MaskSASI_QC_0_100",
    "MaskSASI_syl_0_1",
    "MaskSASI_syl_raw_0_1",
    "MaskSASI_syl_QC_0_1",
    "MaskSASI_QC_syl_0_1",
    "MaskGapScore_raw",
    "MaskGapScore_0_100",
    "MaskGapScore_raw_0_100",
    "MaskGapScore_QC_0_100",
    "MaskGapScore_syl_0_1",
    "MaskGapScore_syl_raw_0_1",
    "MaskGapScore_syl_QC_0_1",
    "MaskGapScore_QC_syl_0_1",
    "MaskAggIndex_raw_0_100",
    "MaskAggIndex_QC_0_100",
    "MaskAggIndex_0_100",
    "MaskAggIndex_syl_0_1",
    "MaskAggIndex_syl_raw_0_1",
    "MaskAggIndex_syl_QC_0_1",
    "MaskAggIndex_QC_syl_0_1",
    "ObjectStructureSyl_0_1",
    "ObjectStructureSyl_raw_0_1",
    "ObjectStructureSyl_QC_0_1",
    "ObjectAggregationSyl_0_1",
    "ObjectAggregationSyl_raw_0_1",
    "ObjectAggregationSyl_QC_0_1",
    "meanI_syl_start_value",
    "meanI_syl_final_value",
    "meanI_syl_amplitude",
    "meanI_syl_AUC",
    "meanI_syl_AUC_norm",
    "meanI_syl_T50",
    "meanI_syl_T90",
    "meanI_syl_initialSlope",
    "meanI_syl_maxSlope",
    "meanI_syl_timeToMaxSlope",
    "meanI_syl_peakValue",
    "meanI_syl_timeToPeak",
    "meanI_syl_plateauValue",
    "meanI_syl_orientation",
    "meanI_syl_fit_R2",
    "meanI_syl_fit_RMSE",
    "meanI_syl_fit_success",
    "cv_syl_start_value",
    "cv_syl_final_value",
    "cv_syl_amplitude",
    "cv_syl_AUC",
    "cv_syl_AUC_norm",
    "cv_syl_T50",
    "cv_syl_T90",
    "cv_syl_initialSlope",
    "cv_syl_maxSlope",
    "cv_syl_timeToMaxSlope",
    "cv_syl_peakValue",
    "cv_syl_timeToPeak",
    "cv_syl_plateauValue",
    "cv_syl_orientation",
    "cv_syl_fit_R2",
    "cv_syl_fit_RMSE",
    "cv_syl_fit_success",
    "maskSASI_syl_start_value",
    "maskSASI_syl_final_value",
    "maskSASI_syl_amplitude",
    "maskSASI_syl_AUC",
    "maskSASI_syl_AUC_norm",
    "maskSASI_syl_T50",
    "maskSASI_syl_T90",
    "maskSASI_syl_initialSlope",
    "maskSASI_syl_maxSlope",
    "maskSASI_syl_timeToMaxSlope",
    "maskSASI_syl_peakValue",
    "maskSASI_syl_timeToPeak",
    "maskSASI_syl_plateauValue",
    "maskSASI_syl_orientation",
    "maskSASI_syl_fit_R2",
    "maskSASI_syl_fit_RMSE",
    "maskSASI_syl_fit_success",
    "maskGapScore_syl_amplitude",
    "maskGapScore_syl_AUC",
    "maskGapScore_syl_AUC_norm",
    "maskGapScore_syl_T50",
    "maskGapScore_syl_T90",
    "maskGapScore_syl_initialSlope",
    "maskGapScore_syl_maxSlope",
    "maskGapScore_syl_timeToMaxSlope",
    "maskGapScore_syl_plateauValue",
    "maskGapScore_syl_orientation",
    "maskGapScore_syl_fit_R2",
    "maskGapScore_syl_fit_RMSE",
    "maskGapScore_syl_fit_success",
    "maskAggIndex_syl_amplitude",
    "maskAggIndex_syl_AUC",
    "maskAggIndex_syl_AUC_norm",
    "maskAggIndex_syl_T50",
    "maskAggIndex_syl_T90",
    "maskAggIndex_syl_initialSlope",
    "maskAggIndex_syl_maxSlope",
    "maskAggIndex_syl_timeToMaxSlope",
    "maskAggIndex_syl_plateauValue",
    "maskAggIndex_syl_orientation",
    "maskAggIndex_syl_fit_R2",
    "maskAggIndex_syl_fit_RMSE",
    "maskAggIndex_syl_fit_success",
    "objectStructure_syl_amplitude",
    "objectStructure_syl_AUC",
    "objectStructure_syl_AUC_norm",
    "objectStructure_syl_T50",
    "objectStructure_syl_T90",
    "objectStructure_syl_initialSlope",
    "objectStructure_syl_maxSlope",
    "objectStructure_syl_timeToMaxSlope",
    "objectStructure_syl_plateauValue",
    "objectStructure_syl_orientation",
    "objectStructure_syl_fit_R2",
    "objectStructure_syl_fit_RMSE",
    "objectStructure_syl_fit_success",
    "objectAggregation_syl_amplitude",
    "objectAggregation_syl_AUC",
    "objectAggregation_syl_AUC_norm",
    "objectAggregation_syl_T50",
    "objectAggregation_syl_T90",
    "objectAggregation_syl_initialSlope",
    "objectAggregation_syl_maxSlope",
    "objectAggregation_syl_timeToMaxSlope",
    "objectAggregation_syl_plateauValue",
    "objectAggregation_syl_orientation",
    "objectAggregation_syl_fit_R2",
    "objectAggregation_syl_fit_RMSE",
    "objectAggregation_syl_fit_success",
    "meanIntensity_syl_source",
    "meanIntensity_syl_norm_minRise",
    "meanIntensity_syl_amplitude",
    "meanIntensity_syl_AUC",
    "meanIntensity_syl_T50",
    "meanIntensity_syl_initialSlope",
    "meanIntensity_syl_maxSlope",
    "meanIntensity_syl_finalValue",
    "meanIntensity_syl_fit_R2",
    "meanIntensity_syl_fit_RMSE",
    "meanIntensity_syl_fit_success",
    "meanIntensity_syl_orientation",
    "meanIntensity_syl_points",
    "SASI_syl_source",
    "SASI_syl_norm01",
    "SASI_syl_amplitude",
    "SASI_syl_AUC",
    "SASI_syl_T50",
    "SASI_syl_initialSlope",
    "SASI_syl_maxSlope",
    "SASI_syl_finalValue",
    "SASI_syl_fit_R2",
    "SASI_syl_fit_RMSE",
    "SASI_syl_fit_success",
    "SASI_syl_orientation",
    "SASI_syl_points",
    "MaskSASI_syl_source",
    "MaskSASI_syl_norm01",
    "MaskSASI_syl_amplitude",
    "MaskSASI_syl_AUC",
    "MaskSASI_syl_T50",
    "MaskSASI_syl_initialSlope",
    "MaskSASI_syl_maxSlope",
    "MaskSASI_syl_finalValue",
    "MaskSASI_syl_fit_R2",
    "MaskSASI_syl_fit_RMSE",
    "MaskSASI_syl_fit_success",
    "MaskSASI_syl_orientation",
    "MaskSASI_syl_points",
    "MaskSASI_noPOD_syl_norm01",
    "MaskSASI_noPOD_syl_amplitude",
    "MaskSASI_noPOD_syl_AUC",
    "MaskSASI_noPOD_syl_T50",
    "MaskSASI_noPOD_syl_initialSlope",
    "MaskSASI_noPOD_syl_maxSlope",
    "MaskSASI_noPOD_syl_finalValue",
    "MaskSASI_noPOD_syl_fit_R2",
    "MaskSASI_noPOD_syl_fit_RMSE",
    "MaskSASI_noPOD_syl_fit_success",
    "MaskSASI_noPOD_syl_orientation",
    "MaskSASI_noPOD_syl_points",
    "MaskSASI_PODremoved_syl_norm01",
    "MaskSASI_PODremoved_syl_amplitude",
    "MaskSASI_PODremoved_syl_AUC",
    "MaskSASI_PODremoved_syl_T50",
    "MaskSASI_PODremoved_syl_initialSlope",
    "MaskSASI_PODremoved_syl_maxSlope",
    "MaskSASI_PODremoved_syl_finalValue",
    "MaskSASI_PODremoved_syl_fit_R2",
    "MaskSASI_PODremoved_syl_fit_RMSE",
    "MaskSASI_PODremoved_syl_fit_success",
    "MaskSASI_PODremoved_syl_orientation",
    "MaskSASI_PODremoved_syl_points",
    "ObjectStructure_syl_source",
    "ObjectStructure_syl_norm01",
    "ObjectStructure_syl_amplitude",
    "ObjectStructure_syl_AUC",
    "ObjectStructure_syl_T50",
    "ObjectStructure_syl_initialSlope",
    "ObjectStructure_syl_maxSlope",
    "ObjectStructure_syl_finalValue",
    "ObjectStructure_syl_fit_R2",
    "ObjectStructure_syl_fit_RMSE",
    "ObjectStructure_syl_fit_success",
    "ObjectStructure_syl_orientation",
    "ObjectStructure_syl_points",
    "ObjectStructure_noPOD_syl_norm01",
    "ObjectStructure_noPOD_syl_amplitude",
    "ObjectStructure_noPOD_syl_AUC",
    "ObjectStructure_noPOD_syl_T50",
    "ObjectStructure_noPOD_syl_initialSlope",
    "ObjectStructure_noPOD_syl_maxSlope",
    "ObjectStructure_noPOD_syl_finalValue",
    "ObjectStructure_noPOD_syl_fit_R2",
    "ObjectStructure_noPOD_syl_fit_RMSE",
    "ObjectStructure_noPOD_syl_fit_success",
    "ObjectStructure_noPOD_syl_orientation",
    "ObjectStructure_noPOD_syl_points",
    "ObjectStructure_PODremoved_syl_norm01",
    "ObjectStructure_PODremoved_syl_amplitude",
    "ObjectStructure_PODremoved_syl_AUC",
    "ObjectStructure_PODremoved_syl_T50",
    "ObjectStructure_PODremoved_syl_initialSlope",
    "ObjectStructure_PODremoved_syl_maxSlope",
    "ObjectStructure_PODremoved_syl_finalValue",
    "ObjectStructure_PODremoved_syl_fit_R2",
    "ObjectStructure_PODremoved_syl_fit_RMSE",
    "ObjectStructure_PODremoved_syl_fit_success",
    "ObjectStructure_PODremoved_syl_orientation",
    "ObjectStructure_PODremoved_syl_points",
    "gSASI_syl_source",
    "gSASI_syl_norm01",
    "gSASI_syl_amplitude",
    "gSASI_syl_AUC",
    "gSASI_syl_T50",
    "gSASI_syl_initialSlope",
    "gSASI_syl_maxSlope",
    "gSASI_syl_finalValue",
    "gSASI_syl_fit_R2",
    "gSASI_syl_fit_RMSE",
    "gSASI_syl_fit_success",
    "gSASI_syl_orientation",
    "gSASI_syl_points",
]

for _syl_prefix in [
    "meanI_syl",
    "meanI_syl_raw",
    "meanI_syl_corrected",
    "cv_syl",
    "cv_syl_raw",
    "cv_syl_corrected",
    "maskSASI_syl",
    "maskSASI_syl_raw",
    "maskSASI_syl_QC",
    "maskGapScore_syl",
    "maskGapScore_syl_raw",
    "maskGapScore_syl_QC",
    "maskAggIndex_syl",
    "maskAggIndex_syl_raw",
    "maskAggIndex_syl_QC",
    "objectStructure_syl",
    "objectStructure_syl_raw",
    "objectStructure_syl_QC",
    "objectAggregation_syl",
    "objectAggregation_syl_raw",
    "objectAggregation_syl_QC",
]:
    for _syl_suffix in [
        "start_value",
        "final_value",
        "finalValue",
        "amplitude",
        "AUC",
        "AUC_norm",
        "T50",
        "T90",
        "initialSlope",
        "maxSlope",
        "timeToMaxSlope",
        "peakValue",
        "timeToPeak",
        "plateauValue",
        "orientation",
        "flatCurveFlag",
        "points",
        "nFiniteValues",
        "nTrustedValues",
        "fracTrustedValues",
        "insufficientTrustedValuesFlag",
        "fit_R2",
        "fit_RMSE",
        "fit_success",
        "fitR2",
        "fitRMSE",
        "fitSuccess",
    ]:
        _syl_col = f"{_syl_prefix}_{_syl_suffix}"
        if _syl_col not in NORMALIZED_SYLLECTOGRAM_METRICS:
            NORMALIZED_SYLLECTOGRAM_METRICS.append(_syl_col)

RBC_AGG_APPROX_METRICS = [
    "aggOccFrac",
    "aggNcomp",
    "aggLargestArea",
    "aggLargestFrac",
    "aggSkelLen",
    "aggBranchPts",
    "aggEndPts",
]


def _safe_float(value: Any) -> float:
    """Convert a scalar to finite float or NaN."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _empty_metric_row(filename: str, mode: str, error: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"filename": filename, "mode": mode}
    for col in CORE_METRICS:
        row[col] = np.nan
    if error:
        row["warning"] = error
    return row


def correlation_length_px(mask: np.ndarray, max_dim: int = 384) -> float:
    """Estimate a radial autocorrelation length in pixels.

    The binary mask is mean-centered, autocorrelated with FFT convolution,
    radially averaged, and the first crossing below 1/e is reported. This is a
    practical correlation-length proxy for structural texture, not a calibrated
    physical length scale.
    """
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0 or np.nanstd(arr) <= 1e-8:
        return np.nan

    scale_back = 1.0
    max_side = max(arr.shape)
    if max_side > max_dim:
        scale = max_dim / float(max_side)
        new_shape = tuple(max(8, int(round(v * scale))) for v in arr.shape)
        arr = transform.resize(
            arr,
            new_shape,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)
        scale_back = 1.0 / scale

    arr = arr - float(np.mean(arr))
    if float(np.var(arr)) <= 1e-10:
        return np.nan

    corr = signal.fftconvolve(arr, arr[::-1, ::-1], mode="full")
    cy, cx = np.array(corr.shape) // 2
    center = corr[cy, cx]
    if abs(center) <= 1e-12:
        return np.nan
    corr = corr / center

    yy, xx = np.indices(corr.shape)
    radii = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    sums = np.bincount(radii.ravel(), corr.ravel())
    counts = np.bincount(radii.ravel())
    profile = sums / np.maximum(counts, 1)

    if profile.size < 3:
        return np.nan
    target = np.exp(-1)
    below = np.where(profile[1:] <= target)[0]
    if below.size == 0:
        return np.nan

    idx = int(below[0] + 1)
    if idx == 1:
        xi = 1.0
    else:
        y0, y1 = profile[idx - 1], profile[idx]
        if abs(y1 - y0) <= 1e-12:
            xi = float(idx)
        else:
            xi = float(idx - 1 + (target - y0) / (y1 - y0))
    return _safe_float(xi * scale_back)


def compute_xi_from_binary_mask(
    mask: np.ndarray,
    xiThr: float = float(np.exp(-1.0)),
    minAreaFracForXi: float = 0.03,
    minObjectsForXi: int = 10,
    maxAreaFracForXi: float = 0.97,
    rejectUniform: bool = True,
) -> float:
    """Estimate Xi from the selected binary gap mask, not the continuous image.

    This is the trusted RBC morphology Xi candidate. When ``rejectUniform`` is
    True it rejects nearly empty (``area_frac < minAreaFracForXi``), nearly full
    (``area_frac > maxAreaFracForXi``), too-few-object, or near-constant masks so
    early visually uniform frames cannot receive a long trusted Xi simply because
    the continuous corrected image has smooth illumination/background correlation.
    Rejected frames return NaN, which keeps SASI/AggIndex from being inflated by
    structureless frames. Set ``rejectUniform=False`` to disable all guards.
    """
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or arr.size == 0:
        return np.nan
    area_frac = float(np.mean(arr))
    if not np.isfinite(area_frac):
        return np.nan
    if rejectUniform:
        if area_frac < float(minAreaFracForXi) or area_frac > float(maxAreaFracForXi):
            return np.nan
        labels, n_objects = ndi.label(arr, structure=np.ones((3, 3), dtype=np.uint8))
        if int(n_objects) < int(minObjectsForXi):
            return np.nan

    values = arr.astype(np.float64)
    if np.nanstd(values) <= 1e-12:
        return np.nan
    values = values - float(np.nanmean(values))
    if np.nanstd(values) <= 1e-12:
        return np.nan

    try:
        ac = np.real(np.fft.ifft2(np.abs(np.fft.fft2(values)) ** 2))
        ac = np.fft.fftshift(ac)
        peak = float(np.nanmax(ac))
        if not np.isfinite(peak) or peak <= 0:
            return np.nan
        ac = ac / peak
        center_y = ac.shape[0] // 2
        center_x = ac.shape[1] // 2
        curve = np.asarray(ac[center_y, center_x:], dtype=np.float64)
        if curve.size < 3:
            return np.nan
        finite = np.isfinite(curve)
        if not finite.any():
            return np.nan
        fill_value = float(np.nanmedian(curve[finite]))
        curve = np.nan_to_num(curve, nan=fill_value, posinf=0.0, neginf=0.0)
        below = np.flatnonzero(curve <= float(xiThr))
        below = below[below > 0]
        if below.size == 0:
            return np.nan
        idx = int(below[0])
        y0 = float(curve[idx - 1])
        y1 = float(curve[idx])
        if not np.isfinite(y0) or not np.isfinite(y1) or abs(y1 - y0) <= 1e-12:
            return _safe_float(float(idx))
        frac = float(np.clip((float(xiThr) - y0) / (y1 - y0), 0.0, 1.0))
        return _safe_float(float((idx - 1) + frac))
    except Exception:
        return np.nan


def lacunarity(mask: np.ndarray, box_size: int = 16) -> float:
    """Compute box-count lacunarity, a measure of structural heterogeneity."""
    arr = np.asarray(mask, dtype=np.float32)
    h, w = arr.shape
    if h < box_size or w < box_size:
        return np.nan
    h2 = (h // box_size) * box_size
    w2 = (w // box_size) * box_size
    boxes = arr[:h2, :w2].reshape(h2 // box_size, box_size, w2 // box_size, box_size)
    counts = boxes.sum(axis=(1, 3)).ravel()
    mean = float(np.mean(counts))
    if mean <= 1e-12:
        return np.nan
    return _safe_float(float(np.var(counts) / (mean * mean) + 1.0))


def skeleton_counts(mask: np.ndarray) -> tuple[int, int, int]:
    """Return skeleton length, branch-point count, and endpoint count."""
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0 or not np.any(arr):
        return 0, 0, 0
    skel = morphology.skeletonize(arr)
    kernel = np.ones((3, 3), dtype=np.int16)
    neighbors = ndi.convolve(skel.astype(np.int16), kernel, mode="constant") - skel
    endpoints = skel & (neighbors == 1)
    branches = skel & (neighbors >= 3)
    return int(np.sum(skel)), int(np.sum(branches)), int(np.sum(endpoints))


def skeleton_metrics(mask: np.ndarray) -> tuple[float, float, float]:
    """Return skeleton, branch-point, and endpoint densities per image pixel."""
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    skel_len, branch_pts, end_pts = skeleton_counts(arr)
    total = float(arr.size)
    return (
        _safe_float(skel_len / total),
        _safe_float(branch_pts / total),
        _safe_float(end_pts / total),
    )


def _component_areas(mask: np.ndarray) -> np.ndarray:
    """Return connected-component areas for a binary mask."""
    labels = measure.label(np.asarray(mask, dtype=bool))
    return np.asarray([region.area for region in measure.regionprops(labels)], dtype=np.float64)


def _resize_continuous_to_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a continuous image field without binarizing it."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.shape == shape:
        return arr
    return transform.resize(
        arr,
        shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float64)


def safe_numeric_series(df: pd.DataFrame, colname: str) -> pd.Series:
    """Return a numeric Series for ``colname``, creating NaNs if missing."""
    if colname not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[colname], errors="coerce").replace([np.inf, -np.inf], np.nan)


def robust_z(series: pd.Series | np.ndarray | list[float]) -> pd.Series:
    """Return a NaN-safe robust z-score based on median and MAD.

    Missing or non-finite entries receive a neutral z-score of 0 so that
    RBC aggregation scores remain computable when one auxiliary metric fails.
    If the MAD is too small, this falls back to a standard z-score; if the
    standard deviation is also invalid, all finite entries receive 0.
    """
    if isinstance(series, pd.Series):
        index = series.index
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    else:
        values = pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)
        index = pd.RangeIndex(len(values))

    z = np.zeros_like(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return pd.Series(z, index=index, dtype="float64")

    finite_values = values[finite]
    center = float(np.nanmedian(finite_values))
    mad = float(np.nanmedian(np.abs(finite_values - center)))
    denom = 1.4826 * mad
    if not np.isfinite(denom) or denom <= 1e-12:
        denom = float(np.nanstd(finite_values))
    if np.isfinite(denom) and denom > 1e-12:
        z[finite] = (finite_values - center) / denom
    return pd.Series(z, index=index, dtype="float64")


def matlab_robust_z(series: pd.Series | np.ndarray | list[float], reference_mask: pd.Series | np.ndarray | None = None) -> pd.Series:
    """Return MATLAB legacy ``robz``: (x - median(x(idx))) / (mad(x(idx),1)+eps).

    MATLAB's RBC reference code uses the raw median absolute deviation, not
    the statistically scaled ``1.4826*MAD`` denominator used by ``robust_z``.
    Keep this helper RBC-only so generic Object Mode scores retain their
    existing scaling.
    """
    if isinstance(series, pd.Series):
        index = series.index
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    else:
        values = pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)
        index = pd.RangeIndex(len(values))

    if reference_mask is None:
        ref = np.ones_like(values, dtype=bool)
    else:
        ref = np.asarray(reference_mask, dtype=bool)
        if ref.size != values.size:
            ref = np.ones_like(values, dtype=bool)

    finite_ref = np.isfinite(values) & ref
    if not finite_ref.any():
        finite_ref = np.isfinite(values)
    z = np.zeros_like(values, dtype=float)
    if not finite_ref.any():
        return pd.Series(z, index=index, dtype="float64")

    ref_values = values[finite_ref]
    center = float(np.nanmedian(ref_values))
    mad = float(np.nanmedian(np.abs(ref_values - center)))
    denom = mad + np.finfo(float).eps
    finite = np.isfinite(values)
    z[finite] = (values[finite] - center) / denom
    return pd.Series(z, index=index, dtype="float64")


def sigmoid100(x: pd.Series | np.ndarray | float) -> pd.Series | float:
    """Map a raw score to 0-100 with a numerically stable sigmoid."""
    if isinstance(x, pd.Series):
        values = pd.to_numeric(x, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        scaled = 100.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))
        return pd.Series(scaled, index=x.index, dtype="float64")
    values = np.asarray(x, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=50.0, neginf=-50.0)
    scaled = 100.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))
    if np.isscalar(x):
        return float(scaled)
    return pd.Series(scaled.ravel(), dtype="float64")


def compute_gap_area_slope(mask: np.ndarray, min_area: int = 4, min_components: int = 5) -> float:
    """Fit the log-log plasma-gap component area distribution slope.

    The slope summarizes how connected plasma-gap object counts scale with
    object area. Negative values are expected when small gaps dominate.
    """
    areas = _component_areas(mask)
    areas = areas[np.isfinite(areas) & (areas >= float(min_area))]
    if areas.size < min_components or np.nanmax(areas) <= np.nanmin(areas):
        return np.nan

    n_bins = int(np.clip(np.sqrt(areas.size), 4, 12))
    bin_edges = np.logspace(np.log10(float(np.min(areas))), np.log10(float(np.max(areas))), n_bins + 1)
    counts, edges = np.histogram(areas, bins=bin_edges)
    nonzero = counts > 0
    if np.count_nonzero(nonzero) < 3:
        return np.nan

    centers = np.sqrt(edges[:-1] * edges[1:])
    x = np.log10(centers[nonzero])
    y = np.log10(counts[nonzero].astype(float))
    if x.size < 3 or np.nanstd(x) <= 1e-12:
        return np.nan
    try:
        slope, _intercept = np.polyfit(x, y, 1)
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return np.nan
    return _safe_float(slope)


def _component_areas_8_connected(mask: np.ndarray) -> np.ndarray:
    """Return 8-connected component areas, matching MATLAB ``bwconncomp``."""
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or arr.size == 0:
        return np.asarray([], dtype=np.float64)
    labels, n_labels = ndi.label(arr, structure=np.ones((3, 3), dtype=np.uint8))
    if n_labels == 0:
        return np.asarray([], dtype=np.float64)
    return np.bincount(labels.ravel())[1:].astype(np.float64)


def gap_area_slope_matlab_debug(mask: np.ndarray, n_bins: int = 12, min_nonzero_bins: int = 3) -> dict[str, Any]:
    """Return MATLAB-compatible gap-area slope details for debugging.

    Mirrors ``computeGapAreaSlopeAllFrames`` in the v14 MATLAB reference:
    ``bwconncomp`` areas, ``logspace(log10(aMin), log10(aMax), nBins+1)``,
    ``histcounts``, geometric bin centres, zero-bin removal, and ``polyfit``.
    """
    areas = _component_areas_8_connected(mask)
    areas = areas[np.isfinite(areas) & (areas > 0)]
    debug: dict[str, Any] = {
        "nComponents": int(areas.size),
        "minComponentArea": _safe_float(np.min(areas)) if areas.size else np.nan,
        "maxComponentArea": _safe_float(np.max(areas)) if areas.size else np.nan,
        "binEdges": [],
        "binCounts": [],
        "nonzeroBinCenters": [],
        "log10BinCenters": [],
        "log10Counts": [],
        "gapAreaSlope": np.nan,
        "gapAreaSlope_R2": np.nan,
        "gapAreaSlope_nBins": 0,
    }
    if areas.size < 3:
        return debug

    area_min = float(np.min(areas))
    area_max = float(np.max(areas))
    if area_min <= 0 or area_max <= area_min:
        return debug

    try:
        edges = np.logspace(np.log10(area_min), np.log10(area_max), int(n_bins) + 1)
        counts, bin_edges = np.histogram(areas, bins=edges)
    except (FloatingPointError, ValueError):
        return debug

    centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    keep = counts > 0
    x = np.log10(centers[keep])
    y = np.log10(counts[keep].astype(float))
    debug.update(
        {
            "binEdges": bin_edges.astype(float).tolist(),
            "binCounts": counts.astype(int).tolist(),
            "nonzeroBinCenters": centers[keep].astype(float).tolist(),
            "log10BinCenters": x.astype(float).tolist(),
            "log10Counts": y.astype(float).tolist(),
            "gapAreaSlope_nBins": int(x.size),
        }
    )
    if x.size < int(min_nonzero_bins):
        return debug

    try:
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + np.finfo(float).eps
        r2 = 1.0 - ss_res / ss_tot
        r2 = max(0.0, min(1.0, r2))
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return debug

    debug["gapAreaSlope"] = _safe_float(slope)
    debug["gapAreaSlope_R2"] = _safe_float(r2)
    return debug


def compute_gap_area_slope_matlab_like(mask: np.ndarray, n_bins: int = 12, min_nonzero_bins: int = 3) -> tuple[float, float, int]:
    """Fit MATLAB-style log-log plasma-gap component area slope."""
    debug = gap_area_slope_matlab_debug(mask, n_bins=n_bins, min_nonzero_bins=min_nonzero_bins)
    return (
        _safe_float(debug.get("gapAreaSlope", np.nan)),
        _safe_float(debug.get("gapAreaSlope_R2", np.nan)),
        int(debug.get("gapAreaSlope_nBins", 0) or 0),
    )


def compute_lf_hf_ratio(image_or_mask: np.ndarray) -> float:
    """Compute a simple low-frequency/high-frequency spectral ratio.

    Milestone 3 uses the final binary plasma-gap mask as the signal. The
    returned value is log10(LF_power / HF_power), with small epsilons for
    numerical stability.
    """
    arr = np.asarray(image_or_mask, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return np.nan
    arr = arr - float(np.nanmean(arr))
    if not np.isfinite(arr).all() or float(np.nanstd(arr)) <= 1e-12:
        return np.nan

    try:
        power = np.abs(np.fft.fftshift(np.fft.fft2(arr))) ** 2
        h, w = arr.shape
        fy = np.fft.fftshift(np.fft.fftfreq(h))
        fx = np.fft.fftshift(np.fft.fftfreq(w))
        yy, xx = np.meshgrid(fy, fx, indexing="ij")
        radii = np.sqrt(xx**2 + yy**2)

        low_band = (radii > 0.0) & (radii <= 0.10)
        high_band = (radii >= 0.18) & (radii <= 0.50)
        if not low_band.any() or not high_band.any():
            return np.nan
        lf_power = float(np.sum(power[low_band]))
        hf_power = float(np.sum(power[high_band]))
        eps = np.finfo(float).eps
        return _safe_float(np.log10((lf_power + eps) / (hf_power + eps)))
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return np.nan


def compute_lf_hf_matlab_like(ximg01: np.ndarray) -> float:
    """Compute MATLAB-compatible low/high frequency power ratio.

    The legacy MATLAB code sums FFT power within normalized radial frequency
    bands. We use sum power here, not mean power, to preserve that behaviour.
    """
    arr = np.asarray(ximg01, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return np.nan
    arr = arr - float(np.nanmean(arr))
    if not np.isfinite(arr).all() or float(np.nanstd(arr)) <= 1e-12:
        return np.nan

    try:
        power = np.abs(np.fft.fftshift(np.fft.fft2(arr))) ** 2
        h, w = arr.shape
        fy = np.fft.fftshift(np.fft.fftfreq(h))
        fx = np.fft.fftshift(np.fft.fftfreq(w))
        yy, xx = np.meshgrid(fy, fx, indexing="ij")
        radii = np.sqrt(xx**2 + yy**2)
        low_band = radii <= 0.03
        high_band = (radii > 0.03) & (radii <= 0.18)
        if not low_band.any() or not high_band.any():
            return np.nan
        lf_power = float(np.sum(power[low_band]))
        hf_power = float(np.sum(power[high_band]))
        eps = np.finfo(float).eps
        return _safe_float(np.log10((lf_power + eps) / (hf_power + eps)))
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return np.nan


def compute_xi_interp_matlab_like(ximg01: np.ndarray) -> float:
    """Estimate MATLAB-style Xi_interp_px from a continuous residual field.

    This reproduces the legacy horizontal autocorrelation crossing calculation
    more closely than the MVP binary-mask correlation length. The input should
    be a continuous RBC preprocessing field, such as a display-scaled Wz or
    residual image, not the final binary plasma-gap mask.
    """
    arr = np.asarray(ximg01, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return np.nan
    arr = arr - float(np.nanmean(arr))
    if not np.isfinite(arr).all() or float(np.nanstd(arr)) <= 1e-12:
        return np.nan

    try:
        corr = np.real(np.fft.ifft2(np.abs(np.fft.fft2(arr)) ** 2))
        corr = np.fft.fftshift(corr)
        max_corr = float(np.nanmax(corr))
        if not np.isfinite(max_corr) or abs(max_corr) <= 1e-12:
            return np.nan
        corr = corr / max_corr

        cy, cx = np.array(corr.shape) // 2
        line = np.asarray(corr[cy, cx:], dtype=np.float64)
        if line.size < 2:
            return np.nan
        target = float(np.exp(-1.0))
        crossings = np.where(line <= target)[0]
        if crossings.size == 0:
            return np.nan

        idx = int(crossings[0])
        if idx <= 0:
            return 0.0
        y0 = float(line[idx - 1])
        y1 = float(line[idx])
        if not np.isfinite(y0) or not np.isfinite(y1):
            return np.nan
        if abs(y1 - y0) <= 1e-12:
            xi = float(idx)
        else:
            xi = float((idx - 1) + (target - y0) / (y1 - y0))
        return _safe_float(xi)
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return np.nan


def compute_shi_tile_cv(mask: np.ndarray, tile_grid: tuple[int, int] = (4, 4)) -> float:
    """Compute spatial heterogeneity as CV of tile-wise gap area fractions.

    MATLAB's ``std`` defaults to sample standard deviation, so the RBC score
    path uses ``ddof=1`` here.
    """
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or arr.size == 0:
        return np.nan

    n_rows, n_cols = tile_grid
    y_edges = np.linspace(0, arr.shape[0], n_rows + 1, dtype=int)
    x_edges = np.linspace(0, arr.shape[1], n_cols + 1, dtype=int)
    fractions: list[float] = []
    for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
        for x0, x1 in zip(x_edges[:-1], x_edges[1:]):
            tile = arr[y0:y1, x0:x1]
            if tile.size:
                fractions.append(float(np.mean(tile)))
    if not fractions:
        return np.nan

    values = np.asarray(fractions, dtype=float)
    mean_value = float(np.mean(values))
    if mean_value <= 1e-12:
        return np.nan
    ddof = 1 if values.size > 1 else 0
    return _safe_float(float(np.std(values, ddof=ddof) / mean_value))


def structure_tensor_anisotropy(
    field: np.ndarray,
    sigma_grad: float = 1.5,
    sigma_tensor: float = 8.0,
) -> dict[str, float]:
    """Structure-tensor anisotropy / orientational order of a grayscale field.

    Captures directional structure (e.g. rouleaux chains) that rotation-blind
    metrics (Xi, diameter, lacunarity) cannot see. Operates on a continuous
    field, ideally the difference-of-Gaussians aggregation field.

    Returns
    -------
    anisotropy : mean local coherence (lam1-lam2)/(lam1+lam2) in [0,1].
    orderParam : anisotropy-weighted global orientational order in [0,1].
    orientation_deg : dominant orientation in degrees.
    """
    field = np.asarray(field, dtype=np.float64)
    if field.size == 0 or not np.any(np.isfinite(field)):
        return {"anisotropy": np.nan, "orderParam": np.nan, "orientation_deg": np.nan}
    smoothed = ndi.gaussian_filter(field, sigma_grad)
    gy, gx = np.gradient(smoothed)
    jxx = ndi.gaussian_filter(gx * gx, sigma_tensor)
    jyy = ndi.gaussian_filter(gy * gy, sigma_tensor)
    jxy = ndi.gaussian_filter(gx * gy, sigma_tensor)

    tmp = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
    lam1 = 0.5 * (jxx + jyy + tmp)
    lam2 = 0.5 * (jxx + jyy - tmp)
    eps = np.finfo(np.float64).eps
    coherence = (lam1 - lam2) / (lam1 + lam2 + eps)

    theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    weight = lam1 - lam2
    wsum = float(np.nansum(weight)) + eps
    cos_term = float(np.nansum(weight * np.cos(2.0 * theta)))
    sin_term = float(np.nansum(weight * np.sin(2.0 * theta)))
    return {
        "anisotropy": float(np.nanmean(coherence)),
        "orderParam": float(np.hypot(cos_term, sin_term) / wsum),
        "orientation_deg": float(np.degrees(0.5 * np.arctan2(sin_term, cos_term))),
    }


def interface_density(mask: np.ndarray) -> float:
    """Two-phase boundary length per image area (px^-1), holes included.

    Robust two-phase microstructure descriptor; unlike object count it does not
    depend on small-speck survival. Implemented as a bwperim equivalent.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return np.nan
    eroded = ndi.binary_erosion(mask, border_value=0)
    perim = mask & ~eroded
    return float(np.count_nonzero(perim)) / float(mask.size)


def percolation_metrics(phase_mask: np.ndarray) -> dict[str, Any]:
    """Connectivity / spanning of a binary phase.

    Flags whether any single connected component spans the field
    (touches both left+right OR top+bottom borders) and reports the
    largest-cluster fraction as a continuous connectivity order parameter.
    """
    phase_mask = np.asarray(phase_mask, dtype=bool)
    out: dict[str, Any] = {
        "percolates": False,
        "spansLR": False,
        "spansTB": False,
        "largestClusterFrac_image": 0.0,
        "largestClusterFrac_phase": 0.0,
    }
    if phase_mask.size == 0 or not np.any(phase_mask):
        return out

    # Vectorized connected components (8-connectivity). Avoids regionprops/coords,
    # which is ~100x slower on busy masks.
    structure = np.ones((3, 3), dtype=bool)
    labels, n_labels = ndi.label(phase_mask, structure=structure)
    if n_labels == 0:
        return out

    # Component sizes via histogram (index 0 is background).
    counts = np.bincount(labels.ravel())
    sizes = counts[1:]
    total = float(sizes.sum())
    if sizes.size:
        largest = float(sizes.max())
        out["largestClusterFrac_image"] = largest / float(phase_mask.size)
        out["largestClusterFrac_phase"] = largest / total if total > 0 else 0.0

    # Spanning: a label that appears on both opposite borders spans the field.
    left = np.unique(labels[:, 0])
    right = np.unique(labels[:, -1])
    top = np.unique(labels[0, :])
    bottom = np.unique(labels[-1, :])
    lr = np.intersect1d(left, right, assume_unique=True)
    tb = np.intersect1d(top, bottom, assume_unique=True)
    spans_lr = bool(np.any(lr > 0))
    spans_tb = bool(np.any(tb > 0))
    out["spansLR"] = spans_lr
    out["spansTB"] = spans_tb
    out["percolates"] = bool(spans_lr or spans_tb)
    return out


def compute_core_metrics(
    image: np.ndarray,
    mask: np.ndarray,
    filename: str = "",
    mode: str = "",
) -> dict[str, Any]:
    """Compute interpretable structural metrics from an image and binary mask."""
    try:
        gray = to_grayscale(image)
        mask = resize_mask_to_shape(mask, gray.shape)
        mask = np.asarray(mask, dtype=bool)
    except Exception as exc:
        return _empty_metric_row(filename, mode, str(exc))

    labels = measure.label(mask)
    regions = measure.regionprops(labels)
    areas = np.asarray([region.area for region in regions], dtype=np.float64)
    diameters = np.asarray(
        [
            region.equivalent_diameter_area
            if hasattr(region, "equivalent_diameter_area")
            else region.equivalent_diameter
            for region in regions
        ],
        dtype=np.float64,
    )
    image_area = float(mask.size) if mask.size else np.nan

    skel_density, branch_density, endpoint_density = skeleton_metrics(mask)

    # --- extra structural biomarkers (directionality, interface, connectivity) ---
    try:
        gray_f = gray.astype(np.float64)
        # Anisotropy/orientation are large-scale (sigma 8-20) measures, so compute
        # them on a downsampled copy for speed; ratios/orientations are ~scale-free.
        max_dim = 220
        big = max(gray_f.shape)
        step = max(1, int(round(big / float(max_dim)))) if big > max_dim else 1
        if step > 1:
            small = gray_f[::step, ::step]
            inv = 1.0 / float(step)
            s1, s2 = 1.5 * inv, 20.0 * inv
            sg, st_ = 1.5 * inv, 8.0 * inv
        else:
            small = gray_f
            s1, s2 = 1.5, 20.0
            sg, st_ = 1.5, 8.0
        agg_field = ndi.gaussian_filter(small, s1) - ndi.gaussian_filter(small, s2)
        aniso = structure_tensor_anisotropy(agg_field, sigma_grad=sg, sigma_tensor=st_)
    except Exception:
        aniso = {"anisotropy": np.nan, "orderParam": np.nan, "orientation_deg": np.nan}
    try:
        ifd = interface_density(mask)
    except Exception:
        ifd = np.nan
    try:
        # Percolation tested on the structure phase (foreground mask) itself.
        perc = percolation_metrics(mask)
    except Exception:
        perc = {"percolates": False, "largestClusterFrac_phase": 0.0, "largestClusterFrac_image": 0.0}
    try:
        # Percolation of the complement (background/plasma) phase.
        perc_comp = percolation_metrics(~mask)
    except Exception:
        perc_comp = {"percolates": False, "largestClusterFrac_phase": 0.0, "largestClusterFrac_image": 0.0}

    row: dict[str, Any] = {
        "filename": filename,
        "mode": mode,
        "areaFrac": _safe_float(np.mean(mask)),
        "nObjects": int(len(regions)),
        "meanArea": _safe_float(np.mean(areas)) if areas.size else 0.0,
        "medianArea": _safe_float(np.median(areas)) if areas.size else 0.0,
        "meanDiam": _safe_float(np.mean(diameters)) if diameters.size else 0.0,
        "medianDiam": _safe_float(np.median(diameters)) if diameters.size else 0.0,
        "largestObjectFrac": _safe_float(np.max(areas) / image_area) if areas.size else 0.0,
        "Xi_px": correlation_length_px(mask),
        "lacunarity16": lacunarity(mask, box_size=16),
        "skelDensity": skel_density,
        "branchDensity": branch_density,
        "endpointDensity": endpoint_density,
        "anisotropy": _safe_float(aniso.get("anisotropy", np.nan)),
        "orientOrderParam": _safe_float(aniso.get("orderParam", np.nan)),
        "orientation_deg": _safe_float(aniso.get("orientation_deg", np.nan)),
        "interfaceDensity_px": _safe_float(ifd),
        "percolates": bool(perc.get("percolates", False)),
        "largestClusterFrac_phase": _safe_float(perc.get("largestClusterFrac_phase", np.nan)),
        "percolates_complement": bool(perc_comp.get("percolates", False)),
        "largestClusterFrac_complement": _safe_float(perc_comp.get("largestClusterFrac_phase", np.nan)),
        "intensityMean": _safe_float(np.mean(gray)),
        "intensityStd": _safe_float(np.std(gray)),
    }
    return row


def compute_object_mode_metrics(
    image: np.ndarray,
    mask: np.ndarray,
    filename: str = "",
    mode: str = "Object Mode",
    enhanced: bool = False,
) -> dict[str, Any]:
    """Compute generic Object Mode morphology metrics without RBC semantics."""
    row = compute_core_metrics(image, mask, filename=filename, mode=mode)
    slope, slope_r2, slope_bins = compute_gap_area_slope_matlab_like(mask)
    median_diameter = _safe_float(row.get("medianDiam", np.nan))
    if not np.isfinite(median_diameter) or median_diameter < 10.0:
        low_resolution_risk = "high"
    elif median_diameter <= 20.0:
        low_resolution_risk = "moderate"
    else:
        low_resolution_risk = "lower"

    area_fraction = _safe_float(row.get("areaFrac", np.nan))
    object_count = _safe_float(row.get("nObjects", np.nan))
    if not np.isfinite(area_fraction) or not np.isfinite(object_count) or object_count <= 0 or area_fraction < 0.001 or area_fraction > 0.90:
        segmentation_confidence = "low"
    elif low_resolution_risk == "high" or area_fraction < 0.01 or area_fraction > 0.80:
        segmentation_confidence = "moderate"
    else:
        segmentation_confidence = "higher"

    row.update(
        {
            "enhancedObjectModeEnabled": bool(enhanced),
            "objectCount": row.get("nObjects", np.nan),
            "meanDiam_px": row.get("meanDiam", np.nan),
            "medianDiam_px": row.get("medianDiam", np.nan),
            "correlationLength_px": row.get("Xi_px", np.nan),
            "sizeDistributionSlope": _safe_float(slope),
            "sizeDistributionSlope_R2": _safe_float(slope_r2),
            "sizeDistributionSlope_nBins": int(slope_bins),
            "textureHeterogeneity_tileCV": compute_shi_tile_cv(mask),
            "LF_log10_LFoverHF_generic": compute_lf_hf_ratio(mask),
            "skeletonDensity": row.get("skelDensity", np.nan),
            "lowResolutionRisk": low_resolution_risk,
            "lowResolutionWarning": (
                "Low-resolution warning: results describe coarse structure only. "
                "Fine object count, boundary shape, and small-object size distribution may be unreliable."
                if low_resolution_risk in {"high", "moderate"}
                else ""
            ),
            "segmentationConfidence": segmentation_confidence,
        }
    )
    return row


def compute_rbc_gap_metrics(
    image: np.ndarray,
    gap_mask: np.ndarray,
    filename: str = "",
    mode: str = "RBC Gap Mode",
    continuous_field: np.ndarray | None = None,
    lf_hf_method: str = "MATLAB-compatible continuous-field",
    minAreaFracForXi: float = 0.03,
    minObjectsForXi: int = 10,
    maxAreaFracForXi: float = 0.97,
    rejectUniformXi: bool = True,
) -> dict[str, Any]:
    """Compute RBC Gap Mode metrics from the stored bright plasma-gap mask.

    White/True pixels in ``gap_mask`` are plasma gaps. The complement metrics
    below use an approximate aggregate network inferred from the gap mask.
    Important: this complement is not true RBC segmentation.
    """
    row = compute_core_metrics(image, gap_mask, filename=filename, mode=mode)
    gray = to_grayscale(image)
    gap_mask = resize_mask_to_shape(gap_mask, gray.shape)
    gap_areas = _component_areas(gap_mask)
    image_area = float(gap_mask.size) if gap_mask.size else np.nan
    largest_gap_area = float(np.max(gap_areas)) if gap_areas.size else 0.0
    median_gap_area = float(np.median(gap_areas)) if gap_areas.size else np.nan
    gap_diams = 2.0 * np.sqrt(gap_areas / np.pi) if gap_areas.size else np.asarray([], dtype=float)
    max_gap_diam = float(np.max(gap_diams)) if gap_diams.size else np.nan
    largest_to_median_ratio = largest_gap_area / (median_gap_area + np.finfo(float).eps) if np.isfinite(median_gap_area) and median_gap_area > 0 else np.nan
    gap_area_slope_mvp = compute_gap_area_slope(gap_mask)
    gap_area_slope, gap_area_slope_r2, gap_area_slope_bins = compute_gap_area_slope_matlab_like(gap_mask)
    xi_mask = compute_xi_from_binary_mask(
        gap_mask,
        minAreaFracForXi=float(minAreaFracForXi),
        minObjectsForXi=int(minObjectsForXi),
        maxAreaFracForXi=float(maxAreaFracForXi),
        rejectUniform=bool(rejectUniformXi),
    )
    lf_hf_mvp = compute_lf_hf_ratio(gap_mask)
    if continuous_field is not None:
        continuous_field = _resize_continuous_to_shape(np.asarray(continuous_field, dtype=float), gray.shape)
        xi_interp = compute_xi_interp_matlab_like(continuous_field)
        lf_hf_matlab = compute_lf_hf_matlab_like(continuous_field)
    else:
        xi_interp = np.nan
        lf_hf_matlab = np.nan
    use_matlab_lf = str(lf_hf_method).lower().startswith("matlab")
    lf_hf_main = lf_hf_matlab if use_matlab_lf else lf_hf_mvp

    row.update(
        {
            "gapAreaFrac": row["areaFrac"],
            "gapNtotal": row["nObjects"],
            "gapMeanDiam_px": row["meanDiam"],
            "gapMedianDiam_px": row["medianDiam"],
            "gapLargestArea_px": _safe_float(largest_gap_area),
            "gapLargestFrac": _safe_float(largest_gap_area / image_area) if image_area else np.nan,
            "largestToMedianGapRatio": _safe_float(largest_to_median_ratio),
            "gapComponentAreas_px": ";".join(f"{area:g}" for area in gap_areas),
            "mask_gapAreaFrac": row["areaFrac"],
            "mask_gapNtotal": row["nObjects"],
            "mask_gapMeanArea": row["meanArea"],
            "mask_gapMeanDiam_px": row["meanDiam"],
            "mask_gapMaxDiam_px": _safe_float(max_gap_diam),
            "mask_largestGapFrac": _safe_float(largest_gap_area / image_area) if image_area else np.nan,
            "Xi_raw_intensity": _safe_float(xi_interp),
            "Xi_mask": _safe_float(xi_mask),
            "Xi_QC": np.nan,
            "xiSource": "raw_intensity_diagnostic",
            "Xi_interp_px": _safe_float(xi_interp),
            "gapAreaSlope": _safe_float(gap_area_slope),
            "gapAreaSlope_mvp": _safe_float(gap_area_slope_mvp),
            "gapAreaSlope_R2": _safe_float(gap_area_slope_r2),
            "gapAreaSlope_nBins": int(gap_area_slope_bins),
            "LF_log10_LFoverHF": _safe_float(lf_hf_main),
            "LF_log10_LFoverHF_mvp": _safe_float(lf_hf_mvp),
            "LF_log10_LFoverHF_matlab": _safe_float(lf_hf_matlab),
            "LF_log10_LFoverHF_source": "MATLAB-compatible continuous-field" if use_matlab_lf else "MVP mask-based",
            "SHI_tileCV": compute_shi_tile_cv(gap_mask),
        }
    )

    # Approximate aggregate network inferred as the complement of the plasma
    # gap mask. This is useful for structural summaries but is not true RBC
    # segmentation and should not be interpreted as individual-cell detection.
    bw_agg_approx = np.logical_not(gap_mask)
    agg_areas = _component_areas(bw_agg_approx)
    largest_agg_area = float(np.max(agg_areas)) if agg_areas.size else 0.0
    agg_pix = float(np.sum(bw_agg_approx))
    agg_skel_len, agg_branch_pts, agg_end_pts = skeleton_counts(bw_agg_approx)

    row.update(
        {
            "aggOccFrac": _safe_float(np.mean(bw_agg_approx)),
            "aggNcomp": int(len(agg_areas)),
            "aggLargestArea": _safe_float(largest_agg_area),
            "aggLargestFrac": _safe_float(largest_agg_area / agg_pix) if agg_pix > 0 else np.nan,
            "aggSkelLen": int(agg_skel_len),
            "aggBranchPts": int(agg_branch_pts),
            "aggEndPts": int(agg_end_pts),
        }
    )
    return row


def _score_xi_series(df: pd.DataFrame) -> pd.Series:
    """Return the legacy intensity-Xi series used for MATLAB comparisons."""
    if "Xi_interp_px" in df.columns:
        xi_interp = safe_numeric_series(df, "Xi_interp_px")
        if "Xi_px" in df.columns:
            return xi_interp.fillna(safe_numeric_series(df, "Xi_px"))
        return xi_interp
    return safe_numeric_series(df, "Xi_px")


def _rbc_score_reference_mask(df: pd.DataFrame) -> np.ndarray | None:
    """Return MATLAB-style score baseline rows when ``isAna`` metadata exists."""
    for column in ["isAna_like_matlab", "isAna"]:
        if column in df.columns:
            values = pd.Series(df[column]).fillna(False).astype(bool).to_numpy()
            if values.any():
                return values
    return None


def compute_sasi(df: pd.DataFrame) -> pd.DataFrame:
    """Add official V3 mask SASI and preserve the legacy intensity-Xi SASI.

    When mask Xi is unavailable, the canonical columns temporarily fall back to
    the legacy score so old standalone callers remain compatible.  The app's
    RBC pipeline supplies mask Xi and publishes the official score.
    """
    out = df.copy()
    reference_mask = _rbc_score_reference_mask(out)
    z_xi = matlab_robust_z(_score_xi_series(out), reference_mask)
    z_diam = matlab_robust_z(safe_numeric_series(out, "gapMeanDiam_px"), reference_mask)
    z_n = matlab_robust_z(safe_numeric_series(out, "gapNtotal"), reference_mask)

    legacy_raw = 0.40 * z_xi + 0.40 * z_diam - 0.20 * z_n
    out["LegacyIntensitySASI_raw"] = legacy_raw
    out["LegacyIntensitySASI_0_100"] = sigmoid100(legacy_raw)
    out["SASI_raw"] = out["LegacyIntensitySASI_raw"]
    out["SASI_0_100"] = out["LegacyIntensitySASI_0_100"]
    out["primarySASISource"] = "legacy_intensity_xi_fallback"

    if "Xi_mask" in out.columns and safe_numeric_series(out, "Xi_mask").notna().any():
        official = compute_official_mask_scores(
            out,
            xi_col="Xi_mask",
            diameter_col="gapMeanDiam_px",
            count_col="gapNtotal",
            area_col="gapAreaFrac",
            reference_mask=reference_mask,
        )
        out["SASI_raw"] = official["officialSASIRaw"]
        out["SASI_0_100"] = official["officialSASI_0_100"]
        out["scoreFormulaVersion"] = official["scoreFormulaVersion"]
        out["scoreNormalizationScope"] = official["scoreNormalizationScope"]
        out["primarySASISource"] = "final_binary_mask_xi"
    return out


def _sasi_syllectogram_axis(df: pd.DataFrame) -> tuple[pd.Series, str, str]:
    """Return the preferred time/index axis for optional SASI pseudo-syllectogram analysis."""
    for column, units in [
        ("matlabTime_s", "seconds"),
        ("time_s", "seconds"),
        ("originalImageIndex_1based", "frame_index_offset"),
        ("analysisIndex_1based", "frame_index_offset"),
        ("imageIndex", "frame_index_offset"),
    ]:
        if column not in df.columns:
            continue
        values = safe_numeric_series(df, column)
        if values.notna().any():
            return values, column, units
    return pd.Series(np.nan, index=df.index, dtype="float64"), "unavailable", "unavailable"


def _double_exponential_curve(t: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Evaluate c + a*exp(-t/tau1) + b*exp(-t/tau2)."""
    c_value, a_value, b_value, tau1, tau2 = np.asarray(parameters, dtype=float)
    return c_value + a_value * np.exp(-t / tau1) + b_value * np.exp(-t / tau2)


def _fit_sasi_double_exponential(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, bool]:
    """Fit a double exponential with soft-L1 residual loss for robust validation output."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    span = float(np.max(t) - np.min(t)) if t.size else 0.0
    if t.size < 5 or not np.isfinite(t).all() or not np.isfinite(y).all() or span <= 0:
        return np.full_like(y, np.nan, dtype=float), False

    positive_steps = np.diff(np.unique(t))
    min_step = float(np.min(positive_steps[positive_steps > 0])) if np.any(positive_steps > 0) else max(span / max(len(t) - 1, 1), 1e-3)
    tau_min = max(0.1 * min_step, 1e-6)
    tau_max = max(100.0 * span, 10.0 * min_step)
    delta = float(y[0] - y[-1])
    initial = np.array([float(y[-1]), 0.65 * delta, 0.35 * delta, max(span / 10.0, tau_min), max(span / 2.0, tau_min)], dtype=float)
    lower = np.array([-500.0, -1000.0, -1000.0, tau_min, tau_min], dtype=float)
    upper = np.array([500.0, 1000.0, 1000.0, tau_max, tau_max], dtype=float)
    initial = np.clip(initial, lower + 1e-9, upper - 1e-9)

    try:
        result = least_squares(
            lambda parameters: _double_exponential_curve(t, parameters) - y,
            x0=initial,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=max(float(np.nanstd(y)), 1.0),
            max_nfev=5000,
        )
        fitted = _double_exponential_curve(t, result.x)
        success = bool(result.success and np.isfinite(fitted).all())
        return fitted if success else np.full_like(y, np.nan, dtype=float), success
    except (FloatingPointError, RuntimeError, ValueError):
        return np.full_like(y, np.nan, dtype=float), False


def _first_half_level_crossing(t: np.ndarray, y: np.ndarray, half_level: float) -> float:
    """Return the first interpolated time/index offset where an oriented curve reaches half amplitude."""
    if t.size == 0 or y.size == 0 or not np.isfinite(half_level):
        return np.nan
    reached = np.flatnonzero(np.isfinite(y) & (y >= half_level))
    if reached.size == 0:
        return np.nan
    position = int(reached[0])
    if position == 0:
        return float(t[0])
    x0, x1 = float(t[position - 1]), float(t[position])
    y0, y1 = float(y[position - 1]), float(y[position])
    if not all(np.isfinite([x0, x1, y0, y1])) or abs(y1 - y0) <= 1e-12:
        return x1
    fraction = float(np.clip((half_level - y0) / (y1 - y0), 0.0, 1.0))
    return x0 + fraction * (x1 - x0)


def compute_sasi_syllectogram_research_output(
    df: pd.DataFrame,
    enabled: bool = False,
    start_threshold: float = 20.0,
    smooth_window: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add optional RBC SASI pseudo-syllectogram validation metrics.

    ``AggIndex_0_100`` remains the broad image aggregation-state score.
    ``SASI_0_100`` remains the focused structural organisation/coarsening score.
    ``SASI_syllectogram_AI_like`` is an optional dynamic RheoScan-like
    area-ratio calculated from fitted SASI(t). It is not the official RheoScan
    AI and must remain a clearly labelled research/validation output.
    """
    out = df.copy()
    summary: dict[str, Any] = {
        "enableSASISyllectogramResearchOutput": bool(enabled),
        "SASI_syllectogram_AI_like": np.nan,
        "SASI_syllectogram_AMP_like": np.nan,
        "SASI_syllectogram_T_half_like": np.nan,
        "SASI_syllectogram_fit_R2": np.nan,
        "SASI_syllectogram_fit_RMSE": np.nan,
        "SASI_syllectogram_fit_success": False,
        "SASI_syllectogram_orientation": "not_calculated",
        "SASI_syllectogram_startFrame_or_startTime": np.nan,
        "SASI_syllectogram_axis": "unavailable",
        "SASI_syllectogram_T_half_units": "unavailable",
        "SASI_syllectogram_fit_method": "disabled",
        "SASI_syllectogram_analysis_points": 0,
        "SASI_syllectogram_warning": "",
    }
    if not enabled:
        return out, summary

    if "SASI_0_100" not in out.columns:
        summary["SASI_syllectogram_fit_method"] = "unavailable"
        summary["SASI_syllectogram_warning"] = "SASI pseudo-syllectogram unavailable: SASI_0_100 is missing."
        for key, value in summary.items():
            out[key] = value
        return out, summary

    axis, axis_name, units = _sasi_syllectogram_axis(out)
    sasi = safe_numeric_series(out, "SASI_0_100")
    selected = pd.DataFrame({"axis": axis, "SASI_0_100": sasi}).replace([np.inf, -np.inf], np.nan).dropna()
    selected = selected.loc[selected["axis"] >= float(start_threshold)].copy()
    selected = selected.groupby("axis", as_index=False, sort=True)["SASI_0_100"].mean()
    summary["SASI_syllectogram_axis"] = axis_name
    summary["SASI_syllectogram_T_half_units"] = units
    summary["SASI_syllectogram_analysis_points"] = int(len(selected))
    if selected.empty:
        summary["SASI_syllectogram_fit_method"] = "unavailable"
        summary["SASI_syllectogram_warning"] = "SASI pseudo-syllectogram unavailable: no finite SASI rows exist at or after image/time 20."
        for key, value in summary.items():
            out[key] = value
        return out, summary

    axis_values = selected["axis"].to_numpy(dtype=float)
    summary["SASI_syllectogram_startFrame_or_startTime"] = float(axis_values[0])
    relative_axis = axis_values - axis_values[0]
    window = max(1, int(smooth_window or 1))
    if window % 2 == 0:
        window += 1
    smooth_sasi = selected["SASI_0_100"].rolling(window=window, center=True, min_periods=1).median().to_numpy(dtype=float)
    fitted_curve, fit_success = _fit_sasi_double_exponential(relative_axis, smooth_sasi)
    if fit_success:
        curve = fitted_curve
        fit_method = "robust_double_exponential"
    else:
        curve = smooth_sasi
        fit_method = "fallback_smoothed_curve"
        summary["SASI_syllectogram_warning"] = "Double-exponential fit failed; SASI pseudo-syllectogram outputs use the lightly smoothed SASI curve fallback."

    if fit_success:
        residual = curve - smooth_sasi
        rmse = float(np.sqrt(np.mean(residual * residual))) if residual.size and np.isfinite(residual).all() else np.nan
        sum_squares_total = float(np.sum((smooth_sasi - np.mean(smooth_sasi)) ** 2)) if smooth_sasi.size else np.nan
        sum_squares_residual = float(np.sum(residual * residual)) if residual.size and np.isfinite(residual).all() else np.nan
        r_squared = 1.0 - (sum_squares_residual / sum_squares_total) if np.isfinite(sum_squares_total) and sum_squares_total > 1e-12 and np.isfinite(sum_squares_residual) else np.nan
    else:
        # A smoothed-curve fallback is intentionally not presented as a fitted
        # curve with perfect diagnostics.
        rmse = np.nan
        r_squared = np.nan

    edge_count = min(5, len(curve))
    initial_median = float(np.median(curve[:edge_count]))
    final_median = float(np.median(curve[-edge_count:]))
    if final_median < initial_median:
        oriented_curve = -curve
        orientation = "decreasing_flipped"
    else:
        oriented_curve = curve
        orientation = "increasing"

    s_min = float(np.min(oriented_curve))
    s_max = float(np.max(oriented_curve))
    amplitude = s_max - s_min
    total_time = float(relative_axis[-1] - relative_axis[0]) if relative_axis.size else np.nan
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    area = float(trapezoid(oriented_curve - s_min, relative_axis)) if relative_axis.size >= 2 else np.nan
    rectangle_area = amplitude * total_time if np.isfinite(amplitude) and np.isfinite(total_time) else np.nan
    ai_like = float(np.clip(100.0 * area / rectangle_area, 0.0, 100.0)) if np.isfinite(rectangle_area) and rectangle_area > 1e-12 and np.isfinite(area) else np.nan
    half_level = s_min + 0.5 * amplitude if np.isfinite(amplitude) else np.nan
    t_half = _first_half_level_crossing(relative_axis, oriented_curve, half_level) if amplitude > 1e-12 else np.nan

    summary.update(
        {
            "SASI_syllectogram_AI_like": ai_like,
            "SASI_syllectogram_AMP_like": amplitude if np.isfinite(amplitude) else np.nan,
            "SASI_syllectogram_T_half_like": t_half,
            "SASI_syllectogram_fit_R2": r_squared,
            "SASI_syllectogram_fit_RMSE": rmse,
            "SASI_syllectogram_fit_success": bool(fit_success),
            "SASI_syllectogram_orientation": orientation,
            "SASI_syllectogram_fit_method": fit_method,
        }
    )
    for key, value in summary.items():
        out[key] = value
    return out, summary


def _moving_median_series(series: pd.Series, window: int) -> pd.Series:
    """Return a lightly smoothed numeric series with a centered moving median."""
    win = max(1, int(window or 1))
    if win % 2 == 0:
        win += 1
    return pd.to_numeric(series, errors="coerce").rolling(window=win, center=True, min_periods=1).median()


def _normalize_min_rise(series: pd.Series) -> pd.Series:
    """Normalize intensity as I / min(I) - 1."""
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = values[np.isfinite(values)]
    if finite.empty:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    min_value = float(finite.min())
    if not np.isfinite(min_value) or abs(min_value) <= 1e-12:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    return (values / min_value) - 1.0


def _normalize_minmax01(series: pd.Series) -> pd.Series:
    """Normalize score curves to 0-1 using min-max scaling."""
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = values[np.isfinite(values)]
    if finite.empty:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    low = float(finite.min())
    high = float(finite.max())
    span = high - low
    if not np.isfinite(span) or span <= 1e-12:
        return pd.Series(0.0, index=series.index, dtype="float64").where(values.notna(), np.nan)
    return (values - low) / span


def _normalized_syllectogram_metrics(
    axis: pd.Series,
    curve: pd.Series,
    prefix: str,
    smooth_window: int,
    fit_double_exponential: bool = True,
) -> dict[str, Any]:
    """Calculate source-specific syllectogram kinetics for a normalized curve."""
    selected = pd.DataFrame({"axis": axis, "curve": curve}).replace([np.inf, -np.inf], np.nan).dropna()
    selected = selected.groupby("axis", as_index=False, sort=True)["curve"].mean()
    base: dict[str, Any] = {
        f"{prefix}_start_value": np.nan,
        f"{prefix}_final_value": np.nan,
        f"{prefix}_finalValue": np.nan,
        f"{prefix}_amplitude": np.nan,
        f"{prefix}_AUC": np.nan,
        f"{prefix}_AUC_norm": np.nan,
        f"{prefix}_T50": np.nan,
        f"{prefix}_T90": np.nan,
        f"{prefix}_initialSlope": np.nan,
        f"{prefix}_maxSlope": np.nan,
        f"{prefix}_timeToMaxSlope": np.nan,
        f"{prefix}_peakValue": np.nan,
        f"{prefix}_timeToPeak": np.nan,
        f"{prefix}_plateauValue": np.nan,
        f"{prefix}_fit_R2": np.nan,
        f"{prefix}_fit_RMSE": np.nan,
        f"{prefix}_fit_success": False,
        f"{prefix}_fitR2": np.nan,
        f"{prefix}_fitRMSE": np.nan,
        f"{prefix}_fitSuccess": False,
        f"{prefix}_orientation": "unavailable",
        f"{prefix}_flatCurveFlag": False,
        f"{prefix}_points": int(len(selected)),
        f"{prefix}_nFiniteValues": int(len(selected)),
        f"{prefix}_nTrustedValues": int(len(selected)),
        f"{prefix}_fracTrustedValues": 1.0 if len(selected) else 0.0,
        f"{prefix}_insufficientTrustedValuesFlag": bool("_QC" in prefix and len(selected) < 3),
    }
    if "_QC" in prefix and len(selected) < 3:
        return base
    if len(selected) < 2:
        return base

    axis_values = selected["axis"].to_numpy(dtype=float)
    relative_axis = axis_values - axis_values[0]
    y = _moving_median_series(selected["curve"], smooth_window).to_numpy(dtype=float)
    if not np.isfinite(relative_axis).all() or not np.isfinite(y).all() or float(np.nanmax(relative_axis) - np.nanmin(relative_axis)) <= 0:
        return base

    fit_success = False
    fit_r2 = np.nan
    fit_rmse = np.nan
    curve_for_metrics = y.copy()
    if fit_double_exponential and len(y) >= 5:
        fitted, fit_success = _fit_sasi_double_exponential(relative_axis, y)
        if fit_success:
            residual = fitted - y
            fit_rmse = float(np.sqrt(np.mean(residual * residual))) if residual.size else np.nan
            ss_total = float(np.sum((y - np.mean(y)) ** 2)) if y.size else np.nan
            ss_resid = float(np.sum(residual * residual)) if residual.size else np.nan
            fit_r2 = 1.0 - (ss_resid / ss_total) if np.isfinite(ss_total) and ss_total > 1e-12 and np.isfinite(ss_resid) else np.nan
            curve_for_metrics = fitted

    start_value = float(curve_for_metrics[0])
    final_value = float(curve_for_metrics[-1])
    s_min = float(np.nanmin(curve_for_metrics))
    s_max = float(np.nanmax(curve_for_metrics))
    amplitude = s_max - s_min
    flat = (not np.isfinite(amplitude)) or amplitude <= 1e-12
    if flat:
        orientation = "flat"
        direction = 0.0
    elif final_value < start_value:
        orientation = "decreasing"
        direction = -1.0
    else:
        orientation = "increasing"
        direction = 1.0
    oriented_for_reach = direction * curve_for_metrics if direction else curve_for_metrics
    oriented_start = float(oriented_for_reach[0]) if oriented_for_reach.size else np.nan
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    auc = float(trapezoid(curve_for_metrics, relative_axis)) if len(relative_axis) >= 2 else np.nan
    duration = float(relative_axis[-1] - relative_axis[0]) if len(relative_axis) >= 2 else np.nan
    auc_norm = float(auc / duration) if np.isfinite(auc) and np.isfinite(duration) and duration > 1e-12 else np.nan
    half_level = oriented_start + 0.5 * amplitude if not flat else np.nan
    t90_level = oriented_start + 0.9 * amplitude if not flat else np.nan
    t50 = _first_half_level_crossing(relative_axis, oriented_for_reach, half_level) if not flat else np.nan
    t90 = _first_half_level_crossing(relative_axis, oriented_for_reach, t90_level) if not flat else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_slopes = np.gradient(curve_for_metrics, relative_axis) if len(curve_for_metrics) >= 2 else np.asarray([], dtype=float)
        reach_slopes = direction * raw_slopes if direction else raw_slopes
    finite_reach_slopes = reach_slopes[np.isfinite(reach_slopes)]
    if np.isfinite(direction) and direction < 0:
        peak_pos = int(np.nanargmin(curve_for_metrics))
        peak_value = float(np.nanmin(curve_for_metrics))
    else:
        peak_pos = int(np.nanargmax(curve_for_metrics))
        peak_value = float(np.nanmax(curve_for_metrics))
    max_slope = float(np.nanmax(finite_reach_slopes)) if finite_reach_slopes.size else np.nan
    if finite_reach_slopes.size and raw_slopes.size:
        slope_pos = int(np.nanargmax(np.where(np.isfinite(reach_slopes), reach_slopes, -np.inf)))
        time_to_max_slope = float(relative_axis[slope_pos])
    else:
        time_to_max_slope = np.nan
    plateau_count = max(1, int(np.ceil(0.2 * len(curve_for_metrics))))
    plateau = float(np.nanmedian(curve_for_metrics[-plateau_count:]))

    base.update(
        {
            f"{prefix}_start_value": start_value if np.isfinite(start_value) else np.nan,
            f"{prefix}_final_value": final_value if np.isfinite(final_value) else np.nan,
            f"{prefix}_finalValue": final_value if np.isfinite(final_value) else np.nan,
            f"{prefix}_amplitude": amplitude if np.isfinite(amplitude) else np.nan,
            f"{prefix}_AUC": auc,
            f"{prefix}_AUC_norm": auc_norm,
            f"{prefix}_T50": t50,
            f"{prefix}_T90": t90,
            f"{prefix}_initialSlope": float(raw_slopes[np.isfinite(raw_slopes)][0]) if np.isfinite(raw_slopes).any() else np.nan,
            f"{prefix}_maxSlope": max_slope,
            f"{prefix}_timeToMaxSlope": time_to_max_slope,
            f"{prefix}_peakValue": peak_value,
            f"{prefix}_timeToPeak": float(relative_axis[peak_pos]) if relative_axis.size else np.nan,
            f"{prefix}_plateauValue": plateau if np.isfinite(plateau) else np.nan,
            f"{prefix}_fit_R2": fit_r2,
            f"{prefix}_fit_RMSE": fit_rmse,
            f"{prefix}_fit_success": bool(fit_success),
            f"{prefix}_fitR2": fit_r2,
            f"{prefix}_fitRMSE": fit_rmse,
            f"{prefix}_fitSuccess": bool(fit_success),
            f"{prefix}_orientation": orientation,
            f"{prefix}_flatCurveFlag": bool(flat),
            f"{prefix}_points": int(len(selected)),
        }
    )
    return base


def _first_available_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column with at least one finite numeric value."""
    for column in candidates:
        if column in df.columns and safe_numeric_series(df, column).notna().any():
            return column
    return None


def add_normalized_syllectogram_outputs(
    df: pd.DataFrame,
    smooth_window: int = 5,
    fit_double_exponential: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add normalized syllectogram curves and sample-level kinetics.

    Intensity uses ``I_norm_minRise = I / min(I) - 1``. SASI, mask-SASI,
    ObjectStructure, and gSASI curves use 0-1 min-max normalization to avoid
    division by near-zero values. These outputs are research/validation
    descriptors and do not change any original segmentation or score formula.
    """
    out = df.copy()
    summary: dict[str, Any] = {}
    if out.empty:
        return out, summary

    axis, axis_name, units = _sasi_syllectogram_axis(out)
    out["syllectogram_axis"] = axis
    out["syllectogram_axis_units"] = units
    summary["syllectogram_axis"] = axis_name
    summary["syllectogram_axis_units"] = units

    raw_intensity_col = _first_available_column(out, ["rawMeanIntensity", "objectRawMeanIntensity", "intensityMean"])
    corrected_intensity_col = _first_available_column(
        out,
        ["correctedMeanIntensity", "objectCorrectedMeanIntensity", "noPODMeanIntensity", "podRemovedMeanIntensity", "intensityMean"],
    )
    raw_std_col = _first_available_column(out, ["rawStdIntensity", "objectRawStdIntensity", "intensityStd"])
    corrected_std_col = _first_available_column(
        out,
        ["correctedStdIntensity", "objectCorrectedStdIntensity", "noPODStdIntensity", "podRemovedStdIntensity", "intensityStd"],
    )
    raw_cv_col = _first_available_column(out, ["rawCV", "objectRawCV"])
    corrected_cv_col = _first_available_column(out, ["correctedCV", "objectCorrectedCV", "noPOD_CV", "podRemoved_CV"])
    gain_col = _first_available_column(out, ["illuminationGain", "objectIlluminationGain"])
    jump_col = next((col for col in ["illuminationJumpFlag", "objectIlluminationJumpFlag"] if col in out.columns), None)

    out["syllectogramSource_intensity"] = "corrected_grayscale"
    out["objectSyllectogramSource_intensity"] = "corrected_grayscale"
    if raw_intensity_col:
        out["opticalMeanIntensity_raw"] = safe_numeric_series(out, raw_intensity_col)
        out["opticalMean_raw"] = out["opticalMeanIntensity_raw"]
        out["opticalMean_raw_normMinRise"] = _normalize_min_rise(out["opticalMean_raw"])
        out["opticalMean_raw_norm01"] = _normalize_minmax01(out["opticalMean_raw"])
        out["objectOpticalMeanIntensity_raw"] = out["opticalMeanIntensity_raw"]
        summary.update(_normalized_syllectogram_metrics(axis, out["opticalMean_raw_normMinRise"], "meanI_syl_raw", smooth_window, fit_double_exponential))
    else:
        out["opticalMeanIntensity_raw"] = np.nan
        out["opticalMean_raw"] = np.nan
        out["opticalMean_raw_normMinRise"] = np.nan
        out["opticalMean_raw_norm01"] = np.nan
        out["objectOpticalMeanIntensity_raw"] = np.nan
    if corrected_intensity_col:
        out["meanIntensity_syl_source"] = corrected_intensity_col
        out["opticalMeanIntensity_corrected"] = safe_numeric_series(out, corrected_intensity_col)
        out["opticalMean_corrected"] = out["opticalMeanIntensity_corrected"]
        out["objectOpticalMeanIntensity_corrected"] = out["opticalMeanIntensity_corrected"]
        out["MeanI_norm_minRise"] = _normalize_min_rise(out["opticalMeanIntensity_corrected"])
        out["MeanI_norm_0_1"] = _normalize_minmax01(out["opticalMeanIntensity_corrected"])
        out["opticalMean_corrected_normMinRise"] = out["MeanI_norm_minRise"]
        out["opticalMean_corrected_norm01"] = out["MeanI_norm_0_1"]
        out["opticalMean_normMinRise"] = out["MeanI_norm_minRise"]
        out["opticalMean_norm01"] = out["MeanI_norm_0_1"]
        out["meanIntensity_syl_norm_minRise"] = out["MeanI_norm_minRise"]
        out["meanIntensity_syl_norm_0_1"] = out["MeanI_norm_0_1"]
        out["opticalMeanIntensity_norm_minRise"] = out["MeanI_norm_minRise"]
        out["opticalMeanIntensity_norm_0_1"] = out["MeanI_norm_0_1"]
        out["objectOpticalMeanIntensity_norm_minRise"] = out["MeanI_norm_minRise"]
        out["objectOpticalMeanIntensity_norm_0_1"] = out["MeanI_norm_0_1"]
        summary["meanIntensity_syl_source"] = corrected_intensity_col
        summary.update(_normalized_syllectogram_metrics(axis, out["MeanI_norm_minRise"], "meanI_syl_corrected", smooth_window, fit_double_exponential))
        summary.update(_normalized_syllectogram_metrics(axis, out["MeanI_norm_minRise"], "meanI_syl", smooth_window, fit_double_exponential))
        summary.update(_normalized_syllectogram_metrics(axis, out["MeanI_norm_minRise"], "meanIntensity_syl", smooth_window, fit_double_exponential))
    else:
        out["meanIntensity_syl_source"] = "unavailable"
        out["MeanI_norm_minRise"] = np.nan
        out["MeanI_norm_0_1"] = np.nan
        out["opticalMeanIntensity_corrected"] = np.nan
        out["opticalMean_corrected"] = np.nan
        out["opticalMean_corrected_normMinRise"] = np.nan
        out["opticalMean_corrected_norm01"] = np.nan
        out["opticalMean_normMinRise"] = np.nan
        out["opticalMean_norm01"] = np.nan
        out["objectOpticalMeanIntensity_corrected"] = np.nan
        out["meanIntensity_syl_norm_minRise"] = np.nan
        summary["meanIntensity_syl_source"] = "unavailable"

    if raw_cv_col:
        out["opticalCV_raw"] = safe_numeric_series(out, raw_cv_col)
        out["objectOpticalCV_raw"] = out["opticalCV_raw"]
    elif raw_intensity_col and raw_std_col:
        raw_mean = safe_numeric_series(out, raw_intensity_col)
        raw_std = safe_numeric_series(out, raw_std_col)
        out["opticalCV_raw"] = raw_std / (raw_mean.abs() + np.finfo(float).eps)
        out["objectOpticalCV_raw"] = out["opticalCV_raw"]
    else:
        out["opticalCV_raw"] = np.nan
        out["objectOpticalCV_raw"] = np.nan
    out["opticalCV_raw_norm01"] = _normalize_minmax01(out["opticalCV_raw"])
    summary.update(_normalized_syllectogram_metrics(axis, out["opticalCV_raw_norm01"], "cv_syl_raw", smooth_window, fit_double_exponential))
    if corrected_cv_col:
        out["opticalCV_corrected"] = safe_numeric_series(out, corrected_cv_col)
        out["objectOpticalCV_corrected"] = out["opticalCV_corrected"]
        out["CV_norm_0_1"] = _normalize_minmax01(out["opticalCV_corrected"])
        out["opticalCV_corrected_norm01"] = out["CV_norm_0_1"]
        out["opticalCV_norm_0_1"] = out["CV_norm_0_1"]
        out["objectOpticalCV_norm_0_1"] = out["CV_norm_0_1"]
        summary["cv_syl_source"] = corrected_cv_col
        summary.update(_normalized_syllectogram_metrics(axis, out["CV_norm_0_1"], "cv_syl_corrected", smooth_window, fit_double_exponential))
        summary.update(_normalized_syllectogram_metrics(axis, out["CV_norm_0_1"], "cv_syl", smooth_window, fit_double_exponential))
    elif corrected_intensity_col and corrected_std_col:
        corrected_mean = safe_numeric_series(out, corrected_intensity_col)
        corrected_std = safe_numeric_series(out, corrected_std_col)
        out["opticalCV_corrected"] = corrected_std / (corrected_mean.abs() + np.finfo(float).eps)
        out["objectOpticalCV_corrected"] = out["opticalCV_corrected"]
        out["CV_norm_0_1"] = _normalize_minmax01(out["opticalCV_corrected"])
        out["opticalCV_corrected_norm01"] = out["CV_norm_0_1"]
        out["opticalCV_norm_0_1"] = out["CV_norm_0_1"]
        out["objectOpticalCV_norm_0_1"] = out["CV_norm_0_1"]
        summary["cv_syl_source"] = f"{corrected_std_col}/{corrected_intensity_col}"
        summary.update(_normalized_syllectogram_metrics(axis, out["CV_norm_0_1"], "cv_syl_corrected", smooth_window, fit_double_exponential))
        summary.update(_normalized_syllectogram_metrics(axis, out["CV_norm_0_1"], "cv_syl", smooth_window, fit_double_exponential))
    else:
        out["opticalCV_corrected"] = np.nan
        out["objectOpticalCV_corrected"] = np.nan
        out["opticalCV_corrected_norm01"] = np.nan
        out["CV_norm_0_1"] = np.nan
        out["opticalCV_norm_0_1"] = np.nan
        out["objectOpticalCV_norm_0_1"] = np.nan
        summary["cv_syl_source"] = "unavailable"
    if gain_col:
        out["opticalIlluminationGain"] = safe_numeric_series(out, gain_col)
        out["objectOpticalIlluminationGain"] = out["opticalIlluminationGain"]
    if jump_col:
        out["opticalIlluminationJumpFlag"] = out[jump_col].fillna(False).astype(bool)
        out["objectOpticalIlluminationJumpFlag"] = out["opticalIlluminationJumpFlag"]

    score_specs = [
        ("SASI_syl", "SASI_syl_norm01", None, ["SASI_0_100"], "SASI_syl_source"),
        ("maskSASI_syl_raw", "MaskSASI_syl_raw_0_1", ["MaskSASI_syl_0_1", "MaskSASI_syl_norm01"], ["MaskSASI_raw_0_100", "MaskSASI_like_final"], "syllectogramSource_SASI"),
        ("maskSASI_syl_QC", "MaskSASI_syl_QC_0_1", "MaskSASI_QC_syl_0_1", ["MaskSASI_QC_0_100", "MaskSASI_0_100"], "syllectogramSource_SASI_QC"),
        ("maskGapScore_syl_raw", "MaskGapScore_syl_raw_0_1", "MaskGapScore_syl_0_1", ["MaskGapScore_raw_0_100", "MaskGapScore_like_final"], "syllectogramSource_MaskGapScore"),
        ("maskGapScore_syl_QC", "MaskGapScore_syl_QC_0_1", "MaskGapScore_QC_syl_0_1", ["MaskGapScore_QC_0_100", "MaskGapScore_0_100"], "syllectogramSource_MaskGapScore_QC"),
        ("maskAggIndex_syl_raw", "MaskAggIndex_syl_raw_0_1", "MaskAggIndex_syl_0_1", ["MaskAggIndex_raw_0_100"], "syllectogramSource_MaskAggIndex"),
        ("maskAggIndex_syl_QC", "MaskAggIndex_syl_QC_0_1", "MaskAggIndex_QC_syl_0_1", ["MaskAggIndex_QC_0_100", "MaskAggIndex_0_100"], "syllectogramSource_MaskAggIndex_QC"),
        ("MaskSASI_noPOD_syl", "MaskSASI_noPOD_syl_norm01", None, ["MaskSASI_like_noPOD"], "syllectogramSource_MaskSASI_noPOD"),
        ("MaskSASI_PODremoved_syl", "MaskSASI_PODremoved_syl_norm01", None, ["MaskSASI_like_PODremoved"], "syllectogramSource_MaskSASI_PODremoved"),
        ("objectStructure_syl_raw", "ObjectStructureSyl_raw_0_1", ["ObjectStructureSyl_0_1", "ObjectStructure_syl_norm01"], ["ObjectStructureScore_raw_0_100", "ObjectStructureScore_0_100"], "objectSyllectogramSource_structure"),
        ("objectStructure_syl_QC", "ObjectStructureSyl_QC_0_1", None, ["ObjectStructureScore_QC_0_100", "ObjectStructureScore_QC"], "objectSyllectogramSource_structure_QC"),
        ("objectAggregation_syl_raw", "ObjectAggregationSyl_raw_0_1", "ObjectAggregationSyl_0_1", ["ObjectAggregationIndex_raw_0_100", "ObjectAggregationIndex_0_100"], "objectSyllectogramSource_aggregation"),
        ("objectAggregation_syl_QC", "ObjectAggregationSyl_QC_0_1", None, ["ObjectAggregationIndex_QC_0_100", "ObjectAggregationIndex_QC"], "objectSyllectogramSource_aggregation_QC"),
        ("ObjectStructure_noPOD_syl", "ObjectStructure_noPOD_syl_norm01", None, ["ObjectStructureScore_noPOD"], "objectSyllectogramSource_structure_noPOD"),
        ("ObjectStructure_PODremoved_syl", "ObjectStructure_PODremoved_syl_norm01", None, ["ObjectStructureScore_PODremoved"], "objectSyllectogramSource_structure_PODremoved"),
        ("gSASI_syl", "gSASI_syl_norm01", None, ["gSASI_0_100"], "gSASI_syl_source"),
    ]
    for prefix, norm_col, compatibility_norm_col, candidates, source_key in score_specs:
        source_col = _first_available_column(out, candidates)
        if source_col:
            source_value = "final_selected_mask" if prefix.startswith(("mask", "MaskSASI")) else (
                "final_selected_object_mask" if prefix.startswith(("object", "ObjectStructure")) else source_col
            )
            out[source_key] = source_value
            if source_value == "final_selected_object_mask":
                out["objectSyllectogramSource"] = source_value
            out[norm_col] = _normalize_minmax01(safe_numeric_series(out, source_col))
            if compatibility_norm_col:
                compatibility_cols = compatibility_norm_col if isinstance(compatibility_norm_col, list) else [compatibility_norm_col]
                for alias_col in compatibility_cols:
                    out[alias_col] = out[norm_col]
            summary[source_key] = source_value
            summary.update(_normalized_syllectogram_metrics(axis, out[norm_col], prefix, smooth_window, fit_double_exponential))
        else:
            out[source_key] = "unavailable"
            if source_key.startswith("objectSyllectogramSource") and "objectSyllectogramSource" not in out.columns:
                out["objectSyllectogramSource"] = "unavailable"
            out[norm_col] = np.nan
            if compatibility_norm_col:
                compatibility_cols = compatibility_norm_col if isinstance(compatibility_norm_col, list) else [compatibility_norm_col]
                for alias_col in compatibility_cols:
                    out[alias_col] = np.nan
            summary[source_key] = "unavailable"

    summary_columns = {key: value for key, value in summary.items() if key not in out.columns}
    if summary_columns:
        out = pd.concat([out, pd.DataFrame(summary_columns, index=out.index)], axis=1)
    return out, summary


def compute_rbc_aggregation_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add RBC Gap Mode research scores using robust dataset z-scores.

    These scores assume that the binary mask represents bright plasma gaps.
    They are not meaningful for generic Object Mode or Ground-Truth Mask Mode
    unless explicitly recalibrated for those domains.
    """
    out = df.copy()
    required = [
        "gapMeanDiam_px",
        "gapAreaFrac",
        "LF_log10_LFoverHF",
        "gapNtotal",
        "gapAreaSlope",
        "SHI_tileCV",
    ]
    for col in required:
        if col not in out.columns:
            out[col] = np.nan
    if "Xi_interp_px" not in out.columns and "Xi_px" not in out.columns:
        out["Xi_px"] = np.nan

    xi = _score_xi_series(out)
    reference_mask = _rbc_score_reference_mask(out)
    z_diam = matlab_robust_z(safe_numeric_series(out, "gapMeanDiam_px"), reference_mask)
    z_area = matlab_robust_z(safe_numeric_series(out, "gapAreaFrac"), reference_mask)
    z_lf = matlab_robust_z(safe_numeric_series(out, "LF_log10_LFoverHF"), reference_mask)
    z_xi = matlab_robust_z(xi, reference_mask)
    z_n = matlab_robust_z(safe_numeric_series(out, "gapNtotal"), reference_mask)
    z_slope = matlab_robust_z(safe_numeric_series(out, "gapAreaSlope"), reference_mask)
    z_shi = matlab_robust_z(safe_numeric_series(out, "SHI_tileCV"), reference_mask)

    gap_score_raw = 0.30 * z_diam + 0.25 * z_area + 0.25 * z_lf + 0.15 * z_xi - 0.05 * z_n
    out["LegacyIntensityGapScore_raw"] = gap_score_raw
    out["LegacyIntensityGapScore_0_100"] = sigmoid100(gap_score_raw)
    out["GapScore_raw"] = out["LegacyIntensityGapScore_raw"]
    out["GapScore_0_100"] = out["LegacyIntensityGapScore_0_100"]
    out["primaryGapScoreSource"] = "legacy_intensity_xi_fallback"

    pti_raw = 0.40 * z_slope + 0.30 * z_xi + 0.30 * z_area
    out["PTI_raw"] = pti_raw
    out["PTI_0_100"] = sigmoid100(pti_raw)

    slope_sign = 1.0
    corr_frame = pd.DataFrame({"slope": safe_numeric_series(out, "gapAreaSlope"), "gap_score": gap_score_raw})
    corr_frame = corr_frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(corr_frame) >= 3 and corr_frame["slope"].nunique(dropna=True) > 1 and corr_frame["gap_score"].nunique(dropna=True) > 1:
        corr = corr_frame["slope"].corr(corr_frame["gap_score"], method="spearman")
        if np.isfinite(corr) and corr < 0:
            slope_sign = -1.0

    # MATLAB-compatible S_struct compresses slope and SHI z-scores with tanh,
    # flips slope sign when its Spearman relationship to the structural
    # reference is negative, and maps the mean argument with a sigmoid.
    # Keep this RBC-only. Legacy MATLAB v14 uses c_val = 2.5.
    c_value = 2.5
    t_slope = np.tanh((slope_sign * z_slope) / c_value)
    t_shi = np.tanh(z_shi / c_value)
    s_struct_raw = (t_slope + t_shi) / 2.0
    out["S_struct_raw"] = s_struct_raw
    out["S_struct_0_100"] = sigmoid100(s_struct_raw)

    out = compute_sasi(out)
    if "Xi_mask" in out.columns and safe_numeric_series(out, "Xi_mask").notna().any():
        official = compute_official_mask_scores(
            out,
            xi_col="Xi_mask",
            diameter_col="gapMeanDiam_px",
            count_col="gapNtotal",
            area_col="gapAreaFrac",
            reference_mask=reference_mask,
        )
        out["GapScore_raw"] = official["officialGapScoreRaw"]
        out["GapScore_0_100"] = official["officialGapScore_0_100"]
        out["scoreFormulaVersion"] = official["scoreFormulaVersion"]
        out["scoreNormalizationScope"] = official["scoreNormalizationScope"]
        out["primaryGapScoreSource"] = "final_binary_mask_xi"
    # AggIndex remains the broad image aggregation-state score. SASI remains
    # the focused structural organisation/coarsening score. Optional dynamic
    # SASI pseudo-syllectogram outputs are calculated separately.
    out["AggIndex_0_100"] = (
        0.50 * safe_numeric_series(out, "GapScore_0_100").fillna(50.0)
        + 0.30 * safe_numeric_series(out, "PTI_0_100").fillna(50.0)
        + 0.20 * safe_numeric_series(out, "S_struct_0_100").fillna(50.0)
    )
    out["AggIndex_raw"] = out["AggIndex_0_100"]
    return out


def compute_generic_structure_scores(df: pd.DataFrame, enabled: bool = False) -> pd.DataFrame:
    """Add optional exploratory generic structure scores for Object Mode.

    These scores borrow mathematical structure from RBC research scores but do
    not carry RBC biological meaning. They are intentionally named ``gSASI``
    and ``gAggIndex`` and should be interpreted only as exploratory image
    structure indices for a consistently processed generic dataset.
    """
    out = df.copy()
    out["experimentalCompositeScoresEnabled"] = bool(enabled)
    if not enabled:
        out["gSASI_available"] = False
        out["gAggIndex_available"] = False
        out["gSASI_warning"] = "gSASI not calculated: experimental composite structure scores are disabled."
        out["gAggIndex_warning"] = "gAggIndex not calculated: experimental composite structure scores are disabled."
        out["gSASIAvailability"] = "Not calculated: experimental composite structure scores are disabled."
        out["gAggIndexAvailability"] = "Not calculated: experimental composite structure scores are disabled."
        return out

    xi = safe_numeric_series(out, "correlationLength_px")
    diam = safe_numeric_series(out, "meanDiam_px")
    count = safe_numeric_series(out, "objectCount")
    if not xi.notna().any():
        out["gSASI_raw"] = np.nan
        out["gSASI_0_100"] = np.nan
        out["gSASI_available"] = False
        out["gSASI_warning"] = "gSASI unavailable: correlationLength_px has no finite values."
        out["gSASIAvailability"] = "Unavailable: correlationLength_px has no finite values."
    else:
        gsasi_raw = 0.40 * robust_z(xi) + 0.40 * robust_z(diam) - 0.20 * robust_z(count)
        out["gSASI_raw"] = gsasi_raw
        out["gSASI_0_100"] = sigmoid100(gsasi_raw)
        out["gSASI_available"] = True
        out["gSASI_warning"] = ""
        out["gSASIAvailability"] = "Available as an exploratory generic structure score."

    generic_agg_components = [
        "meanDiam_px",
        "areaFrac",
        "LF_log10_LFoverHF_generic",
        "correlationLength_px",
        "objectCount",
        "sizeDistributionSlope",
        "textureHeterogeneity_tileCV",
    ]
    missing = [column for column in generic_agg_components if column not in out.columns or not safe_numeric_series(out, column).notna().any()]
    if missing:
        out["gAggIndex_0_100"] = np.nan
        out["gAggIndex_available"] = False
        out["gAggIndex_warning"] = "gAggIndex unavailable: missing finite generic components: " + ", ".join(missing)
        out["gAggIndexAvailability"] = "Unavailable: missing finite generic components: " + ", ".join(missing)
        return out

    z_diam = robust_z(safe_numeric_series(out, "meanDiam_px"))
    z_area = robust_z(safe_numeric_series(out, "areaFrac"))
    z_lf = robust_z(safe_numeric_series(out, "LF_log10_LFoverHF_generic"))
    z_xi = robust_z(safe_numeric_series(out, "correlationLength_px"))
    z_count = robust_z(safe_numeric_series(out, "objectCount"))
    z_slope = robust_z(safe_numeric_series(out, "sizeDistributionSlope"))
    z_heterogeneity = robust_z(safe_numeric_series(out, "textureHeterogeneity_tileCV"))

    generic_gap_score = sigmoid100(0.30 * z_diam + 0.25 * z_area + 0.25 * z_lf + 0.15 * z_xi - 0.05 * z_count)
    generic_pti = sigmoid100(0.40 * z_slope + 0.30 * z_xi + 0.30 * z_area)
    generic_struct = sigmoid100((np.tanh(z_slope / 2.0) + np.tanh(z_heterogeneity / 2.0)) / 2.0)
    out["gAggIndex_0_100"] = 0.50 * generic_gap_score + 0.30 * generic_pti + 0.20 * generic_struct
    out["gAggIndex_available"] = True
    out["gAggIndex_warning"] = ""
    out["gAggIndexAvailability"] = "Available as an exploratory generic structure score."
    return out


def dice_coefficient(auto_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Dice coefficient between an automatic mask and ground truth."""
    gt = np.asarray(gt_mask, dtype=bool)
    auto = resize_mask_to_shape(auto_mask, gt.shape)
    intersection = float(np.logical_and(auto, gt).sum())
    denom = float(auto.sum() + gt.sum())
    return 1.0 if denom == 0 else _safe_float(2.0 * intersection / denom)


def iou_coefficient(auto_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute intersection-over-union between an automatic mask and ground truth."""
    gt = np.asarray(gt_mask, dtype=bool)
    auto = resize_mask_to_shape(auto_mask, gt.shape)
    union = float(np.logical_or(auto, gt).sum())
    return 1.0 if union == 0 else _safe_float(float(np.logical_and(auto, gt).sum()) / union)
