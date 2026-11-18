"""Camera/setup calibration helpers for RBC Gap Mode."""

from __future__ import annotations

import json
import random
import re
from datetime import datetime
from itertools import product
from typing import Any

import numpy as np
import pandas as pd


SEGMENTATION_GRID = {
    "kHi": [1.8, 2.0, 2.2, 2.4, 2.6],
    "kLo": [0.9, 1.0, 1.1, 1.2, 1.3],
    "sigmaIll": [15.0, 25.0, 35.0, 45.0],
    "minAreaPx": [10, 20, 30, 50],
    "closeRadius": [1, 2, 3],
}

AUTO_START_GRID = {
    "autoStartMinDevelopedSignal": [0.35, 0.45, 0.55],
    "autoStartSlopeFrac": [0.30, 0.40, 0.50],
    "autoStartEarlyScaleFrac": [0.25, 0.35, 0.45],
    "autoStartSearchFrac": [0.30, 0.40, 0.50],
    "autoStartSmoothWin_s": [5.0, 7.0, 9.0],
    "autoStartMinPeakSlope": [0.01, 0.02, 0.03],
    "autoStartMinMaxSignal": [0.35, 0.45, 0.55],
}

SAFETY_GRID = {
    "maxGapAreaFilterOption": ["OFF", 0.05, 0.10, 0.15],
}

TUNABLE_PARAMETER_KEYS = list(SEGMENTATION_GRID) + list(AUTO_START_GRID) + list(SAFETY_GRID)

DEFAULT_CAMERA_PRESETS: dict[str, dict[str, Any]] = {
    "USB camera RBC default": {
        "cropFrac": 0.70,
        "useMedian": True,
        "medianK": 3,
        "sigmaIll": 25.0,
        "trackDark": False,
        "kHi": 2.2,
        "kLo": 1.1,
        "minAreaPx": 20,
        "closeRadius": 2,
        "doFillHoles": True,
        "clearBorder": False,
        "boundaryLineWidth": 1,
        "overlay_background": "Illumination-corrected Wz",
        "lf_hf_method": "MATLAB-compatible continuous-field",
        "rbcGapSafetyMode": "MicroStructAI Safe Mode",
        "enableMaxGapAreaFilter": True,
        "maxGapAreaFrac": 0.10,
        "maxGapAreaFilterOption": 0.10,
        "startFrameMode": "Auto first-rise detector",
        "autoStartMinDevelopedSignal": 0.45,
        "autoStartSlopeFrac": 0.40,
        "autoStartEarlyScaleFrac": 0.35,
        "autoStartSearchFrac": 0.40,
        "autoStartSmoothWin_s": 7.0,
        "autoStartMinPeakSlope": 0.02,
        "autoStartMinMaxSignal": 0.45,
        "minSeparationZ": 1.0,
        "minBoundarySupport": 0.30,
        "minReliableFramePercent": 20.0,
        "minTemporalTrendRho": 0.30,
        "minStructuralContrast": 0.10,
        "minSHI_tileCV_forReliable": 0.15,
    },
    "Microscope RBC bright-gap default": {
        "cropFrac": 0.80,
        "useMedian": True,
        "medianK": 3,
        "sigmaIll": 35.0,
        "trackDark": False,
        "kHi": 2.4,
        "kLo": 1.2,
        "minAreaPx": 30,
        "closeRadius": 2,
        "doFillHoles": True,
        "clearBorder": False,
        "boundaryLineWidth": 1,
        "overlay_background": "Illumination-corrected Wz",
        "lf_hf_method": "MATLAB-compatible continuous-field",
        "rbcGapSafetyMode": "MicroStructAI Safe Mode",
        "enableMaxGapAreaFilter": True,
        "maxGapAreaFrac": 0.10,
        "maxGapAreaFilterOption": 0.10,
        "startFrameMode": "Auto first-rise detector",
        "autoStartMinDevelopedSignal": 0.45,
        "autoStartSlopeFrac": 0.40,
        "autoStartEarlyScaleFrac": 0.35,
        "autoStartSearchFrac": 0.40,
        "autoStartSmoothWin_s": 7.0,
        "autoStartMinPeakSlope": 0.02,
        "autoStartMinMaxSignal": 0.45,
        "minSeparationZ": 1.0,
        "minBoundarySupport": 0.30,
        "minReliableFramePercent": 20.0,
        "minTemporalTrendRho": 0.30,
        "minStructuralContrast": 0.10,
        "minSHI_tileCV_forReliable": 0.15,
    },
    "Conservative low-contrast default": {
        "cropFrac": 0.70,
        "useMedian": True,
        "medianK": 3,
        "sigmaIll": 35.0,
        "trackDark": False,
        "kHi": 2.6,
        "kLo": 1.3,
        "minAreaPx": 30,
        "closeRadius": 1,
        "doFillHoles": True,
        "clearBorder": False,
        "boundaryLineWidth": 1,
        "overlay_background": "Illumination-corrected Wz",
        "lf_hf_method": "MATLAB-compatible continuous-field",
        "rbcGapSafetyMode": "MicroStructAI Safe Mode",
        "enableMaxGapAreaFilter": True,
        "maxGapAreaFrac": 0.10,
        "maxGapAreaFilterOption": 0.10,
        "startFrameMode": "Auto first-rise detector",
        "autoStartMinDevelopedSignal": 0.55,
        "autoStartSlopeFrac": 0.30,
        "autoStartEarlyScaleFrac": 0.35,
        "autoStartSearchFrac": 0.40,
        "autoStartSmoothWin_s": 9.0,
        "autoStartMinPeakSlope": 0.03,
        "autoStartMinMaxSignal": 0.55,
        "minSeparationZ": 1.0,
        "minBoundarySupport": 0.30,
        "minReliableFramePercent": 20.0,
        "minTemporalTrendRho": 0.30,
        "minStructuralContrast": 0.10,
        "minSHI_tileCV_forReliable": 0.15,
    },
}

DEFAULT_CAMERA_PRESETS["MATLAB-compatible RBC legacy"] = {
    **DEFAULT_CAMERA_PRESETS["USB camera RBC default"],
    "rbcGapSafetyMode": "MATLAB-compatible mode",
    "enableMaxGapAreaFilter": False,
    "maxGapAreaFrac": 0.10,
    "maxGapAreaFilterOption": "OFF",
}


def safe_filename_token(value: str) -> str:
    """Return a filesystem-friendly preset name token."""
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip()).strip("_")
    return token or "camera_setup"


def preset_settings(preset: dict[str, Any] | None) -> dict[str, Any]:
    """Extract the settings dictionary from a saved preset or plain settings object."""
    if not isinstance(preset, dict):
        return {}
    nested = preset.get("settings")
    return dict(nested) if isinstance(nested, dict) else dict(preset)


def camera_preset_json(
    preset_name: str,
    settings: dict[str, Any],
    calibration_confidence: str,
    negative_control_used: bool,
    camera_setup_notes: str = "",
    user_notes: str = "",
) -> str:
    """Serialize a camera/setup-specific RBC preset."""
    preset = {
        "presetName": str(preset_name).strip() or "Camera setup preset",
        "createdAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "software": "MicroStructAI MVP",
        "mode": "RBC Gap Mode",
        "researchUseOnly": True,
        "calibrationConfidence": calibration_confidence,
        "negativeControlUsed": bool(negative_control_used),
        "cameraSetupNotes": camera_setup_notes,
        "userNotes": user_notes,
        "settings": dict(settings),
    }
    return json.dumps(preset, indent=2, sort_keys=True)


def load_camera_preset_json(payload: bytes | str) -> dict[str, Any]:
    """Load and minimally validate a camera preset JSON payload."""
    text = payload.decode("utf-8-sig") if isinstance(payload, bytes) else str(payload)
    preset = json.loads(text)
    if not isinstance(preset, dict):
        raise ValueError("Camera preset JSON must contain an object.")
    settings = preset_settings(preset)
    if not settings:
        raise ValueError("Camera preset JSON does not contain settings.")
    preset["settings"] = settings
    preset.setdefault("presetName", "Uploaded camera preset")
    return preset


def generate_calibration_candidates(base_settings: dict[str, Any], max_trials: int = 100, seed: int = 17) -> list[dict[str, Any]]:
    """Sample a reproducible tuning grid while always retaining the base preset."""
    max_trials = max(1, int(max_trials))
    grid = {**SEGMENTATION_GRID, **AUTO_START_GRID, **SAFETY_GRID}
    keys = list(grid)
    base = dict(base_settings)
    base_candidate = _apply_max_gap_filter_option(dict(base))
    for key in keys:
        base_candidate.setdefault(key, grid[key][0])

    def signature(candidate: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(candidate.get(key) for key in keys)

    candidates = [base_candidate]
    seen = {signature(base_candidate)}
    total_grid_size = int(np.prod([len(grid[key]) for key in keys]))
    rng = random.Random(seed)

    if total_grid_size <= max_trials:
        iterator = product(*[grid[key] for key in keys])
        for values in iterator:
            candidate = dict(base)
            candidate.update(dict(zip(keys, values)))
            candidate = _apply_max_gap_filter_option(candidate)
            sig = signature(candidate)
            if sig not in seen:
                seen.add(sig)
                candidates.append(candidate)
            if len(candidates) >= max_trials:
                break
        return candidates

    max_attempts = max_trials * 100
    for _attempt in range(max_attempts):
        candidate = dict(base)
        candidate.update({key: rng.choice(values) for key, values in grid.items()})
        candidate = _apply_max_gap_filter_option(candidate)
        sig = signature(candidate)
        if sig in seen:
            continue
        seen.add(sig)
        candidates.append(candidate)
        if len(candidates) >= max_trials:
            break
    return candidates


def _apply_max_gap_filter_option(settings: dict[str, Any]) -> dict[str, Any]:
    """Translate the light calibration safety option into segmentation settings."""
    out = dict(settings)
    option = out.get("maxGapAreaFilterOption")
    if option is None:
        option = out.get("maxGapAreaFrac", 0.10) if bool(out.get("enableMaxGapAreaFilter", True)) else "OFF"
    if str(option).upper() == "OFF":
        out["maxGapAreaFilterOption"] = "OFF"
        out["enableMaxGapAreaFilter"] = False
        out.setdefault("maxGapAreaFrac", 0.10)
    else:
        frac = float(option)
        out["maxGapAreaFilterOption"] = frac
        out["enableMaxGapAreaFilter"] = True
        out["maxGapAreaFrac"] = frac
    return out


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a finite numeric series or an aligned NaN series."""
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _median(df: pd.DataFrame, column: str) -> float:
    values = _numeric(df, column).dropna()
    return float(values.median()) if not values.empty else np.nan


def _summary(analysis: dict[str, Any] | None) -> dict[str, Any]:
    if not analysis:
        return {}
    return analysis.get("params", {}).get("rbc_qc_summary", {}) or {}


def _summary_rows(analysis: dict[str, Any] | None) -> pd.DataFrame:
    if not analysis:
        return pd.DataFrame()
    metrics = analysis.get("metrics", pd.DataFrame())
    if metrics.empty or "usedForMainSummary" not in metrics.columns:
        return metrics
    return metrics.loc[metrics["usedForMainSummary"].fillna(False).astype(bool)].copy()


def _plausibility_fraction(df: pd.DataFrame, plausibility: dict[str, Any]) -> tuple[float, float, float]:
    if df.empty:
        return 0.0, 1.0, 1.0
    area = _numeric(df, "gapAreaFrac")
    n_total = _numeric(df, "gapNtotal")
    diam = _numeric(df, "gapMeanDiam_px")
    largest = _numeric(df, "gapLargestFrac")
    plausible = (
        area.between(float(plausibility.get("minPlausibleGapAreaFrac", 0.01)), float(plausibility.get("maxPlausibleGapAreaFrac", 0.60)))
        & (n_total >= float(plausibility.get("minPlausibleGapNtotal", 20)))
        & (diam >= float(plausibility.get("minPlausibleGapMeanDiam_px", 2.0)))
    )
    tiny = diam < float(plausibility.get("minPlausibleGapMeanDiam_px", 2.0))
    huge = largest > float(plausibility.get("maxHugeConnectedRegionWarning", 0.80))
    return float(plausible.fillna(False).mean()), float(tiny.fillna(False).mean()), float(huge.fillna(False).mean())


def _trend_reward(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    x = _numeric(df, "matlabTime_s")
    if not x.notna().any():
        x = pd.Series(np.arange(len(df)), index=df.index, dtype="float64")
    correlations: list[float] = []
    for column in ["Xi_interp_px", "gapMeanDiam_px", "LF_log10_LFoverHF", "AggIndex_0_100"]:
        y = _numeric(df, column)
        finite = x.notna() & y.notna()
        if int(finite.sum()) >= 3 and y.loc[finite].nunique() > 1:
            correlations.append(float(x.loc[finite].corr(y.loc[finite], method="spearman")))
    best = max(correlations) if correlations else 0.0
    return 10.0 * float(np.clip(best, 0.0, 1.0))


def _instability_penalty(df: pd.DataFrame, plausibility: dict[str, Any]) -> tuple[float, float]:
    plausible_fraction, tiny_fraction, huge_fraction = _plausibility_fraction(df, plausibility)
    penalty = 20.0 * (1.0 - plausible_fraction)
    penalty += 15.0 * max(0.0, tiny_fraction - float(plausibility.get("maxTinyObjectWarning", 0.30)))
    penalty += 15.0 * huge_fraction
    xi = _numeric(df, "Xi_interp_px").dropna()
    if len(xi) >= 3 and abs(float(xi.mean())) > 1e-12:
        penalty += min(10.0, 5.0 * float(xi.std(ddof=1) / abs(xi.mean())))
    removal_fraction, _detected_fraction, max_removed = _large_gap_filter_stats(df)
    penalty += 12.0 * removal_fraction
    penalty += 10.0 * max(0.0, max_removed - 0.20)
    return float(penalty), plausible_fraction


def _large_gap_filter_stats(df: pd.DataFrame) -> tuple[float, float, float]:
    """Return removal fraction, detection fraction, and largest suspicious gap."""
    if df.empty:
        return 0.0, 0.0, 0.0
    removed = _numeric(df, "nLargeGapObjectsRemoved").fillna(0) > 0
    detected = _numeric(df, "nLargeGapObjectsDetected").fillna(0) > 0
    largest = _numeric(df, "largestRemovedGapFrac").dropna()
    return (
        float(removed.mean()),
        float(detected.mean()),
        float(largest.max()) if not largest.empty else 0.0,
    )


def _early_large_gap_residual_penalty(df: pd.DataFrame) -> float:
    """Penalize huge gap objects that remain in the early internal baseline."""
    if df.empty:
        return 0.0
    early_count = max(1, int(round(0.25 * len(df))))
    early = df.iloc[:early_count]
    residual = _numeric(early, "largestGapFrac_afterFilter").fillna(0.0)
    threshold = _numeric(early, "maxGapAreaFrac").fillna(0.10)
    excess = (residual - threshold).clip(lower=0.0)
    return float(min(15.0, 40.0 * float(excess.mean())))


def _negative_false_positive_penalty(analysis: dict[str, Any] | None) -> float:
    if not analysis:
        return 0.0
    summary = _summary(analysis)
    sample_qc = str(summary.get("sampleQC", ""))
    confidence = str(summary.get("scoresConfidence", "")).lower()
    auto_start = summary.get("autoStartFrame", np.nan)
    no_start = bool(summary.get("noStartSafetyTriggered", False))
    reliable_percent = float(summary.get("reliableFramePercent", 0.0) or 0.0)
    max_signal = float(summary.get("maxStartSignalInSearchWindow", 0.0) or 0.0)
    penalty = 0.0
    if np.isfinite(pd.to_numeric(pd.Series([auto_start]), errors="coerce").iloc[0]) and not no_start:
        penalty += 35.0
    if sample_qc == "Reliable aggregation morphology":
        penalty += 30.0
    if confidence in {"high", "moderate"}:
        penalty += 15.0
    penalty += min(20.0, 0.20 * reliable_percent)
    penalty += 10.0 * float(np.clip(max_signal - 0.45, 0.0, 1.0))
    negative_metrics = analysis.get("metrics", pd.DataFrame())
    _removed_fraction, detected_fraction, max_removed = _large_gap_filter_stats(negative_metrics)
    penalty += 15.0 * detected_fraction
    penalty += 10.0 * max(0.0, max_removed - 0.10)
    return float(penalty)


def score_calibration_candidate(
    candidate_id: int,
    settings: dict[str, Any],
    positive_analysis: dict[str, Any],
    negative_analysis: dict[str, Any] | None,
    plausibility: dict[str, Any],
    user_start_frame: float | None = None,
) -> dict[str, Any]:
    """Score one camera/setup candidate without changing the underlying RBC formulas."""
    positive_summary = _summary(positive_analysis)
    positive_rows = _summary_rows(positive_analysis)
    positive_auto_start = pd.to_numeric(pd.Series([positive_summary.get("autoStartFrame", np.nan)]), errors="coerce").iloc[0]
    positive_no_start = bool(positive_summary.get("noStartSafetyTriggered", False))
    positive_qc = str(positive_summary.get("sampleQC", "n/a"))
    reliable_percent = float(positive_summary.get("reliableFramePercent", 0.0) or 0.0)
    instability_penalty, plausible_fraction = _instability_penalty(positive_rows, plausibility)
    positive_score = 0.0
    if np.isfinite(positive_auto_start) and not positive_no_start:
        positive_score += 30.0
    if positive_qc == "Reliable aggregation morphology":
        positive_score += 25.0
    positive_score += min(20.0, 0.20 * reliable_percent)
    positive_score += 15.0 * plausible_fraction
    positive_score += _trend_reward(positive_rows)

    start_error = np.nan
    start_error_penalty = 0.0
    if user_start_frame is not None and np.isfinite(float(user_start_frame)) and np.isfinite(positive_auto_start):
        start_error = abs(float(positive_auto_start) - float(user_start_frame))
        start_error_penalty = min(40.0, 1.5 * start_error)

    negative_summary = _summary(negative_analysis)
    negative_penalty = _negative_false_positive_penalty(negative_analysis)
    positive_removal_fraction, positive_detection_fraction, positive_max_removed = _large_gap_filter_stats(positive_rows)
    negative_removal_fraction, negative_detection_fraction, negative_max_removed = _large_gap_filter_stats(
        negative_analysis.get("metrics", pd.DataFrame()) if negative_analysis else pd.DataFrame()
    )
    option = settings.get("maxGapAreaFilterOption", settings.get("maxGapAreaFrac", "OFF"))
    safety_preference_penalty = 0.0 if option == 0.10 else (0.5 if option in {0.05, 0.15} else 1.0)
    early_large_gap_residual_penalty = _early_large_gap_residual_penalty(positive_analysis.get("metrics", pd.DataFrame()))
    calibration_score = (
        positive_score
        - negative_penalty
        - instability_penalty
        - start_error_penalty
        - safety_preference_penalty
        - early_large_gap_residual_penalty
    )
    row: dict[str, Any] = {
        "candidateID": int(candidate_id),
        **{key: settings.get(key) for key in TUNABLE_PARAMETER_KEYS},
        "calibrationScore": float(calibration_score),
        "positiveDetectionScore": float(positive_score),
        "negativeFalsePositivePenalty": float(negative_penalty),
        "instabilityPenalty": float(instability_penalty),
        "startFrameErrorPenalty": float(start_error_penalty),
        "safetyPreferencePenalty": float(safety_preference_penalty),
        "earlyLargeGapResidualPenalty": float(early_large_gap_residual_penalty),
        "enableMaxGapAreaFilter": bool(settings.get("enableMaxGapAreaFilter", False)),
        "maxGapAreaFrac": float(settings.get("maxGapAreaFrac", 0.10)),
        "positiveFramesWithLargeGapRemovalPercent": 100.0 * positive_removal_fraction,
        "positiveFramesWithLargeGapDetectionPercent": 100.0 * positive_detection_fraction,
        "positiveMaxLargestRemovedGapFrac": positive_max_removed,
        "negativeFramesWithLargeGapRemovalPercent": 100.0 * negative_removal_fraction,
        "negativeFramesWithLargeGapDetectionPercent": 100.0 * negative_detection_fraction,
        "negativeMaxLargestRemovedGapFrac": negative_max_removed,
        "positiveSampleQC": positive_qc,
        "negativeSampleQC": str(negative_summary.get("sampleQC", "not provided")) if negative_analysis else "not provided",
        "positiveAutoStartFrame": float(positive_auto_start) if np.isfinite(positive_auto_start) else np.nan,
        "negativeAutoStartFrame": negative_summary.get("autoStartFrame", np.nan) if negative_analysis else np.nan,
        "noStartSafetyTriggered": positive_no_start,
        "positiveNoStartSafetyTriggered": positive_no_start,
        "negativeNoStartSafetyTriggered": bool(negative_summary.get("noStartSafetyTriggered", False)) if negative_analysis else np.nan,
        "reliableFramePercent_positive": reliable_percent,
        "reliableFramePercent_negative": negative_summary.get("reliableFramePercent", np.nan) if negative_analysis else np.nan,
        "userStartFrame": user_start_frame if user_start_frame is not None else np.nan,
        "absStartErrorFrames": start_error,
        "positivePlausibleFramePercent": 100.0 * plausible_fraction,
    }
    return row


def calibration_confidence(best_row: pd.Series, negative_control_used: bool) -> str:
    """Return a cautious calibration-confidence label for the recommended preset."""
    positive_ok = (
        str(best_row.get("positiveSampleQC", "")) == "Reliable aggregation morphology"
        and not bool(best_row.get("positiveNoStartSafetyTriggered", True))
        and np.isfinite(pd.to_numeric(pd.Series([best_row.get("positiveAutoStartFrame", np.nan)]), errors="coerce").iloc[0])
    )
    negative_ok = (
        negative_control_used
        and str(best_row.get("negativeSampleQC", "")) == "No reliable plasma-gap aggregation detected"
        and bool(best_row.get("negativeNoStartSafetyTriggered", False))
    )
    start_error = pd.to_numeric(pd.Series([best_row.get("absStartErrorFrames", np.nan)]), errors="coerce").iloc[0]
    start_ok = not np.isfinite(start_error) or float(start_error) <= 5.0
    if positive_ok and negative_ok and start_ok:
        return "high"
    if positive_ok and (not negative_control_used or float(best_row.get("negativeFalsePositivePenalty", 0.0)) < 25.0):
        return "moderate"
    return "low"


def calibration_report_markdown(result: dict[str, Any]) -> str:
    """Build a concise research-use camera calibration report."""
    recommended = result.get("recommendedSettings", {})
    best = result.get("bestRow", {}) or {}
    negative_used = bool(result.get("negativeControlUsed", False))
    lines = [
        "# MicroStructAI Camera Calibration Report",
        "",
        "## Calibration Summary",
        f"- Calibration mode: {result.get('calibrationMode', 'n/a')}",
        f"- Calibration image input method: {result.get('calibrationInputMethod', 'Local folder path')}",
        f"- Calibration confidence: {result.get('calibrationConfidence', 'n/a')}",
        f"- Positive sequence source: {result.get('positiveSource', result.get('positiveFolder', 'n/a'))}",
        f"- Negative/control sequence source: {result.get('negativeSource', result.get('negativeFolder', 'not provided')) or 'not provided'}",
        f"- Negative/control type: {result.get('negativeControlType', 'not provided')}",
        f"- Candidate settings tested: {len(result.get('calibrationResults', pd.DataFrame()))}",
        "",
        "## Recommended Preset",
    ]
    for key in TUNABLE_PARAMETER_KEYS:
        lines.append(f"- {key}: {recommended.get(key, 'n/a')}")
    lines += [
        "",
        "## Score Breakdown",
        f"- Calibration score: {best.get('calibrationScore', 'n/a')}",
        f"- Positive detection score: {best.get('positiveDetectionScore', 'n/a')}",
        f"- Negative false-positive penalty: {best.get('negativeFalsePositivePenalty', 'n/a')}",
        f"- Instability penalty: {best.get('instabilityPenalty', 'n/a')}",
        f"- Start-frame error penalty: {best.get('startFrameErrorPenalty', 'n/a')}",
        f"- Safety preference penalty: {best.get('safetyPreferencePenalty', 'n/a')}",
        f"- Maximum single-gap filter enabled: {recommended.get('enableMaxGapAreaFilter', 'n/a')}",
        f"- maxGapAreaFrac: {recommended.get('maxGapAreaFrac', 'n/a')}",
        f"- Positive frames with large-gap removal: {best.get('positiveFramesWithLargeGapRemovalPercent', 'n/a')}%",
        f"- Negative/control frames with large-gap detection: {best.get('negativeFramesWithLargeGapDetectionPercent', 'n/a')}%",
        "",
        "## Interpretation Notes",
        "- Calibration settings are setup-specific. A new camera, microscope, magnification, lighting condition, focus setting, bit depth, or sample preparation may require recalibration.",
        "- The recommended preset should be checked visually using overlays before interpreting SASI/AggIndex.",
        "- This calibration recommends a research-use preset; it does not establish absolute segmentation truth.",
    ]
    if not negative_used:
        lines += [
            "- Calibration confidence is limited because no negative/no-aggregation control was provided.",
            "- If a PBS/no-blood control is later available, rerun full calibration using that control.",
            "- Large-gap artefact filter calibrated using internal early-frame baseline, not true PBS/blank/no-blood control.",
        ]
    if result.get("negativeFalsePositiveDetected", False):
        lines.append("- This preset is not safe for this setup because it produced a false-positive aggregation result on the negative/control sequence.")
    return "\n".join(lines)
