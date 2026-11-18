"""Versioned RBC structural-score formulas used by MicroStructAI V3.

The primary V3 scores are calculated from final binary plasma-gap mask
measurements.  Continuous/intensity autocorrelation is intentionally excluded
from these formulas; it remains available elsewhere as a legacy diagnostic.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


RBC_SCORE_FORMULA_VERSION = "3.0-mask-structural"
RBC_SCORE_NORMALIZATION_WITHIN_SEQUENCE = "within_sequence_robust_z"
RBC_SCORE_NORMALIZATION_FIXED_REFERENCE = "fixed_reference_robust_z"

OFFICIAL_SASI_WEIGHTS = {
    "xi_mask": 0.40,
    "gap_mean_diameter": 0.40,
    "gap_count": -0.20,
}
OFFICIAL_GAP_SCORE_WEIGHTS = {
    "gap_mean_diameter": 0.35,
    "gap_area_fraction": 0.25,
    "xi_mask": 0.25,
    "gap_count": -0.15,
}


def _numeric_series(values: Any, index: pd.Index) -> pd.Series:
    """Return numeric values on ``index``, replacing infinities with NaN."""
    if isinstance(values, pd.Series):
        series = pd.to_numeric(values.reindex(index), errors="coerce")
    else:
        series = pd.to_numeric(pd.Series(values, index=index), errors="coerce")
    return series.replace([np.inf, -np.inf], np.nan).astype("float64")


def sigmoid100_preserve_nan(values: pd.Series) -> pd.Series:
    """Map finite raw scores to 0-100 while preserving unavailable values."""
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    result = pd.Series(np.nan, index=numeric.index, dtype="float64")
    finite = numeric.notna()
    clipped = np.clip(numeric.loc[finite].to_numpy(dtype=float), -50.0, 50.0)
    result.loc[finite] = 100.0 / (1.0 + np.exp(-clipped))
    return result


def robust_z_with_reference(
    values: pd.Series,
    reference_mask: pd.Series | np.ndarray | None = None,
    fixed_stats: Mapping[str, float] | None = None,
) -> tuple[pd.Series, dict[str, float]]:
    """Return raw-MAD robust z values and the reference statistics used.

    ``fixed_stats`` must contain ``median`` and ``mad``.  Without fixed stats,
    statistics are fitted only on finite reference rows (or all finite rows if
    that selection is empty).  Missing input values remain missing.
    """
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
    finite = numeric.notna()

    if fixed_stats is not None:
        if "median" not in fixed_stats or "mad" not in fixed_stats:
            raise ValueError("Fixed reference statistics require both 'median' and 'mad'.")
        center = float(fixed_stats["median"])
        mad = float(fixed_stats["mad"])
        if not np.isfinite(center) or not np.isfinite(mad) or mad < 0:
            raise ValueError("Fixed reference median and MAD must be finite, with MAD >= 0.")
        reference_count = int(fixed_stats.get("n", 0))
    else:
        if reference_mask is None:
            selected = finite.copy()
        else:
            selected_array = np.asarray(reference_mask, dtype=bool)
            if selected_array.size != len(numeric):
                raise ValueError("reference_mask length must match score inputs.")
            selected = pd.Series(selected_array, index=numeric.index) & finite
        if not selected.any():
            selected = finite.copy()
        if not selected.any():
            return (
                pd.Series(np.nan, index=numeric.index, dtype="float64"),
                {"median": np.nan, "mad": np.nan, "n": 0},
            )
        reference_values = numeric.loc[selected].to_numpy(dtype=float)
        center = float(np.median(reference_values))
        mad = float(np.median(np.abs(reference_values - center)))
        reference_count = int(reference_values.size)

    denominator = mad + np.finfo(float).eps
    z_values = pd.Series(np.nan, index=numeric.index, dtype="float64")
    z_values.loc[finite] = (numeric.loc[finite] - center) / denominator
    return z_values, {"median": center, "mad": mad, "n": reference_count}


def compute_official_mask_scores(
    df: pd.DataFrame,
    *,
    xi_col: str,
    diameter_col: str,
    count_col: str,
    area_col: str,
    reference_mask: pd.Series | np.ndarray | None = None,
    reference_stats: Mapping[str, Mapping[str, float]] | None = None,
) -> pd.DataFrame:
    """Calculate the official V3 mask SASI and mask GapScore.

    By default, robust-z reference statistics are fitted within the current
    sequence.  For comparisons on a locked scale, callers may pass precomputed
    reference statistics for ``xi_mask``, ``gap_mean_diameter``, ``gap_count``,
    and ``gap_area_fraction``.  V3 deliberately supplies no universal constants
    because they must be established from an appropriate calibration cohort.
    """
    index = df.index
    values = {
        "xi_mask": _numeric_series(df.get(xi_col, pd.Series(np.nan, index=index)), index),
        "gap_mean_diameter": _numeric_series(df.get(diameter_col, pd.Series(np.nan, index=index)), index),
        "gap_count": _numeric_series(df.get(count_col, pd.Series(np.nan, index=index)), index),
        "gap_area_fraction": _numeric_series(df.get(area_col, pd.Series(np.nan, index=index)), index),
    }

    if reference_stats is not None:
        missing = [name for name in values if name not in reference_stats]
        if missing:
            raise ValueError("Fixed reference statistics missing: " + ", ".join(missing))

    z: dict[str, pd.Series] = {}
    fitted_stats: dict[str, dict[str, float]] = {}
    for name, series in values.items():
        z[name], fitted_stats[name] = robust_z_with_reference(
            series,
            reference_mask=reference_mask,
            fixed_stats=reference_stats.get(name) if reference_stats is not None else None,
        )

    sasi_raw = (
        OFFICIAL_SASI_WEIGHTS["xi_mask"] * z["xi_mask"]
        + OFFICIAL_SASI_WEIGHTS["gap_mean_diameter"] * z["gap_mean_diameter"]
        + OFFICIAL_SASI_WEIGHTS["gap_count"] * z["gap_count"]
    )
    gap_raw = (
        OFFICIAL_GAP_SCORE_WEIGHTS["gap_mean_diameter"] * z["gap_mean_diameter"]
        + OFFICIAL_GAP_SCORE_WEIGHTS["gap_area_fraction"] * z["gap_area_fraction"]
        + OFFICIAL_GAP_SCORE_WEIGHTS["xi_mask"] * z["xi_mask"]
        + OFFICIAL_GAP_SCORE_WEIGHTS["gap_count"] * z["gap_count"]
    )
    sasi_required = pd.concat(
        [values["xi_mask"], values["gap_mean_diameter"], values["gap_count"]], axis=1
    ).notna().all(axis=1)
    gap_required = pd.concat(values.values(), axis=1).notna().all(axis=1)
    sasi_raw = sasi_raw.where(sasi_required, np.nan)
    gap_raw = gap_raw.where(gap_required, np.nan)
    scope = (
        RBC_SCORE_NORMALIZATION_FIXED_REFERENCE
        if reference_stats is not None
        else RBC_SCORE_NORMALIZATION_WITHIN_SEQUENCE
    )

    result = pd.DataFrame(index=index)
    result["officialSASIRaw"] = sasi_raw
    result["officialSASI_0_100"] = sigmoid100_preserve_nan(sasi_raw)
    result["officialGapScoreRaw"] = gap_raw
    result["officialGapScore_0_100"] = sigmoid100_preserve_nan(gap_raw)
    result["scoreFormulaVersion"] = RBC_SCORE_FORMULA_VERSION
    result["scoreNormalizationScope"] = scope
    result["scoreReferenceStats"] = pd.Series(
        [fitted_stats for _ in range(len(result))], index=index, dtype=object
    )
    return result
