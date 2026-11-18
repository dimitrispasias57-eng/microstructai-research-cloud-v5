"""MATLAB CSV comparison utilities for RBC Gap Mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import robust_z, sigmoid100


AUTO_MATCH = "auto: prefer time_s if available"
TIME_MATCH = "match by time_s"
ROW_ORDER_MATCH = "match by row order"
ORIGINAL_INDEX_MATCH = "match by originalImageIndex_1based"
ANALYSIS_INDEX_MATCH = "match by analysisIndex_1based"
MATCH_METHODS = [AUTO_MATCH, TIME_MATCH, ROW_ORDER_MATCH]

FILENAME_COLUMNS = ["filename", "file", "File", "Filename", "imageFile", "image_filename", "name"]
TIME_COLUMNS = ["matlabTime_s", "time_s", "Time_s", "time", "tPOD"]
INDEX_ALIASES = {
    ORIGINAL_INDEX_MATCH: ["originalImageIndex_1based", "frame", "Frame", "frameIndex", "imageIndex", "index"],
    ANALYSIS_INDEX_MATCH: ["analysisIndex_1based", "analysisIndex", "frame", "Frame", "frameIndex", "index"],
}

METRIC_MAP = [
    ("gapMeanDiam_px", "gapMeanDiam_px", "gapMeanDiam_px"),
    ("gapNtotal", "gapNtotal", "gapNtotal"),
    ("gapAreaFrac", "gapAreaFrac", "gapAreaFrac"),
    ("gapAreaSlope", "gapAreaSlope", "gapAreaSlope"),
    ("LF_log10_LFoverHF", "LF_log10_LFoverHF", "LF_log10_LFoverHF"),
    ("SHI_tileCV", "SHI_tileCV", "SHI_tileCV"),
    ("Xi_interp_px", "Xi_interp_px", "Xi_interp_px"),
    ("SASI_0_100", "SASI_0_100", "SASI_0_100"),
    ("AggIndex_0_100", "AggIndex_0_100", "AggIndex_0_100"),
    ("PTI_0_100", "PTI_0_100", "PTI_0_100"),
    ("GapScore_0_100", "GapScore_0_100", "GapScore_0_100"),
]

DERIVED_MATLAB_NOTES = {
    "SASI_0_100": (
        "MATLAB CSV did not contain SASI_0_100, so the comparison table derived it from "
        "MATLAB Xi_interp_px/Xi_px, gapMeanDiam_px, and gapNtotal using the current SASI formula."
    )
}


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column by exact or case-insensitive candidate names."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    lower_map = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def _normalize_filename(value: Any) -> str:
    """Normalize filenames for app/MATLAB row matching."""
    if pd.isna(value):
        return ""
    return Path(str(value)).name.strip().lower()


def _prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Prefix dataframe columns for side-by-side app/MATLAB matching."""
    return df.reset_index(drop=True).add_prefix(prefix)


def _match_by_row_order(app_df: pd.DataFrame, matlab_df: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
    """Pair rows in current order."""
    n_rows = min(len(app_df), len(matlab_df))
    warnings: list[str] = []
    if len(app_df) != len(matlab_df):
        warnings.append(f"Row-order matching used {n_rows} rows; app has {len(app_df)} rows and MATLAB has {len(matlab_df)} rows.")
    matched = pd.concat([_prefix(app_df.iloc[:n_rows], "app__"), _prefix(matlab_df.iloc[:n_rows], "matlab__")], axis=1)
    return matched, ROW_ORDER_MATCH, warnings


def _match_by_filename(app_df: pd.DataFrame, matlab_df: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None, list[str]]:
    """Match rows by normalized filename when both tables have filename columns."""
    app_file_col = _find_column(app_df, FILENAME_COLUMNS)
    matlab_file_col = _find_column(matlab_df, FILENAME_COLUMNS)
    if app_file_col is None or matlab_file_col is None:
        return None, None, []

    app_work = app_df.copy()
    matlab_work = matlab_df.copy()
    app_work["_match_filename"] = app_work[app_file_col].map(_normalize_filename)
    matlab_work["_match_filename"] = matlab_work[matlab_file_col].map(_normalize_filename)
    app_work = app_work[app_work["_match_filename"] != ""]
    matlab_work = matlab_work[matlab_work["_match_filename"] != ""]
    if app_work.empty or matlab_work.empty:
        return None, None, ["Filename columns were found, but no usable filename values were available."]

    merged = pd.merge(
        app_work.add_prefix("app__"),
        matlab_work.add_prefix("matlab__"),
        left_on="app___match_filename",
        right_on="matlab___match_filename",
        how="inner",
    )
    merged = merged.drop(columns=[col for col in ["app___match_filename", "matlab___match_filename"] if col in merged.columns])
    if merged.empty:
        return None, None, ["Filename columns were found, but no filenames matched."]
    return merged.reset_index(drop=True), "filename", []


def _match_by_time(app_df: pd.DataFrame, matlab_df: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
    """Match rows by MATLAB/app time_s when both tables provide it."""
    app_time_col = _find_column(app_df, TIME_COLUMNS)
    matlab_time_col = _find_column(matlab_df, TIME_COLUMNS)
    warnings: list[str] = []
    diagnostics: dict[str, Any] = {
        "appTimeAvailable": app_time_col is not None,
        "matlabTimeAvailable": matlab_time_col is not None,
        "unmatchedAppTimes": [],
        "unmatchedMatlabTimes": [],
    }
    if app_time_col is None or matlab_time_col is None:
        warnings.append(
            "time_s matching failed because one or both tables lack time columns or because times do not overlap. "
            "Row-order matching may be unreliable."
        )
        empty = pd.DataFrame()
        empty.attrs["time_match_diagnostics"] = diagnostics
        return empty, TIME_MATCH, warnings

    app_work = app_df.copy()
    matlab_work = matlab_df.copy()
    app_work["_match_time"] = pd.to_numeric(app_work[app_time_col], errors="coerce")
    matlab_work["_match_time"] = pd.to_numeric(matlab_work[matlab_time_col], errors="coerce")
    app_work = app_work.dropna(subset=["_match_time"])
    matlab_work = matlab_work.dropna(subset=["_match_time"])
    diagnostics.update(
        {
            "appTimeFirst": float(app_work["_match_time"].min()) if not app_work.empty else np.nan,
            "appTimeLast": float(app_work["_match_time"].max()) if not app_work.empty else np.nan,
            "matlabTimeFirst": float(matlab_work["_match_time"].min()) if not matlab_work.empty else np.nan,
            "matlabTimeLast": float(matlab_work["_match_time"].max()) if not matlab_work.empty else np.nan,
        }
    )
    if app_work.empty or matlab_work.empty:
        warnings.append(
            "time_s matching failed because one or both tables lack time columns or because times do not overlap. "
            "Row-order matching may be unreliable."
        )
        empty = pd.DataFrame()
        empty.attrs["time_match_diagnostics"] = diagnostics
        return empty, TIME_MATCH, warnings

    tolerance = 1e-6
    matlab_available = set(matlab_work.index.tolist())
    matched_pairs: list[tuple[int, int]] = []
    for app_idx, app_time in app_work["_match_time"].items():
        candidates = [
            matlab_idx
            for matlab_idx in matlab_available
            if abs(float(matlab_work.loc[matlab_idx, "_match_time"]) - float(app_time)) <= tolerance
        ]
        if not candidates:
            continue
        best_idx = min(candidates, key=lambda idx: abs(float(matlab_work.loc[idx, "_match_time"]) - float(app_time)))
        matched_pairs.append((app_idx, best_idx))
        matlab_available.remove(best_idx)

    matched_app_idx = {app_idx for app_idx, _matlab_idx in matched_pairs}
    matched_matlab_idx = {matlab_idx for _app_idx, matlab_idx in matched_pairs}
    unmatched_app_times = app_work.loc[[idx for idx in app_work.index if idx not in matched_app_idx], "_match_time"].tolist()
    unmatched_matlab_times = matlab_work.loc[[idx for idx in matlab_work.index if idx not in matched_matlab_idx], "_match_time"].tolist()
    diagnostics["unmatchedAppTimes"] = [float(value) for value in unmatched_app_times[:5]]
    diagnostics["unmatchedMatlabTimes"] = [float(value) for value in unmatched_matlab_times[:5]]

    if not matched_pairs:
        warnings.append(
            "time_s matching failed because one or both tables lack time columns or because times do not overlap. "
            "Row-order matching may be unreliable."
        )
        warnings.append(
            f"time_s matching matched 0 rows; unmatched app rows: {len(unmatched_app_times)}, "
            f"unmatched MATLAB rows: {len(unmatched_matlab_times)}."
        )
        if unmatched_app_times:
            warnings.append("First unmatched app matlabTime_s values: " + ", ".join(f"{float(value):.6g}" for value in unmatched_app_times[:5]))
        if unmatched_matlab_times:
            warnings.append("First unmatched MATLAB time_s values: " + ", ".join(f"{float(value):.6g}" for value in unmatched_matlab_times[:5]))
        empty = pd.DataFrame()
        empty.attrs["time_match_diagnostics"] = diagnostics
        return empty, TIME_MATCH, warnings

    matched_app = app_df.loc[[app_idx for app_idx, _matlab_idx in matched_pairs]].reset_index(drop=True).add_prefix("app__")
    matched_matlab = matlab_df.loc[[matlab_idx for _app_idx, matlab_idx in matched_pairs]].reset_index(drop=True).add_prefix("matlab__")
    merged = pd.concat([matched_app, matched_matlab], axis=1)
    merged.attrs["time_match_diagnostics"] = diagnostics
    if unmatched_app_times or unmatched_matlab_times:
        warnings.append(
            f"time_s matching matched {len(matched_pairs)} rows; unmatched app rows: {len(unmatched_app_times)}, "
            f"unmatched MATLAB rows: {len(unmatched_matlab_times)}."
        )
        if unmatched_app_times:
            warnings.append("First unmatched app matlabTime_s values: " + ", ".join(f"{float(value):.6g}" for value in unmatched_app_times[:5]))
        if unmatched_matlab_times:
            warnings.append("First unmatched MATLAB time_s values: " + ", ".join(f"{float(value):.6g}" for value in unmatched_matlab_times[:5]))
    out = merged.reset_index(drop=True)
    out.attrs["time_match_diagnostics"] = diagnostics
    return out, TIME_MATCH, warnings


def _match_by_index(app_df: pd.DataFrame, matlab_df: pd.DataFrame, method: str) -> tuple[pd.DataFrame, str, list[str]]:
    """Match rows by original or analysis index when possible."""
    app_key = method.replace("match by ", "")
    matlab_key = _find_column(matlab_df, INDEX_ALIASES.get(method, [app_key]))
    warnings: list[str] = []
    if app_key not in app_df.columns or matlab_key is None:
        matched, used, row_warnings = _match_by_row_order(app_df, matlab_df)
        warnings.append(f"{method} was requested, but matching index columns were not available on both tables. Falling back to row order.")
        warnings.extend(row_warnings)
        return matched, used, warnings

    app_work = app_df.copy()
    matlab_work = matlab_df.copy()
    app_work["_match_index"] = pd.to_numeric(app_work[app_key], errors="coerce")
    matlab_work["_match_index"] = pd.to_numeric(matlab_work[matlab_key], errors="coerce")
    app_work = app_work.dropna(subset=["_match_index"])
    matlab_work = matlab_work.dropna(subset=["_match_index"])
    merged = pd.merge(
        app_work.add_prefix("app__"),
        matlab_work.add_prefix("matlab__"),
        left_on="app___match_index",
        right_on="matlab___match_index",
        how="inner",
    )
    merged = merged.drop(columns=[col for col in ["app___match_index", "matlab___match_index"] if col in merged.columns])
    if merged.empty:
        matched, used, row_warnings = _match_by_row_order(app_df, matlab_df)
        warnings.append(f"{method} produced no matched rows. Falling back to row order.")
        warnings.extend(row_warnings)
        return matched, used, warnings
    return merged.reset_index(drop=True), method, warnings


def match_app_matlab_rows(app_df: pd.DataFrame, matlab_df: pd.DataFrame, method: str) -> tuple[pd.DataFrame, str, list[str]]:
    """Match MicroStructAI RBC rows to MATLAB CSV rows."""
    if method == AUTO_MATCH:
        if _find_column(app_df, TIME_COLUMNS) is not None and _find_column(matlab_df, TIME_COLUMNS) is not None:
            return _match_by_time(app_df, matlab_df)
        matched, row_used, row_warnings = _match_by_row_order(app_df, matlab_df)
        row_warnings.append("Auto matching could not use time_s because one or both tables lack time columns; row-order matching may be unreliable.")
        return matched, row_used, row_warnings
    if method == TIME_MATCH:
        return _match_by_time(app_df, matlab_df)
    if method == ROW_ORDER_MATCH:
        return _match_by_row_order(app_df, matlab_df)
    if method in {ORIGINAL_INDEX_MATCH, ANALYSIS_INDEX_MATCH}:
        return _match_by_index(app_df, matlab_df, method)
    return _match_by_row_order(app_df, matlab_df)


def _resolve_metric_pairs(app_df: pd.DataFrame, matlab_df: pd.DataFrame) -> tuple[list[dict[str, str]], list[str]]:
    """Resolve requested comparison metrics to columns present in both tables."""
    pairs: list[dict[str, str]] = []
    missing: list[str] = []
    for metric_name, app_col, matlab_col_default in METRIC_MAP:
        if metric_name == "Xi_interp_px":
            app_col = "Xi_interp_px" if "Xi_interp_px" in app_df.columns else "Xi_px"
            matlab_col = "Xi_interp_px" if "Xi_interp_px" in matlab_df.columns else "Xi_px"
        else:
            matlab_col = matlab_col_default
        if app_col in app_df.columns and matlab_col in matlab_df.columns:
            pairs.append({"metric": metric_name, "app_col": app_col, "matlab_col": matlab_col})
        else:
            missing.append(f"{metric_name} (app: {app_col if app_col in app_df.columns else 'missing'}, MATLAB: {matlab_col if matlab_col in matlab_df.columns else 'missing'})")
    return pairs, missing


def _comparison_index(matched: pd.DataFrame) -> pd.Series:
    """Choose a stable frame/index value for comparison plots."""
    for col in ["app__matlabTime_s", "app__time_s", "matlab__time_s", "app__originalImageIndex_1based", "app__analysisIndex_1based", "app__imageIndex"]:
        if col in matched.columns:
            values = pd.to_numeric(matched[col], errors="coerce")
            if values.notna().any():
                return values.fillna(pd.Series(np.arange(len(matched)) + 1, index=matched.index))
    return pd.Series(np.arange(len(matched)) + 1, index=matched.index, dtype="float64")


def _filename_series(matched: pd.DataFrame) -> pd.Series:
    """Return app filenames when available."""
    if "app__filename" in matched.columns:
        return matched["app__filename"].astype(str)
    return pd.Series([f"row_{idx + 1}" for idx in range(len(matched))], index=matched.index)


def _add_missing_derived_matlab_columns(matlab_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add comparison-only MATLAB columns that older CSV exports omitted."""
    out = matlab_df.copy()
    notes: list[str] = []
    if "SASI_0_100" not in out.columns:
        xi_col = _find_column(out, ["Xi_interp_px", "Xi_px", "xi_px"])
        diam_col = _find_column(out, ["gapMeanDiam_px"])
        n_col = _find_column(out, ["gapNtotal"])
        if xi_col and diam_col and n_col:
            z_xi = robust_z(pd.to_numeric(out[xi_col], errors="coerce"))
            z_diam = robust_z(pd.to_numeric(out[diam_col], errors="coerce"))
            z_n = robust_z(pd.to_numeric(out[n_col], errors="coerce"))
            out["SASI_raw"] = 0.40 * z_xi + 0.40 * z_diam - 0.20 * z_n
            out["SASI_0_100"] = sigmoid100(out["SASI_raw"])
            notes.append(DERIVED_MATLAB_NOTES["SASI_0_100"])
    return out, notes


def _safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    """Return correlation only when both paired vectors vary."""
    if len(a) < 2 or a.nunique(dropna=True) <= 1 or b.nunique(dropna=True) <= 1:
        return np.nan
    try:
        return float(a.corr(b, method=method))
    except Exception:
        return np.nan


def _time_column_stats(df: pd.DataFrame, candidates: list[str]) -> tuple[bool, float, float]:
    """Return availability and first/last finite time values."""
    column = _find_column(df, candidates)
    if column is None:
        return False, np.nan, np.nan
    values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return True, np.nan, np.nan
    return True, float(values.min()), float(values.max())


def compare_matlab_metrics(app_df: pd.DataFrame, matlab_df: pd.DataFrame, match_method: str = AUTO_MATCH) -> dict[str, Any]:
    """Compare shared RBC Gap Mode metrics between app output and MATLAB CSV."""
    app_df = app_df.copy()
    matlab_df, derived_notes = _add_missing_derived_matlab_columns(matlab_df)
    app_time_available, app_time_first, app_time_last = _time_column_stats(app_df, ["matlabTime_s"])
    matlab_time_available, matlab_time_first, matlab_time_last = _time_column_stats(matlab_df, ["time_s"])
    matched, used_method, warnings = match_app_matlab_rows(app_df, matlab_df, match_method)
    time_diagnostics = matched.attrs.get("time_match_diagnostics", {}) if isinstance(matched, pd.DataFrame) else {}
    metric_pairs, missing_metrics = _resolve_metric_pairs(app_df, matlab_df)

    frame_index = _comparison_index(matched)
    filenames = _filename_series(matched)
    summary_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []

    for pair in metric_pairs:
        metric = pair["metric"]
        app_col = f"app__{pair['app_col']}"
        matlab_col = f"matlab__{pair['matlab_col']}"
        if app_col not in matched.columns or matlab_col not in matched.columns:
            continue
        app_values = pd.to_numeric(matched[app_col], errors="coerce")
        matlab_values = pd.to_numeric(matched[matlab_col], errors="coerce")
        finite = app_values.notna() & matlab_values.notna() & np.isfinite(app_values) & np.isfinite(matlab_values)
        paired_app = app_values[finite]
        paired_matlab = matlab_values[finite]
        n_points = int(finite.sum())
        diff = paired_app - paired_matlab
        pearson = _safe_corr(paired_app, paired_matlab, method="pearson")
        spearman = _safe_corr(paired_app, paired_matlab, method="spearman")
        summary_rows.append(
            {
                "metric": metric,
                "appColumn": pair["app_col"],
                "matlabColumn": pair["matlab_col"],
                "nPaired": n_points,
                "Pearson_r": pearson,
                "Spearman_r": spearman,
                "MAE": float(np.mean(np.abs(diff))) if n_points else np.nan,
                "medianAbsError": float(np.median(np.abs(diff))) if n_points else np.nan,
                "appMean": float(np.mean(paired_app)) if n_points else np.nan,
                "matlabMean": float(np.mean(paired_matlab)) if n_points else np.nan,
                "appMedian": float(np.median(paired_app)) if n_points else np.nan,
                "matlabMedian": float(np.median(paired_matlab)) if n_points else np.nan,
                "bias": float(np.median(paired_app) - np.median(paired_matlab)) if n_points else np.nan,
            }
        )
        for idx in paired_app.index:
            long_rows.append(
                {
                    "metric": metric,
                    "frameIndex": frame_index.loc[idx],
                    "filename": filenames.loc[idx],
                    "appValue": float(app_values.loc[idx]),
                    "matlabValue": float(matlab_values.loc[idx]),
                    "difference": float(app_values.loc[idx] - matlab_values.loc[idx]),
                }
            )

    summary = pd.DataFrame(summary_rows)
    long = pd.DataFrame(long_rows)
    return {
        "summary": summary,
        "long": long,
        "matchedRows": int(len(matched)),
        "matchMethodRequested": match_method,
        "matchMethodUsed": used_method,
        "missingMetrics": missing_metrics,
        "derivedMetricNotes": derived_notes,
        "appMatlabTimeAvailable": bool(app_time_available),
        "matlabTimeAvailable": bool(matlab_time_available),
        "appMatlabTimeFirst": app_time_first,
        "appMatlabTimeLast": app_time_last,
        "matlabTimeFirst": matlab_time_first,
        "matlabTimeLast": matlab_time_last,
        "timeMatchDiagnostics": time_diagnostics,
        "warnings": warnings,
    }


def matlab_comparison_report_lines(result: dict[str, Any] | None) -> list[str]:
    """Return report lines summarizing MATLAB comparison quality."""
    if not result:
        return ["", "## MATLAB Comparison Summary", "- MATLAB comparison was not run."]
    summary = result.get("summary", pd.DataFrame())
    lines = [
        "",
        "## MATLAB Comparison Summary",
        f"- Match method used: {result.get('matchMethodUsed', 'n/a')}",
        f"- Matched rows: {result.get('matchedRows', 0)}",
    ]
    if "matlabTotalRows" in result:
        lines.append(f"- MATLAB total rows: {result.get('matlabTotalRows', 'n/a')}")
    if "matlabAnalysisRows" in result:
        lines.append(f"- MATLAB analysis rows (isAna == 1): {result.get('matlabAnalysisRows', 'n/a')}")
    if "matlabRowsCompared" in result:
        lines.append(f"- MATLAB rows compared: {result.get('matlabRowsCompared', 'n/a')}")
    if "appAnalysedRows" in result:
        lines.append(f"- App analysed rows: {result.get('appAnalysedRows', 'n/a')}")
    if "isAnaFilterUsed" in result:
        lines.append(f"- MATLAB isAna filter used: {result.get('isAnaFilterUsed')}")
    if "appMatlabTimeAvailable" in result:
        lines.append(f"- App matlabTime_s available: {result.get('appMatlabTimeAvailable')}")
    if "matlabTimeAvailable" in result:
        lines.append(f"- MATLAB time_s available: {result.get('matlabTimeAvailable')}")
    if "appMatlabTimeFirst" in result:
        first = result.get("appMatlabTimeFirst", np.nan)
        last = result.get("appMatlabTimeLast", np.nan)
        lines.append(f"- App matlabTime_s range: {first:.4g} to {last:.4g}" if np.isfinite(first) and np.isfinite(last) else "- App matlabTime_s range: n/a")
    if "matlabTimeFirst" in result:
        first = result.get("matlabTimeFirst", np.nan)
        last = result.get("matlabTimeLast", np.nan)
        lines.append(f"- MATLAB time_s range: {first:.4g} to {last:.4g}" if np.isfinite(first) and np.isfinite(last) else "- MATLAB time_s range: n/a")
    for note in result.get("derivedMetricNotes", []):
        lines.append(f"- {note}")
    if summary.empty:
        lines.append("- No shared finite metrics were available for MATLAB comparison.")
        return lines

    strong: list[str] = []
    moderate: list[str] = []
    sensitive: list[str] = []
    needs_calibration: list[str] = []
    for _, row in summary.iterrows():
        metric = str(row["metric"])
        pearson = row.get("Pearson_r", np.nan)
        n_points = row.get("nPaired", 0)
        mae = row.get("MAE", np.nan)
        bias = row.get("bias", np.nan)
        matlab_median = row.get("matlabMedian", np.nan)

        if metric == "gapAreaSlope":
            sensitive.append(metric)
            continue

        if n_points < 3 or not np.isfinite(pearson):
            needs_calibration.append(metric)
            continue

        scale = abs(float(matlab_median)) if np.isfinite(matlab_median) and abs(float(matlab_median)) > 1e-9 else np.nan
        rel_mae = abs(float(mae)) / scale if np.isfinite(mae) and np.isfinite(scale) else np.nan
        rel_bias = abs(float(bias)) / scale if np.isfinite(bias) and np.isfinite(scale) else np.nan
        error_noticeable = (
            (np.isfinite(rel_mae) and rel_mae > 0.15)
            or (np.isfinite(rel_bias) and rel_bias > 0.10)
        )

        if pearson >= 0.90 and not error_noticeable:
            strong.append(metric)
        elif pearson >= 0.75:
            moderate.append(metric)
        else:
            needs_calibration.append(metric)

    lines.append(f"- Strong agreement: {', '.join(strong) if strong else 'none yet'}")
    lines.append(f"- Moderate agreement: {', '.join(moderate) if moderate else 'none yet'}")
    lines.append(f"- Sensitive / caution metrics: {', '.join(sensitive) if sensitive else 'none flagged'}")
    lines.append(f"- Needs calibration/check: {', '.join(needs_calibration) if needs_calibration else 'none flagged'}")
    if "gapAreaSlope" in sensitive:
        lines.append(
            "- gapAreaSlope is a sensitive size-distribution descriptor. Its median/scale may be close to MATLAB, "
            "but frame-level agreement can be lower because small differences in connected-component splitting, "
            "boundary pixels, or log-bin assignment can change the fitted slope."
        )
    if "PTI_0_100" in moderate:
        lines.append(
            "- PTI partly depends on gapAreaSlope; therefore PTI absolute differences may partly reflect gapAreaSlope sensitivity."
        )
    return lines
