"""Multi-sample RBC and generic Object Mode comparison dashboard helpers.

This module is a reporting layer over existing batch outputs. It does not
change segmentation, QC, auto-start, or score calculations.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .qc_plotting import bool_values, count_caption, draw_curve, first_column, metric_spec, metric_view


SAMPLE_ID_COLUMNS = ["sampleName", "sampleKey", "sample_id", "Sample"]
PRIMARY_RBC_METRICS = [
    "SASI_0_100",
    "AggIndex_0_100",
    "Xi_interp_px",
    "gapMeanDiam_px",
    "gapNtotal",
    "gapAreaFrac",
    "LF_log10_LFoverHF",
    "SHI_tileCV",
]
BAR_METRICS = ["SASI_0_100", "AggIndex_0_100", "Xi_interp_px", "gapMeanDiam_px", "gapNtotal"]
TIME_CURVE_METRICS = ["SASI_0_100", "AggIndex_0_100", "Xi_interp_px", "gapMeanDiam_px"]
GROUP_METRICS = ["SASI_0_100", "AggIndex_0_100", "Xi_interp_px", "gapMeanDiam_px", "gapNtotal"]
PRIMARY_GENERIC_METRICS = [
    "gSASI_0_100",
    "gAggIndex_0_100",
    "correlationLength_px",
    "meanDiam_px",
    "objectCount",
    "areaFrac",
    "lacunarity16",
    "textureHeterogeneity_tileCV",
]
GENERIC_BAR_METRICS = ["gSASI_0_100", "gAggIndex_0_100", "correlationLength_px", "meanDiam_px", "objectCount"]
GENERIC_TIME_CURVE_METRICS = ["gSASI_0_100", "gAggIndex_0_100", "correlationLength_px", "meanDiam_px"]
ALL_DASHBOARD_METRICS = list(dict.fromkeys(PRIMARY_RBC_METRICS + PRIMARY_GENERIC_METRICS))


def _clean_text(value: Any) -> str:
    """Return stripped text or an empty string for missing values."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """Return one numeric dataframe column without raising for missing values."""
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def dashboard_metric_profile(summary_df: pd.DataFrame, frame_df: pd.DataFrame) -> dict[str, Any]:
    """Return mode-aware dashboard metric names without mixing score types."""
    columns = set(summary_df.columns) | set(frame_df.columns)
    has_rbc_scores = bool({"SASI_0_100", "AggIndex_0_100"} & columns)
    has_generic_scores = bool({"gSASI_0_100", "gAggIndex_0_100"} & columns)
    generic_mode = has_generic_scores or "objectCount" in columns or "correlationLength_px" in columns
    if generic_mode and not has_rbc_scores:
        return {
            "scoreType": "generic",
            "primaryMetrics": PRIMARY_GENERIC_METRICS,
            "barMetrics": GENERIC_BAR_METRICS,
            "timeCurveMetrics": GENERIC_TIME_CURVE_METRICS,
            "groupMetrics": GENERIC_BAR_METRICS,
            "mixedScores": False,
        }
    return {
        "scoreType": "mixed" if has_rbc_scores and has_generic_scores else "rbc",
        "primaryMetrics": PRIMARY_RBC_METRICS,
        "barMetrics": BAR_METRICS,
        "timeCurveMetrics": TIME_CURVE_METRICS,
        "groupMetrics": GROUP_METRICS,
        "mixedScores": bool(has_rbc_scores and has_generic_scores),
    }


def find_sample_identifier_column(df: pd.DataFrame, allow_filename_fallback: bool = False) -> str | None:
    """Find the preferred sample identifier column or a folder-root fallback."""
    lower_map = {str(column).lower(): str(column) for column in df.columns}
    for candidate in SAMPLE_ID_COLUMNS:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    for candidate in ["sampleFolder", "imageFolder", "folder", "root", "rootName"]:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    if allow_filename_fallback:
        for candidate in ["filename", "file"]:
            if candidate in df.columns:
                return candidate
            if candidate.lower() in lower_map:
                return lower_map[candidate.lower()]
    return None


def _sample_name_from_value(value: Any, source_column: str) -> str:
    """Normalize direct sample IDs while extracting folder basenames when needed."""
    text = _clean_text(value)
    if not text:
        return ""
    if source_column.lower() in {"samplefolder", "imagefolder", "folder", "root", "rootname"}:
        return Path(text.rstrip("\\/")).name
    if source_column.lower() in {"filename", "file"}:
        return Path(text).stem
    return text


def infer_experimental_labels(sample_name: str) -> dict[str, str]:
    """Infer donor, flow-history, and stent labels from common RBC sample names."""
    name = _clean_text(sample_name)
    donor_match = re.search(r"(?i)(?:^|[^A-Z0-9])((?:H|D)0*\d+)(?:[^A-Z0-9]|$)", name)
    donor = donor_match.group(1).upper() if donor_match else "unknown"

    flow_match = re.search(r"(?i)WB[\s_\-]*(\d+(?:[P.]\d+)?)", name)
    if flow_match:
        flow_value = flow_match.group(1).replace("p", ".").replace("P", ".")
        try:
            numeric_flow = float(flow_value)
            flow_text = f"{numeric_flow:g}"
        except ValueError:
            flow_text = flow_value
        flow_history = f"WB{flow_text}"
    else:
        flow_history = "unknown"

    if re.search(r"(?i)NO[\s_\-]*STENT|NOSTENT", name):
        stent_status = "NOSTENT"
    elif re.search(r"(?i)(?:^|[^A-Z])STENT(?:[^A-Z]|$)", name):
        stent_status = "STENT"
    else:
        stent_status = "unknown"
    return {"donor": donor, "flowHistory": flow_history, "stentStatus": stent_status}


def _use_existing_or_inferred(existing: Any, inferred: str) -> str:
    """Use parsed filename labels when available and retain metadata as a fallback."""
    if inferred and inferred.lower() != "unknown":
        return inferred
    text = _clean_text(existing)
    if text and text.lower() not in {"nan", "n/a", "unknown", "none"}:
        return text
    return "unknown"


def _summary_frame_rows(frame_group: pd.DataFrame) -> pd.DataFrame:
    """Return rows used for main summaries, falling back to all rows when unavailable."""
    for column in ["usedInSummary", "usedForMainSummary", "includedInMainSummary", "isAnaFinal"]:
        if column not in frame_group.columns:
            continue
        mask = bool_values(frame_group[column])
        # An explicit all-excluded selection is empty, not permission to use all frames.
        return frame_group.loc[mask].copy()
    return frame_group.copy()


def _sort_frame_rows(frame_group: pd.DataFrame) -> pd.DataFrame:
    """Sort frames consistently before taking final values."""
    for column in ["matlabTime_s", "time_s", "originalImageIndex_1based", "analysisIndex_1based"]:
        if column in frame_group.columns:
            values = pd.to_numeric(frame_group[column], errors="coerce")
            if values.notna().any():
                out = frame_group.copy()
                out["_dashboard_sort"] = values
                return out.sort_values("_dashboard_sort", kind="stable").drop(columns="_dashboard_sort")
    return frame_group.copy()


def _derived_sample_rows(frame_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate reusable sample-level summaries from existing per-frame metrics."""
    rows: list[dict[str, Any]] = []
    for sample_name, group in frame_df.groupby("sampleName", dropna=False, sort=False):
        group = _sort_frame_rows(group)
        summary_rows = _sort_frame_rows(_summary_frame_rows(group))
        row: dict[str, Any] = {
            "sampleName": str(sample_name),
            "nFramesAnalysed": int(len(group)),
            "nFramesUsedForSummary": int(len(summary_rows)),
        }
        for metric in ALL_DASHBOARD_METRICS:
            values = _numeric(summary_rows, metric).dropna()
            row[f"mean_{metric}"] = float(values.mean()) if not values.empty else np.nan
            row[f"median_{metric}"] = float(values.median()) if not values.empty else np.nan
            row[f"final_{metric}"] = float(values.iloc[-1]) if not values.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _merge_derived_summary(summary_df: pd.DataFrame, derived_df: pd.DataFrame) -> pd.DataFrame:
    """Preserve authoritative batch columns while filling missing dashboard summaries."""
    if summary_df.empty:
        return derived_df.copy()
    if derived_df.empty:
        return summary_df.copy()
    merged = pd.merge(summary_df, derived_df, on="sampleName", how="outer", suffixes=("", "__derived"))
    for column in list(merged.columns):
        if not column.endswith("__derived"):
            continue
        target = column[: -len("__derived")]
        if target in merged.columns:
            merged[target] = merged[target].where(merged[target].notna(), merged[column])
        else:
            merged[target] = merged[column]
        merged = merged.drop(columns=column)
    return merged


def _summary_metric_column(summary_df: pd.DataFrame, metric: str, statistic: str = "median") -> str | None:
    """Find an available sample-level summary column for a metric."""
    candidates = [f"{statistic}_{metric}", f"median_{metric}", f"mean_{metric}", f"final_{metric}", metric]
    for column in candidates:
        if column in summary_df.columns and _numeric(summary_df, column).notna().any():
            return column
    return None


def _settings_warnings(summary_df: pd.DataFrame, frame_df: pd.DataFrame) -> list[str]:
    """Warn when metadata suggests samples used different presets or safety profiles."""
    warnings: list[str] = []
    for column in ["cameraPresetName", "presetName", "rbcGapSafetyMode", "LFHF_method"]:
        values: list[str] = []
        for df in [summary_df, frame_df]:
            if column in df.columns:
                values.extend([text for text in df[column].dropna().map(_clean_text).tolist() if text])
        unique = sorted(set(values))
        if len(unique) > 1:
            warnings.append(f"Samples contain multiple {column} values: {', '.join(unique)}. Compare only equivalent imaging settings and presets.")
    return warnings


def build_group_summary(summary_df: pd.DataFrame, group_metrics: list[str] | None = None) -> pd.DataFrame:
    """Build flow-history, stent-status, and combined group summaries."""
    group_metrics = list(group_metrics or GROUP_METRICS)
    rows: list[dict[str, Any]] = []
    grouping_specs = [
        ("flowHistory", ["flowHistory"]),
        ("stentStatus", ["stentStatus"]),
        ("flowHistory x stentStatus", ["flowHistory", "stentStatus"]),
    ]
    for grouping_name, columns in grouping_specs:
        if not set(columns).issubset(summary_df.columns):
            continue
        valid = summary_df.copy()
        for column in columns:
            valid = valid.loc[valid[column].fillna("unknown").astype(str).str.lower() != "unknown"]
        if valid.empty:
            continue
        group_key: str | list[str] = columns[0] if len(columns) == 1 else columns
        for keys, group in valid.groupby(group_key, dropna=False, sort=True):
            values = (keys,) if not isinstance(keys, tuple) else keys
            group_label = " ".join(map(str, values))
            row: dict[str, Any] = {
                "grouping": grouping_name,
                "groupLabel": group_label,
                "nSamples": int(len(group)),
                "warning": "Fewer than 2 samples; interpret cautiously." if len(group) < 2 else "",
            }
            for metric in group_metrics:
                column = _summary_metric_column(group, metric, "median")
                data = _numeric(group, column).dropna() if column else pd.Series(dtype="float64")
                row[f"median_{metric}"] = float(data.median()) if not data.empty else np.nan
                row[f"IQR_{metric}"] = float(data.quantile(0.75) - data.quantile(0.25)) if not data.empty else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def prepare_multi_sample_comparison(
    sample_summary_df: pd.DataFrame,
    frame_metrics_df: pd.DataFrame,
    matlab_comparison_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Validate and prepare dashboard tables without rerunning image analysis."""
    summary = sample_summary_df.copy() if isinstance(sample_summary_df, pd.DataFrame) else pd.DataFrame()
    frames = frame_metrics_df.copy() if isinstance(frame_metrics_df, pd.DataFrame) else pd.DataFrame()
    matlab = matlab_comparison_df.copy() if isinstance(matlab_comparison_df, pd.DataFrame) else pd.DataFrame()
    warnings: list[str] = []
    errors: list[str] = []

    summary_id = find_sample_identifier_column(summary, allow_filename_fallback=True)
    frame_id = find_sample_identifier_column(frames)
    if summary_id is None and frame_id is None:
        errors.append("Cannot compare samples because no sample identifier column was found.")
        return {"summary": pd.DataFrame(), "frames": pd.DataFrame(), "groups": pd.DataFrame(), "matlab": matlab, "warnings": warnings, "errors": errors}

    if frame_id is not None:
        frames["sampleName"] = frames[frame_id].map(lambda value: _sample_name_from_value(value, frame_id))
        frames = frames.loc[frames["sampleName"] != ""].copy()
    if summary_id is not None:
        summary["sampleName"] = summary[summary_id].map(lambda value: _sample_name_from_value(value, summary_id))
        summary = summary.loc[summary["sampleName"] != ""].copy()
    elif not frames.empty:
        summary = pd.DataFrame({"sampleName": frames["sampleName"].drop_duplicates().tolist()})
        warnings.append("sample_summary.csv had no sample identifier; the summary table was rebuilt from frame_metrics.csv sample identifiers.")

    if frames.empty:
        warnings.append("Per-frame metrics are missing or empty. Sample-level tables remain available, but time curves and derived summaries are limited.")
    elif frame_id is None:
        if len(summary) == 1:
            frames["sampleName"] = str(summary["sampleName"].iloc[0])
            warnings.append("frame_metrics.csv had no sample identifier; all frame rows were assigned to the only sample in sample_summary.csv.")
        else:
            warnings.append("frame_metrics.csv has no sample identifier. Time curves and grouped per-frame backfilling are unavailable.")

    derived = _derived_sample_rows(frames) if not frames.empty and "sampleName" in frames.columns else pd.DataFrame()
    summary = _merge_derived_summary(summary, derived)
    summary = summary.drop_duplicates(subset=["sampleName"], keep="first").reset_index(drop=True)

    labels = summary["sampleName"].map(infer_experimental_labels)
    inferred_df = pd.DataFrame(labels.tolist(), index=summary.index)
    summary["donor"] = [
        _use_existing_or_inferred(existing, inferred)
        for existing, inferred in zip(summary.get("donor", pd.Series("", index=summary.index)), inferred_df["donor"])
    ]
    summary["flowHistory"] = [
        _use_existing_or_inferred(existing, inferred)
        for existing, inferred in zip(summary.get("flowHistory", summary.get("flowGroup", pd.Series("", index=summary.index))), inferred_df["flowHistory"])
    ]
    summary["stentStatus"] = [
        _use_existing_or_inferred(existing, inferred)
        for existing, inferred in zip(summary.get("stentStatus", summary.get("stentCondition", pd.Series("", index=summary.index))), inferred_df["stentStatus"])
    ]
    summary["flowGroup"] = summary["flowHistory"]
    summary["stentCondition"] = summary["stentStatus"]

    if "reliableFramePercentage" in summary.columns and "reliableFramePercent" not in summary.columns:
        summary["reliableFramePercent"] = summary["reliableFramePercentage"]
    if "reliableFramePercent" in summary.columns and "reliableFramePercentage" not in summary.columns:
        summary["reliableFramePercentage"] = summary["reliableFramePercent"]

    parsing_failures = summary.loc[
        (summary["donor"] == "unknown") | (summary["flowHistory"] == "unknown") | (summary["stentStatus"] == "unknown"),
        "sampleName",
    ].astype(str).tolist()
    if parsing_failures:
        warnings.append("Could not infer all donor/flow/stent labels for: " + ", ".join(parsing_failures))

    profile = dashboard_metric_profile(summary, frames)
    active_metrics = profile["primaryMetrics"]
    missing_metrics = [metric for metric in active_metrics if _summary_metric_column(summary, metric, "median") is None]
    if missing_metrics:
        warnings.append("Optional comparison metrics missing: " + ", ".join(missing_metrics))

    warnings.extend(_settings_warnings(summary, frames))
    if profile["mixedScores"]:
        warnings.append("Do not compare RBC SASI/AggIndex directly with Object Mode gSASI/gAggIndex. They are different score types.")
    elif profile["scoreType"] == "generic" and any(metric in summary.columns or metric in frames.columns for metric in ["gSASI_0_100", "gAggIndex_0_100"]):
        warnings.append("Object Mode gSASI/gAggIndex are exploratory composite structure scores. They are not RBC aggregation scores.")
    if "scoresConfidence" in summary.columns:
        low_conf = summary.loc[summary["scoresConfidence"].fillna("").astype(str).str.lower() == "low", "sampleName"].astype(str).tolist()
        if low_conf:
            warnings.append("Low-confidence samples: " + ", ".join(low_conf))
    if "sampleQC" in summary.columns:
        unreliable = summary.loc[summary["sampleQC"].fillna("").astype(str) != "Reliable aggregation morphology", "sampleName"].astype(str).tolist()
        if unreliable:
            warnings.append("Samples without reliable aggregation morphology QC: " + ", ".join(unreliable))
    if "largeGapArtefactRisk" in summary.columns:
        risky = summary.loc[summary["largeGapArtefactRisk"].fillna("").astype(str).str.lower().isin(["moderate", "high"]), "sampleName"].astype(str).tolist()
        if risky:
            warnings.append("Moderate/high large-gap artefact risk samples: " + ", ".join(risky))
    if "percentFramesWithLargeGapRemoval" in summary.columns:
        high_removal = summary.loc[_numeric(summary, "percentFramesWithLargeGapRemoval") > 20.0, "sampleName"].astype(str).tolist()
        if high_removal:
            warnings.append("More than 20% of frames had large-gap removal for: " + ", ".join(high_removal))

    groups = build_group_summary(summary, group_metrics=profile["groupMetrics"])
    if not groups.empty:
        sparse_groups = groups.loc[groups["nSamples"] < 2, ["grouping", "groupLabel"]]
        if not sparse_groups.empty:
            labels = [f"{row['grouping']}: {row['groupLabel']}" for _, row in sparse_groups.iterrows()]
            warnings.append("Groups with fewer than 2 samples should be interpreted cautiously: " + ", ".join(labels))
    if not matlab.empty:
        matlab_id = find_sample_identifier_column(matlab, allow_filename_fallback=True)
        if matlab_id is None:
            warnings.append("Optional MATLAB comparison table has no sample identifier column.")
        correlation_columns = [column for column in matlab.columns if "pearson" in str(column).lower() or str(column).lower().startswith("r_")]
        if not correlation_columns:
            warnings.append("Optional MATLAB comparison table has no Pearson/correlation columns.")
    metrics_available = [metric for metric in active_metrics if _summary_metric_column(summary, metric, "median") is not None]
    return {
        "summary": summary,
        "frames": frames,
        "groups": groups,
        "matlab": matlab,
        "warnings": list(dict.fromkeys(warnings)),
        "errors": errors,
        "metricsAvailable": metrics_available,
        **profile,
    }


def _empty_figure(message: str) -> plt.Figure:
    """Create a compact placeholder plot."""
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    return fig


def plot_multi_sample_bar(
    summary_df: pd.DataFrame,
    metric: str,
    statistic: str = "median",
    group_by: str = "Sample",
) -> plt.Figure:
    """Plot one existing RBC metric across samples or inferred groups."""
    column = _summary_metric_column(summary_df, metric, statistic)
    if summary_df.empty or column is None:
        return _empty_figure(f"No finite {metric} summary values are available.")
    plot_df = summary_df.copy()
    plot_df["_value"] = _numeric(plot_df, column)
    if group_by == "Sample":
        values = plot_df["_value"]
        labels = plot_df["sampleName"].astype(str)
        title = f"{statistic.title()} {metric} by sample"
    else:
        group_columns = {
            "flowHistory": ["flowHistory"],
            "stentStatus": ["stentStatus"],
            "flowHistory x stentStatus": ["flowHistory", "stentStatus"],
        }.get(group_by, ["sampleName"])
        valid = plot_df.dropna(subset=["_value"]).copy()
        for group_column in group_columns:
            valid = valid.loc[valid[group_column].fillna("unknown").astype(str).str.lower() != "unknown"]
        if valid.empty:
            return _empty_figure(f"No parsed {group_by} labels are available for {metric}.")
        grouped = valid.groupby(group_columns, sort=True)["_value"].median()
        values = grouped.reset_index()["_value"]
        labels = pd.Series([" ".join(map(str, value if isinstance(value, tuple) else (value,))) for value in grouped.index])
        title = f"{statistic.title()} {metric} grouped by {group_by}"
    finite = values.notna() & np.isfinite(values)
    if not finite.any():
        return _empty_figure(f"No finite {metric} values are available.")
    values = values.loc[finite]
    labels = labels.loc[finite]
    fig, ax = plt.subplots(figsize=(max(7.5, 0.48 * len(values)), 4))
    ax.bar(np.arange(len(values)), values.to_numpy(dtype=float), color="#2F6F8F")
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels.astype(str), rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_multi_sample_time_curves(
    frame_df: pd.DataFrame,
    metric: str,
    selected_samples: list[str],
    time_axis: str,
    show_excluded: bool = False,
) -> tuple[plt.Figure, list[str]]:
    """Plot stored QC-supported values, keeping missing frames as line breaks."""
    warnings: list[str] = []
    spec = metric_spec(metric)
    if frame_df.empty or first_column(frame_df, (*spec.trusted, *spec.raw)) is None or "sampleName" not in frame_df.columns:
        return _empty_figure(f"Per-frame {metric} values are unavailable."), warnings
    requested_axis = time_axis
    if time_axis == "time_s if available":
        time_axis = "matlabTime_s" if "matlabTime_s" in frame_df.columns else ("time_s" if "time_s" in frame_df.columns else "analysisIndex_1based")
    if time_axis not in frame_df.columns:
        fallback = "originalImageIndex_1based" if "originalImageIndex_1based" in frame_df.columns else "analysisIndex_1based"
        warnings.append(f"Requested time axis {requested_axis!r} is unavailable; using {fallback}.")
        time_axis = fallback
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plotted = 0
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for sample_index, sample in enumerate(selected_samples):
        subset = frame_df.loc[frame_df["sampleName"].astype(str) == str(sample)].copy()
        if subset.empty:
            continue
        x = _numeric(subset, time_axis)
        if not x.notna().any():
            x = pd.Series(np.arange(len(subset)) + 1, index=subset.index, dtype="float64")
            warnings.append(f"{sample}: {time_axis} values were not finite; using row position.")
        elif x.isna().any():
            warnings.append(f"{sample}: some {time_axis} values were missing; affected frames were omitted from the time curve.")
        view = metric_view(subset, metric)
        if not view["value"].notna().any():
            warnings.append(f"{sample}: no accepted {metric} values. {count_caption(view)}")
        if not view.attrs["qc_available"]:
            warnings.append(f"{sample}: metric-specific QC is unavailable; any measured curve is dotted grey, not labelled trusted.")
        # Keep every frame and every NaN: never connect across an excluded observation.
        subset = subset.copy()
        subset[time_axis] = x
        n_before = len(ax.lines) + len(ax.collections)
        x_label = draw_curve(ax, subset, view, time_axis, show_excluded, str(sample), colors[sample_index % len(colors)])
        plotted += int(len(ax.lines) + len(ax.collections) > n_before)
    if not plotted:
        plt.close(fig)
        return _empty_figure(f"No accepted values for {metric}. See QC counts and reasons; missing is not zero."), warnings
    ax.set_xlabel(x_label)
    ax.set_ylabel(spec.unit)
    ax.set_title(f"{spec.label} — QC-aware time curves")
    if spec.family == "score":
        ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8)
    fig.tight_layout()
    return fig, warnings


def _top_samples(summary_df: pd.DataFrame, metric: str, statistic: str = "median", ascending: bool = False) -> str:
    """Return a compact ranked sample list for one metric."""
    column = _summary_metric_column(summary_df, metric, statistic)
    if column is None:
        return "not available"
    values = _numeric(summary_df, column)
    ranked = summary_df.assign(_value=values).dropna(subset=["_value"]).sort_values("_value", ascending=ascending).head(3)
    if ranked.empty:
        return "not available"
    return ", ".join(f"{row['sampleName']} ({float(row['_value']):.3g})" for _, row in ranked.iterrows())


def multi_sample_report(
    summary_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    group_df: pd.DataFrame,
    matlab_df: pd.DataFrame,
    warnings: list[str],
) -> str:
    """Generate a cautious, exportable multi-sample comparison report."""
    profile = dashboard_metric_profile(summary_df, frame_df)
    available = [metric for metric in profile["primaryMetrics"] if _summary_metric_column(summary_df, metric, "median") is not None]
    reliable = int((summary_df.get("sampleQC", pd.Series(dtype=str)).fillna("").astype(str) == "Reliable aggregation morphology").sum())
    low_conf = int((summary_df.get("scoresConfidence", pd.Series(dtype=str)).fillna("").astype(str).str.lower() == "low").sum())
    risk = int(summary_df.get("largeGapArtefactRisk", pd.Series(dtype=str)).fillna("").astype(str).str.lower().isin(["moderate", "high"]).sum())
    parsed_flow = int((summary_df.get("flowHistory", pd.Series(dtype=str)).fillna("unknown").astype(str).str.lower() != "unknown").sum())
    parsed_stent = int((summary_df.get("stentStatus", pd.Series(dtype=str)).fillna("unknown").astype(str).str.lower() != "unknown").sum())
    preset_warning = next((warning for warning in warnings if "multiple " in warning and "values:" in warning), "No mixed-preset metadata warning detected.")
    lines = [
        "# MicroStructAI Multi-Sample Comparison Report",
        "",
        "## Dataset Summary",
        f"- Number of samples: {len(summary_df)}",
        f"- Total frame rows: {len(frame_df)}",
        f"- Metrics available: {', '.join(available) if available else 'none'}",
        f"- Samples with parsed flowHistory labels: {parsed_flow}",
        f"- Samples with parsed stentStatus labels: {parsed_stent}",
        f"- Optional MATLAB comparison rows available: {len(matlab_df)}",
        f"- Settings/preset check: {preset_warning}",
        "",
        "## QC Summary",
        f"- Reliable samples: {reliable}",
        f"- Low-confidence samples: {low_conf}",
        f"- Samples with moderate/high large-gap artefact risk: {risk}",
        "",
    ]
    lines += ["## Main Comparison Findings"]
    if profile["scoreType"] == "generic":
        lines += [
            f"- Top samples by median gSASI: {_top_samples(summary_df, 'gSASI_0_100')}",
            f"- Top samples by median gAggIndex: {_top_samples(summary_df, 'gAggIndex_0_100')}",
            f"- Top samples by median correlationLength_px: {_top_samples(summary_df, 'correlationLength_px')}",
            f"- Samples with highest median meanDiam_px: {_top_samples(summary_df, 'meanDiam_px')}",
            f"- Samples with highest median objectCount: {_top_samples(summary_df, 'objectCount')}",
            "- Higher gSASI may indicate larger, less fragmented coarse structural organisation.",
            "- Object Mode gSASI/gAggIndex are exploratory indices, not RBC aggregation scores.",
            "- Do not compare RBC SASI/AggIndex directly with Object Mode gSASI/gAggIndex. They are different score types.",
        ]
    else:
        lines += [
            f"- Top samples by median SASI: {_top_samples(summary_df, 'SASI_0_100')}",
            f"- Top samples by median AggIndex: {_top_samples(summary_df, 'AggIndex_0_100')}",
            f"- Top samples by median Xi_interp_px: {_top_samples(summary_df, 'Xi_interp_px')}",
            f"- Samples with highest median gapMeanDiam_px: {_top_samples(summary_df, 'gapMeanDiam_px')}",
            f"- Samples with highest median gapNtotal: {_top_samples(summary_df, 'gapNtotal')}",
            f"- Samples with lowest median gapNtotal: {_top_samples(summary_df, 'gapNtotal', ascending=True)}",
            "- Higher SASI generally indicates stronger structural organisation of plasma-gap morphology.",
            "- Higher Xi_interp_px suggests a larger characteristic structural length scale.",
            "- Lower gapNtotal together with larger gapMeanDiam_px may indicate more consolidated plasma-gap morphology.",
        ]
    if not group_df.empty and profile["scoreType"] == "rbc":
        for grouping in ["stentStatus", "flowHistory", "flowHistory x stentStatus"]:
            subset = group_df.loc[group_df["grouping"] == grouping]
            if subset.empty:
                continue
            labels = ", ".join(
                f"{row['groupLabel']} (n={int(row['nSamples'])}, "
                f"median SASI={float(row['median_SASI_0_100']):.3g}, "
                f"median AggIndex={float(row['median_AggIndex_0_100']):.3g})"
                for _, row in subset.iterrows()
            )
            lines.append(f"- {grouping} summary: {labels}")
    lines += [
        "",
        "## Warnings and Limitations",
        "- Comparisons are valid only when samples were processed with the same settings/preset and comparable imaging conditions.",
        "- These are research-use structural indices, not clinical diagnostic outputs.",
        "- Overlay inspection remains required before interpretation; RBC QC review is additionally required for RBC Gap Mode outputs.",
        "- Calibration and negative-control limitations should be considered when comparing setups.",
        "- This dashboard reports associations and structural differences; it does not claim biological causality.",
    ]
    if warnings:
        lines += ["", "## Processing Notes"]
        lines += [f"- {warning}" for warning in warnings]
    return "\n".join(lines)
