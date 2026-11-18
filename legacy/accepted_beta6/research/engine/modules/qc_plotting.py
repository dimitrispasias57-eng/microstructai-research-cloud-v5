"""Read-only, metric-specific plot views over already calculated V3 results.

Never recompute scores, replace missing QC scores with raw scores, or mutate the
analysis table. Retain excluded rows as NaN so time curves cannot bridge them.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlotMetric:
    key: str
    label: str
    unit: str
    raw: tuple[str, ...]
    trusted: tuple[str, ...] = ()
    family: str = "mask"
    trust_flags: tuple[str, ...] = ()


METRICS = [
    *[PlotMetric(name, name, "Score (0–100)",
                 (f"{name}_raw_0_100", f"{name}_0_100"),
                 (f"{name}_QC_0_100",), "score", (f"{name}_isTrusted",))
      for name in ("SASI", "AggIndex", "GapScore", "PTI")],
    PlotMetric("Xi", "Horizontal correlation length ξ — gap mask", "Pixels",
               ("Xi_mask", "raw_Xi"), ("Xi_QC", "trusted_Xi"), "shape", ("Xi_isTrusted",)),
    PlotMetric("diameter", "Mean equivalent gap diameter", "Pixels",
               ("raw_gapMeanDiam", "gapMeanDiamFinal", "gapMeanDiam_px"), ("trusted_gapMeanDiam",), "shape"),
    PlotMetric("area", "Gap area fraction", "Fraction of image",
               ("gapAreaFracFinal", "gapAreaFrac", "mask_gapAreaFrac")),
    PlotMetric("count", "Gap count", "Objects",
               ("gapNtotalFinal", "gapNtotal", "mask_gapNtotal")),
    PlotMetric("largest", "Largest gap fraction", "Fraction of image",
               ("raw_largestGapFrac", "largestGapFracFinal", "gapLargestFrac"), ("trusted_largestGapFrac",), "shape"),
    PlotMetric("lacunarity", "Lacunarity (16-pixel scale)", "Lacunarity", ("lacunarity16",), family="shape"),
    PlotMetric("interface", "Gap boundary density", "Pixels⁻¹", ("interfaceDensity_px",), family="shape"),
    PlotMetric("connectivity", "Largest connected foreground fraction", "Fraction",
               ("largestClusterFrac_phase",), family="shape"),
    PlotMetric("complement_connectivity", "Largest connected background fraction", "Fraction",
               ("largestClusterFrac_complement",), family="shape"),
    PlotMetric("area_slope", "Gap area-distribution slope", "Slope", ("gapAreaSlope",), family="shape"),
    PlotMetric("optical_mean", "Mean grayscale intensity", "Intensity",
               ("opticalMean_corrected", "opticalMean_raw"), family="image"),
    PlotMetric("optical_cv", "Grayscale intensity variation (CV)", "CV",
               ("opticalCV_corrected", "opticalCV_raw"), family="image"),
    PlotMetric("shi", "Spatial heterogeneity (SHI)", "Tile CV", ("SHI_tileCV",), family="image"),
    PlotMetric("directionality", "Image-field directionality", "Anisotropy", ("anisotropy",), family="image"),
    PlotMetric("order", "Image-field orientational order", "Order parameter", ("orientOrderParam",), family="image"),
    PlotMetric("orientation", "Dominant image-field orientation", "Degrees", ("orientation_deg",), family="image"),
    PlotMetric("lf_hf", "Low-/high-frequency texture ratio", "log10(LF/HF)", ("LF_log10_LFoverHF",), family="image"),
    PlotMetric("image_structure", "Image-field structure score (diagnostic)", "Score (0–100)", ("S_struct_0_100",), family="image"),
    PlotMetric("image_xi", "Continuous-field correlation length (diagnostic)", "Pixels",
               ("Xi_interp_px", "Xi_px"), family="image"),
]
BY_KEY = {spec.key: spec for spec in METRICS}
BY_COLUMN = {col: spec for spec in METRICS for col in (*spec.raw, *spec.trusted)}
SUMMARY_FLAGS = ("usedInSummary", "usedForMainSummary", "includedInMainSummary")


def bool_values(values: pd.Series) -> pd.Series:
    """Parse native and CSV booleans; 'False' must never become True."""
    return values.map(lambda value: str(value).strip().lower() in {"true", "1", "1.0", "yes"}).astype(bool)


def first_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    # Select by presence, NOT by finite values: an all-missing QC column is final.
    return next((col for col in candidates if col in df.columns), None)


def numeric(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def metric_spec(column: str) -> PlotMetric:
    if column in BY_KEY:
        return BY_KEY[column]
    if column in BY_COLUMN:
        return BY_COLUMN[column]
    # Generic metrics require explicit object-QC metadata, not RBC assumptions.
    return PlotMetric(column, column, "Metric value", (column,), family="object")


def available_metrics(df: pd.DataFrame, families: set[str] | None = None) -> list[PlotMetric]:
    return [spec for spec in METRICS if (families is None or spec.family in families)
            and first_column(df, (*spec.trusted, *spec.raw)) is not None]


def metric_view(df: pd.DataFrame, spec: PlotMetric | str) -> pd.DataFrame:
    """Return display values/status/reasons; use existing gates, no new thresholds."""
    if isinstance(spec, str):
        spec = metric_spec(spec)
    raw_col = first_column(df, spec.raw)
    qc_col = first_column(df, spec.trusted)
    source = qc_col or raw_col
    raw = numeric(df, raw_col)
    candidate = numeric(df, source)
    allowed = pd.Series(True, index=df.index)
    warning = pd.Series(False, index=df.index)
    reasons = pd.Series("", index=df.index, dtype=object)
    known = qc_col is not None

    def reject(mask: pd.Series, reason: str) -> None:
        nonlocal allowed, reasons
        allowed &= ~mask
        reasons = reasons.mask(mask, reasons + reason + "; ")

    if spec.family == "image":
        # Mask failures say nothing conclusive about separately measured optics.
        trust = first_column(df, (f"{spec.key}_isTrusted", "opticalFrameReliable"))
        if trust:
            known = True
            reject(~bool_values(df[trust]), "Optical/image QC exclusion")
        if "opticalWarningFlag" in df:
            warning = bool_values(df["opticalWarningFlag"])
    elif spec.family == "object":
        trust = first_column(df, (f"{spec.key}_isTrusted", "objectMaskReliable"))
        summary = first_column(df, ("objectUsedInSummary", *SUMMARY_FLAGS))
        for col in (trust, summary):
            if col:
                known = True
                reject(~bool_values(df[col]), f"Excluded by {col}")
        if "hardBadObjectMaskFlag" in df:
            known = True
            reject(bool_values(df["hardBadObjectMaskFlag"]), "Hard object-mask failure")
    else:
        summary = first_column(df, SUMMARY_FLAGS)
        if summary:
            known = True
            reject(~bool_values(df[summary]), "Outside the selected main-summary frames")
        for col in ("hardFailure", "hardBadMaskFlag", "hugeGapFlag"):
            if col in df:
                known = True
                reject(bool_values(df[col]), f"Mask QC: {col}")
        trust = first_column(df, spec.trust_flags)
        if trust is None and spec.family == "score":
            trust = first_column(df, ("trustedForScore", "score_isTrusted"))
        if trust:
            known = True
            reject(~bool_values(df[trust]), "Not trusted for this metric")
        if spec.family in {"shape", "score"} and "sparseMaskMorphologyFiltered" in df:
            known = True
            reject(bool_values(df["sparseMaskMorphologyFiltered"]), "Insufficient mask support for morphology")
        for col in ("warningMaskFlag", "visualWarningOnly"):
            if col in df:
                warning |= bool_values(df[col])

    finite = candidate.notna()
    displayed = allowed & finite
    calculated = raw.notna() | finite
    status = pd.Series("Unavailable", index=df.index, dtype=object)
    status.loc[calculated & ~displayed] = "Excluded"
    status.loc[displayed] = "Accepted" if known else "QC unavailable"
    status.loc[displayed & warning & known] = "Warning (accepted)"
    reasons = reasons.mask(~finite & raw.notna(), reasons + "No finite QC-supported value; ")
    reasons = reasons.mask(~calculated, reasons + "No finite calculated value; ")
    reason_col = first_column(df, (f"{spec.key}_QC_reason", "score_QC_reason", "rbcScoreQCReason"))
    if spec.family != "image" and reason_col:
        extra = df[reason_col].fillna("").astype(str)
        relevant = (~displayed | warning) & extra.ne("")
        reasons = reasons.mask(relevant, reasons + extra + "; ")
    reasons = reasons.mask(status.eq("Accepted"), "Accepted under stored QC and frame-selection rules")
    reasons = reasons.mask(status.eq("QC unavailable"), "No metric-specific QC available; measured values, not QC-validated results")
    reasons = reasons.mask(status.eq("Warning (accepted)") & reasons.eq(""), "Warning only; retained under stored QC rules")
    view = pd.DataFrame({
        "value": candidate.where(displayed), "raw_value": raw,
        "status": status, "reason": reasons.str.rstrip("; "),
        "source_column": source or "unavailable", "raw_column": raw_col or "unavailable",
    }, index=df.index)
    view.attrs["metric"] = spec
    view.attrs["qc_available"] = known
    return view


def view_counts(view: pd.DataFrame) -> dict[str, int]:
    status = view["status"]
    return {"Frames": len(view), "Accepted": int(status.isin(["Accepted", "Warning (accepted)"]).sum()),
            "Warnings (included)": int(status.eq("Warning (accepted)").sum()),
            "Excluded": int(status.eq("Excluded").sum()), "Unavailable": int(status.eq("Unavailable").sum()),
            "QC unavailable": int(status.eq("QC unavailable").sum())}


def count_caption(view: pd.DataFrame) -> str:
    counts = view_counts(view)
    return (f"{counts['Accepted']}/{counts['Frames']} accepted (including {counts['Warnings (included)']} warnings)"
            f" · {counts['Excluded']} excluded · {counts['Unavailable']} unavailable"
            + (f" · {counts['QC unavailable']} measured without metric-specific QC" if counts['QC unavailable'] else ""))


def display_x(df: pd.DataFrame, x_axis: str) -> tuple[np.ndarray, str]:
    values = numeric(df, x_axis if x_axis in df else None)
    if values.notna().any():
        label = {"analysisIndex_1based": "Analysed frame number", "originalImageIndex_1based": "Original image number",
                 "time_s": "Time (s)", "matlabTime_s": "Time (s, stored axis)"}.get(x_axis, x_axis)
        return values.to_numpy(float), label
    return np.arange(1, len(df) + 1, dtype=float), "Frame position (requested axis unavailable)"


def draw_curve(ax, df: pd.DataFrame, view: pd.DataFrame, x_axis: str,
               show_excluded: bool = False, label: str = "Accepted", color: str = "#236685") -> str:
    """Keep full row positions/NaNs in the line; overlay warnings and raw exclusions."""
    x, x_label = display_x(df, x_axis)
    order = np.argsort(x, kind="stable")
    values = view["value"].to_numpy(float)
    status = view["status"].to_numpy()
    unassessed = status == "QC unavailable"
    # Both lines retain NaNs. A missing frame cannot become a connecting segment.
    if np.isfinite(np.where(~unassessed, values, np.nan)).any():
        ax.plot(x[order], np.where(~unassessed, values, np.nan)[order], "o-", color=color,
                markersize=3.5, linewidth=1.6, label=label)
    if unassessed.any():
        measured_label = "Measured (QC unavailable)" if label == "Accepted" else f"{label} (QC unavailable)"
        ax.plot(x[order], np.where(unassessed, values, np.nan)[order], "s:", color="#687582",
                markersize=3, label=measured_label)
    warning = status == "Warning (accepted)"
    if warning.any():
        ax.scatter(x[warning], values[warning], marker="^", s=42, color="#DAA520",
                   edgecolor="#725100", zorder=4, label="Warning (included)")
    if show_excluded:
        raw = view["raw_value"].to_numpy(float)
        excluded = (status == "Excluded") & np.isfinite(raw)
        if excluded.any():
            ax.scatter(x[excluded], raw[excluded], marker="o", s=24, facecolors="none",
                       edgecolors="#999999", alpha=.8, label="Excluded raw (diagnostic only)")
    return x_label


def plot_qc_metric(df: pd.DataFrame, spec: PlotMetric | str, x_axis: str,
                   show_excluded: bool = False, show_strip: bool = True):
    if isinstance(spec, str):
        spec = metric_spec(spec)
    view = metric_view(df, spec)
    if show_strip:
        fig, (ax, strip) = plt.subplots(2, 1, figsize=(7, 4.2), sharex=True, layout="constrained",
                                       gridspec_kw={"height_ratios": [7, .65], "hspace": .12})
    else:
        fig, ax = plt.subplots(figsize=(7, 3.9), layout="constrained")
        strip = None
    x_label = draw_curve(ax, df, view, x_axis, show_excluded)
    ax.set(title=spec.label, ylabel=spec.unit)
    if spec.family == "score":
        ax.set_ylim(0, 100)
    ax.grid(alpha=.2)
    if not view["value"].notna().any():
        ax.text(.5, .5, "No accepted values available\nSee QC details; missing does not mean zero.",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8, loc="best")
    if strip is not None:
        x, _ = display_x(df, x_axis)
        for status, color, marker in [("Accepted", "#236685", "|"), ("Warning (accepted)", "#DAA520", "^"),
                                      ("Excluded", "#999999", "x"), ("Unavailable", "#9B4352", "|"),
                                      ("QC unavailable", "#687582", "s")]:
            select = view["status"].eq(status).to_numpy()
            if select.any():
                strip.scatter(x[select], np.zeros(select.sum()), color=color, marker=marker, s=25)
        strip.set(yticks=[], ylabel="QC", xlabel=x_label, ylim=(-1, 1))
        for spine in ["left", "right", "top"]:
            strip.spines[spine].set_visible(False)
    else:
        ax.set_xlabel(x_label)
    return fig


def qc_details(df: pd.DataFrame, spec: PlotMetric | str) -> pd.DataFrame:
    view = metric_view(df, spec)
    identifiers = [col for col in ("sampleName", "filename", "analysisIndex_1based", "time_s", "matlabTime_s") if col in df]
    return pd.concat([df[identifiers].reset_index(drop=True), view.reset_index(drop=True)], axis=1)
