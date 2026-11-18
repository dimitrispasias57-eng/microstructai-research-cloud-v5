"""Matplotlib plotting functions for MicroStructAI MVP."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .qc_plotting import bool_values


def _clean_numeric(df: pd.DataFrame, metric: str) -> pd.Series:
    if metric not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df.get(metric), errors="coerce")


def _x_values(df: pd.DataFrame, x_axis: str = "analysisIndex_1based") -> tuple[np.ndarray | list[str], str, bool]:
    """Return x values, label, and whether labels are categorical filenames."""
    if x_axis == "filename" and "filename" in df.columns:
        return df["filename"].astype(str).tolist(), "filename", True
    if x_axis in df.columns:
        values = pd.to_numeric(df[x_axis], errors="coerce")
        if values.notna().any():
            fallback = pd.Series(np.arange(len(df)), index=df.index, dtype="float64")
            return values.where(values.notna(), fallback).to_numpy(), x_axis, False
    return np.arange(len(df)), "analysis index", False


def _format_x_axis(ax, x_values, x_label: str, categorical: bool) -> None:
    """Apply consistent x-axis labels for numeric index or filenames."""
    ax.set_xlabel(x_label)
    if categorical:
        positions = np.arange(len(x_values))
        ax.set_xticks(positions)
        ax.set_xticklabels(x_values, rotation=60, ha="right", fontsize=8)


def group_metrics_by_order_of_magnitude(df: pd.DataFrame, metrics: list[str]) -> list[tuple[str, list[str]]]:
    """Group metrics by their typical absolute order of magnitude for readable trend plots."""
    groups: dict[int, list[str]] = {}
    for metric in metrics:
        values = _clean_numeric(df, metric).replace([np.inf, -np.inf], np.nan).dropna().abs()
        if values.empty:
            continue
        nonzero = values.loc[values > 1e-12]
        order = 0 if nonzero.empty else int(math.floor(math.log10(float(nonzero.median()))))
        groups.setdefault(order, []).append(metric)
    return [(f"Typical scale: 10^{order}", groups[order]) for order in sorted(groups)]


def plot_metric_lines(
    df: pd.DataFrame,
    metrics: list[str],
    x_axis: str = "analysisIndex_1based",
    title: str = "Metric trends",
):
    """Plot selected metrics against image index."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_values, x_label, categorical = _x_values(df, x_axis)
    x = np.arange(len(df)) if categorical else x_values
    for metric in metrics:
        values = _clean_numeric(df, metric)
        if values.notna().any():
            ax.plot(x, values, marker="o", linewidth=1.8, label=metric)
    _format_x_axis(ax, x_values, x_label, categorical)
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if metrics:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_rbc_metric_lines_qc(
    df: pd.DataFrame,
    metrics: list[str],
    x_axis: str = "analysisIndex_1based",
    plot_frames: str = "Both, with excluded frames greyed out",
    title: str = "RBC metric trends with QC inclusion",
):
    """Plot RBC metrics while marking frames included/excluded from the main QC summary."""
    summary_col = "usedInSummary" if "usedInSummary" in df.columns else "usedForMainSummary"
    # Keep excluded row positions so the line breaks rather than bridging them.
    plot_df = df.copy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if plot_df.empty:
        ax.text(0.5, 0.5, "No frames available for selected plot option", ha="center", va="center")
        ax.axis("off")
        return fig

    x_values, x_label, categorical = _x_values(plot_df, x_axis)
    x = np.arange(len(plot_df)) if categorical else x_values
    score_plot = any(str(metric).endswith("_raw_0_100") or str(metric) in {"SASI_0_100", "AggIndex_0_100", "PTI_0_100", "GapScore_0_100"} for metric in metrics)
    if score_plot and "visualGreyExcluded" in plot_df.columns:
        included = ~plot_df["visualGreyExcluded"].fillna(True).astype(bool).to_numpy()
    elif score_plot and "visualQCExcluded" in plot_df.columns:
        included = ~plot_df["visualQCExcluded"].fillna(True).astype(bool).to_numpy()
    elif score_plot and "score_isTrusted" in plot_df.columns:
        included = plot_df["score_isTrusted"].fillna(False).astype(bool).to_numpy()
    elif summary_col in plot_df.columns:
        included = bool_values(plot_df[summary_col]).to_numpy()
    else:
        included = np.ones(len(plot_df), dtype=bool)
    if plot_frames == "Only frames used for main summary" and summary_col in plot_df.columns:
        included &= bool_values(plot_df[summary_col]).to_numpy()
    isolated = plot_df["isolatedReliableFrame"].fillna(False).astype(bool).to_numpy() if "isolatedReliableFrame" in plot_df.columns else np.zeros(len(plot_df), dtype=bool)
    reliable_outside = isolated & ~included
    warning_only = (
        plot_df["visualWarningOnly"].fillna(False).astype(bool).to_numpy()
        if "visualWarningOnly" in plot_df.columns
        else np.zeros(len(plot_df), dtype=bool)
    )
    qc_excluded = (~included) & ~reliable_outside
    show_excluded = plot_frames in {"All analysed frames", "Both, with excluded frames greyed out"}
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#245C6D", "#7E4E1F", "#3C7A57"])

    any_plotted = False
    for idx, metric in enumerate(metrics):
        values = _clean_numeric(plot_df, metric)
        if not values.notna().any():
            continue
        color = colors[idx % len(colors)]
        values_np = values.to_numpy(dtype=float)
        if plot_frames == "Both, with excluded frames greyed out":
            included_values = np.where(included, values_np, np.nan)
            isolated_values = np.where(reliable_outside, values_np, np.nan)
            excluded_values = np.where(qc_excluded, values_np, np.nan)
            warning_values = np.where(warning_only & included, values_np, np.nan)
            if np.isfinite(included_values).any():
                ax.plot(x, included_values, marker="o", linewidth=1.8, color=color, label=metric)
            if np.isfinite(isolated_values).any():
                ax.plot(x, isolated_values, marker="D", linestyle="None", color="#C77C2E", alpha=0.78)
            if np.isfinite(warning_values).any():
                ax.plot(
                    x,
                    warning_values,
                    marker="^",
                    linestyle="None",
                    markerfacecolor="#F4B942",
                    markeredgecolor="#6B5B00",
                    color="#F4B942",
                    alpha=0.9,
                )
            if np.isfinite(excluded_values).any():
                ax.plot(
                    x,
                    excluded_values,
                    marker="o",
                    linestyle="None",
                    markerfacecolor="none",
                    markeredgecolor="#8A8A8A",
                    color="#8A8A8A",
                    alpha=0.75,
                )
            any_plotted = True
        elif plot_frames == "All analysed frames" and show_excluded:
            ax.plot(x, values_np, marker="o", linewidth=1.5, color=color, alpha=0.9, label=metric)
            if reliable_outside.any():
                ax.scatter(np.asarray(x)[reliable_outside], values_np[reliable_outside], s=46, color="#C77C2E", marker="D", alpha=0.78, zorder=4)
            if warning_only.any():
                ax.scatter(
                    np.asarray(x)[warning_only & included],
                    values_np[warning_only & included],
                    s=48,
                    facecolors="#F4B942",
                    edgecolors="#6B5B00",
                    marker="^",
                    alpha=0.9,
                    zorder=5,
                )
            if qc_excluded.any():
                ax.scatter(
                    np.asarray(x)[qc_excluded],
                    values_np[qc_excluded],
                    s=44,
                    facecolors="none",
                    edgecolors="#8A8A8A",
                    marker="o",
                    alpha=0.8,
                    zorder=3,
                )
            any_plotted = True
        else:
            ax.plot(x, np.where(included, values_np, np.nan), marker="o", linewidth=1.8, color=color, label=metric)
            any_plotted = True

    if not any_plotted:
        ax.text(0.5, 0.5, "No finite values to plot", ha="center", va="center")
        ax.axis("off")
        return fig

    if plot_frames in {"All analysed frames", "Both, with excluded frames greyed out"} and "suggestedStableStartFrame" in plot_df.columns:
        stable_allowed = True
        if "stableReliableWindowDetected" in plot_df.columns and not plot_df["stableReliableWindowDetected"].fillna(False).astype(bool).any():
            stable_allowed = False
        if "reliableFramesUsedForMainSummary" in plot_df.columns:
            reliable_used = pd.to_numeric(plot_df["reliableFramesUsedForMainSummary"], errors="coerce").dropna()
            if reliable_used.empty:
                stable_allowed = False
            else:
                min_stable = 5.0
                if "minConsecutiveReliableFramesForStableWindow" in plot_df.columns:
                    min_values = pd.to_numeric(plot_df["minConsecutiveReliableFramesForStableWindow"], errors="coerce").dropna()
                    if not min_values.empty:
                        min_stable = float(min_values.iloc[0])
                if float(reliable_used.iloc[0]) < min_stable:
                    stable_allowed = False
        stable_values = pd.to_numeric(plot_df["suggestedStableStartFrame"], errors="coerce").dropna()
        if stable_allowed and not stable_values.empty and not categorical:
            stable_frame = float(stable_values.iloc[0])
            if x_axis == "analysisIndex_1based" and "suggestedStableStartAnalysisIndex" in plot_df.columns:
                alt_values = pd.to_numeric(plot_df["suggestedStableStartAnalysisIndex"], errors="coerce").dropna()
                if not alt_values.empty:
                    stable_frame = float(alt_values.iloc[0])
            ax.axvline(stable_frame, color="#4B7F52", linestyle="--", linewidth=1.2, alpha=0.75, label="Suggested stable start")

    if plot_frames in {"All analysed frames", "Both, with excluded frames greyed out"} and not categorical:
        auto_start = np.nan
        if x_axis in {"matlabTime_s", "time_s"} and "autoStartTime_s" in plot_df.columns:
            auto_values = pd.to_numeric(plot_df["autoStartTime_s"], errors="coerce").dropna()
            if not auto_values.empty:
                auto_start = float(auto_values.iloc[0])
        elif x_axis == "analysisIndex_1based" and "autoStartFrame_POD" in plot_df.columns:
            auto_values = pd.to_numeric(plot_df["autoStartFrame_POD"], errors="coerce").dropna()
            if not auto_values.empty:
                auto_start = float(auto_values.iloc[0])
        elif x_axis == "originalImageIndex_1based" and "autoStart_originalImageIndex" in plot_df.columns:
            auto_values = pd.to_numeric(plot_df["autoStart_originalImageIndex"], errors="coerce").dropna()
            if not auto_values.empty:
                auto_start = float(auto_values.iloc[0])
        elif "autoStartFrame_POD" in plot_df.columns:
            auto_values = pd.to_numeric(plot_df["autoStartFrame_POD"], errors="coerce").dropna()
            if not auto_values.empty:
                auto_start = float(auto_values.iloc[0])
        if np.isfinite(auto_start):
            ax.axvline(auto_start, color="#B33A3A", linestyle="-.", linewidth=1.3, alpha=0.85, label="Auto start frame")

    if plot_frames in {"All analysed frames", "Both, with excluded frames greyed out"} and ("usedInSummary" in plot_df.columns or "usedForMainSummary" in plot_df.columns or "score_isTrusted" in plot_df.columns):
        ax.scatter([], [], color="#245C6D", marker="o", label="Used in trusted analysis")
        ax.scatter([], [], facecolors="#F4B942", edgecolors="#6B5B00", marker="^", alpha=0.9, label="Warning only; still trusted")
        ax.scatter([], [], color="#C77C2E", marker="D", alpha=0.78, label="Passed frame QC but excluded from summary")
        ax.scatter([], [], facecolors="none", edgecolors="#8A8A8A", marker="o", alpha=0.8, label="Calculated but excluded from trusted analysis")

    _format_x_axis(ax, x_values, x_label, categorical)
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_rbc_spearman_correlation_matrix(
    df: pd.DataFrame,
    only_main_summary_frames: bool = False,
):
    """Plot an AggIndex-first Spearman matrix for important RBC Gap Mode metrics."""
    metric_labels = [
        ("AggIndex_0_100", "AggIndex"),
        ("SASI_0_100", "SASI"),
        ("GapScore_0_100", "GapScore"),
        ("PTI_0_100", "PTI"),
        ("S_struct_0_100", "S_struct"),
        ("SHI_tileCV", "SHI"),
        ("gapMeanDiam_px", "GapMeanDiam"),
        ("gapAreaFrac", "GapAreaFrac"),
        ("gapNtotal", "GapNtotal"),
        ("gapAreaSlope", "GapAreaSlope"),
        ("LF_log10_LFoverHF", "LF"),
        ("Xi_interp_px", "XiInterp"),
        ("Xi_px", "Xi"),
        ("lacunarity16", "Lacunarity16"),
        ("aggLargestFrac", "AggLargestFrac"),
        ("aggSkelLen", "AggSkelLen"),
        ("aggBranchPts", "AggBranchPts"),
        ("aggEndPts", "AggEndPts"),
        ("motionD", "Motion D"),
    ]
    plot_df = df.copy()
    summary_col = "usedInSummary" if "usedInSummary" in plot_df.columns else "usedForMainSummary"
    if only_main_summary_frames and summary_col in plot_df.columns:
        plot_df = plot_df.loc[plot_df[summary_col].fillna(False).astype(bool)].copy()

    available = [(metric, label) for metric, label in metric_labels if metric in plot_df.columns and _clean_numeric(plot_df, metric).notna().sum() >= 2]
    fig_size = max(7.5, 0.62 * len(available) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.82))
    if len(available) < 2:
        ax.text(0.5, 0.5, "At least two RBC metrics with finite values are required", ha="center", va="center")
        ax.axis("off")
        return fig

    metric_names = [metric for metric, _label in available]
    labels = [label for _metric, label in available]
    numeric = pd.DataFrame({metric: _clean_numeric(plot_df, metric) for metric in metric_names})
    corr = numeric.corr(method="spearman", min_periods=2)
    matrix = corr.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#E6E6E6")
    image = ax.imshow(masked, vmin=-1, vmax=1, cmap=cmap, aspect="equal")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=50, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Spearman correlation matrix - AggIndex-first ordering", fontweight="bold")

    for row in range(len(labels)):
        for col in range(len(labels)):
            value = matrix[row, col]
            if not np.isfinite(value):
                continue
            text_color = "white" if value < -0.35 or value > 0.72 else "black"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=7.5)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label("Spearman rho")
    fig.tight_layout()
    return fig


def plot_histograms(df: pd.DataFrame, metrics: list[str]):
    """Plot compact histograms for selected metrics."""
    metrics = [m for m in metrics if _clean_numeric(df, m).notna().any()]
    if not metrics:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No finite values to plot", ha="center", va="center")
        ax.axis("off")
        return fig

    ncols = min(3, len(metrics))
    nrows = int(math.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, metric in zip(axes_arr, metrics):
        values = _clean_numeric(df, metric).dropna()
        ax.hist(values, bins=min(20, max(5, len(values))), color="#4267B2", alpha=0.8)
        ax.set_title(metric)
        ax.grid(True, alpha=0.2)
    for ax in axes_arr[len(metrics) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_group_boxplot(df: pd.DataFrame, metric: str, label_col: str = "groupLabel"):
    """Plot a metric as boxplots grouped by a label column."""
    plot_df = df[[metric, label_col]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric, label_col])
    fig, ax = plt.subplots(figsize=(7, 4))
    if plot_df.empty:
        ax.text(0.5, 0.5, "No grouped data to plot", ha="center", va="center")
        ax.axis("off")
        return fig

    groups = [g for g in sorted(plot_df[label_col].astype(str).unique())]
    values = [plot_df.loc[plot_df[label_col].astype(str) == g, metric].values for g in groups]
    ax.boxplot(values, labels=groups, showmeans=True)
    ax.set_xlabel("Group")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by group")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_pca_scatter(scores: pd.DataFrame, explained: pd.Series):
    """Plot PC1 vs PC2 scores."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    if scores.empty or "PC1" not in scores:
        ax.text(0.5, 0.5, "PCA not available", ha="center", va="center")
        ax.axis("off")
        return fig

    pc2 = scores["PC2"] if "PC2" in scores else np.zeros(len(scores))
    ax.scatter(scores["PC1"], pc2, s=48, color="#245C6D", alpha=0.85)
    for _, row in scores.iterrows():
        label_value = row.get("analysisIndex_1based", row.get("imageIndex", ""))
        ax.annotate(str(int(label_value)) if pd.notna(label_value) and str(label_value).replace(".", "", 1).isdigit() else str(label_value), (row["PC1"], row["PC2"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    pc1_var = explained.get("PC1", np.nan) * 100
    pc2_var = explained.get("PC2", np.nan) * 100
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}% var)" if np.isfinite(pc1_var) else "PC1")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}% var)" if np.isfinite(pc2_var) else "PC2")
    ax.set_title("PCA scatter")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.25)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.25)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_pc1_over_index(scores: pd.DataFrame, x_axis: str = "analysisIndex_1based"):
    """Plot PC1 score as a function of image index."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    if scores.empty or "PC1" not in scores:
        ax.text(0.5, 0.5, "PCA not available", ha="center", va="center")
        ax.axis("off")
        return fig
    x_values, x_label, categorical = _x_values(scores, x_axis)
    x = np.arange(len(scores)) if categorical else x_values
    ax.plot(x, scores["PC1"], marker="o", color="#7E4E1F", linewidth=1.8)
    _format_x_axis(ax, x_values, x_label, categorical)
    ax.set_ylabel("PC1 score")
    ax.set_title("PC1 trend")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_pc1_loadings(loadings: pd.DataFrame):
    """Plot PC1 loadings as a horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(loadings))))
    if loadings.empty or "PC1" not in loadings:
        ax.text(0.5, 0.5, "PCA loadings not available", ha="center", va="center")
        ax.axis("off")
        return fig
    ordered = loadings["PC1"].sort_values()
    colors = ["#A34E4E" if v < 0 else "#3C7A57" for v in ordered]
    ax.barh(ordered.index, ordered.values, color=colors, alpha=0.9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("PC1 loading")
    ax.set_title("PC1 metric loadings")
    fig.tight_layout()
    return fig


def plot_gt_summary(df: pd.DataFrame, x_axis: str = "analysisIndex_1based"):
    """Plot key ground-truth morphology metrics across image index."""
    metrics = ["GT_nObjects", "GT_meanArea", "GT_meanDiam", "GT_Xi_px", "GT_lacunarity16", "GT_skelDensity"]
    return plot_metric_lines(df, [metric for metric in metrics if metric in df.columns], x_axis=x_axis)


def plot_segmentation_quality(df: pd.DataFrame, x_axis: str = "analysisIndex_1based"):
    """Plot Dice and IoU bars for automatic segmentation quality."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    x_values, x_label, categorical = _x_values(df, x_axis)
    x = np.arange(len(df)) if categorical else x_values
    for ax, metric, title in zip(axes, ["Dice_auto_vs_GT", "IoU_auto_vs_GT"], ["Dice auto vs GT", "IoU auto vs GT"]):
        values = pd.to_numeric(df.get(metric), errors="coerce")
        if values.notna().any():
            ax.bar(x, values, color="#3C7A57", alpha=0.85)
            ax.set_ylim(0, 1)
            _format_x_axis(ax, x_values, x_label, categorical)
            ax.set_title(title)
            ax.grid(True, axis="y", alpha=0.25)
        else:
            ax.text(0.5, 0.5, f"{metric} not available", ha="center", va="center")
            ax.axis("off")
    axes[0].set_ylabel("Score")
    fig.tight_layout()
    return fig


def _metric_comparison_values(comparison_long: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return finite paired app/MATLAB values for one metric."""
    if comparison_long.empty:
        return pd.DataFrame()
    data = comparison_long.loc[comparison_long["metric"] == metric].copy()
    for col in ["frameIndex", "appValue", "matlabValue", "difference"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.replace([np.inf, -np.inf], np.nan).dropna(subset=["appValue", "matlabValue"])


def plot_matlab_scatter(comparison_long: pd.DataFrame, metric: str):
    """Plot app vs MATLAB values with a y=x reference line."""
    data = _metric_comparison_values(comparison_long, metric)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    if data.empty:
        ax.text(0.5, 0.5, f"No paired values for {metric}", ha="center", va="center")
        ax.axis("off")
        return fig

    ax.scatter(data["matlabValue"], data["appValue"], s=34, color="#245C6D", alpha=0.82)
    lo = float(np.nanmin([data["matlabValue"].min(), data["appValue"].min()]))
    hi = float(np.nanmax([data["matlabValue"].max(), data["appValue"].max()]))
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, alpha=0.55)
    ax.set_xlabel("MATLAB")
    ax.set_ylabel("MicroStructAI")
    ax.set_title(f"{metric}: app vs MATLAB")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_matlab_difference(comparison_long: pd.DataFrame, metric: str):
    """Plot app-MATLAB difference across frame/index."""
    data = _metric_comparison_values(comparison_long, metric).dropna(subset=["difference", "frameIndex"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    if data.empty:
        ax.text(0.5, 0.5, f"No paired differences for {metric}", ha="center", va="center")
        ax.axis("off")
        return fig

    ax.plot(data["frameIndex"], data["difference"], marker="o", color="#A34E4E", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.45)
    ax.set_xlabel("Frame/index")
    ax.set_ylabel("App - MATLAB")
    ax.set_title(f"{metric}: difference vs frame/index")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_matlab_trend(comparison_long: pd.DataFrame, metric: str):
    """Plot app and MATLAB values across frame/index."""
    data = _metric_comparison_values(comparison_long, metric).dropna(subset=["frameIndex"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    if data.empty:
        ax.text(0.5, 0.5, f"No paired trends for {metric}", ha="center", va="center")
        ax.axis("off")
        return fig

    ax.plot(data["frameIndex"], data["appValue"], marker="o", color="#245C6D", linewidth=1.6, label="MicroStructAI")
    ax.plot(data["frameIndex"], data["matlabValue"], marker="s", color="#7E4E1F", linewidth=1.4, label="MATLAB")
    ax.set_xlabel("Frame/index")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric}: app and MATLAB over frame/index")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_matlab_spearman_matrix_comparison(paired_df: pd.DataFrame):
    """Plot app, MATLAB, and difference Spearman matrices for matched RBC frames."""
    metric_labels = [
        ("AggIndex_0_100", "AggIndex"),
        ("SASI_0_100", "SASI"),
        ("GapScore_0_100", "GapScore"),
        ("PTI_0_100", "PTI"),
        ("SHI_tileCV", "SHI"),
        ("gapMeanDiam_px", "GapMeanDiam"),
        ("gapAreaFrac", "GapAreaFrac"),
        ("gapNtotal", "GapNtotal"),
        ("gapAreaSlope", "GapAreaSlope"),
        ("LF_log10_LFoverHF", "LF"),
        ("Xi_interp_px", "XiInterp"),
    ]
    available = [
        (metric, label)
        for metric, label in metric_labels
        if f"{metric}_app" in paired_df.columns
        and f"{metric}_MATLAB" in paired_df.columns
        and _clean_numeric(paired_df, f"{metric}_app").notna().sum() >= 2
        and _clean_numeric(paired_df, f"{metric}_MATLAB").notna().sum() >= 2
    ]
    fig_width = max(14.0, 1.45 * len(available) + 5.0)
    fig_height = max(5.5, 0.70 * len(available) + 2.2)
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
    if len(available) < 2:
        for ax in axes:
            ax.text(0.5, 0.5, "At least two shared paired RBC metrics are required", ha="center", va="center")
            ax.axis("off")
        fig.tight_layout()
        return fig

    metric_names = [metric for metric, _label in available]
    labels = [label for _metric, label in available]
    app_values = pd.DataFrame({metric: _clean_numeric(paired_df, f"{metric}_app") for metric in metric_names})
    matlab_values = pd.DataFrame({metric: _clean_numeric(paired_df, f"{metric}_MATLAB") for metric in metric_names})
    app_corr = app_values.corr(method="spearman", min_periods=2)
    matlab_corr = matlab_values.corr(method="spearman", min_periods=2)
    difference = app_corr - matlab_corr
    matrices = [
        ("MicroStructAI Spearman rho", app_corr.to_numpy(dtype=float), -1.0, 1.0, "viridis"),
        ("MATLAB Spearman rho", matlab_corr.to_numpy(dtype=float), -1.0, 1.0, "viridis"),
        ("Difference: app - MATLAB", difference.to_numpy(dtype=float), -1.0, 1.0, "coolwarm"),
    ]

    for ax, (title, matrix, vmin, vmax, cmap_name) in zip(axes, matrices):
        masked = np.ma.masked_invalid(matrix)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#E6E6E6")
        image = ax.imshow(masked, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontweight="bold")
        for row in range(len(labels)):
            for col in range(len(labels)):
                value = matrix[row, col]
                if not np.isfinite(value):
                    continue
                if cmap_name == "coolwarm":
                    text_color = "white" if abs(value) > 0.55 else "black"
                else:
                    text_color = "white" if value < -0.35 or value > 0.72 else "black"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=6.8)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
        colorbar.ax.tick_params(labelsize=8)

    fig.suptitle("Matched-frame RBC correlation structure: MicroStructAI vs MATLAB", fontweight="bold")
    fig.tight_layout()
    return fig
