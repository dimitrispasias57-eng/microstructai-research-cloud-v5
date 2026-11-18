"""PCA utilities for MicroStructAI MVP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAResult:
    """Container for PCA outputs used by the app and report."""

    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance: pd.Series
    used_metrics: list[str]
    warning: str | None = None


def finite_numeric_columns(df: pd.DataFrame, min_finite: int = 5) -> list[str]:
    """Return numeric columns with enough finite values for PCA."""
    columns: list[str] = []
    for col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() >= min_finite and values.nunique(dropna=True) > 1:
            columns.append(col)
    return columns


def run_pca(
    metrics_df: pd.DataFrame,
    selected_metrics: list[str],
    min_images: int = 5,
    flip_metric: str | None = None,
    flip_direction: str = "positive",
) -> PCAResult:
    """Run standardized PCA on selected finite metric columns."""
    if metrics_df.empty:
        return PCAResult(pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), [], "No metrics available.")

    selected_metrics = [m for m in selected_metrics if m in metrics_df.columns]
    if len(selected_metrics) < 2:
        return PCAResult(pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), [], "Select at least two metrics.")

    data = metrics_df[selected_metrics].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    data = data.loc[:, data.nunique(dropna=True) > 1]

    if data.shape[0] < min_images:
        return PCAResult(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.Series(dtype=float),
            list(data.columns),
            f"PCA needs at least {min_images} complete images; found {data.shape[0]}.",
        )
    if data.shape[1] < 2:
        return PCAResult(pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), list(data.columns), "PCA needs at least two variable metrics.")

    scaled = StandardScaler().fit_transform(data.values)
    n_components = min(2, scaled.shape[0], scaled.shape[1])
    pca = PCA(n_components=n_components)
    scores_arr = pca.fit_transform(scaled)

    scores = pd.DataFrame(index=data.index)
    scores["filename"] = metrics_df.loc[data.index, "filename"].values if "filename" in metrics_df.columns else data.index.astype(str)
    if "imageIndex" in metrics_df.columns:
        scores["imageIndex"] = pd.to_numeric(metrics_df.loc[data.index, "imageIndex"], errors="coerce").fillna(data.index.to_series()).values
    else:
        scores["imageIndex"] = data.index.to_numpy()
    for index_col in [
        "originalImageIndex_0based",
        "originalImageIndex_1based",
        "analysisIndex_0based",
        "analysisIndex_1based",
    ]:
        if index_col in metrics_df.columns:
            scores[index_col] = pd.to_numeric(metrics_df.loc[data.index, index_col], errors="coerce").values
    scores["PC1"] = scores_arr[:, 0]
    if n_components > 1:
        scores["PC2"] = scores_arr[:, 1]
    else:
        scores["PC2"] = np.nan

    loadings = pd.DataFrame(index=data.columns)
    loadings["PC1"] = pca.components_[0, :]
    if n_components > 1:
        loadings["PC2"] = pca.components_[1, :]
    else:
        loadings["PC2"] = np.nan

    explained = pd.Series(pca.explained_variance_ratio_, index=[f"PC{i + 1}" for i in range(n_components)])
    if "PC2" not in explained.index:
        explained.loc["PC2"] = np.nan

    desired_sign = 1 if flip_direction == "positive" else -1
    if flip_metric and flip_metric in loadings.index:
        current = loadings.loc[flip_metric, "PC1"]
        if np.isfinite(current) and current != 0 and np.sign(current) != desired_sign:
            scores["PC1"] = -scores["PC1"]
            loadings["PC1"] = -loadings["PC1"]

    return PCAResult(scores=scores, loadings=loadings, explained_variance=explained, used_metrics=list(data.columns))
