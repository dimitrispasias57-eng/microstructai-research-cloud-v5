"""Research-grade static morphology for the trusted MicroStructAI gap mask.

The white/True phase is always interpreted as plasma gap / void.  Nothing in
this module creates or changes the authoritative segmentation used by the
existing application.  The functions consume an already-selected final mask
and return separate, versioned research descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, pi, sqrt
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from skimage import measure, morphology


STATIC_ALGORITHM_VERSION = "3.1.0-gap-static"


@dataclass(frozen=True)
class StaticConfig:
    connectivity: int = 2  # skimage: 2 means 8-connected in 2-D
    min_object_area_px: int = 20
    min_network_object_area_px: int = 20
    granulometry_radii_px: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12)
    topology_scales_px: tuple[int, ...] = (-4, -2, -1, 0, 1, 2, 4)
    grid_sizes: tuple[int, ...] = (2, 4, 8)
    contour_min_area_px: int = 100
    contour_points: int = 128
    contour_smoothing_sigma: float = 1.0
    contour_max_mode: int = 10
    max_contours_per_frame: int = 5
    max_graph_pixels: int = 12000


def as_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Return a finite 2-D boolean mask or raise a clear input error."""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError("Trusted gap mask must be a 2-D array.")
    if arr.size == 0:
        raise ValueError("Trusted gap mask is empty.")
    return np.nan_to_num(arr, nan=0.0).astype(bool, copy=False)


def safe_percentile(values: np.ndarray | Iterable[float], percentile: float) -> float:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, percentile)) if arr.size else np.nan


def _safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def _safe_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=1)) if arr.size >= 2 else np.nan


def _safe_cv(values: Iterable[float]) -> float:
    mean = _safe_mean(values)
    std = _safe_std(values)
    return float(std / abs(mean)) if np.isfinite(mean) and abs(mean) > 1e-12 and np.isfinite(std) else np.nan


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    if coords.size == 0:
        return False
    rows, cols = coords[:, 0], coords[:, 1]
    return bool((rows == 0).any() or (cols == 0).any() or (rows == shape[0] - 1).any() or (cols == shape[1] - 1).any())


def gap_object_features(mask: np.ndarray, config: StaticConfig | None = None) -> pd.DataFrame:
    """Describe every valid 8-connected plasma-gap component.

    Border-touching components remain in the all-object population and are
    explicitly marked so shape summaries can also use a fully observed subset.
    """
    cfg = config or StaticConfig()
    binary = as_binary_mask(mask)
    labels = measure.label(binary, connectivity=cfg.connectivity)
    centre_y = (binary.shape[0] - 1) / 2.0
    centre_x = (binary.shape[1] - 1) / 2.0
    diag = max(float(np.hypot(binary.shape[0], binary.shape[1])), 1.0)
    rows: list[dict[str, Any]] = []
    for region in measure.regionprops(labels):
        if region.area < cfg.min_object_area_px:
            continue
        perimeter = float(region.perimeter)
        area = float(region.area)
        major = float(region.axis_major_length)
        minor = float(region.axis_minor_length)
        centroid_y, centroid_x = (float(region.centroid[0]), float(region.centroid[1]))
        border = _touches_border(region.coords, binary.shape)
        feret = getattr(region, "feret_diameter_max", np.nan)
        circularity = float(4.0 * pi * area / (perimeter * perimeter)) if perimeter > 0 else np.nan
        rows.append(
            {
                "gapObjectID": int(region.label),
                "area_px2": area,
                "perimeter_px": perimeter,
                "equivalentDiameter_px": float(region.equivalent_diameter_area),
                "majorAxisLength_px": major,
                "minorAxisLength_px": minor,
                "aspectRatio": float(major / minor) if minor > 1e-12 else np.nan,
                "eccentricity": float(region.eccentricity),
                "circularity": circularity,
                "solidity": float(region.solidity),
                "convexArea_px2": float(region.area_convex),
                "extent": float(region.extent),
                "orientation_deg": float(np.degrees(region.orientation)),
                "centroidY_px": centroid_y,
                "centroidX_px": centroid_x,
                "perimeterToArea_px_inv": float(perimeter / area) if area > 0 else np.nan,
                "feretDiameterMax_px": float(feret) if np.isfinite(feret) else np.nan,
                "distanceFromROICentre_px": float(np.hypot(centroid_y - centre_y, centroid_x - centre_x)),
                "distanceFromROICentre_normalized": float(np.hypot(centroid_y - centre_y, centroid_x - centre_x) / diag),
                "touchesROIBorder": border,
                "fullyObserved": not border,
                "population": "fully_observed" if not border else "border_truncated",
                "algorithmVersion": STATIC_ALGORITHM_VERSION,
            }
        )
    return pd.DataFrame(rows)


def _top_area_fraction(areas: np.ndarray, fraction: float) -> float:
    if areas.size == 0 or float(np.sum(areas)) <= 0:
        return np.nan
    count = max(1, int(np.ceil(float(fraction) * len(areas))))
    return float(np.sum(np.sort(areas)[-count:]) / np.sum(areas))


def _orientation_order(angles_deg: np.ndarray) -> tuple[float, float]:
    angles = np.deg2rad(angles_deg[np.isfinite(angles_deg)])
    if angles.size == 0:
        return np.nan, np.nan
    vector = np.mean(np.exp(2j * angles))
    order = float(abs(vector))
    dispersion = float(1.0 - order)
    return order, dispersion


def gap_frame_summary(mask: np.ndarray, objects: pd.DataFrame | None = None, config: StaticConfig | None = None) -> dict[str, Any]:
    cfg = config or StaticConfig()
    binary = as_binary_mask(mask)
    obj = gap_object_features(binary, cfg) if objects is None else objects
    full = obj.loc[obj.get("fullyObserved", pd.Series(False, index=obj.index)).fillna(False).astype(bool)].copy() if not obj.empty else obj
    areas = pd.to_numeric(obj.get("area_px2", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    full_areas = pd.to_numeric(full.get("area_px2", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    orient = pd.to_numeric(full.get("orientation_deg", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    order, dispersion = _orientation_order(orient)
    total_gap = float(binary.sum())
    image_area = float(binary.size)
    summary: dict[str, Any] = {
        "gapAreaFraction": float(np.mean(binary)),
        "gapObjectCount_all": int(len(obj)),
        "gapObjectCount_fullyObserved": int(len(full)),
        "borderTouchingObjectCount": int(len(obj) - len(full)),
        "borderTouchingAreaFractionOfGap": float(obj.loc[~obj["fullyObserved"], "area_px2"].sum() / total_gap) if total_gap > 0 and not obj.empty else 0.0,
        "gapAreaMean_px2": _safe_mean(areas),
        "gapAreaMedian_px2": safe_percentile(areas, 50),
        "gapAreaSD_px2": _safe_std(areas),
        "gapAreaCV": _safe_cv(areas),
        "gapAreaP10_px2": safe_percentile(areas, 10),
        "gapAreaP25_px2": safe_percentile(areas, 25),
        "gapAreaP50_px2": safe_percentile(areas, 50),
        "gapAreaP75_px2": safe_percentile(areas, 75),
        "gapAreaP90_px2": safe_percentile(areas, 90),
        "gapAreaP95_px2": safe_percentile(areas, 95),
        "gapAreaMaximum_px2": float(np.max(areas)) if areas.size else np.nan,
        "largestGapFractionOfImage": float(np.max(areas) / image_area) if areas.size else 0.0,
        "largestGapFractionOfGapArea": float(np.max(areas) / total_gap) if areas.size and total_gap > 0 else 0.0,
        "top1PercentObjectsAreaFraction": _top_area_fraction(areas, 0.01),
        "top5PercentObjectsAreaFraction": _top_area_fraction(areas, 0.05),
        "top10PercentObjectsAreaFraction": _top_area_fraction(areas, 0.10),
        "fullyObservedGapAreaMedian_px2": safe_percentile(full_areas, 50),
        "medianEccentricity_fullyObserved": safe_percentile(pd.to_numeric(full.get("eccentricity", pd.Series(dtype=float)), errors="coerce").to_numpy(), 50),
        "medianCircularity_fullyObserved": safe_percentile(pd.to_numeric(full.get("circularity", pd.Series(dtype=float)), errors="coerce").to_numpy(), 50),
        "medianSolidity_fullyObserved": safe_percentile(pd.to_numeric(full.get("solidity", pd.Series(dtype=float)), errors="coerce").to_numpy(), 50),
        "medianPerimeterToArea_px_inv_fullyObserved": safe_percentile(pd.to_numeric(full.get("perimeterToArea_px_inv", pd.Series(dtype=float)), errors="coerce").to_numpy(), 50),
        "orientationOrder": order,
        "orientationDispersion": dispersion,
        "geometryUnits": "pixels and pixels^2; no physical calibration applied",
        "algorithmVersion": STATIC_ALGORITHM_VERSION,
    }
    return summary


def gap_size_spectrum(objects: pd.DataFrame, roi_area_px2: int, max_bins: int = 24) -> pd.DataFrame:
    """Return adaptive log-area bins with object and occupied-area fractions."""
    if objects.empty or "area_px2" not in objects:
        return pd.DataFrame(columns=["binLower_px2", "binUpper_px2", "objectCount", "objectFraction", "gapAreaFraction", "cumulativeObjectFraction", "cumulativeGapAreaFraction"])
    areas = pd.to_numeric(objects["area_px2"], errors="coerce").dropna().to_numpy(dtype=float)
    areas = areas[areas > 0]
    if areas.size == 0:
        return pd.DataFrame()
    logs = np.log10(areas)
    if areas.size < 3 or float(np.ptp(logs)) < 1e-12:
        edges = np.array([logs.min() - 0.01, logs.max() + 0.01])
    else:
        q25, q75 = np.percentile(logs, [25, 75])
        width = 2.0 * (q75 - q25) / np.cbrt(areas.size)
        bins = int(np.ceil(np.ptp(logs) / width)) if width > 1e-12 else int(np.ceil(np.sqrt(areas.size)))
        bins = int(np.clip(bins, 3, max_bins))
        edges = np.linspace(logs.min(), logs.max() + np.finfo(float).eps, bins + 1)
    counts, _ = np.histogram(logs, bins=edges)
    rows: list[dict[str, Any]] = []
    total_area = float(np.sum(areas))
    cumulative_count = 0
    cumulative_area = 0.0
    for idx in range(len(edges) - 1):
        lower, upper = 10.0 ** edges[idx], 10.0 ** edges[idx + 1]
        select = (areas >= lower) & ((areas < upper) if idx < len(edges) - 2 else (areas <= upper))
        bin_area = float(np.sum(areas[select]))
        cumulative_count += int(counts[idx])
        cumulative_area += bin_area
        rows.append(
            {
                "binIndex": idx + 1,
                "binLower_px2": float(lower),
                "binUpper_px2": float(upper),
                "objectCount": int(counts[idx]),
                "objectFraction": float(counts[idx] / len(areas)),
                "gapAreaInBin_px2": bin_area,
                "gapAreaFraction": float(bin_area / total_area) if total_area > 0 else np.nan,
                "roiAreaFraction": float(bin_area / max(int(roi_area_px2), 1)),
                "cumulativeObjectFraction": float(cumulative_count / len(areas)),
                "cumulativeGapAreaFraction": float(cumulative_area / total_area) if total_area > 0 else np.nan,
                "binning": "Freedman-Diaconis on log10(area), bounded to 3-24 bins",
                "algorithmVersion": STATIC_ALGORITHM_VERSION,
            }
        )
    return pd.DataFrame(rows)


def local_thickness_analysis(mask: np.ndarray) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    """Measure EDT-derived local gap radius and thickness (2 * radius)."""
    binary = as_binary_mask(mask)
    radius_map = ndi.distance_transform_edt(binary).astype(np.float32)
    thickness_map = (2.0 * radius_map).astype(np.float32)
    radii = radius_map[binary]
    thickness = thickness_map[binary]
    if thickness.size:
        narrow_threshold = float(np.percentile(thickness, 25))
        broad_threshold = float(np.percentile(thickness, 75))
    else:
        narrow_threshold = broad_threshold = np.nan
    summary = {
        "localRadiusMean_px": _safe_mean(radii),
        "localRadiusMedian_px": safe_percentile(radii, 50),
        "localRadiusP10_px": safe_percentile(radii, 10),
        "localRadiusP25_px": safe_percentile(radii, 25),
        "localRadiusP50_px": safe_percentile(radii, 50),
        "localRadiusP75_px": safe_percentile(radii, 75),
        "localRadiusP90_px": safe_percentile(radii, 90),
        "localRadiusMaximum_px": float(np.max(radii)) if radii.size else np.nan,
        "localThicknessMean_px": _safe_mean(thickness),
        "localThicknessMedian_px": safe_percentile(thickness, 50),
        "localThicknessP10_px": safe_percentile(thickness, 10),
        "localThicknessP25_px": safe_percentile(thickness, 25),
        "localThicknessP50_px": safe_percentile(thickness, 50),
        "localThicknessP75_px": safe_percentile(thickness, 75),
        "localThicknessP90_px": safe_percentile(thickness, 90),
        "localThicknessMaximum_px": float(np.max(thickness)) if thickness.size else np.nan,
        "localThicknessNormalizedMedian": float(np.median(thickness) / max(mask.shape)) if thickness.size else np.nan,
        "narrowGapThreshold_px": narrow_threshold,
        "broadGapThreshold_px": broad_threshold,
        "narrowGapPixelFraction": float(np.mean(thickness <= narrow_threshold)) if thickness.size else np.nan,
        "broadGapPixelFraction": float(np.mean(thickness >= broad_threshold)) if thickness.size else np.nan,
        "thicknessConvention": "Euclidean distance transform radius multiplied by 2",
        "thresholdConvention": "within-frame P25/P75 of gap-pixel thickness",
        "algorithmVersion": STATIC_ALGORITHM_VERSION,
    }
    if thickness.size:
        bin_count = int(np.clip(np.ceil(np.sqrt(thickness.size)), 8, 48))
        counts, edges = np.histogram(thickness, bins=bin_count)
        distribution = pd.DataFrame(
            {
                "binLowerThickness_px": edges[:-1],
                "binUpperThickness_px": edges[1:],
                "gapPixelCount": counts,
                "gapPixelFraction": counts / max(int(counts.sum()), 1),
            }
        )
    else:
        distribution = pd.DataFrame(columns=["binLowerThickness_px", "binUpperThickness_px", "gapPixelCount", "gapPixelFraction"])
    return summary, thickness_map, distribution


def granulometry(mask: np.ndarray, radii_px: Iterable[int]) -> pd.DataFrame:
    binary = as_binary_mask(mask)
    original_area = float(binary.sum())
    rows = [{"radius_px": 0, "remainingGapFraction": 1.0 if original_area > 0 else np.nan, "removedGapFraction": 0.0 if original_area > 0 else np.nan}]
    previous_remaining = original_area
    for radius in sorted({max(1, int(value)) for value in radii_px}):
        opened = morphology.binary_opening(binary, morphology.disk(radius))
        remaining = float(opened.sum())
        rows.append(
            {
                "radius_px": radius,
                "remainingGapFraction": float(remaining / original_area) if original_area > 0 else np.nan,
                "removedGapFraction": float(1.0 - remaining / original_area) if original_area > 0 else np.nan,
                "incrementalRemovedFraction": float((previous_remaining - remaining) / original_area) if original_area > 0 else np.nan,
                "structuringElement": "binary disk",
                "algorithmVersion": STATIC_ALGORITHM_VERSION,
            }
        )
        previous_remaining = remaining
    return pd.DataFrame(rows)


def granulometry_summary(table: pd.DataFrame) -> dict[str, Any]:
    if table.empty or table["remainingGapFraction"].dropna().empty:
        return {"dominantGranulometricRadius_px": np.nan, "granulometryAUC": np.nan}
    work = table.dropna(subset=["remainingGapFraction"]).sort_values("radius_px")
    increments = pd.to_numeric(work.get("incrementalRemovedFraction", pd.Series(np.nan, index=work.index)), errors="coerce")
    valid = increments.dropna()
    dominant = float(work.loc[valid.idxmax(), "radius_px"]) if not valid.empty else np.nan
    auc = float(np.trapezoid(work["remainingGapFraction"], work["radius_px"])) if len(work) >= 2 else np.nan
    return {"dominantGranulometricRadius_px": dominant, "granulometryAUC": auc, "algorithmVersion": STATIC_ALGORITHM_VERSION}


def _run_lengths(vector: np.ndarray) -> np.ndarray:
    # Use a signed type so falling edges are represented as -1 rather than
    # wrapping to 255.
    padded = np.pad(np.asarray(vector, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return (ends - starts).astype(float)


def chord_length_analysis(mask: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    binary = as_binary_mask(mask)
    horizontal = np.concatenate([_run_lengths(row) for row in binary]) if binary.shape[0] else np.array([], dtype=float)
    vertical = np.concatenate([_run_lengths(col) for col in binary.T]) if binary.shape[1] else np.array([], dtype=float)
    rows = []
    for orientation, values in (("horizontal", horizontal), ("vertical", vertical)):
        if values.size == 0:
            continue
        maximum = max(1.0, float(np.max(values)))
        # A bounded log-spaced distribution preserves the chord-size spectrum
        # without exporting one provenance-heavy CSV row per raster run.
        edge_count = min(33, max(3, int(np.ceil(np.log2(maximum + 1))) * 3))
        edges = np.unique(np.rint(np.geomspace(1.0, maximum + 1.0, edge_count)).astype(int)).astype(float)
        edges = np.unique(np.concatenate([[0.5], edges - 0.5, [maximum + 0.5]]))
        counts, bin_edges = np.histogram(values, bins=edges)
        for lower, upper, count in zip(bin_edges[:-1], bin_edges[1:], counts):
            if int(count) == 0:
                continue
            rows.append(
                {
                    "orientation": orientation,
                    "chordLengthLower_px": float(max(0.0, lower)),
                    "chordLengthUpper_px": float(upper),
                    "chordLengthMidpoint_px": float((lower + upper) / 2.0),
                    "chordCount": int(count),
                    "chordProbability": float(count / len(values)),
                    "distribution": "bounded log-spaced histogram of complete raster-run chord lengths",
                }
            )
    mean_h, mean_v = _safe_mean(horizontal), _safe_mean(vertical)
    anisotropy = float(abs(mean_h - mean_v) / (mean_h + mean_v)) if np.isfinite(mean_h) and np.isfinite(mean_v) and (mean_h + mean_v) > 0 else np.nan
    summary = {
        "horizontalChordCount": int(len(horizontal)),
        "horizontalChordMean_px": mean_h,
        "horizontalChordMedian_px": safe_percentile(horizontal, 50),
        "horizontalChordP90_px": safe_percentile(horizontal, 90),
        "verticalChordCount": int(len(vertical)),
        "verticalChordMean_px": mean_v,
        "verticalChordMedian_px": safe_percentile(vertical, 50),
        "verticalChordP90_px": safe_percentile(vertical, 90),
        "chordLengthAnisotropy": anisotropy,
        "algorithmVersion": STATIC_ALGORITHM_VERSION,
    }
    return summary, pd.DataFrame(rows)


def topology_features(mask: np.ndarray, connectivity: int = 2) -> dict[str, Any]:
    binary = as_binary_mask(mask)
    labels = measure.label(binary, connectivity=connectivity)
    components = int(labels.max())
    inverse_labels = measure.label(~binary, connectivity=connectivity)
    border_labels = np.unique(np.concatenate([inverse_labels[0], inverse_labels[-1], inverse_labels[:, 0], inverse_labels[:, -1]]))
    all_inverse = np.unique(inverse_labels)
    holes = int(len([value for value in all_inverse if value != 0 and value not in set(border_labels.tolist())]))
    areas = np.bincount(labels.ravel())[1:].astype(float) if components else np.array([], dtype=float)
    total = float(areas.sum())
    probabilities = areas / total if total > 0 else np.array([], dtype=float)
    entropy = float(-np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))) if probabilities.size else np.nan
    normalized_entropy = float(entropy / np.log2(len(probabilities))) if probabilities.size > 1 else 0.0 if probabilities.size == 1 else np.nan
    euler = int(components - holes)
    return {
        "topologyConnectedComponents": components,
        "topologyEnclosedHoles": holes,
        "topologyEulerCharacteristic": euler,
        "topologyLargestComponentFractionOfGap": float(areas.max() / total) if areas.size and total > 0 else 0.0,
        "topologyComponentSizeEntropy_bits": entropy,
        "topologyComponentSizeEntropyNormalized": normalized_entropy,
        "topologyConnectivity": "8-connected" if connectivity == 2 else "4-connected",
        "topologyHoleDefinition": "inverse components not touching the ROI boundary",
        "algorithmVersion": STATIC_ALGORITHM_VERSION,
    }


def multiscale_topology(mask: np.ndarray, scales_px: Iterable[int], connectivity: int = 2) -> tuple[pd.DataFrame, dict[str, Any]]:
    binary = as_binary_mask(mask)
    rows: list[dict[str, Any]] = []
    for scale in sorted(set(int(v) for v in scales_px)):
        if scale < 0:
            transformed = morphology.binary_erosion(binary, morphology.disk(abs(scale)))
            operation = "erosion"
        elif scale > 0:
            transformed = morphology.binary_dilation(binary, morphology.disk(scale))
            operation = "dilation"
        else:
            transformed = binary
            operation = "identity"
        topo = topology_features(transformed, connectivity)
        perimeter = float(measure.perimeter(transformed, neighborhood=8))
        rows.append(
            {
                "scale_px": scale,
                "operation": operation,
                "areaFraction": float(np.mean(transformed)),
                "perimeter_px": perimeter,
                "connectedComponents": topo["topologyConnectedComponents"],
                "enclosedHoles": topo["topologyEnclosedHoles"],
                "eulerCharacteristic": topo["topologyEulerCharacteristic"],
                "algorithmVersion": STATIC_ALGORITHM_VERSION,
            }
        )
    table = pd.DataFrame(rows).sort_values("scale_px", ignore_index=True)
    if table.empty:
        return table, {}
    x = table["scale_px"].to_numpy(dtype=float)
    components = table["connectedComponents"].to_numpy(dtype=float)
    holes = table["enclosedHoles"].to_numpy(dtype=float)
    euler = table["eulerCharacteristic"].to_numpy(dtype=float)
    changes = np.abs(np.diff(components))
    transition_scale = float(table.iloc[int(np.argmax(changes)) + 1]["scale_px"]) if changes.size else np.nan
    summary = {
        "multiscaleComponentCountAUC": float(np.trapezoid(components, x)) if len(x) >= 2 else np.nan,
        "multiscaleHoleCountAUC": float(np.trapezoid(holes, x)) if len(x) >= 2 else np.nan,
        "multiscaleEulerAUC": float(np.trapezoid(euler, x)) if len(x) >= 2 else np.nan,
        "strongestConnectivityTransitionScale_px": transition_scale,
        "peakHoleCountScale_px": float(table.iloc[int(np.argmax(holes))]["scale_px"]),
        "topologyStability": float(1.0 / (1.0 + np.std(euler))) if len(euler) else np.nan,
        "algorithmVersion": STATIC_ALGORITHM_VERSION,
    }
    return table, summary


def _skeleton_edge_table(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.argwhere(skeleton)
    if coords.size == 0:
        return coords, np.empty((0, 2), dtype=int), np.empty(0, dtype=float)
    index = {tuple(coord): idx for idx, coord in enumerate(coords)}
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for idx, (row, col) in enumerate(coords):
        for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
            other = index.get((int(row + dr), int(col + dc)))
            if other is not None:
                edges.append((idx, other))
                weights.append(sqrt(2.0) if dr and dc else 1.0)
    return coords, np.asarray(edges, dtype=int), np.asarray(weights, dtype=float)


def _graph_diameter(coords: np.ndarray, edges: np.ndarray, weights: np.ndarray, max_pixels: int) -> float:
    n = len(coords)
    if n == 0 or n > max_pixels or edges.size == 0:
        return np.nan
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([weights, weights])
    graph = csr_matrix((data, (row, col)), shape=(n, n))
    n_components, labels = connected_components(graph, directed=False)
    longest = 0.0
    for component in range(n_components):
        nodes = np.flatnonzero(labels == component)
        if nodes.size <= 1:
            continue
        seed = int(nodes[0])
        first = dijkstra(graph, directed=False, indices=seed)
        far = int(nodes[np.argmax(first[nodes])])
        second = dijkstra(graph, directed=False, indices=far)
        value = float(np.max(second[nodes][np.isfinite(second[nodes])]))
        longest = max(longest, value)
    return longest


def gap_network_features(mask: np.ndarray, config: StaticConfig | None = None) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    """Describe the medial architecture of the plasma-gap / void network."""
    cfg = config or StaticConfig()
    binary = as_binary_mask(mask)
    cleaned = morphology.remove_small_objects(binary, min_size=cfg.min_network_object_area_px, connectivity=cfg.connectivity)
    skeleton = morphology.skeletonize(cleaned)
    neighbour_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbours = ndi.convolve(skeleton.astype(np.uint8), neighbour_kernel, mode="constant", cval=0) - skeleton.astype(np.uint8)
    endpoints = skeleton & (neighbours == 1)
    branchpoints = skeleton & (neighbours >= 3)
    coords, edges, weights = _skeleton_edge_table(skeleton)
    degrees = np.zeros(len(coords), dtype=int)
    if edges.size:
        np.add.at(degrees, edges[:, 0], 1)
        np.add.at(degrees, edges[:, 1], 1)
    graph_components = measure.label(skeleton, connectivity=2)
    component_count = int(graph_components.max())
    edge_count = int(len(edges))
    node_count = int(len(coords))
    cycles = max(0, edge_count - node_count + component_count)
    angles = []
    for (a, b) in edges:
        dy, dx = coords[b] - coords[a]
        angles.append(atan2(float(dy), float(dx)))
    angle_arr = np.asarray(angles, dtype=float)
    if angle_arr.size:
        nematic = float(abs(np.mean(np.exp(2j * angle_arr))))
        hist, _ = np.histogram(np.mod(angle_arr, pi), bins=18, range=(0, pi))
        probs = hist[hist > 0] / max(hist.sum(), 1)
        orientation_entropy = float(-np.sum(probs * np.log2(probs)))
    else:
        nematic = orientation_entropy = np.nan
    total_length = float(np.sum(weights))
    longest_geodesic = _graph_diameter(coords, edges, weights, cfg.max_graph_pixels)
    euclidean_extent = float(np.hypot(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]))) if len(coords) >= 2 else np.nan
    tortuosity = float(longest_geodesic / euclidean_extent) if np.isfinite(longest_geodesic) and np.isfinite(euclidean_extent) and euclidean_extent > 0 else np.nan
    unique_degrees, degree_counts = np.unique(degrees, return_counts=True) if degrees.size else (np.array([], dtype=int), np.array([], dtype=int))
    degree_df = pd.DataFrame(
        {
            "nodeDegree": unique_degrees,
            "nodeCount": degree_counts,
            "nodeFraction": degree_counts / max(int(degree_counts.sum()), 1),
            "interpretation": "compact pixel-skeleton node-degree distribution; raw pixel adjacency edges are intentionally not exported",
        }
    )
    summary = {
        "gapNetworkSkeletonLength_px": total_length,
        "gapNetworkSkeletonDensity": float(total_length / binary.size),
        "gapNetworkPixelCount": node_count,
        "gapNetworkEndpointCount": int(endpoints.sum()),
        "gapNetworkEndpointDensity": float(endpoints.sum() / binary.size),
        "gapNetworkBranchPointCount": int(branchpoints.sum()),
        "gapNetworkBranchDensity": float(branchpoints.sum() / binary.size),
        "gapNetworkJunctionDensity": float(branchpoints.sum() / max(node_count, 1)),
        "gapNetworkConnectedComponents": component_count,
        "gapNetworkMeanNodeDegree": float(np.mean(degrees)) if degrees.size else np.nan,
        "gapNetworkLongestGeodesicPath_px": longest_geodesic,
        "gapNetworkLoopCycleCount": cycles,
        "gapNetworkTortuosityApprox": tortuosity,
        "gapNetworkStraightnessApprox": float(1.0 / tortuosity) if np.isfinite(tortuosity) and tortuosity > 0 else np.nan,
        "gapNetworkOrientationEntropy_bits": orientation_entropy,
        "gapNetworkNematicOrder": nematic,
        "gapNetworkInterpretation": "medial architecture of the plasma-gap / void network; not an RBC rouleaux skeleton",
        "algorithmVersion": STATIC_ALGORITHM_VERSION,
    }
    return summary, skeleton, degree_df


def _resample_closed_contour(contour: np.ndarray, n_points: int) -> np.ndarray:
    points = np.asarray(contour, dtype=float)
    if len(points) < 3:
        return np.empty((0, 2), dtype=float)
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    segment = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    if cumulative[-1] <= 1e-12:
        return np.empty((0, 2), dtype=float)
    samples = np.linspace(0.0, cumulative[-1], int(n_points), endpoint=False)
    row = np.interp(samples, cumulative, points[:, 0])
    col = np.interp(samples, cumulative, points[:, 1])
    return np.column_stack([row, col])


def normalized_contour_modes(contour: np.ndarray, max_mode: int = 10, n_points: int = 128, smooth_sigma: float = 1.0) -> dict[str, Any]:
    """Return translation/scale/rotation/start-point invariant Fourier magnitudes."""
    resampled = _resample_closed_contour(contour, n_points)
    if len(resampled) < max(8, 2 * max_mode + 1):
        return {"valid": False, "reason": "contour_too_short", "modes": pd.DataFrame(), "resampled": resampled, "smoothed": resampled}
    smoothed = np.column_stack(
        [ndi.gaussian_filter1d(resampled[:, axis], sigma=max(0.0, float(smooth_sigma)), mode="wrap") for axis in (0, 1)]
    )
    complex_curve = (smoothed[:, 1] - np.mean(smoothed[:, 1])) + 1j * (smoothed[:, 0] - np.mean(smoothed[:, 0]))
    coeff = np.fft.fft(complex_curve) / len(complex_curve)
    scale = float(np.sqrt(np.sum(np.abs(coeff[1 : max_mode + 1]) ** 2)))
    if scale <= 1e-12:
        return {"valid": False, "reason": "zero_contour_scale", "modes": pd.DataFrame(), "resampled": resampled, "smoothed": smoothed}
    amplitudes = np.abs(coeff[1 : max_mode + 1]) / scale
    energy = amplitudes**2
    total_energy = float(np.sum(energy))
    rows = []
    cumulative = 0.0
    for mode, (amplitude, mode_energy) in enumerate(zip(amplitudes, energy), start=1):
        cumulative += float(mode_energy)
        rows.append(
            {
                "mode": mode,
                "normalizedAmplitude": float(amplitude),
                "normalizedEnergy": float(mode_energy / total_energy) if total_energy > 0 else np.nan,
                "cumulativeNormalizedEnergy": float(cumulative / total_energy) if total_energy > 0 else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    low = float(table.loc[table["mode"].between(2, min(4, max_mode)), "normalizedEnergy"].sum())
    high = float(table.loc[table["mode"] >= min(5, max_mode), "normalizedEnergy"].sum())
    return {
        "valid": True,
        "reason": "ok",
        "modes": table,
        "resampled": resampled,
        "smoothed": smoothed,
        "lowOrderEnergy_modes2to4": low,
        "lowHighOrderEnergyRatio": float(low / high) if high > 1e-12 else np.nan,
        "normalization": "centroid removed; magnitude spectrum normalized by modes 1..N (translation, scale, rotation and start-point invariant)",
        "algorithmVersion": STATIC_ALGORITHM_VERSION,
    }


def contour_mode_analysis(mask: np.ndarray, objects: pd.DataFrame, config: StaticConfig | None = None) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    cfg = config or StaticConfig()
    binary = as_binary_mask(mask)
    valid_objects = objects.loc[(objects.get("fullyObserved", False) == True) & (pd.to_numeric(objects.get("area_px2"), errors="coerce") >= cfg.contour_min_area_px)].copy() if not objects.empty else objects
    if not valid_objects.empty:
        valid_objects = valid_objects.nlargest(cfg.max_contours_per_frame, "area_px2")
    all_rows: list[pd.DataFrame] = []
    visual: list[dict[str, Any]] = []
    for _, row in valid_objects.iterrows():
        label_id = int(row["gapObjectID"])
        component = measure.label(binary, connectivity=cfg.connectivity) == label_id
        contours = measure.find_contours(component.astype(float), 0.5, fully_connected="high")
        if not contours:
            continue
        contour = max(contours, key=len)
        result = normalized_contour_modes(contour, cfg.contour_max_mode, cfg.contour_points, cfg.contour_smoothing_sigma)
        if not result.get("valid"):
            continue
        table = result["modes"].copy()
        table.insert(0, "gapObjectID", label_id)
        table["objectArea_px2"] = float(row["area_px2"])
        table["algorithmVersion"] = STATIC_ALGORITHM_VERSION
        all_rows.append(table)
        visual.append(
            {
                "gapObjectID": label_id,
                "originalContour": contour,
                "resampledContour": result["resampled"],
                "smoothedContour": result["smoothed"],
                "lowOrderEnergy_modes2to4": result["lowOrderEnergy_modes2to4"],
                "lowHighOrderEnergyRatio": result["lowHighOrderEnergyRatio"],
            }
        )
    modes = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if modes.empty:
        summary = {"contourObjectsAnalysed": 0, "contourHighestComputedMode": 0, "contourLowOrderEnergyMedian": np.nan, "contourLowHighEnergyRatioMedian": np.nan}
    else:
        per_object = modes.groupby("gapObjectID", as_index=False).agg(
            lowOrderEnergy=("normalizedEnergy", lambda x: float(np.sum(np.asarray(x)[1:4])) if len(x) >= 4 else np.nan)
        )
        ratios = [item["lowHighOrderEnergyRatio"] for item in visual]
        summary = {
            "contourObjectsAnalysed": int(modes["gapObjectID"].nunique()),
            "contourHighestComputedMode": int(modes["mode"].max()),
            "contourLowOrderEnergyMedian": safe_percentile(per_object["lowOrderEnergy"].to_numpy(), 50),
            "contourLowHighEnergyRatioMedian": safe_percentile(np.asarray(ratios, dtype=float), 50),
            "contourReliabilityNote": "mode stability requires robustness analysis; computed mode does not imply validated mode",
            "algorithmVersion": STATIC_ALGORITHM_VERSION,
        }
    return modes, visual, summary


def _moran_i(values: np.ndarray, valid: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(arr)
    if int(ok.sum()) < 3:
        return np.nan
    mean = float(np.mean(arr[ok]))
    denominator = float(np.sum((arr[ok] - mean) ** 2))
    if denominator <= 1e-12:
        return np.nan
    numerator = 0.0
    weights = 0
    for row, col in np.argwhere(ok):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = row + dr, col + dc
            if 0 <= rr < arr.shape[0] and 0 <= cc < arr.shape[1] and ok[rr, cc]:
                numerator += (arr[row, col] - mean) * (arr[rr, cc] - mean)
                weights += 1
    return float(ok.sum() / weights * numerator / denominator) if weights else np.nan


def spatial_grid_analysis(mask: np.ndarray, thickness_map: np.ndarray, grid_sizes: Iterable[int]) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    binary = as_binary_mask(mask)
    rows: list[dict[str, Any]] = []
    maps: dict[str, np.ndarray] = {}
    for grid in sorted({int(value) for value in grid_sizes if int(value) >= 2}):
        if min(binary.shape) < grid:
            continue
        y_edges = np.linspace(0, binary.shape[0], grid + 1, dtype=int)
        x_edges = np.linspace(0, binary.shape[1], grid + 1, dtype=int)
        area_map = np.full((grid, grid), np.nan, dtype=float)
        thickness_median_map = np.full((grid, grid), np.nan, dtype=float)
        count_map = np.full((grid, grid), np.nan, dtype=float)
        for gy in range(grid):
            for gx in range(grid):
                tile = binary[y_edges[gy] : y_edges[gy + 1], x_edges[gx] : x_edges[gx + 1]]
                tile_thickness = thickness_map[y_edges[gy] : y_edges[gy + 1], x_edges[gx] : x_edges[gx + 1]]
                if tile.size == 0:
                    continue
                labels = measure.label(tile, connectivity=2)
                component_areas = np.bincount(labels.ravel())[1:]
                area = float(np.mean(tile))
                thick_values = tile_thickness[tile]
                area_map[gy, gx] = area
                thickness_median_map[gy, gx] = float(np.median(thick_values)) if thick_values.size else 0.0
                count_map[gy, gx] = int(labels.max())
                rows.append(
                    {
                        "gridSize": grid,
                        "tileRow": gy,
                        "tileColumn": gx,
                        "tileHeight_px": int(tile.shape[0]),
                        "tileWidth_px": int(tile.shape[1]),
                        "localGapFraction": area,
                        "localGapCount": int(labels.max()),
                        "localMedianGapArea_px2": float(np.median(component_areas)) if component_areas.size else np.nan,
                        "localLargestGapFraction": float(component_areas.max() / tile.size) if component_areas.size else 0.0,
                        "localThicknessMedian_px": float(np.median(thick_values)) if thick_values.size else 0.0,
                        "localTopologyEuler": int(measure.euler_number(tile, connectivity=2)),
                        "validPixelFraction": 1.0,
                        "algorithmVersion": STATIC_ALGORITHM_VERSION,
                    }
                )
        maps[f"gap_fraction_{grid}x{grid}"] = area_map
        maps[f"thickness_median_{grid}x{grid}"] = thickness_median_map
        maps[f"gap_count_{grid}x{grid}"] = count_map
    table = pd.DataFrame(rows)
    summary: dict[str, Any] = {"algorithmVersion": STATIC_ALGORITHM_VERSION}
    if not table.empty:
        preferred = 4 if 4 in set(table["gridSize"]) else int(table["gridSize"].min())
        subset = table.loc[table["gridSize"] == preferred]
        values = pd.to_numeric(subset["localGapFraction"], errors="coerce").to_numpy(dtype=float)
        map_values = maps[f"gap_fraction_{preferred}x{preferred}"]
        valid = np.isfinite(map_values)
        median = safe_percentile(values, 50)
        summary.update(
            {
                "spatialSummaryGrid": preferred,
                "spatialGapFractionMean": _safe_mean(values),
                "spatialGapFractionSD": _safe_std(values),
                "spatialGapFractionCV": _safe_cv(values),
                "spatialGapFractionIQR": float(safe_percentile(values, 75) - safe_percentile(values, 25)) if len(values) else np.nan,
                "spatialGapFractionMinimum": float(np.nanmin(values)) if len(values) else np.nan,
                "spatialGapFractionMaximum": float(np.nanmax(values)) if len(values) else np.nan,
                "spatialGapFractionMaxMedianRatio": float(np.nanmax(values) / median) if np.isfinite(median) and median > 0 else np.nan,
                "spatialHotspotFraction": float(np.mean(values >= safe_percentile(values, 75))) if len(values) else np.nan,
                "spatialColdspotFraction": float(np.mean(values <= safe_percentile(values, 25))) if len(values) else np.nan,
                "spatialGapFractionMoranI": _moran_i(map_values, valid),
                "spatialLeftRightGradient": float(np.nanmean(map_values[:, -1]) - np.nanmean(map_values[:, 0])),
                "spatialTopBottomGradient": float(np.nanmean(map_values[-1, :]) - np.nanmean(map_values[0, :])),
                "spatialCentreEdgeDifference": float(np.nanmean(map_values[preferred // 4 : preferred - preferred // 4, preferred // 4 : preferred - preferred // 4]) - np.nanmean(np.concatenate([map_values[0], map_values[-1], map_values[1:-1, 0], map_values[1:-1, -1]]))) if preferred >= 4 else np.nan,
            }
        )
    return table, maps, summary


def analyze_static_gap_structure(mask: np.ndarray, config: StaticConfig | None = None) -> dict[str, Any]:
    """Run the complete static structural layer on one authoritative gap mask."""
    cfg = config or StaticConfig()
    binary = as_binary_mask(mask)
    objects = gap_object_features(binary, cfg)
    frame = gap_frame_summary(binary, objects, cfg)
    size = gap_size_spectrum(objects, binary.size)
    thickness_summary, thickness_map, thickness_distribution = local_thickness_analysis(binary)
    gran = granulometry(binary, cfg.granulometry_radii_px)
    gran_summary = granulometry_summary(gran)
    chord_summary, chords = chord_length_analysis(binary)
    topology = topology_features(binary, cfg.connectivity)
    multiscale, multiscale_summary = multiscale_topology(binary, cfg.topology_scales_px, cfg.connectivity)
    network, skeleton, network_edges = gap_network_features(binary, cfg)
    contour_modes, contour_visuals, contour_summary = contour_mode_analysis(binary, objects, cfg)
    spatial, spatial_maps, spatial_summary = spatial_grid_analysis(binary, thickness_map, cfg.grid_sizes)
    combined = {
        **frame,
        **thickness_summary,
        **gran_summary,
        **chord_summary,
        **topology,
        **multiscale_summary,
        **network,
        **contour_summary,
        **spatial_summary,
    }
    return {
        "summary": combined,
        "objects": objects,
        "sizeSpectrum": size,
        "thicknessMap": thickness_map,
        "thicknessDistribution": thickness_distribution,
        "granulometry": gran,
        "chordLengths": chords,
        "multiscaleTopology": multiscale,
        "skeleton": skeleton,
        "networkDegreeDistribution": network_edges,
        "contourModes": contour_modes,
        "contourVisuals": contour_visuals,
        "spatialTiles": spatial,
        "spatialMaps": spatial_maps,
        "algorithmVersion": STATIC_ALGORITHM_VERSION,
    }
