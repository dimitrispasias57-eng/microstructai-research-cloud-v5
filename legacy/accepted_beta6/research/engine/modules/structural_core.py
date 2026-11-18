"""Authoritative V4 mask-measurement facade over the preserved V3 primitives.

No segmentation or domain scores here. Legacy formulas have ONE implementation
in metrics.py / morphology_intelligence.py, not a second rewritten copy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, morphology

from .metrics import compute_core_metrics as legacy_image_mask_metrics
from .metrics import compute_xi_from_binary_mask, skeleton_counts
from .morphology_intelligence import topology_features, gap_network_features, StaticConfig
from .gap_dynamics import mask_turnover

CORE_VERSION = "4.0-mask-core-1"
SCHEMA_VERSION = "4.0-dev-mask-1"
METRIC_NOTES = {
    "Xi_horizontal_px": "pixels; directional along image rows AFTER recorded orientation. Not aggregate/cell size or known flow-direction length.",
    "Xi_radial_px": "pixels; legacy radial autocorrelation, maximum working dimension 384; not interchangeable with horizontal Xi.",
    "lacunarity16": "dimensionless; non-overlapping 16×16 pixel boxes, incomplete edge boxes discarded. Scale-dependent; unavailable below 16 pixels.",
    "Euler_legacy_8_8": "8-connected foreground minus enclosed 8-connected background components. Historical V3 definition, preserved.",
    "Euler_canonical_8_4_v1": "8-connected foreground minus enclosed 4-connected background components (dual connectivity). Preferred digital topology definition.",
    "branch_pixels": "Count of Zhang skeleton pixels with at least 3 neighbours in the 8-neighbourhood, not anatomical junctions.",
    "junction_count_cluster8_v1": "8-connected clusters of adjacent branch pixels on the unfiltered Zhang skeleton. Close junctions may merge; no anatomical validation.",
    "network_length_px": "Sum of 8-adjacency skeleton edge lengths (1 or sqrt(2)), after removing components smaller than 20 pixels². Pixel-graph, not anatomical path length.",
    "mask_anisotropy": "Legacy multiscale structure-tensor calculation applied to the binary mask, not the original image. Pixel-scale dependent.",
    "interfaceDensity_px": "4-neighbour erosion-defined foreground-boundary pixel count / image pixel count (legacy bwperim-like definition); inverse-pixel units, not an unbiased perimeter.",
    "network_pixel_cycles": "E−V+C of an 8-adjacency pixel graph; diagonal triangles can form graph cycles, not necessarily biological loops.",
}


def validate_binary(mask):
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.size == 0 or min(arr.shape) < 2:
        raise ValueError("Structural mask must be a non-empty 2-D array with both dimensions at least 2 pixels.")
    if arr.dtype != np.bool_:
        raise ValueError("Structural core requires a validated boolean mask; conversion belongs to input validation.")
    return np.ascontiguousarray(arr)


def compute_core_metrics(image, mask, filename="", mode=""):
    """Legacy-compatible image+mask entry point. Intensity anisotropy stays legacy.

    Existing app calls and generic/supplied calls share the same underlying
    primitives; extra V4 descriptors are separate, never injected into old rows.
    """
    return legacy_image_mask_metrics(image, mask, filename=filename, mode=mode)


def canonical_topology(mask):
    binary = validate_binary(mask)
    _, foreground_n = ndi.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    background, _ = ndi.label(~binary, structure=ndi.generate_binary_structure(2, 1))
    border = set(np.unique(np.concatenate([background[0], background[-1], background[:, 0], background[:, -1]])))
    holes = sum(int(label != 0 and label not in border) for label in np.unique(background))
    return {"Euler_canonical_8_4_v1": int(foreground_n - holes), "holes_background4_v1": int(holes)}


def consolidated_junctions(mask):
    skeleton = morphology.skeletonize(validate_binary(mask))
    degree = ndi.convolve(skeleton.astype(np.int16), np.ones((3, 3), dtype=np.int16), mode="constant") - skeleton
    branches = skeleton & (degree >= 3)
    _, count = ndi.label(branches, structure=np.ones((3, 3), dtype=np.uint8))
    return int(count)


def measure_mask(mask):
    """Direct descriptors for any algorithmic foreground; no RBC semantics.

    All 24 previously checked values delegate to their original implementation.
    Two new topology fields and one new junction definition have versioned names.
    """
    binary = validate_binary(mask)
    values = compute_core_metrics(binary.astype(np.uint8) * 255, binary)
    topology = topology_features(binary)
    # Skip the optional all-pairs graph diameter: it is NOT one of these outputs.
    # Length/count/cycle quantities are unaffected by max_graph_pixels.
    network, _, _ = gap_network_features(binary, StaticConfig(max_graph_pixels=0))
    skeleton, branches, endpoints = skeleton_counts(binary)
    keys = ["areaFrac", "nObjects", "meanArea", "medianArea", "meanDiam", "medianDiam",
            "largestObjectFrac", "largestClusterFrac_phase", "lacunarity16", "interfaceDensity_px"]
    result = {key: values[key] for key in keys}
    result.update(Xi_horizontal_px=compute_xi_from_binary_mask(binary, rejectUniform=False),
                  Xi_radial_px=values["Xi_px"], mask_anisotropy=values["anisotropy"],
                  Euler_legacy_8_8=topology["topologyEulerCharacteristic"], holes_background8_legacy=topology["topologyEnclosedHoles"],
                  skeleton_pixels=skeleton, branch_pixels=branches, endpoints=endpoints,
                  network_length_px=network["gapNetworkSkeletonLength_px"], network_pixels=network["gapNetworkPixelCount"],
                  network_endpoints=network["gapNetworkEndpointCount"], network_branch_pixels=network["gapNetworkBranchPointCount"],
                  network_components=network["gapNetworkConnectedComponents"], network_pixel_cycles=network["gapNetworkLoopCycleCount"],
                  junction_count_cluster8_v1=consolidated_junctions(binary), **canonical_topology(binary))
    # Distribution uses the same 8-connected labelling and region properties as
    # the legacy scalar descriptors; it is not a second estimator.
    regions = measure.regionprops(measure.label(binary, connectivity=2))
    objects = pd.DataFrame([{"component": r.label, "area_px2": r.area,
                             "equivalent_diameter_px": r.equivalent_diameter_area} for r in regions],
                           columns=["component", "area_px2", "equivalent_diameter_px"])
    return result, objects


def measure_temporal(masks):
    for mask in masks:
        validate_binary(mask)
    if len({m.shape for m in masks}) > 1:
        raise ValueError("Temporal descriptors require identical dimensions.")
    return mask_turnover(masks)  # unchanged definitions, unregistered coordinates


def metric_unit(name):
    if name in ("meanArea", "medianArea") or name.endswith("_px2"):
        return "pixels²"
    if name in ("meanDiam", "medianDiam", "Xi_horizontal_px", "Xi_radial_px", "network_length_px"):
        return "pixels"
    if name == "interfaceDensity_px":
        return "1/pixel"
    if name in ("areaFrac", "largestObjectFrac", "largestClusterFrac_phase", "lacunarity16", "mask_anisotropy"):
        return "dimensionless"
    return "count"
