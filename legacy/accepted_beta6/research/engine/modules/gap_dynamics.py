"""Registration, gap tracking and descriptive dynamics for MicroStructAI V3.

Tracked objects are plasma-gap structures.  They are never interpreted as
individual RBCs.  Ambiguous associations are retained as QC information and
are not forced into one-to-one identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage import measure
from skimage.registration import phase_cross_correlation

from .morphology_intelligence import as_binary_mask


DYNAMICS_ALGORITHM_VERSION = "3.1.0-gap-dynamics"


@dataclass(frozen=True)
class DynamicsConfig:
    registration_upsample_factor: int = 10
    max_registration_fraction: float = 0.10
    max_registration_error: float = 0.75
    min_object_area_px: int = 20
    min_iou: float = 0.05
    min_overlap_fraction: float = 0.15
    max_centroid_distance_px: float = 35.0
    max_area_ratio: float = 4.0
    min_track_length_frames: int = 3


def _normalize_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Registration images must be 2-D grayscale arrays.")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr -= float(np.mean(arr))
    scale = float(np.std(arr))
    return arr / scale if scale > 1e-8 else arr


def register_sequence(images: list[np.ndarray], masks: list[np.ndarray], config: DynamicsConfig | None = None) -> dict[str, Any]:
    """Register every frame to the previous frame using phase correlation."""
    cfg = config or DynamicsConfig()
    if len(images) != len(masks):
        raise ValueError("Registration image and mask sequence lengths differ.")
    if not images:
        return {"qc": pd.DataFrame(), "registeredMasks": [], "registeredImages": [], "registrationReliable": False}
    reference_shape = np.asarray(images[0]).shape
    if any(np.asarray(image).shape != reference_shape for image in images) or any(np.asarray(mask).shape != reference_shape for mask in masks):
        raise ValueError("All registration frames and masks must have the same dimensions.")
    registered_images = [np.asarray(images[0], dtype=np.float32)]
    registered_masks = [as_binary_mask(masks[0]).copy()]
    rows = [
        {
            "frameIndex": 1,
            "referenceFrameIndex": 1,
            "incrementalShiftY_px": 0.0,
            "incrementalShiftX_px": 0.0,
            "cumulativeShiftY_px": 0.0,
            "cumulativeShiftX_px": 0.0,
            "registrationError": 0.0,
            "registrationConfidence": 1.0,
            "registrationApplied": False,
            "registrationReliable": True,
            "registrationFailureReason": "reference_frame",
            "algorithmVersion": DYNAMICS_ALGORITHM_VERSION,
        }
    ]
    cumulative = np.array([0.0, 0.0], dtype=float)
    max_shift = cfg.max_registration_fraction * float(np.hypot(*reference_shape))
    previous = _normalize_image(images[0])
    for idx in range(1, len(images)):
        current_raw = np.asarray(images[idx], dtype=np.float32)
        current = _normalize_image(current_raw)
        reliable = True
        reason = "ok"
        try:
            shift, error, _phase = phase_cross_correlation(previous, current, upsample_factor=cfg.registration_upsample_factor)
            shift = np.asarray(shift[:2], dtype=float)
            error = float(error)
            if not np.all(np.isfinite(shift)) or not np.isfinite(error):
                reliable, reason = False, "non_finite_registration"
            elif float(np.linalg.norm(shift)) > max_shift:
                reliable, reason = False, "shift_exceeds_plausible_limit"
            elif error > cfg.max_registration_error:
                reliable, reason = False, "registration_error_too_high"
        except Exception as exc:
            shift = np.array([0.0, 0.0], dtype=float)
            error = np.nan
            reliable, reason = False, f"registration_failed:{type(exc).__name__}"
        if reliable:
            cumulative += shift
            registered_image = ndi.shift(current_raw, cumulative, order=1, mode="nearest", prefilter=False)
            registered_mask = ndi.shift(as_binary_mask(masks[idx]).astype(np.uint8), cumulative, order=0, mode="constant", cval=0, prefilter=False).astype(bool)
        else:
            registered_image = current_raw
            registered_mask = as_binary_mask(masks[idx]).copy()
        registered_images.append(registered_image)
        registered_masks.append(registered_mask)
        confidence = float(np.clip(1.0 - error, 0.0, 1.0)) if np.isfinite(error) else 0.0
        rows.append(
            {
                "frameIndex": idx + 1,
                "referenceFrameIndex": idx,
                "incrementalShiftY_px": float(shift[0]),
                "incrementalShiftX_px": float(shift[1]),
                "cumulativeShiftY_px": float(cumulative[0]),
                "cumulativeShiftX_px": float(cumulative[1]),
                "registrationError": error,
                "registrationConfidence": confidence,
                "registrationApplied": bool(reliable and np.linalg.norm(cumulative) > 1e-6),
                "registrationReliable": reliable,
                "registrationFailureReason": reason,
                "algorithmVersion": DYNAMICS_ALGORITHM_VERSION,
            }
        )
        previous = current
    qc = pd.DataFrame(rows)
    reliable_fraction = float(qc["registrationReliable"].mean()) if not qc.empty else 0.0
    return {
        "qc": qc,
        "registeredMasks": registered_masks,
        "registeredImages": registered_images,
        "registrationReliable": reliable_fraction >= 0.80,
        "registrationReliableFrameFraction": reliable_fraction,
        "algorithmVersion": DYNAMICS_ALGORITHM_VERSION,
    }


def _objects(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels = measure.label(as_binary_mask(mask), connectivity=2)
    rows = []
    keep = np.zeros_like(labels, dtype=np.int32)
    next_label = 1
    for region in measure.regionprops(labels):
        if region.area < min_area:
            continue
        keep[labels == region.label] = next_label
        rows.append(
            {
                "localObjectID": next_label,
                "area_px2": float(region.area),
                "perimeter_px": float(region.perimeter),
                "centroidY_px": float(region.centroid[0]),
                "centroidX_px": float(region.centroid[1]),
                "eccentricity": float(region.eccentricity),
                "orientation_deg": float(np.degrees(region.orientation)),
            }
        )
        next_label += 1
    return keep, rows


def _overlap_matrices(previous_labels: np.ndarray, current_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_prev = int(previous_labels.max())
    n_curr = int(current_labels.max())
    if n_prev == 0 or n_curr == 0:
        return np.zeros((n_prev, n_curr)), np.zeros((n_prev, n_curr)), np.zeros((n_prev, n_curr))
    combined = previous_labels.ravel().astype(np.int64) * (n_curr + 1) + current_labels.ravel().astype(np.int64)
    table = np.bincount(combined, minlength=(n_prev + 1) * (n_curr + 1)).reshape(n_prev + 1, n_curr + 1)[1:, 1:].astype(float)
    prev_area = np.bincount(previous_labels.ravel(), minlength=n_prev + 1)[1:].astype(float)[:, None]
    curr_area = np.bincount(current_labels.ravel(), minlength=n_curr + 1)[1:].astype(float)[None, :]
    union = prev_area + curr_area - table
    iou = np.divide(table, union, out=np.zeros_like(table), where=union > 0)
    prev_overlap = np.divide(table, prev_area, out=np.zeros_like(table), where=prev_area > 0)
    curr_overlap = np.divide(table, curr_area, out=np.zeros_like(table), where=curr_area > 0)
    return iou, prev_overlap, curr_overlap


def track_gap_objects(masks: list[np.ndarray], registration_qc: pd.DataFrame | None = None, config: DynamicsConfig | None = None) -> dict[str, Any]:
    """Track gap structures with explicit merge/split and ambiguity handling."""
    cfg = config or DynamicsConfig()
    if not masks:
        return {"tracks": pd.DataFrame(), "events": pd.DataFrame(), "summary": {}}
    label_frames: list[np.ndarray] = []
    object_frames: list[list[dict[str, Any]]] = []
    for mask in masks:
        labels, objects = _objects(mask, cfg.min_object_area_px)
        label_frames.append(labels)
        object_frames.append(objects)

    next_track_id = 1
    track_map: dict[int, int] = {}
    records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for obj in object_frames[0]:
        track_map[int(obj["localObjectID"])] = next_track_id
        records.append({"frameIndex": 1, "trackID": next_track_id, **obj, "associationConfidence": 1.0, "associationAmbiguous": False, "eventRole": "birth"})
        events.append({"frameIndex": 1, "eventType": "BIRTH", "parentTrackIDs": "", "childTrackIDs": str(next_track_id), "confidence": 1.0, "ambiguityFlag": False})
        next_track_id += 1

    for frame_idx in range(1, len(masks)):
        prev_labels = label_frames[frame_idx - 1]
        curr_labels = label_frames[frame_idx]
        prev_objects = {int(item["localObjectID"]): item for item in object_frames[frame_idx - 1]}
        curr_objects = {int(item["localObjectID"]): item for item in object_frames[frame_idx]}
        iou, prev_overlap, curr_overlap = _overlap_matrices(prev_labels, curr_labels)
        reliable_registration = True
        if registration_qc is not None and frame_idx < len(registration_qc):
            reliable_registration = bool(registration_qc.iloc[frame_idx].get("registrationReliable", True))
        previous_track_map = dict(track_map)
        track_map = {}

        candidate_prev_for_curr = {curr: [] for curr in curr_objects}
        candidate_curr_for_prev = {prev: [] for prev in prev_objects}
        for prev_id, prev_obj in prev_objects.items():
            for curr_id, curr_obj in curr_objects.items():
                overlap_ok = prev_overlap[prev_id - 1, curr_id - 1] >= cfg.min_overlap_fraction or curr_overlap[prev_id - 1, curr_id - 1] >= cfg.min_overlap_fraction
                dy = curr_obj["centroidY_px"] - prev_obj["centroidY_px"]
                dx = curr_obj["centroidX_px"] - prev_obj["centroidX_px"]
                distance = float(np.hypot(dy, dx))
                ratio = max(curr_obj["area_px2"] / max(prev_obj["area_px2"], 1.0), prev_obj["area_px2"] / max(curr_obj["area_px2"], 1.0))
                compatible = iou[prev_id - 1, curr_id - 1] >= cfg.min_iou or (overlap_ok and distance <= cfg.max_centroid_distance_px and ratio <= cfg.max_area_ratio)
                if compatible:
                    candidate_prev_for_curr[curr_id].append(prev_id)
                    candidate_curr_for_prev[prev_id].append(curr_id)

        handled_curr: set[int] = set()
        handled_prev: set[int] = set()
        for curr_id, parents in candidate_prev_for_curr.items():
            if len(parents) >= 2:
                parent_tracks = sorted({previous_track_map[p] for p in parents if p in previous_track_map})
                track_id = min(parent_tracks) if parent_tracks else next_track_id
                if not parent_tracks:
                    next_track_id += 1
                track_map[curr_id] = track_id
                confidence = float(np.mean([curr_overlap[p - 1, curr_id - 1] for p in parents]))
                events.append({"frameIndex": frame_idx + 1, "eventType": "MERGE", "parentTrackIDs": ";".join(map(str, parent_tracks)), "childTrackIDs": str(track_id), "confidence": confidence, "ambiguityFlag": not reliable_registration})
                handled_curr.add(curr_id)
                handled_prev.update(parents)

        for prev_id, children in candidate_curr_for_prev.items():
            unhandled = [child for child in children if child not in handled_curr]
            if len(unhandled) >= 2:
                parent_track = previous_track_map.get(prev_id)
                child_tracks = []
                for rank, curr_id in enumerate(sorted(unhandled, key=lambda c: iou[prev_id - 1, c - 1], reverse=True)):
                    if rank == 0 and parent_track is not None:
                        track_id = parent_track
                    else:
                        track_id = next_track_id
                        next_track_id += 1
                    track_map[curr_id] = track_id
                    child_tracks.append(track_id)
                    handled_curr.add(curr_id)
                confidence = float(np.mean([prev_overlap[prev_id - 1, child - 1] for child in unhandled]))
                events.append({"frameIndex": frame_idx + 1, "eventType": "SPLIT", "parentTrackIDs": str(parent_track or ""), "childTrackIDs": ";".join(map(str, child_tracks)), "confidence": confidence, "ambiguityFlag": not reliable_registration})
                handled_prev.add(prev_id)

        remaining_prev = [p for p in prev_objects if p not in handled_prev]
        remaining_curr = [c for c in curr_objects if c not in handled_curr]
        if remaining_prev and remaining_curr:
            cost = np.full((len(remaining_prev), len(remaining_curr)), 1e6, dtype=float)
            for pi, prev_id in enumerate(remaining_prev):
                for ci, curr_id in enumerate(remaining_curr):
                    if prev_id in candidate_prev_for_curr[curr_id]:
                        distance = np.hypot(
                            curr_objects[curr_id]["centroidY_px"] - prev_objects[prev_id]["centroidY_px"],
                            curr_objects[curr_id]["centroidX_px"] - prev_objects[prev_id]["centroidX_px"],
                        )
                        cost[pi, ci] = 1.0 - iou[prev_id - 1, curr_id - 1] + 0.25 * distance / max(cfg.max_centroid_distance_px, 1.0)
            row_idx, col_idx = linear_sum_assignment(cost)
            for pi, ci in zip(row_idx, col_idx):
                if cost[pi, ci] >= 1e5:
                    continue
                prev_id, curr_id = remaining_prev[pi], remaining_curr[ci]
                track_map[curr_id] = previous_track_map.get(prev_id, next_track_id)
                if prev_id not in previous_track_map:
                    next_track_id += 1
                handled_prev.add(prev_id)
                handled_curr.add(curr_id)

        for curr_id in curr_objects:
            if curr_id not in track_map:
                track_map[curr_id] = next_track_id
                events.append({"frameIndex": frame_idx + 1, "eventType": "BIRTH", "parentTrackIDs": "", "childTrackIDs": str(next_track_id), "confidence": 1.0 if reliable_registration else 0.25, "ambiguityFlag": not reliable_registration})
                next_track_id += 1
        for prev_id, previous_track in previous_track_map.items():
            if prev_id not in handled_prev and not candidate_curr_for_prev.get(prev_id):
                events.append({"frameIndex": frame_idx + 1, "eventType": "DISAPPEARANCE", "parentTrackIDs": str(previous_track), "childTrackIDs": "", "confidence": 1.0 if reliable_registration else 0.25, "ambiguityFlag": not reliable_registration})

        for curr_id, obj in curr_objects.items():
            parents = candidate_prev_for_curr.get(curr_id, [])
            confidence = float(max([iou[p - 1, curr_id - 1] for p in parents], default=0.0))
            ambiguous = bool(len(parents) > 1 or not reliable_registration)
            role = "merge_child" if len(parents) > 1 else "tracked" if parents else "birth"
            records.append({"frameIndex": frame_idx + 1, "trackID": track_map[curr_id], **obj, "associationConfidence": confidence, "associationAmbiguous": ambiguous, "eventRole": role})

    tracks = pd.DataFrame(records)
    event_table = pd.DataFrame(events)
    if not tracks.empty:
        tracks = tracks.sort_values(["trackID", "frameIndex"], ignore_index=True)
        tracks["trackAge_frames"] = tracks.groupby("trackID").cumcount() + 1
        tracks["areaChange_px2_per_frame"] = tracks.groupby("trackID")["area_px2"].diff()
        tracks["perimeterChange_px_per_frame"] = tracks.groupby("trackID")["perimeter_px"].diff()
        tracks["centroidDisplacement_px_per_frame"] = np.hypot(
            tracks.groupby("trackID")["centroidY_px"].diff(),
            tracks.groupby("trackID")["centroidX_px"].diff(),
        )
        lifetime = tracks.groupby("trackID")["frameIndex"].agg(lambda s: int(s.max() - s.min() + 1))
        tracks["trackLifetime_frames"] = tracks["trackID"].map(lifetime)
        tracks["trackReliable"] = (tracks["trackLifetime_frames"] >= cfg.min_track_length_frames) & (~tracks["associationAmbiguous"].fillna(True))
    event_counts = event_table["eventType"].value_counts() if not event_table.empty else pd.Series(dtype=int)
    reliable_tracks = tracks.loc[tracks.get("trackReliable", pd.Series(False, index=tracks.index)).fillna(False)] if not tracks.empty else tracks
    transitions = max(len(masks) - 1, 1)
    summary = {
        "gapTrackCount": int(tracks["trackID"].nunique()) if not tracks.empty else 0,
        "reliableGapTrackCount": int(reliable_tracks["trackID"].nunique()) if not reliable_tracks.empty else 0,
        "medianTrackPersistence_frames": float(reliable_tracks.groupby("trackID")["trackLifetime_frames"].first().median()) if not reliable_tracks.empty else np.nan,
        "gapBirthRate_perFrame": float(event_counts.get("BIRTH", 0) / transitions),
        "gapDisappearanceRate_perFrame": float(event_counts.get("DISAPPEARANCE", 0) / transitions),
        "gapMergeRate_perFrame": float(event_counts.get("MERGE", 0) / transitions),
        "gapSplitRate_perFrame": float(event_counts.get("SPLIT", 0) / transitions),
        "meanGapAreaGrowth_px2_perFrame": float(reliable_tracks["areaChange_px2_per_frame"].dropna().clip(lower=0).mean()) if not reliable_tracks.empty else np.nan,
        "meanGapClosure_px2_perFrame": float((-reliable_tracks["areaChange_px2_per_frame"].dropna().clip(upper=0)).mean()) if not reliable_tracks.empty else np.nan,
        "meanCentroidVelocity_px_perFrame": float(reliable_tracks["centroidDisplacement_px_per_frame"].dropna().mean()) if not reliable_tracks.empty else np.nan,
        "trackingUnits": "frames and pixels; no frame-rate or spatial calibration assumed",
        "trackingInterpretation": "plasma-gap structures, not individual RBCs",
        "algorithmVersion": DYNAMICS_ALGORITHM_VERSION,
    }
    return {"tracks": tracks, "events": event_table, "summary": summary, "algorithmVersion": DYNAMICS_ALGORITHM_VERSION}


def mask_turnover(masks: list[np.ndarray]) -> pd.DataFrame:
    rows = []
    for idx in range(1, len(masks)):
        previous = as_binary_mask(masks[idx - 1])
        current = as_binary_mask(masks[idx])
        intersection = float(np.logical_and(previous, current).sum())
        union = float(np.logical_or(previous, current).sum())
        rows.append(
            {
                "frameIndex": idx + 1,
                "maskIoUFromPrevious": float(intersection / union) if union > 0 else 1.0,
                "temporalMaskTurnover": float(np.logical_xor(previous, current).mean()),
                "gapAreaChangeFraction": float(current.mean() - previous.mean()),
                "algorithmVersion": DYNAMICS_ALGORITHM_VERSION,
            }
        )
    return pd.DataFrame(rows)


def analyze_gap_dynamics(images: list[np.ndarray], masks: list[np.ndarray], config: DynamicsConfig | None = None) -> dict[str, Any]:
    cfg = config or DynamicsConfig()
    registration = register_sequence(images, masks, cfg)
    if registration["registrationReliable"]:
        analysis_masks = registration["registeredMasks"]
        tracking_suppressed = False
        reason = "ok"
    else:
        analysis_masks = masks
        tracking_suppressed = True
        reason = "registration failed reliability threshold; displacement-dependent tracking suppressed"
    turnover = mask_turnover(analysis_masks)
    if tracking_suppressed:
        tracking = {"tracks": pd.DataFrame(), "events": pd.DataFrame(), "summary": {"trackingSuppressed": True, "trackingSuppressionReason": reason}}
    else:
        tracking = track_gap_objects(analysis_masks, registration["qc"], cfg)
        tracking["summary"].update({"trackingSuppressed": False, "trackingSuppressionReason": ""})
    temporal_summary = {
        **tracking.get("summary", {}),
        "registrationReliable": bool(registration["registrationReliable"]),
        "registrationReliableFrameFraction": float(registration["registrationReliableFrameFraction"]),
        "medianTemporalMaskTurnover": float(turnover["temporalMaskTurnover"].median()) if not turnover.empty else np.nan,
        "algorithmVersion": DYNAMICS_ALGORITHM_VERSION,
    }
    return {"registration": registration, "tracking": tracking, "turnover": turnover, "summary": temporal_summary, "algorithmVersion": DYNAMICS_ALGORITHM_VERSION}
