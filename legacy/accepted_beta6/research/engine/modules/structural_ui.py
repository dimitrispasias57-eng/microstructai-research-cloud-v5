"""Streamlit presentation for the additive Morphology & Dynamics layer."""

from __future__ import annotations

import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.stats import theilslopes
from .ui_cache import remembered_view, remembered_upload

from .structural_intelligence import (
    compare_bulk_measurements,
    device_distance_features,
    estimate_device_influence_profile,
    find_similar_score_different_structure,
    future_mechanism_lab_schema,
    passport_markdown,
    structural_export_zip,
)


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value) if value is not None else "n/a"
    return f"{numeric:.{digits}f}" if np.isfinite(numeric) else "n/a"


def _metric_cards(values: list[tuple[str, Any]], columns: int = 4) -> None:
    containers = st.columns(columns)
    for idx, (label, value) in enumerate(values):
        containers[idx % columns].metric(label, _fmt(value))


def _selected_frame(structural: dict[str, Any], key: str) -> tuple[int, dict[str, Any]]:
    frames = structural.get("frames", [])
    if not frames:
        return 0, {}
    options = [f"{frame['metadata'].get('sourceFrameNumber', idx + 1)} · {frame['metadata'].get('filename', '')}" for idx, frame in enumerate(frames)]
    selected = st.select_slider("Frame", options=range(len(options)), format_func=lambda idx: options[idx], key=key)
    return int(selected), frames[int(selected)]


def _show_dataframe(table: pd.DataFrame, caption: str | None = None, height: int = 320) -> None:
    if caption:
        st.caption(caption)
    if table is None or table.empty:
        st.info("No QC-supported values are available for this view.")
    else:
        st.dataframe(table, use_container_width=True, height=height)


def _show_figure(fig) -> None:
    """Release the figure manager after rendering; avoid accumulation on navigation."""
    try:
        st.pyplot(fig, clear_figure=True)
    finally:
        plt.close(fig)


def render_structural_intelligence(structural: dict[str, Any] | None) -> None:
    """Render the complete, progressively disclosed research interface."""
    if not structural or not structural.get("enabled"):
        st.info("Run Conservative RBC Gap Mode to generate Morphology & Dynamics outputs.")
        return
    st.subheader("Morphology & Dynamics")
    st.info(
        "Primary representation: the selected final plasma-gap mask from the existing V3 workflow. "
        "Gap skeletons describe void-network architecture, not RBC rouleaux. Cell-rich domains are secondary and QC-gated."
    )
    if structural.get("warnings"):
        with st.expander("Advanced-analysis warnings", expanded=False):
            st.write("\n".join(structural["warnings"]))
    if st.session_state.get("_structural_export_owner") is not structural:
        st.session_state["_structural_export_owner"] = structural
        st.session_state.pop("_structural_export_bytes", None)
    if st.button("Prepare complete structural bundle", key="prepare_structural_bundle"):
        with st.spinner("Preparing the requested download…"):
            st.session_state["_structural_export_bytes"] = structural_export_zip(structural)
    if "_structural_export_bytes" in st.session_state:
        st.download_button(
            "Download complete structural-intelligence bundle (.zip)",
            data=st.session_state["_structural_export_bytes"],
            file_name="MicroStructAI_V3_structural_intelligence_bundle.zip",
            mime="application/zip", key="download_structural_intelligence_bundle", on_click="ignore",
        )

    names = [
        "Structural Fingerprint", "Gap Morphology", "Gap Sizes", "Gap Thickness",
        "Spatial Maps", "Topology", "Gap Network", "Contour Modes",
        "Cell-Rich Structure", "Dynamics", "Merge/Split Events", "Robustness & QC",
        "Device Interaction", "Bulk Comparator", "Mechanism Lab (Future)",
    ]
    selected_view = remembered_view("Explore structural results", names, "structural_results_view", dropdown=True)
    st.caption("Only the selected view is generated. The stored analysis and frame measurements are unchanged.")
    passport = structural.get("passport", {})
    tables = structural.get("tables", {})

    if selected_view == names[0]:
        existing = passport.get("existingMicroStructAI", {})
        population = passport.get("gapPopulation", {})
        scale = passport.get("gapSpaceScale", {})
        spatial = passport.get("spatialStructure", {})
        topology = passport.get("topology", {})
        network = passport.get("voidNetwork", {})
        st.markdown("#### Existing MicroStructAI")
        _metric_cards([(key, value) for key, value in existing.items()])
        st.markdown("#### Gap population")
        _metric_cards([(key, value) for key, value in population.items()])
        st.markdown("#### Gap space scale")
        _metric_cards([(key, value) for key, value in scale.items()])
        st.markdown("#### Spatial, topology and void-network architecture")
        _metric_cards([(key, value) for key, value in {**spatial, **topology, **network}.items()])
        reliability = passport.get("reliability", {})
        st.markdown("#### Reliability")
        st.json(reliability, expanded=False)
        st.download_button(
            "Download RBC Structural Passport (.md)",
            data=passport_markdown(passport).encode("utf-8"),
            file_name="RBC_Structural_Passport.md",
            mime="text/markdown",
            key="download_structural_passport_md",
        )
        st.caption("No arbitrary overall structural score is calculated.")

    if selected_view == names[1]:
        frame_summary = tables.get("gap_frame_summary", pd.DataFrame())
        objects = tables.get("gap_object_features", pd.DataFrame())
        _show_dataframe(frame_summary, "Frame-level robust summaries; border-truncated objects remain flagged.")
        with st.expander("All gap-object descriptors", expanded=False):
            _show_dataframe(objects, height=420)

    if selected_view == names[2]:
        _, frame = _selected_frame(structural, "structural_gap_sizes_frame")
        spectrum = frame.get("static", {}).get("sizeSpectrum", pd.DataFrame())
        objects = frame.get("static", {}).get("objects", pd.DataFrame())
        if not objects.empty:
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
            areas = pd.to_numeric(objects["area_px2"], errors="coerce").dropna()
            axes[0].hist(areas, bins=min(24, max(3, int(np.sqrt(len(areas))))), color="#2F6F8F")
            axes[0].set(xlabel="Gap area (px²)", ylabel="Objects", title="Gap area distribution")
            axes[1].hist(np.log10(areas[areas > 0]), bins=min(24, max(3, int(np.sqrt(len(areas))))), color="#2A9D8F")
            axes[1].set(xlabel="log10 gap area (px²)", ylabel="Objects", title="Log-area distribution")
            fig.tight_layout()
            _show_figure(fig)
        _show_dataframe(spectrum, "Adaptive bins use the Freedman–Diaconis rule on log-area.")

    if selected_view == names[3]:
        _, frame = _selected_frame(structural, "structural_thickness_frame")
        thickness = frame.get("static", {}).get("thicknessMap")
        if thickness is not None:
            fig, axis = plt.subplots(figsize=(7, 5))
            shown = axis.imshow(np.ma.masked_where(np.asarray(thickness) <= 0, thickness), cmap="magma")
            axis.set_title("Gap local thickness map")
            axis.axis("off")
            colorbar = fig.colorbar(shown, ax=axis)
            colorbar.set_label("Local gap thickness (px; 2 × EDT radius)")
            _show_figure(fig)
            st.caption("This is plasma-gap geometry, not RBC size.")
        summary = frame.get("static", {}).get("summary", {})
        _metric_cards([(key, value) for key, value in summary.items() if "Thickness" in key][:12])

    if selected_view == names[4]:
        _, frame = _selected_frame(structural, "structural_spatial_frame")
        maps = frame.get("static", {}).get("spatialMaps", {})
        if maps:
            selected = st.selectbox("Spatial metric", list(maps), key="structural_spatial_map_name")
            values = np.asarray(maps[selected], dtype=float)
            fig, axis = plt.subplots(figsize=(6, 5))
            shown = axis.imshow(np.ma.masked_invalid(values), cmap="viridis", interpolation="nearest")
            axis.set(title=selected.replace("_", " "), xlabel="Tile column", ylabel="Tile row")
            fig.colorbar(shown, ax=axis, label="metric units shown in title/table")
            _show_figure(fig)
        _show_dataframe(frame.get("static", {}).get("spatialTiles", pd.DataFrame()), "No interpolation is used between invalid tiles.")

    if selected_view == names[5]:
        _, frame = _selected_frame(structural, "structural_topology_frame")
        summary = frame.get("static", {}).get("summary", {})
        _metric_cards([(key, value) for key, value in summary.items() if "topology" in key.lower()][:12])
        multiscale = frame.get("static", {}).get("multiscaleTopology", pd.DataFrame())
        if not multiscale.empty:
            fig, axis = plt.subplots(figsize=(8, 4))
            for column in ["connectedComponents", "enclosedHoles", "eulerCharacteristic"]:
                axis.plot(multiscale["scale_px"], multiscale[column], marker="o", label=column)
            axis.set(xlabel="Morphological scale (px; negative=erosion, positive=dilation)", ylabel="Count / characteristic", title="Multiscale gap topology")
            axis.axvline(0, color="#777777", linewidth=1)
            axis.legend()
            axis.grid(alpha=0.25)
            _show_figure(fig)
        _show_dataframe(multiscale)

    if selected_view == names[6]:
        _, frame = _selected_frame(structural, "structural_network_frame")
        skeleton = frame.get("static", {}).get("skeleton")
        if skeleton is not None:
            fig, axis = plt.subplots(figsize=(7, 5))
            axis.imshow(skeleton, cmap="gray")
            axis.set_title("Plasma-gap / void-network medial skeleton")
            axis.axis("off")
            _show_figure(fig)
        summary = frame.get("static", {}).get("summary", {})
        _metric_cards([(key, value) for key, value in summary.items() if "gapNetwork" in key][:14])
        st.warning("The displayed skeleton is not an RBC rouleaux or individual-cell network.")

    if selected_view == names[7]:
        st.info("This is the outline of a bright plasma gap in the selected mask—not an individual RBC, "
                "a physical string or a measurement of membrane tension.")
        st.caption("Grey: original pixel-based outline. Blue: points spaced evenly along the outline. "
                   "Red: Gaussian-smoothed outline used to reduce pixel-scale jaggedness before Fourier shape analysis.")
        _, frame = _selected_frame(structural, "structural_contour_frame")
        modes = frame.get("static", {}).get("contourModes", pd.DataFrame())
        visuals = frame.get("static", {}).get("contourVisuals", [])
        if visuals:
            selected = st.selectbox("Gap contour object", range(len(visuals)), format_func=lambda idx: f"Object {visuals[idx]['gapObjectID']}", key="contour_object_select")
            item = visuals[int(selected)]
            fig, axis = plt.subplots(figsize=(6, 5))
            original, resampled, smoothed = item["originalContour"], item["resampledContour"], item["smoothedContour"]
            original = np.vstack([original, original[0]])
            smoothed = np.vstack([smoothed, smoothed[0]])
            axis.plot(original[:, 1], original[:, 0], color="#777777", label="Original outline")
            axis.plot(resampled[:, 1], resampled[:, 0], ".", color="#2F6F8F", markersize=2, label="resampled")
            axis.plot(smoothed[:, 1], smoothed[:, 0], color="#D1495B", label="Smoothed outline")
            axis.invert_yaxis()
            axis.set_aspect("equal")
            axis.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=3, fontsize=8)
            axis.set(xlabel="Horizontal position (pixels)", ylabel="Vertical position (pixels)")
            axis.set_title("Plasma-gap outline preparation", pad=42)
            fig.tight_layout()
            _show_figure(fig)
        else:
            st.info("No suitable fully observed gap contours were available for this frame.")
        st.caption("Fourier modes describe shape at different levels of detail: low modes describe broad shape; "
                   "higher modes describe finer boundary variations, which can also arise from pixels or segmentation noise. "
                   "The mode 'energy' is a mathematical share of squared shape amplitude, not physical energy. "
                   "Smoothing does not validate the boundary; robustness must be checked before interpreting these descriptors.")
        st.caption("Contour/Fourier modes describe a gap's outline. They are different from the POD modes used "
                   "in optional image preprocessing; viewing this panel does not enable POD removal.")
        _show_dataframe(modes, "Computed modes are exploratory until perturbation stability establishes the highest reliable order.")

    if selected_view == names[8]:
        _, frame = _selected_frame(structural, "structural_cellrich_frame")
        cell = frame.get("cellrich", {})
        mask = cell.get("mask")
        if mask is not None:
            st.image((np.asarray(mask, dtype=np.uint8) * 255), caption="Secondary dark/cell-rich connected-domain mask", use_container_width=True)
        summary = cell.get("summary", {})
        _metric_cards([(key, value) for key, value in summary.items() if isinstance(value, (int, float, np.number))])
        reliability = summary.get("cellRichReliability", "NOT_ASSESSED")
        if reliability in {"UNSTABLE", "UNRELIABLE"}:
            st.warning("Cell-rich representation unreliable. Gap representation retained as primary analysis.")
        else:
            st.info(f"Cell-rich representation stability: {reliability}. Results remain separate secondary evidence.")
        _show_dataframe(cell.get("robustness", pd.DataFrame()))

    if selected_view == names[9]:
        dynamics = structural.get("dynamics", {})
        summary = dynamics.get("summary", {})
        _metric_cards([(key, value) for key, value in summary.items() if isinstance(value, (int, float, bool, np.number))])
        _show_dataframe(dynamics.get("registration", {}).get("qc", pd.DataFrame()), "Registration precedes any displacement-dependent interpretation.")
        turnover = dynamics.get("turnover", pd.DataFrame())
        if not turnover.empty:
            fig, axis = plt.subplots(figsize=(8, 3.5))
            axis.plot(turnover["frameIndex"], turnover["temporalMaskTurnover"], color="#2F6F8F")
            axis.set(xlabel="Frame", ylabel="XOR area fraction", title="Mask-level structural turnover")
            axis.grid(alpha=0.25)
            _show_figure(fig)
        _show_dataframe(dynamics.get("tracking", {}).get("tracks", pd.DataFrame()), "Tracked identities are plasma-gap structures and are QC-gated.")

    if selected_view == names[10]:
        events = structural.get("dynamics", {}).get("tracking", {}).get("events", pd.DataFrame())
        if events.empty:
            st.info("No reliable merge/split events are available, or tracking was suppressed by registration QC.")
        else:
            event_types = st.multiselect("Event type", sorted(events["eventType"].unique()), default=[value for value in ["MERGE", "SPLIT"] if value in set(events["eventType"])])
            shown = events.loc[events["eventType"].isin(event_types)] if event_types else events
            _show_dataframe(shown, "Associations include confidence and ambiguity flags; ambiguous matches are not forced.", height=450)

    if selected_view == names[11]:
        dual = structural.get("dualRepresentation", {})
        st.json(dual.get("summary", {}), expanded=True)
        _show_dataframe(dual.get("table", pd.DataFrame()), height=420)
        with st.expander("Gap boundary-robustness perturbations", expanded=False):
            _show_dataframe(tables.get("gap_feature_robustness", pd.DataFrame()), height=420)
        with st.expander("Cell-rich threshold/blur perturbations", expanded=False):
            _show_dataframe(tables.get("cellrich_feature_robustness", pd.DataFrame()), height=420)
        st.caption("RELIABLE: median Dice ≥0.90 and median feature CV ≤0.10. MODERATE: Dice ≥0.75 and CV ≤0.25. Lower support is UNSTABLE/UNRELIABLE.")

    if selected_view == names[12]:
        st.markdown("#### Device Interaction Mode")
        st.caption(
            "Upload a researcher-defined binary stent/device mask. Automatic stent segmentation is intentionally not assumed. "
            "Distance is reported in pixels because physical calibration is not available in the current dataset."
        )
        uploaded_mask = remembered_upload(
            "Device mask (PNG, TIFF, JPEG; non-zero pixels are device)",
            types=["png", "tif", "tiff", "jpg", "jpeg"],
            key="structural_device_mask",
        )
        band_text = st.text_input("Distance-band edges (px)", value="0, 10, 25, 50, 100", key="device_band_edges")
        if uploaded_mask is None:
            st.info("No device mask supplied. Device-distance analysis is not calculated.")
        else:
            try:
                device = np.asarray(Image.open(io.BytesIO(uploaded_mask.getvalue())).convert("L")) > 0
                edges = [float(value.strip()) for value in band_text.split(",") if value.strip()]
                all_profiles = []
                for frame_index, frame in enumerate(structural.get("frames", []), start=1):
                    thickness = frame.get("static", {}).get("thicknessMap")
                    if thickness is None:
                        continue
                    gap_mask = np.asarray(thickness) > 0
                    frame_profile = device_distance_features(gap_mask, device, edges)
                    frame_profile.insert(0, "frameIndex", frame_index)
                    frame_profile.insert(1, "filename", frame.get("metadata", {}).get("filename", f"frame_{frame_index}"))
                    all_profiles.append(frame_profile)
                device_table = pd.concat(all_profiles, ignore_index=True) if all_profiles else pd.DataFrame()
                _show_dataframe(device_table, "Band-boundary clipping is explicitly flagged in the export.", height=400)
                if not device_table.empty:
                    st.download_button(
                        "Download device_distance_features.csv",
                        data=device_table.to_csv(index=False).encode("utf-8-sig"),
                        file_name="device_distance_features.csv",
                        mime="text/csv",
                        key="download_device_distance_features",
                    )
                    profile_metrics = [
                        column for column in [
                            "gapAreaFractionWithinBand", "medianLocalThickness_px", "spatialGapFractionCV_4x4",
                            "topologyEulerCharacteristic", "gapNetworkSkeletonDensity", "gapNetworkBranchDensity",
                            "gapObjectNematicOrder",
                        ] if column in device_table
                    ]
                    selected_metric = st.selectbox("Device influence profile metric", profile_metrics, key="device_profile_metric")
                    influence = estimate_device_influence_profile(device_table, selected_metric)
                    profile = influence.get("profile", pd.DataFrame())
                    if not profile.empty:
                        fig, axis = plt.subplots(figsize=(8, 3.8))
                        x = profile["distanceLower_px"].to_numpy(dtype=float)
                        mean = profile["mean"].to_numpy(dtype=float)
                        lower = profile["ci95Lower"].to_numpy(dtype=float)
                        upper = profile["ci95Upper"].to_numpy(dtype=float)
                        axis.plot(x, mean, marker="o", color="#2F6F8F")
                        axis.fill_between(x, lower, upper, alpha=0.22, color="#2F6F8F")
                        axis.set(xlabel="Lower edge of distance band (px)", ylabel=selected_metric, title="Device influence profile across frames")
                        axis.grid(alpha=0.25)
                        _show_figure(fig)
                    if influence.get("supported"):
                        st.success(f"Exploratory return-to-far-field distance: {_fmt(influence['influenceDistance_px'], 1)} px")
                    else:
                        st.info(f"Influence distance not reported: {influence.get('reason', 'insufficient support')}.")
                    st.warning("This profile is exploratory and does not establish a causal device effect.")
            except Exception as exc:
                st.error(f"Device analysis unavailable: {exc}")

    if selected_view == names[13]:
        st.markdown("#### External bulk-measurement / RheoScan comparator")
        st.caption("Upload sample-level tables and explicitly map their identifier columns. Missing measurement fields are allowed.")
        left, right = st.columns(2)
        structural_upload = remembered_upload("Sample structural summary CSV", ["csv"], "bulk_structural_csv", container=left)
        bulk_upload = remembered_upload("Bulk/RheoScan CSV", ["csv"], "bulk_measurement_csv", container=right)
        if structural_upload is None or bulk_upload is None:
            st.info("Upload both CSV files to activate matching, robust scatterplots, rank comparison, and pair discovery.")
        else:
            try:
                structural_table = pd.read_csv(io.BytesIO(structural_upload.getvalue()))
                bulk_table = pd.read_csv(io.BytesIO(bulk_upload.getvalue()))
                left, right = st.columns(2)
                structural_id = left.selectbox("Structural sample-ID column", structural_table.columns.tolist(), key="bulk_structural_id")
                bulk_id = right.selectbox("Bulk sample-ID column", bulk_table.columns.tolist(), key="bulk_bulk_id")
                comparison = compare_bulk_measurements(structural_table, bulk_table, structural_id, bulk_id)
                matched = comparison["matched"]
                st.metric("Matched samples", len(matched))
                _show_dataframe(comparison["correlations"], "Exploratory Spearman rank correlations; no causal interpretation.")
                if len(matched) >= 3:
                    structural_numeric = [column for column in structural_table.select_dtypes(include=[np.number]).columns if column in matched]
                    bulk_numeric = [column for column in bulk_table.select_dtypes(include=[np.number]).columns if column in matched]
                    if structural_numeric and bulk_numeric:
                        left, right = st.columns(2)
                        x_column = left.selectbox("Bulk metric", bulk_numeric, key="bulk_x_metric")
                        y_column = right.selectbox("Structural metric", structural_numeric, key="bulk_y_metric")
                        valid = matched[[x_column, y_column]].apply(pd.to_numeric, errors="coerce").dropna()
                        if len(valid) >= 3:
                            slope, intercept, _lo, _hi = theilslopes(valid[y_column], valid[x_column])
                            predicted = intercept + slope * valid[x_column]
                            rank_x = valid[x_column].rank(method="average")
                            rank_y = valid[y_column].rank(method="average")
                            plot_table = valid.copy()
                            plot_table["robustResidual"] = valid[y_column] - predicted
                            plot_table["bulkRank"] = rank_x
                            plot_table["structuralRank"] = rank_y
                            fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
                            axes[0].scatter(valid[x_column], valid[y_column], color="#2F6F8F")
                            order = np.argsort(valid[x_column].to_numpy())
                            axes[0].plot(valid[x_column].to_numpy()[order], predicted.to_numpy()[order], color="#D1495B")
                            axes[0].set(xlabel=x_column, ylabel=y_column, title="Theil–Sen robust fit")
                            axes[1].scatter(rank_x, rank_y, color="#2A9D8F")
                            axes[1].plot([1, len(valid)], [1, len(valid)], linestyle="--", color="#777777")
                            axes[1].set(xlabel="Bulk rank", ylabel="Structural rank", title="Rank comparison")
                            fig.tight_layout()
                            _show_figure(fig)
                            with st.expander("Robust residuals and ranks", expanded=False):
                                _show_dataframe(plot_table)

                    st.markdown("#### Same Bulk — Different Structure")
                    numeric_columns = matched.select_dtypes(include=[np.number]).columns.tolist()
                    default_scores = [column for column in bulk_numeric if column in numeric_columns][:2]
                    default_structure = [column for column in structural_numeric if column in numeric_columns][:6]
                    score_columns = st.multiselect("Bulk/existing-score columns", numeric_columns, default=default_scores, key="same_bulk_scores")
                    structural_columns = st.multiselect("Structural columns", numeric_columns, default=default_structure, key="same_bulk_structures")
                    pair_input = matched.copy()
                    pair_input["sampleID"] = pair_input[structural_id].astype(str)
                    pairs = find_similar_score_different_structure(pair_input, score_columns, structural_columns)
                    _show_dataframe(pairs.head(100), "Pairs are ranked for researcher discovery only; no biological significance is assigned.", height=420)
                    if not pairs.empty:
                        st.download_button(
                            "Download similar_score_different_structure.csv",
                            data=pairs.to_csv(index=False).encode("utf-8-sig"),
                            file_name="similar_score_different_structure.csv",
                            mime="text/csv",
                            key="download_same_bulk_pairs",
                        )
            except Exception as exc:
                st.error(f"Bulk comparison unavailable: {exc}")

    if selected_view == names[14]:
        st.markdown("#### Mechanism-Relevant Dynamics — Research Only")
        st.warning("Mechanism inference is not validated from current imaging data.")
        st.write(
            "Current exports preserve onset, growth, closure/coarsening proxies, merge/split rates, persistence, "
            "topology turnover, and registration QC where supported. Future controlled flow, breakup, recovery, "
            "hysteresis, and concentration-series data can be added through the documented interface below."
        )
        st.json(future_mechanism_lab_schema(), expanded=False)
        _show_dataframe(tables.get("gap_dynamic_summary", pd.DataFrame()), "Current mechanism-relevant measurements; no depletion/bridging label is produced.")
