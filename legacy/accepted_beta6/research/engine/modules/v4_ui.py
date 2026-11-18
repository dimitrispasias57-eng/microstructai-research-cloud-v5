"""Small explicit supplied-mask workflow; expensive analysis is button-triggered."""
import io
import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from .development_safety import input_selection_signature
from .v4_pipeline import prepare_supplied, analyze_supplied, ensure_measurements, digest
from .v4_report import build_v4_report, roi_panel
from .v4_ingestion import display_u8
from .structural_core import METRIC_NOTES


def render_v4_results(result):
    ensure_measurements(result)
    st.subheader("Direct structural measurements")
    st.caption("Mask source: " + result["v4_context"]["mask_source"] + ". Raw values remain visible despite cautions. Computational PASS does not verify biological segmentation.")
    section = st.radio("Structural results", ["Measurements", "Mask & ROI", "Diagnostics", "Temporal", "Report"], horizontal=True, key="v4_results_section")
    if section == "Measurements":
        st.dataframe(result["v4_structural_metrics"], hide_index=True)
        names = [c for c in result["v4_structural_metrics"] if c != "filename"]
        selected = st.selectbox("Direct descriptor to plot", names, key="v4_metric_plot")
        if len(result["v4_structural_metrics"]) == 1:
            value = result["v4_structural_metrics"][selected].iloc[0]
            st.metric(selected, f"{value:.6g}" if np.isfinite(value) else "Unavailable")
        else:
            plot = result["v4_structural_metrics"][[selected]].copy()
            plot.index = pd.RangeIndex(1, len(plot)+1, name="Frame (1-based)")
            st.line_chart(plot)
        st.caption("All finite raw values are plotted, including cautions. Unavailable values are NaN, not zero. Units: " + result["v4_metric_availability"].set_index("metric").loc[selected, "units"])
        st.info(METRIC_NOTES.get(selected, "Descriptor of the selected binary foreground, not automatically a cell or aggregate measurement."))
        with st.expander("Component distribution / metric availability"):
            st.dataframe(result["v4_component_distribution"], hide_index=True)
            st.dataframe(result["v4_metric_availability"], hide_index=True)
    elif section == "Mask & ROI":
        preview = st.selectbox("Frame to inspect", range(len(result["previews"])), format_func=lambda i: result["previews"][i]["filename"])
        _show_preview(result["previews"][preview])
    elif section == "Diagnostics":
        st.warning("Heuristics are not a mask-accuracy detector. CAUTION is not a biological rejection; PASS is not truth. No generic warning hides raw measurements.")
        for category in ("IMAGE QUALITY", "MASK PLAUSIBILITY", "TEMPORAL / REFERENCE ELIGIBILITY"):
            st.subheader(category)
            table = result["v4_diagnostics"]
            from .v4_diagnostics import display_table
            st.dataframe(display_table(table.loc[table.category.eq(category)]), hide_index=True)
    elif section == "Temporal":
        st.info("Unregistered pixel-mask overlap/change only. Motion can look like structural change. Single masks have no temporal descriptors; seconds require confirmed timing.")
        st.dataframe(result["v4_temporal"], hide_index=True)
    else:
        if st.session_state.get("v4_report_owner") is not result:
            with st.spinner("Building report from stored results…"):
                st.session_state["v4_report_bundle"] = build_v4_report(result)
            st.session_state["v4_report_owner"] = result
        bundle = st.session_state["v4_report_bundle"]
        st.warning("Review filenames, paths and image thumbnails before sharing. No source images are included in the ZIP.")
        st.download_button("Download V4 report (HTML)", bundle["html"], "MicroStructAI_V4_report.html", "text/html", on_click="ignore")
        st.download_button("Download V4 reproducibility bundle (ZIP)", bundle["zip"], "MicroStructAI_V4_bundle.zip", "application/zip", on_click="ignore")
        st.download_button("Download direct metrics + provenance (CSV)", bundle["csv"], "MicroStructAI_V4_metrics.csv", "text/csv", on_click="ignore")
        with st.expander("Captured settings / mask provenance"):
            st.json(result["v4_context"])


def _show_preview(frame):
    if frame.get("original_image") is not None:
        st.image(roi_panel(frame["original_image"], frame["provenance"]["roi_1based_xywh"]), caption="Original, selected ROI and crop. Display scaling only; analysis preserves original numeric pixels.")
    columns = st.columns(2)
    columns[0].image(frame["mask"].astype(np.uint8)*255, caption="Selected foreground = white. Check polarity and crop.")
    if frame.get("image") is not None:
        from .development_report import mask_panel
        panel = mask_panel(frame)
        if panel:
            columns[1].image(panel, caption="Image, mask and boundary: check correspondence yourself.")
    else:
        columns[1].info("Mask-only input: image quality and biological phase identity are not assessed.")
    st.json(frame["provenance"], expanded=False)


def verified_mask_workspace():
    st.header("Verified Mask Analysis")
    st.info("Here ‘verified’ means format, polarity and alignment checks plus your preview review—not independent biological validation. Automatic segmentation is BYPASSED.")
    input_type = st.radio("Input type", ["Mask only", "Image + corresponding mask"], horizontal=True, key="v4_input_type")
    masks = st.file_uploader("Upload mask files", type=["png", "tif", "tiff", "bmp"], accept_multiple_files=True, key="v4_masks") or []
    images = st.file_uploader("Upload corresponding images", type=["png", "tif", "tiff", "jpg", "jpeg", "bmp"], accept_multiple_files=True, key="v4_images") or [] if input_type != "Mask only" else []
    options = {"Binary (zero / one nonzero value)": "binary", "Labels (nonzero labels → foreground union)": "labels", "Grayscale (explicit threshold)": "threshold", "RGB mask (any nonzero channel → foreground)": "rgb_nonzero"}
    interpretation = options[st.selectbox("Mask interpretation", list(options), key="v4_interpretation")]
    polarity = st.selectbox("Foreground polarity — choose explicitly", ["Choose…", "nonzero/high", "zero/low"], key="v4_polarity")
    threshold = st.number_input("Mask threshold in original numeric units", value=127.0, key="v4_mask_threshold") if interpretation == "threshold" else None
    policies = {"Reject stacks until I choose": "reject", "Treat TIFF pages as sequence frames": "sequence", "Analyse one selected TIFF page": "selected"}
    policy = policies[st.selectbox("TIFF pages / stack interpretation", list(policies), key="v4_stack")]
    page = int(st.number_input("TIFF page (1-based)", min_value=1, value=1, key="v4_page")) if policy == "selected" else 1
    roi = None
    use_roi = st.checkbox("Experimental rectangular ROI — same coordinates for every frame", key="v4_roi_enabled")
    if use_roi:
        fields = st.columns(4)
        roi = [int(c.number_input(label, min_value=1, value=default, key="v4_roi_"+label)) for c, label, default in zip(fields, ["x", "y", "width", "height"], [1, 1, 100, 100])]
        st.caption("Coordinates are 1-based. Out-of-bounds or inconsistent dimensions are rejected, never silently clipped/resized.")
    pairing = False
    if images:
        from .io_utils import natural_sort_key
        st.dataframe(pd.DataFrame({"image (natural order)": pd.Series([f.name for f in sorted(images, key=natural_sort_key)]), "mask (natural order)": pd.Series([f.name for f in sorted(masks, key=natural_sort_key)])}), hide_index=True)
        pairing = st.checkbox("I confirm this one-to-one image/mask pairing (and matching TIFF page order)", key="v4_pairing")
    with st.expander("Sample label, calibration and timing"):
        label = st.text_input("Sample label (no patient identifiers)", key="v4_sample_label")
        timing = st.checkbox("I confirm these are consecutive time frames in the displayed order", key="v4_timing")
        interval = st.number_input("Time interval between frames (seconds)", min_value=.000001, value=1.0, key="v4_interval") if timing else None
        scale_yes = st.checkbox("Measured spatial calibration available", key="v4_scale")
        scale = st.number_input("Micrometres per pixel", min_value=.000001, value=1.0, key="v4_scale_value") if scale_yes else None
        scale_source = st.text_input("Spatial calibration source / date", key="v4_scale_source") if scale_yes else ""
        st.caption("Calibration metadata is recorded. Native metrics and size filters remain pixels/pixels²; no silent conversion.")
    acquisition = {"sample_label": label, "timing_confirmed": timing, "time_interval_s": interval,
                   "um_per_pixel": scale, "scale_source": scale_source, "experimental_roi": use_roi}
    signature = digest({"masks": input_selection_signature(masks), "images": input_selection_signature(images),
                        "interpretation": interpretation, "foreground": polarity, "threshold": threshold,
                        "policy": policy, "page": page, "roi": roi, "pairing": pairing, "acquisition": acquisition, "input_type": input_type})
    if st.button("1. Validate inputs and show mask preview", key="v4_preview", disabled=not masks):
        try:
            if input_type != "Mask only" and not images:
                raise ValueError("Image + mask mode requires corresponding images.")
            with st.spinner("Checking bytes, dimensions, orientation, polarity and ROI…"):
                prepared = prepare_supplied(masks, images, interpretation=interpretation, foreground=polarity,
                                             threshold=threshold, stack_policy=policy, selected_page=page,
                                             roi=roi, pairing_confirmed=pairing, acquisition=acquisition)
            st.session_state["v4_prepared"] = prepared
            st.session_state["v4_prepared_signature"] = signature
            st.session_state["v4_preview_accepted"] = False
        except (ValueError, OSError) as exc:
            st.error(str(exc))
            st.session_state.pop("v4_prepared", None)
    prepared = st.session_state.get("v4_prepared")
    current = prepared is not None and st.session_state.get("v4_prepared_signature") == signature
    if prepared and not current:
        st.warning("Inputs/settings changed. Validate again; the earlier preview cannot authorize this run.")
    if current:
        st.success("Format checks complete. MASK SOURCE: User supplied. No automatic segmentation has run.")
        for warning in prepared["warnings"]:
            st.warning(warning)
        pos = st.selectbox("Preview frame", range(len(prepared["frames"])), format_func=lambda i: prepared["frames"][i]["filename"], key="v4_preview_position")
        _show_preview(prepared["frames"][pos])
        accepted = st.checkbox("I reviewed foreground polarity, orientation, pairing and ROI in the previews", key="v4_preview_accepted")
        if st.button("2. Analyse the supplied masks", disabled=not accepted, key="v4_analyse"):
            try:
                with st.spinner("Measuring stored masks — segmentation bypassed…"):
                    st.session_state["v4_mask_analysis"] = analyze_supplied(prepared)
                    st.session_state["v4_result_signature"] = signature
            except (ValueError, OSError) as exc:
                st.error(str(exc))
    result = st.session_state.get("v4_mask_analysis")
    if result:
        if st.session_state.get("v4_result_signature") != signature:
            st.warning("Results below belong to the previous completed run; inputs/settings now differ.")
        render_v4_results(result)
