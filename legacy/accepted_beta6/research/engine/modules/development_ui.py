"""Development-only UI; scientific engine stays in the original V3 modules."""
import io

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from .development_report import build_sample_report, image_u8
from .development_safety import (inspect_inputs, interpretation_notes, manual_roi,
                                 metric_availability, qc_category_table, reference_diagnostics)
from .v4_ingestion import read_uploaded_image, load_sequence
from .segmentation import apply_crop


def roi_preview(files, params, mode):
    if not files or mode != "RBC Gap Mode" or params.get("rbcAnalysisMethod", "Standard RBC Gap Mode") == "Standard RBC Gap Mode":
        return None
    frames = load_sequence(files, stack_policy=params.get("v4StackPolicy", "reject"), selected_page=params.get("v4SelectedPage", 1))
    pixels = frames[0].pixels
    h, w = pixels.shape[:2]
    manual_roi(params, w, h)
    _, bounds = apply_crop(pixels, params)
    from .v4_report import roi_panel
    rect = [bounds["cropX0_0based"]+1, bounds["cropY0_0based"]+1,
            bounds["cropX1_0based_exclusive"]-bounds["cropX0_0based"], bounds["cropY1_0based_exclusive"]-bounds["cropY0_0based"]]
    return roi_panel(pixels, rect)


def input_controls(files, params, mode):
    effective = dict(params)
    with st.expander("Input checks, timing and optional channel ROI", expanded=False):
        st.caption("Checks run automatically before analysis. They never resize/delete frames. Source files and the locked PATEND preset remain unchanged.")
        stack_choice = st.selectbox("TIFF stack handling", ["Reject until explicitly chosen", "Pages are sequence frames", "One selected page"], key="dev_tiff_policy")
        if stack_choice != "Reject until explicitly chosen":
            effective["v4StackPolicy"] = "sequence" if stack_choice == "Pages are sequence frames" else "selected"
            if stack_choice == "One selected page":
                effective["v4SelectedPage"] = int(st.number_input("Selected TIFF page (1-based)", min_value=1, value=1, key="dev_tiff_page"))
            st.warning("This changes input interpretation. TIFF pages are not necessarily time points. Confirm timing and inspect the expanded frame manifest. Complex OME/Z/channel stacks require one explicit page.")
        label = st.text_input("Sample label (avoid patient identifiers)", key="dev_sample_label")
        modality = st.selectbox("Image modality (report metadata only)", ["Not specified", "Bright-field unstained", "DIC / phase contrast", "Stained smear", "Other"], key="dev_modality")
        st.caption(f"Engine frame rate: {params.get('matlab_fps', 1.0)} fps. Change it using the existing MATLAB fps / timing control. Confirm timing only if this rate describes the selected frames, including any subsampling.")
        timing = st.checkbox("I confirm the FPS and uniform time spacing of the selected frames", key="dev_timing_confirmed")
        scale_confirmed = st.checkbox("I have a measured spatial calibration", key="dev_scale_confirmed")
        scale, source = None, ""
        if scale_confirmed:
            scale = st.number_input("Micrometres per pixel", min_value=0.000001, value=1.0, format="%.6f", key="dev_um_per_pixel")
            source = st.text_input("Calibration source / date / objective", key="dev_scale_source")
            if not source.strip():
                st.warning("Enter the calibration source. Until then, the report will retain unknown spatial calibration.")
                scale = None
        st.caption("Spatial calibration is recorded as metadata only in this release. Existing metric columns and thresholds remain in pixels; camera segmentation presets are not a physical length calibration.")
        supported = mode == "RBC Gap Mode" and params.get("rbcAnalysisMethod", "Standard RBC Gap Mode") != "Standard RBC Gap Mode"
        experimental = st.checkbox("Experimental channel-interior ROI (override this run only)", value=False, disabled=not supported, key="dev_roi_enabled") if supported else False
        if experimental:
            rect = st.text_input("Channel interior: x, y, width, height (1-based pixels)", value="", key="dev_roi_rect", help="Example: 21,11,100,80 selects columns 21–120 and rows 11–90. Exclude walls/background yourself; no automatic channel detector is used.")
            effective.update(useCrop=True, cropMode="manual", manualCropRect=rect)
            st.warning("This changes the analysed region and can change all metrics. Treat it as a development experiment, not a reanalysis on the frozen PATEND protocol. Check the preview before interpreting results.")
        elif not supported:
            st.caption("The guided channel ROI is available with Conservative or Experimental POD-removed RBC mode. Existing Object/GT workflows retain their own settings.")
        if st.button("Check selected images and preview ROI", disabled=not files, key="dev_check_inputs"):
            with st.spinner("Checking image bytes and dimensions…"):
                try:
                    loaded = load_sequence(files, stack_policy=effective.get("v4StackPolicy", "reject"), selected_page=effective.get("v4SelectedPage", 1))
                    check = inspect_inputs(loaded, effective)
                except (ValueError, OSError) as exc:
                    check = {"ok": False, "errors": [str(exc)], "warnings": [], "records": []}
            show_preflight(check)
            if check["ok"]:
                png = roi_preview(files, effective, mode)
                if png:
                    st.image(png, caption="First selected image. Red: crop location. Display thumbnail only; full-resolution pixels are used for analysis.")
    return effective, {"sample_label": label, "modality": modality, "timing_confirmed": timing,
                       "time_interval_s": 1.0/float(params.get("matlab_fps", 1.0)) if timing else None,
                       "um_per_pixel": scale, "scale_source": source, "experimental_roi": experimental}


def show_preflight(check):
    for error in check["errors"]:
        st.error(error)
    if check["ok"]:
        st.success(f"{len(check['records'])} input files passed format/dimension checks — not biological validation.")
    for warning in check["warnings"]:
        st.warning(warning)
    st.dataframe(pd.DataFrame(check["records"]), hide_index=True)


def render_quality(analysis):
    if not analysis or analysis.get("metrics", pd.DataFrame()).empty:
        st.info("Run analysis to see metric-specific QC, calibration and reference diagnostics.")
        return
    st.subheader("What can this sample tell me?")
    for note in interpretation_notes(analysis):
        st.info(note)
    st.subheader("Metric availability — excluded is not zero")
    st.dataframe(metric_availability(analysis), hide_index=True)
    st.subheader("Three separate QC questions")
    if analysis.get("v4_context"):
        from .v4_pipeline import ensure_measurements
        ensure_measurements(analysis)
        for category in ("IMAGE QUALITY", "MASK PLAUSIBILITY", "TEMPORAL / REFERENCE ELIGIBILITY"):
            st.subheader(category)
            st.caption("PASS is not biological truth; CAUTION is not biological invalidity. These observable heuristics do not remove raw values or change legacy RBC gates.")
            table = analysis["v4_diagnostics"]
            from .v4_diagnostics import display_table
            st.dataframe(display_table(table.loc[table.category.eq(category)]), hide_index=True)
        st.subheader("Legacy per-metric gates (unchanged)")
    st.caption("Image quality, mask plausibility and temporal eligibility are different questions. Categories overlap; a warning does not always mean exclusion, and an unavailable check is not a pass. Stored rules/formulas have not changed.")
    st.dataframe(qc_category_table(analysis["metrics"]), hide_index=True)
    st.subheader("Why can scores move when masks look similar?")
    st.write("A score depends on the measurements AND the reference used to centre/scale them. QC or start-frame changes can change the reference. A high mask Dice therefore does not by itself prove score robustness. Raw median absolute deviation (MAD) describes reference spread; when it is zero, V3 adds only machine epsilon, so tiny changes can have a large score effect.")
    diagnostics = reference_diagnostics(analysis)
    if diagnostics.empty:
        st.info("Trusted SASI/GapScore reference diagnostics are unavailable for this run.")
    else:
        st.dataframe(diagnostics, hide_index=True)
    st.caption("Read-only diagnostic for trusted mask SASI/GapScore, not a full AggIndex sensitivity test. No new reference constants or clinical cut-offs are introduced.")
    with st.expander("Captured input check and settings"):
        context = analysis.get("development_context", {})
        if context.get("input_check"):
            show_preflight(context["input_check"])
        st.json(context)


def render_sample_report(analysis):
    if not analysis or analysis.get("metrics", pd.DataFrame()).empty:
        return
    # Session-local ownership cache. Never hash the large metric table on each view.
    if st.session_state.get("_dev_report_owner") is not analysis:
        with st.spinner("Building sample report from stored results (no reanalysis)…"):
            st.session_state["_dev_sample_report"] = build_sample_report(analysis)
        st.session_state["_dev_report_owner"] = analysis
    report = st.session_state["_dev_sample_report"]
    st.subheader("Clear sample report")
    st.write("Includes interpretation limits, metric availability, accepted/excluded frames, real image/mask panels, settings, calibration metadata and input hashes.")
    st.warning("Review filenames, labels and local paths before sharing. The report contains image thumbnails, but does not include original image files.")
    st.download_button("Download sample report (HTML)", report["html"], "MicroStructAI_sample_report.html", "text/html", key="dev_html_report", on_click="ignore")
    st.download_button("Download reproducible sample bundle (ZIP)", report["zip"], "MicroStructAI_sample_bundle.zip", "application/zip", key="dev_zip_report", on_click="ignore")
    st.caption("Open HTML in a browser; use Print → Save as PDF if needed. The native V3 text report below is retained for traceability.")
