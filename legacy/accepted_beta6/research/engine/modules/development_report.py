"""Self-contained sample report from stored analysis; never rerun segmentation."""
from __future__ import annotations

import base64
import html
import io
import json
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .development_safety import interpretation_notes, metric_availability, qc_category_table, reference_diagnostics


def json_bytes(value):
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple, np.ndarray)):
            return [clean(x) for x in v]
        if isinstance(v, (float, np.floating)):
            return float(v) if np.isfinite(v) else None
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v
    return json.dumps(clean(value), indent=2, ensure_ascii=False, default=str, allow_nan=False).encode("utf-8")


def image_u8(arr):
    from .v4_ingestion import display_u8
    return display_u8(np.nan_to_num(np.asarray(arr), nan=0, posinf=0, neginf=0))


def mask_panel(preview):
    """Actual analysis image, final binary mask and red boundary, display only."""
    from scipy.ndimage import binary_erosion
    mask = preview.get("mask", preview.get("gt_filled_mask"))
    raw = preview.get("camera_crop_gray", preview.get("image"))
    if mask is None or raw is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    image = Image.fromarray(image_u8(raw)).convert("RGB")
    if image.size != (mask.shape[1], mask.shape[0]):
        return None  # never align a mask by silently resizing it
    overlay = np.array(image)
    overlay[mask & ~binary_erosion(mask)] = [230, 45, 45]
    panels = [image, Image.fromarray(mask.astype(np.uint8) * 255).convert("RGB"), Image.fromarray(overlay)]
    for panel in panels:
        panel.thumbnail((440, 330))  # report thumbnail only, not analysis input
    width, height = panels[0].size
    canvas = Image.new("RGB", (3 * width, height + 26), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, ("Analysis crop", "Final mask", "Mask boundary"))):
        canvas.paste(panel, (index * width, 26))
        draw.text((index * width + 5, 6), label, fill="black")
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def detailed_frame_table(analysis):
    df = analysis["metrics"]
    names = ["filename", "originalImageIndex_1based", "analysisIndex_1based", "matlabTime_s",
             "usedInSummary", "usedForMainSummary", "trustedForScore", "rbcScoreQCReason", "score_QC_reason",
             "hardBadMaskFlag", "warningMaskFlag", "hugeGapFlag", "sparseMaskMorphologyFiltered",
             "Xi_QC", "SASI_QC_0_100", "AggIndex_QC_0_100"]
    return df[[c for c in names if c in df]].copy()


def build_sample_report(analysis):
    """Build on-demand once per result in session cache; no private uploads embedded."""
    ctx = analysis.get("development_context", {})
    df = analysis["metrics"]
    tables = {
        "metric_availability": metric_availability(analysis),
        "qc_categories": qc_category_table(df),
        "reference_diagnostics": reference_diagnostics(analysis),
        "frame_decisions": detailed_frame_table(analysis),
        "input_manifest": pd.DataFrame(ctx.get("input_check", {}).get("records", [])),
    }
    notes = interpretation_notes(analysis)
    title = str(ctx.get("acquisition", {}).get("sample_label", "") or "Sample structural-analysis report")
    esc = html.escape
    blocks = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>MicroStructAI sample report</title>",
              "<style>body{font:16px/1.55 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#183044}h1,h2{color:#14536b}table{border-collapse:collapse;font-size:12px;display:block;overflow-x:auto}th,td{border:1px solid #cad4df;padding:6px;text-align:left}img{max-width:100%}.caution{background:#fff4d5;padding:18px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>",
              f"<h1>{esc(title)}</h1><p>MicroStructAI development copy • {esc(str(ctx.get('created_utc', 'Date unavailable')))}</p>",
              f"<p>Mode: {esc(str(analysis.get('mode')))}. Processed frames: {len(df)}. Core formulas preserved from V3; development changes are not independently validated.</p>",
              "<div class='caution'><h2>What can be concluded?</h2><ul>" + "".join(f"<li>{esc(n)}</li>" for n in notes) + "</ul></div>"]
    warnings = ctx.get("input_check", {}).get("warnings", []) + analysis.get("warnings", [])
    blocks += ["<h2>Run warnings</h2><ul>" + "".join(f"<li>{esc(str(w))}</li>" for w in dict.fromkeys(warnings)) + "</ul>"]
    blocks += ["<h2>How to read the measurements</h2>",
               "<p>Gap area fraction describes how much of the analysed crop belongs to the selected bright phase. Gap count counts connected regions, not red blood cells. Equivalent gap diameter is the diameter of a circle with the same area, not a cell diameter. Horizontal correlation length ξ measures how far the binary pattern stays similar along image rows; it depends on orientation and is not a literal gap length.</p>",
               "<p>SASI (sometimes called SASSI in earlier discussion) is the code's mask-structure score: 0.40 z(ξ) + 0.40 z(diameter) − 0.20 z(count), followed by a 0–100 logistic transform. GapScore combines 0.35 z(diameter) + 0.25 z(area fraction) + 0.25 z(ξ) − 0.15 z(count). The z values use a reference median and raw median absolute deviation (MAD), with machine epsilon in the denominator.</p>",
               "<p>The trusted RBC AggIndex is 0.50 GapScore + 0.30 PTI + 0.20 image-field structure score. The legacy code substitutes 50 for missing component values inside this mixture and additionally gates the final result on eligible QC and finite GapScore. This is disclosed, not changed here. AggIndex is not simply another name for SASI, and neither is the commercial optical aggregation index. Fixed mask-score references do not make all AggIndex components cross-sample calibrated.</p>",
               "<p>Lacunarity describes unevenness of occupied/empty regions at the stated pixel scale. Topology and contours describe connectivity and the shape of selected gaps; a contour is not the average shape of all RBCs. Correlation/PCA describe associations in these inputs, not independent biological validation.</p>"]
    table_titles = {"metric_availability": "Measurements and availability", "qc_categories": "Three separate QC questions", "reference_diagnostics": "SASI/GapScore reference diagnostics (read-only)", "frame_decisions": "Every processed frame: retained/excluded decisions", "input_manifest": "Selected input manifest"}
    for key, table in tables.items():
        blocks.append(f"<h2>{table_titles[key]}</h2>")
        if key == "metric_availability":
            blocks.append("<p>NaN means unavailable, never zero. QC counts describe stored engine gates; reportable n additionally withholds one-image sequence scores. Medians of raw descriptors with QC unavailable are unassessed measurements, not approved results.</p>")
        if key == "qc_categories":
            blocks.append("<p>These are overlapping diagnostic categories, not three independent validation tests. A warning flag is not always a hard rejection. Use the exact metric-specific accepted/excluded counts above; a missing diagnostic means unassessed, not passed.</p>")
        if key == "reference_diagnostics":
            blocks.append("<p>Reconstructed using the unchanged V3 helper and stored trusted-score inputs. These statistics cover SASI/GapScore only, not all PTI or AggIndex components. Zero MAD or a fallback reference deserves caution even when masks look stable.</p>")
        blocks.append(table.to_html(index=False, escape=True, na_rep="Unavailable") if not table.empty else "<p>Not available for this analysis.</p>")
    if analysis.get("v4_context"):
        from .v4_report import provenance_sections
        blocks.append(provenance_sections(analysis))
    images = {}
    if analysis.get("development_roi_png"):
        images["roi_preview.png"] = analysis["development_roi_png"]
        blocks.append("<h2>First selected image: crop location</h2><p>Red rectangle is the selected crop. This is a display thumbnail; no image resampling was introduced into analysis.</p>")
        blocks.append("<img alt='Crop preview' src='data:image/png;base64," + base64.b64encode(images["roi_preview.png"]).decode() + "'>")
    previews = analysis.get("previews", [])
    if previews:
        positions = {0, len(previews) // 2, len(previews) - 1}
        # Include the first excluded frame as well, if a matching preview exists.
        if "trustedForScore" in df and "filename" in df:
            from .qc_plotting import bool_values
            excluded = set(df.loc[~bool_values(df["trustedForScore"]), "filename"].astype(str))
            first_excluded = next((i for i, p in enumerate(previews) if str(p.get("filename")) in excluded), None)
            if first_excluded is not None:
                positions.add(first_excluded)
        blocks.append("<h2>Internal image/overlay check</h2><p>First, middle, last available preview and first excluded frame (when available). These are deterministic examples, not selected to look favourable and not independent validation. Red outlines follow the stored final masks. Images are display thumbnails.</p>")
        for pos in sorted(positions):
            preview = previews[pos]
            png = mask_panel(preview)
            if png:
                name = f"mask_check_{pos+1}.png"
                images[name] = png
                blocks.append(f"<h3>{esc(str(preview.get('filename', pos+1)))}</h3><img alt='Image, mask and overlay' src='data:image/png;base64,{base64.b64encode(png).decode()}'>")
    blocks.append("<h2>Reproducibility</h2><p>The ZIP contains the complete native metrics, exact pre-run and resolved engine settings, source-code hashes, per-input hashes, frame decisions and this report. Original images are not included. Filenames/settings may still contain identifiers or local paths: review before sharing. The selected-input manifest can be longer than processed output when V3's configured frame-window selection applies.</p>")
    blocks.append("<details><summary>Captured run context</summary><pre>" + esc(json_bytes(ctx).decode()) + "</pre></details></html>")
    report_html = "\n".join(blocks).encode("utf-8")
    text = title + "\n\n" + "\n\n".join(notes)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("sample_report.html", report_html)
        z.writestr("interpretation.txt", text.encode("utf-8"))
        z.writestr("run_context.json", json_bytes(ctx))
        z.writestr("resolved_engine_settings.json", json_bytes(analysis.get("params", {})))
        z.writestr("native_metrics.csv", df.to_csv(index=False).encode("utf-8-sig"))
        for key, table in tables.items():
            z.writestr(f"{key}.csv", table.to_csv(index=False).encode("utf-8-sig"))
        for name, png in images.items():
            z.writestr("figures/" + name, png)
        if analysis.get("v4_context"):
            from .v4_report import export_members
            for name, data in export_members(analysis).items():
                z.writestr(name, data)
    return {"html": report_html, "zip": payload.getvalue(), "notes": notes, "tables": tables}
