"""V4 provenance sections and self-contained reports. Stored masks only."""
import base64
import html
import io
import json
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .v4_ingestion import display_u8
from .v4_pipeline import ensure_measurements
from .structural_core import METRIC_NOTES


def json_data(value):
    from .development_report import json_bytes
    return json_bytes(value)


def roi_panel(original, roi):
    """Original, ROI overlay, crop; display normalization never enters analysis."""
    from .v4_masks import crop, validate_roi
    rect = validate_roi(roi, original.shape)
    base = Image.fromarray(display_u8(original)).convert("RGB")
    overlay = base.copy()
    x, y, w, h = rect
    ImageDraw.Draw(overlay).rectangle((x-1, y-1, x+w-2, y+h-2), outline=(230, 45, 45), width=max(1, base.width//200))
    cropped = Image.fromarray(display_u8(crop(original, rect))).convert("RGB")
    panels = [base, overlay, cropped]
    canvas = Image.new("RGB", (1200, 330), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (panel, label) in enumerate(zip(panels, ["Original (display scaled)", "ROI overlay (1-based coordinates)", "Analysis crop (display scaled)"])):
        panel.thumbnail((395, 295))
        canvas.paste(panel, (400*i, 30))
        draw.text((400*i+4, 5), label, fill="black")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def provenance_sections(analysis):
    ensure_measurements(analysis)
    context = analysis["v4_context"]
    esc = html.escape
    blocks = ["<h2>Segmentation source</h2><pre>" + esc(json_data(context["segmentation"]).decode()) + "</pre>",
              "<p>Mask source: <strong>" + esc(context["mask_source"]) + "</strong>. Successful computation is not evidence that the mask is biologically correct.</p>",
              f"<p>App {esc(context['app_version'])}; schema {esc(context['analysis_schema'])}; core {esc(context['structural_core_version'])}. Historical V3 fields retain legacy schema 3.1.</p>"]
    frames = context["frames"]
    for title, payload in [
        ("Mask provenance", [{k: v for k, v in f.items() if k not in ("image_ingestion", "mask_ingestion")} for f in frames]),
        ("Image ingestion", [{"filename": f["filename"], "image": f.get("image_ingestion"), "mask": f.get("mask_ingestion")} for f in frames]),
        ("ROI", [{"filename": f["filename"], "roi_1based_xywh": f.get("roi_1based_xywh"), "original_dimensions": f.get("original_mask_dimensions"), "analysed_dimensions": f.get("analysed_dimensions")} for f in frames]),
        ("Calibration and timing", context.get("acquisition", {})),
    ]:
        blocks.append(f"<h2>{title}</h2><details><summary>All captured records (expand)</summary><pre>{esc(json_data(payload).decode())}</pre></details>")
    blocks.append("<p>Spatial scale is user-declared metadata. Metrics remain pixels/pixels²; no hidden physical calibration. Horizontal Xi is along image rows, not automatically the flow direction or a cell/aggregate size.</p>")
    diagnostics = analysis["v4_diagnostics"]
    for category in ("IMAGE QUALITY", "MASK PLAUSIBILITY", "TEMPORAL / REFERENCE ELIGIBILITY"):
        blocks.append(f"<h2>{category}</h2><p>PASS means no listed heuristic fired, not biological truth. CAUTION does not mean biologically invalid. FAIL is an explicit input/computation failure. Heuristics do not remove raw metrics.</p>")
        blocks.append(diagnostics.loc[diagnostics.category.eq(category)].to_html(index=False, escape=True))
    blocks.append("<h2>Direct structural descriptors: availability and units</h2>" + analysis["v4_metric_availability"].to_html(index=False, escape=True))
    blocks.append("<h2>Metric definitions and cautions</h2><ul>" + "".join(f"<li><strong>{esc(k)}</strong>: {esc(v)}</li>" for k, v in METRIC_NOTES.items()) + "</ul>")
    blocks.append("<p>Temporal overlap/turnover is measured in the recorded pixel coordinates, WITHOUT registration. It can reflect motion as well as structural change. Empty/empty overlap is 1 by legacy convention and is not evidence of persistent biological structure.</p>")
    if context["segmentation"].get("domain") == "RBC experimental":
        blocks.append("<p>Experimental RBC-specific derived indices (SASI, GapScore, PTI, AggIndex) use the unchanged V3 route and its reference/QC gates. New generic cautions do not overwrite those gates. Their biological meaning and equivalence to commercial indices are not validated.</p>")
    else:
        blocks.append("<p>No RBC composite indices are reported for this generic mask route. Structural measurement does not establish a biological phase identity.</p>")
    return "\n".join(blocks)


def export_members(analysis):
    ensure_measurements(analysis)
    result = {"v4_run_provenance.json": json_data(analysis["v4_context"]),
              "v4_settings.json": json_data(analysis.get("params", {})),
              "v4_metric_definitions.json": json_data(METRIC_NOTES)}
    for key in ("v4_structural_metrics", "v4_component_distribution", "v4_metric_availability", "v4_diagnostics", "v4_temporal"):
        result[key + ".csv"] = analysis[key].to_csv(index=False).encode("utf-8-sig")
    # Per-row provenance accompanies even the standalone generic CSV.
    table = analysis["v4_structural_metrics"].copy()
    context = analysis["v4_context"]
    rows = context["frames"]
    if len(rows) == len(table):
        table["app_version"] = context["app_version"]
        table["analysis_schema"] = context["analysis_schema"]
        table["segmentation_source"] = context["mask_source"]
        table["settings_sha256"] = context["settings_sha256"]
        table["frame_provenance_json"] = [json_data(r).decode() for r in rows]
        table["acquisition_json"] = json_data(context.get("acquisition", {})).decode()
        table["warnings_json"] = json_data(analysis.get("warnings", [])).decode()
        table["metric_availability_json"] = json_data(analysis["v4_metric_availability"].to_dict("records")).decode()
        table["segmentation_module_json"] = json_data(context["segmentation"]).decode()
    result["v4_metrics_with_provenance.csv"] = table.to_csv(index=False).encode("utf-8-sig")
    return result


def build_v4_report(analysis):
    ensure_measurements(analysis)
    ctx = analysis["v4_context"]
    esc = html.escape
    label = ctx.get("acquisition", {}).get("sample_label") or "MicroStructAI V4 structural report"
    blocks = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>MicroStructAI V4 report</title>",
              "<style>body{font:15px/1.55 system-ui;max-width:1250px;margin:32px auto;padding:20px;color:#183044}table{border-collapse:collapse;font-size:12px;display:block;overflow-x:auto}td,th{border:1px solid #ccd7df;padding:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%}h1,h2{color:#14536b}</style>",
              f"<h1>{esc(label)}</h1><p>DEVELOPMENT / REGRESSION — research use only. Internal checks, not independent or biological validation.</p>",
              "<h2>Run warnings</h2><ul>" + "".join(f"<li>{esc(str(w))}</li>" for w in analysis.get("warnings", [])) + "</ul>",
              provenance_sections(analysis), "<h2>Direct measurements (all frames, including cautions)</h2>",
              analysis["v4_structural_metrics"].to_html(index=False, escape=True, na_rep="Unavailable"),
              "<h2>Temporal pair descriptors</h2>", analysis["v4_temporal"].to_html(index=False, escape=True, na_rep="Unavailable")]
    files = export_members(analysis)
    from .development_report import mask_panel
    previews = analysis.get("previews", [])
    for i in sorted({0, len(previews)//2, len(previews)-1}) if previews else []:
        p = previews[i]
        panel_input = dict(p)
        if panel_input.get("image") is None:
            panel_input["image"] = p["mask"].astype(np.uint8)*255
        panel = mask_panel(panel_input)
        if panel:
            files[f"figures/mask_{i+1}.png"] = panel
            blocks += [f"<h2>Internal preview: {esc(p['filename'])}</h2><p>Display-only scaling. Mask-only views repeat the mask in the image position; there is no underlying microscopy image. First/middle/last examples, not independent validation.</p>",
                       "<img alt='Image mask and boundary preview' src='data:image/png;base64," + base64.b64encode(panel).decode() + "'>"]
        if p.get("original_image") is not None:
            roi = roi_panel(p["original_image"], p["provenance"]["roi_1based_xywh"])
            files[f"figures/roi_{i+1}.png"] = roi
            blocks.append("<img alt='Original image ROI overlay and crop' src='data:image/png;base64," + base64.b64encode(roi).decode() + "'>")
    blocks += ["<h2>Complete reproducibility context</h2><details><summary>Expand JSON</summary><pre>" + esc(json_data(ctx).decode()) + "</pre></details>",
               "<p>Original source files are not embedded. The ZIP includes per-input hashes, final mask hashes, native units, settings and diagnostic values. Review image thumbnails and filenames before sharing.</p></html>"]
    markup = "\n".join(blocks).encode("utf-8")
    files["sample_report.html"] = markup
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return {"html": markup, "zip": buffer.getvalue(), "csv": files["v4_metrics_with_provenance.csv"]}
