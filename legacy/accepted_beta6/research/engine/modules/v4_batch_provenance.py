"""Reporting-only wrappers; preserve the original batch function bodies/values.

ContextVar makes collection local to the current call/thread, not a global
research-image cache. Only provenance/derived tables survive, not image arrays.
"""
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps

from .v4_pipeline import ensure_measurements
from .v4_report import json_data

_collector = ContextVar("microstructai_v4_batch_provenance", default=None)


def collect_sample(analysis):
    records = _collector.get()
    if records is None:
        return
    ensure_measurements(analysis)
    records.append({"sample_source": analysis.get("params", {}).get("image_folder_path", "not supplied"),
                    "context": deepcopy(analysis["v4_context"]),
                    "direct_metrics": analysis["v4_structural_metrics"].to_dict("records"),
                    "metric_availability": analysis["v4_metric_availability"].to_dict("records"),
                    "diagnostics": analysis["v4_diagnostics"].to_dict("records"),
                    "temporal_descriptors": analysis["v4_temporal"].to_dict("records")})


def with_batch_provenance(engine):
    @wraps(engine)
    def wrapped(*args, **kwargs):
        records = []
        token = _collector.set(records)
        try:
            result = engine(*args, **kwargs)
        finally:
            _collector.reset(token)
        result["v4_batch_provenance"] = {
            "scope": "V4 DEVELOPMENT. Not new independent validation. V3 native batch numbers remain unchanged.",
            "samples": records,
            "success_count": result.get("success_count"), "failure_count": result.get("failure_count"),
            "all_processing_warnings": result.get("warnings", []),
            "note": "Input failures and unprocessed selections remain in legacy failure rows/warnings. Missing metadata is not reconstructed or asserted."}
        suffix = batch_suffix(result)
        for key in ("report_md", "report_preview_md"):
            if key in result:
                result[key] += suffix
        return result
    return wrapped


def batch_suffix(result):
    provenance = result.get("v4_batch_provenance")
    if not provenance:
        return ""
    return ("\n\n## V4 segmentation / mask provenance / ingestion / ROI\n"
            "Complete per-sample context and direct descriptor availability follow. IMAGE QUALITY, MASK PLAUSIBILITY, "
            "and TEMPORAL / REFERENCE ELIGIBILITY remain separate diagnostic categories. PASS is not segmentation truth. "
            "CAUTION never silently removes raw values. RBC scores are experimental and are not generic descriptors.\n\n"
            "```json\n" + json_data(provenance).decode() + "\n```\n")


def with_batch_report_provenance(renderer):
    @wraps(renderer)
    def wrapped(result):
        full, preview, total, shown, exported = renderer(result)
        suffix = batch_suffix(result)
        # The legacy renderer may rebuild text from tables or use stored text.
        marker = "## V4 segmentation / mask provenance / ingestion / ROI"
        if suffix and marker not in full:
            full += suffix
        if suffix and marker not in preview:
            preview += suffix
        return full, preview, total, shown, exported
    return wrapped
