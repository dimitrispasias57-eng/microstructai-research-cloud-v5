"""Versioned adapters dispatch to preserved algorithms; no new segmentation math."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentationAdapter:
    name: str
    version: str
    domain: str
    implementation: str

    def provenance(self):
        return {"name": self.name, "version": self.version, "domain": self.domain,
                "implementation": self.implementation, "source": "Automatically segmented"}


def adapter_for(mode, params):
    if mode == "RBC Gap Mode":
        return SegmentationAdapter("RBC Plasma Gap — V3 compatible", "3.1-legacy-1", "RBC experimental",
                                   params.get("rbcAnalysisMethod", "Standard RBC Gap Mode"))
    if mode == "Object Mode":
        return SegmentationAdapter("Standard Objects — Legacy v1", "3.1-legacy-1", "Generic objects",
                                   params.get("objectAnalysisMethod", "Standard Object Mode"))
    raise ValueError("No automatic adapter for this route; supplied masks bypass segmentation.")


def run_legacy_adapter(engine, images, mode, params):
    adapter = adapter_for(mode, params)
    result = engine(images, mode, params)
    result["segmentation_adapter"] = adapter.provenance()
    return result
