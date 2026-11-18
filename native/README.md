# MicroStructAI-RBC Native Cloud v1

Self-contained CPU segmentation and supervised research morphology. No OpenAI,
Gemini or other inference service; no API key. This is a separate experimental
native model, not the historical Astra/M2 product.

## Supported input and limitations

- Already-centred **96 × 96 native-pixel, 8-bit grayscale DPC single-cell crops**.
- RGB is accepted only when all three channels are identical. No silent channel
  selection, resize, contrast correction, cropping or bit-depth conversion.
- No full-field detector. Arbitrary brightfield, phase contrast, stained blood,
  dense flow, other resolutions and other objects have not been validated.
- Maximum 50 observations and 16 MiB per batch, 1 MiB per file. Export before
  clearing the session. Repeated observations are not independent cells/donors.

The pipeline uses a centre-conditioned width-8 U-Net trained from scratch,
fixed probability threshold 0.5, the frozen component/filled-mask adapter,
unchanged historical M2 deterministic edge refinement and numerical QC, and
unchanged morphology formulas. The historical multimodal proposer/reviewer is
not run or simulated. Every output requires human review. No automatic acceptance.

## Evidence

`NATIVE_RBC_MODEL_MIXED` / `INTERNAL_VALIDATION_ONLY`.

Training: 4,000 crops/10 participants; validation: 500/5; held-out internal test:
454/5. Reference-assisted crop localisation. Published algorithmic references,
not manual segmentation ground truth. Two model candidates only.

Refined test participant-macro Dice 0.94571; IoU 0.90175; absolute relative area
error 8.13%; mean boundary distance 0.659 px. Frozen refinement reduced agreement
from raw Dice 0.95279; this negative result is retained, not tuned away.

Blinded expert quality audit (50 prospectively selected cases): 43 GOOD (86%),
6 MINOR_ERROR (12%), 1 MAJOR_ERROR (2%), no uncertain ratings. 49/50 (98%) GOOD or
MINOR. The 90% GOOD supportive threshold was not met. Five-participant cluster
95% intervals: GOOD 76–94%, GOOD+MINOR 94–100%, MAJOR 0–6%. One reviewer; small
sample. Numerical QC did not flag the observed major error. Human review remains
essential. No AUTO_ACCEPT denominator exists, so false-auto-accept and PPV are
not estimable, not measured zero-risk values.

These public data were used previously in project research. No independent
external validation of the native model is claimed. No diagnosis, disease,
clinical validity, aggregation, mechanical deformability or stiffness claim.

## Run and deploy

Use Python 3.13, `pip install -r requirements.txt`, then
`streamlit run app.py`. Linux installs CPU-only PyTorch. No secrets needed.
On Streamlit Community Cloud, choose this repository, `main`, `app.py`, and
Python 3.13. Scientific modules and model hashes are checked before inference.

Only model weights are cached globally. Images, predictions, reviews, report
bytes and filenames are session-local and are not written to shared app disk.
Hosting necessarily receives uploads; use only authorised non-identifiable
research data. Download the ZIP before closing; browser/session memory is not a
durable clinical record. This is not an authenticated clinical-data platform.

## Attribution

Training source: **erythroSight**, Shrestha, Li, Prakash, Merkeley, Kapoor, Pande
and Stoeber (2024), FRDR DOI [10.20383/103.0916](https://doi.org/10.20383/103.0916),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Changes: reference-outline rasterisation, centred cropping, train-only
augmentation and from-scratch model training. Published outlines were generated
algorithmically with Cellpose, not individually traced manual ground truth.
No Cellpose weights or Astra/M2 outputs were used for training. Dataset authors
do not endorse this application. No raw research images, participant mappings,
human identities or credentials are included in this app repository.

Local inference has no per-image API charge. Free hosting, if deployed, has
finite resources and no promise of unlimited scale. Local hardware/electricity
costs still exist. See the research project's deployment reports for measured
platform results; this README alone is not evidence of hosted validation.
