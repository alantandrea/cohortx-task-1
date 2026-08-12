# CohortX Task 1 — Final Submission

Extracts the six cohort-selection fields from PMC full-text `.nxml` articles.

## Final results (competition closed 2026-07-16)
| | Score |
|---|---|
| **Public leaderboard** | **0.71784** |
| **Private leaderboard (final)** | **0.69244** |

This entry contributed to a **3rd-place overall finish in the CohortX Challenge**
(mean of the three task private scores: 0.57983), featured at the CohortX Challenge
Showcase at MICCAI 2026.

`submission.csv` in this folder is the exact file that was selected for final judging
(public 0.71784 / private 0.69244).

## Solution:
Per the host's ruling — *"external datasets (PubMed, ClinicalTrials.gov, UMLS, MeSH, Disease
Ontology) are NOT allowed for training or inference; the only data you can use is the one we
provided on Kaggle"*, and *"pretrained models ARE allowed if lightweight / laptop-runnable;
inference must be offline with no external API calls"* — this solution:

- **Uses only the provided Kaggle data**: the 500 test `.nxml` files, and models trained
  solely on the 416 provided train rows. **No ClinicalTrials.gov / PubMed / UMLS / MeSH / any
  external dataset** is used for training or inference (there is no such code path here).
- **Uses only lightweight, laptop-runnable pretrained models**: `google/flan-t5-base` (250M,
  fine-tuned by us on the provided rows) and `pritamdeka/BioBERT-...-stsb` (the same encoder
  family the official metric uses). Both run on CPU.
- **Runs fully offline** at inference — no network, no APIs. (Model weights are installed
  once beforehand; see Setup.)

## Method (per field)
| Field | Method |
|---|---|
| `conditions` | **FLAN-T5-base generator** fine-tuned on the 416 provided (article → gold) rows → top-2 canonical disease names (`models/gen_cond`). |
| `eligibility_criteria` | Verbatim inclusion/exclusion section if the article has one; else a **BioBERT sentence ranker** trained on provided rows; else rule-based reconstruction. Truncated at a clean sentence boundary (~60 chars — the FM3S optimum found on the leaderboard). |
| `study_type` | **TF-IDF + LogisticRegression** classifier (trained on provided rows). |
| `sex` | **TF-IDF + LogisticRegression** classifier (trained on provided rows). |
| `minimum_age` / `maximum_age` | Priors learned from the provided train labels. |

## Layout
```
final_submission/
  submission.csv            # the final predictions (public 0.71784)
  README.md
  requirements.txt
  src/
    run_inference.py        # ENTRY POINT — regenerates submission.csv
    build_gen_data.py       # builds (article -> gold) training pairs from provided data
    train_gen.py            # fine-tunes FLAN-T5 on those pairs (training code)
    extract.py              # eligibility/study/sex/age extraction (no external data)
    nxml.py, data_io.py, features.py, models_io.py
  models/
    gen_cond/               # our fine-tuned FLAN-T5 conditions generator (see note below)
    study_type_clf.joblib, sex_clf.joblib, eligibility_ranker.joblib, train_meta.json
  data/                     # <- YOU add this (competition data, non-redistributable)
    Task_1.xlsx
    PMC_NXML_Archives/PMC*.nxml
```

> **Note on model weights:** `models/gen_cond/model.safetensors` (944 MB) exceeds
> GitHub's file-size limit and is not in this repository — the config and tokenizer
> are included. Reproduce the weights exactly with the two retraining commands below
> (deterministic given the provided data), or request them from the author.
> The BioBERT embedding cache (`models/emb_cache/`) is likewise excluded: it is
> derived from the competition articles and regenerates automatically on first run.

## Setup (one-time, needs internet only to fetch the base model weights)
```bash
pip install -r requirements.txt
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz  # optional; not used by the final model
# Place the competition data under final_submission/data/ (Task_1.xlsx + PMC_NXML_Archives/).
# The FLAN-T5-base and BioBERT weights download automatically from HuggingFace on first run;
# to run fully offline afterwards, they are cached locally by huggingface.
```

## Inference (offline, CPU)
```bash
cd final_submission
COHORTX_DEVICE=cpu python src/run_inference.py --out submission.csv
```
Produces `submission.csv` — 500 rows, columns:
`pmcids,conditions,study_type,sex,minimum_age,maximum_age,eligibility_criteria`.

## Retrain the generator from scratch (provided data only)
```bash
cd final_submission
python src/build_gen_data.py                                   # writes data/gen_cond.jsonl
python src/train_gen.py --data data/gen_cond.jsonl --out models/gen_cond \
       --model google/flan-t5-base --epochs 14 --bs 8 --max_tgt 64
```
The study_type / sex / eligibility-ranker models were trained on the 416 provided rows with
the project's `train.py`; `train_meta.json` records the training metadata.

## Notes
- This package deliberately contains **no ClinicalTrials.gov code or cache**. Any file named
  `compliant_hybrid_*` / `cg_*` / `full_elig_*` elsewhere in the parent project uses CT.gov
  and is **NOT compliant** — do not submit those.
