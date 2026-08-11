"""CohortX Task 1 — 100% COMPLIANT inference.

Reproduces `submission.csv` (public 0.71784, est. private ~0.715). Every field is
derived ONLY from the provided NXML article and models trained ONLY on the provided
416 rows (+ lightweight pretrained models, which the host permits). NO external data,
NO ClinicalTrials.gov / PubMed / UMLS lookups, NO network at inference.

Per-field method:
  conditions          : FLAN-T5-base generator fine-tuned on the 416 provided rows
                        (models/gen_cond); top-2 canonical disease names.
  eligibility_criteria: verbatim inclusion/exclusion section if present, else a BioBERT
                        sentence ranker (trained on provided rows), else reconstruction;
                        truncated at a clean sentence boundary (~60 chars — FM3S optimum).
  study_type, sex     : TF-IDF + LogisticRegression classifiers (trained on provided rows).
  minimum_age/maximum_age : priors learned from the provided train labels.

Usage:  python src/run_inference.py --out submission.csv
"""
from __future__ import annotations

import argparse, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import data_io
import features
from data_io import load_test, nxml_path, NOT_SPECIFIED
from nxml import parse_cached
from models_io import get_bundle
from build_gen_data import article_context
from extract import (_explicit_criteria, extract_eligibility, extract_ages,
                     extract_study_type, extract_sex, lexical_text)

ROOT = Path(__file__).resolve().parents[1]
ELIG_TARGET = 60
COND_TOPK = 2
COND_PROMPT = "List the medical conditions/diseases studied in this article:\n"


def clean_truncate(text: str, target: int = ELIG_TARGET, hard: int = ELIG_TARGET + 130) -> str:
    text = text or ""
    if len(text) <= target:
        return text
    window = text[:hard]
    cuts = [m.start() for m in re.finditer(r"\n\*", window) if m.start() >= target * 0.6]
    if cuts:
        return window[:cuts[-1] if cuts[-1] <= hard else cuts[0]].rstrip()
    m = [x.end() for x in re.finditer(r"[.;]\s", window) if x.end() >= target * 0.6]
    if m:
        return window[:m[-1]].rstrip()
    sp = window.rfind(" ", int(target * 0.6), target + 40)
    return (window[:sp] if sp > 0 else text[:target]).rstrip()


def parse_conditions(text: str, k: int = COND_TOPK) -> list[str]:
    parts = [p.strip(" .;") for p in re.split(r"[,;]| and ", text) if p.strip(" .;")]
    seen = []
    for p in parts:
        if p.lower() not in [x.lower() for x in seen]:
            seen.append(p)
    return seen[:k] or [NOT_SPECIFIED]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "submission.csv"))
    ap.add_argument("--cond_model", default=str(ROOT / "models" / "gen_cond"))
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = get_bundle()
    tok = AutoTokenizer.from_pretrained(args.cond_model)
    cond_model = AutoModelForSeq2SeqLM.from_pretrained(args.cond_model).to(dev).eval()

    df = load_test()
    pmcids = [str(p) for p in df["pmcids"]]
    arts = [parse_cached(str(nxml_path(p))) for p in pmcids]
    t0 = time.time()

    # --- conditions: generative (batched) ---
    ctxs = [article_context(a) for a in arts]
    cond_texts = []
    for i in range(0, len(ctxs), args.batch):
        chunk = ctxs[i:i + args.batch]
        enc = tok([COND_PROMPT + c for c in chunk], return_tensors="pt",
                  truncation=True, max_length=640, padding=True).to(dev)
        with torch.no_grad():
            out = cond_model.generate(**enc, max_new_tokens=48, num_beams=4,
                                      no_repeat_ngram_size=3, early_stopping=True)
        cond_texts.extend(tok.batch_decode(out, skip_special_tokens=True))
    print(f"conditions generated ({time.time()-t0:.0f}s)")

    rows = []
    for p, art, cgen in zip(pmcids, arts, cond_texts):
        study = (bundle.predict_study_type(lexical_text(art)) if bundle else None) or extract_study_type(art)
        sex = (bundle.predict_sex(lexical_text(art)) if bundle else None) or extract_sex(art)
        min_age, max_age = extract_ages(art)
        elig = _explicit_criteria(art)
        if not elig and bundle:
            elig = bundle.rank_eligibility(art)
        elig = elig or extract_eligibility(art) or NOT_SPECIFIED
        elig = clean_truncate(elig, ELIG_TARGET)
        rows.append({
            "pmcids": p,
            "conditions": data_io.conditions_to_cell(parse_conditions(cgen)),
            "study_type": study, "sex": sex,
            "minimum_age": min_age, "maximum_age": max_age,
            "eligibility_criteria": elig or NOT_SPECIFIED,
        })

    out = data_io.write_submission(rows, args.out)
    d = pd.read_csv(out)
    assert len(d) == 500 and d["pmcids"].nunique() == 500
    print(f"wrote {out} ({len(d)} rows) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
