"""Build COMPLIANT training data for a generative eligibility/conditions model.

Input  = article text from the provided NXML (title + abstract + methods/patient sections).
Target = the GOLD field from the provided 416 train rows.
NO external data, NO test distillation — trains only on Kaggle-provided (article -> gold) pairs.

Output: data/gen_elig.jsonl and data/gen_cond.jsonl with {"input","target","pmcids"}.
"""
from __future__ import annotations

import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_io
from data_io import load_train, nxml_path, parse_conditions
from nxml import parse_cached

ROOT = Path(__file__).resolve().parents[1]
MAX_IN = 3000   # chars of article context


def article_context(art) -> str:
    """Patient-selection-relevant article text (title + abstract + methods/eligibility)."""
    parts = [art.title, art.abstract,
             art.section_by(r"inclus|exclus|eligib|selection criteria|patient|particip|method|population|cohort|enroll|recruit")]
    txt = re.sub(r"\s+", " ", " . ".join(p for p in parts if p)).strip()
    return txt[:MAX_IN]


def main():
    tr = load_train()
    elig_rows, cond_rows = [], []
    for _, r in tr.iterrows():
        art = parse_cached(str(nxml_path(r["pmcids"])))
        ctx = article_context(art)
        if not ctx:
            continue
        gold_elig = str(r["eligibility_criteria"]).strip()
        gold_cond = ", ".join(parse_conditions(r["conditions"]))
        if gold_elig and gold_elig.lower() != "nan":
            elig_rows.append({"pmcids": r["pmcids"],
                              "input": "Generate the clinical trial eligibility criteria (inclusion and exclusion) for this study:\n" + ctx,
                              "target": gold_elig})
        if gold_cond:
            cond_rows.append({"pmcids": r["pmcids"],
                              "input": "List the medical conditions/diseases studied in this article:\n" + ctx,
                              "target": gold_cond})
    (ROOT / "data" / "gen_elig.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in elig_rows), encoding="utf-8")
    (ROOT / "data" / "gen_cond.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in cond_rows), encoding="utf-8")
    print(f"gen_elig: {len(elig_rows)} pairs | gen_cond: {len(cond_rows)} pairs")


if __name__ == "__main__":
    main()
