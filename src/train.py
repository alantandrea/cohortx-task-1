"""Train the CohortX Task 1 models on the 416 labeled rows and save artifacts.

Produces (in models/):
  study_type_clf.joblib   - LogisticRegression on BioBERT(title+abstract) features
  sex_clf.joblib          - LogisticRegression on the same features
  eligibility_ranker.joblib - 3-class (inclusion/exclusion/none) sentence classifier
  train_meta.json         - CV scores, config, library versions (reproducibility)

Offline / CPU. Embeddings are cached by features.embed so re-runs are fast.

Usage:  python src/train.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_io
import features
from extract import candidate_sentences, cue_features
from nxml import parse_cached

MODELS = Path(__file__).resolve().parents[1] / "models"
RANKER_REL_THRESH = 0.45   # min cosine to a gold bullet to count as a positive
TOPK_INC, TOPK_EXC = 6, 5  # bullets to emit at inference


def _doc_text(art) -> str:
    """Semantic doc text for embeddings (title + abstract)."""
    return f"{art.title}. {art.abstract}".strip()


def _lexical_text(art) -> str:
    """Lexically rich text for TF-IDF classifiers (captures keyword cues)."""
    return f"{art.title}. {art.abstract} {art.methods_text}".strip()


def _counts(y):
    from collections import Counter
    c = Counter(y)
    top = c.most_common(1)[0]
    return dict(c), top[1] / len(y)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    MODELS.mkdir(parents=True, exist_ok=True)
    df = data_io.load_train()
    print(f"Loading + parsing {len(df)} train articles...")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    import joblib

    arts, doc_texts, lex_texts = [], [], []
    for pmcid in df["pmcids"]:
        art = parse_cached(str(data_io.nxml_path(pmcid)))
        arts.append(art)
        doc_texts.append(_doc_text(art))
        lex_texts.append(_lexical_text(art))

    print("Embedding documents (BioBERT, cached)...")
    t0 = time.time()
    X_doc = features.embed(doc_texts)
    print(f"  doc embeddings {X_doc.shape} in {time.time()-t0:.0f}s")

    meta = {"n_train": len(df), "biobert": features.BIOBERT}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    lex = np.array(lex_texts, dtype=object)

    def tfidf_clf(balanced):
        return Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                      sublinear_tf=True, max_features=30000,
                                      stop_words="english")),
            ("lr", LogisticRegression(max_iter=3000, C=4.0,
                                      class_weight="balanced" if balanced else None)),
        ])

    # ---- study_type (TF-IDF captures "randomized"/"retrospective" cues) ----
    # .to_numpy(object) — pandas 3.0 defaults to pyarrow strings which sklearn can't index.
    y_st = np.array(df["study_type"].astype(str).str.upper().tolist(), dtype=object)
    dist, maj = _counts(y_st)
    clf_st = tfidf_clf(balanced=True)
    acc = cross_val_score(clf_st, lex, y_st, cv=cv, scoring="accuracy").mean()
    clf_st.fit(lex, y_st)
    meta["study_type_cv_acc"] = round(float(acc), 4)
    print(f"study_type: 5-fold CV acc = {acc:.3f}  (majority {maj:.3f})  {dist}")
    joblib.dump(clf_st, MODELS / "study_type_clf.joblib")

    # ---- sex (unbalanced: respect the strong ALL prior, flip only on signal) ----
    y_sex = np.array(df["sex"].astype(str).str.upper().str.strip().tolist(), dtype=object)
    dist, maj = _counts(y_sex)
    best_acc, best_clf, best_bal = -1, None, None
    for bal in (False, True):
        c = tfidf_clf(balanced=bal)
        a = cross_val_score(c, lex, y_sex, cv=cv, scoring="accuracy").mean()
        print(f"  sex tfidf balanced={bal}: CV acc {a:.3f}")
        if a > best_acc:
            best_acc, best_clf, best_bal = a, c, bal
    clf_sex = best_clf
    clf_sex.fit(lex, y_sex)
    meta["sex_cv_acc"] = round(float(best_acc), 4)
    meta["sex_balanced"] = best_bal
    print(f"sex: best CV acc = {best_acc:.3f}  (majority {maj:.3f})  {dist}")
    joblib.dump(clf_sex, MODELS / "sex_clf.joblib")

    # ---- eligibility ranker (weak supervision) ----
    print("Building weakly-supervised sentence dataset for eligibility ranker...")
    feats, labels = [], []
    n_pos_inc = n_pos_exc = n_none = 0
    for art, (_, row) in zip(arts, df.iterrows()):
        cands = candidate_sentences(art)
        if not cands:
            continue
        inc_b, exc_b = data_io.parse_eligibility_bullets(row["eligibility_criteria"])
        cand_emb = features.embed(cands)
        inc_emb = features.embed(inc_b) if inc_b else np.zeros((0, cand_emb.shape[1]), np.float32)
        exc_emb = features.embed(exc_b) if exc_b else np.zeros((0, cand_emb.shape[1]), np.float32)
        sim_inc = features.cosine_matrix(cand_emb, inc_emb).max(axis=1) if inc_emb.size else np.zeros(len(cands))
        sim_exc = features.cosine_matrix(cand_emb, exc_emb).max(axis=1) if exc_emb.size else np.zeros(len(cands))
        for i, sent in enumerate(cands):
            si, se = float(sim_inc[i]), float(sim_exc[i])
            if max(si, se) < RANKER_REL_THRESH:
                lab = "none"; n_none += 1
            elif si >= se:
                lab = "inclusion"; n_pos_inc += 1
            else:
                lab = "exclusion"; n_pos_exc += 1
            feats.append(np.concatenate([cand_emb[i], cue_features(sent)]))
            labels.append(lab)

    Xr = np.vstack(feats)
    yr = np.array(labels)
    print(f"  ranker samples: {len(yr)}  (inc={n_pos_inc}, exc={n_pos_exc}, none={n_none})")
    clf_rank = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    try:
        racc = cross_val_score(clf_rank, Xr, yr, cv=3, scoring="f1_macro").mean()
        meta["ranker_cv_f1_macro"] = round(float(racc), 4)
        print(f"  ranker 3-fold CV macro-F1 = {racc:.3f}")
    except Exception as e:
        print(f"  ranker CV skipped: {e}")
    clf_rank.fit(Xr, yr)
    joblib.dump(
        {"clf": clf_rank, "topk_inc": TOPK_INC, "topk_exc": TOPK_EXC,
         "classes": list(clf_rank.classes_)},
        MODELS / "eligibility_ranker.joblib",
    )

    # ---- reproducibility metadata ----
    import sklearn, sentence_transformers
    meta.update({
        "sklearn": sklearn.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "ranker_rel_thresh": RANKER_REL_THRESH,
        "topk_inc": TOPK_INC, "topk_exc": TOPK_EXC,
    })
    (MODELS / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved models + train_meta.json to {MODELS}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
