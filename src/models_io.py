"""Load trained artifacts and run model-backed inference.

Used by extract.py: if models/ contains the trained artifacts, the extractor uses
them; otherwise it degrades gracefully to the heuristic path. Inference is fully
offline/CPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parents[1] / "models"


class ModelBundle:
    def __init__(self):
        import joblib
        self.study_type = self._load(joblib, "study_type_clf.joblib")
        self.sex = self._load(joblib, "sex_clf.joblib")
        self.ranker = self._load(joblib, "eligibility_ranker.joblib")

    @staticmethod
    def _load(joblib, name):
        fp = MODELS / name
        return joblib.load(fp) if fp.exists() else None

    @property
    def available(self) -> bool:
        return any([self.study_type, self.sex, self.ranker])

    # -- field predictors ---------------------------------------------------- #
    def predict_study_type(self, lex_text: str) -> str | None:
        if self.study_type is None:
            return None
        return str(self.study_type.predict([lex_text])[0])

    def predict_sex(self, lex_text: str) -> str | None:
        if self.sex is None:
            return None
        return str(self.sex.predict([lex_text])[0])

    def rank_eligibility(self, art) -> str | None:
        """Score candidate sentences, pick top inclusion/exclusion, format CT.gov-style."""
        if self.ranker is None:
            return None
        import features
        from extract import candidate_sentences, cue_features, _format_criteria

        cands = candidate_sentences(art)
        if not cands:
            return None
        clf = self.ranker["clf"]
        classes = list(self.ranker["classes"])
        emb = features.embed(cands)
        X = np.hstack([emb, np.array([cue_features(s) for s in cands], np.float32)])
        proba = clf.predict_proba(X)
        idx = {c: i for i, c in enumerate(classes)}
        p_inc = proba[:, idx["inclusion"]] if "inclusion" in idx else np.zeros(len(cands))
        p_exc = proba[:, idx["exclusion"]] if "exclusion" in idx else np.zeros(len(cands))

        topk_inc = self.ranker.get("topk_inc", 6)
        topk_exc = self.ranker.get("topk_exc", 5)
        min_p = 0.33

        inc = [cands[i] for i in np.argsort(-p_inc)[:topk_inc] if p_inc[i] >= min_p]
        exc = [cands[i] for i in np.argsort(-p_exc)[:topk_exc] if p_exc[i] >= min_p]
        # Preserve document order for readability.
        order = {s: i for i, s in enumerate(cands)}
        inc.sort(key=lambda s: order[s])
        exc.sort(key=lambda s: order[s])
        crit = _format_criteria(inc, exc)
        return crit or None


_BUNDLE: ModelBundle | None = None
_TRIED = False


def get_bundle() -> ModelBundle | None:
    global _BUNDLE, _TRIED
    if not _TRIED:
        _TRIED = True
        try:
            b = ModelBundle()
            _BUNDLE = b if b.available else None
        except Exception:
            _BUNDLE = None
    return _BUNDLE
