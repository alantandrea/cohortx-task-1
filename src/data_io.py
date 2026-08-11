"""Load the Task_1.xlsx train/test sheets and write submissions."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
XLSX = DATA / "Task_1.xlsx"
NXML_DIR = DATA / "PMC_NXML_Archives"

FIELDS = [
    "conditions",
    "study_type",
    "sex",
    "minimum_age",
    "maximum_age",
    "eligibility_criteria",
]
COLUMNS = ["pmcids"] + FIELDS
NOT_SPECIFIED = "Not Specified"


def pmcid_str(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return str(int(float(v)))
    except (ValueError, TypeError):
        return str(v).strip() or None


def nxml_path(pmcid: str) -> Path:
    return NXML_DIR / f"PMC{pmcid}.nxml"


def _load_sheet(sheet: str) -> pd.DataFrame:
    df = pd.read_excel(XLSX, sheet_name=sheet, dtype=object)
    df = df[[c for c in COLUMNS if c in df.columns]].copy()
    df["pmcids"] = df["pmcids"].map(pmcid_str)
    df = df[df["pmcids"].notna()].reset_index(drop=True)
    return df


def load_train() -> pd.DataFrame:
    """416 labeled rows (empty trailing rows dropped via pmcid filter)."""
    return _load_sheet("Train")


def load_test() -> pd.DataFrame:
    """500 rows; only pmcids populated."""
    return _load_sheet("Test")


def parse_conditions(v) -> list[str]:
    """Gold `conditions` is a stringified Python list; parse leniently."""
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    s = str(v).strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    return [s]


def conditions_to_cell(conds: list[str]) -> str:
    """Serialise conditions back to the stringified-list format used in the data."""
    clean = [c.strip() for c in conds if c and c.strip()]
    if not clean:
        clean = [NOT_SPECIFIED]
    return repr(clean)


def parse_eligibility_bullets(text) -> tuple[list[str], list[str]]:
    """Split a gold/predicted eligibility string into (inclusion, exclusion) bullets."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return [], []
    s = str(text).replace("\r", "\n")
    low = s.lower()
    inc_pos = low.find("inclusion crit")
    exc_pos = low.find("exclusion crit")

    def bulletize(chunk: str) -> list[str]:
        items = []
        for line in re.split(r"\n+|(?<=\.)\s*\*", chunk):
            line = re.sub(r"^\s*[\*•\-–·]\s*|^\s*\(?\d+\)?[\.\)]\s*", "", line).strip()
            line = re.sub(r"^(inclusion|exclusion) criteria[:\s]*", "", line, flags=re.I).strip()
            if len(line) >= 5:
                items.append(re.sub(r"\s+", " ", line))
        return items

    if inc_pos == -1 and exc_pos == -1:
        return bulletize(s), []
    if exc_pos == -1:
        return bulletize(s[inc_pos:]), []
    if inc_pos == -1:
        return [], bulletize(s[exc_pos:])
    if inc_pos < exc_pos:
        return bulletize(s[inc_pos:exc_pos]), bulletize(s[exc_pos:])
    return bulletize(s[inc_pos:]), bulletize(s[exc_pos:inc_pos])


def write_submission(rows: list[dict], path: str | Path) -> Path:
    """Write predictions to CSV with the canonical column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = NOT_SPECIFIED
    df = df[COLUMNS]
    df.to_csv(path, index=False, encoding="utf-8")
    return path
