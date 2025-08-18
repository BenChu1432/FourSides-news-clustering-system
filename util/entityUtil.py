import ast
import re
from typing import Iterable, Set
from unicodedata import normalize as ucnorm
import pandas as pd

def normalize_entity_text(s: str) -> str:
    # Lowercase + collapse whitespace; you can add more normalization if needed
    return re.sub(r"\s+", " ", s.strip().lower())

def extract_entity_texts(ents: Iterable[tuple[str, str]]) -> Set[str]:
    # ents is a set of (text, label)
    return {normalize_entity_text(t) for (t, _lbl) in ents}

def parse_top_entities_texts(raw_top_ents) -> Set[str]:
    # Robustly parse DB field (string/list/dict) → set of normalized texts
    texts = set()
    try:
        if isinstance(raw_top_ents, str):
            raw_top_ents = raw_top_ents.strip()
            parsed = ast.literal_eval(raw_top_ents) if raw_top_ents else []
        else:
            parsed = raw_top_ents or []
    except Exception:
        parsed = []

    # Accept list of dicts, tuples, or strings
    for e in parsed:
        if isinstance(e, dict):
            txt = e.get("text")
            if isinstance(txt, str):
                texts.add(normalize_entity_text(txt))
        elif isinstance(e, (list, tuple)) and e:
            txt = e[0]
            if isinstance(txt, str):
                texts.add(normalize_entity_text(txt))
        elif isinstance(e, str):
            texts.add(normalize_entity_text(e))
    return texts

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b: 
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def as_unix_seconds(x) -> int:
    # x can be int, float, str, pandas.Timestamp, or whatever.
    if x is None:
        return 0
    try:
        # Already int-like?
        if isinstance(x, (int,)) and not isinstance(x, bool):
            return int(x)
        if isinstance(x, float):
            return int(x)
        # pandas/py datetime
        if hasattr(x, "timestamp"):
            return int(x.timestamp())
        # string?
        s = str(x).strip()
        if s.isdigit():
            return int(s)
        # float-like string
        return int(float(s))
    except Exception:
        return int(time.time())