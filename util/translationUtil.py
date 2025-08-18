# util/translationUtil.py
import os
import re
from typing import List, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    logging as hf_logging,
)

# ---- ENV CONFIG ----
def _as_int(env: str | None, default: int) -> int:
    try:
        return int(env) if env is not None else default
    except Exception:
        return default

TRANSLATE_MODEL = os.getenv("TRANSLATE_MODEL") or "Helsinki-NLP/opus-mt-zh-en"
TRANSLATION_BATCH_SIZE = _as_int(os.getenv("TRANSLATION_BATCH_SIZE"), 8)

# Keep well under 512 to avoid the 0.9*max_length warnings
TRANSLATE_MAX_INPUT_TOKENS = _as_int(os.getenv("TRANSLATE_MAX_INPUT_TOKENS"), 448)
TRANSLATE_MAX_NEW_TOKENS = _as_int(os.getenv("TRANSLATE_MAX_NEW_TOKENS"), 256)

# Hard character cap to avoid pathological inputs
TRANSLATE_MAX_CHARS = _as_int(os.getenv("TRANSLATE_MAX_CHARS"), 8000)

# Device selection
TRANSLATE_DEVICE = _as_int(os.getenv("TRANSLATE_DEVICE"), (0 if torch.cuda.is_available() else -1))
DEVICE = torch.device("cuda:0" if TRANSLATE_DEVICE >= 0 and torch.cuda.is_available() else "cpu")

# Debug lengths (set TRANSLATE_DEBUG=1 to log chunk token lengths)
TRANSLATE_DEBUG = _as_int(os.getenv("TRANSLATE_DEBUG"), 0)

# Reduce HF log noise
if os.getenv("TRANSFORMERS_VERBOSITY", "").lower() == "error":
    hf_logging.set_verbosity_error()

_model = None
_tokenizer = None

def _get_mt():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TRANSLATE_MODEL)
        _model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATE_MODEL)
        _model.to(DEVICE)
        _model.eval()
    return _model, _tokenizer


_SENT_SPLIT_RE = re.compile(r"([。！？!?；;：:、,\n])")

def _split_sentences_zh(text: str) -> List[str]:
    if not text:
        return []
    text = text.strip()
    parts = _SENT_SPLIT_RE.split(text)
    out = []
    for i in range(0, len(parts), 2):
        seg = (parts[i] or "").strip()
        if not seg:
            continue
        if i + 1 < len(parts):
            seg = seg + (parts[i + 1] or "")
        out.append(seg.strip())
    if not out:
        out = [text]
    return out


def _pack_by_token_limit(sentences: List[str], tokenizer, max_tokens: int) -> List[str]:
    chunks: List[str] = []
    cur: List[str] = []

    def tok_len(s: str) -> int:
        # Add special tokens to mirror encoder behavior
        return len(tokenizer.encode(s, add_special_tokens=True))

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) > TRANSLATE_MAX_CHARS:
            s = s[:TRANSLATE_MAX_CHARS]

        if not cur:
            cur = [s]
            continue

        trial = " ".join(cur + [s])
        if tok_len(trial) <= max_tokens:
            cur.append(s)
        else:
            chunks.append(" ".join(cur))
            cur = [s]

    if cur:
        chunks.append(" ".join(cur))

    # Final safety: a single sentence might still exceed max_tokens; keep as its own chunk
    # We'll hard-truncate via tokenizer in encode step.
    return chunks


def _chunk_text(text: str, tokenizer, max_tokens: int) -> List[str]:
    if not text:
        return [""]
    text = text[:TRANSLATE_MAX_CHARS]
    sents = _split_sentences_zh(text)
    chunks = _pack_by_token_limit(sents, tokenizer, max_tokens)
    return chunks


def _encode_truncate(batch_texts: List[str], tokenizer, max_tokens: int):
    # Hard truncate to max_tokens at tokenizer level, padding to batch max
    return tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tokens,
    )


def batch_translate(texts: List[str]) -> List[str]:
    """
    Translate Chinese -> English, chunking inputs under encoder token limits.
    Uses direct model.generate() to avoid pipeline auto-warnings.
    """
    if not texts:
        return []

    model, tok = _get_mt()

    # Prepare chunks
    flat: List[str] = []
    index_map: List[Tuple[int, int]] = []  # (orig_idx, chunk_idx)
    per_doc_counts: List[int] = []

    for i, t in enumerate(texts):
        t = (t or "")
        if len(t) > TRANSLATE_MAX_CHARS:
            t = t[:TRANSLATE_MAX_CHARS]
        chunks = _chunk_text(t, tok, TRANSLATE_MAX_INPUT_TOKENS)
        per_doc_counts.append(len(chunks))
        for j, c in enumerate(chunks):
            flat.append(c)
            index_map.append((i, j))

    # Translate chunks in batches
    translated_flat: List[str] = ["" for _ in flat]
    bs = max(1, TRANSLATION_BATCH_SIZE)

    for start in range(0, len(flat), bs):
        batch = flat[start:start + bs]

        # Debug: show lengths to verify no chunk > limit
        if TRANSLATE_DEBUG:
            lens = [len(tok.encode(x, add_special_tokens=True)) for x in batch]
            print(f"[translate] token lengths (pre-trunc) min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.1f}")

        enc = _encode_truncate(batch, tok, TRANSLATE_MAX_INPUT_TOKENS)
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=TRANSLATE_MAX_NEW_TOKENS,
                num_beams=1,
                do_sample=False,
                length_penalty=1.0,
                early_stopping=True,
            )

        out_texts = tok.batch_decode(gen, skip_special_tokens=True)
        for k, t_out in enumerate(out_texts):
            translated_flat[start + k] = (t_out or "").strip()

    # Reassemble per original doc
    joined: List[List[str]] = [[] for _ in texts]
    for idx, (doc_idx, _chunk_idx) in enumerate(index_map):
        joined[doc_idx].append(translated_flat[idx])

    final = [" ".join(parts for parts in parts_list if parts).strip() for parts_list in joined]
    return final