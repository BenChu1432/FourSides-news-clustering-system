import hashlib
from opencc import OpenCC
import os
import re
from collections import defaultdict
from unicodedata import normalize as ucnorm

# ---- CHINESE COPY-PASTE DETECTION CONFIG ----
ZH_DUP_WINDOW_DAYS = int(os.getenv("ZH_DUP_WINDOW_DAYS") or 7)
ZH_DUP_MAX_ROWS    = int(os.getenv("ZH_DUP_MAX_ROWS") or 500)
ZH_DUP_NGRAM       = int(os.getenv("ZH_DUP_NGRAM") or 3)   # simhash shingle size
ZH_DUP_MAX_HAMMING = int(os.getenv("ZH_DUP_MAX_HAMMING") or 3)
ZH_DUP_REFINE_N    = int(os.getenv("ZH_DUP_REFINE_N") or 5)   # refine n-gram size
ZH_DUP_MIN_J       = float(os.getenv("ZH_DUP_MIN_J") or 0.90) # refine Jaccard threshold

# NEW: treat timestamps within this window as a "tie"
ZH_DUP_TIE_SECS    = int(os.getenv("ZH_DUP_TIE_SECS") or 60)

# Optional: normalize Simplified→Traditional so SC/TC copies also match
try:
    _opencc_zh = OpenCC('s2t')  # convert Simplified to Traditional
except Exception:
    _opencc_zh = None

# --- Normalization and boilerplate stripping for Chinese ---
_BOILERPLATE_RX = re.compile(
    r"^(責任編輯|责任编辑|延伸閱讀|延伸阅读|更多報導|更多报道|來源|来源|綜合報導|综合报导|圖／|圖/|文／|文/|本文經授權|本文经授权|看更多|相關報導|相关新闻)\b.*",
    re.MULTILINE
)

def _ensure_secs(ts) -> int:
    try:
        t = int(ts or 0)
    except Exception:
        return 0
    # 若是毫秒，轉成秒
    return t // 1000 if t > 1_000_000_000_000 else t

def _zh_to_tc(s: str) -> str:
    if not s:
        return ""
    try:
        return _opencc_zh.convert(s) if _opencc_zh else s
    except Exception:
        return s

def zh_canonicalize(text: str) -> str:
    # 1) unify width/case; 2) convert SC->TC; 3) squeeze spaces; 4) keep CJK + digits + letters
    s = ucnorm("NFKC", text or "")
    s = _zh_to_tc(s)
    s = s.lower()
    s = _BOILERPLATE_RX.sub("", s)
    s = re.sub(r"\s+", "", s)  # remove all whitespace for pure char shingles
    # keep CJK, letters, digits
    s = re.sub(r"[^\u4e00-\u9fff0-9a-z]", "", s)
    return s

def _char_shingles_zh(s: str, n: int) -> list[str]:
    # FIX: return [] for short strings to avoid degenerate shingles/Jaccard
    if len(s) < n:
        return []
    return [s[i:i+n] for i in range(len(s)-n+1)]

def _hash64_stable(s: str) -> int:
    # stable 64-bit hash (process-independent)
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")

def simhash64_zh(text: str, ngram: int = 3) -> int:
    s = zh_canonicalize(text)
    shingles = _char_shingles_zh(s, ngram)
    if not shingles:
        return 0
    weights = [0] * 64
    for sh in shingles:
        h = _hash64_stable(sh)
        for i in range(64):
            weights[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(64):
        if weights[i] >= 0:
            out |= (1 << i)
    return out

def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def jaccard_char_ngrams_zh(a: str, b: str, n: int = 5) -> float:
    # FIX: avoid spuriously high J from very short canonical texts
    sa = zh_canonicalize(a)
    sb = zh_canonicalize(b)
    if len(sa) < 2 * n or len(sb) < 2 * n:
        return 0.0
    A = set(_char_shingles_zh(sa, n))
    B = set(_char_shingles_zh(sb, n))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def is_origin_native(origin: str | None) -> bool:
    """
    Normalize origin and decide if it's 'native'.
    Accept a few synonyms; extend as needed for your data.
    """
    if not origin:
        return False
    s = str(origin).strip().lower()
    return s in {"native", "original", "self", "自採", "自采", "本報", "本報自採", "本台", "自製", "自制"}

class SimHashIndexZH:
    def __init__(self, bands: int = 4, bits: int = 64):
        assert bits % bands == 0
        self.bands = bands
        self.bandbits = bits // bands
        self.tables = [defaultdict(list) for _ in range(bands)]
        self.items = {}  # id -> (simhash, meta)

    def _bandkey(self, h: int, band: int) -> int:
        shift = band * self.bandbits
        mask = (1 << self.bandbits) - 1
        return (h >> shift) & mask

    def add(self, item_id: str, h: int, meta: dict):
        self.items[item_id] = (h, meta)
        for b in range(self.bands):
            self.tables[b][self._bandkey(h, b)].append(item_id)

    def query(self, h: int, max_candidates: int = 64) -> list[tuple[str, int, dict]]:
        cand_ids = set()
        for b in range(self.bands):
            cand_ids.update(self.tables[b].get(self._bandkey(h, b), []))
        out = []
        for cid in cand_ids:
            ch, meta = self.items[cid]
            out.append((cid, hamming64(h, ch), meta))
        out.sort(key=lambda x: x[1])
        return out[:max_candidates]