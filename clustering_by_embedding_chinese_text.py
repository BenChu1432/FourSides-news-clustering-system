# clustering_by_embedding_chinese_text.py
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Dict, List, Tuple, Optional
import uuid
import time
import concurrent
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sqlalchemy import create_engine
from dotenv import load_dotenv
import umap
from util.traditionaliseUtil import simplified_to_traditional
from util import awsUtil, dbUtil, entityUtil, placeUtil, strUtil,copyPasteUtil,llmSummarisationUtil
import plotly.express as px
from datetime import UTC  # Add this import if not already present
from plotly.io import to_html
from unicodedata import normalize as ucnorm
import re
import json
import html as html_lib
import textwrap
import traceback
from zoneinfo import ZoneInfo

# ---- CONFIGURATION ----
load_dotenv()
SYNC_DATABASE_URL = os.getenv("SYNC_DATABASE_URL")
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE")) if os.getenv("SAMPLE_SIZE") else 500
OFFSET = int(os.getenv("OFFSET")) if os.getenv("OFFSET") else 500
EMBED_MODEL        = os.getenv("EMBED_MODEL") or "sentence-transformers/multi-qa-mpnet-base-dot-v1"
SIM_THRESH         = float(os.getenv("SIM_THRESH") or 0.8)
TIME_WINDOW_DAYS   = int(os.getenv("TIME_WINDOW_DAYS") or 3)
ENTITY_LABELS      = set(os.getenv("ENTITY_LABELS").split(",")) if os.getenv("ENTITY_LABELS") else {"PERSON","ORG","GPE","LOC"}
TIME_WINDOW_SECS = TIME_WINDOW_DAYS * 86400  # pure Unix seconds

# The following are used but we no longer hard-filter by SIM_LOOSE during attachment.
SIM_LOOSE  = float(os.getenv("SIM_LOOSE") or 0.60)

ALPHA_ENT   = float(os.getenv("ALPHA_ENT") or 0.27)
ALPHA_PLACE = float(os.getenv("ALPHA_PLACE") or 0.05)   # weight for place overlap (Jaccard)
TIME_BONUS  = float(os.getenv("TIME_BONUS") or 0.05)
ASSIGN_THRESH = float(os.getenv("ASSIGN_THRESH") or 0.74)
TOP_K_ATTACH  = int(os.getenv("TOP_K_ATTACH") or 20)    # number of nearest clusters to consider regardless of sim

ZH_MODEL = "zh_core_web_trf"  # "zh_core_web_trf" for best quality;

engine = create_engine(
    SYNC_DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Detect broken connections
    pool_recycle=1800,   # Recycle connections every 30 mins
    future=True
)
embedder = SentenceTransformer('moka-ai/m3e-large')

# ---- CATEGORY CLASSIFICATION SETUP ----
CATEGORIES = [
    ("國際", "International / 國際新聞：世界局勢、地緣政治、外交、外長峰會、條約、制裁、戰爭、政變、跨國事件。"),
    ("政治", "Politics / 政治與政府：選舉、政黨(民眾黨/國民黨/共產黨)、黨員、國會／議會、內閣、總統／首相、地方政府、政策與法案、預算、監察、司法、罷免、公投、抗議、貪腐、兩岸議題、統獨、國家認同、民主、意識形態爭議、政治人物或名嘴針對公共議題之發言與政治直播／自媒體政治言論。"),
    ("財經", "Business & Economy / 財經：股市、投資、公司財報、併購、央行、利率、通膨、景氣、就業、產業動態、匯率、貿易。"),
    ("科技", "Tech / 科技：軟體、硬體、AI／人工智慧、網路、資安、半導體、雲端、通訊。"),
    ("教育", "Education / 教育：學校與大學、招生與考試、課綱、教師、學習政策、校園事件。"),
    ("社會", "Society / 社會與治安：人際糾紛、暴力犯罪、警方與逮捕、檢調偵辦、公共服務、火災、爆炸、起火原因調查、意外事故（非交通工具為主）、救援、送醫、傷亡、治安新聞。"),
    ("文化", "Culture / 文化與藝文：文化資產、博物館、美術展、文學、表演藝術、非物質文化資產、文化創意產業、文化政策與藝文活動。"),
    ("環境", "Environment / 環境與生態：氣候變遷、永續、保育、污染、廢棄物、環評、生態破壞、空氣與水質。"),
    ("娛樂", "Entertainment / 娛樂與流行文化：影視、音樂、演唱會、綜藝、藝人、名人八卦與近況、粉絲活動。"),
    ("健康", "Health / 個人健康與醫療：症狀、治療、藥物、臨床照護、營養、運動、睡眠、心理健康、預防保健。"),
    ("軍隊", "Military / 軍事與國防：軍種、演訓、武器、軍售、國防政策、軍事衝突。"),
    ("運動", "Sports / 體育：賽事、聯賽、球隊、選手、教練、成績與轉會、奧運、世錦賽。"),
    ("觀點", "Opinion / 評論與社論：時評、專欄、社論、觀點、深度分析、投書、政論節目與名嘴評論、公共議題評論。"),
    ("天氣", "Weather / 天氣預報與即時：降雨、溫度、風力、紫外線、特報、路況受天氣影響之提醒。"),
    ("能源", "Energy / 能源：可再生能源（太陽能、風能、水力發電、生質能、地熱能）、不可再生能源（核能、煤炭、石油、天然氣）、清潔能源、綠色能源。"),
    ("公共衞生", "Public Health / 公共衛生：疫情與傳染病、疫苗接種政策、檢測、隔離、群聚感染、醫療量能、環境衛生、空污健康警示。"),
    ("災害", "Natural Disasters / 天然災害（僅限自然成因）：地震、火山爆發、海嘯、颱風／颶風、暴雨洪水、土石流、野火、乾旱、熱浪、寒潮、暴雪、雪崩。"),
    ("交通", "Transportation / 交通與運輸：公路、鐵路、捷運、客運、航空、海運、港口、物流、基礎建設、通勤政策、車輛法規、交通事故（含車禍、列車事故、航空與船舶事故、遊艇起火／爆炸）。"),
    ("地產", "Real Estate / 地產與房市：房價、租金、貸款／房貸利率、建案與施工、都更與重劃、土地使用與都市計畫、預售屋、成交量、房屋稅與持有稅、囤房／社宅政策、包租代管、商辦與工業不動產、REITs、不動產開發商與代銷、建築成本與材料。"),
    ("生活", "Lifestyle / 生活：日常民生與瑣事、消費與優惠資訊、量販／超商活動、食衣住行育樂實用資訊、居家與家政、旅遊與休閒、餐飲美食與開店消息、地方活動、寵物與園藝、穿搭與美妝、親子與人際、節慶與風俗、各種生活小撇步。")
]

CATEGORY_LABELS = [c[0] for c in CATEGORIES]
CATEGORY_EMBS = np.vstack([
    embedder.encode(c[1], show_progress_bar=False, normalize_embeddings=True) for c in CATEGORIES
])
TOPIC_MIN_SIM        = float(os.getenv("TOPIC_MIN_SIM") or 0.30)
TOPIC_MARGIN         = float(os.getenv("TOPIC_MARGIN") or 0.04)
TOPIC_SECONDARY_MIN  = float(os.getenv("TOPIC_SECONDARY_MIN") or 0.28)
TOPIC_MAX_CANDIDATES = int(os.getenv("TOPIC_MAX_CANDIDATES") or 3)

TAIPEI_TZ = ZoneInfo("Asia/Taipei")

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def classify_topic_candidates(centroid: np.ndarray):
    if centroid is None or not np.isfinite(centroid).all():
        return None, 0.0, None, 0.0, True, []

    c = _normalize(centroid)
    sims = (CATEGORY_EMBS @ c).astype(float)  # cosine
    order = np.argsort(-sims)

    candidates = [
        {"label": CATEGORY_LABELS[idx], "score": float(sims[idx]), "rank": rank}
        for rank, idx in enumerate(order[:TOPIC_MAX_CANDIDATES], start=1)
    ]

    best_idx = order[0]
    main_topic = CATEGORY_LABELS[best_idx]
    main_score = float(sims[best_idx])

    secondary_topic, secondary_score = None, 0.0
    if len(order) >= 2:
        second_idx = order[1]
        secondary_topic = CATEGORY_LABELS[second_idx]
        secondary_score = float(sims[second_idx])
        if secondary_score < TOPIC_SECONDARY_MIN:
            secondary_topic, secondary_score = None, 0.0

    ambiguous = False
    if main_score < TOPIC_MIN_SIM:
        ambiguous = True
    elif secondary_topic is not None and (main_score - secondary_score) < TOPIC_MARGIN:
        ambiguous = True

    return main_topic, main_score, secondary_topic, secondary_score, ambiguous, candidates

# Create variables to store topics for the cluster
cluster_topics: Dict[str, str] = {}
cluster_topic_scores: Dict[str, float] = {}
cluster_secondary_topics: Dict[str, str] = {}
cluster_secondary_scores: Dict[str, float] = {}
cluster_ambiguous: Dict[str, bool] = {}
cluster_topic_candidates: Dict[str, list] = {}

MEDIA_NAME_TO_REGION = {
    # Taiwan
    "CTS": "台灣", "TSSDNews": "台灣", "CTWant": "台灣", "TaiwanNews": "台灣",
    "TTV": "台灣", "CTINews": "台灣", "UnitedDailyNews": "台灣", "LibertyTimesNet": "台灣",
    "ChinaTimes": "台灣", "CNA": "台灣", "PTSNews": "台灣", "CTEE": "台灣",
    "MyPeopleVol": "台灣", "TaiwanTimes": "台灣", "ChinaDailyNews": "台灣",
    "SETN": "台灣", "NextAppleNews": "台灣", "MirrorMedia": "台灣", "NowNews": "台灣",
    "StormMedia": "台灣", "TVBS": "台灣", "EBCNews": "台灣", "ETtoday": "台灣",
    "NewTalk": "台灣", "FTV": "台灣","TaroNews":"台灣",

    # Hong Kong
    "HongKongFreePress": "香港", "HKFreePress": "香港", "MingPaoNews": "香港",
    "SingTaoDaily": "香港", "SCMP": "香港", "WenWeiPo": "香港",
    "OrientalDailyNews": "香港", "TaKungPao": "香港", "HK01": "香港",
    "InitiumMedia": "香港", "HKCD": "香港", "NowTV": "香港", "HKCourtNews": "香港",
    "ICable": "香港", "HKGovernmentNews": "香港", "OrangeNews": "香港",
    "TheStandard": "香港", "HKEJ": "香港", "HKET": "香港", "RTHK": "香港",
    "TheWitness": "香港", "InMediaHK": "香港",

    # China
    "PeopleDaily": "中國", "XinhuaNewsAgency": "中國", "GlobalTimes": "中國", "CCTV": "中國",

    # International
    "ChineseNewYorkTimes": "美國",
    "DeutscheWelle": "歐盟",
    "ChineseBBC": "歐盟",
    "TheEpochTimes": "美國",
    "YahooNews": "台灣",
    "VOC": "美國",
}

def extract_places_zh(doc) -> list[str]:
    out = []
    for ent in doc.ents:
        if ent.label_ == "GPE" or ent.label_ == "LOC":
            out.append(ucnorm("NFKC", ent.text.strip()))
    return out

def compute_article_places_detail(
    places: List[str],
    media_region: Optional[str] = None,
) -> Tuple[list[dict], list[str]]:
    """
    Build per-article place details using absolute frequency counts (no normalization).
    Returns:
      - detail: list[{"place": str, "frequency": int, "place_source": "ENTITIES"|"MEDIA_NAME"}]
      - concern: list[str] ordered by descending frequency (ENTITY mentions only).
    """
    pc = Counter(places)
    total = sum(pc.values())

    if total > 0:
        # Sort by absolute frequency
        pairs = pc.most_common()  # [(place, count), ...]
        detail = [
            {"place": p, "frequency": int(c), "place_source": "ENTITIES"}
            for p, c in pairs
        ]
        concern = [p for p, _ in pairs]
        concern = _unique_in_order(concern)
        return detail, concern

    # No entity mentions in this article — optionally hint with media region,
    # but frequency should be 0 to avoid implying an entity mention.
    if media_region:
        return [{"place": media_region, "frequency": 0, "place_source": "MEDIA_NAME"}], []
    return [], []

def _unique_in_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def compute_places_detail(
    place_counts: Counter,
    media_regions: list[str]
) -> Tuple[list[dict], list[str]]:
    """
    Return:
      - detail: every place sorted by absolute frequency (no threshold).
      - concern: the same places (ENTITY mentions only) in the same order.
    """
    total_ent = sum(place_counts.values())

    if total_ent > 0:
        # Sort by absolute frequency of entity mentions
        pairs = place_counts.most_common()  # [(place, count), ...]
        detail = [
            {
                "place": p,
                "frequency": int(c),
                "place_source": "ENTITIES",
            }
            for p, c in pairs
        ]
        concern = [p for p, _ in pairs]
        return detail, concern

    # Fallback: no entity mentions → use media regions, counted by frequency of appearance
    r_counts = Counter([r for r in media_regions if r])
    total_media = sum(r_counts.values())
    if total_media > 0:
        pairs = r_counts.most_common()
        detail = [
            {
                "place": p,
                "frequency": int(c),
                "place_source": "MEDIA_NAME",
            }
            for p, c in pairs
        ]
        concern = [p for p, _ in pairs]  # concern mirrors the displayed order
        return detail, concern

    # Final fallback (no entities, no media hint). Use neutral frequency 0.
    detail = [{"place": "台灣", "frequency": 0, "place_source": "DEFAULT_TAIWAN_IF_LOCAL"}]
    concern = ["台灣"]
    return detail, concern

def decide_place_for_cluster(place_counts: Counter, media_regions: list[str]) -> Tuple[List[str], List[dict]]:
    """
    Optional helper mirroring compute_places_detail; returns (places_in_concern, places_in_detail).
    """
    detail, concern = compute_places_detail(place_counts, media_regions)
    return concern, detail

# Caches for per-cluster aggregation
cluster_place_counts: dict[str, Counter] = defaultdict(Counter)   # entities -> places
cluster_media_regions: dict[str, list[str]] = defaultdict(list)   # derived from media_name
attached_counts: Dict[str, int] = defaultdict(int)
# Track members for clusters created in this job
new_clusters_created: set[str] = set()
cluster_members: dict[str, list[dict]] = defaultdict(list)

def _select_central_articles(
    centroid_vec: np.ndarray,
    members: list[dict],
    k: int = 3,
    redundancy_sim_thresh: float = 0.92
) -> list[dict]:
    """
    members: list of dicts with keys: id, title, content, embedding
    Return up to k members most central to centroid, applying redundancy filter
    so selected items aren't near-duplicates of each other.
    """
    if not members:
        return []

    X = np.vstack([m["embedding"] for m in members])
    centroid = centroid_vec.reshape(1, -1)
    sims = cosine_similarity(X, centroid).ravel()

    # Rank by similarity to centroid
    order = np.argsort(-sims)
    selected = []
    selected_vecs = []

    for idx in order:
        cand = members[idx]
        cand_vec = cand["embedding"].reshape(1, -1)

        # Redundancy filter: skip if too similar to any already selected
        redundant = False
        if selected_vecs:
            S = cosine_similarity(cand_vec, np.vstack(selected_vecs)).ravel()
            if float(S.max()) >= redundancy_sim_thresh:
                redundant = True

        if not redundant:
            selected.append(cand)
            selected_vecs.append(cand["embedding"])
            if len(selected) >= k:
                break

    # If nothing passed the redundancy filter (edge case), fall back to the top-1
    if not selected:
        selected = [members[order[0]]]
    return selected

# Load spaCy
try:
    nlp_zh = spacy.load(ZH_MODEL)
except Exception as e:
    raise RuntimeError(
        f"Failed to load Chinese spaCy model '{ZH_MODEL}'. "
        f"Install it with: python -m spacy download {ZH_MODEL}"
    ) from e

def _norm_surface(s: str) -> str:
    return ucnorm("NFKC", s).strip().lower()

KEEP_LABELS = {"PERSON", "ORG", "GPE", "LOC"}

def _is_junk(surface: str) -> bool:
    if len(surface) <= 1:
        return True
    if all(ch.isdigit() for ch in surface):
        return True
    return False

def extract_ents_zh(text_zh: str):
    doc = nlp_zh(text_zh or "")
    items = []
    for ent in doc.ents:
        if ent.label_ in KEEP_LABELS:
            s = _norm_surface(ent.text)
            if not _is_junk(s):
                items.append((s, ent.label_))
    return Counter(items), doc

# ---- HELPERS FOR PLACE OVERLAP ----
def _parse_top_entities_any(raw):
    data = raw
    if isinstance(raw, str):
        try:
            print("LLM raw output:", raw)
            data = json.loads(raw)
        except Exception:
            try:
                data = ast.literal_eval(raw)
            except Exception:
                data = []
    if not isinstance(data, (list, tuple)):
        return []
    out = []
    for item in data:
        if isinstance(item, dict):
            text = item.get("text") or item.get("surface") or item.get("name")
            label = item.get("label") or item.get("type")
            cnt = item.get("count")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            text, label = item[0], item[1]
            cnt = item[2] if len(item) > 2 else None
        else:
            continue
        if isinstance(text, str) and isinstance(label, str):
            out.append({"text": _norm_surface(text), "label": label, "count": cnt})
    return out

def _as_place_set_from_cluster(df: pd.DataFrame, idx: int) -> set[str]:
    """
    Prefer the cluster.places_in_concern (enum array) if available; otherwise
    fall back to parsing GPE/LOC from top_entities.
    """
    s = set()
    raw = df.at[idx, "places_in_concern"] if "places_in_concern" in df.columns else None
    if isinstance(raw, (list, tuple)):
        for p in raw:
            p2 = placeUtil.canonicalize_place(str(p))
            if p2:
                s.add(p2)
    if s:
        return s
    # Fallback to top_entities
    return parse_places_from_top_entities(df.at[idx, "top_entities"])

def parse_places_from_top_entities(raw) -> set[str]:
    items = _parse_top_entities_any(raw)
    places = set()
    for it in items:
        if it.get("label") in {"GPE", "LOC"}:
            p = placeUtil.canonicalize_place(it.get("text") or "")
            if p:
                places.add(p)
    return places

# ---- EMBEDDING FUNCTION ----
def embed_long_text(text, max_tokens=512):
    sentences = text.replace("\n", " ").split(". ")
    chunks, curr = [], ""
    for sent in sentences:
        if len(curr.split()) + len(sent.split()) <= max_tokens:
            curr += sent + ". "
        else:
            chunks.append(embedder.encode(curr, show_progress_bar=False))
            curr = sent + ". "
    if curr:
        chunks.append(embedder.encode(curr, show_progress_bar=False))
    return np.mean(chunks, axis=0)

def embed_long_text_zh(text: str, char_budget: int | None = None, overlap: int = 40) -> np.ndarray:
    """
    Chunk Chinese text by sentence, encode chunks with normalize_embeddings=True,
    mean-pool, then L2-normalize. Returns zero-vector if input is empty.
    """
    text = (text or "").strip()
    get_dim = getattr(embedder, "get_sentence_embedding_dimension", None)
    dim = get_dim() if callable(get_dim) else 1024

    if not text:
        return np.zeros(dim, dtype=np.float32)

    if char_budget is None:
        char_budget = int(os.getenv("EMBED_MAX_CHARS") or 380)

    sents = [s for s in SENT_SPLIT_RE.split(text) if s]
    chunks, curr = [], ""

    def flush():
        nonlocal curr
        if curr:
            chunks.append(curr)
            curr = ""

    for s in sents if sents else [text]:
        if not curr:
            # Start or hard-split long sentence
            if len(s) <= char_budget:
                curr = s
            else:
                # Hard split long sentence with overlap
                i = 0
                while i < len(s):
                    end = min(i + char_budget, len(s))
                    piece = s[i:end]
                    chunks.append(piece)
                    if end >= len(s):
                        break
                    i = end - overlap
            continue

        if len(curr) + 1 + len(s) <= char_budget:
            curr = f"{curr} {s}"
        else:
            flush()
            nxt = (curr[-overlap:] + s) if curr else s
            if len(nxt) <= char_budget:
                curr = nxt
            else:
                # Split the overrun
                i = 0
                while i < len(nxt):
                    end = min(i + char_budget, len(nxt))
                    piece = nxt[i:end]
                    chunks.append(piece)
                    if end >= len(nxt):
                        break
                    i = end - overlap
                curr = ""

    flush()

    if not chunks:
        return np.zeros(dim, dtype=np.float32)

    embs = embedder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
    embs = np.vstack(embs) if isinstance(embs, list) else embs
    v = np.mean(embs, axis=0).astype(np.float32)

    # L2 normalize
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-12 else np.zeros_like(v, dtype=np.float32)

# ---- LOAD EXISTING CLUSTERS ----
ten_days_ago = int((datetime.now(TAIPEI_TZ) - timedelta(days=10)).timestamp())

df_clusters = pd.read_sql(
    f"""
    SELECT id, 
           centroid_embedding, 
           top_entities, 
           latest_published, 
           main_topic,
           main_topic_score,
           secondary_topic,
           secondary_topic_score,
           ambiguous,
           topic_candidates,
           places_in_concern
    FROM cluster
    WHERE latest_published >= {ten_days_ago}
    """,
    engine
)
print(f"✅ Successfully loaded the existing ({len(df_clusters)}) clusters")

if df_clusters.empty:
    print("ℹ️ No existing clusters found; starting fresh.")
    cluster_embeddings = None
else:
    df_clusters["centroid_embedding"] = df_clusters["centroid_embedding"].apply(
        lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x)
    )

    # Make latest_published timezone-aware datetime
    df_clusters["latest_published"] = pd.to_datetime(df_clusters["latest_published"], unit="s").dt.tz_localize("Asia/Taipei")

    cluster_embeddings = np.vstack(df_clusters["centroid_embedding"].values)
    print("✅ Successfully converted centroid embeddings and timestamps")

# Ensure required object-typed columns exist so .at assignments won’t error
for col in ("places_in_concern", "places_in_detail", "top_entities", "topic_candidates"):
    if col not in df_clusters.columns:
        df_clusters[col] = pd.Series([[] for _ in range(len(df_clusters))], dtype="object")
    else:
        df_clusters[col] = df_clusters[col].astype("object")

if "cluster_size" not in df_clusters.columns:
    df_clusters["cluster_size"] = 1

time_window = timedelta(days=TIME_WINDOW_DAYS)

if "cluster_size" not in df_clusters.columns:
    df_clusters["cluster_size"] = 1

if "places_in_concern" not in df_clusters.columns:
    df_clusters["places_in_concern"] = [[] for _ in range(len(df_clusters))]

time_window = timedelta(days=TIME_WINDOW_DAYS)

# Track updates to existing clusters for centroid recalculation
cluster_centroid_embedding_updates: Dict[str, List[np.ndarray]] = {}
cluster_latest_pub: Dict[str, int] = {}
cluster_top_entities: Dict[str, Counter] = {}

# ---- FETCH UNCLUSTERED ARTICLES ----
df_new = pd.read_sql(f"""
    SELECT id, content, published_at, url, title, "media_name", origin
    FROM news
    WHERE "clusterId" IS NULL
      AND content IS NOT NULL
      AND title IS NOT NULL
      AND published_at IS NOT NULL
   ORDER BY published_at DESC NULLS LAST
    LIMIT {SAMPLE_SIZE}
""", engine)
    #   AND published_at >= EXTRACT(EPOCH FROM NOW()) - {COPYPASTE_WINDOW_SECS}
print("✅ Successfully loaded unclustered articles")

if df_new.empty:
    print("🚫 No new articles to cluster.")
    raise SystemExit(0)

job_id = dbUtil.create_clustering_job(engine)
processed_rows = []

# ---- FETCH ARTICLES FOR COPY-PASTE DETECTION FROM THE LAST X DAYS ----
ZH_DUP_WINDOW_SECS = copyPasteUtil.ZH_DUP_WINDOW_DAYS * 86400
# Must fetch from the last three days of articles for copy-paste detection
df_recent_zh = pd.read_sql(f"""
    SELECT id, content, "media_name", published_at
    FROM news
    WHERE content IS NOT NULL
    AND published_at IS NOT NULL
    ORDER BY published_at DESC
    LIMIT {SAMPLE_SIZE}
""", engine)
print("✅ Successfully loaded articles from the last 3 days")

simidx_zh = copyPasteUtil.SimHashIndexZH(bands=4, bits=64)
_recent_zh_payloads: dict[str, str] = {}

SENT_SPLIT_RE = re.compile(r'(?<=[。！？；.!?])\s*')

for _, r in df_recent_zh.iterrows():
    rid = str(r["id"])
    txt = r.get("content") or ""
    h = copyPasteUtil.simhash64_zh(txt, ngram=copyPasteUtil.ZH_DUP_NGRAM)
    meta = {
        "media_name": str(r.get("media_name") or ""),
        "published_at": int(r.get("published_at") or 0),
    }
    simidx_zh.add(rid, h, meta)
    _recent_zh_payloads[rid] = txt

print("✅ Successfully hashed articles for copy-paste reporting detection")

# Embed → Find Similar → Score (sim + bonuses) → Assign
for _, row in df_new.iterrows():
    def should_embed_article(title: str | None, content: str | None) -> bool:
        """Only embed if we have meaningful text (title or content)."""
        t = (title or "").strip()
        c = (content or "").strip()
        return bool(t or c)

    def is_zero_vec(vec: np.ndarray, eps: float = 1e-8) -> bool:
        try:
            return not np.isfinite(vec).all() or float(np.linalg.norm(vec)) < eps
        except Exception:
            return True
    article_id = str(row["id"])
    content = row["content"]
    title_zh = (row.get("title") or "").strip()
    content_zh = (row.get("content") or "").strip()
    if not should_embed_article(title_zh, content_zh):
        # Skip processing entirely if both title and content are empty/missing
        continue
    text_zh = (title_zh + "\n" + content_zh).strip()
    
    
    url = row["url"]
    media_name = str(row.get("media_name") or "")
    media_region = MEDIA_NAME_TO_REGION.get(media_name)

    # --- Chinese-only copy-paste detection (mark ONLY later copies; no lead secs) ---
    # --- Chinese-only copy-paste detection (mark ONLY later copies) ---
    base_text_zh = content_zh or ""
    zh_simhash64 = 0
    dup_matches_zh = []
    copypaste_flag = False
    copypaste_origin = None  # presumed earlier source if flagged
    copypaste_rule = "original"  # diagnostic
    pub_unix = copyPasteUtil._ensure_secs(row.get("published_at"))
    tie_secs = getattr(copyPasteUtil, "ZH_DUP_TIE_SECS", 60)

    if base_text_zh:
        zh_simhash64 = copyPasteUtil.simhash64_zh(base_text_zh, ngram=copyPasteUtil.ZH_DUP_NGRAM)
        if zh_simhash64 != 0:
            candidates = simidx_zh.query(zh_simhash64, max_candidates=64)
            for cid, dist, meta in candidates:
                if cid == article_id:
                    continue  # skip self
                if dist <= copyPasteUtil.ZH_DUP_MAX_HAMMING:
                    other = _recent_zh_payloads.get(cid, "")
                    j = copyPasteUtil.jaccard_char_ngrams_zh(
                        base_text_zh, other, n=copyPasteUtil.ZH_DUP_REFINE_N
                    )
                    if j >= copyPasteUtil.ZH_DUP_MIN_J:
                        cand_pub = copyPasteUtil._ensure_secs(meta.get("published_at"))
                        dup_matches_zh.append({
                            "id": cid,
                            "media_name": meta.get("media_name"),
                            "hamming": dist,
                            "jaccard_ngram": round(j, 3),
                            "published_at": cand_pub,
                        })

            # sort by similarity then time (early→late)
            dup_matches_zh.sort(key=lambda d: (d["hamming"], -d["jaccard_ngram"], d["published_at"]))

            # Cross-outlet only
            cross = [
                m for m in dup_matches_zh
                if m["media_name"] and m["media_name"] != media_name and m["published_at"] > 0
            ]

            if cross:
                # 1) Strictly earlier if at least tie_secs earlier
                earlier = [m for m in cross if m["published_at"] <= (pub_unix - tie_secs)]
                if earlier:
                    origin = sorted(earlier, key=lambda d: (d["published_at"], d["hamming"], -d["jaccard_ngram"]))[0]
                    copypaste_flag = True
                    copypaste_rule = "earlier_cross_outlet"
                    copypaste_origin = {
                        "id": origin["id"],
                        "media_name": origin["media_name"],
                        "published_at": origin["published_at"],
                        "delta_secs": pub_unix - origin["published_at"],
                        "hamming": origin["hamming"],
                        "jaccard_ngram": origin["jaccard_ngram"],
                    }
                else:
                    # 2) Same-time tie within tie_secs → rely on origin
                    same_time = [m for m in cross if abs(m["published_at"] - pub_unix) <= tie_secs]
                    origin_str = (row.get("origin") or "").strip()
                    if same_time and not copyPasteUtil.is_origin_native(origin_str):
                        origin = sorted(same_time, key=lambda d: (d["hamming"], -d["jaccard_ngram"]))[0]
                        copypaste_flag = True
                        copypaste_rule = "tie_origin_not_native"
                        copypaste_origin = {
                            "id": origin["id"],
                            "media_name": origin["media_name"],
                            "published_at": origin["published_at"],
                            "delta_secs": pub_unix - origin["published_at"],
                            "hamming": origin["hamming"],
                            "jaccard_ngram": origin["jaccard_ngram"],
                        }
                    else:
                        copypaste_flag = False
                        copypaste_rule = "original"
            else:
                copypaste_flag = False
                copypaste_rule = "original"

    # Embedding        
    embedding = embed_long_text_zh(text_zh if text_zh else (row.get("content") or ""))
    if is_zero_vec(embedding):
        # Optionally log and continue
        print(f"Skip embedding for article {row['id']} (zero/invalid vector)")
        continue

    ents_counter, doc_ner = extract_ents_zh(text_zh)
    raw_places = extract_places_zh(doc_ner)
    article_places = placeUtil.expand_tokens_to_allowed_places(raw_places, text=text_zh)
    places = sorted(article_places)  # 若下游要 list

    published_at = int(row["published_at"])

    # Similarity to cluster centroids
    if cluster_embeddings is None or (isinstance(cluster_embeddings, np.ndarray) and cluster_embeddings.shape[0] == 0):
        similarities = np.array([])
    else:
        similarities = cosine_similarity([embedding], cluster_embeddings)[0]

    pub_unix = int(published_at)

    ents_set = set(ents_counter.keys())
    ents_texts = entityUtil.extract_entity_texts(ents_set)

    article_places = set(
        p for p in (placeUtil.canonicalize_place(p) for p in raw_places) if p
    )

    best_idx = None
    best_score = -1.0
    best_metrics = None  # (sim, ent_overlap, place_overlap, time_boost)

    if similarities.size > 0:
        order = np.argsort(-similarities)[:min(TOP_K_ATTACH, similarities.shape[0])]
    else:
        order = []
    candidate_count = len(order)

    for i in order:
        sim = float(similarities[i])
        cluster_latest_unix = entityUtil.as_unix_seconds(df_clusters.at[i, "latest_published"])
        raw_top_ents_i = df_clusters.at[i, "top_entities"]
        top_texts_i = entityUtil.parse_top_entities_texts(raw_top_ents_i)
        ent_overlap = entityUtil.jaccard(ents_texts, top_texts_i) if top_texts_i else 0.0

        cluster_places = _as_place_set_from_cluster(df_clusters, i)
        place_overlap = entityUtil.jaccard(article_places, cluster_places) if (article_places and cluster_places) else 0.0

        time_boost = TIME_BONUS if abs(pub_unix - cluster_latest_unix) <= TIME_WINDOW_SECS else 0.0

        score = sim + ALPHA_ENT * ent_overlap + ALPHA_PLACE * place_overlap + time_boost
        if score > best_score:
            best_score = score
            best_idx = i
            best_metrics = (sim, float(ent_overlap), float(place_overlap), float(time_boost))

    if best_idx is not None:
        sim_dbg, ent_dbg, place_dbg, time_dbg = best_metrics
        print(f"[attach] best_idx={best_idx} sim={sim_dbg:.3f} ent={ent_dbg:.3f} place={place_dbg:.3f} time={time_dbg:.3f} combined={best_score:.3f} thresh={ASSIGN_THRESH}")
    else:
        sim_dbg = ent_dbg = place_dbg = time_dbg = None

    attached_existing = False
    if best_idx is not None and best_score >= ASSIGN_THRESH:
        assigned_cluster_id = df_clusters.at[best_idx, "id"]
        cluster_idx = best_idx
        is_new_cluster = False
        attached_existing = True
    else:
        assigned_cluster_id = str(uuid.uuid4())
        is_new_cluster = True

    # update place stats (cluster-level aggregation)
    places = [p for p in (placeUtil.canonicalize_place(p) for p in raw_places) if p]
    if places:
        cluster_place_counts[assigned_cluster_id].update(places)
    if media_region:
        cluster_media_regions[assigned_cluster_id].append(media_region)

    # 1) Per-article places (ONLY mentions in this article)
    article_places_in_detail, article_places_in_concern = compute_article_places_detail(
        places, media_region
    )

    # 2) Aggregated per-cluster places (mentions across the cluster; concern = mentions only)
    cluster_places_in_detail, cluster_places_in_concern = compute_places_detail(
        cluster_place_counts[assigned_cluster_id],
        cluster_media_regions[assigned_cluster_id]
    )

    cluster_members[assigned_cluster_id].append({
        "id": article_id,
        "title": simplified_to_traditional(title_zh),
        "content": simplified_to_traditional(content_zh),
        "embedding": embedding,
        "published_at": pub_unix,
        "media_name": media_name,
    })


    if is_new_cluster:
        # add new clusters for summarisation and headline generation
        new_clusters_created.add(assigned_cluster_id)
        # Classify topics from the initial centroid (embedding)
        main_topic, main_topic_score, secondary_topic, secondary_score, ambiguous, topic_candidates = classify_topic_candidates(embedding)

        # Seed top_entities from current article's entities
        top_ents_seed = [
            {"text": t, "label": l, "count": cnt}
            for ((t, l), cnt) in ents_counter.items()
        ]
        published_at = int(row["published_at"])

        new_row = {
            "id": assigned_cluster_id,
            "centroid_embedding": embedding,
            "top_entities": top_ents_seed,
            "latest_published": pd.Timestamp(published_at, unit="s", tz="Asia/Taipei"),
            "main_topic": main_topic,
            "main_topic_score": main_topic_score,
            "secondary_topic": secondary_topic,
            "secondary_topic_score": secondary_score,
            "ambiguous": ambiguous,
            "topic_candidates": topic_candidates,
            "places_in_concern": cluster_places_in_concern,
            "places_in_detail":  cluster_places_in_detail,
            "cluster_size": 1,
        }
        cluster_top_entities[assigned_cluster_id] = Counter(ents_counter)

        # NEW: ensure DB updater sees the latest timestamp for this brand-new cluster
        cluster_latest_pub[assigned_cluster_id] = int(row["published_at"])

        df_clusters = pd.concat([df_clusters, pd.DataFrame([new_row])], ignore_index=True)

        if cluster_embeddings is None or (isinstance(cluster_embeddings, np.ndarray) and cluster_embeddings.size == 0):
            cluster_embeddings = embedding.reshape(1, -1)
        else:
            cluster_embeddings = np.vstack([cluster_embeddings, embedding])

        print(f"✅ added the main topic {main_topic} to the cluster")
        print("⚠️ new clusterId:", assigned_cluster_id)

    else:
        # Weighted centroid update using cluster_size
        n = int(df_clusters.at[cluster_idx, "cluster_size"] or 1)
        old_centroid = df_clusters.at[cluster_idx, "centroid_embedding"]
        new_centroid = (old_centroid * n + embedding) / (n + 1)

        df_clusters.at[cluster_idx, "centroid_embedding"] = new_centroid
        df_clusters.at[cluster_idx, "cluster_size"] = n + 1
        cluster_embeddings[cluster_idx] = new_centroid
        attached_counts[assigned_cluster_id] += 1
        cluster_centroid_embedding_updates.setdefault(assigned_cluster_id, []).append(embedding)

        # Update latest_published (tz-aware)
        prev_latest_unix = entityUtil.as_unix_seconds(df_clusters.at[cluster_idx, "latest_published"])
        new_latest_unix = max(prev_latest_unix, pub_unix)
        df_clusters.at[cluster_idx, "latest_published"] = pd.Timestamp(new_latest_unix, unit="s", tz="Asia/Taipei")
        cluster_latest_pub[assigned_cluster_id] = new_latest_unix

        # Update top entities
        c = cluster_top_entities.setdefault(assigned_cluster_id, Counter())
        c.update(ents_counter)
        top_ent_counter = c.most_common(10)
        df_clusters.at[cluster_idx, "top_entities"] = [
            {"text": t, "label": l, "count": count} for ((t, l), count) in top_ent_counter
        ]

        # Update topics
        main_topic, main_topic_score, secondary_topic, secondary_score, ambiguous, topic_candidates = classify_topic_candidates(new_centroid)
        df_clusters.at[cluster_idx, "main_topic"] = main_topic
        df_clusters.at[cluster_idx, "main_topic_score"] = main_topic_score
        df_clusters.at[cluster_idx, "secondary_topic"] = secondary_topic
        df_clusters.at[cluster_idx, "secondary_topic_score"] = secondary_score
        df_clusters.at[cluster_idx, "ambiguous"] = ambiguous
        df_clusters.at[cluster_idx, "topic_candidates"] = topic_candidates

        # Update places (cluster-level)
        df_clusters.at[cluster_idx, "places_in_concern"] = cluster_places_in_concern
        df_clusters.at[cluster_idx, "places_in_detail"]  = cluster_places_in_detail
        

        print(f"✅ added the main topic {main_topic} to the cluster")
        print(f"✅ attached news article ({cluster_idx}) clusterId: {assigned_cluster_id}")

    # Get top 10 entities by frequency for this cluster
    centroid_for_row = embedding if is_new_cluster else new_centroid
    top_ent_counter = cluster_top_entities.get(assigned_cluster_id, Counter()).most_common(10)
    top_ents_out = [{"text": t, "label": l, "count": count} for ((t, l), count) in top_ent_counter]
    sim_dbg, ent_dbg, place_dbg, time_dbg = (best_metrics if best_metrics is not None else (None, None, None, None))

    # Allows subsequent articles to be marked copy-paste
    if zh_simhash64 != 0 and base_text_zh:
        simidx_zh.add(article_id, zh_simhash64, {
            "media_name": media_name,
            "published_at": pub_unix,
        })
        _recent_zh_payloads[article_id] = base_text_zh

    # Save processed row (per-article fields)
    processed_rows.append({
        "id": article_id,
        "title": simplified_to_traditional(row["title"]),
        "content": simplified_to_traditional(content_zh),
        "embedding": embedding,
        "published_at": pub_unix,
        "url": url,
        "media_name": media_name,  # show in the table
        "cluster_id": assigned_cluster_id,
        "centroid_embedding": centroid_for_row,
        "latest_published": pub_unix,
        "top_entities": top_ents_out,
        "main_topic": main_topic,
        "main_topic_score": main_topic_score,
        "secondary_topic": secondary_topic,
        "secondary_topic_score": secondary_score,
        "ambiguous": ambiguous,
        "topic_candidates": topic_candidates,

        # Places (per-article, ONLY mentions)
        "places_in_concern": article_places_in_concern,
        "places_in_detail":  article_places_in_detail,

        # Attach diagnostics
        "attach_candidates": candidate_count,
        "attach_sim": sim_dbg,
        "attach_ent_overlap": ent_dbg,
        "attach_place_overlap": place_dbg,
        "attach_time_bonus": time_dbg,
        "attach_combined": best_score if best_idx is not None else None,
        "attach_thresh": ASSIGN_THRESH,
        "attached_existing": attached_existing,

        # Copy-paste diagnostics
        "content_zh_simhash64": str(zh_simhash64),
        "copypaste_flag": copypaste_flag,        # 1 only for suspected copies
        "copypaste_matches": dup_matches_zh[:5],     # context for review
        "copypaste_origin": copypaste_origin,     # presumed original (if any)
        "copypaste_rule": copypaste_rule,          # NEW
        "copypaste_tie_secs": tie_secs,               # NEW
        # LLM summary+headline
        "headline": None,
        "summary": None,
    })


def _format_article_for_llm(a: dict, max_chars_per_article: int = 2000) -> str:
    # Simple truncation safeguard; you can make this smarter (lead-only, etc.)
    title = (a.get("title") or "").strip()
    content = (a.get("content") or "").strip()
    text = f"【標題】{title}\n【內文】{content}"
    return text[:max_chars_per_article]

def _llm_worker_for_cluster(cid: str, cidx: int, centroid_vec: np.ndarray, members: list[dict]) -> dict:
    """
    Returns dict with keys: cid, cidx, ok(bool), headline(str), summary(list[str]), err(str|None)
    """
    try:
        selected = _select_central_articles(centroid_vec, members, k=3, redundancy_sim_thresh=0.92)
        articles_text = "\n\n".join(_format_article_for_llm(a) for a in selected)

        # llmSummarisationUtil.get_headline_and_summary now returns a Python dict:
        # {"headline": str, "summary": list[str]}
        parsed = llmSummarisationUtil.get_headline_summary_and_question(articles_text) or {}
        print("parsed:",parsed)
        headline = parsed.get("headline", "")
        summary = parsed.get("summary", "")
        question = parsed.get("question", "")

        return {"cid": cid, "cidx": cidx, "ok": True, "headline": headline, "summary": summary, "question":question, "err": None}
    except Exception as e:
        return {
            "cid": cid,
            "cidx": cidx,
            "ok": False,
            "headline": None,
            "summary": None,
            "err": f"{e}\n{traceback.format_exc()}",
        }

if new_clusters_created:
    if "headline" not in df_clusters.columns:
        df_clusters["headline"] = None
    if "summary" not in df_clusters.columns:
        df_clusters["summary"] = None

    tasks = []
    with ThreadPoolExecutor(max_workers=int(os.getenv("LLM_THREADPOOL_SIZE") or 6)) as executor:
        for cid in list(new_clusters_created):
            idxs = df_clusters.index[df_clusters["id"] == cid].tolist()
            if not idxs:
                continue
            cidx = idxs[0]

            centroid_vec = df_clusters.at[cidx, "centroid_embedding"]
            members = cluster_members.get(cid, [])

            if not isinstance(centroid_vec, np.ndarray) or centroid_vec.size == 0 or not members:
                continue

            # Submit a task
            fut = executor.submit(_llm_worker_for_cluster, cid, cidx, centroid_vec, members)
            tasks.append(fut)

        # Collect results and update df_clusters on main thread
        for fut in concurrent.futures.as_completed(tasks):
            res = fut.result()
            cid = res["cid"]
            cidx = res["cidx"]
            if res["ok"]:
                df_clusters.at[cidx, "headline"] = res["headline"]
                df_clusters.at[cidx, "summary"] = res["summary"]
                df_clusters.at[cidx, "question"] = res["question"]
                print(f"📝 Cluster {cid}: generated headline/summary/question.")
            else:
                print(f"⚠️ LLM summarization failed for cluster {cid}: {res['err']}")


# ---- STORE TO DATABASE ----
if processed_rows:
    df_clustered = pd.DataFrame(processed_rows)
    for col in ("headline", "summary"):
        if col not in df_clustered.columns:
            df_clustered[col] = None
    for cid in list(new_clusters_created):
        idxs = df_clusters.index[df_clusters["id"] == cid].tolist()
        if not idxs:
            continue
        cidx = idxs[0]
        h = df_clusters.at[cidx, "headline"]
        s = df_clusters.at[cidx, "summary"]
        q= df_clusters.at[cidx, "question"]
        if h or s or q:
            mask = df_clustered["cluster_id"] == cid
            df_clustered.loc[mask, "headline"] = h
            df_clustered.loc[mask, "summary"] = s
            df_clustered.loc[mask, "question"] = q

    # ---- SANITIZE BEFORE DB WRITE ----
    def to_py(obj):
        """Convert numpy scalars/arrays and nested structures to pure Python."""
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_py(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_py(x) for x in obj]
        return obj

    JSON_COLS = [
        "top_entities", "topic_candidates",
        "places_in_concern", "places_in_detail",
        "copypaste_matches", "copypaste_origin",
        "headline", "summary", "question",
    ]
    FLOAT_COLS = [
        "main_topic_score", "secondary_topic_score",
        "attach_sim", "attach_ent_overlap", "attach_place_overlap",
        "attach_time_bonus", "attach_combined", "attach_thresh",
    ]
    INT_COLS = ["published_at", "latest_published"]

    for col in JSON_COLS:
        if col in df_clustered.columns:
            df_clustered[col] = df_clustered[col].apply(to_py)

    for col in FLOAT_COLS:
        if col in df_clustered.columns:
            df_clustered[col] = (
                pd.to_numeric(df_clustered[col], errors="coerce")
                .astype(float)
                .where(df_clustered[col].notna(), None)
            )

    for col in INT_COLS:
        if col in df_clustered.columns:
            df_clustered[col] = pd.to_numeric(df_clustered[col], errors="coerce").astype("Int64")
            
    dbUtil.store_clusters_to_db(
        df_clustered,
        engine,
        cluster_centroid_embedding_updates=cluster_centroid_embedding_updates,
        cluster_latest_pub=cluster_latest_pub,
        cluster_top_entities=cluster_top_entities,
        job_id=job_id,
        attached_counts=attached_counts
    )
else:
    print("⚠️ No articles were processed.")


# ---- UMAP VISUALIZATION ----
if not processed_rows:
    print("⚠️ No articles were processed. Skipping visualization.")
    try:
        dbUtil.finish_clustering_job(engine, job_id=job_id, url=None)
    except Exception:
        pass
    raise SystemExit(0)

df_vis = pd.DataFrame(processed_rows)

# Make sure IDs are strings for merging
df_vis["cluster_id"] = df_vis["cluster_id"].astype(str)
if "id" in df_clusters.columns:
    df_clusters["id"] = df_clusters["id"].astype(str)

# Guarantee the column and dtype before using .str
if "content" not in df_vis.columns:
    df_vis["content"] = ""
df_vis["content"] = df_vis["content"].fillna("").astype(str)

# Merge cluster-level place aggregates onto df_vis
df_clusters_brief = (
    df_clusters[["id", "places_in_concern", "places_in_detail"]]
    .rename(columns={
        "id": "cluster_id",
        "places_in_concern": "places_in_concern_cluster",
        "places_in_detail":  "places_in_detail_cluster"
    })
)
df_vis = df_vis.merge(df_clusters_brief, on="cluster_id", how="left")

# Optional fallback: if cluster-level missing, use article-level so tooltip never shows empty
df_vis["places_in_concern_cluster"] = df_vis["places_in_concern_cluster"].where(
    df_vis["places_in_concern_cluster"].notna(), df_vis["places_in_concern"]
)
df_vis["places_in_detail_cluster"] = df_vis["places_in_detail_cluster"].where(
    df_vis["places_in_detail_cluster"].notna(), df_vis["places_in_detail"]
)

# Pretty numbers for hover
for col in ["attach_sim", "attach_ent_overlap", "attach_place_overlap", "attach_time_bonus", "attach_combined",
            "main_topic_score", "secondary_topic_score"]:
    df_vis[col] = pd.to_numeric(df_vis[col], errors="coerce").round(3)
df_vis["id"] = df_vis["id"].astype(str)
df_vis["cluster_vis_id"] = df_vis["cluster_id"]
corpus_embeddings = np.vstack(df_vis["embedding"].values)

# Reduce to 2D
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.05, random_state=42)
coords = reducer.fit_transform(corpus_embeddings)

df_vis["x"] = coords[:, 0]
df_vis["y"] = coords[:, 1]
df_vis["short_text"] = df_vis["content"].str.replace("\n", " ", regex=False).str.slice(0, 100) + "..."
df_vis["published_at"] = pd.to_datetime(df_vis["published_at"], unit="s", utc=True)
df_vis["main_topic_filled"] = df_vis["main_topic"].fillna("未分類")

# Build strings for both CLUSTER and ARTICLE concerns
df_vis["places_in_concern_article_str"] = df_vis["places_in_concern"].apply(strUtil._json_str)
df_vis["places_in_concern_cluster_str"] = df_vis["places_in_concern_cluster"].apply(strUtil._json_str)
# Cluster-level details (used for a compact multiline hover)
df_vis["places_in_detail_str"] = df_vis["places_in_detail_cluster"].apply(strUtil._json_str)
# Article-level details (used for a compact multiline hover)
df_vis["places_in_detail_article_str"] = df_vis["places_in_detail"].apply(strUtil._json_str)

palette = px.colors.qualitative.Alphabet
topic_colors = {label: palette[i % len(palette)] for i, label in enumerate(CATEGORY_LABELS)}
topic_colors.setdefault("未分類", "#9e9e9e")

# ----- Compact multiline summarizers (keep tooltip narrow)
def _clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: max(1, n - 1)] + "…"

def summarize_places_multiline(s, max_items=6, per_item_chars=40):
    # For places_in_detail: list of dicts with frequency/source
    if pd.isna(s):
        return ""
    s = str(s)
    try:
        items = json.loads(s)
    except Exception:
        items = None

    if isinstance(items, list):
        parts = []
        for d in items[:max_items]:
            place = (d.get("place") or d.get("name") or "").strip()
            # Prefer "frequency"; fall back to "count"; then to "confidence" (old data)
            freq = d.get("frequency", d.get("count", None))
            if freq is None:
                conf = d.get("confidence", None)
                # If only confidence exists, show it so old rows remain readable
                if isinstance(conf, (int, float)):
                    freq_str = f"confidence={conf:.2f}"
                else:
                    freq_str = ""
            else:
                try:
                    freq_int = int(freq)
                    freq_str = f"frequency: {freq_int}"
                except Exception:
                    freq_str = f"frequency: {freq}"
            src = d.get("place_source")
            place = _clip(place, per_item_chars)
            if freq_str and src:
                parts.append(f"• {place}, {freq_str}, place_source: {src}")
            elif freq_str:
                parts.append(f"• {place}, {freq_str}")
            elif src:
                parts.append(f"• {place}, place_source: {src}")
            else:
                parts.append(f"• {place}")
        if len(items) > max_items:
            parts.append(f"+{len(items)-max_items} more")
        return "<br>".join(parts)

    raw = _clip(s, 240)
    return "<br>".join(textwrap.wrap(raw, width=60))

def summarize_concern_list(s, max_items=8, per_item_chars=40):
    # For places_in_concern: list[str]
    if pd.isna(s):
        return ""
    s = str(s)
    try:
        items = json.loads(s)
    except Exception:
        items = None
    if isinstance(items, list):
        parts = []
        for p in items[:max_items]:
            place = _clip(str(p), per_item_chars)
            parts.append(f"• {place}")
        if len(items) > max_items:
            parts.append(f"+{len(items)-max_items} more")
        return "<br>".join(parts)
    return _clip(s, 240)

def summarize_copypaste_origin(o):
    # o can be dict or JSON string; returns one short line
    if o is None or (isinstance(o, float) and pd.isna(o)):
        return ""
    if isinstance(o, str):
        try:
            o = json.loads(o)
        except Exception:
            return _clip(o, 240)
    if not isinstance(o, dict):
        return ""
    mid = str(o.get("id", ""))[:12]
    media = str(o.get("media_name", "") or "")
    ham = o.get("hamming")
    jacc = o.get("jaccard_ngram")
    delta = o.get("delta_secs")
    ts = o.get("published_at")
    try:
        ts_str = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z") if ts else ""
    except Exception:
        ts_str = str(ts or "")
    parts = []
    if media: parts.append(media)
    if mid: parts.append(f"id={mid}")
    if ham is not None: parts.append(f"H={ham}")
    if jacc is not None: parts.append(f"J={jacc:.3f}")
    if delta is not None: parts.append(f"Δs={int(delta)}")
    if ts_str: parts.append(ts_str)
    return " • ".join(parts)

def summarize_copypaste_matches(o, max_items=5, per_item_chars=36):
    # o can be list[dict] or JSON string; returns compact multiline bullets
    if o is None or (isinstance(o, float) and pd.isna(o)):
        return ""
    if isinstance(o, str):
        try:
            o = json.loads(o)
        except Exception:
            return _clip(o, 240)
    if not isinstance(o, (list, tuple)):
        return ""
    parts = []
    for d in list(o)[:max_items]:
        media = _clip(str(d.get("media_name", "") or ""), per_item_chars)
        mid = str(d.get("id", ""))[:10]
        ham = d.get("hamming")
        jacc = d.get("jaccard_ngram")
        ts = d.get("published_at")
        try:
            ts_str = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%m-%d %H:%M") if ts else ""
        except Exception:
            ts_str = str(ts or "")
        item = f"• {media}"
        if mid: item += f" (id={mid})"
        if ham is not None: item += f", H={ham}"
        if jacc is not None: item += f", J={jacc:.3f}"
        if ts_str: item += f", {ts_str}"
        parts.append(item)
    if len(o) > max_items:
        parts.append(f"+{len(o)-max_items} more")
    return "<br>".join(parts)

# Build plotting DataFrame with compact hover fields
df_plot = df_vis.copy()
df_plot["places_in_detail_cluster_hover"] = df_plot["places_in_detail_str"].apply(
    lambda s: summarize_places_multiline(s, max_items=6, per_item_chars=40)
)
df_plot["places_in_detail_article_hover"] = df_plot["places_in_detail_article_str"].apply(
    lambda s: summarize_places_multiline(s, max_items=6, per_item_chars=40)
)
df_plot["places_in_concern_cluster_hover"] = df_plot["places_in_concern_cluster_str"].apply(
    lambda s: summarize_concern_list(s, max_items=8, per_item_chars=40)
)
df_plot["places_in_concern_article_hover"] = df_plot["places_in_concern_article_str"].apply(
    lambda s: summarize_concern_list(s, max_items=8, per_item_chars=40)
)
df_plot["content"] = df_plot["content"].fillna("").astype(str)
df_plot["short_text_hover"] = (
    df_plot["content"].str.replace("\n", " ", regex=False).str.slice(0, 60) + "…"
)
df_plot["published_at_str"] = pd.to_datetime(df_plot["published_at"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S %Z")

df_plot["copypaste_flag_hover"] = df_plot.get("copypaste_flag", False).map(lambda v: "1" if bool(v) else "0")
df_plot["copypaste_origin_hover"] = df_plot.get("copypaste_origin", None).apply(summarize_copypaste_origin)
df_plot["copypaste_matches_hover"] = df_plot.get("copypaste_matches", None).apply(
    lambda s: summarize_copypaste_matches(s, max_items=6, per_item_chars=36)
)

# Tooltip fields. We include BOTH cluster and article concerns.
hover_cols = [
    "cluster_id",
    "title",
    "short_text_hover",
    "published_at_str",
    "main_topic",
    "main_topic_score",
    "secondary_topic",
    "secondary_topic_score",
    "ambiguous",
    # NEW copy-paste fields
    "copypaste_flag_hover",
    "copypaste_origin_hover",
    "copypaste_matches_hover",
    
    "places_in_concern_article_hover",   # NEW: article-level concern
    "places_in_detail_article_hover",   # NEW: article-level detail
    "places_in_concern_cluster_hover",   # NEW: cluster-level concern
    "places_in_detail_cluster_hover",            # cluster-level details
    "attached_existing",
    "attach_combined",
    "attach_sim",
    "attach_ent_overlap",
    "attach_time_bonus",
    "attach_place_overlap",
    "attach_thresh",
    "attach_candidates",
]
labels_for_hover = [
    "cluster_id", "title", "short_text", "published_at",
    "main_topic", "main_topic_score",
    "secondary_topic", "secondary_topic_score",
    "ambiguous",
    # NEW labels (aligned with the above)
    "copypaste_flag",
    "copypaste_origin",
    "copypaste_matches",

    "places_in_concern (article)",
    "places_in_detail (article)",
    "places_in_concern (cluster)",
    "places_in_detail (cluster)",
    "attached_existing",
    "attach_combined", "attach_sim", "attach_ent_overlap",
    "attach_time_bonus", "attach_place_overlap", "attach_thresh", "attach_candidates",
]

custom_cols = hover_cols + ["url"]
hover_lines = [f"<b>{lbl}:</b> %{{customdata[{i}]}}" for i, lbl in enumerate(labels_for_hover)]
hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

# 1) UMAP colored by cluster, using custom hovertemplate (no hover_data)
fig_clusters_big = px.scatter(
    df_plot,
    x="x",
    y="y",
    color=df_plot["cluster_vis_id"].astype(str),
    custom_data=custom_cols,
    title="🌀 UMAP Visualization of News Clusters (hover shows attach diagnostics)",
    color_discrete_sequence=px.colors.qualitative.Plotly,
)
fig_clusters_big.update_traces(marker=dict(size=8, opacity=0.9), hovertemplate=hovertemplate)
fig_clusters_big.update_layout(height=720, hovermode="closest", hoverdistance=1, hoverlabel=dict(font_size=11, align="left"))

# 2) Summary bar (unchanged)
topic_counts = (
    df_vis.groupby("main_topic_filled")
          .size()
          .reset_index(name="count")
          .sort_values("count", ascending=False)
)
fig_bar = px.bar(
    topic_counts,
    x="main_topic_filled",
    y="count",
    color="main_topic_filled",
    color_discrete_map=topic_colors,
    title="📊 Article Count by Main Topic",
    text="count"
)
fig_bar.update_traces(textposition="outside")
fig_bar.update_layout(xaxis_title="Main Topic", yaxis_title="Article Count", showlegend=False, height=520)

# Build the "All Articles" table HTML
def _fmt_cell(value):
    if isinstance(value, pd.Timestamp):
        try:
            return value.strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (list, dict)):
        try:
            return html_lib.escape(json.dumps(value, ensure_ascii=False))
        except Exception:
            return html_lib.escape(str(value))
    if value is None:
        return ""
    return html_lib.escape(str(value))

excluded_cols = {"short_text", "embedding", "centroid_embedding", "x", "y", "cluster_vis_id", "main_topic_filled"}
first_cols = ["cluster_id", "title", "main_topic"]

priority_cols = [
    "secondary_topic", "ambiguous",
    "copypaste_flag", "copypaste_matches", "content_zh_simhash64",
    # Updated places columns
    "places_in_concern", "places_in_detail",
    "published_at", "attached_existing",
    "attach_combined", "attach_sim", "attach_ent_overlap", "attach_place_overlap",
    "attach_time_bonus", "attach_thresh", "attach_candidates",
    "media_name", "id", "content", "top_entities", "topic_candidates",
    "url"
]

existing_priority = [c for c in priority_cols if c in df_vis.columns and c not in first_cols and c not in excluded_cols]
remaining_cols = [c for c in df_vis.columns if c not in first_cols and c not in excluded_cols and c not in existing_priority]
cols_for_table = first_cols + existing_priority + remaining_cols

thead_cells = "".join(f"<th scope='col'>{html_lib.escape(c)}</th>" for c in cols_for_table)
thead_html = f"<thead><tr>{thead_cells}</tr></thead>"

df_table = (
    df_vis
    .sort_values(by=["cluster_id", "published_at"], ascending=[True, False])
    .reset_index(drop=True)
)

rows_html = []
for _, rec in df_table.iterrows():
    url = rec.get("url") or ""
    cells = []
    for col in cols_for_table:
        val = rec.get(col, "")
        if col == "title":
            title_text = _fmt_cell(val)
            if url:
                cell_html = f"<td class='title'><a href='{html_lib.escape(url)}' target='_blank' rel='noopener noreferrer'>{title_text}</a></td>"
            else:
                cell_html = f"<td class='title'>{title_text}</td>"
            cells.append(cell_html)
            continue
        if col == "url":
            if url:
                cell_html = f"<td><a class='link' href='{html_lib.escape(url)}' target='_blank' rel='noopener noreferrer' aria-label='Open article'>🔗 Open</a></td>"
            else:
                cell_html = "<td></td>"
            cells.append(cell_html)
            continue
        if col == "content":
            text = _fmt_cell(val)
            cell_html = f"<td class='clamp' title='{text}'>{text}</td>"
            cells.append(cell_html)
            continue
        cells.append(f"<td>{_fmt_cell(val)}</td>")
    row_html = f"<tr data-url='{html_lib.escape(url)}'>{''.join(cells)}</tr>"
    rows_html.append(row_html)

tbody_html = f"<tbody>{''.join(rows_html)}</tbody>"

html_table = f"""
<div class="table-wrap" role="region" aria-label="All Articles Table">
  <table id="articles-table">
    {thead_html}
    {tbody_html}
  </table>
</div>
"""

html_clusters = to_html(fig_clusters_big, full_html=False, include_plotlyjs='cdn')
html_bar = to_html(fig_bar, full_html=False, include_plotlyjs=False)

page_html = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>News Clusters & Topic Summary</title>
<style>
  :root {{
    --max-width: 1200px;
    --gap: 16px;
    --border: #eee;
    --muted: #555;
    --hover: #f9fbff;
    --th-bg: #fafafa;
    --link: #1565c0;
  }}
  body {{
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    margin: 0;
    color: #111;
    background: #fff;
  }}
  header {{
    padding: 16px;
    border-bottom: 1px solid var(--border);
  }}
  header h1 {{
    font-size: 18px;
    margin: 0 0 6px;
  }}
  header p {{
    margin: 0;
    color: var(--muted);
    font-size: 14px;
  }}
  main {{
    max-width: var(--max-width);
    margin: 0 auto;
    padding: 16px;
    display: grid;
    grid-template-columns: 1fr;
    gap: var(--gap);
  }}
  section {{
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px;
  }}
  section h2 {{
    font-size: 16px;
    margin: 6px 8px 12px;
  }}

  .table-wrap {{
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    line-height: 1.35;
    table-layout: auto;
  }}
  th, td {{
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
    text-align: left;
    white-space: nowrap;
  }}
  th {{
    position: sticky;
    top: 0;
    background: var(--th-bg);
    z-index: 1;
  }}
  tr:hover {{
    background: var(--hover);
    cursor: pointer;
  }}
  td.title a {{
    color: var(--link);
    text-decoration: none;
  }}
  td.title a:hover {{
    text-decoration: underline.
  }}
  td.clamp {{
    max-width: 560px;
    white-space: normal;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
  }}
  a.link {{
    color: var(--link);
    text-decoration: none;
    font-weight: 600;
  }}
  a.link:hover {{
    text-decoration: underline;
  }}

  @media (max-width: 640px) {{
    th, td {{
      white-space: normal;
    }}
  }}
</style>
</head>
<body>
<header>
  <h1>News Clusters & Topic Summary</h1>
  <p>Top: original UMAP colored by cluster. Middle: article count by main topic. Bottom: all articles table (click a row or title to open the article).</p>
</header>
<main>
  <section aria-label="UMAP clusters (original)">
    {html_clusters}
  </section>
  <section aria-label="Topic counts">
    {html_bar}
  </section>
  <section aria-label="All articles table">
    <h2>All Articles</h2>
    {html_table}
  </section>
</main>
<script>
  // Open URL on point click: URL is the last element of customdata
  document.querySelectorAll(".plotly-graph-div").forEach(function(gd) {{
    gd.on('plotly_click', function(data) {{
      try {{
        const cd = data.points[0].customdata;
        const url = cd && cd[cd.length - 1];
        if (url) window.open(url, '_blank');
      }} catch (e) {{}}
    }});
  }});
  // Open URL on row click
  document.querySelectorAll('#articles-table tbody tr').forEach(function(tr) {{
    tr.addEventListener('click', function(e) {{
      if (e.target.closest('a')) return;
      const url = tr.getAttribute('data-url');
      if (url) window.open(url, '_blank');
    }});
  }});
</script>
</body>
</html>
"""

file_name = f"{int(time.time())}_news_clusters_and_topic_counts.html"
try:
    url = asyncio.run(awsUtil.upload_html_to_s3(page_html, file_name))
    dbUtil.finish_clustering_job(engine, job_id=job_id, url=url)
    print("✅ Visualization saved and uploaded to S3.")
except Exception as e:
    print(f"❌ Error saving visualization: {e}")