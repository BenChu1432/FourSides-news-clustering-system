import os
import uuid
from dotenv import load_dotenv
import numpy as np
from sqlalchemy import (
    Column,
    Float,
    Integer,
    String,
    Boolean,
    ForeignKey,
    Enum,
    Text,
    Index,
    ARRAY,
    func,
    text,
    update,
)
from typing import Counter, Iterable, Tuple, List, Dict, Any
import sqlalchemy
import json
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.postgresql import UUID, JSONB, ENUM
from sqlalchemy.orm import declarative_base, relationship
import enum
from sqlalchemy.orm import sessionmaker
import time
from sqlalchemy.dialects.postgresql import insert
import numpy as _np
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

load_dotenv()
SYNC_DATABASE_URL = os.getenv("SYNC_DATABASE_URL")
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE")) if os.getenv("SAMPLE_SIZE") else None
BATCH_SIZE = int(os.getenv("BATCH_SIZE")) if os.getenv("BATCH_SIZE") else 1000
EMBED_MODEL = os.getenv("EMBED_MODEL")
SIM_THRESH = float(os.getenv("SIM_THRESH")) if os.getenv("SIM_THRESH") else 0.0
TIME_WINDOW_DAYS = int(os.getenv("TIME_WINDOW_DAYS")) if os.getenv("TIME_WINDOW_DAYS") else 7
ENTITY_LABELS = os.getenv("ENTITY_LABELS")

Base = declarative_base()

# --- Enums ---
class ClusteringErrorTypeEnum(str, enum.Enum):
    SUMMARY_JSON_FORMATTING = "SUMMARY_JSON_FORMATTING"
    ATTACHMENT_FAILURE = "ATTACHMENT_FAILURE"
    OTHERS = "OTHERS"

class EntityLabelEnum(str, enum.Enum):
    PERSON = "PERSON"
    ORG = "ORG"
    GPE = "GPE"
    LOC = "LOC"

# Placeholder enums for MediaNameEnum and OriginEnum
class MediaNameEnum(str, enum.Enum):
    CTS = "CTS"
    TSSDNews = "TSSDNews"
    CTWant = "CTWant"
    TaiwanNews = "TaiwanNews"
    TTV = "TTV"
    CTINews = "CTINews"
    HongKongFreePress = "HongKongFreePress"
    MingPaoNews = "MingPaoNews"
    SingTaoDaily = "SingTaoDaily"
    SCMP = "SCMP"
    ChineseNewYorkTimes = "ChineseNewYorkTimes"
    DeutscheWelle = "DeutscheWelle"
    HKFreePress = "HKFreePress"
    WenWeiPo = "WenWeiPo"
    OrientalDailyNews = "OrientalDailyNews"
    TaKungPao = "TaKungPao"
    HK01 = "HK01"
    InitiumMedia = "InitiumMedia"
    YahooNews = "YahooNews"
    HKCD = "HKCD"
    TheEpochTimes = "TheEpochTimes"
    NowTV = "NowTV"
    ChineseBBC = "ChineseBBC"
    VOC = "VOC"
    HKCourtNews = "HKCourtNews"
    ICable = "ICable"
    HKGovernmentNews = "HKGovernmentNews"
    OrangeNews = "OrangeNews"
    TheStandard = "TheStandard"
    HKEJ = "HKEJ"
    HKET = "HKET"
    RTHK = "RTHK"
    TheWitness = "TheWitness"
    InMediaHK = "InMediaHK"
    PeopleDaily = "PeopleDaily"
    XinhuaNewsAgency = "XinhuaNewsAgency"
    GlobalTimes = "GlobalTimes"
    CCTV = "CCTV"
    UnitedDailyNews = "UnitedDailyNews"
    LibertyTimesNet = "LibertyTimesNet"
    ChinaTimes = "ChinaTimes"
    CNA = "CNA"
    PTSNews = "PTSNews"
    CTEE = "CTEE"
    MyPeopleVol = "MyPeopleVol"
    TaiwanTimes = "TaiwanTimes"
    ChinaDailyNews = "ChinaDailyNews"
    SETN = "SETN"
    NextAppleNews = "NextAppleNews"
    MirrorMedia = "MirrorMedia"
    NowNews = "NowNews"
    StormMedia = "StormMedia"
    TVBS = "TVBS"
    EBCNews = "EBCNews"
    ETtoday = "ETtoday"
    NewTalk = "NewTalk"
    FTV = "FTV"

class OriginEnum(str, enum.Enum):
    native = "native"
    CTS = "CTS"
    TSSDNews = "TSSDNews"
    CTWant = "CTWant"
    TaiwanNews = "TaiwanNews"
    TTV = "TTV"
    CTINews = "CTINews"
    HongKongFreePress = "HongKongFreePress"
    MingPaoNews = "MingPaoNews"
    SingTaoDaily = "SingTaoDaily"
    SCMP = "SCMP"
    ChineseNewYorkTimes = "ChineseNewYorkTimes"
    DeutscheWelle = "DeutscheWelle"
    ChineseBBC = "ChineseBBC"
    TheEpochTimes = "TheEpochTimes"
    YahooNews = "YahooNews"
    VOC = "VOC"
    HKFreePress = "HKFreePress"
    WenWeiPo = "WenWeiPo"
    OrientalDailyNews = "OrientalDailyNews"
    TaKungPao = "TaKungPao"
    HK01 = "HK01"
    InitiumMedia = "InitiumMedia"
    HKCD = "HKCD"
    NowTV = "NowTV"
    HKCourtNews = "HKCourtNews"
    ICable = "ICable"
    HKGovernmentNews = "HKGovernmentNews"
    OrangeNews = "OrangeNews"
    TheStandard = "TheStandard"
    HKEJ = "HKEJ"
    HKET = "HKET"
    RTHK = "RTHK"
    TheWitness = "TheWitness"
    InMediaHK = "InMediaHK"
    PeopleDaily = "PeopleDaily"
    XinhuaNewsAgency = "XinhuaNewsAgency"
    GlobalTimes = "GlobalTimes"
    CCTV = "CCTV"
    UnitedDailyNews = "UnitedDailyNews"
    LibertyTimesNet = "LibertyTimesNet"
    ChinaTimes = "ChinaTimes"
    CNA = "CNA"
    PTSNews = "PTSNews"
    CTEE = "CTEE"
    MyPeopleVol = "MyPeopleVol"
    TaiwanTimes = "TaiwanTimes"
    ChinaDailyNews = "ChinaDailyNews"
    SETN = "SETN"
    NextAppleNews = "NextAppleNews"
    MirrorMedia = "MirrorMedia"
    NowNews = "NowNews"
    StormMedia = "StormMedia"
    TVBS = "TVBS"
    EBCNews = "EBCNews"
    ETtoday = "ETtoday"
    NewTalk = "NewTalk"
    FTV = "FTV"

# Postgres ENUMs: must match Prisma-created enum type names and values exactly.
InterestingRegionOrCountryPG = ENUM(
    # Counties / cities
    '台北市','新北市','桃園市','台中市','台南市','高雄市','基隆市','新竹市','嘉義市',
    '宜蘭縣','花蓮縣','台東縣','南投縣','彰化縣','雲林縣','屏東縣','苗栗縣',
    '新竹縣','嘉義縣','澎湖縣','金門縣','連江縣',
    # Regions / countries
    '香港','中國','美國','加沙','以色列','烏克蘭','歐盟','日本','韓國',
    # Default bucket
    '台灣',
    name='InterestingRegionOrCountry',
    create_type=False,  # Prisma manages the type
)

# Validation set for incoming list values
VALID_INTERESTING_PLACES = set(getattr(InterestingRegionOrCountryPG, "enums", []) or [
    '台北市','新北市','桃園市','台中市','台南市','高雄市','基隆市','新竹市','嘉義市',
    '宜蘭縣','花蓮縣','台東縣','南投縣','彰化縣','雲林縣','屏東縣','苗栗縣',
    '新竹縣','嘉義縣','澎湖縣','金門縣','連江縣',
    '香港','中國','美國','加沙','以色列','烏克蘭','歐盟','日本','韓國',
    '台灣',
])

# --- Models ---
class Cluster(Base):
    __tablename__ = "cluster"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    cluster_name = Column(String, unique=True, nullable=True)
    cluster_summary = Column(String, unique=True, nullable=True)
    cluster_question = Column(String, unique=True, nullable=True)

    processed_at = Column(Integer, nullable=True)
    latest_published = Column(Integer, nullable=True)  # Unix timestamp
    article_count = Column(Integer, nullable=True)

    # Embeddings
    centroid_embedding = Column(ARRAY(Float), nullable=True)

    # Entities and topics
    top_entities = Column(JSONB, nullable=True)

    main_topic = Column(String, nullable=True)
    main_topic_score = Column(Float, nullable=True)
    secondary_topic = Column(String, nullable=True)
    secondary_topic_score = Column(Float, nullable=True)

    ambiguous = Column(Boolean, nullable=False, server_default=text("false"))
    topic_candidates = Column(JSONB, nullable=True)

    # Places (updated)
    places_in_concern = Column(ARRAY(InterestingRegionOrCountryPG), nullable=True)
    # [{"place":"台北市","confidence":0.72,"place_source":"ENTITIES"}, ...]
    places_in_detail = Column(JSONB, nullable=True)

    # NEW: link to the job that produced this cluster
    clustering_job_id = Column(Integer, ForeignKey("clustering_jobs.id"), nullable=True, index=True)
    clustering_job = relationship("ClusteringJob", back_populates="clusters")

    # Relationships
    news = relationship("News", back_populates="cluster")


class ClusteringJob(Base):
    __tablename__ = "clustering_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    visualisation_url = Column(String, nullable=True)
    start_time = Column(Integer, nullable=False)
    end_time = Column(Integer, nullable=True)

    failures = relationship("ClusteringFailure", back_populates="job")
    clusters = relationship("Cluster", back_populates="clustering_job")


class ClusteringFailure(Base):
    __tablename__ = "clustering_failures"

    id = Column(Integer, primary_key=True, autoincrement=True)
    failure_type = Column(Enum(ClusteringErrorTypeEnum), nullable=False)
    detail = Column(String, nullable=True)
    timestamp = Column(Integer, nullable=False)
    resolved = Column(Boolean, default=False, nullable=False)

    jobId = Column(Integer, ForeignKey("clustering_jobs.id"), nullable=True)
    job = relationship("ClusteringJob", back_populates="failures")


class News(Base):
    __tablename__ = "news"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default="gen_random_uuid()")
    media_name = Column(Enum(MediaNameEnum), nullable=True)
    url = Column(String, unique=True, nullable=True)
    title = Column(String, nullable=True)
    origin = Column(Enum(OriginEnum), nullable=True)
    content = Column(Text, nullable=True)
    content_en = Column(Text, nullable=True)
    published_at = Column(Integer, nullable=True)
    authors = Column(ARRAY(String), nullable=True)
    images = Column(ARRAY(String), nullable=True)
    clusterId = Column(UUID(as_uuid=True), ForeignKey("cluster.id"), nullable=True)

    embedding = Column(String, nullable=True)  # Placeholder for vector type

    cluster = relationship("Cluster", back_populates="news")

    authorships = relationship("NewsAuthor", back_populates="news")
    userArticleReads = relationship("UserArticleRead", back_populates="news")
    entities = relationship("NewsEntity", back_populates="news", cascade="all, delete-orphan")

    places_in_concern = Column(ARRAY(InterestingRegionOrCountryPG), nullable=True)
    places_in_detail = Column(JSONB, nullable=True)

    copypaste_flag = Column(Boolean, nullable=False, server_default=text("false"))

    __table_args__ = (
        Index("ix_news_id", "id"),
    )

class NewsEntity(Base):
    __tablename__ = "news_entities"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default="gen_random_uuid()")
    entity = Column(String, nullable=False)
    label = Column(Enum(EntityLabelEnum), nullable=False)

    newsId = Column(UUID(as_uuid=True), ForeignKey("news.id"), nullable=False)
    news = relationship("News", back_populates="entities")

    __table_args__ = (
        Index("ix_news_entities_entity", "entity"),
        Index("ix_news_entities_news_id", "newsId"),
        Index("ix_news_entities_entity_news_id", "entity", "newsId", unique=True),
    )

# --- Additional Models (placeholders) ---

class NewsAuthor(Base):
    __tablename__ = "news_authors"

    id = Column(Integer, primary_key=True)
    news_id = Column(UUID(as_uuid=True), ForeignKey("news.id"))
    news = relationship("News", back_populates="authorships")


class UserArticleRead(Base):
    __tablename__ = "user_article_reads"

    id = Column(Integer, primary_key=True)
    news_id = Column(UUID(as_uuid=True), ForeignKey("news.id"))
    news = relationship("News", back_populates="userArticleReads")

RETRYABLE_EXC = (sqlalchemy.exc.OperationalError, sqlalchemy.exc.InterfaceError)

# helper to chunk iterables
def chunked(iterable, size):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf

def create_clustering_job(engine):
    SessionLocal = sessionmaker(bind=engine, future=True)
    with SessionLocal() as session:
        try:
            job = ClusteringJob(start_time=int(time.time()))
            session.add(job)
            session.commit()
            print(f"🟢 Started clustering job {job.id}")
            return job.id
        finally:
            session.close()

def finish_clustering_job(engine, job_id, url):
    SessionLocal = sessionmaker(bind=engine, future=True)
    with SessionLocal() as session:
        job = session.get(ClusteringJob, job_id)
        if job:
            job.end_time = int(time.time())
            job.visualisation_url = url
            session.commit()
            print(f"✅ Finished clustering job {job_id}")

def log_clustering_failure(engine, job_id, failure_type, detail):
    SessionLocal = sessionmaker(bind=engine, future=True)
    with SessionLocal() as session:
        failure = ClusteringFailure(
            failure_type=failure_type,
            detail=detail,
            timestamp=int(time.time()),
            jobId=job_id,
        )
        session.add(failure)
        session.commit()
        print(f"❌ Logged failure for job {job_id}")
        

def store_clusters_to_db(
    df,
    engine,
    job_id,
    max_retries=3,
    backoff_base=1.5,
    insert_chunk_size=1000,
    cluster_centroid_embedding_updates=None,
    cluster_latest_pub=None,
    cluster_top_entities=None,
    attached_counts=None
):
    """
    要求 df 至少包含：
      - id (news UUID or str)
      - cluster_id (cluster UUID or str)

    可選欄位：
      - centroid_embedding (np.ndarray|list)
      - latest_published (int)
      - main_topic, main_topic_score, secondary_topic, secondary_topic_score
      - ambiguous (bool), topic_candidates (list[dict])
      - cluster_places_in_concern / places_in_concern (list[str] of InterestingRegionOrCountry)
      - cluster_places_in_detail / places_in_detail (list[dict])
      - embedding (np.ndarray|list|str)
      - copypaste_flag (bool)
    """

    import numpy as _np
    from sqlalchemy import select, update
    from sqlalchemy.exc import IntegrityError

    # ---------- 安全檢查：避免 NoneType len ----------
    if df is None:
        raise ValueError("df is None. Expecting a pandas DataFrame with at least columns ['id','cluster_id'].")

    required_cols = {"id", "cluster_id"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    if len(df) == 0:
        # 沒資料可寫，直接返回
        print("ℹ️ store_clusters_to_db: received empty df, nothing to do.")
        return

    # ---------- 公用轉換器 ----------
    def py(obj):
        """遞迴把 numpy 類型轉成原生 Python，tuple 轉 list，便於 JSON/ARRAY 序列化。"""
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: py(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [py(x) for x in obj]
        if isinstance(obj, tuple):
            return [py(x) for x in obj]
        return obj

    def to_uuid(x):
        if x is None:
            raise ValueError("UUID value is None")
        try:
            return uuid.UUID(str(x))
        except Exception as e:
            raise ValueError(f"Invalid UUID: {x}") from e

    # ---------- Enum 轉換器：接受字串或 Enum ----------
    def coerce_enum(enum_cls, val):
        if val is None:
            return None
        if isinstance(val, enum_cls):
            return val  # 已是 Enum
        if isinstance(val, str):
            try:
                # 允許以名稱或值來匹配
                for m in enum_cls:
                    if val == m.value or val == m.name:
                        return m
            except Exception:
                pass
        raise ValueError(f"Invalid enum value for {enum_cls.__name__}: {val}")

    SessionLocal = sessionmaker(bind=engine, future=True)
    cluster_centroid_embedding_updates = cluster_centroid_embedding_updates or {}
    cluster_latest_pub = cluster_latest_pub or {}
    cluster_top_entities = cluster_top_entities or {}
    attached_counts = attached_counts or {}

    # ---------- 地點欄位驗證 ----------
    def normalize_places_list(val):
        if val is None:
            return None
        if isinstance(val, str):
            candidates = [val]
        elif isinstance(val, (list, tuple)):
            candidates = list(val)
        else:
            return None
        cleaned = [p for p in candidates if isinstance(p, str) and p in VALID_INTERESTING_PLACES]
        return cleaned or None

    def normalize_places_detail(tp):
        if not tp:
            return None
        out = []
        for obj in tp:
            if not isinstance(obj, dict):
                continue
            place = obj.get("place") or obj.get("name") or obj.get("text")
            if not isinstance(place, str) or place not in VALID_INTERESTING_PLACES:
                continue
            rec = {"place": place}
            if "frequency" in obj:
                try:
                    rec["frequency"] = int(obj["frequency"])
                except Exception:
                    pass
            conf_val = obj.get("confidence", obj.get("score"))
            if conf_val is not None:
                try:
                    rec["confidence"] = float(conf_val)
                except Exception:
                    pass
            ps = obj.get("place_source") or obj.get("source")
            if isinstance(ps, str):
                rec["place_source"] = ps
            out.append(rec)
        return out or None

    # ---------- 準備每個 cluster 的匯總資訊 ----------
    try:
        grouped = df.groupby("cluster_id")
    except Exception as e:
        raise ValueError("df.groupby('cluster_id') failed. Make sure 'cluster_id' column has valid values.") from e

    cluster_meta_by_id = {}
    for cluster_id, group in grouped:
        first = group.iloc[0]

        centroid_emb = first.get("centroid_embedding")
        centroid_emb = py(centroid_emb)

        meta = {
            "cluster_name": first.get("headline") or None,
            "cluster_summary": first.get("summary") or None,
            "cluster_question": first.get("question") or None,
            "centroid_embedding": centroid_emb,
            "top_entities": py(first.get("top_entities")),
            "latest_published": int(first["latest_published"]) if first.get("latest_published") is not None else None,
            "article_count": int(group.shape[0]),
            "main_topic": first.get("main_topic"),
            "main_topic_score": float(first.get("main_topic_score")) if first.get("main_topic_score") is not None else None,
            "secondary_topic": first.get("secondary_topic"),
            "secondary_topic_score": float(first.get("secondary_topic_score")) if first.get("secondary_topic_score") is not None else None,
            "ambiguous": bool(first.get("ambiguous")) if first.get("ambiguous") is not None else False,
            "topic_candidates": py(first.get("topic_candidates")),
            "places_in_concern": normalize_places_list(py(first.get("cluster_places_in_concern") or first.get("places_in_concern"))),
            "places_in_detail": normalize_places_detail(py(first.get("cluster_places_in_detail") or first.get("places_in_detail"))),
            "job_id": job_id
        }
        cluster_meta_by_id[str(cluster_id)] = py(meta)

    all_cluster_ids = list(cluster_meta_by_id.keys())
    if not all_cluster_ids:
        print("ℹ️ No clusters to write. Skipping.")
        return

    try:
        all_cluster_uuids = [to_uuid(cid) for cid in all_cluster_ids]
    except Exception as e:
        raise ValueError(f"Invalid cluster_id detected: {e}") from e

    # ---------- 查已有的 cluster ----------
    existing_ids = set()
    if all_cluster_uuids:
        with SessionLocal() as session:
            for batch in chunked(all_cluster_uuids, 1000):
                existing_ids.update(
                    row[0] for row in session.query(Cluster.id).filter(Cluster.id.in_(batch)).all()
                )
    existing_str_ids = {str(x) for x in existing_ids}
    new_cluster_ids = [cid for cid in all_cluster_ids if cid not in existing_str_ids]

    cluster_uuid_to_id = {str(eid): eid for eid in existing_ids}
    for cid in new_cluster_ids:
        cluster_uuid_to_id[cid] = to_uuid(cid)

    now_ts = int(time.time())
    print(f"✅ Existing clusters fetched. existing={len(existing_ids)}, new={len(new_cluster_ids)}")

    # ---------- 寫入新 clusters ----------
    if new_cluster_ids:
        with SessionLocal.begin() as session:
            rows = []
            for cid in new_cluster_ids:
                meta = cluster_meta_by_id[cid]
                rows.append({
                    "id": to_uuid(cid),
                    "processed_at": now_ts,
                    "centroid_embedding": meta.get("centroid_embedding"),
                    "top_entities": meta.get("top_entities"),
                    "latest_published": meta.get("latest_published"),
                    "article_count": meta.get("article_count"),
                    "main_topic": meta.get("main_topic"),
                    "main_topic_score": meta.get("main_topic_score"),
                    "secondary_topic": meta.get("secondary_topic"),
                    "secondary_topic_score": meta.get("secondary_topic_score"),
                    "ambiguous": meta.get("ambiguous", False),
                    "topic_candidates": meta.get("topic_candidates"),
                    "places_in_concern": meta.get("places_in_concern"),
                    "places_in_detail": meta.get("places_in_detail"),
                    "clustering_job_id": meta.get("job_id"),
                    "cluster_name": meta.get("cluster_name"),
                    "cluster_summary": meta.get("cluster_summary"),
                    "cluster_question": meta.get("cluster_question"),
                })
            rows = [py(r) for r in rows]
            stmt = insert(Cluster).values(rows).on_conflict_do_nothing(index_elements=["id"])
            session.execute(stmt)
        print(f"🆕 Inserted {len(new_cluster_ids)} new clusters")

    # ---------- 更新新聞的 clusterId 及其他欄位 ----------
    with SessionLocal.begin() as session:
        print("📝 Updating News rows ...")

        try:
            article_ids = [to_uuid(x) for x in df["id"].tolist()]
        except Exception as e:
            raise ValueError(f"Invalid news.id found: {e}") from e

        existing_news_ids = set(
            session.execute(
                select(News.id).where(News.id.in_(article_ids))
            ).scalars().all()
        )

        updates, inserts = [], []

        for row in df.itertuples(index=False):
            nid = to_uuid(getattr(row, "id"))
            cluster_uuid_str = str(getattr(row, "cluster_id"))
            mapped_cluster = cluster_uuid_to_id.get(cluster_uuid_str) or to_uuid(cluster_uuid_str)

            payload = {"id": nid, "clusterId": mapped_cluster}

            # 可選欄位：embedding
            # Only persist embedding when the source article had content AND the vector is valid
            emb = getattr(row, "embedding", None)
            src_content = getattr(row, "content", None)

            def _embed_payload_value(e):
                if isinstance(e, _np.ndarray):
                    return e.tolist()
                if isinstance(e, (list, str)):
                    return e
                # Fallback: stringify if needed
                return str(e)

            # Require non-null content and a non-zero vector to store embedding
            if src_content is not None and emb is not None:
                try:
                    v = _np.array(emb, dtype=float)
                    if _np.isfinite(v).all() and float(_np.linalg.norm(v)) > 1e-8:
                        payload["embedding"] = _embed_payload_value(emb)
                except Exception:
                    # Don’t write an invalid embedding
                    pass

            # places
            art_places_concern = getattr(row, "places_in_concern", None)
            art_places_detail = getattr(row, "places_in_detail", None)
            if art_places_concern is not None:
                payload["places_in_concern"] = normalize_places_list(py(art_places_concern))
            if art_places_detail is not None:
                payload["places_in_detail"] = normalize_places_detail(py(art_places_detail))

            # copypaste_flag
            if hasattr(row, "copypaste_flag"):
                cp_flag = getattr(row, "copypaste_flag")
                if cp_flag is not None:
                    payload["copypaste_flag"] = bool(cp_flag)

            # 媒體來源等 Enum 欄位（若你在插入時有一起寫入）
            for col, enum_cls in [
                ("media_name", MediaNameEnum),
                ("origin", OriginEnum),
            ]:
                if hasattr(row, col):
                    val = getattr(row, col)
                    if val is not None:
                        try:
                            payload[col] = coerce_enum(enum_cls, val)
                        except ValueError as e:
                            # 記錄但不阻塞整批；可依需求改成 raise
                            print(f"⚠️ Skip invalid enum {col}={val} for news {nid}: {e}")

            payload = py(payload)

            if nid in existing_news_ids:
                updates.append(payload)
            else:
                inserts.append(payload)

        if updates:
            session.bulk_update_mappings(News, updates)
        if inserts:
            try:
                session.execute(insert(News), inserts)
            except IntegrityError:
                session.rollback()
                ins_ids = [r["id"] for r in inserts]
                just_existing = set(
                    session.execute(select(News.id).where(News.id.in_(ins_ids))).scalars().all()
                )
                collided = [r for r in inserts if r["id"] in just_existing]
                still_new = [r for r in inserts if r["id"] not in just_existing]
                if collided:
                    session.bulk_update_mappings(News, collided)
                if still_new:
                    session.execute(insert(News), still_new)

        print(f"✅ News updated. updates={len(updates)}, inserts={len(inserts)}")

    # ---------- 更新既有 clusters 的彙整欄位 ----------
    existing_str_ids = {str(x) for x in existing_ids}
    if existing_str_ids:
        with SessionLocal.begin() as session:
            print(f"🔁 Updating {len(existing_str_ids)} existing clusters ...")
            for cid in existing_str_ids:
                cid_uuid = uuid.UUID(cid)
                inc = int(attached_counts.get(cid, 0) or 0)

                stmt_values = {}
                if inc:
                    stmt_values["article_count"] = func.coalesce(Cluster.article_count, 0) + inc

                embeddings = cluster_centroid_embedding_updates.get(cid)
                if embeddings:
                    try:
                        arrs = []
                        for e in embeddings:
                            if isinstance(e, _np.ndarray):
                                arrs.append(e)
                            elif isinstance(e, (list, tuple)):
                                arrs.append(_np.array(e))
                        if arrs:
                            new_centroid = _np.mean(_np.vstack(arrs), axis=0).tolist()
                            stmt_values["centroid_embedding"] = new_centroid
                        # 若格式不正確，則略過
                    except Exception as e:
                        print(f"⚠️ centroid update skipped for {cid}: {e}")

                latest_pub = cluster_latest_pub.get(cid)
                if latest_pub is not None:
                    try:
                        stmt_values["latest_published"] = func.greatest(
                            func.coalesce(Cluster.latest_published, 0),
                            int(latest_pub)
                        )
                    except Exception:
                        pass

                entity_counter = cluster_top_entities.get(cid)
                if entity_counter:
                    try:
                        # 允許傳入 Counter 或 list[tuple]
                        if hasattr(entity_counter, "most_common"):
                            items = entity_counter.most_common(10)
                        else:
                            items = list(entity_counter)[:10]
                        stmt_values["top_entities"] = [{"text": t, "count": int(c)} for t, c in items]
                    except Exception:
                        pass

                meta = cluster_meta_by_id.get(cid, {})
                if meta.get("main_topic") is not None:
                    stmt_values["main_topic"] = meta.get("main_topic")
                    stmt_values["main_topic_score"] = meta.get("main_topic_score")
                    stmt_values["secondary_topic"] = meta.get("secondary_topic")
                    stmt_values["secondary_topic_score"] = meta.get("secondary_topic_score")

                if meta.get("ambiguous") is not None:
                    stmt_values["ambiguous"] = bool(meta.get("ambiguous"))

                if meta.get("topic_candidates") is not None:
                    stmt_values["topic_candidates"] = meta.get("topic_candidates")

                if meta.get("places_in_concern") is not None:
                    stmt_values["places_in_concern"] = meta.get("places_in_concern")

                if meta.get("places_in_detail") is not None:
                    stmt_values["places_in_detail"] = meta.get("places_in_detail")

                stmt_values = py(stmt_values)
                if stmt_values:
                    stmt = update(Cluster).where(Cluster.id == cid_uuid).values(**stmt_values)
                    session.execute(stmt)
            print("✅ Existing clusters updated")