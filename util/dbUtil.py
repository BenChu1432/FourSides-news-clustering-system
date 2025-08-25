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
    Expects df to include at least:
      - id (UUID of news)
      - cluster_id (UUID for cluster)
    Optionally:
      - centroid_embedding (np.ndarray|list)
      - latest_published (int)
      - main_topic, main_topic_score, secondary_topic, secondary_topic_score
      - ambiguous (bool), topic_candidates (list[dict])
      - places_in_concern (list[str] of InterestingRegionOrCountry)
      - places_in_detail (list[{"place": str, "confidence"/"frequency", "place_source": str}])
      - entities (list[(entity_text, label)])
    """

    # --- New: recursive sanitizer to strip numpy types everywhere ---
    import numpy as _np

    def py(obj):
        """Recursively convert numpy scalars/arrays and tuples to pure Python."""
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
            return [py(x) for x in obj]  # convert tuple → list for JSON/ARRAY
        return obj

    SessionLocal = sessionmaker(bind=engine, future=True)
    cluster_centroid_embedding_updates = cluster_centroid_embedding_updates or {}
    cluster_latest_pub = cluster_latest_pub or {}
    cluster_top_entities = cluster_top_entities or {}
    attached_counts = attached_counts or {}

    def _to_uuid(x):
        return uuid.UUID(str(x))

    def _normalize_places_list(val) -> List[str] | None:
        """Ensure we return a list of valid InterestingRegionOrCountry values."""
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

    def _normalize_places_detail(tp) -> List[Dict[str, Any]] | None:
        """
        Normalize places_in_detail to include 'place' and pass through known metrics.
        Accepts {'frequency', 'confidence', 'place_source'}; filters invalid places.
        """
        if not tp:
            return None
        normed = []
        for obj in tp:
            if not isinstance(obj, dict):
                continue
            place = obj.get("place") or obj.get("name") or obj.get("text")
            if not isinstance(place, str) or place not in VALID_INTERESTING_PLACES:
                continue

            out = {"place": place}

            if "frequency" in obj:
                try:
                    out["frequency"] = int(obj["frequency"])
                except Exception:
                    pass

            # accept either "confidence" or "score"
            conf_val = obj.get("confidence", obj.get("score", None))
            if conf_val is not None:
                try:
                    out["confidence"] = float(conf_val)
                except Exception:
                    pass

            ps = obj.get("place_source") or obj.get("source")
            if isinstance(ps, str):
                out["place_source"] = ps

            normed.append(out)
        return normed or None

    attempt = 0
    while True:
        try:
            # ---- Build per-cluster metadata from df (in-memory) ----
            cluster_meta_by_id: Dict[str, dict] = {}
            for cluster_uuid, group in df.groupby("cluster_id"):
                first_row = group.iloc[0]

                # Sanitize raw values early
                centroid_emb = first_row.get("centroid_embedding")
                centroid_emb = centroid_emb.tolist() if isinstance(centroid_emb, _np.ndarray) else centroid_emb
                centroid_emb = py(centroid_emb)

                top_entities = py(first_row.get("top_entities"))
                topic_candidates = py(first_row.get("topic_candidates"))
                cluster_places_concern_raw = first_row.get("cluster_places_in_concern") or first_row.get("places_in_concern")
                cluster_places_detail_raw = first_row.get("cluster_places_in_detail") or first_row.get("places_in_detail")

                meta = {
                    "cluster_name": first_row.get("headline"),
                    "cluster_summary": first_row.get("summary"),
                    "cluster_question": first_row.get("question"),
                    "centroid_embedding": centroid_emb,
                    "top_entities": top_entities,
                    "latest_published": int(first_row["latest_published"]) if first_row.get("latest_published") is not None else None,
                    "article_count": int(group.shape[0]),
                    # Topic fields
                    "main_topic": first_row.get("main_topic"),
                    "main_topic_score": float(first_row.get("main_topic_score")) if first_row.get("main_topic_score") is not None else None,
                    "secondary_topic": first_row.get("secondary_topic"),
                    "secondary_topic_score": float(first_row.get("secondary_topic_score")) if first_row.get("secondary_topic_score") is not None else None,
                    "ambiguous": bool(first_row.get("ambiguous")) if first_row.get("ambiguous") is not None else False,
                    "topic_candidates": topic_candidates,
                    # Places (normalize to enum list and cleaned detail)
                    "places_in_concern": _normalize_places_list(py(cluster_places_concern_raw)),
                    "places_in_detail": _normalize_places_detail(py(cluster_places_detail_raw)),
                    "job_id": job_id
                }

                # Final sanitize the meta dict
                cluster_meta_by_id[str(cluster_uuid)] = py(meta)

            all_cluster_ids = list(cluster_meta_by_id.keys())
            all_cluster_uuids = [_to_uuid(cid) for cid in all_cluster_ids]

            # ---- Session A (read): find which clusters already exist ----
            existing_ids = set()
            if all_cluster_uuids:
                with SessionLocal() as session:
                    for batch in chunked(all_cluster_uuids, 1000):
                        existing_ids.update(
                            row[0] for row in session.query(Cluster.id).filter(Cluster.id.in_(batch)).all()
                        )

            existing_str_ids = {str(cid) for cid in existing_ids}
            new_cluster_ids = [cid for cid in all_cluster_ids if cid not in existing_str_ids]

            # Prepare trivial mapping (UUID identity)
            cluster_uuid_to_id: Dict[str, uuid.UUID] = {str(eid): eid for eid in existing_ids}
            for cid in new_cluster_ids:
                cluster_uuid_to_id[cid] = _to_uuid(cid)

            now_ts = int(time.time())
            print("✅ fetch clusters that already exist")

            # ---- Session B (write): create only new clusters ----
            if new_cluster_ids:
                with SessionLocal.begin() as session:
                    rows = []
                    for cluster_uuid in new_cluster_ids:
                        meta = cluster_meta_by_id[cluster_uuid]
                        rows.append({
                            "id": _to_uuid(cluster_uuid),
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
                            "cluster_name": meta.get("cluster_name") or None,
                            "cluster_summary": meta.get("cluster_summary") or None,
                            "cluster_question": meta.get("cluster_question") or None
                        })

                    if rows:
                        # Final safety: sanitize rows list
                        rows = [py(r) for r in rows]
                        stmt = insert(Cluster).values(rows).on_conflict_do_nothing(index_elements=["id"])
                        session.execute(stmt)

            # ---- Session C (write): update News rows ----
            with SessionLocal.begin() as session:
                print("📝 Updating news.clusterId ...")

                # Normalize article IDs to UUID
                article_ids: list[uuid.UUID] = [_to_uuid(x) for x in df["id"].tolist()]

                # Fetch existing as UUIDs
                existing_news_ids: set[uuid.UUID] = set(
                    session.execute(
                        select(News.id).where(News.id.in_(article_ids))
                    ).scalars().all()
                )

                updates: list[dict] = []
                inserts: list[dict] = []

                for row in df.itertuples(index=False):
                    nid = _to_uuid(row.id)
                    mapped_cluster = cluster_uuid_to_id.get(str(row.cluster_id)) or _to_uuid(row.cluster_id)

                    payload = {
                        "id": nid,
                        "clusterId": mapped_cluster,
                    }

                    emb = getattr(row, "embedding", None)
                    if emb is not None:
                        if isinstance(emb, _np.ndarray):
                            emb = emb.tolist()
                        payload["embedding"] = emb

                    art_places_concern = getattr(row, "places_in_concern", None)
                    art_places_detail  = getattr(row, "places_in_detail", None)
                    if art_places_concern is not None:
                        payload["places_in_concern"] = _normalize_places_list(py(art_places_concern))
                    if art_places_detail is not None:
                        payload["places_in_detail"] = _normalize_places_detail(py(art_places_detail))

                    cp_flag = getattr(row, "copypaste_flag", None)
                    if cp_flag is not None:
                        payload["copypaste_flag"] = bool(cp_flag)

                    # Final sanitize payload
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

                print("✅ Successfully updated news")

            # ---- Session E (write): update existing clusters without reading them ----
            if existing_str_ids:
                with SessionLocal.begin() as session:
                    print(f"🔁 Updating {len(existing_str_ids)} existing clusters...")
                    for cid in existing_str_ids:
                        cid_uuid = uuid.UUID(cid)
                        inc = int(attached_counts.get(cid, 0) or 0)
                        stmt_values = {}

                        if inc:
                            stmt_values["article_count"] = func.coalesce(Cluster.article_count, 0) + inc

                        embeddings = cluster_centroid_embedding_updates.get(cid)
                        if embeddings and all(isinstance(e, _np.ndarray) for e in embeddings):
                            new_centroid = _np.mean(embeddings, axis=0).tolist()
                            stmt_values["centroid_embedding"] = new_centroid

                        latest_pub = cluster_latest_pub.get(cid)
                        if latest_pub is not None:
                            stmt_values["latest_published"] = func.greatest(
                                func.coalesce(Cluster.latest_published, 0),
                                int(latest_pub)
                            )

                        entity_counter = cluster_top_entities.get(cid)
                        if entity_counter:
                            stmt_values["top_entities"] = [
                                {"text": text, "count": count}
                                for text, count in entity_counter.most_common(10)
                            ]

                        meta = cluster_meta_by_id.get(cid, {})
                        if meta.get("main_topic") is not None:
                            stmt_values["main_topic"] = meta.get("main_topic")
                            stmt_values["main_topic_score"] = float(meta.get("main_topic_score")) if meta.get("main_topic_score") is not None else None
                            stmt_values["secondary_topic"] = meta.get("secondary_topic")
                            stmt_values["secondary_topic_score"] = float(meta.get("secondary_topic_score")) if meta.get("secondary_topic_score") is not None else None
                        if meta.get("ambiguous") is not None:
                            stmt_values["ambiguous"] = bool(meta.get("ambiguous"))
                        if meta.get("topic_candidates") is not None:
                            stmt_values["topic_candidates"] = meta.get("topic_candidates")

                        if meta.get("places_in_concern") is not None:
                            stmt_values["places_in_concern"] = meta.get("places_in_concern")
                        if meta.get("places_in_detail") is not None:
                            stmt_values["places_in_detail"] = meta.get("places_in_detail")

                        # Final sanitize update dict
                        stmt_values = py(stmt_values)

                        if stmt_values:
                            stmt = update(Cluster).where(Cluster.id == cid_uuid).values(**stmt_values)
                            session.execute(stmt)

                    print(f"✅ Updated {len(existing_str_ids)} existing clusters")

            # Success
            break

        except RETRYABLE_EXC as e:
            attempt += 1
            engine.dispose()
            if attempt >= max_retries:
                raise
            sleep_for = backoff_base ** attempt
            print(f"⚠️ DB disconnect detected ({e.__class__.__name__}). Retrying in {sleep_for:.1f}s (attempt {attempt}/{max_retries})...")
            time.sleep(sleep_for)

        except sqlalchemy.exc.IntegrityError:
            engine.dispose()
            raise