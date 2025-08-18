import time
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
import spacy
from spacy.cli import download as spacy_download
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import umap
import plotly.express as px
from datetime import timedelta
import uuid
import asyncio
from util import awsUtil, dbUtil
from collections import Counter

cluster_summaries = {}

# —— CONFIGURATION —— 
SAMPLE_SIZE        = 300
BATCH_SIZE         = 16
EMBED_MODEL        = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
SIM_THRESH         = 0.8            # cosine similarity threshold
TIME_WINDOW_DAYS   = 3              # ± days for event window
ENTITY_LABELS      = {"PERSON","ORG","GPE","LOC"}

# Postgres credentials
load_dotenv()
SYNC_DATABASE_URL=os.getenv("SYNC_DATABASE_URL")
print(f"✅ Loaded environment variables")
# —— LOAD DATA ——
engine = create_engine(SYNC_DATABASE_URL)
query = f"""
    SELECT id, title, content_en, published_at, url, "clusterId"
    FROM news
    WHERE content_en IS NOT NULL AND "clusterId" IS NULL
    ORDER BY published_at DESC NULLS LAST
    LIMIT {SAMPLE_SIZE}
"""
df = pd.read_sql(query, engine, parse_dates=["published_at"])
print(f"✅ Loaded {len(df)} articles")

# —— EMBEDDING: handle long texts by chunking ——
embedder = SentenceTransformer(EMBED_MODEL)

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

print("🔢 Embedding articles...")
df["embedding"] = df["content_en"].apply(embed_long_text)
corpus_embeddings = np.vstack(df["embedding"].values)

# —— NAMED ENTITY RECOGNITION ——
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Model en_core_web_sm not found. Downloading...")
    spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

print("🗂 Extracting named entities...")
doc_ents = []
for doc in nlp.pipe(df["content_en"], batch_size= BATCH_SIZE, disable=["parser"]):
    ents = {ent.text for ent in doc.ents if ent.label_ in ENTITY_LABELS}
    doc_ents.append(ents)

df["entities"] = [";".join(sorted(ents)) for ents in doc_ents]

# —— BUILD SIMILARITY GRAPH ——
print("🔗 Building event graph...")
sim_matrix = cosine_similarity(corpus_embeddings)
time_window = timedelta(days=TIME_WINDOW_DAYS)

G = nx.Graph()
G.add_nodes_from(range(len(df)))

for i in range(len(df)):
    for j in range(i+1, len(df)):
        # Time constraint only if both dates exist
        di = df.at[i, "published_at"]
        dj = df.at[j, "published_at"]
        if pd.notna(di) and pd.notna(dj):
            if abs(di - dj) > time_window:
                continue
        # Content similarity
        if sim_matrix[i, j] < SIM_THRESH:
            continue
        # Shared entity
        if not (doc_ents[i] & doc_ents[j]):
            continue
        G.add_edge(i, j)

# —— EXTRACT CLUSTERS ——
print("🎯 Extracting clusters...")
components = list(nx.connected_components(G))
uuid_map = {}      # cluster_idx -> uuid
label_map = {}     # article idx -> uuid
vis_label_map = {} # article idx -> visual cluster id
for cluster_idx, comp in enumerate(components):
    cluster_uuid = str(uuid.uuid4())
    uuid_map[cluster_idx] = cluster_uuid
    for idx in comp:
        label_map[idx] = cluster_uuid
        vis_label_map[idx] = cluster_idx

# Singletons for any node not in the graph (isolated or no edges)
singleton_count = len(components)
for idx in range(len(df)):
    if idx not in label_map:
        cluster_uuid = str(uuid.uuid4())
        label_map[idx] = cluster_uuid
        vis_label_map[idx] = singleton_count
        singleton_count += 1
        

# Store both internal (UUID) and visual (integer) cluster IDs
df["cluster_id"] = df.index.map(label_map)         # UUID for internal use
df["cluster_vis_id"] = df.index.map(vis_label_map) # Integer for visualization


# Storing cluster Ids in the DB
# dbUtil.store_clusters_to_db(df, engine)
# print("✅ Stored cluster ids in the DB")


# —— VISUALIZE ——
print("📊 Plotting clusters with UMAP...")
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.05)
coords = reducer.fit_transform(corpus_embeddings)
df["x"], df["y"] = coords[:, 0], coords[:, 1]
df["short_text"] = df["content_en"].str[:100] + "..."

fig = px.scatter(
    df, x="x", y="y",
    color=df["cluster_vis_id"].astype(str),  # use visual ID for colors
    hover_data=["cluster_id","title", "short_text", "published_at", "url"],
    custom_data=["url"],
    title="Event-Based News Clustering"
)
# Convert the figure to HTML
plotly_html = fig.to_html(full_html=True, include_plotlyjs='cdn')

# Inject JavaScript to handle dot clicks
plotly_html = plotly_html.replace(
    '</body>',
    """
    <script>
    document.querySelectorAll(".plotly-graph-div").forEach(function(gd) {
        gd.on('plotly_click', function(data) {
            const url = data.points[0].customdata[0];
            window.open(url, '_blank');  // Use '_self' to open in same tab
        });
    });
    </script>
    </body>
    """
)
# Save the modified HTML as a file
file_name=f"{int(time.time())}_news_clusters.html"
try:
    result = asyncio.run(awsUtil.upload_html_to_s3(plotly_html, file_name))
    print("result:", result)
except Exception as e:
    print(f"❌ Cannot save the HTML file in S3: {e}")
print("✅ Saved the file to S3")

# Saved the filename to the DB
print("✅ Saved the filename to DB")

# —— EXPORT ——
export_cols = ["id", "title", "cluster_id", "cluster_vis_id", "entities", "published_at"]
df[export_cols].to_csv("news_clusters_with_entities.csv", index=False)
print("✅ Exported clusters + entities to news_clusters_with_entities.csv")