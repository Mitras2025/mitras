"""
incident_ai.py
- Handles CSV ingestion, chunking, embeddings, DB storage, and recommendations.
"""

import os
import json
import textwrap
import pandas as pd
from tqdm import tqdm
from typing import List
from io import StringIO
import requests
import json
from google import genai

from sqlalchemy import create_engine, text
from google import genai
from dotenv import load_dotenv

# ---------- config ----------
load_dotenv()

DB_URL = os.getenv(
    "DATABASE_URL",
    os.getenv("DB_URL", "postgresql://pguser:pgpass@localhost:5432/incidentsdb")
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-gecko-001")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-pro")
TOP_K = int(os.getenv("TOP_K", "6"))
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "400"))

# GenAI client
cred_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if cred_json:
    import io
    with open("/tmp/gcloud_sa.json", "w") as f:
        f.write(cred_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcloud_sa.json"

# Initialize GenAI client
client = genai.Client(
    vertexai=True,
    project="mitras-469413",
    location="us-central1"
)

# ---------- helpers ----------
def chunk_text(text: str, max_tokens: int = 400) -> List[str]:
    words = text.split()
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= max_tokens:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
    return [embedding.values for embedding in resp.embeddings]


def generate_recommendation(prompt: str, max_output_tokens: int = 500, temperature: float = 0.15) -> str:
    resp = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        generation_config={"temperature": temperature, "maxOutputTokens": max_output_tokens, "topP": 0.9}
    )
    return resp.text

# ---------- DB ----------
def init_db(engine):
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS incident_segments (
            id SERIAL PRIMARY KEY,
            incident_id TEXT,
            segment_text TEXT,
            metadata JSONB,
            embedding vector(1536)
        );
        """))
        conn.commit()


def upsert_segment(engine, incident_id: str, segment_text: str, metadata: dict, embedding: List[float]):
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO incident_segments (incident_id, segment_text, metadata, embedding) VALUES (:iid, :txt, :meta, :emb)"),
            {"iid": incident_id, "txt": segment_text, "meta": json.dumps(metadata), "emb": embedding}
        )
        conn.commit()


def query_similar(engine, query_embedding: List[float], top_k: int = 6):
    if not query_embedding:
        return []
    with engine.connect() as conn:
        q = text("""
            SELECT incident_id, segment_text, metadata
            FROM incident_segments
            ORDER BY embedding <#> :q_emb
            LIMIT :k
        """)
        rows = conn.execute(q, {"q_emb": query_embedding, "k": top_k}).fetchall()
        return [{"incident_id": r[0], "segment_text": r[1], "metadata": r[2]} for r in rows]

# ---------- ingestion ----------
def ingest_csv(csv_path_or_url, engine):
    """
    Ingest CSV data into DB.
    Accepts: URL, local path, or StringIO
    """
    # URL
    if isinstance(csv_path_or_url, str) and csv_path_or_url.startswith("http"):
        resp = requests.get(csv_path_or_url)
        resp.raise_for_status()
        csv_file_like = StringIO(resp.text)
    # File-like
    elif isinstance(csv_path_or_url, StringIO):
        csv_file_like = csv_path_or_url
    # Local path
    else:
        if not os.path.exists(csv_path_or_url):
            raise FileNotFoundError(f"CSV file not found: {csv_path_or_url}")
        csv_file_like = csv_path_or_url

    df = pd.read_csv(csv_file_like, sep=";")
    if df.empty:
        raise ValueError(f"No data found in CSV: {csv_path_or_url}")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        incident_id = str(row.get("incident_id") or f"row_{idx}")
        full_text = " ".join([str(row.get(col, "") or "") for col in ["description", "root_cause", "resolution"]]).strip()
        if not full_text:
            continue

        chunks = chunk_text(full_text, max_tokens=MAX_CHUNK_TOKENS)
        if not chunks:
            continue
        embeddings = embed_texts(chunks)
        for chunk, emb in zip(chunks, embeddings):
            metadata = {"category": row.get("category"), "impact": row.get("impact"), "date": str(row.get("date"))}
            upsert_segment(engine, incident_id, chunk, metadata, emb)


# ---------- recommendation ----------
def build_prompt(current_incident: dict, retrieved: List[dict]) -> str:
    header = "You are an incident triage assistant. Produce a JSON with keys: recommended_solution (list), rationale (brief), confidence_score (0-1)."
    current = f"CURRENT INCIDENT:\nID: {current_incident.get('incident_id','N/A')}\nCategory: {current_incident.get('category','')}\nDescription: {current_incident.get('description','')}\nImpact: {current_incident.get('impact','')}\nEnvironment: {current_incident.get('environment','')}\n\n"
    examples = "RELEVANT HISTORIC SEGMENTS:\n"
    for i, seg in enumerate(retrieved, start=1):
        meta = seg.get("metadata") or {}
        examples += f"---\nMatch #{i}\nincident_id: {seg.get('incident_id')}\nsegment: {seg.get('segment_text')[:800]}\nmetadata: {json.dumps(meta)}\n\n"
    instruction = textwrap.dedent("""
        Using the current incident and historic segments above:
        1) Give step-by-step recommended_solution (short actionable steps).
        2) Provide a rationale linking historic segments.
        3) Provide a confidence_score 0-1.
        Output ONLY valid JSON.
    """)
    return "\n".join([header, current, examples, instruction])


def recommend_for_incident(engine, current_incident: dict, top_k: int = TOP_K):
    q_text = current_incident.get("description", "") + " " + current_incident.get("environment", "")
    if not q_text.strip():
        raise ValueError("Incident description and environment are empty.")
    q_emb = embed_texts([q_text])[0]
    retrieved = query_similar(engine, q_emb, top_k)
    prompt = build_prompt(current_incident, retrieved)
    gen_text = generate_recommendation(prompt)
    try:
        result = json.loads(gen_text)
    except Exception:
        import re
        m = re.search(r"\{.*\}", gen_text, flags=re.S)
        if m:
            result = json.loads(m.group(0))
        else:
            raise RuntimeError("Failed to parse model output as JSON:\n" + gen_text)
    return {"recommendation": result, "raw_model_output": gen_text, "retrieved": retrieved}
