"""
incident_ai.py
- Ingest CSV of historical incidents (one row per incident)
- Chunk/tokenize segments, compute embeddings via Gemini/GenAI API
- Store embeddings + metadata into Postgres (pgvector)
- Query pipeline: embed incoming incident, retrieve top-k similar segments
- Call Gemini to produce structured JSON recommendation
"""

import os
import json
import textwrap
from typing import List
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# DB
from sqlalchemy import create_engine, text

# GenAI client
from google import genai

# ---------- config ----------
load_dotenv()

DB_URL = os.getenv(
    "DB_URL", "postgresql://pguser:pgpass@localhost:5432/incidentsdb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-gecko-001")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-pro")
TOP_K = int(os.getenv("TOP_K", "6"))
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "400"))

# ✅ Initialize GenAI client using service account JSON
# The client will automatically read GOOGLE_APPLICATION_CREDENTIALS
client = genai.Client()

# ---------- helpers ----------


def chunk_text(text: str, max_tokens: int = 400) -> List[str]:
    words = text.split()
    chunks = []
    cur = []
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
    resp = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts
    )
    return [embedding.values for embedding in resp.embeddings]


def generate_recommendation(prompt: str, max_output_tokens: int = 500, temperature: float = 0.15) -> str:
    resp = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        generation_config={
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "topP": 0.9,
        }
    )
    return resp.text

# ---------- DB setup ----------


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
            {"iid": incident_id, "txt": segment_text,
             "meta": json.dumps(metadata), "emb": embedding}
        )
        conn.commit()


def query_similar(engine, query_embedding: List[float], top_k: int = 6):
    if not query_embedding:
        return []
    with engine.connect() as conn:
        q = text("""
            SELECT incident_id, segment_text, metadata, embedding
            FROM incident_segments
            ORDER BY embedding <#> :q_emb
            LIMIT :k
        """)
        rows = conn.execute(
            q, {"q_emb": query_embedding, "k": top_k}).fetchall()
        results = []
        for r in rows:
            results.append({
                "incident_id": r[0],
                "segment_text": r[1],
                "metadata": r[2]
            })
        return results


def build_prompt(current_incident: dict, retrieved: List[dict]) -> str:
    header = "You are an incident triage assistant. Produce a JSON with keys: recommended_solution (list of step strings), rationale (brief), confidence_score (0-1)."
    current = f"CURRENT INCIDENT:\nID: {current_incident.get('incident_id', 'N/A')}\nCategory: {current_incident.get('category', '')}\nDescription: {current_incident.get('description', '')}\nImpact: {current_incident.get('impact', '')}\nEnvironment: {current_incident.get('environment', '')}\n\n"
    examples = "RELEVANT HISTORIC SEGMENTS (top matches):\n"
    for i, seg in enumerate(retrieved, start=1):
        meta = seg.get("metadata") or {}
        examples += f"---\nMatch #{i}\nincident_id: {seg.get('incident_id')}\nsegment: {seg.get('segment_text')[:800]}\nmetadata: {json.dumps(meta)}\n\n"
    instruction = textwrap.dedent("""
    Using the current incident and the historic segments above:
    1) Give a step-by-step recommended_solution (short actionable steps).
    2) Provide a one-paragraph rationale linking to which historic segments inspired which steps.
    3) Provide a confidence_score between 0 and 1.
    Output ONLY valid JSON.
    """)
    return "\n".join([header, current, examples, instruction])

# ---------- ingestion ----------


def ingest_csv(csv_path: str, engine):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path, sep=";")
    if df.empty:
        raise ValueError(f"No data found in CSV file: {csv_path}")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        incident_id = str(row.get("incident_id") or f"row_{idx}")
        full_text = " ".join([
            str(row.get("description", "") or ""),
            str(row.get("root_cause", "") or ""),
            str(row.get("resolution", "") or ""),
        ]).strip()
        if not full_text:
            continue  # skip empty rows
        chunks = chunk_text(full_text, max_tokens=MAX_CHUNK_TOKENS)
        if not chunks:
            continue
        embeddings = embed_texts(chunks)
        for chunk, emb in zip(chunks, embeddings):
            metadata = {
                "category": row.get("category"),
                "impact": row.get("impact"),
                "date": str(row.get("date")),
            }
            upsert_segment(engine, incident_id, chunk, metadata, emb)

# ---------- realtime query pipeline ----------


def recommend_for_incident(engine, current_incident: dict, top_k: int = TOP_K):
    q_text = current_incident.get(
        "description", "") + " " + current_incident.get("environment", "")
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
            raise RuntimeError(
                "Failed to parse model output as JSON:\n" + gen_text)
    return {"recommendation": result, "raw_model_output": gen_text, "retrieved": retrieved}


# ---------- main ----------
if __name__ == "__main__":
    engine = create_engine(DB_URL)
    init_db(engine)

    # Example ingestion
    # ingest_csv("E:/AI Build/mitras/sample_incidents.csv", engine)

    # Example recommendation
    example_incident = {
        "incident_id": "INCOMING-001",
        "category": "Database",
        "description": "Users experiencing timeouts in order placement during peak. Error: DB connection timeout. Instance: orders-db-prod-1",
        "impact": "Order placement failure for ~20% users",
        "environment": "prod, db cluster: orders-db-cluster",
    }

    # result = recommend_for_incident(engine, example_incident)
    # print(json.dumps(result, indent=2))
