from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine
from . import incident_ai
import os
import json
from incident_ai import (
    init_db,
    ingest_csv,
    recommend_for_incident,
    TOP_K,
    DB_URL,
)

# ---------- FastAPI setup ----------
app = FastAPI(title="Incident AI API")
engine = create_engine(DB_URL)
init_db(engine)

# ---------- Pydantic models ----------


class IncidentInput(BaseModel):
    incident_id: str
    category: str
    description: str
    impact: str
    environment: str


class CSVIngestInput(BaseModel):
    csv_path: str

# ---------- Test endpoint ----------


@app.get("/ping")
def ping():
    return {"status": "ok", "message": "Server is running!"}

# ---------- API Endpoints ----------


@app.post("/ingest_csv")
def ingest_csv_endpoint(data: CSVIngestInput):
    try:
        print(f"➡️ Trying to ingest file: {data.csv_path}")
        ingest_csv(data.csv_path, engine)
        return {"status": "success", "message": f"Ingested CSV {data.csv_path}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.post("/recommend")
def recommend_incident_endpoint(incident: IncidentInput):
    incident_dict = incident.dict()
    try:
        result = recommend_for_incident(engine, incident_dict, top_k=TOP_K)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
