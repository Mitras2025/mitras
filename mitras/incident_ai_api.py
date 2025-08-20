from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine
import os
import pandas as pd
import requests
from io import StringIO
import json
from .incident_ai import (
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
        csv_path_or_url = data.csv_path

        if csv_path_or_url.startswith("http"):
            # download CSV from URL
            response = requests.get(csv_path_or_url)
            response.raise_for_status()  # fail if URL is invalid
            csv_file_like = StringIO(response.text)
        else:
            # local file path
            csv_file_like = csv_path_or_url

        # ingest_csv must be able to accept a file-like object (StringIO)
        ingest_csv(csv_file_like, engine)
        return {"status": "success", "message": f"Ingested CSV {csv_path_or_url}"}

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
