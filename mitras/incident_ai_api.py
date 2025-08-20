from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine
from io import StringIO
import requests

from .incident_ai import init_db, ingest_csv, recommend_for_incident, TOP_K, DB_URL

# ---------- FastAPI ----------
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

# ---------- Test ----------
@app.get("/ping")
def ping():
    return {"status": "ok", "message": "Server running"}

# ---------- Endpoints ----------
@app.post("/ingest_csv")
def ingest_csv_endpoint(data: CSVIngestInput):
    try:
        csv_path_or_url = data.csv_path
        if csv_path_or_url.startswith("http"):
            resp = requests.get(csv_path_or_url)
            resp.raise_for_status()
            csv_file_like = StringIO(resp.text)
        else:
            csv_file_like = csv_path_or_url
        ingest_csv(csv_file_like, engine)
        return {"status": "success", "message": f"Ingested CSV {csv_path_or_url}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.post("/recommend")
def recommend_incident_endpoint(incident: IncidentInput):
    try:
        result = recommend_for_incident(engine, incident.dict(), top_k=TOP_K)
        return {"status": "success", "result": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
