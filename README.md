# Mitras Incident AI

## Overview
**Mitras Incident AI** is a FastAPI-based service that helps organizations ingest historical incidents, compute embeddings using **Google Vertex AI (Gemini/GenAI)**, and provide **AI-powered recommendations** for incoming incidents.  

It leverages a **PostgreSQL database with pgvector** to store embeddings and metadata for efficient similarity search.

---

## ✨ Features

### 1. CSV Ingestion
- Accepts historical incidents in CSV format.
- Chunks incident text (description, root cause, resolution) into token segments.
- Computes embeddings using **Vertex AI’s embedding model**.
- Stores embeddings and metadata in **PostgreSQL (with pgvector)**.

### 2. Realtime Incident Recommendation
- Takes an incoming incident JSON.
- Computes embedding of the incident description and environment.
- Retrieves **top-k most similar** historical incident segments from the database.
- Generates structured JSON recommendation using **Vertex AI**.

### 3. API Endpoints
- `POST /ingest_csv` → Ingest historical incidents from CSV.
- `POST /recommend` → Get AI-generated recommendation for a new incident.
- `GET /ping` → Test endpoint to verify server health.

---

## 🏗️ Architecture

```
+----------------+      +----------------+      +------------------+
| Historical CSV | ---> | Mitras Backend | ---> | PostgreSQL +     |
| (incidents)    |      | (FastAPI)      |      | pgvector         |
+----------------+      +----------------+      +------------------+
                                |
                                v
                        +------------------+
                        | Vertex AI /      |
                        | Gemini Models    |
                        +------------------+
                                |
                                v
                    +-------------------------+
                    | Structured JSON Output  |
                    +-------------------------+
```

- **Backend**: Python 3.13 + FastAPI  
- **Database**: PostgreSQL + pgvector  
- **AI Models**: Vertex AI (GenAI / Gemini)  
- **Deployment**: Render.com (or any cloud service)  
- **Environment Variables**: `.env` or secret JSON for Vertex AI credentials  

---

## ⚙️ Setup

### 1. Clone Repository
```bash
git clone https://github.com/Mitras2025/mitras.git
cd mitras
```

### 2. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# OR
.venv\Scriptsctivate      # Windows

pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file:
```ini
DATABASE_URL=postgresql://pguser:pgpass@localhost:5432/incidentsdb
EMBEDDING_MODEL=text-embedding-gecko-001
GENERATION_MODEL=gemini-2.5-pro
TOP_K=6
MAX_CHUNK_TOKENS=400
GOOGLE_APPLICATION_CREDENTIALS_JSON='{
  "type": "service_account",
  "project_id": "...",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "...",
  ...
}'
```

### 4. Run Locally
```bash
uvicorn mitras.incident_ai_api:app --host 0.0.0.0 --port 10000
```

---

## 📡 API Usage

### 1. Health Check
```bash
curl -X GET http://localhost:10000/ping
```

### 2. Ingest CSV
```bash
curl -X POST http://localhost:10000/ingest_csv \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "https://raw.githubusercontent.com/Mitras2025/mitras/main/mitras/sample_incidents.csv"}'
```
- `csv_path` can be a URL or local file path.  
- CSV must have columns:  
  `incident_id, description, root_cause, resolution, category, impact, date`

### 3. Get Recommendation
```bash
curl -X POST http://localhost:10000/recommend \
  -H "Content-Type: application/json" \
  -d '{
        "incident_id": "INCOMING-001",
        "category": "Database",
        "description": "Users experiencing timeouts in order placement during peak. Error: DB connection timeout. Instance: orders-db-prod-1",
        "impact": "Order placement failure for ~20% users",
        "environment": "prod, db cluster: orders-db-cluster"
      }'
```

**Sample Response:**
```json
{
  "recommendation": {
    "recommended_solution": ["Step 1", "Step 2", "..."],
    "rationale": "Explanation...",
    "confidence_score": 0.85
  },
  "raw_model_output": "...",
  "retrieved": [
    {
      "incident_id": "...",
      "segment_text": "...",
      "metadata": {...}
    }
  ]
}
```

---

## 🔄 Internal Workflow

1. **CSV Ingestion**
   - Read CSV → skip empty rows → join description/root_cause/resolution → chunk → embed → store in Postgres.

2. **Recommendation Pipeline**
   - Embed incoming incident → retrieve top-k similar segments → build prompt → call Gemini → parse JSON → return recommendation.

---

## 📌 Notes
- **Billing Required**: Vertex AI requires an active billing account.  
- **Embedding Dimension**: `1536` (for `text-embedding-gecko-001`).  
- **Database**: pgvector extension must be enabled.  
- **Deployment**: On Render.com, store JSON key securely in environment variables.  

---

## 🚀 Roadmap / Improvements
- Async ingestion for large CSVs.  
- Error logging and retry for Vertex AI API failures.  
- Authentication for API endpoints.  
- Store raw CSV and ingestion logs for auditing.  

---

## 📊 Data Flow Diagram

```
CSV Files → FastAPI → Parsing/Chunking → Vertex AI Embeddings
         → PostgreSQL + pgvector → Similarity Search
         → Gemini Model → JSON Recommendation → Client
```

---

## License
MIT License © 2025 Mitras
