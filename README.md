# High-Throughput Multimodal Serving & Drift Engine 🚀

A production-grade, asynchronous multimodal inference microservice built with **FastAPI**, **Pydantic V2**, thread-safe **$O(1)$ LRU caching**, header-level input sanitization, and automated test coverage.

---

## 🏗️ System Architecture

```text
Client Request (REST / JSON)
        │
        ├──► /v1/predict/vector  (Tabular Dense Features)
        └──► /v1/predict/image   (Base64 Image Streams)
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│         FastAPI Gateway & Ingestion Layer              │
│  • Pydantic V2 Strict Contract Enforcement             │
│  • Dimension Bounds Check (1 <= len <= 2048)          │
│  • Header-Level Magic Byte Sanitization (JFIF / PNG)  │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│            Thread-Safe LRU Cache Layer                 │
│  • O(1) Cache Hit Intercept (OrderedDict + Lock)      │
│  • Bypasses Model Inference on Repeated Queries        │
│  • Live Hit / Miss Ratio Telemetry                     │
└────────────┬──────────────────────────────┬────────────┘
             │ (Hit)                        │ (Miss)
             ▼                              ▼
    [ Return Cached Payload ]      [ Model Inference Routine ]
                                            │
                                            ▼
                                   [ Update Cache Entry ]
                                            │
                                            ▼
┌────────────────────────────────────────────────────────┐
│               Standardized Response                    │
│   { modality, prediction, confidence, cached, latency }│
└────────────────────────────────────────────────────────┘

```

---

## 📁 Repository Structure

```text
multimodal-serving-engine/
│
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI gateway, routes (/health, /predict), and cache integration
│   ├── api/                  # Modular endpoint routers
│   ├── core/
│   │   ├── __init__.py
│   │   └── cache.py          # Thread-safe O(1) LRU Inference Cache (OrderedDict + Mutex Lock)
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── payload.py        # Strict Pydantic V2 contracts (Vector & Image models with field validators)
│   └── services/             # Downstream model inference adapters & scoring pipelines
│
├── tests/
│   ├── __init__.py
│   └── test_api.py           # Pytest suite: health status, cache hit/miss, and payload validation
│
├── .gitignore                # Standard Python, pytest, virtualenv, and IDE exclusions
├── requirements.txt          # Production and testing runtime dependencies
└── README.md                 # System architecture documentation and operational guide

```

---

## ⚡ Key Engineering Features

* **Strict Pydantic V2 Data Contracts (`app/schemas/payload.py`):** Enforces input vector dimension caps ($1 \le d \le 2048$) and strips empty payloads via `@field_validator` before downstream processing.
* **Thread-Safe $O(1)$ LRU Cache (`app/core/cache.py`):** Mitigates repeated computation overhead for duplicate feature vectors using Python's `threading.Lock` across concurrent asynchronous event loops.
* **Low-Level Header Sanitization (`app/main.py`):** Intercepts raw base64 image streams, enforcing valid JPEG (`/9j/`) or PNG (`iVBORw0KGgo`) byte headers and rejecting malformed payloads with `HTTP 422 Unprocessable Content`.
* **Automated Test Harness (`tests/test_api.py`):** Full end-to-end integration tests using `pytest` and `httpx`, validating cache transitions, input rejections, and latency profiling.

---

## 📊 API Specification

### Endpoints Overview

| Method | Endpoint | Description | Request Schema | Response Status |
| --- | --- | --- | --- | --- |
| `GET` | `/` | Service health status and navigation index | None | `200 OK` |
| `GET` | `/health` | Diagnostic check and active LRU cache metrics | None | `200 OK` |
| `POST` | `/v1/predict/vector` | Tabular vector inference with $O(1)$ LRU caching | `VectorInferenceRequest` | `200 OK` / `422` |
| `POST` | `/v1/predict/image` | Base64 image intake with byte header validation | `ImageInferenceRequest` | `200 OK` / `422` |

---

## 🛠️ Installation & Local Execution

### 1. Clone the Repository

```bash
git clone https://github.com/soumilichanda/multimodal-serving-engine.git
cd multimodal-serving-engine

```

### 2. Set Up Virtual Environment & Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

```

### 3. Run the Automated Test Suite

```powershell
python -m pytest tests/test_api.py -v

```

### 4. Launch the Microservice

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

```

Interactive OpenAPI documentation is available at `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)`.

---

### Step to Save and Commit

1. Open `README.md` in VS Code, paste the markdown above, and save (**Ctrl + S**).
2. Run these terminal commands to commit and push:

```powershell
git add README.md
git commit -m "docs: write comprehensive architectural README with design diagram, API specs, and execution instructions"
git push origin main

```