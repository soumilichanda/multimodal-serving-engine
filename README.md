# High-Throughput Multimodal Serving & Drift Engine ðŸš€
[![CI Test & Build Verification](https://github.com/soumilichanda/multimodal-serving-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/soumilichanda/multimodal-serving-engine/actions/workflows/ci.yml)
![Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)
![Architecture](https://img.shields.io/badge/Architecture-Asynchronous%20FastAPI%20%2B%20ONNX-success)

A production-grade, asynchronous multimodal inference microservice built with **FastAPI**, **Pydantic V2**, thread-safe **(1)$ LRU caching**, dynamic micro-batching, live KS-test feature drift monitoring, a two-tier **Semantic Out-of-Distribution (OOD) Gatekeeper**, and a **Quantized ONNX Runtime** backend.

---

## ðŸ—ï¸ System Architecture

```text
Client Request (REST / Async JSON)
  â”‚
  â”œâ”€â”€ /v1/predict/vector  (Dense Feature Ingestion)
  â””â”€â”€ /v1/predict/image   (Base64 Image Streams)
  â”‚
  â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚             FastAPI Gateway & Ingestion Layer          â”‚
â”‚ â€¢ Pydantic V2 Strict Contract Enforcement              â”‚
â”‚ â€¢ Dimension Bounds & Magic Byte Header Validation      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                           â”‚
                           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                Thread-Safe LRU Cache Layer             â”‚
â”‚ â€¢ O(1) Cache Hit Intercept (OrderedDict + Mutex Lock)  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â”‚ (Cache Hit)               â”‚ (Cache Miss)
               â–¼                           â–¼
       [ Return Cached Payload ]  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                                  â”‚   Dynamic Micro-Batcher Worker    â”‚
                                  â”‚ â€¢ Capacity Flush: B_max = 16      â”‚
                                  â”‚ â€¢ Timeout Flush: delta_t_max = 8msâ”‚
                                  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                                    â”‚
                                                    â–¼
                                  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                                  â”‚   Tier-1: Semantic OOD Gatekeeper â”‚
                                  â”‚ â€¢ Magic Byte Pre-Validation       â”‚
                                  â”‚ â€¢ Synset Boundary: [151, 268] U   â”‚
                                  â”‚                    [281, 285]     â”‚
                                  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                                    â”‚ (In-Distribution)
                                                    â–¼
                                  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                                  â”‚   Tier-2: Quantized ONNX Runtime  â”‚
                                  â”‚ â€¢ Graph Optimization (ORT_ALL)    â”‚
                                  â”‚ â€¢ SIMD NCHW Tensor Execution      â”‚
                                  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                                    â”‚
                                                    â–¼
                                  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                                  â”‚    Live Telemetry & KS-Drift      â”‚
                                  â”‚ â€¢ Windowed p50/p95/p99 Latency    â”‚
                                  â”‚ â€¢ Two-Sample KS-Test Feature      â”‚
                                  â”‚   Drift Detection (p < 0.05)      â”‚
                                  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
`

---

## ðŸ“ Repository Structure

```text
multimodal-serving-engine/
â”‚
â”œâ”€â”€ app/
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ main.py              # FastAPI gateway, lifespan orchestration & routing
â”‚   â”œâ”€â”€ core/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”œâ”€â”€ cache.py         # Thread-safe O(1) LRU Inference Cache
â”‚   â”‚   â”œâ”€â”€ telemetry.py     # Latency profiler & Two-Sample KS-test drift engine
â”‚   â”‚   â””â”€â”€ batcher.py       # Asynchronous dynamic request micro-batcher
â”‚   â”œâ”€â”€ schemas/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â””â”€â”€ payload.py       # Pydantic V2 strict data contracts & field validators
â”‚   â””â”€â”€ services/
â”‚       â”œâ”€â”€ __init__.py
â”‚       â”œâ”€â”€ gatekeeper.py    # Tier-1 Semantic OOD Filter & NCHW Preprocessor
â”‚       â””â”€â”€ onnx_backend.py  # Tier-2 Quantized ONNX Runtime Inference Engine
â”‚
â”œâ”€â”€ benchmarks/
â”‚   â””â”€â”€ load_test.py         # Asynchronous 50-worker concurrency load harness
â”‚
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ test_api.py          # Endpoint integration, OOD rejection & caching tests
â”‚   â”œâ”€â”€ test_telemetry.py    # Latency profiler & Kolmogorov-Smirnov drift tests
â”‚   â””â”€â”€ test_batcher.py      # Micro-batcher capacity and timeout flush tests
â”‚
â”œâ”€â”€ Dockerfile               # Multi-stage non-root hardened container definition
â”œâ”€â”€ pytest.ini               # Pytest root path & asyncio execution flags
â”œâ”€â”€ requirements.txt         # Production and testing runtime dependencies
â””â”€â”€ README.md                # System architecture documentation & operational guide
`

---

## âš¡ Key Engineering Features

- **Two-Tier Inference & OOD Defense (pp/services/):**
  - **Tier-1 Gatekeeper (gatekeeper.py):** Pre-validates base64 magic byte headers (JPEG /9j/ and PNG iVBORw0KGgo) and filters out-of-distribution classes before inference.
  - **Tier-2 Quantized Backend (onnx_backend.py):** Executes quantized ONNX models using optimized graph execution (CPUExecutionProvider) with deterministic evaluation fallbacks.
- **Thread-Safe (1)$ LRU Cache (pp/core/cache.py):** Intercepts repeated feature vectors using SHA-256 state hashing and an eviction mutex lock.
- **Dynamic Micro-Batching (pp/core/batcher.py):** Coalesces point queries into dense 2D matrices using capacity ({max} = 16$) and temporal ($\Delta t_{max} = 8	ext{ ms}$) flushes.
- **Statistical Drift & Telemetry (pp/core/telemetry.py):** Tracks rolling latency percentiles ($, $, $, $) and Kolmogorov-Smirnov two-sample drift detection ( < 0.05$).
- **Automated Unit Testing (	ests/):** 12 comprehensive unit and integration tests covering API contracts, micro-batching, and OOD defenses.

---

## ðŸš¦ API Specification

| Method | Endpoint | Description | Request Schema | Response Status |
| :---: | :--- | :--- | :--- | :---: |
| **GET** | / | Service health status and navigation index | None | 200 OK |
| **GET** | /health | Health probe & active LRU cache metrics | None | 200 OK |
| **GET** | /v1/telemetry | Rolling latency percentiles & KS drift report | None | 200 OK |
| **POST** | /v1/predict/vector | Dynamic micro-batching inference & (1)$ cache | VectorInferenceRequest | 200 OK |
| **POST** | /v1/predict/image | OOD gatekeeper filtering & ONNX runtime inference | ImageInferenceRequest | 200 OK |

---

## ðŸ› ï¸ How to Run & Verify

### 1. Run Automated Test Suite
`powershell
python -m pytest tests/ -v
`

### 2. Launch Local Server
`powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
`
Interactive OpenAPI documentation is available at http://127.0.0.1:8000/docs.

### 3. Run High-Concurrency Benchmark
`powershell
python benchmarks/load_test.py
`

### 4. Containerized Execution (Docker)
`powershell
docker build -t multimodal-serving-engine:v1 .
docker run -d -p 8000:8000 --name serving-engine multimodal-serving-engine:v1
`

---

## âš¡ Concurrency & Latency Benchmark Results

High-concurrency synthetic load simulation executed via enchmarks/load_test.py across 50 asynchronous client workers:

| Metric | Target Specification | Empirical Result |
| :--- | :--- | :--- |
| **Concurrent Workers** | 50 Async Workers | 50 Simultaneous Workers |
| **Total Ingested Payloads** | 500 Requests | 500 / 500 (100% Success) |
| **Total Wall Time** | Real-Time Execution | 5.97s |
| **System Throughput** | High-Throughput Saturation | 83.7 req/s |
| **Median Latency ($)** | Low-Latency SLA (< 400 ms) | 293.13 ms |
| **Tail Latency ($)** | Sub-2.0s Bound | 1,660.29 ms |
| **Extreme Tail Latency ($)** | Worst-Case Tail Bound | 2,761.03 ms |

> Dynamic micro-batching coalesces point requests into vectorized SIMD evaluations under load, minimizing thread lock contention and preventing GPU/CPU starvation.

