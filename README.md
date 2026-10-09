# High-Throughput Multimodal Serving & Drift Engine 🚀
[![CI Test & Build Verification](https://github.com/soumilichanda/multimodal-serving-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/soumilichanda/multimodal-serving-engine/actions/workflows/ci.yml)
![Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)
![Architecture](https://img.shields.io/badge/Architecture-Asynchronous%20FastAPI%20%2B%20ONNX-success)

A production-grade, asynchronous multimodal inference microservice built with **FastAPI**, **Pydantic V2**, thread-safe **(1)$ LRU caching**, dynamic micro-batching, live KS-test feature drift monitoring, a two-tier **Semantic Out-of-Distribution (OOD) Gatekeeper**, and a **Quantized ONNX Runtime** backend.

---

## 🏗️ System Architecture

```Text
Client Request (REST / Async JSON)
  │
  ├── /v1/predict/vector  (Dense Feature Ingestion)
  └── /v1/predict/image   (Base64 Image Streams)
  │
  ▼
┌────────────────────────────────────────────────────────┐
│             FastAPI Gateway & Ingestion Layer          │
│ • Pydantic V2 Strict Contract Enforcement              │
│ • Dimension Bounds & Magic Byte Header Validation      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                Thread-Safe LRU Cache Layer             │
│ • O(1) Cache Hit Intercept (OrderedDict + Mutex Lock)  │
└──────────────┬───────────────────────────┬─────────────┘
               │ (Cache Hit)               │ (Cache Miss)
               ▼                           ▼
       [ Return Cached Payload ]  ┌───────────────────────────────────┐
                                  │   Dynamic Micro-Batcher Worker    │
                                  │ • Capacity Flush: B_max = 16      │
                                  │ • Timeout Flush: delta_t_max = 8ms│
                                  └─────────────────┬─────────────────┘
                                                    │
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │   Tier-1: Semantic OOD Gatekeeper │
                                  │ • Magic Byte Pre-Validation       │
                                  │ • Synset Boundary: [151, 268] U   │
                                  │                    [281, 285]     │
                                  └─────────────────┬─────────────────┘
                                                    │ (In-Distribution)
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │   Tier-2: Quantized ONNX Runtime  │
                                  │ • Graph Optimization (ORT_ALL)    │
                                  │ • SIMD NCHW Tensor Execution      │
                                  └─────────────────┬─────────────────┘
                                                    │
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │    Live Telemetry & KS-Drift      │
                                  │ • Windowed p50/p95/p99 Latency    │
                                  │ • Two-Sample KS-Test Feature      │
                                  │   Drift Detection (p < 0.05)      │
                                  └───────────────────────────────────┘
`

---

## 📁 Repository Structure

```Text
multimodal-serving-engine/
│
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI gateway, lifespan orchestration & routing
│   ├── core/
│   │   ├── __init__.py
│   │   ├── cache.py         # Thread-safe O(1) LRU Inference Cache
│   │   ├── telemetry.py     # Latency profiler & Two-Sample KS-test drift engine
│   │   └── batcher.py       # Asynchronous dynamic request micro-batcher
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── payload.py       # Pydantic V2 strict data contracts & field validators
│   └── services/
│       ├── __init__.py
│       ├── gatekeeper.py    # Tier-1 Semantic OOD Filter & NCHW Preprocessor
│       └── onnx_backend.py  # Tier-2 Quantized ONNX Runtime Inference Engine
│
├── benchmarks/
│   └── load_test.py         # Asynchronous 50-worker concurrency load harness
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py          # Endpoint integration, OOD rejection & caching tests
│   ├── test_telemetry.py    # Latency profiler & Kolmogorov-Smirnov drift tests
│   └── test_batcher.py      # Micro-batcher capacity and timeout flush tests
│
├── Dockerfile               # Multi-stage non-root hardened container definition
├── pytest.ini               # Pytest root path & asyncio execution flags
├── requirements.txt         # Production and testing runtime dependencies
└── README.md                # System architecture documentation & operational guide
`

---

## ⚡ Key Engineering Features

- **Two-Tier Inference & OOD Defense (pp/services/):**
  - **Tier-1 Gatekeeper (gatekeeper.py):** Pre-validates base64 magic byte headers (JPEG /9j/ and PNG iVBORw0KGgo) and filters out-of-distribution classes before inference.
  - **Tier-2 Quantized Backend (onnx_backend.py):** Executes quantized ONNX models using optimized graph execution (CPUExecutionProvider) with deterministic evaluation fallbacks.
- **Thread-Safe (1)$ LRU Cache (pp/core/cache.py):** Intercepts repeated feature vectors using SHA-256 state hashing and an eviction mutex lock.
- **Dynamic Micro-Batching (pp/core/batcher.py):** Coalesces point queries into dense 2D matrices using capacity ({max} = 16$) and temporal ($\Delta t_{max} = 8	ext{ ms}$) flushes.
- **Statistical Drift & Telemetry (pp/core/telemetry.py):** Tracks rolling latency percentiles ($, $, $, $) and Kolmogorov-Smirnov two-sample drift detection ( < 0.05$).
- **Automated Unit Testing (	ests/):** 12 comprehensive unit and integration tests covering API contracts, micro-batching, and OOD defenses.

---

## 🚦 API Specification

| Method | Endpoint | Description | Request Schema | Response Status |
| :---: | :--- | :--- | :--- | :---: |
| **GET** | / | Service health status and navigation index | None | 200 OK |
| **GET** | /health | Health probe & active LRU cache metrics | None | 200 OK |
| **GET** | /v1/telemetry | Rolling latency percentiles & KS drift report | None | 200 OK |
| **POST** | /v1/predict/vector | Dynamic micro-batching inference & (1)$ cache | VectorInferenceRequest | 200 OK |
| **POST** | /v1/predict/image | OOD gatekeeper filtering & ONNX runtime inference | ImageInferenceRequest | 200 OK |

---

## 🛠️ How to Run & Verify

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

## ⚡ Concurrency & Latency Benchmark Results

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
