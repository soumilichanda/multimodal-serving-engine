# High-Throughput Multimodal Serving & Drift Engine 🚀

A production-grade, asynchronous multimodal inference microservice built with **FastAPI**, **Pydantic V2**, thread-safe **$O(1)$ LRU caching**, dynamic micro-batching, live KS-test feature drift monitoring, and multi-stage container deployment.

---

## 🏗️ System Architecture

Client Request (REST / Async JSON)
        │
        ├──► /v1/predict/vector  (Dense Feature Ingestion)
        └──► /v1/predict/image   (Base64 Image Streams)
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│         FastAPI Gateway & Ingestion Layer              │
│  • Pydantic V2 Strict Contract Enforcement             │
│  • Dimension Bounds & Magic Byte Header Validation     │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│            Thread-Safe LRU Cache Layer                 │
│  • O(1) Cache Hit Intercept (OrderedDict + Mutex Lock) │
│  • Bypasses Model Inference on Repeated Queries        │
└────────────┬──────────────────────────────┬────────────┘
             │ (Hit)                        │ (Miss)
             ▼                              ▼
    [ Return Cached Payload ]    ┌───────────────────────────────────┐
                                 │   Dynamic Micro-Batcher Worker    │
                                 │   • Capacity Flush: B_max = 16    │
                                 │   • Timeout Flush: Δt_max = 8ms   │
                                 └─────────────────┬─────────────────┘
                                                   │
                                                   ▼
                                         [ Vectorized SIMD Eval ]
                                                   │
                                                   ▼
                                 ┌───────────────────────────────────┐
                                 │    Live Telemetry & KS-Drift      │
                                 │   • Windowed p50/p95/p99 Latency  │
                                 │   • Two-Sample KS-Test Feature Drift│
                                 └───────────────────────────────────┘
📁 Repository StructurePlaintextmultimodal-serving-engine/
│
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI gateway, lifespan management & routing
│   ├── core/
│   │   ├── __init__.py
│   │   ├── cache.py          # Thread-safe O(1) LRU Inference Cache
│   │   ├── telemetry.py      # Latency profiler & Two-Sample KS-test drift engine
│   │   └── batcher.py        # Asynchronous dynamic request micro-batcher
│   └── schemas/
│       ├── __init__.py
│       └── payload.py        # Pydantic V2 strict data contracts & field validators
│
├── benchmarks/
│   └── load_test.py          # Asynchronous high-concurrency benchmark harness
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py           # Endpoint integration & payload rejection tests
│   ├── test_telemetry.py     # Profiler & Kolmogorov-Smirnov drift alerts tests
│   └── test_batcher.py       # Micro-batcher capacity and timeout flush invariants
│
├── Dockerfile                # Multi-stage non-root container definition
├── pytest.ini                # Pytest path and asyncio configuration
├── requirements.txt          # Production and testing runtime dependencies
└── README.md                 # System architecture documentation and operational guide
⚡ Key Engineering FeaturesStrict Pydantic V2 Data Contracts (app/schemas/payload.py): Enforces vector dimension boundaries ($1 \le d \le 2048$) and cleans base64 image strings with @field_validator before downstream processing.Thread-Safe $O(1)$ LRU Cache (app/core/cache.py): Prevents duplicate computation under concurrent request workloads using OrderedDict guarded by a thread mutex.Dynamic Micro-Batching (app/core/batcher.py): Coalesces incoming point queries into dense 2D matrices for SIMD vectorized inference. Flushes automatically upon reaching capacity ($B_{\max} = 16$) or timeout ($\Delta t_{\max} = 8\text{ ms}$).Statistical Drift & Telemetry (app/core/telemetry.py): Tracks rolling latency percentiles ($p50$, $p90$, $p95$, $p99$) and applies two-sample Kolmogorov-Smirnov tests (scipy.stats.ks_2samp) to flag production distribution drift ($p < 0.05$).Hardened Multi-Stage Dockerfile (Dockerfile): Multi-stage build isolating dependencies into an unprivileged non-root user (appuser, UID 10001) runtime with native container health checks.Concurrency Load Testing (benchmarks/load_test.py): Asynchronous load generator driving 50 concurrent workers over 500 requests to measure endpoint saturation and batch efficiency.📊 API SpecificationMethodEndpointDescriptionRequest SchemaResponse StatusGET/Service health status and navigation indexNone200 OKGET/healthHealth probe & active LRU cache metricsNone200 OKGET/v1/telemetryRolling latency percentiles & KS drift reportNone200 OKPOST/v1/predict/vectorDynamic micro-batching inference & $O(1)$ cacheVectorInferenceRequest200 OK / 422POST/v1/predict/imageHeader byte sanitization & vision validationImageInferenceRequest200 OK / 422🛠️ Verification & Execution1. Run Automated Test SuitesPowerShellpython -m pytest tests/ -v
2. Launch Local ServerPowerShelluvicorn app.main:app --host 127.0.0.1 --port 8000
Interactive OpenAPI documentation is available at http://127.0.0.1:8000/docs.3. Run High-Concurrency BenchmarkWith the server running, execute in a separate terminal:PowerShellpython benchmarks/load_test.py
4. Containerized Execution (Docker)PowerShelldocker build -t multimodal-serving-engine:v1 .
docker run -d -p 8000:8000 --name serving-engine multimodal-serving-engine:v1
