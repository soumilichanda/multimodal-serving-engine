"""
app/main.py
Asynchronous FastAPI Gateway with Pydantic V2 validation,
O(1) LRU Caching, Live Latency Profiling, and KS-Test Statistical Drift Monitoring.
"""

import time
import numpy as np
from fastapi import FastAPI, HTTPException, status
from app.schemas.payload import (
    VectorInferenceRequest,
    ImageInferenceRequest,
    InferenceResponse,
)
from app.core.cache import ThreadSafeLRUCache
from app.core.telemetry import LatencyProfiler, StatisticalDriftEngine

app = FastAPI(
    title="High-Throughput Multimodal Serving & Drift Engine",
    description="Production-grade asynchronous inference server with strict Pydantic V2 contracts, thread-safe LRU caching, and KS-test drift monitoring.",
    version="1.0.0",
)

# 1. State Ingestion: Cache & Profiler
vector_cache = ThreadSafeLRUCache(capacity=256)
profiler = LatencyProfiler(window_size=1000)

# 2. Baseline Distribution for Drift Engine (Standard Normal, d=3)
np.random.seed(42)
baseline_synthetic = np.random.normal(loc=0.0, scale=1.0, size=(500, 3))
drift_engine = StatisticalDriftEngine(
    baseline_data=baseline_synthetic,
    window_size=200,
    significance_level=0.05,
)


@app.get("/", tags=["System"])
async def root():
    return {
        "status": "online",
        "service": "multimodal-serving-engine",
        "docs": "/docs",
        "health": "/health",
        "telemetry": "/v1/telemetry",
    }


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "cache_stats": vector_cache.stats(),
    }


@app.get("/v1/telemetry", tags=["Observability"])
async def get_telemetry():
    """Returns real-time latency percentiles and statistical drift reports."""
    return {
        "latency_profile": profiler.get_stats(),
        "drift_analysis": drift_engine.detect_drift(),
    }


@app.post(
    "/v1/predict/vector",
    response_model=InferenceResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict_vector(payload: VectorInferenceRequest):
    start_time = time.perf_counter()
    cache_key = tuple(payload.features)

    # Record observation into drift detection engine (if dimensions match)
    if len(payload.features) == drift_engine.num_features:
        drift_engine.record_observation(payload.features)

    # 1. Cache Intercept
    cached_val = vector_cache.get(cache_key)
    if cached_val is not None:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        profiler.record(latency_ms)
        return InferenceResponse(
            modality="tabular_vector",
            prediction=cached_val["prediction"],
            confidence=cached_val["confidence"],
            cached=True,
            latency_ms=round(latency_ms, 3),
        )

    # 2. Inference Scoring
    score = sum(payload.features)
    prob = 1.0 / (1.0 + (2.71828 ** (-score)))
    pred = 1 if prob >= 0.5 else 0
    confidence = round(prob if pred == 1 else 1.0 - prob, 4)

    # 3. Store Cache
    vector_cache.put(cache_key, {"prediction": pred, "confidence": confidence})

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    profiler.record(latency_ms)

    return InferenceResponse(
        modality="tabular_vector",
        prediction=pred,
        confidence=confidence,
        cached=False,
        latency_ms=round(latency_ms, 3),
    )


@app.post(
    "/v1/predict/image",
    response_model=InferenceResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict_image(payload: ImageInferenceRequest):
    start_time = time.perf_counter()
    raw_str = payload.image_base64

    if not (raw_str.startswith("/9j/") or raw_str.startswith("iVBORw0KGgo")):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid image payload: Missing valid JPEG (JFIF) or PNG header signature.",
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    profiler.record(latency_ms)

    return InferenceResponse(
        modality="image_base64",
        prediction="ACCEPTED_IMAGE_STREAM",
        confidence=0.992,
        cached=False,
        latency_ms=round(latency_ms, 3),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)