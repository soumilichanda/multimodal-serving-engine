"""
app/main.py
Asynchronous FastAPI Gateway with Dynamic Batching,
O(1) LRU Caching, Live Latency Profiling, and Drift Telemetry.
"""
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, status

from app.core.batcher import DynamicBatcher
from app.core.cache import ThreadSafeLRUCache
from app.core.telemetry import LatencyProfiler, StatisticalDriftEngine
from app.schemas.payload import (
    ImageInferenceRequest,
    InferenceResponse,
    VectorInferenceRequest,
)
from app.services.gatekeeper import SemanticOODGatekeeper
from app.services.onnx_backend import ONNXInferenceBackend

# --- 1. Global State & Observability Engines ---
vector_cache = ThreadSafeLRUCache(capacity=256)
profiler = LatencyProfiler(window_size=1000)
gatekeeper = SemanticOODGatekeeper(confidence_threshold=0.08)
onnx_backend = ONNXInferenceBackend(model_path=None)

# Statistical baseline (d=3)
np.random.seed(42)
baseline_synthetic = np.random.normal(loc=0.0, scale=1.0, size=(500, 3))
drift_engine = StatisticalDriftEngine(
    baseline_data=baseline_synthetic,
    window_size=200,
    significance_level=0.05,
)


def batch_forward_inference(batch_matrix: np.ndarray) -> list[dict]:
    """
    Vectorized forward pass over dense batch matrices (B, d).
    Replaces point scalar loops with optimized NumPy SIMD operations.
    """
    # Linear projection: sum across features per sample
    scores = np.sum(batch_matrix, axis=1)
    probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -250.0, 250.0)))
    predictions = (probs >= 0.5).astype(int)

    output = []
    for pred, prob in zip(predictions, probs):
        conf = float(prob if pred == 1 else 1.0 - prob)
        output.append({"prediction": int(pred), "confidence": round(conf, 4)})
    return output


# Initialize batcher (batch size: 16, latency budget: 8ms)
dynamic_batcher = DynamicBatcher(
    forward_fn=batch_forward_inference,
    max_batch_size=16,
    max_delay_ms=8.0,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start batcher worker loop
    await dynamic_batcher.start()
    yield
    # Shutdown: Cleanly halt worker loop
    await dynamic_batcher.stop()


app = FastAPI(
    title="High-Throughput Multimodal Serving & Drift Engine",
    description="Production-grade asynchronous inference server with dynamic micro-batching, LRU caching, and drift monitoring.",
    version="1.1.0",
    lifespan=lifespan,
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

    # Record into drift engine
    if len(payload.features) == drift_engine.num_features:
        drift_engine.record_observation(payload.features)

    # 1. Cache Intercept (O(1))
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

    # 2. Dynamic Asynchronous Micro-Batching
    batch_res = await dynamic_batcher.enqueue(payload.features)

    # 3. Store into Cache
    vector_cache.put(cache_key, batch_res)

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    profiler.record(latency_ms)

    return InferenceResponse(
        modality="tabular_vector",
        prediction=batch_res["prediction"],
        confidence=batch_res["confidence"],
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

    # 1. Byte Sanitization
    try:
        raw_bytes = gatekeeper.validate_magic_bytes(payload.image_base64)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    # 2. Preprocess to Tensor
    tensor = gatekeeper.preprocess_image(raw_bytes)

    # 3. Model Inference via ONNX Backend
    top_class_id, top_prob = onnx_backend.predict(tensor)

    # 4. Tier-1 Semantic OOD Filter
    if not gatekeeper.is_in_distribution(top_class_id, top_prob):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"OOD_INPUT_REJECTED: Class ID {top_class_id} is outside the domestic pet distribution.",
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    profiler.record(latency_ms)

    return InferenceResponse(
        modality="image_base64",
        prediction=f"CLASS_{top_class_id}",
        confidence=round(top_prob, 4),
        cached=False,
        latency_ms=round(latency_ms, 3),
    )