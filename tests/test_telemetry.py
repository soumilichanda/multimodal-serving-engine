"""
tests/test_telemetry.py
Unit tests verifying LatencyProfiler percentiles and StatisticalDriftEngine KS-tests.
"""

import numpy as np
import pytest

from app.core.telemetry import LatencyProfiler, StatisticalDriftEngine


def test_latency_profiler_percentiles():
    profiler = LatencyProfiler(window_size=100)
    for ms in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]:
        profiler.record(ms)

    stats = profiler.get_stats()
    assert stats["total_requests"] == 10
    assert stats["window_size"] == 10
    assert stats["p50_ms"] == pytest.approx(55.0, 0.5)
    assert stats["avg_ms"] == pytest.approx(55.0, 0.1)


def test_drift_engine_insufficient_samples():
    baseline = np.random.normal(0, 1, (100, 2))
    engine = StatisticalDriftEngine(baseline, window_size=50)

    # Ingest fewer than minimum samples (threshold is 30)
    for _ in range(10):
        engine.record_observation([0.1, -0.2])

    result = engine.detect_drift()
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["drift_detected"] is False


def test_drift_engine_stable_distribution():
    np.random.seed(42)
    baseline = np.random.normal(loc=0.0, scale=1.0, size=(300, 2))
    engine = StatisticalDriftEngine(baseline, window_size=100)

    # Feed samples from the identical distribution
    for _ in range(80):
        live_sample = np.random.normal(loc=0.0, scale=1.0, size=2).tolist()
        engine.record_observation(live_sample)

    result = engine.detect_drift()
    assert result["status"] == "STABLE"
    assert result["drift_detected"] is False


def test_drift_engine_detects_distribution_shift():
    np.random.seed(42)
    baseline = np.random.normal(loc=0.0, scale=1.0, size=(300, 2))
    engine = StatisticalDriftEngine(baseline, window_size=100)

    # Inject heavy distribution shift (mean shifted from 0.0 to 4.5)
    for _ in range(80):
        shifted_sample = np.random.normal(loc=4.5, scale=1.0, size=2).tolist()
        engine.record_observation(shifted_sample)

    result = engine.detect_drift()
    assert result["status"] == "DRIFT_DETECTED"
    assert result["drift_detected"] is True
    assert result["feature_reports"][0]["drift"] is True