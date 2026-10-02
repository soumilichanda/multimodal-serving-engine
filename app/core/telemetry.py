"""
app/core/telemetry.py
Statistical Feature Drift Engine (Two-Sample KS-Test) and
Thread-Safe Latency Profiler for Real-Time Inference Observability.
"""

from collections import deque
from threading import Lock
import time
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.stats import ks_2samp


class LatencyProfiler:
    """
    Thread-safe latency tracker computing streaming rolling percentiles
    (p50, p90, p95, p99) and active request throughput (QPS).
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies: deque[float] = deque(maxlen=window_size)
        self.lock = Lock()
        self.total_requests: int = 0
        self.start_epoch: float = time.time()

    def record(self, latency_ms: float) -> None:
        """Records a single inference execution latency in milliseconds."""
        with self.lock:
            self.latencies.append(latency_ms)
            self.total_requests += 1

    def get_stats(self) -> Dict[str, Any]:
        """Calculates windowed percentiles and lifetime system throughput."""
        with self.lock:
            if not self.latencies:
                return {
                    "total_requests": 0,
                    "window_size": 0,
                    "avg_ms": 0.0,
                    "p50_ms": 0.0,
                    "p90_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                    "qps": 0.0,
                }

            arr = np.array(self.latencies)
            elapsed_time = max(time.time() - self.start_epoch, 1e-6)
            qps = self.total_requests / elapsed_time

            return {
                "total_requests": self.total_requests,
                "window_size": len(self.latencies),
                "avg_ms": round(float(np.mean(arr)), 3),
                "p50_ms": round(float(np.percentile(arr, 50)), 3),
                "p90_ms": round(float(np.percentile(arr, 90)), 3),
                "p95_ms": round(float(np.percentile(arr, 95)), 3),
                "p99_ms": round(float(np.percentile(arr, 99)), 3),
                "qps": round(qps, 2),
            }


class StatisticalDriftEngine:
    """
    Two-sample Kolmogorov-Smirnov (KS-Test) engine monitoring statistical drift
    between a pre-computed baseline (training distribution) and live production arrays.
    """

    def __init__(
        self,
        baseline_data: np.ndarray,
        window_size: int = 500,
        significance_level: float = 0.05,
    ):
        """
        :param baseline_data: 2D array of shape (N, D) representing the reference distribution.
        :param window_size: Buffer capacity for live feature observations.
        :param significance_level: Alpha threshold (standard: 0.05) to reject the null hypothesis.
        """
        if baseline_data.ndim == 1:
            baseline_data = baseline_data.reshape(-1, 1)

        self.baseline = baseline_data
        self.num_features = baseline_data.shape[1]
        self.window_size = window_size
        self.alpha = significance_level
        self.lock = Lock()

        # Allocate rolling buffers per feature dimension
        self.live_buffers: List[deque[float]] = [
            deque(maxlen=window_size) for _ in range(self.num_features)
        ]

    def record_observation(self, feature_vector: List[float]) -> None:
        """Appends an incoming production vector into the rolling inspection window."""
        with self.lock:
            if len(feature_vector) != self.num_features:
                raise ValueError(
                    f"Dimension mismatch: expected {self.num_features}, got {len(feature_vector)}"
                )
            for idx, val in enumerate(feature_vector):
                self.live_buffers[idx].append(val)

    def detect_drift(self) -> Dict[str, Any]:
        """
        Runs two-sample KS-tests across all feature dimensions:
        H0: The live production feature comes from the same distribution as baseline.
        H1: The live production feature has drifted (diverged).
        """
        with self.lock:
            current_samples = len(self.live_buffers[0])
            # Require minimum sample support for valid statistical power
            if current_samples < min(30, self.window_size):
                return {
                    "status": "INSUFFICIENT_DATA",
                    "samples_observed": current_samples,
                    "min_required": min(30, self.window_size),
                    "drift_detected": False,
                    "feature_reports": [],
                }

            drift_flag = False
            reports = []

            for idx in range(self.num_features):
                base_col = self.baseline[:, idx]
                live_col = np.array(self.live_buffers[idx])

                # ks_2samp calculates statistic D and empirical p-value
                statistic, p_value = ks_2samp(base_col, live_col)
                feature_drift = bool(p_value < self.alpha)

                if feature_drift:
                    drift_flag = True

                reports.append(
                    {
                        "feature_index": idx,
                        "ks_statistic": round(float(statistic), 4),
                        "p_value": round(float(p_value), 6),
                        "drift": feature_drift,
                    }
                )

            return {
                "status": "DRIFT_DETECTED" if drift_flag else "STABLE",
                "samples_observed": current_samples,
                "drift_detected": drift_flag,
                "feature_reports": reports,
            }