"""
tests/test_batcher.py
Asyncio unit tests verifying dynamic batching invariants and concurrent execution.
"""

import asyncio

import numpy as np
import pytest

from app.core.batcher import DynamicBatcher


def mock_vectorized_model(batch: np.ndarray):
    # Sum along features and return prediction dicts
    scores = np.sum(batch, axis=1)
    return [{"prediction": int(s > 0), "score": float(s)} for s in scores]


@pytest.mark.asyncio
async def test_batcher_size_flush():
    batcher = DynamicBatcher(
        forward_fn=mock_vectorized_model,
        max_batch_size=4,
        max_delay_ms=50.0,
    )
    await batcher.start()

    # Enqueue 4 items concurrently to trigger size-based flush
    inputs = [[1.0, 2.0], [-1.0, -2.0], [0.5, 0.5], [2.0, -1.0]]
    results = await asyncio.gather(*(batcher.enqueue(x) for x in inputs))

    assert len(results) == 4
    assert results[0]["prediction"] == 1
    assert results[1]["prediction"] == 0

    await batcher.stop()


@pytest.mark.asyncio
async def test_batcher_timeout_flush():
    batcher = DynamicBatcher(
        forward_fn=mock_vectorized_model,
        max_batch_size=10,  # High batch limit
        max_delay_ms=15.0,  # Fast timeout
    )
    await batcher.start()

    # Enqueue only 1 item; must flush on timeout rather than hanging
    result = await batcher.enqueue([3.0, 4.0])

    assert result["prediction"] == 1
    assert result["score"] == 7.0

    await batcher.stop()