"""
app/core/batcher.py
Asynchronous Dynamic Request Batcher with Time-Window and Size Thresholds.
"""

import asyncio
import time
from typing import Any, Callable, List, Tuple
import numpy as np


class BatchItem:
    """Encapsulates an incoming payload vector and its completion future."""

    __slots__ = ("features", "future", "arrival_time")

    def __init__(self, features: List[float]):
        self.features = features
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()
        self.arrival_time: float = time.perf_counter()


class DynamicBatcher:
    """
    Asynchronously aggregates concurrent point inference requests into
    dense 2D matrices, dispatching inference either when max_batch_size is reached
    or when max_delay_ms expires.
    """

    def __init__(
        self,
        forward_fn: Callable[[np.ndarray], List[dict]],
        max_batch_size: int = 32,
        max_delay_ms: float = 10.0,
    ):
        self.forward_fn = forward_fn
        self.max_batch_size = max_batch_size
        self.max_delay_seconds = max_delay_ms / 1000.0

        self._queue: asyncio.Queue[BatchItem] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._is_running = False

    async def start(self) -> None:
        """Starts the background batch collection loop."""
        if not self._is_running:
            self._is_running = True
            self._worker_task = asyncio.create_task(self._process_loop())

    async def stop(self) -> None:
        """Flushes remaining queue items and halts the background worker."""
        self._is_running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def enqueue(self, features: List[float]) -> dict:
        """
        Pushes a single request payload onto the queue and awaits completion.
        Automatically starts the background worker if not already running.
        """
        if not self._is_running:
            await self.start()

        item = BatchItem(features)
        await self._queue.put(item)
        return await item.future

    async def _process_loop(self) -> None:
        """Background coroutine that gathers batches and dispatches execution."""
        while self._is_running:
            # 1. Block awaiting the first item of a batch
            try:
                first_item = await self._queue.get()
            except asyncio.CancelledError:
                break

            batch: List[BatchItem] = [first_item]
            start_window = time.perf_counter()

            # 2. Accumulate items until max_batch_size is reached or timeout expires
            while len(batch) < self.max_batch_size:
                elapsed = time.perf_counter() - start_window
                time_remaining = self.max_delay_seconds - elapsed

                if time_remaining <= 0:
                    break

                try:
                    item = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=time_remaining,
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            # 3. Vectorize and execute inference
            try:
                matrix = np.array([b.features for b in batch], dtype=np.float32)
                results = self.forward_fn(matrix)

                for item, res in zip(batch, results):
                    if not item.future.done():
                        item.future.set_result(res)
            except Exception as exc:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)
            finally:
                for _ in batch:
                    self._queue.task_done()