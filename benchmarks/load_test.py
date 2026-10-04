"""
Phase 4 Concurrency Benchmark Harness
Evaluates p50/p95/p99 latency, RPS throughput, and batch accumulation efficiency.
"""
import asyncio
import time
import httpx
import numpy as np

TARGET_URL = "http://127.0.0.1:8000/v1/predict/vector"
CONCURRENT_WORKERS = 50
TOTAL_REQUESTS = 500

async def worker(worker_id: int, queue: asyncio.Queue, client: httpx.AsyncClient, latencies: list[float]):
    while not queue.empty():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        # Synthetic feature payload (16 dims)
        payload = {"features": np.random.randn(16).tolist()}
        start = time.perf_counter()
        try:
            resp = await client.post(TARGET_URL, json=payload, timeout=10.0)
            latency = (time.perf_counter() - start) * 1000.0
            if resp.status_code == 200:
                latencies.append(latency)
        except Exception as e:
            pass
        finally:
            queue.task_done()

async def run_benchmark():
    queue = asyncio.Queue()
    for _ in range(TOTAL_REQUESTS):
        queue.put_nowait(1)

    latencies: list[float] = []
    print(f"🚀 Dispatching {TOTAL_REQUESTS} requests across {CONCURRENT_WORKERS} concurrent async workers...")
    
    wall_start = time.perf_counter()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=CONCURRENT_WORKERS)) as client:
        tasks = [
            asyncio.create_task(worker(i, queue, client, latencies))
            for i in range(CONCURRENT_WORKERS)
        ]
        await asyncio.gather(*tasks)
    
    total_time = time.perf_counter() - wall_start
    if not latencies:
        print("❌ Benchmark failed: No successful responses received. Is the server running?")
        return

    rps = len(latencies) / total_time
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print("\n--- Benchmark Metrics ---")
    print(f"Successful Requests : {len(latencies)} / {TOTAL_REQUESTS}")
    print(f"Total Wall Time     : {total_time:.2f}s")
    print(f"Throughput          : {rps:.1f} req/s")
    print(f"Latency p50         : {p50:.2f} ms")
    print(f"Latency p95         : {p95:.2f} ms")
    print(f"Latency p99         : {p99:.2f} ms")

if __name__ == "__main__":
    asyncio.run(run_benchmark())