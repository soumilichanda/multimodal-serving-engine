"""
app/core/cache.py
Thread-safe O(1) Least Recently Used (LRU) Inference Cache.
"""

from collections import OrderedDict
from threading import Lock
from typing import Any


class ThreadSafeLRUCache:
    """
    In-memory LRU cache with threading locks to prevent race conditions
    under concurrent FastAPI asynchronous request workloads.
    """

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self.cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self.lock = Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[Any, ...]) -> Any | None:
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: tuple[Any, ...], value: Any) -> None:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def stats(self) -> dict:
        with self.lock:
            total = self.hits + self.misses
            hit_ratio = (self.hits / total) if total > 0 else 0.0
            return {
                "capacity": self.capacity,
                "size": len(self.cache),
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": round(hit_ratio, 3),
            }