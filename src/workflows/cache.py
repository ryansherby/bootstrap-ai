import threading
from typing import Any
from dataclasses import dataclass

@dataclass
class CacheResult():
    value: Any
    cache_hit: bool

class SharedCache:
    def __init__(self):
        self._cache = {}
        self._locks = {}
        self._locks_lock = threading.Lock()

    def _get_lock(self, key):
        with self._locks_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()

            return self._locks[key]

    def get(self, key):
        value = self._cache.get(key)
        if value is not None:
            return CacheResult(value=value, cache_hit=True)
        return CacheResult(value=None, cache_hit=False)
    def get_or_compute(self, key, compute_fn):
        lock = self._get_lock(key)

        with lock:
            # Check AFTER acquiring the key lock
            if key in self._cache:
                return CacheResult(value=self._cache[key], cache_hit=True)

            result = compute_fn()

            self._cache[key] = result

            return CacheResult(value=result, cache_hit=False)
        
    def clear_cache(self):
        with self._locks_lock:
            self._cache.clear()
            self._locks.clear()