"""A disk cache around an OCR backend.

Experiments that vary the tracker, the plate format or the aggregation rule
all read the same plate crops. Caching by crop content means each crop is
read once per backend, however many configurations are replayed over it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import TracebackType

import numpy as np

from alpr.types import PlateRead


class CachedOCR:
    """Wraps an OCR backend and remembers its answers on disk.

    Entries are keyed by a hash of the crop's pixels, so a hit is exactly the
    answer the backend gave for that image. Use one cache file per backend and
    preprocessing combination.

    Args:
        backend: The backend to wrap.
        path: JSON file holding the cache. Created on :meth:`save`.
    """

    def __init__(self, backend, path: str | Path) -> None:
        self._backend = backend
        self._path = Path(path)
        self.name = backend.name
        self.preprocess_name = getattr(backend, "preprocess_name", "none")
        self.hits = 0
        self.misses = 0
        self._entries: dict[str, list | None] = {}
        if self._path.exists():
            self._entries = json.loads(self._path.read_text())

    @staticmethod
    def _key(crop: np.ndarray) -> str:
        digest = hashlib.sha1(np.ascontiguousarray(crop).tobytes(), usedforsecurity=False)
        digest.update(str(crop.shape).encode())
        return digest.hexdigest()

    def read(self, crop: np.ndarray) -> PlateRead | None:
        """Return the cached read for this crop, computing it on a miss."""
        key = self._key(crop)
        if key in self._entries:
            self.hits += 1
            entry = self._entries[key]
            return None if entry is None else PlateRead(entry[0], entry[1])
        self.misses += 1
        result = self._backend.read(crop)
        self._entries[key] = None if result is None else [result.text, result.confidence]
        return result

    def save(self) -> None:
        """Write the cache to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._entries))

    def __enter__(self) -> CachedOCR:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.save()
