"""Song and stem session ownership for reusable audio analysis state."""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


ANALYSIS_VERSION = 1


@dataclass(frozen=True)
class StemCacheKey:
    path: str
    size: int
    modified_ns: int
    bpm: float
    beats_per_bar: int
    analysis_version: int = ANALYSIS_VERSION


@dataclass
class StemSession:
    key: StemCacheKey
    markers: Any


class SongSession:
    """Own analyzed stems and the currently active stem for one song."""

    def __init__(self, max_cached_stems: int = 4) -> None:
        self._max_cached_stems = max(1, int(max_cached_stems))
        self._stems: OrderedDict[StemCacheKey, StemSession] = OrderedDict()
        self._active_key: StemCacheKey | None = None
        self._generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def active(self) -> StemSession | None:
        if self._active_key is None:
            return None
        return self._stems.get(self._active_key)

    @staticmethod
    def cache_key(path: str, bpm: float, beats_per_bar: int) -> StemCacheKey:
        normalized = os.path.normcase(os.path.abspath(path))
        stat = os.stat(normalized)
        return StemCacheKey(
            path=normalized,
            size=int(stat.st_size),
            modified_ns=int(stat.st_mtime_ns),
            bpm=round(float(bpm), 6),
            beats_per_bar=int(beats_per_bar),
        )

    def activate_cached(
        self,
        path: str,
        bpm: float,
        beats_per_bar: int,
    ) -> StemSession | None:
        key = self.cache_key(path, bpm, beats_per_bar)
        self._discard_stale_path_entries(key)
        session = self._stems.get(key)
        if session is None:
            return None
        self._stems.move_to_end(key)
        self._activate(key)
        return session

    def store(
        self,
        markers: Any,
        bpm: float,
        beats_per_bar: int,
    ) -> StemSession:
        key = self.cache_key(markers.wav_path, bpm, beats_per_bar)
        self._discard_stale_path_entries(key)
        session = StemSession(key=key, markers=markers)
        self._stems[key] = session
        self._stems.move_to_end(key)
        self._activate(key)
        self._evict_inactive_stems()
        return session

    def clear(self) -> None:
        self._stems.clear()
        self._active_key = None
        self._generation += 1

    def _activate(self, key: StemCacheKey) -> None:
        if self._active_key != key:
            self._generation += 1
        self._active_key = key

    def _discard_stale_path_entries(self, current_key: StemCacheKey) -> None:
        stale = [
            key for key in self._stems
            if key.path == current_key.path and key != current_key
        ]
        for key in stale:
            self._stems.pop(key, None)
            if self._active_key == key:
                self._active_key = None

    def _evict_inactive_stems(self) -> None:
        while len(self._stems) > self._max_cached_stems:
            oldest = next(iter(self._stems))
            if oldest == self._active_key:
                self._stems.move_to_end(oldest)
                continue
            self._stems.pop(oldest)