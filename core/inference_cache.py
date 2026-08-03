"""Versioned per-unit cache for generated model predictions."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from core.learning_runtime import LearningDomain, RuntimeToken


@dataclass(frozen=True)
class InferenceUnitKey:
    token: RuntimeToken
    domain: LearningDomain
    unit: Hashable


class InferenceCache:
    """Store generated results without making them authoritative user state."""

    def __init__(self) -> None:
        self._entries: dict[InferenceUnitKey, Any] = {}

    def merge(
        self,
        result_token: RuntimeToken,
        current_token: RuntimeToken,
        domain: LearningDomain,
        values: Mapping[Hashable, Any],
    ) -> bool:
        """Merge a worker result only when all generation fields are current."""
        if result_token != current_token:
            return False
        for unit, value in values.items():
            self._entries[InferenceUnitKey(result_token, domain, unit)] = value
        return True

    def get(
        self,
        token: RuntimeToken,
        domain: LearningDomain,
        unit: Hashable,
    ) -> Any | None:
        return self._entries.get(InferenceUnitKey(token, domain, unit))

    def values(
        self,
        token: RuntimeToken,
        domain: LearningDomain,
        units: Iterable[Hashable],
    ) -> dict[Hashable, Any]:
        found = {}
        for unit in units:
            value = self.get(token, domain, unit)
            if value is not None:
                found[unit] = value
        return found

    def missing(
        self,
        token: RuntimeToken,
        domain: LearningDomain,
        units: Iterable[Hashable],
    ) -> list[Hashable]:
        return [
            unit for unit in units
            if InferenceUnitKey(token, domain, unit) not in self._entries
        ]

    def completion(
        self,
        token: RuntimeToken,
        domain: LearningDomain,
        units: Iterable[Hashable],
    ) -> tuple[int, int]:
        requested = list(units)
        missing_count = len(self.missing(token, domain, requested))
        return len(requested) - missing_count, len(requested)

    def discard_other_sessions(self, session: int) -> None:
        self._entries = {
            key: value for key, value in self._entries.items()
            if key.token.session == int(session)
        }
