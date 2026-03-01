"""
Model variants system: provider-specific reasoning effort levels.

Allows users to define named model variants with different reasoning effort
levels (low/medium/high) that map to provider-specific parameters.

Usage in config (ollamacode.yaml):
  variants:
    fast:
      model: gpt-4o-mini
      reasoning_effort: low
    balanced:
      model: gpt-4o
      reasoning_effort: medium
    deep:
      model: o1
      reasoning_effort: high
  default_variant: balanced

Switch via /variant command or OLLAMACODE_VARIANT env var.
"""

from __future__ import annotations

import enum
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class ReasoningEffort(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ModelVariant:
    """A named model configuration with reasoning effort level."""

    name: str
    model_id: str
    reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM
    provider: str | None = None  # Override provider for this variant
    extra_params: dict[str, Any] | None = None  # Provider-specific params


# Provider-specific parameter mapping for reasoning effort
_EFFORT_PARAMS: dict[str, dict[ReasoningEffort, dict[str, Any]]] = {
    "openai": {
        ReasoningEffort.LOW: {"temperature": 0.3, "max_tokens": 2048},
        ReasoningEffort.MEDIUM: {"temperature": 0.7, "max_tokens": 4096},
        ReasoningEffort.HIGH: {"temperature": 1.0, "max_tokens": 8192},
    },
    "anthropic": {
        ReasoningEffort.LOW: {"max_tokens": 2048},
        ReasoningEffort.MEDIUM: {"max_tokens": 4096},
        ReasoningEffort.HIGH: {"max_tokens": 8192},
    },
    "ollama": {
        ReasoningEffort.LOW: {"num_predict": 2048, "temperature": 0.3},
        ReasoningEffort.MEDIUM: {"num_predict": 4096, "temperature": 0.7},
        ReasoningEffort.HIGH: {"num_predict": 8192, "temperature": 1.0},
    },
}


class VariantManager:
    """Manages model variant selection and parameter resolution."""

    def __init__(self) -> None:
        self._variants: dict[str, ModelVariant] = {}
        self._current: str | None = None

    def load_from_config(self, config: dict[str, Any]) -> int:
        """Load variants from config dict.

        Expects config keys: 'variants' (dict of name -> variant config),
        'default_variant' (str).
        Returns number of variants loaded.
        """
        raw = config.get("variants")
        if not isinstance(raw, dict):
            return 0

        loaded = 0
        for name, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            name = str(name).strip().lower()
            model_id = (entry.get("model") or entry.get("model_id") or "").strip()
            if not name or not model_id:
                continue
            effort_str = (entry.get("reasoning_effort") or "medium").strip().lower()
            try:
                effort = ReasoningEffort(effort_str)
            except ValueError:
                effort = ReasoningEffort.MEDIUM

            self._variants[name] = ModelVariant(
                name=name,
                model_id=model_id,
                reasoning_effort=effort,
                provider=(entry.get("provider") or "").strip() or None,
                extra_params=entry.get("extra_params"),
            )
            loaded += 1

        # Set default variant
        default = (
            os.environ.get("OLLAMACODE_VARIANT", "").strip()
            or (config.get("default_variant") or "").strip()
        )
        if default and default in self._variants:
            self._current = default

        logger.debug("Loaded %d model variants", loaded)
        return loaded

    def select(self, name: str) -> ModelVariant | None:
        """Select a variant by name. Returns the variant or None if not found."""
        name = name.strip().lower()
        variant = self._variants.get(name)
        if variant is not None:
            self._current = name
            logger.debug("Selected variant: %s (%s)", name, variant.model_id)
        return variant

    @property
    def current(self) -> ModelVariant | None:
        """Return the currently selected variant, or None."""
        if self._current is None:
            return None
        return self._variants.get(self._current)

    def get_provider_params(self, provider_name: str) -> dict[str, Any]:
        """Get provider-specific parameters for the current variant's reasoning effort.

        Returns an empty dict if no variant is selected or no mapping exists.
        """
        variant = self.current
        if variant is None:
            return {}

        # Start with effort-based params
        provider_map = _EFFORT_PARAMS.get(provider_name.lower(), {})
        params = dict(provider_map.get(variant.reasoning_effort, {}))

        # Merge any extra_params from the variant config
        if variant.extra_params:
            params.update(variant.extra_params)

        return params

    def list_variants(self) -> list[dict[str, str]]:
        """Return info about all available variants."""
        return [
            {
                "name": v.name,
                "model_id": v.model_id,
                "reasoning_effort": v.reasoning_effort.value,
                "provider": v.provider or "",
                "current": "yes" if v.name == self._current else "",
            }
            for v in sorted(self._variants.values(), key=lambda v: v.name)
        ]

    @property
    def variant_names(self) -> list[str]:
        """Return sorted list of variant names for autocomplete."""
        return sorted(self._variants.keys())
