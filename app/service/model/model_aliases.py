from __future__ import annotations

from typing import Dict, Tuple


DERIVED_MODEL_SUFFIXES: Tuple[str, ...] = (
    "-image-generation",
    "-non-thinking",
    "-search",
    "-image",
    "-chat",
)


class ModelAliasResolutionError(ValueError):
    """Raised when model alias resolution encounters an invalid mapping."""


def normalize_model_aliases(alias_map: Dict[str, str] | None) -> Dict[str, str]:
    """Normalize raw alias mappings by trimming keys and values."""
    normalized: Dict[str, str] = {}
    if not isinstance(alias_map, dict):
        return normalized

    for alias, target in alias_map.items():
        alias_name = str(alias).strip()
        target_name = str(target).strip()
        if alias_name and target_name:
            normalized[alias_name] = target_name
    return normalized


def split_model_suffixes(model: str) -> Tuple[str, Tuple[str, ...]]:
    """
    Split a model name into its base name and known derived suffixes.

    This supports chained suffixes such as `foo-search-non-thinking`.
    """
    value = (model or "").strip()
    if not value:
        return "", ()

    suffixes = []
    base_name = value
    while base_name:
        matched_suffix = next(
            (suffix for suffix in DERIVED_MODEL_SUFFIXES if base_name.endswith(suffix)),
            None,
        )
        if not matched_suffix:
            break
        base_name = base_name[: -len(matched_suffix)]
        suffixes.append(matched_suffix)

    suffixes.reverse()
    return base_name or value, tuple(suffixes)


def get_base_model_name(model: str) -> str:
    """Return the base model name without derived suffixes."""
    base_name, _ = split_model_suffixes(model)
    return base_name


def _get_next_alias_target(model: str, alias_map: Dict[str, str]) -> str | None:
    exact_target = alias_map.get(model)
    if exact_target and exact_target != model:
        return exact_target

    base_name, suffixes = split_model_suffixes(model)
    if not suffixes or not base_name:
        return None

    base_target = alias_map.get(base_name)
    if not base_target or base_target == base_name:
        return None

    suffix_string = "".join(suffixes)
    if base_target.endswith(suffix_string):
        return base_target
    return f"{base_target}{suffix_string}"


def resolve_model_alias(model: str, alias_map: Dict[str, str] | None) -> str:
    """
    Resolve a model alias to its final upstream model.

    Resolution order:
    1. Exact full-name alias match.
    2. Base-name alias match, then re-append known derived suffixes.
    3. Continue resolving chained aliases until a stable tail is reached.
    """
    value = (model or "").strip()
    if not value:
        return value

    normalized_aliases = normalize_model_aliases(alias_map)
    if not normalized_aliases:
        return value

    current = value
    visited = {current}

    while True:
        target = _get_next_alias_target(current, normalized_aliases)
        if not target or target == current:
            return current
        if target in visited:
            raise ModelAliasResolutionError(
                f"Model alias mapping contains a cycle: {value}"
            )
        visited.add(target)
        current = target
