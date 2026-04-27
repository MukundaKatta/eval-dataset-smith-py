"""Core implementation of dataset forging and stratified splitting.

Mirrors the JS sibling's ``buildEvalDataset`` / ``stratifiedSplit`` semantics
and adds a top-level ``forge_dataset`` function that returns a typed
``Dataset`` containing ``EvalCase`` records.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------------------
# Public dataclasses


@dataclass(frozen=True)
class EvalCase:
    """One eval case in a forged dataset.

    Mirrors the ``{id, input, expected, tags, source}`` shape produced by the
    JS sibling. ``tags`` is a tuple so the case is hashable and copy-safe.
    """

    id: str
    input: Any
    expected: Any
    tags: tuple[str, ...] = ()
    source: str | None = None


@dataclass
class Dataset:
    """A forged eval dataset.

    ``cases`` is the flat list of ``EvalCase`` records. ``balance`` records the
    per-balance-key counts (e.g. ``{"type": {"bug": 5, "doc": 5}}``) so callers
    can audit how well the input was balanced.
    """

    cases: list[EvalCase] = field(default_factory=list)
    balance: dict[str, dict[str, int]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)


# ---------------------------------------------------------------------------
# Internal helpers


def _group_by(items: Iterable[dict[str, Any]], key: str, default: str) -> dict[str, list[dict[str, Any]]]:
    """Bucket ``items`` by ``key`` (falling back to ``default`` when absent)."""
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        bucket = str(item.get(key, default)) if isinstance(item, dict) else default
        out[bucket].append(item)
    return dict(out)


# ---------------------------------------------------------------------------
# JS-port functions


def build_eval_dataset(items: Sequence[dict[str, Any]], max_per_type: int = 20) -> list[dict[str, Any]]:
    """Build a flat list of eval-case dicts; direct port of JS ``buildEvalDataset``.

    Each input item may use any of these field-name aliases:

    * input: ``input`` / ``question`` / ``prompt``
    * expected: ``expected`` / ``answer`` / ``acceptance``
    * type: ``type`` (defaults to ``"general"``)
    * tags: ``tags`` (list[str]) -- merged with the type tag, deduped

    Items missing both an input and an expected are silently skipped (matching
    the JS sibling's ``filter(item => item.input && item.expected)``).
    """
    if not isinstance(items, (list, tuple)):
        raise TypeError("items must be a list or tuple of dicts")
    if max_per_type < 0:
        raise TypeError("max_per_type must be >= 0")

    grouped = _group_by(items, "type", "general")
    result: list[dict[str, Any]] = []
    for type_name, values in grouped.items():
        for index, item in enumerate(values[:max_per_type]):
            item_id = item.get("id") or f"{type_name}-{index + 1}"
            inp = item.get("input") or item.get("question") or item.get("prompt")
            exp = item.get("expected") or item.get("answer") or item.get("acceptance")
            extra_tags = item.get("tags") or []
            # Dedupe while preserving order: type first, then user tags.
            seen: set[str] = set()
            tags: list[str] = []
            for t in [type_name, *extra_tags]:
                if t in seen:
                    continue
                seen.add(t)
                tags.append(t)
            row = {
                "id": item_id,
                "input": inp,
                "expected": exp,
                "tags": tags,
                "source": item.get("source"),
            }
            if row["input"] and row["expected"]:
                result.append(row)
    return result


def stratified_split(items: Sequence[dict[str, Any]], ratio: float = 0.8) -> dict[str, list[dict[str, Any]]]:
    """Split ``items`` into train/test, preserving per-tag balance.

    Uses each item's first tag (or ``"general"`` if absent) as the strata key
    and slices each group at ``ceil(len(group) * ratio)``.
    """
    if not 0.0 <= ratio <= 1.0:
        raise TypeError("ratio must be in [0.0, 1.0]")

    import math

    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        tags = item.get("tags") or ["general"]
        first = tags[0] if tags else "general"
        by_tag[first].append(item)
    for group in by_tag.values():
        cut = math.ceil(len(group) * ratio)
        train.extend(group[:cut])
        test.extend(group[cut:])
    return {"train": train, "test": test}


# ---------------------------------------------------------------------------
# Top-level Pythonic API


def forge_dataset(
    sources: Sequence[dict[str, Any]],
    balance_keys: Sequence[str] | None = None,
    max_per_type: int = 20,
) -> Dataset:
    """Forge a balanced eval ``Dataset`` from raw source items.

    ``sources`` is any sequence of dicts. Each dict is interpreted as an eval
    candidate (see :func:`build_eval_dataset` for field aliases).
    ``balance_keys`` is the list of keys whose distribution should be tracked
    in ``Dataset.balance`` (defaults to ``("type", "difficulty")``).

    The returned ``Dataset`` contains a flat list of typed ``EvalCase``
    records; the ``balance`` field shows per-key counts (computed against the
    original sources, before deduping or capping) so callers can audit input
    skew.
    """
    if not isinstance(sources, (list, tuple)):
        raise TypeError("sources must be a list or tuple of dicts")

    keys: tuple[str, ...] = tuple(balance_keys) if balance_keys else ("type", "difficulty")

    rows = build_eval_dataset(sources, max_per_type=max_per_type)
    cases = [
        EvalCase(
            id=r["id"],
            input=r["input"],
            expected=r["expected"],
            tags=tuple(r["tags"]),
            source=r["source"],
        )
        for r in rows
    ]

    # Compute balance histograms across the *original* sources, not the
    # filtered/capped output -- callers want to see the raw input skew so they
    # can decide whether to add more examples of under-represented buckets.
    balance: dict[str, dict[str, int]] = {}
    for key in keys:
        counts: dict[str, int] = defaultdict(int)
        for item in sources:
            if not isinstance(item, dict):
                continue
            counts[str(item.get(key, "unknown"))] += 1
        balance[key] = dict(counts)

    return Dataset(cases=cases, balance=balance)
