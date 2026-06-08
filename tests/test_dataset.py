"""Tests for ``eval_dataset_smith`` core APIs."""

from __future__ import annotations

import pytest

from eval_dataset_smith import (
    Dataset,
    EvalCase,
    build_eval_dataset,
    forge_dataset,
    stratified_split,
)


def _bug(i: int) -> dict:
    return {
        "type": "bug",
        "id": f"bug-{i}",
        "input": f"reproduce step {i}",
        "expected": "no crash",
        "difficulty": "easy",
    }


def _doc(i: int) -> dict:
    return {
        "type": "doc",
        "question": f"how does feature {i} work?",
        "answer": f"see chapter {i}",
        "difficulty": "med",
    }


def test_build_eval_dataset_normalizes_aliases():
    items = [_bug(1), _doc(1)]
    rows = build_eval_dataset(items)
    by_id = {r["id"]: r for r in rows}
    assert by_id["bug-1"]["input"] == "reproduce step 1"
    # doc-1 came from doc with no `id` -> auto-generated as "doc-1"
    assert by_id["doc-1"]["input"] == "how does feature 1 work?"
    assert by_id["doc-1"]["expected"] == "see chapter 1"


def test_build_eval_dataset_caps_per_type():
    items = [_bug(i) for i in range(50)]
    rows = build_eval_dataset(items, max_per_type=10)
    assert len(rows) == 10
    # All survivors are tagged "bug"
    assert all("bug" in r["tags"] for r in rows)


def test_build_eval_dataset_drops_incomplete_items():
    items = [
        {"type": "bug", "input": "x"},  # no expected -> drop
        {"type": "bug", "expected": "y"},  # no input -> drop
        _bug(7),
    ]
    rows = build_eval_dataset(items)
    assert len(rows) == 1
    assert rows[0]["id"] == "bug-7"


def test_build_eval_dataset_dedupes_tags():
    items = [{"type": "bug", "input": "i", "expected": "e", "tags": ["bug", "p0"]}]
    rows = build_eval_dataset(items)
    assert rows[0]["tags"] == ["bug", "p0"]


def test_forge_dataset_returns_typed_cases():
    items = [_bug(1), _bug(2), _doc(1)]
    ds = forge_dataset(items)
    assert isinstance(ds, Dataset)
    assert len(ds) == 3
    for c in ds.cases:
        assert isinstance(c, EvalCase)
        assert c.input
        assert c.expected
    # Tags carry the type
    types_in_tags = {c.tags[0] for c in ds.cases}
    assert {"bug", "doc"} <= types_in_tags


def test_forge_dataset_records_balance_for_default_keys():
    items = [_bug(1), _bug(2), _doc(1)]
    ds = forge_dataset(items)
    assert ds.balance["type"] == {"bug": 2, "doc": 1}
    # difficulty was set on each item too
    assert ds.balance["difficulty"]["easy"] == 2
    assert ds.balance["difficulty"]["med"] == 1


def test_forge_dataset_custom_balance_keys():
    items = [
        {"type": "bug", "severity": "p0", "input": "i", "expected": "e"},
        {"type": "bug", "severity": "p1", "input": "i", "expected": "e"},
        {"type": "bug", "severity": "p0", "input": "i", "expected": "e"},
    ]
    ds = forge_dataset(items, balance_keys=["severity"])
    assert ds.balance == {"severity": {"p0": 2, "p1": 1}}


def test_stratified_split_keeps_per_tag_balance():
    items = [
        {"tags": ["bug"]},
        {"tags": ["bug"]},
        {"tags": ["doc"]},
        {"tags": ["doc"]},
    ]
    out = stratified_split(items, ratio=0.5)
    # ratio=0.5 with ceil -> 1 train per group of 2
    assert len(out["train"]) == 2
    assert len(out["test"]) == 2


def test_stratified_split_handles_missing_tags():
    items = [{"x": 1}, {"x": 2}]
    out = stratified_split(items, ratio=0.5)
    assert len(out["train"]) + len(out["test"]) == 2


def test_stratified_split_ratio_validation():
    with pytest.raises(TypeError):
        stratified_split([], ratio=1.5)


def test_build_eval_dataset_validates_input_type():
    with pytest.raises(TypeError):
        build_eval_dataset({"not": "a list"})  # type: ignore[arg-type]


def test_build_eval_dataset_skips_non_dict_items():
    # Non-dict entries are tolerated (skipped), not crashed on.
    rows = build_eval_dataset(["nope", 42, None, _bug(3)])
    assert len(rows) == 1
    assert rows[0]["id"] == "bug-3"


def test_forge_dataset_skips_non_dict_items():
    # forge_dataset already guards non-dicts in its balance loop; the cases
    # path should be equally tolerant rather than raising AttributeError.
    ds = forge_dataset(["nope", _bug(4)])
    assert len(ds) == 1
    assert ds.cases[0].id == "bug-4"


def test_forge_dataset_validates_input_type():
    with pytest.raises(TypeError):
        forge_dataset("nope")  # type: ignore[arg-type]
