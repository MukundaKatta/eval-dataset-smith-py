# eval-dataset-smith-py

[![PyPI](https://img.shields.io/pypi/v/eval-dataset-smith-py.svg)](https://pypi.org/project/eval-dataset-smith-py/)
[![Python](https://img.shields.io/pypi/pyversions/eval-dataset-smith-py.svg)](https://pypi.org/project/eval-dataset-smith-py/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Generate balanced AI eval fixtures from your bugs, docs, examples, and policies.** Zero runtime dependencies.

Python port of [@mukundakatta/eval-dataset-smith](https://github.com/MukundaKatta/eval-dataset-smith). The JS sibling has the full design notes; this README sticks to the Python API.

## Install

```bash
pip install eval-dataset-smith-py
```

## Usage

```python
from eval_dataset_smith import forge_dataset, stratified_split

sources = [
    {"type": "bug",  "id": "B-1", "input": "repro: click X",         "expected": "no crash",        "difficulty": "easy"},
    {"type": "bug",  "id": "B-2", "input": "repro: open file Y",     "expected": "no crash",        "difficulty": "med"},
    {"type": "doc",                "question": "how does foo work?", "answer": "see chapter 3",     "difficulty": "easy"},
    {"type": "policy",             "input": "is PII allowed?",       "expected": "redact",          "difficulty": "hard"},
]

ds = forge_dataset(sources, balance_keys=["type", "difficulty"])

ds.cases       # list[EvalCase]   -- the eval fixtures
ds.balance     # {"type": {...}, "difficulty": {...}} -- audit input skew
len(ds)        # 4

# Per-tag stratified split (preserves type balance across train/test)
parts = stratified_split([c.__dict__ for c in ds.cases], ratio=0.8)
parts["train"], parts["test"]
```

## API

### `forge_dataset(sources, balance_keys=("type","difficulty"), max_per_type=20) -> Dataset`

Top-level Pythonic entry point. Returns a typed `Dataset` of `EvalCase` records plus a `balance` histogram you can use to audit input skew.

### `build_eval_dataset(items, max_per_type=20) -> list[dict]`

Direct port of the JS `buildEvalDataset`. Accepts the JS field-name aliases:

| Field      | Aliases                              |
|------------|--------------------------------------|
| `input`    | `input` / `question` / `prompt`      |
| `expected` | `expected` / `answer` / `acceptance` |
| `type`     | `type` (defaults to `"general"`)     |
| `tags`     | `tags: list[str]`                    |

### `stratified_split(items, ratio=0.8) -> {"train": [...], "test": [...]}`

Direct port of the JS `stratifiedSplit`. Splits by the first tag of each item, slicing each group at `ceil(len(group) * ratio)`.

## API differences from the JS sibling

* `forge_dataset` is a Python addition that returns typed dataclasses (`Dataset`, `EvalCase`).
* `build_eval_dataset` and `stratified_split` mirror the JS function names with `snake_case`.

See the JS sibling's [README](https://github.com/MukundaKatta/eval-dataset-smith) for the full design notes.
