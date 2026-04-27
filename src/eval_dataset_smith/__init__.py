"""eval-dataset-smith -- generate balanced AI eval fixtures.

Public surface (mirrors the JS sibling, plus a higher-level Pythonic API):

    from eval_dataset_smith import (
        forge_dataset,
        build_eval_dataset,
        stratified_split,
        Dataset,
        EvalCase,
    )

* ``forge_dataset(sources, balance_keys=...)`` -- top-level Pythonic API. Takes
  raw source items (bugs, docs, examples, policies) and returns a balanced
  ``Dataset`` of ``EvalCase`` records.
* ``build_eval_dataset(items, max_per_type=20)`` -- direct port of the JS
  ``buildEvalDataset`` function; returns a list of dicts.
* ``stratified_split(items, ratio=0.8)`` -- split a dataset into train/test
  groups while keeping per-tag balance.

Zero runtime dependencies; pure stdlib. Heuristic-only -- no LLM calls.
"""

from .dataset import (
    Dataset,
    EvalCase,
    build_eval_dataset,
    forge_dataset,
    stratified_split,
)

__version__ = "0.1.0"
VERSION = __version__

__all__ = [
    "VERSION",
    "Dataset",
    "EvalCase",
    "build_eval_dataset",
    "forge_dataset",
    "stratified_split",
]
