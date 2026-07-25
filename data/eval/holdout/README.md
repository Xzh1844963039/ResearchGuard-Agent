# Holdout Corpus v1

This split is intentionally separate from the five-paper development corpus and
from all earlier retrieval benchmark annotations.

- `manifest.jsonl` lists 10 public arXiv papers and their download URLs.
- `blind_questions.jsonl` contains 25 questions with routing and document-level
  evaluation labels only. It contains no reference answers and no chunk IDs.
- PDFs, parser outputs, indexes, caches, and run reports are generated under
  ignored project directories and are not committed.

Run `scripts/prepare_holdout_v1.py` to download and parse the corpus, then build
the dedicated index and Chroma collection before running
`scripts/validate_holdout_v1.py`.
