# C:\Users\18449\Desktop\researchguard_workspace\scripts\build_chroma_v1.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(r"C:\Users\18449\Desktop\researchguard_workspace")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchguard.indexing.chroma_index import build_or_sync_chroma  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "chroma_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or incrementally synchronize Chroma backend v1.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--incremental", action="store_true", help="Explicitly request incremental synchronization.")
    parser.add_argument(
        "--confirm-large-delete",
        action="store_true",
        help="Confirm deletion when stale IDs exceed the configured safety ratio.",
    )
    return parser.parse_args()


def write_build_summary(path: Path, summary: dict[str, Any]) -> None:
    history: list[dict[str, Any]] = []
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        history = list(previous.get("run_history", []))
        prior_latest = previous.get("latest")
        if prior_latest and (not history or history[-1] != prior_latest):
            history.append(prior_latest)
    history.append(summary)
    payload = {"latest": summary, "run_history": history}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        summary, settings = build_or_sync_chroma(
            args.config,
            allow_large_delete=bool(args.confirm_large_delete),
        )
        summary["incremental_requested"] = bool(args.incremental)
        output_path = settings.validation_output_directory / "chroma_build_summary.json"
        write_build_summary(output_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
