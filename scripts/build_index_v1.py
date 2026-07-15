# C:\Users\18449\Desktop\researchguard_workspace\scripts\build_index_v1.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from researchguard.indexing.index_v1 import build_index, dry_run, validate_existing_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ResearchGuard embedding and persistent index v1.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "indexing_v1.yaml"))
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    input_root = Path(args.input_root) if args.input_root else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        if args.dry_run:
            result = dry_run(config_path, input_root_override=input_root, output_dir_override=output_dir)
        elif args.validate_only:
            result = validate_existing_index(config_path, output_dir_override=output_dir)
        else:
            result = build_index(
                config_path,
                input_root_override=input_root,
                output_dir_override=output_dir,
                force_rebuild=True if args.force_rebuild else None,
                incremental=True if args.incremental else None,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
