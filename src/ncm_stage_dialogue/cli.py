"""Command-line entry points for corpus construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ncm_stage_dialogue.corpus import CorpusError, build_corpus


def main() -> int:
    """Build the configured corpus and print deterministic summary counts."""
    parser = argparse.ArgumentParser(
        description="Build a hash-pinned public-domain stage-play dialogue corpus."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("sources/plays.json"),
        help="JSON source manifest (default: sources/plays.json)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Generated corpus root (default: data)",
    )
    args = parser.parse_args()
    try:
        counts = build_corpus(args.manifest, args.data_root)
    except (CorpusError, OSError, ValueError) as error:
        parser.exit(1, f"build-corpus: {error}\n")
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
