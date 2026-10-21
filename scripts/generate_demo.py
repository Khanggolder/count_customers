from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from busreid.scenario import write_scenario  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "sample" / "generated" / "scenario.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = write_scenario(args.output, args.seed)
    print(f"Generated deterministic simulated scenario: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
