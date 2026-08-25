"""Truth-last scorer CLI for the slow/fast diagnostic matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.slow_fast_scorer import score_slow_fast_matrix  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-config", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--truth-map", required=True)
    parser.add_argument("--class-binding", default=None)
    args = parser.parse_args()
    matrix = json.loads(Path(args.matrix_config).read_text(encoding="utf-8-sig"))
    truth_map = json.loads(Path(args.truth_map).read_text(encoding="utf-8-sig"))
    summary = score_slow_fast_matrix(
        matrix,
        args.prediction_root,
        truth_map,
        class_binding_path=args.class_binding,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
