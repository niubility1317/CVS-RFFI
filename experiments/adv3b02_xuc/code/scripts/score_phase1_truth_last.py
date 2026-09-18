from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from cvsrffi.truth_last import score_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenarios", default="clean,leo_clear_weak,leo_low_elev_weak,leo_rain_weak")
    args = parser.parse_args()
    from cvsrffi.original_leo import validate_scenarios
    scenarios = validate_scenarios(args.scenarios.split(","))
    result = score_predictions(args.predictions, args.truth, output_path=args.output, scenarios=scenarios)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
