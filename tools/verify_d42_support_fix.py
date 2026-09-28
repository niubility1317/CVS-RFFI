"""Replay captured inner-support fits; accepts no IQ, query or truth artifact."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import numpy as np
from cvsrffi.stage2_d42_compact_runtime import d42_runtime_for_feature_dim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, action="append", type=Path)
    args = parser.parse_args()
    for path in args.artifact:
        with np.load(path, allow_pickle=False) as data:
            x, y = data["transformed"], data["targets"]
            runtime = d42_runtime_for_feature_dim(x.shape[1])
            c, b, audit = runtime._fit_equal_prior_lda(x, y, int(data["class_count"]), int(data["k_shot"]))
            scores = x.astype(np.float64) @ c.astype(np.float64).T + b.astype(np.float64)
            reference = x.astype(np.float64) @ data["sklearn_coefficients"].T + data["sklearn_intercept"]
            if not np.array_equal(scores.argmax(1), reference.argmax(1)):
                raise AssertionError("Captured support predictions changed")
        print(json.dumps(dict(artifact=str(path), status="PASS", query_access=False, audit=audit)), flush=True)


if __name__ == "__main__":
    main()
