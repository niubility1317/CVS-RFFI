from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
OLD = ("14-10", "14-7", "20-15", "20-19", "6-15", "8-20")
NEW = ("1-16", "1-18")


def _ids(data: np.lib.npyio.NpzFile) -> set[str]:
    return {
        "|".join(str(data[key][i]) for key in ("dataset_role", "tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids"))
        for i in range(len(data["tx_ids"]))
    }


def validate(root: Path) -> dict:
    rows = []
    id_sets = {}
    for scenario in SCENARIOS:
        path = root / f"{scenario}.npz"
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing feature cache: {path}")
        with np.load(path, allow_pickle=False) as data:
            roles = data["dataset_role"].astype(str)
            tx = data["tx_ids"].astype(str)
            rx = data["rx_ids"].astype(str)
            sat = data["sat_scenarios"].astype(str)
            target = np.isin(roles, ["target_old", "target_new"])
            if not np.all(sat[target] == scenario):
                raise ValueError(f"{scenario}: target rows contain wrong satellite scenario")
            if np.any(np.isin(roles, ["target_unknown", "proxy_unknown"])):
                raise ValueError(f"{scenario}: unknown/proxy rows are forbidden")
            counts = Counter(zip(roles.tolist(), tx.tolist(), rx.tolist()))
            shortages = []
            for receiver in RECEIVERS:
                for label in OLD:
                    if counts[("target_old", label, receiver)] < 40:
                        shortages.append(("target_old", label, receiver, counts[("target_old", label, receiver)]))
                for label in NEW:
                    if counts[("target_new", label, receiver)] < 40:
                        shortages.append(("target_new", label, receiver, counts[("target_new", label, receiver)]))
            if shortages:
                raise ValueError(f"{scenario}: insufficient maxK20+query20 coverage: {shortages[:8]}")
            id_sets[scenario] = _ids(data)
            rows.append({"scenario": scenario, "path": str(path), "bytes": path.stat().st_size,
                         "row_count": len(tx), "feature_dim": int(data["features"].shape[1]),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    base = id_sets[SCENARIOS[0]]
    if any(id_sets[scenario] != base for scenario in SCENARIOS[1:]):
        raise ValueError("sample ID sets differ across LEO scenarios")
    return {"status": "PASS", "scenario_rows": rows, "shared_sample_id_count": len(base)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = validate(args.root)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
