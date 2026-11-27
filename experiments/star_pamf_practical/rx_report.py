"""Report the requested cross-receiver-only verdict for the running screen."""
import argparse
import csv
from pathlib import Path
import time

from experiments.cric_dura_comparison.io import read, write
from .rx_screen import MARGIN_PP, METHODS, READOUTS, ROOT, SEED, SIX, TOLERANCE_PP, VARIANTS, VIEWS


def summarize(root):
    root = Path(root)
    with (root / "paired_comparison.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["population"] == "cross_rx"]
    rows_by_key = {(r["candidate"], r["readout"], r["view"]): r for r in rows}
    expected = {(candidate, route, view) for candidate in VARIANTS
                for route in READOUTS for view in (*VIEWS, SIX)}
    if len(rows) != len(expected) or rows_by_key.keys() != expected:
        raise ValueError("Incomplete cross-receiver matched cells")

    routes = {}
    for route in READOUTS:
        practical = [rows_by_key[c, route, SIX] for c in VARIANTS]
        previous = {m: float(practical[0][m]) for m in METHODS.values()}
        if any(float(r[m]) != value for r in practical for m, value in previous.items()):
            raise ValueError("Cross-receiver baselines differ between candidate rows")
        incumbent = max(previous, key=previous.__getitem__)
        candidates = {}
        for r in practical:
            candidate = r["candidate"]
            clean = rows_by_key[candidate, route, "clean"]
            acc = float(r["accuracy_pct"])
            clean_delta = float(clean["gain_vs_native_pp"])
            margin = acc - previous[incumbent]
            candidates[candidate] = dict(practical_accuracy_pct=acc,
                                         practical_macro_f1_pct=float(r["macro_f1_pct"]),
                                         gain_vs_original_star_pp=float(r["gain_vs_native_pp"]),
                                         gain_vs_best_baseline_pp=margin, clean_accuracy_pct=float(clean["accuracy_pct"]),
                                         clean_gain_vs_original_star_pp=clean_delta)
            if route == "raw":
                candidates[candidate]["passes_screen"] = margin >= MARGIN_PP and clean_delta >= -TOLERANCE_PP
        routes[route] = dict(baseline_ranked=dict(sorted(previous.items(), key=lambda x: -x[1])),
                             best_baseline=incumbent, candidates=candidates)

    out = root / "cross_rx_report"
    out.mkdir(exist_ok=False)
    with (out / "comparison.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(status="ANALYZED", model_seed=SEED, population="cross_rx", primary_route="raw",
                  primary_view=SIX, minimum_gain_vs_best_baseline_pp=MARGIN_PP,
                  maximum_clean_drop_vs_original_star_pp=TOLERANCE_PP, routes=routes)
    write(out / "complete.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(ROOT))
    args = p.parse_args()
    root = args.root
    while not (root / "complete.json").is_file():
        state = read(root / "pipeline_state.json")
        if state["status"] == "FAILED":
            raise RuntimeError("Original run failed before all truth-last scores were ready")
        time.sleep(30)
    if read(root / "complete.json")["truth_last"] is not True:
        raise ValueError("Cross-receiver reporting requires the original independent scores")
    print(summarize(root), flush=True)


if __name__ == "__main__":
    main()
