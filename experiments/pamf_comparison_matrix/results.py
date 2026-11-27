"""Emit seven-method tables from independently scored frozen decisions."""
from collections import defaultdict
import csv
from pathlib import Path
import statistics
import time

import numpy as np

from experiments.cric_dura_comparison.evaluation import DATES
from experiments.cric_dura_comparison.io import read, write
from experiments.practical_full_2x2.plan import TASKS, VIEWS
from .plan import METHODS, NATURAL, PROJECT, SHEN, V, published_scores, settings


def table(path, rows):
    if not rows:
        return
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pooled(confusion):
    cm = np.asarray(confusion, dtype=np.int64)
    divisor = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), divisor, out=np.zeros(len(cm)), where=divisor > 0)
    return dict(accuracy=float(cm.trace() / cm.sum()), macro_f1=float(f1.mean()), count=int(cm.sum()))


def summarize(root, *, main_only=False, main_raw_only=False, methods=METHODS,
              resolve_scores=published_scores, columns_file="seven_method_columns.csv"):
    root = Path(root)
    output = root / ("main_existing_raw" if main_raw_only else "main_snapshot" if main_only else "tables")
    settings_by_id = settings()
    rows = []
    missing = []
    common_six = {}
    for t in V:
        n6 = published_scores("m6-natural", settings_by_id["m6-natural"], "pamf", 2026092701)
        data = read(n6 / "scores" / "scores.json")["results"]
        common_six[t] = {x["stratum"] for x in data if x["test"] == t and x["dimension"] == "receiver"}
        if len(common_six[t]) != 6:
            raise ValueError("Six common target receivers required")

    def append(method, setting, seed, task, route, values):
        by_view = {x["view"]: x for x in values}
        if set(by_view) != set(VIEWS):
            raise ValueError(f"Missing seven-view scores for {setting}, {method}, {task}, {route}")
        six = {kind: statistics.mean(by_view[v][kind] for v in VIEWS[1:])
               for kind in ("accuracy", "macro_f1")}
        six["count"] = by_view["clean"]["count"]
        for view, metric in dict(by_view, six_practical_mean=six).items():
            rows.append(dict(setting=setting, seed=seed, population=task, route=route, view=view,
                             method=method, count=metric["count"], accuracy=100 * metric["accuracy"],
                             macro_f1=100 * metric["macro_f1"]))

    for setting, info in settings_by_id.items():
        if (main_only or main_raw_only) and setting != "F0":
            continue
        for seed in info["seeds"]:
            for method in methods:
                source = resolve_scores(setting, info, method, seed)
                if main_raw_only and method == "shen":
                    source = PROJECT / "runs" / SHEN / f"s{seed}"
                path = Path(source) / "scores" / "scores.json"
                if not path.is_file():
                    missing.append(dict(setting=setting, method=methods[method], seed=seed))
                    continue
                data = read(path)
                if not data["truth_last"] or data["model_seed"] != seed:
                    raise ValueError(f"Invalid frozen score {source}")
                records = data["results"]
                tasks = TASKS if info["kind"] == "paper" and not main_raw_only else V
                routes = ("raw", "DURA-LU") if info["kind"] == "paper" and not main_raw_only else ("raw",)
                for task in tasks:
                    endpoint = info.get("endpoint")
                    if endpoint is not None and task != ("cross_both" if endpoint == 0 else "cross_rx"):
                        continue
                    for route in routes:
                        items = [x for x in records if (x["test"], x["readout"]) == (task, route)]
                        dim = "overall" if endpoint is None else "date"
                        values = [x for x in items if x["dimension"] == dim and
                                  (endpoint is None or x["stratum"] == DATES[endpoint])]
                        name = task if endpoint is None else f"held_D{endpoint}"
                        append(method, setting, seed, name, route, values)
                        if setting == "F0" and task in V:
                            receiver_rows = [x for x in items if x["dimension"] == "receiver"
                                             and x["stratum"] in common_six[task]]
                            values = []
                            for view in VIEWS:
                                matching = [x for x in receiver_rows if x["view"] == view]
                                if len(matching) != 6:
                                    raise ValueError("Missing exact five-source common-receiver cells")
                                cm = np.sum([x["confusion"] for x in matching], axis=0)
                                values.append(dict(view=view, **pooled(cm)))
                            append(method, "m5-natural (common receivers)", seed, task, route, values)

    agg = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in ("setting", "population", "route", "view", "method"))
        agg[key].append(r)
    wide = {}
    stats = []
    for (setting, population, route, view, method), rs in agg.items():
        row = dict(setting=setting, population=population, route=route, view=view,
                   method=methods[method], n=len(rs), count=rs[0]["count"])
        cell = wide.setdefault((setting, population, route, view),
                        dict(setting=setting, population=population, route=route, view=view,
                             **{label: "" for label in methods.values()}))
        for metric in ("accuracy", "macro_f1"):
            values = [r[metric] for r in rs]
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else ""
        cell[methods[method]] = f"{row['accuracy_mean']:.4f}"
        stats.append(row)

    group_means = []
    for n in NATURAL:
        groups = {s for s, d in settings_by_id.items() if d.get("family") == "receiver_count"
                  and d.get("n_source_rx") == n}
        for method in methods:
            for task in V:
                for view in ("clean", "six_practical_mean"):
                    selected = [r for r in rows if r["setting"] in groups and
                                (r["method"], r["population"], r["view"]) == (method, task, view)]
                    if groups and len(selected) == len(groups):
                        group_means.append(dict(source_rx_count=n, receiver_combinations=len(groups),
                            population=task, view=view, method=methods[method],
                            accuracy=statistics.mean(r["accuracy"] for r in selected),
                            macro_f1=statistics.mean(r["macro_f1"] for r in selected)))
    output.mkdir(parents=True, exist_ok=True)
    table(output / "per_seed.csv", [dict(r, method=methods[r["method"]]) for r in rows])
    table(output / "summary.csv", stats)
    table(output / columns_file, list(wide.values()))
    table(output / "equal_size_receiver_groups.csv", group_means)
    table(output / "missing.csv", missing)
    result = dict(status="ANALYZED" if not missing else "PARTIAL", missing_rows=len(missing),
                  scored_cells=len(rows), settings="Main only" if main_only or main_raw_only else 35)
    write(output / "complete.json", result)
    return result


def follow(root):
    """Finish reporting for the already-started immutable release."""
    jobs = read(root / "jobs.json")
    scores = {job["name"]: root / "readouts" / job["name"] / "scores" / "complete.json"
              for job in jobs if job["stage"] == "score"}
    print("MAIN_EXISTING_RAW", summarize(root, main_raw_only=True), flush=True)
    while True:
        state = read(root / "pipeline_state.json")
        main = root / "main_snapshot" / "complete.json"
        if (not main.exists() and all(p.is_file() for k, p in scores.items() if k.startswith("F0-"))):
            print("MAIN_SNAPSHOT", summarize(root, main_only=True), flush=True)
        if state["status"] in {"FAILED", "PARTIAL", "ANALYZED"}:
            # Let the dispatcher finish its own last reporting attempt before filling the reserved output.
            final = root / "tables" / "complete.json"
            print("FINAL_TABLES", read(final) if final.exists() else summarize(root), flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--main-existing-raw", action="store_true")
    mode.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    if args.follow:
        follow(args.root)
    else:
        print(summarize(args.root, main_raw_only=True))
