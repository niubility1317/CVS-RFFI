"""Recompute the comments' paired statistics from retained, closed artifacts."""
import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np

from experiments.cric_dura_comparison.evaluation import VIEWS, handle, metrics
from experiments.cric_dura_comparison.io import read
from .register import CONTEXT, PROJECT, PAPER_RUN

SEEDS = (2026092701, 2026092702, 2026092703)
TASKS = ("cross_rx", "cross_both")
SCENES = (*VIEWS, "six_practical_mean")


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def ssl_deltas(root):
    with (root / "analysis" / "per_seed.csv").open(encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    outcomes = {}
    for r in records:
        if (r["family"] == "factorial" and r["method"] in ("pl", "ssl")
                and r["readout"] == "raw" and r["test"] in TASKS):
            key = (r["data_setting"], r["test"], r["view"], r["metric"], int(r["model_seed"]), r["method"])
            if key in outcomes:
                raise ValueError("Duplicate historical cell")
            outcomes[key] = float(r["percent"])
    rows = []
    for setting in sorted({k[0] for k in outcomes}):
        for task in TASKS:
            for view in SCENES:
                for metric in ("accuracy", "macro_f1"):
                    for seed in SEEDS:
                        key = (setting, task, view, metric, seed)
                        base, ssl = [outcomes[(*key, m)] for m in ("pl", "ssl")]
                        rows.append(dict(setting=setting, test=task, view=view, metric=metric,
                                         seed=seed, ordinary_pl_pct=base, full_ssl_pct=ssl,
                                         change_pp=ssl-base))
    return rows


def load_decisions(row, task, expected, classes):
    base = Path(row["predictions"])
    done = read(base / "complete.json")
    if (done["status"] != "ALL_PREDICTIONS_COMPLETE" or done["truth_read"]
            or done["query_fit"] or task not in done["tests"] or done["views"] != list(VIEWS)):
        raise ValueError("Only complete, frozen rows may be inspected")
    scored = read(Path(row["scores"]) / "complete.json")
    if scored["status"] != "SCORED_COMPLETE":
        raise ValueError("Original independent truth-last score missing")
    keys = {f"{route}__{view}" for route in ("raw", "DURA-LU") for view in VIEWS}
    with np.load(base / task / "predictions.npz", allow_pickle=False) as archive:
        ids = archive["ids"]
        if not np.array_equal(ids, expected) or not keys.issubset(archive.files):
            raise ValueError("Frozen row views or physical population differ")
        result = {k: archive[k] for k in keys}
    if any(x.shape != ids.shape or x.dtype.kind not in "iu" or x.min() < 0 or x.max() >= classes
           for x in result.values()):
        raise ValueError("Invalid frozen decisions")
    return result


def matched_dura(root):
    ctx = read(CONTEXT)
    a, b = [ctx["handles"][k] for k in ("F0", "m6-natural")]
    classes = a["classes"]
    if classes != b["classes"]:
        raise ValueError("Historical class map differs")
    index = read(root / "sources.json")
    cells = []
    # Validate and retain all declared predictions for every seed before truth.
    for task in TASKS:
        ha, hb = a["tests"][task], b["tests"][task]
        ma, ia, _ = handle(ha["views_root"], classes)
        mb, ib, _ = handle(hb["views_root"], classes)
        if not set(mb["target_rxs"]) < set(ma["target_rxs"]):
            raise ValueError("Six target RX are not a strict Main subset")
        offsets = {i: j for j, i in enumerate(ia)}
        subset = np.asarray([offsets[i] for i in ib])
        for seed in SEEDS:
            ra = index[f"main_split-factorial-ssl-full-permuted-s{seed}"]
            rb = index[f"source_rx_6-factorial-ssl-full-permuted-s{seed}"]
            va = load_decisions(ra, task, ia, len(classes))
            vb = load_decisions(rb, task, ib, len(classes))
            cells.append((task, seed, hb["truth"], ib, {k: x[subset] for k, x in va.items()}, vb))

    rows = []
    truths = {}
    for task, seed, path, ids, five, six in cells:
        if task not in truths:
            truth = read(path)
            truths[task] = (np.asarray([truth[i]["label"] for i in ids]),
                            np.asarray([truth[i]["receiver"] for i in ids]))
        y, rx = truths[task]
        strata = [("overall", "all", np.ones(len(y), dtype=bool))]
        strata.extend(("transmitter", name, y == i) for i, name in enumerate(classes))
        strata.extend(("receiver", str(i), rx == i) for i in np.unique(rx))
        for view in VIEWS:
            a_raw, b_raw = five[f"raw__{view}"], six[f"raw__{view}"]
            a_dura, b_dura = five[f"DURA-LU__{view}"], six[f"DURA-LU__{view}"]
            for dimension, name, use in strata:
                labels = y[use]
                aa, ab, ba, bb = [x[use] == labels for x in (a_raw,a_dura,b_raw,b_dura)]
                rows.append(dict(test=task, seed=seed, view=view, dimension=dimension, stratum=name,
                    n=len(labels), raw5_pct=100*mean(aa.tolist()), raw6_pct=100*mean(ba.tolist()),
                    dura5_pct=100*mean(ab.tolist()), dura6_pct=100*mean(bb.tolist()),
                    raw6_minus5_pp=100*(ba.mean()-aa.mean()), dura6_minus5_pp=100*(bb.mean()-ab.mean()),
                    dura5_repairs=int((~aa&ab).sum()), dura5_damage=int((aa&~ab).sum()),
                    dura6_repairs=int((~ba&bb).sum()), dura6_damage=int((ba&~bb).sum()),
                    dura5_net=int(ab.sum()-aa.sum()), dura6_net=int(bb.sum()-ba.sum()),
                    dura6_vs5_repairs=int((~ab&bb).sum()), dura6_vs5_damage=int((ab&~bb).sum())))
    return rows


def run(output):
    root = Path(PROJECT) / "runs" / PAPER_RUN
    output.mkdir(parents=True, exist_ok=False)
    deltas = ssl_deltas(root)
    write_csv(output / "ssl_deltas.csv", deltas)
    matched = matched_dura(root)
    write_csv(output / "dura_five_six_rx.csv", matched)
    ten = [r for r in deltas if r["setting"]=="10-class setting" and r["test"]=="cross_both"
           and r["view"]=="clean" and r["metric"]=="accuracy"]
    result = {"ten_class_double_cross": ten, "ssl_rows": len(deltas), "dura_rows": len(matched)}
    result["dura_six_scene"] = []
    for task in TASKS:
        subset = [r for r in matched if r["test"]==task and r["dimension"]=="overall" and r["view"]!="clean"]
        result["dura_six_scene"].append(dict(test=task, **{
            k: mean(r[k] for r in subset) for k in ("raw5_pct","raw6_pct","dura5_pct","dura6_pct",
                                                      "raw6_minus5_pp","dura6_minus5_pp")}))
    with (output / "summary.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
