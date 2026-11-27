"""Two preregistered source-only STAR receiver-generalization screens."""
import argparse
import csv
from pathlib import Path
from statistics import mean
import os
import subprocess
import time

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_comparison_matrix.plan import METHODS, RUN as COMPARISON
from experiments.practical_full_2x2.cache import atomic
from experiments.practical_full_2x2.plan import TASKS, VIEWS
from .runner import CONTEXT, GPUS, PROJECT, PYTHON, RUN as ORIGINAL, resources

RUN = "20261127-phase1-star-cross-rx-practical-s2026092701-m2-r01"
MODULE = "experiments.star_pamf_practical.rx_screen"
ROOT = PROJECT / "runs" / RUN
RELEASE = PROJECT / "releases" / (RUN + "-release")
COMMAND = [PYTHON, "-u", "-m", MODULE, "dispatch", "--config", str(RELEASE / "runtime.json")]
SEED = 2026092701
VARIANTS = {
    "cross-rx": dict(cross_rx_identity_weight=0.20, labeled_ce_floor=0.10),
    "cross-rx-label-anchor": dict(cross_rx_identity_weight=0.20, labeled_ce_floor=0.30),
}
SOURCE_PAIRS = PROJECT / "runs" / ORIGINAL / "source_pairs"
BASELINE = PROJECT / "runs" / COMPARISON / "tables" / "per_seed.csv"
MIN_FREE_MIB = 12000
RX_TASKS = ("cross_rx", "cross_both")
READOUTS = ("raw", "DURA-LU")
SIX = "six_practical_mean"
MARGIN_PP = 0.50
TOLERANCE_PP = 1.00


def row_id(variant):
    return f"{variant}-s{SEED}"


def prepare(runtime, root, context):
    h = context["handles"]["F0"]
    source = read(h["source_contract"])
    pairs = read(SOURCE_PAIRS / "complete.json")
    if any(pairs[key] for key in ("U_TX_truth_read", "validation_used", "target_read")):
        raise ValueError("The matched practical pairs must be source L/U only")
    g = context["caches"]["F0"]["LU"]["path"]
    for variant in VARIANTS:
        row = row_id(variant)
        src = root / "F" / row / "source"
        dest = root / "readouts" / row
        common = dict(run_id=RUN, row_id=row, model_seed=SEED, protocol_id="F0",
                      classes=source["classes"], source_days=source["source_days"],
                      source_contract=h["source_contract"], readouts=list(READOUTS), views=list(VIEWS))
        train = dict(run_id=RUN, row_id=row, model_seed=SEED, variant=variant,
                     source_cache=h["source_cache"], pairs=str(SOURCE_PAIRS),
                     source_output=str(src), code_commit=runtime["code_commit"])
        pred = dict(common, G_roles="LU", restored_cache=g, source_output=str(src),
                    frozen_source=dict(run_id=RUN, row_id=row), output_root=str(dest / "predictions"),
                    tests={k: {"views_root": h["tests"][k]["views_root"]} for k in TASKS})
        score = dict(common, tests={k: h["tests"][k] for k in TASKS},
                     prediction_output=str(dest / "predictions"), output_root=str(dest / "scores"))
        for stage, c in (("train", train), ("predict", pred), ("score", score)):
            write(root / "configs" / f"{stage}__{row}.json", c)


def smoke(root, context):
    import torch
    from experiments.pamf_comparison_matrix.runner import star_reader
    from experiments.practical_full_2x2.compat import tensor
    from .data import load_source
    from .training import forward

    h = context["handles"]["F0"]
    original_row = f"main-star-s{SEED}"
    folder = PROJECT / "runs" / ORIGINAL / "F" / original_row / "source"
    f = star_reader(folder, read(h["source_contract"]), SEED,
                    dict(run_id=ORIGINAL, row_id=original_row), "cpu")
    signal = load_source(h["source_cache"], ("L_s",))["L_s"][0][:2]
    with torch.inference_mode():
        outputs = [forward(f, tensor(signal), bank)[1] for bank in ("clean", "sg")]
        if any(x.shape != (2, 6) or not bool(x.isfinite().all()) for x in outputs):
            raise ValueError("Frozen native STAR checkpoint source-only smoke failed")
    write(root / "smoke.json", dict(status="PASS", source_checkpoint=ORIGINAL,
                                    role="L_s", query_reads=0, inherited_weights=False))
    print("STAR_RX_SCREEN_SOURCE_SMOKE_PASS", flush=True)


def worker(stage, c):
    from experiments.practical_full_2x2.compat import numerical_policy
    numerical_policy()
    if stage == "train":
        from dataclasses import replace
        from .training import RECIPE, train
        if (c["run_id"], c["row_id"], c["model_seed"]) != (RUN, row_id(c["variant"]), SEED):
            raise ValueError("Incorrect STAR screen candidate")
        return train(c, cfg=replace(RECIPE, **VARIANTS[c["variant"]]))
    from experiments.practical_full_2x2 import evaluation
    if stage == "score":
        return evaluation.score(c)
    from experiments.pamf_comparison_matrix.runner import star_reader
    from .training import forward
    evaluation.reject_truth(c)
    model = star_reader(c["source_output"], read(c["source_contract"]), c["model_seed"],
                        c["frozen_source"], "cuda:0")
    return evaluation.predict(c, classify_view=lambda x, view:
                              forward(model, x, "clean" if view == "clean" else "sg")[1])


def start(root, stage, row, gpu):
    config = root / "configs" / f"{stage}__{row}.json"
    argv = [PYTHON, "-u", "-m", MODULE, stage, "--config", str(config)]
    log = root / "logs" / f"{stage}__{row}.log"
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID",
               CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
               PYTHONPATH=f"{RELEASE}:{RELEASE / 'code'}", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2")
    with log.open("x") as stream:
        proc = subprocess.Popen(argv, cwd=RELEASE, env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT)
    print("STAR_RX_WORKER", row, stage, proc.pid, gpu, flush=True)
    return proc, dict(pid=proc.pid, gpu=gpu, stage=stage, cwd=str(RELEASE), argv=argv, log=str(log))


def compare(root):
    with Path(BASELINE).open(encoding="utf-8-sig", newline="") as f:
        controls = {(r["method"], r["population"], r["route"], r["view"]):
                    (float(r["accuracy"]), float(r["macro_f1"]), int(r["count"]))
                    for r in csv.DictReader(f) if r["setting"] == "F0" and int(r["seed"]) == SEED}
    views = tuple(VIEWS) + (SIX,)
    expected = {(p, r, v) for p in TASKS for r in READOUTS for v in views}
    if controls.keys() != {(m, *key) for m in METHODS.values() for key in expected}:
        raise ValueError("The full seven-method matched-seed comparator is required")
    rows = []
    verdicts = {}
    for variant in VARIANTS:
        row = row_id(variant)
        path = root / "readouts" / row / "scores" / "scores.json"
        scores = read(path)
        if (scores["run_id"], scores["row_id"], scores["model_seed"], scores["truth_last"]) != (RUN, row, SEED, True):
            raise ValueError("The candidate requires exact frozen truth-last scores")
        cells = {(r["test"], r["readout"], r["view"]): (100*r["accuracy"], 100*r["macro_f1"], r["count"])
                 for r in scores["results"] if r["dimension"] == "overall"}
        for task in TASKS:
            for route in READOUTS:
                practical = [cells[task, route, v] for v in VIEWS if v != "clean"]
                cells[task, route, SIX] = (mean(t[0] for t in practical), mean(t[1] for t in practical), practical[0][2])
        if cells.keys() != expected:
            raise ValueError("Candidate is missing a preregistered test cell")
        for (task, route, view), (acc, macro_f1, count) in cells.items():
            comparable = {m: controls[m, task, route, view] for m in METHODS.values()}
            if any(ref[2] != count for ref in comparable.values()):
                raise ValueError("Candidate and comparator have different target populations")
            highest = max(comparable, key=lambda m: comparable[m][0])
            native = comparable[METHODS["star"]][0]
            rows.append(dict(candidate=variant, population=task, readout=route, view=view,
                             count=count, accuracy_pct=acc, macro_f1_pct=macro_f1,
                             native_star_pct=native, gain_vs_native_pp=acc-native,
                             best_baseline=highest, best_baseline_pct=comparable[highest][0],
                             gain_vs_best_pp=acc-comparable[highest][0],
                             **{m: values[0] for m, values in comparable.items()}))
        rx_gains = {p: cells[p, "raw", SIX][0] - max(controls[m, p, "raw", SIX][0] for m in METHODS.values())
                    for p in RX_TASKS}
        clean_gains = {p: cells[p, "raw", "clean"][0] - controls[METHODS["star"], p, "raw", "clean"][0]
                       for p in TASKS}
        date_gain = cells["cross_date", "raw", SIX][0] - controls[METHODS["star"], "cross_date", "raw", SIX][0]
        overall = mean(cells[p, "raw", SIX][0] for p in TASKS)
        best_overall = max(mean(controls[m, p, "raw", SIX][0] for p in TASKS) for m in METHODS.values())
        passes = (min(rx_gains.values()) >= MARGIN_PP and overall > best_overall
                  and min(clean_gains.values()) >= -TOLERANCE_PP and date_gain >= -TOLERANCE_PP)
        verdicts[variant] = dict(raw_gain_vs_best_others_pp=rx_gains, raw_clean_gain_vs_native_pp=clean_gains,
                                cross_date_practical_gain_vs_native_pp=date_gain,
                                three_population_raw_mean_pct=overall,
                                best_baseline_three_population_raw_mean_pct=best_overall,
                                passes_screen=passes)
    with (root / "paired_comparison.csv").open("x", encoding="utf-8", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    result = dict(status="ANALYZED", model_seed=SEED, fixed=verdicts, truth_last=True,
                  scope="Single-seed screen; no selection or automatic expansion")
    write(root / "complete.json", result)
    return result


def dispatch(runtime):
    root = Path(ROOT)
    root.mkdir(exist_ok=False)
    (root / "logs").mkdir()
    location = root / "pipeline_state.json"
    state = dict(controller_pid=os.getpid(), code_commit=runtime["code_commit"])
    atomic(location, dict(state, status="SOURCE_SMOKE"))
    stages = {row_id(v): "train" for v in VARIANTS}
    active, failed = {}, {}
    launches = []
    next_stage = {"train": "predict", "predict": "score", "score": "SCORED"}
    try:
        context = read(CONTEXT)
        smoke(root, context)
        prepare(runtime, root, context)
        while active or any(x in next_stage for x in stages.values()):
            for row, (p, info) in list(active.items()):
                code = p.poll()
                if code is None:
                    continue
                del active[row]
                stages[row] = "FAILED" if code else next_stage[info["stage"]]
                if code:
                    failed[row] = dict(info, exit_code=code)
            free = resources()
            busy = {info["gpu"] for _, info in active.values()}
            for row, stage in stages.items():
                if row in active or stage not in next_stage:
                    continue
                if stage == "score":
                    if None in busy:
                        continue
                    gpu = None
                else:
                    gpu = next((g for g in GPUS if g not in busy and free.get(g, 0) >= MIN_FREE_MIB), None)
                    if gpu is None:
                        continue
                active[row] = start(root, stage, row, gpu)
                launches.append(dict(row=row, **active[row][1]))
                busy.add(gpu)
            atomic(location, dict(state, status="RUNNING", stages=stages, failed=failed, launches=launches,
                                  active={row: info for row, (_, info) in active.items()}))
            if active or any(x in next_stage for x in stages.values()):
                time.sleep(12)
        if failed:
            raise RuntimeError("One or more screen rows failed; retain completed artifacts")
        result = compare(root)
        atomic(location, dict(state, status=result["status"], stages=stages, launches=launches))
    except Exception as exc:
        atomic(location, dict(state, status="FAILED", error=str(exc), stages=stages,
                              failed=failed, launches=launches))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("dispatch", "train", "predict", "score"))
    p.add_argument("--config", required=True)
    a = p.parse_args()
    c = read(a.config)
    if a.stage == "dispatch":
        dispatch(c)
    else:
        worker(a.stage, c)
