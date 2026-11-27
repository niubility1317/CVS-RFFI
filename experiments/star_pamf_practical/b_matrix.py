"""Run the fixed STAR B recipe on every original matched training setting."""
import argparse
from collections import Counter
import csv
from pathlib import Path
from statistics import mean, stdev
import os
import time

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_comparison_matrix.plan import (
    CONTEXT, GPUS, GPU_WORKERS, METHODS, PROJECT, PYTHON, RUN as BASELINE, STAR, published_scores, settings,
)
from experiments.pamf_comparison_matrix.runner import spawn
from experiments.practical_full_2x2.cache import atomic
from .rx_screen import RUN as PILOT, SEED, VARIANTS, MIN_FREE_MIB, row_id as pilot_row_id

RUN = "20261127-phase1-star-b-practical-m57-r01"
MODULE = "experiments.star_pamf_practical.b_matrix"
ROOT = PROJECT / "runs" / RUN
RELEASE = PROJECT / "releases" / (RUN + "-release")
COMMAND = [PYTHON, "-u", "-m", MODULE, "dispatch", "--config", str(RELEASE / "runtime.json")]
VARIANT = "cross-rx-label-anchor"
OPTIONS = VARIANTS[VARIANT]
PILOT_ROW = pilot_row_id(VARIANT)
PILOT_ROOT = PROJECT / "runs" / PILOT
EXISTING = ("F0", SEED)
NEW = "STAR-RFFI B"
METHOD_COLUMNS = dict(METHODS, star_b=NEW)


def rows():
    return [(name, seed) for name, info in settings().items() for seed in info["seeds"]]


def label(setting, seed):
    return f"{setting}-star-b-s{seed}"


def original(setting, seed):
    if setting == "F0":
        return PROJECT / "runs" / STAR, f"main-star-s{seed}"
    return PROJECT / "runs" / BASELINE, f"{setting}-star-s{seed}"


def prepare(commit, root):
    queue = {}
    for setting, seed in rows():
        if (setting, seed) == EXISTING:
            continue
        parent, old = original(setting, seed)
        train, predict, score = (read(parent / "configs" / f"{stage}__{old}.json")
                                 for stage in ("train", "predict", "score"))
        if any((c["row_id"], c["model_seed"]) != (old, seed) for c in (train, predict, score)):
            raise ValueError("Original STAR configuration identity differs")
        pair_receipt = read(Path(train["pairs"]) / "complete.json")
        if any(pair_receipt[k] for k in ("U_TX_truth_read", "target_read", "validation_used")):
            raise ValueError("Original paired practical cache must use only source L/U")
        name = label(setting, seed)
        source = root / "F" / name / "source"
        dest = root / "readouts" / name
        shared = dict(row_id=name, model_seed=seed, run_id=RUN)
        train.update(shared, code_commit=commit, source_output=str(source), variant=VARIANT)
        if "source_dimensions" not in train:
            contract = read(Path(train["source_cache"]) / "source_contract.json")
            train["source_dimensions"] = {k: contract[k] for k in ("counts", "classes", "source_rxs", "source_days")}
        predict.update(shared, source_output=str(source), output_root=str(dest / "predictions"),
                       frozen_source=dict(run_id=RUN, row_id=name), reuse_raw=None)
        score.update(shared, prediction_output=str(dest / "predictions"), output_root=str(dest / "scores"))
        queue[name] = "train"
        for stage, c in (("train", train), ("predict", predict), ("score", score)):
            write(root / "configs" / f"{stage}__{name}.json", c)
    if len(queue) != 56:
        raise ValueError("The matrix must contain 56 new B training rows")
    return queue


def smoke(root):
    import torch
    from experiments.pamf_comparison_matrix.runner import star_reader
    from experiments.practical_full_2x2.compat import tensor
    from .data import load_source
    from .training import forward

    h = read(CONTEXT)["handles"]["F0"]
    source = PILOT_ROOT / "F" / PILOT_ROW / "source"
    frozen = read(source / "resolved_config.json")
    if (frozen["source_cache"] != h["source_cache"] or frozen["checkpoint_init"] != "scratch"
            or any(frozen["recipe"][k] != v for k, v in OPTIONS.items())):
        raise ValueError("Existing B Main checkpoint does not match the registered training")
    f = star_reader(source, read(h["source_contract"]), SEED, dict(run_id=PILOT, row_id=PILOT_ROW), "cpu")
    x = tensor(load_source(h["source_cache"], ("L_s",))["L_s"][0][:2])
    with torch.inference_mode():
        for bank in ("clean", "sg"):
            y = forward(f, x, bank)[1]
            if y.shape != (2, 6) or not bool(y.isfinite().all()):
                raise FloatingPointError("B source-only smoke failed")
    write(root / "smoke.json", dict(status="PASS", checkpoint_run=PILOT, input_role="L_s",
                                    target_iq_reads=0, query_truth_reads=0, inherited_weights=False))
    print("STAR_B_SOURCE_CHECKPOINT_PASS", flush=True)


def worker(stage, method, c):
    if stage == "train":
        from dataclasses import replace
        from experiments.practical_full_2x2.compat import numerical_policy
        from .training import RECIPE, train
        numerical_policy()
        if c["run_id"] != RUN or c["variant"] != VARIANT:
            raise ValueError("Unexpected STAR B row configuration")
        return train(c, cfg=replace(RECIPE, **OPTIONS))
    from experiments.pamf_comparison_matrix.runner import worker as evaluate
    return evaluate(stage, method, c)


def analyze(root, *, main_only=False):
    from experiments.pamf_comparison_matrix.results import summarize, table

    def resolve(setting, info, method, seed):
        if method != "star_b":
            return published_scores(setting, info, method, seed)
        if (setting, seed) == EXISTING:
            return PILOT_ROOT / "readouts" / PILOT_ROW
        return root / "readouts" / label(setting, seed)

    receipt = summarize(root, main_only=main_only, methods=METHOD_COLUMNS, resolve_scores=resolve,
                        columns_file="eight_method_columns.csv")
    output = root / ("main_snapshot" if main_only else "tables")
    with (output / "per_seed.csv").open(encoding="utf-8-sig", newline="") as f:
        per_seed = list(csv.DictReader(f))
    keys = ("setting", "seed", "population", "route", "view")
    old = {tuple(r[k] for k in keys): r for r in per_seed if r["method"] == METHODS["star"]}
    pairs = []
    grouped = {}
    for new in (r for r in per_seed if r["method"] == NEW):
        key = tuple(new[k] for k in keys)
        base = old[key]
        if new["count"] != base["count"]:
            raise ValueError(f"STAR comparison population mismatches for {key}")
        delta = {m: float(new[m]) - float(base[m]) for m in ("accuracy", "macro_f1")}
        pairs.append(dict(zip(keys, key), count=base["count"], original_accuracy=base["accuracy"],
                          b_accuracy=new["accuracy"], accuracy_delta_pp=delta["accuracy"],
                          original_macro_f1=base["macro_f1"], b_macro_f1=new["macro_f1"],
                          macro_f1_delta_pp=delta["macro_f1"]))
        if new["population"] == "cross_rx":
            grouped.setdefault((key[0], *key[2:]), []).append((base, new, delta))
    if receipt["missing_rows"] == 0 and len(pairs) != len(old):
        raise ValueError("Complete B matrix does not pair every original metric")
    table(output / "paired_star_per_seed.csv", pairs)
    primary = []
    for (setting, population, route, view), rs in grouped.items():
        ds = [r[2]["accuracy"] for r in rs]
        primary.append(dict(setting=setting, population=population, route=route, view=view, n=len(rs),
                            original_accuracy_mean=mean(float(r[0]["accuracy"]) for r in rs),
                            b_accuracy_mean=mean(float(r[1]["accuracy"]) for r in rs),
                            paired_accuracy_delta_mean_pp=mean(ds), paired_accuracy_delta_sd_pp=stdev(ds) if len(ds) > 1 else "",
                            paired_macro_f1_delta_mean_pp=mean(r[2]["macro_f1"] for r in rs)))
    table(output / "cross_rx_paired_summary.csv", primary)
    final = dict(receipt, primary_population="cross_rx", primary_route="raw",
                 primary_view="six_practical_mean", paired_cells=len(pairs), reused_main_seed=SEED)
    write(output / "paired_complete.json", final)
    return final


def dispatch(runtime, *, resume_state=None, adopted=None, max_workers=GPU_WORKERS):
    from .runner import resources

    root = Path(ROOT)
    target = root / "pipeline_state.json"
    base = dict(controller_pid=os.getpid(), code_commit=runtime["code_commit"], gpu_workers=max_workers)
    active, failed, stages = {}, {}, {}
    snapshot_attempted = False
    snapshot_error = None
    next_stage = {"train": "predict", "predict": "score", "score": "SCORED"}
    order = {"score": 0, "predict": 1, "train": 2, "SCORED": 3, "FAILED": 3, "BLOCKED": 3}
    try:
        if resume_state is None:
            root.mkdir(exist_ok=False)
            (root / "logs").mkdir()
            atomic(target, dict(base, status="SOURCE_SMOKE"))
            smoke(root)
            stages = prepare(runtime["code_commit"], root)
        else:
            base["scheduler_commit"] = runtime["scheduler_commit"]
            active, failed, stages = adopted, resume_state["failed"], resume_state["stages"]
            snapshot_error = resume_state.get("snapshot_error")
            snapshot_attempted = bool(snapshot_error) or (root / "main_snapshot").exists()
        while active or any(s in next_stage for s in stages.values()):
            for row, (proc, info) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                del active[row]
                stages[row] = "FAILED" if code else next_stage[info["stage"]]
                if code:
                    failed[row] = dict(info, exit_code=code)
            if not snapshot_attempted and all(status == "SCORED" for row, status in stages.items() if row.startswith("F0-")):
                snapshot_attempted = True
                try:
                    analyze(root, main_only=True)
                except (ValueError, OSError, KeyError) as exc:
                    snapshot_error = str(exc)
                    print("STAR_B_SNAPSHOT_FAILED", snapshot_error, flush=True)
            failed_by_stage = Counter(info["stage"] for info in failed.values())
            free = resources()
            busy = {info["gpu"] for _, info in active.values()}
            for row, stage in sorted(stages.items(), key=lambda r: order[r[1]]):
                if row in active or stage not in next_stage:
                    continue
                if failed_by_stage[stage] >= 2:
                    stages[row] = "BLOCKED"
                    continue
                if stage == "score":
                    if None in busy:
                        continue
                    gpu = None
                else:
                    if len(busy - {None}) >= max_workers:
                        continue
                    floor = MIN_FREE_MIB if stage == "train" else 7000
                    gpu = next((g for g in GPUS if g not in busy and free.get(g, 0) >= floor), None)
                    if gpu is None:
                        continue
                job = dict(key=f"{stage}__{row}", stage=stage, method="star",
                           config=str(root / "configs" / f"{stage}__{row}.json"))
                active[row] = spawn(root, job, gpu, release=RELEASE, module=MODULE)
                busy.add(gpu)
                print("STAR_B_WORKER", row, stage, active[row][0].pid, gpu, flush=True)
            atomic(target, dict(base, status="RUNNING", active={k: v[1] for k, v in active.items()},
                                stages=stages, failed=failed, snapshot_error=snapshot_error))
            if active or any(s in next_stage for s in stages.values()):
                time.sleep(12)
        result = analyze(root)
        atomic(target, dict(base, status=result["status"], stages=stages, failed=failed,
                            snapshot_error=snapshot_error, summary=result))
    except Exception as exc:
        if resume_state is not None or stages or (
                target.is_file() and read(target).get("controller_pid") == os.getpid()):
            atomic(target, dict(base, status="FAILED", error=str(exc), stages=stages, failed=failed,
                                active={k: v[1] for k, v in active.items()}))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("dispatch", "train", "predict", "score"))
    p.add_argument("--method", default="star")
    p.add_argument("--config", required=True)
    a = p.parse_args()
    c = read(a.config)
    if a.stage == "dispatch":
        dispatch(c)
    else:
        worker(a.stage, a.method, c)
