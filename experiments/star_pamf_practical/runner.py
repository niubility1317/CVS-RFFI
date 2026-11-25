"""N607 dispatcher, frozen prediction and independent score for the Main split."""
import argparse
import csv
from pathlib import Path
import os
import statistics
import subprocess
import time

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.practical_full_2x2.cache import atomic
from experiments.practical_full_2x2.plan import SEEDS, TASKS, VIEWS
from .data import CLASSES, DAYS, PROTOCOL, load_source, prepare_pairs

RUN = "20261125-phase1-star-pamf-main-practical-m3-r01"
MODULE = "experiments.star_pamf_practical.runner"
PROJECT = Path("/home/szu2070436088/2510044040/CV-SincNet")
PYTHON = "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
PAPER = "20261027-phase1-pamf-paper-completion-m119-r03"
CONTEXT = PROJECT / "releases" / f"{PAPER}-release" / "context.json"
RUN_ROOT = PROJECT / "runs" / RUN
RELEASE = PROJECT / "releases" / f"{RUN}-release"
COMMAND = [PYTHON, "-u", "-m", MODULE, "dispatch", "--config", str(RELEASE / "runtime.json")]
GPUS = (2, 7, 6, 3)
MIN_FREE_MIB = 16000


def row_id(seed):
    return f"main-star-s{seed}"


def worker_env(gpu=None):
    return dict(os.environ, CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
                CUDA_DEVICE_ORDER="PCI_BUS_ID",
                PYTHONPATH=os.pathsep.join((str(RELEASE), str(RELEASE / "code"))),
                OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")


def prepare(rt, ctx, root):
    handle = ctx["handles"][PROTOCOL]
    if handle["classes"] != CLASSES or tuple(handle["tests"]) != TASKS:
        raise ValueError("Unexpected Main split test contract")
    cache = ctx["caches"][PROTOCOL]["LU"]
    for seed in SEEDS:
        row = row_id(seed)
        frozen = root / "F" / row / "source"
        prediction = root / "readouts" / row / "predictions"
        common = dict(run_id=RUN, row_id=row, model_seed=seed, protocol_id=PROTOCOL,
                      classes=CLASSES, source_days=list(DAYS), readouts=["raw", "DURA-LU"],
                      source_contract=handle["source_contract"], views=list(VIEWS),
                      G_roles="LU")
        write(root / "configs" / f"train__{row}.json",
              dict(run_id=RUN, row_id=row, model_seed=seed, source_cache=handle["source_cache"],
                   pairs=str(root / "source_pairs"), source_output=str(frozen),
                   code_commit=rt["code_commit"]))
        write(root / "configs" / f"predict__{row}.json",
              dict(common, tests={k: {"views_root": handle["tests"][k]["views_root"]} for k in TASKS},
                   source_output=str(frozen), restored_cache=cache["path"], output_root=str(prediction)))
        write(root / "configs" / f"score__{row}.json",
              dict(common, tests=handle["tests"], prediction_output=str(prediction),
                   output_root=str(prediction.parent / "scores")))


def smoke(ctx, root):
    import torch
    from experiments.practical_full_2x2.cache import transform
    from experiments.practical_full_2x2.compat import tensor
    from .model import Ni_model_2part
    from .training import encode_iq, forward, new_model
    from .view_bn import encode_view

    # Exercise the historical checkpoint only on authorized Main split L IQ.
    # Its state is isolated from the new six-class model.
    old = PROJECT.parent / "STAR-RFFI" / "STAR-test" / "补实验"
    history = old / "smooth_e_bn_budget_dual_rx1_s16_t90_e100_20261008_r01"
    worker = history / "runs" / "smooth-e-bn-budget-rx1-s302-t90-e100-r01"
    checkpoint, = worker.rglob("final_last_epoch_source_bn.pth")
    historical = Ni_model_2part(10).eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["epoch"] != 100:
        raise ValueError("Historical fixed checkpoint is unavailable")
    historical.load_state_dict(payload["state_dict"], strict=True)
    with torch.inference_mode():
        pairs = load_source(ctx["handles"][PROTOCOL]["source_cache"], ("L_s",))
        x, meta = pairs["L_s"]
        sg = np.stack([transform(x[i], meta["ids"][i], int(meta["rx_i"][i]), int(meta["day_i"][i]),
                                 1, 1, 1.0)[0] for i in range(2)])
        for bank, iq in (("clean", x[:2]), ("sg", sg)):
            features = encode_view(historical.encoder, encode_iq(tensor(iq)), bank)
            logits = historical.classifier(features[:, 128:], stochastic=False)
            if logits.shape != (2, 10) or not bool(torch.isfinite(logits).all()):
                raise ValueError("Historical source-only STAR checkpoint smoke failed")
        scratch = new_model("cpu").eval()
        logits = forward(scratch, tensor(sg), "sg")[1]
        if logits.shape != (2, 6) or not bool(torch.isfinite(logits).all()):
            raise ValueError("Main split practical source smoke failed")
    write(root / "smoke.json", dict(status="PASS", input_role="L_s",
                                    inherited_weights=False, query_reads=0))
    print("STAR_SOURCE_ONLY_SMOKE_PASS", flush=True)


def predict(c, device="cuda:0"):
    import torch
    from .training import forward, new_model
    from experiments.practical_full_2x2 import evaluation

    evaluation.reject_truth(c)
    frozen = read(Path(c["source_output"]) / "complete.json")
    if frozen["status"] != "SOURCE_TRAINED_FROZEN" or frozen["target_access"]:
        raise ValueError("Target evaluation requires a frozen source checkpoint")
    checkpoint = torch.load(frozen["checkpoint"], map_location=device, weights_only=True)
    if (checkpoint["run_id"], checkpoint["row_id"], checkpoint["model_seed"], checkpoint["epoch"]) != (
            RUN, c["row_id"], c["model_seed"], 100):
        raise ValueError("The fixed E100 STAR checkpoint must match the exact registered row")
    model = new_model(device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    evaluation.predict(c, device=device, classify_view=lambda x, view:
                       forward(model, x, "clean" if view == "clean" else "sg")[1])


def analyze(root):
    records = []
    paper = PROJECT / "runs" / PAPER / "readouts"
    groups = {}
    for seed in SEEDS:
        row = row_id(seed)
        star = read(root / "readouts" / row / "scores" / "scores.json")
        ref = read(paper / f"main_split-factorial-ssl-full-permuted-s{seed}" / "scores" / "scores.json")
        if not (star["truth_last"] and ref["truth_last"] and star["model_seed"] == ref["model_seed"] == seed):
            raise ValueError("The paired PAMF and STAR scores must be frozen and independently scored")
        key = lambda r: (r["test"], r["readout"], r["view"])
        ref_results = {key(r): r for r in ref["results"] if r["dimension"] == "overall"}
        results = [r for r in star["results"] if r["dimension"] == "overall"]
        if len(results) != len(TASKS) * len(VIEWS) * 2:
            raise ValueError("Incomplete STAR result matrix")
        for r in results:
            comparator = ref_results[key(r)]
            if r["count"] != comparator["count"]:
                raise ValueError("PAMF and STAR populations differ")
            record = dict(seed=seed, task=r["test"], route=r["readout"], view=r["view"], count=r["count"],
                          star_accuracy=r["accuracy"], star_macro_f1=r["macro_f1"],
                          pamf_accuracy=comparator["accuracy"], pamf_macro_f1=comparator["macro_f1"],
                          accuracy_delta_pp=100 * (r["accuracy"] - comparator["accuracy"]))
            records.append(record)
            groups.setdefault(key(r), []).append(record)
    summary = []
    fields = ("star_accuracy", "pamf_accuracy", "star_macro_f1", "pamf_macro_f1", "accuracy_delta_pp")
    for (task, route, view), rows in groups.items():
        result = dict(task=task, route=route, view=view, seeds=len(rows))
        for field in fields:
            values = [r[field] for r in rows]
            result[field + "_mean"] = statistics.mean(values)
            result[field + "_sample_sd"] = statistics.stdev(values)
        summary.append(result)
    for task in TASKS:
        for route in ("raw", "DURA-LU"):
            rows = [r for r in records if r["task"] == task and r["route"] == route and r["view"] != "clean"]
            result = dict(task=task, route=route, view="six_practical_mean", seeds=len(SEEDS))
            for field in fields:
                values = [statistics.mean(r[field] for r in rows if r["seed"] == seed) for seed in SEEDS]
                result[field + "_mean"] = statistics.mean(values)
                result[field + "_sample_sd"] = statistics.stdev(values)
            summary.append(result)
    for file, rows in (("paired_per_seed.csv", records), ("paired_summary.csv", summary)):
        with (root / file).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write(root / "complete.json", dict(status="ANALYZED", rows=list(map(row_id, SEEDS)),
                                        paired_summary="paired_summary.csv", truth_last=True))


def resources():
    lines = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
                                     "--format=csv,noheader,nounits"], text=True).splitlines()
    return dict(tuple(map(int, line.split(","))) for line in lines)


def start(root, action, row, gpu=None):
    config = root / "configs" / f"{action}__{row}.json"
    command = [PYTHON, "-u", "-m", MODULE, action, "--config", str(config)]
    log_path = root / "logs" / f"{action}__{row}.log"
    with log_path.open("x") as log:
        proc = subprocess.Popen(command, cwd=RELEASE, env=worker_env(gpu), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
    return proc, dict(pid=proc.pid, cwd=str(RELEASE), argv=command, gpu=gpu,
                      log=str(log_path), action=action)


def dispatch(rt):
    from experiments.practical_full_2x2.compat import numerical_policy
    numerical_policy()
    root = RUN_ROOT
    root.mkdir(exist_ok=False)
    (root / "logs").mkdir()
    atomic(root / "pipeline_state.json", dict(status="SOURCE_SMOKE"))
    ctx = read(CONTEXT)
    smoke(ctx, root)
    prepare(rt, ctx, root)
    atomic(root / "pipeline_state.json", dict(status="PREPARING_PRACTICAL_SOURCE_PAIRS"))
    prepare_pairs(ctx["handles"][PROTOCOL]["source_cache"], root / "source_pairs", workers=8)

    queue = [row_id(seed) for seed in SEEDS]
    stages = {row: "train" for row in queue}
    active = {}
    failures = {}
    trace = []
    while active or any(stage in ("train", "predict", "score") for stage in stages.values()):
        for row, (proc, data) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            del active[row]
            if code:
                failures[row] = dict(data, exit_code=code)
                stages[row] = "FAILED"
            else:
                stages[row] = {"train": "predict", "predict": "score", "score": "SCORED"}[data["action"]]
        if len(failures) >= 2:
            for row in stages:
                if row not in active and stages[row] == "train":
                    stages[row] = "STOPPED"
        free = resources()
        occupied = {data["gpu"] for _, data in active.values()}
        for row, stage in stages.items():
            if row in active or stage not in ("train", "predict", "score"):
                continue
            gpu = None
            if stage == "score":
                if None in occupied:
                    continue
            else:
                gpu = next((index for index in GPUS if index not in occupied
                            and free.get(index, 0) >= MIN_FREE_MIB), None)
                if gpu is None:
                    continue
            active[row] = start(root, stage, row, gpu)
            occupied.add(gpu)
            trace.append(dict(row_id=row, **active[row][1]))
            print("STAR_WORKER_START", trace[-1], flush=True)
        atomic(root / "pipeline_state.json", dict(status="RUNNING", stages=stages, failures=failures,
              active={row: data for row, (_, data) in active.items()}, launches=trace, gpu_free_mib=free))
        if active or any(s in ("train", "predict", "score") for s in stages.values()):
            time.sleep(10)
    if failures:
        atomic(root / "pipeline_state.json", dict(status="FAILED", stages=stages, failures=failures,
                                                launches=trace))
        raise RuntimeError("One or more registered rows failed; see pipeline_state.json")
    analyze(root)
    atomic(root / "pipeline_state.json", dict(status="ANALYZED", stages=stages, launches=trace))


def main():
    from experiments.practical_full_2x2.compat import numerical_policy
    from .training import train
    from experiments.practical_full_2x2 import evaluation
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["dispatch", "train", "predict", "score"])
    p.add_argument("--config", required=True)
    opts = p.parse_args()
    numerical_policy()
    action = {"dispatch": dispatch, "train": train, "predict": predict, "score": evaluation.score}
    action[opts.action](read(opts.config))


if __name__ == "__main__":
    main()
