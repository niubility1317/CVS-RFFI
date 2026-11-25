"""Run one predeclared matched seed, then freeze, predict, score and compare."""
import argparse
import csv
import os
from pathlib import Path
import statistics
import subprocess
import time

from experiments.cric_dura_comparison.io import read, write
from experiments.practical_full_2x2.cache import atomic
from experiments.practical_full_2x2.plan import TASKS, VIEWS
from .protocol import (BASE_READOUT, COMMAND, CONTEXT, GPUS, MIN_MIB, MODULE, PREFIX,
                       RELEASE, ROOT, ROUTES, ROW, RUN, SEED, SOURCE_OUT, PYTHON,
                       configuration, prefix_record)


def prepare(runtime):
    from experiments.cric_dura_comparison.evaluation import handle
    ctx = read(CONTEXT)
    h = ctx["handles"]["F0"]
    root = Path(ROOT)
    candidate = configuration(runtime["code_commit"])
    prefix_record(candidate, PREFIX)
    contract = read(candidate["source_contract"])
    if contract != read(h["source_contract"]):
        raise ValueError("Main split training and practical readout contracts differ")
    cache = ctx["caches"]["F0"]["LU"]["path"]
    restored = read(Path(cache) / "complete.json")
    if read(restored["G"]["source_contract"]) != contract:
        raise ValueError("DURA-LU has a different source contract")
    for task in TASKS:
        # Only the previously validated manifest is checked before training.
        manifest = read(Path(h["tests"][task]["views_root"]) / "manifest.json")
        if manifest["status"] != "VALIDATED_ONCE" or manifest["classes"] != contract["classes"]:
            raise ValueError("Missing validated Phase1 test handle")
    pred = root / "readouts" / ROW / "predictions"
    common = dict(run_id=RUN, row_id=ROW, model_seed=SEED, protocol_id="F0", classes=contract["classes"],
                  source_days=[1, 2, 3], views=list(VIEWS), readouts=list(ROUTES), G_roles="LU")
    configs = dict(
        train=candidate,
        predict=dict(common, source_contract=candidate["source_contract"], source_output=str(SOURCE_OUT),
                     tests={t: {"views_root": h["tests"][t]["views_root"]} for t in TASKS},
                     restored_cache=cache, frozen_sources=candidate["frozen_sources"], output_root=str(pred)),
        score=dict(common, tests={t: h["tests"][t] for t in TASKS}, prediction_output=str(pred),
                   output_root=str(pred.parent / "scores")),
    )
    for stage, config in configs.items():
        write(root / "configs" / f"{stage}.json", config)


def smoke(c):
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison import f_train, t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_scope
    from experiments.pamf_repl_view_shuffle.source import assert_structure, installed

    prefix_record(c, PREFIX)
    payload = torch.load(Path(PREFIX) / "final_ssdg.pth", map_location="cpu", weights_only=False)
    if (payload["run_id"] != prefix_record(c, PREFIX)["config"]["run_id"]
            or payload["candidate_id"] != ROW or payload["epoch"] != 20
            or payload["args"]["baseline_ckpt"] or payload["args"]["teacher_ckpt"]
            or payload["a1_ema_successful_updates"] != 4440):
        raise ValueError("Real E20 checkpoint has a different ancestry")
    cfg = dict(c, hard_pl_smoke=True, epoch_offset=20, step_offset=4440, segment_epochs=10,
               parent_source=None)
    holder = {}
    device = "cuda:0"
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    with installed(cfg, holder, True):
        loaders = f_train.context(cfg)
        with f_train.class_dimension(cfg), native_scope(f_train.original_config(SEED)) as native:
            args = f_train.arguments(cfg, device)
            args.input_len = 256
            net = t1_model.install(native.build_baseline_model(args, device), cfg).to(device)
        assert_structure(net, cfg)
        net.load_state_dict(payload["model"], strict=True)
        net.train()
        net._cd_U_loader = loaders["unlabeled_loader"]
        net._cd_U_iter = iter(net._cd_U_loader)
        batch = next(iter(loaders["train_loader"]))
        x, y, domain = [v.to(device) for v in batch[:3]]
        meta = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch[3].items()}
        with net._phase1_labeled_feature_context():
            output = net(x, return_aux=True, domain_labels=domain)
        ce = F.cross_entropy(output["tx_logits"], y, label_smoothing=.01)
        correction, metrics = net._source_labeled_feature_hook(
            model=net, labeled_output=output, labels=y, extra=(None, meta), domains=domain,
            optimizer_step=1, identity_loss=ce, labeled_count=128, labeled_inputs=x)
        loss = ce + correction
        loss.backward()
        state = holder["objective"].state_dict()
        head = net.id_backbone.encoder.core.id_backbone.cls_head.weight
        if (not bool(torch.isfinite(loss)) or head.grad is None or not bool(head.grad.isfinite().all())
                or float(head.grad.norm()) <= 0 or state["calls"] != 1 or not state["accepted"]
                or state["student_views"] != 1536 or state["physical_seen"] != 128
                or state["physical_student_views"] != 256 or metrics["T1_L_packets"] != 768
                or state["U_gradient_guard"]["calls"] != 1 or state["U_truth_read"]
                or sum(net._cd_practical.coverage.values()) != 64):
            raise ValueError("Real E20 no-query U As forward/backward differs")
    record = dict(status="PASS", real_source_checkpoint=True, query_reads=0, optimizer_updates=0,
                  source_U_samples=256, practical_U_slots=128, U_scene_counts=state["U_practical"]["scene_counts"])
    write(Path(ROOT) / "smoke.json", record)
    print("SOURCE_CHECKPOINT_SMOKE_PASS", record, flush=True)


def predict(c):
    from experiments.cric_dura_comparison.evaluation import reject_truth
    from experiments.pamf_repl_view_shuffle.source import load_reader
    from experiments.practical_full_2x2 import evaluation

    reject_truth(c)
    if (c["run_id"], c["row_id"], c["model_seed"]) != (RUN, ROW, SEED):
        raise ValueError("Unexpected candidate row")
    reader, cfg = load_reader(c["source_output"], read(c["source_contract"]), "cuda:0")
    if (cfg["run_id"], cfg["row_id"], cfg["model_seed"]) != (RUN, ROW, SEED):
        raise ValueError("Mismatched frozen candidate")
    evaluation.predict(c, classify_view=lambda x, _: reader.classify_features(reader.features(x)))


def worker(stage, config):
    c = read(config)
    if stage == "score":
        from experiments.practical_full_2x2.evaluation import score
        return score(c)
    from experiments.pamf_repl_view_shuffle.runner import activate
    activate(c)
    if stage == "smoke":
        return smoke(c)
    if stage == "train":
        from experiments.pamf_repl_view_shuffle.source import train
        return train(c, prefix=PREFIX)
    return predict(c)


def analyze(root):
    import numpy as np

    out = root / "readouts" / ROW
    star = read(out / "scores" / "scores.json")
    base = read(Path(BASE_READOUT) / "scores" / "scores.json")
    if any(s["model_seed"] != SEED or not s["truth_last"] for s in (star, base)):
        raise ValueError("Independent same-seed frozen scores required")
    for task in TASKS:
        def ids(folder):
            with np.load(folder / "predictions" / task / "predictions.npz", allow_pickle=False) as d:
                return d["ids"].copy()
        if not np.array_equal(ids(out), ids(Path(BASE_READOUT))):
            raise ValueError("Paired target physical IDs differ")
    def values(score):
        return {(r["test"], r["readout"], r["view"]): r for r in score["results"]
                if r["dimension"] == "overall"}
    candidate, reference = map(values, (star, base))
    records = []
    means = {}
    for task in TASKS:
        for route in ROUTES:
            deltas = {}
            for view in VIEWS:
                key = task, route, view
                u, v = candidate[key], reference[key]
                if u["count"] != v["count"]:
                    raise ValueError("Paired sample counts differ")
                deltas[view] = 100 * (u["accuracy"] - v["accuracy"])
                records.append(dict(task=task, route=route, view=view, count=u["count"],
                                    candidate_accuracy=u["accuracy"], baseline_accuracy=v["accuracy"],
                                    candidate_macro_f1=u["macro_f1"], baseline_macro_f1=v["macro_f1"],
                                    accuracy_delta_pp=deltas[view]))
            means[(task, route)] = statistics.mean(deltas[v] for v in VIEWS[1:])
    with (root / "paired_scores.csv").open("x", newline="", encoding="utf-8") as stream:
        table = csv.DictWriter(stream, fieldnames=list(records[0]))
        table.writeheader()
        table.writerows(records)
    delta = {(r["task"], r["route"], r["view"]): r["accuracy_delta_pp"] for r in records}
    clean = delta["cross_rx", "raw", "clean"]
    low = delta["cross_rx", "raw", "practical_low_urban"]
    raw, dura = (means["cross_rx", route] for route in ROUTES)
    write(root / "complete.json", dict(status="ANALYZED", seed=SEED, truth_last=True,
        cross_rx=dict(clean_delta_pp=clean, low_urban_delta_pp=low,
                      six_practical_raw_delta_pp=raw, six_practical_DURA_delta_pp=dura),
        exploratory_threshold_met=clean >= -.5 and low >= 2 and raw >= 0 and dura >= -.5))


def dispatch(rt):
    from experiments.pamf_repl_completion.runner import capacity, foreign_CPU_score, processes

    root = Path(ROOT)
    root.mkdir(exist_ok=False)
    (root / "logs").mkdir()
    state = dict(status="PREPARING", pid=os.getpid(), run_id=RUN, code_commit=rt["code_commit"], stages={})
    state_file = root / "pipeline_state.json"
    atomic(state_file, state)
    stage = "prepare"
    try:
        prepare(rt)
        for stage in ("smoke", "train", "predict", "score"):
            gpu = None
            state.update(status="QUEUED", stage=stage)
            atomic(state_file, state)
            while stage != "score":
                choices = []
                pending = processes()
                for index in GPUS:
                    cuda, free = capacity(index)
                    if free >= MIN_MIB:
                        reserved = cuda | {pid for pid, g in pending.items() if g == index}
                        choices.append((free, -len(reserved), -GPUS.index(index), index))
                if choices:
                    gpu = max(choices)[3]
                    break
                time.sleep(15)
            while stage == "score" and foreign_CPU_score(set()):
                time.sleep(15)
            conf = root / "configs" / ("train.json" if stage == "smoke" else f"{stage}.json")
            argv = [PYTHON, "-u", "-m", MODULE, stage, "--config", str(conf)]
            env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(RELEASE), str(RELEASE / "code"))),
                       CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            log = root / "logs" / f"{stage}.log"
            with log.open("x") as output:
                p = subprocess.Popen(argv, cwd=RELEASE, env=env, stdin=subprocess.DEVNULL,
                                     stdout=output, stderr=subprocess.STDOUT)
            state["stages"][stage] = dict(pid=p.pid, gpu=gpu, argv=argv, cwd=str(RELEASE), log=str(log))
            state.update(status="RUNNING", stage=stage)
            atomic(state_file, state)
            status = p.wait()
            if status:
                raise RuntimeError(f"{stage} failed with exit code {status}; see {log}")
            state["stages"][stage]["exit_code"] = status
        stage = "analysis"
        analyze(root)
    except Exception as exc:
        atomic(state_file, dict(state, status="FAILED", stage=stage, error=str(exc)))
        raise
    atomic(state_file, dict(state, status="ANALYZED", stage="complete"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("dispatch", "smoke", "train", "predict", "score"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.action == "dispatch":
        dispatch(read(args.config))
    else:
        worker(args.action, args.config)
