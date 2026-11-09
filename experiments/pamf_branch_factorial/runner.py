"""One owner for source-only training, raw frozen prediction and truth-last scoring."""
import argparse
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import re
import subprocess
import time

from experiments.cric_dura_comparison.io import read, write
from .method import RUN, rows, configuration, TASKS, READOUTS, SPECS


def resource_sample():
    line = Path("/proc/stat").read_text().splitlines()[0].split()
    if line[0] != "cpu" or len(line) < 6:
        raise ValueError("Missing CPU accounting")
    ticks = [int(v) for v in line[1:]]
    mem = next((int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")), None)
    if mem is None:
        raise ValueError("Missing available memory")
    return dict(total=sum(ticks), idle=ticks[3], iowait=ticks[4], available_kib=mem)


def resource_guard(previous, current):
    delta = current["total"] - previous["total"]
    if delta <= 0:
        return dict(allowed=False, reason="insufficient CPU accounting")
    idle = current["idle"] - previous["idle"]
    wait = current["iowait"] - previous["iowait"]
    busy = 100. * (delta - idle - wait) / delta
    io = 100. * wait / delta
    return dict(allowed=busy < 75. and io < 5. and current["available_kib"] >= 64 * 1024**2,
                cpu_busy_percent=busy, iowait_percent=io, available_kib=current["available_kib"])


def activate(rt):
    from experiments.pamf_repl_completion.runner import activate as start
    start(rt)


def prepare(rt):
    from experiments.practical_full_2x2.plan import VIEWS
    context = read(rt["context"])
    root = Path(rt["run_root"])
    jobs, sources = [], {}
    config_dir = root / "configs"
    config_dir.mkdir()

    def add(kind, row, config, completion, status, dependencies=()):
        path = config_dir / (kind + "__" + row["row_id"] + ".json")
        write(path, config)
        jobs.append(dict(id=kind + "__" + row["row_id"], kind=kind, config=str(path),
                         completion=str(completion), successful_statuses=[status], dependencies=list(dependencies)))

    contract = read(context["handle"]["source_contract"])
    for row in rows():
        rid = row["row_id"]
        if row["reuse"]:
            old = context["reuse"][rid]
            source, prediction, scores = (Path(old[k]) for k in ("source_output", "predictions", "scores"))
            done = read(source / "complete.json")
            config = done["config"]
            if (done["status"] != "SOURCE_FROZEN" or done["epoch"] != 50 or done["updates"] != 11100 or
                    done["target_access"] is not False or done["U_truth_read"] is not False or
                    done["head_fit_updates"] != 0 or config["model_seed"] != row["model_seed"] or
                    config["ablation"] != "F15" or config["U_use"] is not False or
                    config["satellite_concat"] is not True or config["satellite_CE_weight"] != .68 or
                    read(config["source_contract"]) != contract or not (source / "final_ssdg.pth").is_file()):
                raise ValueError("CHECKPOINT_DATA_CONTRACT_MISMATCH: baseline")
            expected = {"base": "NO_REFERENCE", "no_signature": "NS_NO_REFERENCE",
                        "no_signature_no_pa": "NS_NO_PA_REFERENCE"}[row["arm"]]
            if config["repl_row"]["structure"] != expected:
                raise ValueError("Reused structural baseline differs")
            pd, sd = read(prediction / "complete.json"), read(scores / "complete.json")
            if (pd["status"] != "ALL_PREDICTIONS_COMPLETE" or pd["truth_read"] is not False or
                    sd["status"] != "SCORED_COMPLETE"):
                raise ValueError("Reused frozen prediction/score incomplete")
            sources[rid] = dict(row=row, source_output=str(source), predictions=str(prediction), scores=str(scores), reused=True)
            continue
        c = configuration(context, row, rt["project"], rt["code_commit"])
        c["frozen_sources"] = rt["frozen_sources"]
        source = Path(c["output_root"])
        add("train", row, c, source / "complete.json", "SOURCE_FROZEN")
        out = root / "readouts" / rid
        h = context["handle"]
        predictor = dict(run_id=RUN, row_id=rid, protocol_id="F0", model_seed=row["model_seed"],
            source_output=str(source), source_contract=h["source_contract"], classes=h["classes"],
            source_days=[1, 2, 3], tests={t: dict(views_root=h["tests"][t]["views_root"]) for t in TASKS},
            views=list(VIEWS), readouts=READOUTS, output_root=str(out / "predictions"),
            frozen_sources=rt["frozen_sources"])
        scorer = deepcopy(predictor)
        scorer.update(prediction_output=predictor["output_root"], output_root=str(out / "scores"),
                      tests={t: h["tests"][t] for t in TASKS})
        for key in ("source_output", "frozen_sources"):
            scorer.pop(key)
        add("predict", row, predictor, out / "predictions/complete.json", "ALL_PREDICTIONS_COMPLETE", [str(source / "complete.json")])
        add("score", row, scorer, out / "scores/complete.json", "SCORED_COMPLETE", [str(out / "predictions/complete.json")])
        sources[rid] = dict(row=row, source_output=str(source), predictions=str(out / "predictions"), scores=str(out / "scores"), reused=False)
    if len(sources) != 36 or len(jobs) != 81 or sum(v["reused"] for v in sources.values()) != 9:
        raise ValueError("Expected 36 logical, 9 reused and 81 stage jobs")
    write(root / "jobs.json", jobs)
    write(root / "sources.json", sources)
    return jobs


def smoke(rt):
    import numpy as np
    import torch
    from experiments.cric_dura_comparison.io import tensor
    from experiments.cric_dura_comparison.f_train import arguments, class_dimension, original_config
    from experiments.cvs_phase1_stack.runtime import installed as native_scope
    from experiments.pamf_repl_completion.objectives import installed
    from experiments.cric_dura_comparison import t1_model
    from .method import architecture_scope, load_reader
    context = read(rt["context"])
    contract = read(context["handle"]["source_contract"])
    source = context["reuse"]["base-s2026092701"]["source_output"]
    reader, _ = load_reader(source, contract, "cuda:0")
    x = tensor(np.ascontiguousarray(np.load(Path(context["handle"]["source_cache"]) / "L_s_x.npy", mmap_mode="r")[:2], np.float32), "cuda:0")
    with torch.no_grad():
        result = reader.classify_features(reader.features(x))
        if result.shape != (2, 6) or not bool(torch.isfinite(result).all()):
            raise ValueError("Actual matching checkpoint no-query smoke failed")
    del reader
    checks = []
    for arm in SPECS:
        row = next(r for r in rows() if r["arm"] == arm and r["model_seed"] == 2026092701)
        c = configuration(context, row, rt["project"], rt["code_commit"])
        torch.manual_seed(c["model_seed"])
        with architecture_scope(), installed(c, {}, False), class_dimension(c), native_scope(original_config(c["model_seed"])) as native:
            args = arguments(c, "cuda:0")
            args.input_len = 256
            net = t1_model.install(native.build_baseline_model(args, "cuda:0"), c).to("cuda:0")
        net.eval()
        logits = net(x)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1], device="cuda:0"))
        loss.backward()
        if logits.shape != (2, 6) or not bool(torch.isfinite(loss)) or not all(
                bool(torch.isfinite(p.grad).all()) for p in net.parameters() if p.grad is not None):
            raise FloatingPointError("Branch graph source smoke failed")
        checks.append(dict(arm=arm, finite_backward=True))
        del net, logits, loss
    write(Path(rt["run_root"]) / "smoke.json", dict(status="PASS", source_samples=2, target_reads=0,
          U_truth_reads=0, formal_updates=0, checkpoint=source, checks=checks))


def worker(c, kind):
    if kind == "score":
        from .evaluation import score
        return score(c)
    activate(c)
    if kind == "smoke":
        return smoke(c)
    if kind == "train":
        from .method import train
        return train(c)
    if kind == "predict":
        from .evaluation import predict
        return predict(c)
    raise ValueError("Unknown worker kind")


def dispatch(rt):
    from experiments.pamf_repl_completion.runner import processes, capacity, foreign_CPU_score
    from experiments.cric_dura_comparison.pipeline import done, ready, block_failed_dependencies
    from experiments.practical_full_2x2.cache import atomic
    root = Path(rt["run_root"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    state = dict(status="PREPARING", pid=os.getpid(), cwd=os.getcwd(), code_commit=rt["code_commit"],
                 owner=rt["launch_owner"], jobs={})
    atomic(root / "pipeline_state.json", state)
    try:
        jobs = prepare(rt)
    except Exception as error:
        state.update(status="FAILED_PREPARATION", error=repr(error), ended_at=time.time())
        atomic(root / "pipeline_state.json", state)
        raise
    by = {j["id"]: j for j in jobs}
    active, failures = {}, Counter()
    state.update(status="SMOKE_PENDING", jobs={j["id"]: dict(status="QUEUED") for j in jobs})
    atomic(root / "pipeline_state.json", state)

    def launch(job, gpu):
        path = root / "logs" / (job["id"] + ".log")
        cmd = [rt["python"], "-u", "-m", "experiments.pamf_branch_factorial.runner", "worker",
               "--kind", job["kind"], "--config", job["config"]]
        env = dict(os.environ, PYTHONPATH=rt["release_root"], CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        with path.open("x") as log:
            child = subprocess.Popen(cmd, cwd=rt["release_root"], env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return child, dict(status="RUNNING", pid=child.pid, gpu=gpu, cwd=rt["release_root"], argv=cmd,
                           log=str(path), started_at=time.time())

    previous = resource_sample()
    time.sleep(1)
    while True:
        current = resource_sample()
        guard = resource_guard(previous, current)
        previous = current
        procs = processes()
        gpu = next((g for g in rt["allowed_gpus"] if guard["allowed"] and len(capacity(g)[0] | {p for p, v in procs.items() if v == g}) < 2
                    and capacity(g)[1] >= 12000), None)
        if gpu is not None:
            break
        time.sleep(15)
    child, item = launch(dict(id="smoke", kind="smoke", config=rt["runtime_file"]), gpu)
    state["smoke"] = item
    atomic(root / "pipeline_state.json", state)
    status = child.wait()
    state["smoke"].update(status="FAILED" if status else "PASS", exit_code=status, ended_at=time.time())
    if status or read(root / "smoke.json")["status"] != "PASS":
        state["status"] = "FAILED_NO_QUERY_SMOKE"
        atomic(root / "pipeline_state.json", state)
        return
    state["status"] = "RUNNING"
    turn = 0
    while True:
        current = resource_sample()
        guard = resource_guard(previous, current)
        previous = current
        for key, (child, item) in list(active.items()):
            if child.poll() is None:
                continue
            ok = done(by[key])
            state["jobs"][key] = dict(item, status="COMPLETE" if ok else "FAILED", ended_at=time.time())
            if not ok:
                lines = Path(item["log"]).read_text(errors="replace").splitlines()
                reason = re.sub(r"s202609270[123]", "sSEED", lines[-1] if lines else "missing completion")
                failures[reason] += 1
                state["jobs"][key]["failure"] = reason
            del active[key]
        block_failed_dependencies(jobs, state, [])
        systemic = any(n >= 2 or any(t in reason for t in ("CONTRACT_MISMATCH", "PROVENANCE", "Truth closed", "U truth", "Output collision"))
                       for reason, n in failures.items())
        pending = [j for j in jobs if state["jobs"][j["id"]]["status"] == "QUEUED"]
        candidates = [j for j in pending if ready(j)] if not systemic and guard["allowed"] else []
        score = next((j for j in candidates if j["kind"] == "score"), None)
        if score and not any(by[k]["kind"] == "score" for k in active) and not foreign_CPU_score({p.pid for p, _ in active.values()}):
            child, item = launch(score, None)
            active[score["id"]] = (child, item)
            state["jobs"][score["id"]] = item
            candidates.remove(score)
        procs = processes()
        for gpu in rt["allowed_gpus"]:
            pids, free = capacity(gpu)
            pids |= {pid for pid, g in procs.items() if g == gpu}
            own = sum(item["gpu"] == gpu for _, item in active.values())
            if len(pids) >= 2 or own >= 2 or free < 12000:
                continue
            usable = [j for j in candidates if j["kind"] != "score"]
            if not usable:
                continue
            preferred = "predict" if turn % 3 == 2 else "train"
            job = next((j for j in usable if j["kind"] == preferred), usable[0])
            turn += 1
            child, item = launch(job, gpu)
            active[job["id"]] = (child, item)
            state["jobs"][job["id"]] = item
            candidates.remove(job)
            procs[child.pid] = gpu
        state.update(updated_at=time.time(), counts=dict(Counter(v["status"] for v in state["jobs"].values())),
                     failures=dict(failures), systemic_stop_new_launches=systemic, resource_guard=guard)
        atomic(root / "pipeline_state.json", state)
        if not active and (not pending or systemic):
            break
        time.sleep(15)
    if all(v["status"] == "COMPLETE" for v in state["jobs"].values()):
        from .analyze import run
        run(rt)
        state["status"] = "ANALYZED"
    else:
        state["status"] = "PARTIAL_TECHNICAL_FAILURE"
    state["ended_at"] = time.time()
    atomic(root / "pipeline_state.json", state)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["dispatch", "worker"])
    p.add_argument("--config", required=True)
    p.add_argument("--kind")
    a = p.parse_args()
    config = read(a.config)
    dispatch(config) if a.action == "dispatch" else worker(config, a.kind)
