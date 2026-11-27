"""Schedule the comparison matrix using frozen source/test handles."""
import argparse
from collections import Counter
from pathlib import Path
import os
import subprocess
import time

from experiments.cric_dura_comparison.io import read, write
from experiments.practical_full_2x2.cache import atomic
from experiments.practical_full_2x2.plan import TASKS, VIEWS
from .plan import (ADVISOR, CONTEXT, GPUS, GPU_WORKERS, MODULE, NEW_METHODS, PROJECT,
                   PYTHON, RELEASE, ROOT, RUN, SHEN, STAR, V, row_name, settings)


def prepare(commit, root):
    ctx = read(CONTEXT)
    runs = PROJECT / "runs"
    jobs = []

    def add(stage, name, method, config, deps=()):
        key = f"{stage}__{name}"
        config_path = root / "configs" / f"{key}.json"
        write(config_path, config)
        jobs.append(dict(key=key, name=name, stage=stage, method=method, config=str(config_path),
                         dependencies=list(deps)))
        return key

    for key, spec in settings().items():
        paper = spec["kind"] == "paper"
        if paper:
            h = ctx["handles"][key]
            folder = h["source_cache"]
            tests = h["tests"]
        else:
            folder = str(runs / ADVISOR / "data" / key)
            base = runs / ADVISOR / "configs" / f"score__{key}-ssl-s{spec['model_seed']}.json"
            tests = read(base)["tests"]
        source_contract = str(Path(folder) / "source_contract.json")
        contract = read(source_contract)
        expected = {k: contract[k] for k in ("source_rxs", "source_days", "counts", "classes")}
        if not paper and any(expected[k] != spec[k] for k in ("source_rxs", "source_days", "counts")):
            raise ValueError(f"Supplement source contract differs: {key}")
        tasks = TASKS if paper else V
        cache = str(root / "pairs" / key)
        cache_job = None
        if key != "F0":
            cache_job = add("cache", key, "star", dict(source_cache=folder, expected=expected, output=cache))
        for seed in spec["seeds"]:
            for method in NEW_METHODS:
                if key == "F0" and method == "star":
                    continue
                name = row_name(key, method, seed)
                base = root / "readouts" / name
                src = root / "F" / name / "source"
                common = dict(run_id=RUN, row_id=name, model_seed=seed)
                if key == "F0":
                    past = runs / SHEN / f"s{seed}"
                    src = Path(past / "source")
                    source_id = dict(run_id=SHEN, row_id=f"shen-wisig-main-s{seed}")
                    reuse = dict(source_id, path=str(past / "predictions"))
                    deps = ()
                else:
                    source_id = {k: common[k] for k in ("run_id", "row_id")}
                    reuse = None
                    if method == "shen":
                        train = dict(common, code_commit=commit, source_cache=folder,
                                     source_contract=source_contract, output_root=str(src), recipe="paper_pdp_half")
                        deps = (add("train", name, method, train),)
                    else:
                        train = dict(common, code_commit=commit, source_cache=folder, pairs=cache,
                                     source_dimensions=expected, source_output=str(src))
                        deps = (add("train", name, method, train, (cache_job,)),)
                pred = base / "predictions"
                shared = dict(common, protocol_id=key, classes=expected["classes"], source_days=expected["source_days"],
                              readouts=["raw", "DURA-LU"] if paper else ["raw"], views=list(VIEWS))
                px = dict(shared, source_output=str(src), source_contract=source_contract,
                          output_root=str(pred), frozen_source=source_id, reuse_raw=reuse,
                          tests={t: {"views_root": tests[t]["views_root"]} for t in tasks})
                if paper:
                    px.update(G_roles="LU", restored_cache=ctx["caches"][key]["LU"]["path"])
                dep = add("predict", name, method, px, deps)
                sx = dict(shared, output_root=str(base / "scores"), prediction_output=str(pred),
                          tests={t: tests[t] for t in tasks})
                add("score", name, method, sx, (dep,))
    write(root / "jobs.json", jobs)
    return jobs


def star_reader(folder, contract, seed, identity, device):
    import torch
    from experiments.star_pamf_practical.training import new_model

    folder = Path(folder)
    final = read(folder / "complete.json")
    c = read(folder / "resolved_config.json")
    if (final["status"] != "SOURCE_TRAINED_FROZEN" or final["target_access"] or final["U_TX_truth_read"]
            or read(Path(c["source_cache"]) / "source_contract.json") != contract):
        raise ValueError("STAR frozen model must match the recorded exact source contract")
    d = torch.load(final["checkpoint"], map_location=device, weights_only=True)
    if (d["epoch"], d["model_seed"], d["run_id"], d["row_id"]) != (100, seed, identity["run_id"], identity["row_id"]):
        raise ValueError("Unexpected STAR model identity")
    model = new_model(device, len(contract["classes"])).eval()
    model.load_state_dict(d["state_dict"], strict=True)
    return model


def shen_reader(folder, contract, seed, identity, device):
    from experiments.shen_wisig.runner import load_reader
    model, d = load_reader(folder, contract, device)
    if (d["recipe"], d["epoch"], d["model_seed"], d["run_id"], d["row_id"]) != (
            "paper_pdp_half", 50, seed, identity["run_id"], identity["row_id"]):
        raise ValueError("Unexpected Receiver-Agnostic RFFI r04 identity")
    return model


def smoke():
    import torch
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.star_pamf_practical.training import forward

    h = read(CONTEXT)["handles"]["F0"]
    contract = read(h["source_contract"])
    x = SourceDataset(h["source_cache"], "L_s", contract).x[:2].to("cuda:0")
    runs = PROJECT / "runs"
    seed = 2026092701
    star = star_reader(runs / STAR / "F" / f"main-star-s{seed}" / "source", contract, seed,
                       dict(run_id=STAR, row_id=f"main-star-s{seed}"), "cuda:0")
    shen = shen_reader(runs / SHEN / f"s{seed}" / "source", contract, seed,
                       dict(run_id=SHEN, row_id=f"shen-wisig-main-s{seed}"), "cuda:0")
    with torch.inference_mode():
        values = (forward(star, x, "clean")[1], forward(star, x, "sg")[1],
                  shen.classify_features(shen.features(x)))
        if any(y.shape != (2, 6) or not bool(y.isfinite().all()) for y in values):
            raise ValueError("The two frozen model source-only readers failed")
    write(Path(ROOT) / "smoke.json", dict(status="PASS", checkpoint_readers=2, source_IQ_batches=1,
                                        truth_reads=0, target_reads=0))
    print("TWO_REAL_CHECKPOINTS_NO_QUERY_PASS", flush=True)


def worker(stage, method, c):
    if stage == "cache":
        from experiments.star_pamf_practical.data import prepare_pairs
        return prepare_pairs(c["source_cache"], c["output"], 8, expected=c["expected"])
    from experiments.practical_full_2x2.compat import numerical_policy
    numerical_policy()
    if stage == "smoke":
        return smoke()
    if stage == "train":
        if method == "shen":
            from experiments.shen_wisig.runner import train
            return train(c, "cuda:0")
        from experiments.star_pamf_practical.training import train
        return train(c)
    from experiments.practical_full_2x2 import evaluation
    tasks = tuple(c["tests"])
    if stage == "score":
        return evaluation.score(c, task_names=tasks)
    evaluation.reject_truth(c)
    contract = read(c["source_contract"])
    if method == "shen":
        model = shen_reader(c["source_output"], contract, c["model_seed"], c["frozen_source"], "cuda:0")
        classify = lambda x, _: model.classify_features(model.features(x))
    else:
        from experiments.star_pamf_practical.training import forward
        model = star_reader(c["source_output"], contract, c["model_seed"], c["frozen_source"], "cuda:0")
        classify = lambda x, view: forward(model, x, "clean" if view == "clean" else "sg")[1]
    return evaluation.predict(c, classify_view=classify, task_names=tasks, reuse_raw=c["reuse_raw"])


def spawn(root, job, gpu, *, release=RELEASE, module=MODULE):
    log = root / "logs" / (job["key"] + ".log")
    cmd = [PYTHON, "-u", "-m", module, job["stage"], "--method", job["method"], "--config", job["config"]]
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID",
               CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
               PYTHONPATH=f"{release}:{release / 'code'}", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2")
    with log.open("x") as out:
        p = subprocess.Popen(cmd, cwd=release, env=env, stdout=out, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL)
    print("MATRIX_LAUNCH", job["key"], p.pid, gpu, flush=True)
    return p, dict(stage=job["stage"], method=job["method"], pid=p.pid, gpu=gpu, argv=cmd,
                   cwd=str(release), log=str(log))


def dispatch(runtime):
    from experiments.star_pamf_practical.runner import resources
    from .results import summarize

    root = Path(ROOT)
    root.mkdir(exist_ok=False)
    (root / "logs").mkdir()
    (root / "pairs").mkdir()
    status = root / "pipeline_state.json"
    state = dict(status="PREPARING", controller_pid=os.getpid(), code_commit=runtime["code_commit"])
    atomic(status, state)
    try:
        jobs = prepare(runtime["code_commit"], root)
        free = resources()
        gpu = next((g for g in GPUS if free.get(g, 0) >= 11000), None)
        while gpu is None:
            atomic(status, dict(state, status="WAITING_SMOKE_GPU", gpu_free=free))
            time.sleep(15)
            free = resources()
            gpu = next((g for g in GPUS if free.get(g, 0) >= 11000), None)
        probe = dict(key="smoke", stage="smoke", method="both", config=str(RELEASE / "runtime.json"))
        p, smoke_info = spawn(root, probe, gpu)
        atomic(status, dict(state, status="SMOKE_RUNNING", active={"smoke": smoke_info}))
        if p.wait() or read(root / "smoke.json")["status"] != "PASS":
            raise ValueError("No-query real-checkpoint smoke failed")
        active = {}
        done = {}
        failures = Counter()
        for_main = {j["key"] for j in jobs if j["name"].startswith("F0-")}
        while len(done) < len(jobs):
            for key, (p, info) in list(active.items()):
                if (code := p.poll()) is None:
                    continue
                done[key] = dict(info, status="FAILED" if code else "COMPLETE", exit_code=code)
                if code:
                    failures[info["method"], info["stage"]] += 1
                del active[key]
            if for_main <= done.keys() and not (root / "main_snapshot").exists():
                summarize(root, main_only=True)
            free = resources()
            busy = {info["gpu"] for _, info in active.values() if info["gpu"] is not None}
            cpu_stages = {info["stage"] for _, info in active.values() if info["gpu"] is None}
            priority = {"score": 0, "predict": 1, "cache": 2, "train": 3}
            for job in sorted(jobs, key=lambda j: priority[j["stage"]]):
                key, stage, method = job["key"], job["stage"], job["method"]
                deps = job["dependencies"]
                if key in active or key in done:
                    continue
                if any(dep in done and done[dep]["status"] != "COMPLETE" for dep in deps) or failures[method, stage] >= 2:
                    done[key] = dict(status="BLOCKED", stage=stage, method=method)
                    continue
                if not all(dep in done for dep in deps):
                    continue
                gpu = None
                if stage in ("cache", "score"):
                    if stage in cpu_stages:
                        continue
                    cpu_stages.add(stage)
                else:
                    if len(busy) >= GPU_WORKERS:
                        continue
                    minimum = 12000 if stage == "train" and method == "star" else 7000
                    gpu = next((g for g in GPUS if g not in busy and free.get(g, 0) >= minimum), None)
                    if gpu is None:
                        continue
                    busy.add(gpu)
                active[key] = spawn(root, job, gpu)
            atomic(status, dict(state, status="RUNNING", smoke=smoke_info,
                                active={key: info for key, (_, info) in active.items()}, finished=done,
                                counts=dict(Counter(v["status"] for v in done.values())), total_jobs=len(jobs)))
            if len(done) < len(jobs):
                time.sleep(12)
        summary = summarize(root)
        atomic(status, dict(state, status=summary["status"], finished=done, summary=summary))
    except Exception as exc:
        atomic(status, dict(state, status="FAILED", error=str(exc)))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("dispatch", "smoke", "cache", "train", "predict", "score"))
    p.add_argument("--method")
    p.add_argument("--config", required=True)
    args = p.parse_args()
    if args.action == "dispatch":
        dispatch(read(args.config))
    else:
        worker(args.action, args.method, read(args.config))
