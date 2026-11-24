"""Repeat the frozen six-source DURA lineage and its original matched test."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RUN = "20261124-phase1-pamf-dura-six-source-retrain-s2026092701-r01"
G_RUN = "20261021-phase1-T1-DURA-G-lineage-m35-r01"
G_ROW = "m6-natural-G-LU-s2026092701"
TRAIN_RELEASE = "20261022-phase1-DURA-teacher-cache-control-r01"
TRAIN_COMMIT = "34d1357614be0d813cc1e68d8526612d3abdfcb8"
PAPER = "20261027-phase1-pamf-paper-completion-m119-r03"
PAPER_COMMIT = "c9ebb6db301815159710b4c5cc461c9fd808dc55"
SEEDS = (2026092701, 2026092702, 2026092703)
TASKS = ("cross_rx", "cross_both")
READOUTS = ("raw", "DURA-legacy", "DURA-retrained")
STAGES = ("coarse", "action", "r17", "r38", "r44", "r46", "r65", "r69")


def paths(rt):
    project = Path(rt["project"])
    return project/"runs"/G_RUN/G_ROW/"source", project/"releases"/f"{PAPER}-release"


def validate_training(config, original):
    expected = dict(original, run_id=RUN, row_id=G_ROW, output_root=config["output_root"])
    if (config != expected or original["code_commit"] != TRAIN_COMMIT
            or config["model_seed"] != SEEDS[0] or config["cache"]["roles"] != "LU"
            or Path(config["output_root"]).resolve() == Path(original["output_root"]).resolve()
            or any(k in config for k in ("checkpoint", "resume", "F_checkpoint", "truth", "tests"))):
        raise ValueError("Only a fresh exact-rereun source config is allowed")


def prepare(rt):
    from experiments.cric_dura_comparison.io import read, write
    from experiments.cric_dura_comparison.evaluation import handle, VIEWS

    project = Path(rt["project"])
    root = Path(rt["run_root"])
    oldg, release = paths(rt)
    old = read(oldg/"config.json")
    audit = read(Path(rt["release_root"])/"pretraining_audit.json")
    if (not audit["passed"] or audit["training_commit"] != old["code_commit"]
            or audit["stdout_records"] != 160 or audit["stdout_abnormal_marker_lines"]):
        raise ValueError("Historical source audit failed")
    ctx = read(release/"context.json")
    h = ctx["handles"]["m6-natural"]
    if read(h["source_contract"]) != read(old["cache"]["source_contract"]):
        raise ValueError("Original G and frozen F physical contracts differ")
    contract = read(h["source_contract"])
    if contract["source_rxs"] != [1,3,4,6,8,11] or contract["counts"] != {
            "L_s":7560, "U_s":68040, "V":32400}:
        raise ValueError("Six-source natural-budget contract required")
    original_rows = read(project/"runs"/PAPER/"sources.json")
    jobs = []

    def add(key, kind, config, artifact, status):
        write(root/f"{key}.json", config)
        jobs.append(dict(key=key, kind=kind, completion=str(artifact), expected=status))

    train_output, restored = root/"source", root/"restored"
    new = dict(old, run_id=RUN, row_id=G_ROW, output_root=str(train_output))
    validate_training(new, old)
    g = dict(ctx["caches"]["m6-natural"]["LU"]["G"], checkpoint=str(train_output/"r69"/"final.pt"))
    view_paths = {task:dict(views_root=h["tests"][task]["views_root"]) for task in TASKS}
    for task, count in zip(TASKS, (108000,36000)):
        meta, ids, _ = handle(view_paths[task]["views_root"], h["classes"])
        if len(ids) != count or meta["target_rxs"] != [0,2,5,7,9,10]:
            raise ValueError("Common six-receiver test population differs")
    common = dict(protocol_id="m6-natural", source_contract=h["source_contract"], frozen_sources=old["frozen_sources"])
    write(root/"smoke.json", dict(original_g=ctx["caches"]["m6-natural"]["LU"]["G"],
                                 original_f=original_rows[f"source_rx_6-factorial-ssl-full-permuted-s{SEEDS[0]}"],
                                 cache=old["cache"], verdict=str(root/"smoke_result.json"), **common))
    add("train","train",new,train_output/"complete.json","SOURCE_FROZEN")
    add("restore","restore",dict(run_id=RUN,row_id=G_ROW,readout="DURA-LU", G=g, tests=view_paths,
        source_cache=h["source_cache"], output_root=str(restored),source_output=str(train_output),**common),
        restored/"complete.json","DURA_IQ_CACHE_COMPLETE")
    scorers = []
    for seed in SEEDS:
        key = f"m6-natural-s{seed}"
        old_key = f"source_rx_6-factorial-ssl-full-permuted-s{seed}"
        entry = original_rows[old_key]
        identity = read(Path(entry["source_output"])/"complete.json")
        if (identity["status"]!="SOURCE_FROZEN" or identity["target_access"] or identity["U_truth_read"]
                or identity["epoch"]!=50 or identity["updates"]!=11100
                or identity["config"]["model_seed"]!=seed or identity["config"]["row_id"]!=old_key
                or read(identity["config"]["source_contract"])!=contract):
            raise ValueError("Frozen original paper F differs")
        run_row = root/"readouts"/key
        p = dict(run_id=RUN,row_id=key,model_seed=seed,paper_row=entry["row"],
                 source_output=entry["source_output"], restored_caches={"LU":str(restored)},
                 original_predictions=entry["predictions"], original_row=old_key,
                 output_root=str(run_row/"new_decisions"), comparison_output=str(run_row/"predictions"),
                 tests=view_paths,views=list(VIEWS),classes=h["classes"],readouts=["raw","DURA-LU"],**common)
        add(f"predict-{seed}","predict",p,run_row/"predictions"/"complete.json","ALL_PREDICTIONS_COMPLETE")
        s = dict(run_id=RUN,row_id=key,model_seed=seed,protocol_id="m6-natural",classes=h["classes"],
                 source_days=contract["source_days"],views=list(VIEWS),readouts=list(READOUTS),
                 tests={task:h["tests"][task] for task in TASKS},prediction_output=p["comparison_output"],
                 output_root=str(run_row/"scores"))
        scorers.append((f"score-{seed}","score",s,run_row/"scores"/"complete.json","SCORED_COMPLETE"))
    for item in scorers:
        add(*item)
    return jobs


def activate_eval(c):
    from experiments.pamf_paper_completion.runner import activate
    activate(c)


def smoke(c):
    import numpy as np
    import torch

    activate_eval(c)
    from experiments.cric_dura_comparison.frozen_inference import load_G
    from experiments.cric_dura_comparison.g_cache import CacheView
    from experiments.cric_dura_comparison.io import read, tensor, write
    from experiments.pamf_paper_completion.model import load_reader
    from experiments.pamf_stage2_inference.engine import InferenceEngine

    contract = read(c["source_contract"])
    f = c["original_f"]
    reader,_ = load_reader(f["source_output"], contract, f["row"], "cuda:0")
    model = load_G(c["original_g"], contract, "cuda:0")
    view = CacheView(c["cache"])
    if not view.ready() or len(view)!=75600:
        raise ValueError("Source-only sample cache unavailable")
    sample = tensor(view.batch(np.array([0, len(view)-1]),1)["received"], "cuda:0")
    with torch.inference_mode():
        eager = model(sample)
        optimized = InferenceEngine(model,reuse=True,graph=True,native=False)(sample)[0]
        logits = reader.classify_features(reader.features(optimized))
    if logits.shape!=(2,6) or not bool(torch.isfinite(logits).all()) or not torch.allclose(
            eager,optimized,atol=1e-5,rtol=1e-5):
        raise ValueError("Source-only frozen G/F graph smoke failed")
    write(c["verdict"],dict(status="PASS",real_checkpoints=True,source_iq_samples=2,query_reads=0))
    print("SIX_SOURCE_DURA_NO_QUERY_SMOKE_PASS",flush=True)


def restore(c):
    from experiments.cric_dura_comparison import external_dura as native
    from experiments.cric_dura_comparison.io import read

    finished = read(Path(c["source_output"])/"complete.json")
    if (finished["status"]!="SOURCE_FROZEN" or finished["updates"]!=39520 or finished["stages"]!=list(STAGES)
            or finished["checkpoint"]!=c["G"]["checkpoint"]
            or any(finished[k]!=0 for k in ("TX_labels_read","F_reads","target_reads"))):
        raise ValueError("New G must be fully frozen before target prediction")
    activate_eval(c)
    native.tasks, native.CACHE = lambda _:list(TASKS), RUN
    native.cache(c)


def predict(c):
    import numpy as np
    from experiments.cric_dura_comparison.io import read, write
    from experiments.pamf_paper_completion.evaluation import predict as native
    from experiments.pamf_single_view_paper import evaluation as base

    activate_eval(c)
    base.TASKS,base.READOUTS = TASKS,("raw","DURA-LU")
    native(c)
    original = Path(c["original_predictions"])
    new = Path(c["output_root"])
    output = Path(c["comparison_output"])
    a,b = [read(p/"complete.json") for p in (original,new)]
    if (any(x["status"]!="ALL_PREDICTIONS_COMPLETE" or x["truth_read"] or x["query_fit"] for x in (a,b))
            or a["run_id"]!=PAPER or a["row_id"]!=c["original_row"] or b["row_id"]!=c["row_id"]
            or b["run_id"]!=RUN or a["views"]!=b["views"] or "DURA-LU" not in a["readouts"]):
        raise ValueError("Historical and new frozen predictions differ in identity or coverage")
    output.mkdir(exist_ok=False)
    diffs = []
    for task in TASKS:
        with np.load(original/task/"predictions.npz",allow_pickle=False) as old, np.load(
                new/task/"predictions.npz",allow_pickle=False) as fresh:
            if not np.array_equal(old["ids"],fresh["ids"]):
                raise ValueError("Prior predictions cover a different sample population")
            values = {"ids":fresh["ids"]}
            for view in c["views"]:
                raw = f"raw__{view}"
                g = f"DURA-LU__{view}"
                values.update({raw:fresh[raw],f"DURA-legacy__{view}":old[g],f"DURA-retrained__{view}":fresh[g]})
                diffs.append(dict(task=task,view=view,count=len(values["ids"]),
                                  raw_changes=int(np.count_nonzero(old[raw]!=fresh[raw]))))
            target = output/task
            target.mkdir()
            np.savez(target/"predictions.npz",**values)
    write(output/"raw_agreement.json",diffs)
    tests = {task:dict(b["tests"][task],readouts=list(READOUTS)) for task in TASKS}
    write(output/"complete.json",dict(b,readouts=list(READOUTS),tests=tests))


def score(c):
    from experiments.cric_dura_comparison import evaluation

    if set(c)&{"source_cache","source_output","frozen_sources","restored_caches","G"}:
        raise ValueError("Truth process must not receive model or source handles")
    evaluation.tasks = lambda _:list(TASKS)
    evaluation.score(c)


def analyze(root):
    from collections import defaultdict
    from statistics import mean
    from experiments.cric_dura_comparison.io import read, write

    rows = []
    averages = defaultdict(list)
    for seed in SEEDS:
        base = root/"readouts"/f"m6-natural-s{seed}"
        scored = read(base/"scores"/"scores.json")
        if not scored["truth_last"] or scored["model_seed"]!=seed:
            raise ValueError("Independent truth score missing")
        raw_changes = {(r["task"],r["view"]):r["raw_changes"] for r in
                       read(base/"predictions"/"raw_agreement.json")}
        groups = defaultdict(dict)
        for r in scored["results"]:
            groups[(r["test"],r["view"],r["dimension"],r["stratum"])][r["readout"]]=r
        paired = {(r["test"],r["view"],r["dimension"],r["stratum"],r["readout"]):r for r in scored["paired_raw"]}
        for key,methods in groups.items():
            task,view,dim,stratum = key
            raw,old,new = [methods[k] for k in READOUTS]
            p = paired[(*key,"DURA-retrained")]
            row = dict(seed=seed,test=task,view=view,dimension=dim,stratum=stratum,count=new["count"],
                       raw_accuracy=raw["accuracy"],legacy_accuracy=old["accuracy"],retrained_accuracy=new["accuracy"],
                       raw_macro_f1=raw["macro_f1"],legacy_macro_f1=old["macro_f1"],retrained_macro_f1=new["macro_f1"],
                       retrained_minus_legacy_pp=100*(new["accuracy"]-old["accuracy"]),
                       retrained_minus_raw_pp=100*(new["accuracy"]-raw["accuracy"]),
                       repaired=p["repairs"],damaged=p["damage"],net_vs_raw=p["net"])
            rows.append(row)
            if dim=="overall":
                averages[task,view].append(dict(row,raw_changes=raw_changes[task,view]))
    with (root/"paired_scores.csv").open("x",encoding="utf-8",newline="") as stream:
        output = csv.DictWriter(stream,fieldnames=list(rows[0]))
        output.writeheader()
        output.writerows(rows)
    summary = {task:{} for task in TASKS}
    metrics = ("raw_accuracy","legacy_accuracy","retrained_accuracy","retrained_minus_legacy_pp",
               "retrained_minus_raw_pp","raw_changes")
    for (task,view), values in averages.items():
        if len(values)!=len(SEEDS):
            raise ValueError("Every view must contain all three F seeds")
        summary[task][view] = {k:mean(r[k] for r in values) for k in metrics}
    for task, values in summary.items():
        values["six_practical_mean"] = {k:mean(row[k] for view,row in values.items() if view!="clean") for k in metrics}
    write(root/"summary.json",summary)


def dispatch(rt):
    from experiments.cric_dura_comparison.io import read
    from experiments.practical_full_2x2.cache import atomic

    root = Path(rt["run_root"])
    root.mkdir(exist_ok=False)
    (root/"logs").mkdir()
    state = dict(status="PREPARING",pid=os.getpid(),code_commit=rt["code_commit"],jobs={})
    status_file = root/"pipeline_state.json"
    atomic(status_file,state)
    jobs = prepare(rt)
    smoke_job = dict(key="smoke",kind="smoke",completion=str(root/"smoke_result.json"),expected="PASS")
    for job in [smoke_job,*jobs]:
        state["jobs"][job["key"]] = dict(status="QUEUED")
    atomic(status_file,state)

    def choose():
        output = subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],text=True)
        choices = [tuple(map(int,line.split(","))) for line in output.splitlines()]
        return max(choices,key=lambda x:x[1])

    for job in [smoke_job,*jobs]:
        kind,key = job["kind"],job["key"]
        gpu = None
        if kind!="score":
            state["status"] = "WAITING_FOR_GPU"
            atomic(status_file,state)
            while gpu is None:
                card,free = choose()
                if free>=rt["free_mib"]:
                    gpu = card
                else:
                    time.sleep(20)
        config = root/f"{key}.json"
        cmd = [rt["python"],"-u",__file__,kind,"--runtime",rt["runtime_file"],"--config",str(config)]
        env = dict(os.environ,CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
                   OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
        log = root/"logs"/f"{key}.log"
        with log.open("x") as stream:
            child = subprocess.Popen(cmd,cwd=rt["release_root"],env=env,stdin=subprocess.DEVNULL,
                                     stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            item = dict(status="RUNNING",pid=child.pid,gpu=gpu,argv=cmd,cwd=rt["release_root"],log=str(log))
            state["jobs"][key] = item
            state["status"] = "RUNNING"
            atomic(status_file,state)
            code = child.wait()
        artifact = Path(job["completion"])
        ok = not code and artifact.is_file() and read(artifact)["status"]==job["expected"]
        item.update(status="COMPLETE" if ok else "FAILED",exit_code=code)
        state["status"] = "RUNNING" if ok else "PARTIAL_TECHNICAL_FAILURE"
        atomic(status_file,state)
        if not ok:
            raise RuntimeError(f"{key} did not produce its registered artifact")
    analyze(root)
    state["status"] = "ANALYZED"
    atomic(status_file,state)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind",choices=("dispatch","smoke","train","restore","predict","score"))
    parser.add_argument("--runtime",required=True,type=Path)
    parser.add_argument("--config",type=Path)
    args = parser.parse_args()
    runtime = json.loads(args.runtime.read_text())
    old_root,paper_release = paths(runtime)
    if args.kind=="train":
        sys.path.insert(0,str(Path(runtime["project"])/"releases"/TRAIN_RELEASE))
        from experiments.cric_dura_comparison.io import read
        from experiments.cric_dura_comparison.pipeline import worker

        c = read(args.config)
        validate_training(c,read(old_root/"config.json"))
        worker(c)
    else:
        sys.path.insert(0,str(paper_release))
        from experiments.cric_dura_comparison.io import read

        dispatch(runtime) if args.kind=="dispatch" else globals()[args.kind](read(args.config))
