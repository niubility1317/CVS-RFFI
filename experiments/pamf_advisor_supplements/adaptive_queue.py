"""Run paired Adaptive Tri and PL on the exact advisor populations."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import os
import subprocess
import sys
import time

from adaptive_profiles import BASE, RUN, SOURCE_RELEASE, TASKS, matrix


def config_path(root, kind, row):
    return root/"configs"/f"{kind}__{row}.json"


def prepare(rt):
    from experiments.cric_dura_comparison.io import read, write

    root, old = Path(rt["run_root"]), Path(rt["prior_root"])
    live = read(old/"pipeline_state.json")["jobs"]
    rows = list(matrix())
    if [entry["data_overrides"] for entry in read(Path(rt["release_root"])/"experiment.json")["rows"]] != rows:
        raise ValueError("Registered paired matrix differs")
    jobs = []

    def add(kind, row_id, config, artifact, statuses, dependencies=()):
        path = config_path(root, kind, row_id)
        write(path, config)
        jobs.append(dict(id=f"{kind}__{row_id}", kind=kind, config=str(path),
                         completion=str(artifact), successful_statuses=statuses,
                         dependencies=list(map(str, dependencies))))

    for row in rows:
        key, setting = row["row_id"], row["setting"]
        seed = row["model_seed"]
        prior_method = "pl" if row["method"] == "pl" else "ssl"
        prior = f"{setting}-{prior_method}-s{seed}"
        prior_ssl = f"{setting}-ssl-s{seed}"
        source = old/"F"/prior/"source"
        previous = read(config_path(old,"train",prior_ssl))
        if row["reuse"]:
            old_scores = old/"readouts"/prior/"scores"
            prior_pred = read(config_path(old,"predict",prior))
            wanted = read(config_path(old,"predict",prior_ssl))
            if any(prior_pred[k] != wanted[k] for k in ("tests","views","classes","source_contract","readouts")):
                raise ValueError("Reused predictions are on a different target population")
            receipt = root/"reused"/f"{key}.json"
            cfg = dict(row=row, current_source=previous["source_contract"], checkpoint=str(source),
                       old_row=prior, old_scores=str(old_scores), receipt=str(receipt),
                       prediction_output=prior_pred["output_root"], views=prior_pred["views"])
            add("reuse", key, cfg, receipt, ["REUSED_VERIFIED"], (old_scores/"complete.json",))
            continue
        c = dict(previous)
        if row["method"] == "pl":
            template = read(config_path(old,"train",f"budget630-pl-s{seed}"))
            c = dict(template, **{k:previous[k] for k in ("source_contract","source_cache","source_rxs","source_days")})
        c.update(run_id=RUN, row_id=key, code_commit=rt["code_commit"], protocol_id=setting,
                 advisor_row=row, output_root=str(root/"F"/key/"source"),
                 paper_row=dict(c["paper_row"], row_id=key, protocol_id=setting),
                 algorithm_id="pamf_advisor_adaptive_profiles_v1" if row["method"]=="adaptive" else c["algorithm_id"])
        deps = []
        if row["method"]=="adaptive" and live[f"train__{prior_ssl}"]["status"] in ("RUNNING","COMPLETE"):
            prefix = old/"F"/prior_ssl/"source"/"segment20"
            c["shared_prefix"] = str(prefix)
            deps.append(prefix/"complete.json")
        add("train", key, c, Path(c["output_root"])/"complete.json", ["SOURCE_FROZEN"], deps)
        p = read(config_path(old,"predict",prior_ssl))
        out = root/"readouts"/key
        p.update(run_id=RUN,row_id=key,paper_row=c["paper_row"],source_output=c["output_root"],
                 output_root=str(out/"predictions"))
        add("predict",key,p,out/"predictions"/"complete.json",["ALL_PREDICTIONS_COMPLETE"],
            (Path(c["output_root"])/"complete.json",))
        s = read(config_path(old,"score",prior_ssl))
        s.update(run_id=RUN,row_id=key,prediction_output=p["output_root"],output_root=str(out/"scores"))
        add("score",key,s,out/"scores"/"complete.json",["SCORED_COMPLETE"],(out/"predictions"/"complete.json",))
    write(root/"jobs.json",jobs)
    return jobs


def reuse(c):
    from experiments.cric_dura_comparison.io import read, write
    from adaptive_profiles import count_coverage

    row = c["row"]
    d = read(Path(c["checkpoint"])/"complete.json")
    original = d["config"]
    done = read(Path(c["old_scores"])/"complete.json")
    metrics = read(Path(c["old_scores"])/"scores.json")
    close = read(Path(c["prediction_output"])/"complete.json")
    if (d["status"] != "SOURCE_FROZEN" or d["epoch"] != 50 or d["updates"] != 11100
            or d["target_access"] or d["U_truth_read"] or d["head_fit_updates"]
            or original["run_id"] != BASE or original["row_id"] != c["old_row"]
            or original["model_seed"] != row["model_seed"]
            or read(original["source_contract"]) != read(c["current_source"])
            or done["status"] != "SCORED_COMPLETE" or not metrics["truth_last"]
            or metrics["row_id"] != c["old_row"] or close["status"] != "ALL_PREDICTIONS_COMPLETE"
            or close["truth_read"] or close["query_fit"] or close["views"] != c["views"]):
        raise ValueError("Reused predictions or source contract do not match")
    is_tri = row["method"] == "adaptive"
    if (original["paper_row"]["permuted"] != is_tri or original["plain_PL_threshold"] != .95
            or original["plain_PL_weight"] != 1. or
            (is_tri and (count_coverage(original) < 14 or original["hard_pl"]["weight"] != 6.))):
        raise ValueError("Reused method differs from the registered PL or no-trigger fallback")
    write(c["receipt"],dict(status="REUSED_VERIFIED",scores=c["old_scores"],truth_last=True))


def smoke(rt):
    import torch
    from experiments.cric_dura_comparison.io import read, write
    from experiments.pamf_repl_completion.runner import activate
    from experiments.pamf_branch_training.source import checked_e20_parent
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.pamf_paper_completion.model import reader_scope
    from adaptive_profiles import count_coverage

    key = "budget630-adaptive-s2026092701"
    c = read(config_path(Path(rt["run_root"]), "train", key))
    activate(c)
    prefix = checked_e20_parent(c)
    with reader_scope(c["paper_row"]):
        payload = torch.load(Path(prefix["folder"])/"final_ssdg.pth",map_location="cpu",weights_only=False)
        reader = reader_from_payload(prefix["config"],payload,"cuda:0")
    x = SourceDataset(c["source_cache"],"L_s",read(c["source_contract"])).x[:2].cuda()
    with torch.inference_mode():
        logits = reader.classify_features(reader.features(x))
    if logits.shape != (2,6) or not bool(logits.isfinite().all()) or count_coverage(c) >= 14:
        raise ValueError("Real source checkpoint and label coverage smoke failed")
    write(Path(rt["run_root"])/"smoke.json",dict(status="PASS",real_checkpoint=True,query_reads=0))
    print("ADVISOR_ADAPTIVE_NO_QUERY_SMOKE_PASS",flush=True)


def worker(c, kind):
    if kind == "reuse":
        return reuse(c)
    if kind == "score":
        from experiments.cric_dura_comparison import evaluation as native
        native.tasks = lambda _:list(TASKS)
        return native.score(c)
    if kind == "smoke":
        return smoke(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    if kind == "train":
        from adaptive_profiles import train
        return train(c)
    if kind == "predict":
        from experiments.pamf_single_view_paper import evaluation as native
        from experiments.pamf_paper_completion.model import load_reader
        native.TASKS, native.READOUTS, native.load_reader = TASKS, ("raw",), load_reader
        return native.predict(c)
    raise ValueError(f"Unknown worker kind: {kind}")


def analysis(root):
    from experiments.cric_dura_comparison.io import read
    from experiments.cric_dura_comparison.evaluation import DATES
    results = {}
    for row in matrix():
        key = row["row_id"]
        path = (Path(read(root/"reused"/f"{key}.json")["scores"]) if row["reuse"]
                else root/"readouts"/key/"scores")
        score = read(path/"scores.json")
        if not score["truth_last"]:
            raise ValueError("Independent final scoring required")
        data = {}
        for record in score["results"]:
            endpoint = row["endpoint"]
            if endpoint is None and record["dimension"] != "overall":
                continue
            if endpoint is not None and (record["dimension"] != "date" or record["stratum"] != DATES[endpoint]):
                continue
            data[(record["test"],record["view"])] = record
        wanted = 14 if row["endpoint"] is None else 7
        if len(data) != wanted:
            raise ValueError("Seven views required per declared endpoint")
        results.setdefault(row["setting"],{})[row["method"]] = (row,data)
    rows = []
    for setting, pair in results.items():
        (row,ada),(_,pl) = pair["adaptive"],pair["pl"]
        if set(ada) != set(pl):
            raise ValueError("Unmatched target readout")
        for (task,view),a in ada.items():
            b = pl[task,view]
            if a["count"] != b["count"]:
                raise ValueError("Unmatched target population")
            rows.append(dict(setting=setting, target_partition=task, view=view,
                 endpoint=row["endpoint"], profile=row["profile"], early_anchor=row["early_anchor"], count=a["count"],
                 adaptive_accuracy=a["accuracy"], pl_accuracy=b["accuracy"], delta_pp=100*(a["accuracy"]-b["accuracy"]),
                 adaptive_macro_f1=a["macro_f1"], pl_macro_f1=b["macro_f1"]))
    with (root/"paired_results.csv").open("x",encoding="utf-8",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)


def free_memory(gpu):
    raw = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
                                   "--format=csv,noheader,nounits"], text=True).strip()
    return int(raw)


def launch(rt, job, gpu, script="adaptive_queue.py"):
    cmd = [rt["python"], "-u", str(Path(rt["release_root"])/script), "worker",
           "--runtime", rt["runtime_file"], "--kind", job["kind"], "--config", job["config"]]
    name = Path(rt["run_root"])/"logs"/f"{job['id']}.log"
    with name.open("x") as stream:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        process = subprocess.Popen(cmd, cwd=rt["release_root"], env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    return process, dict(status="RUNNING", pid=process.pid, gpu=gpu, argv=cmd, cwd=rt["release_root"], log=str(name))


def dispatch(rt):
    from experiments.cric_dura_comparison.io import read
    from experiments.pamf_repl_completion.runner import foreign_CPU_score
    from experiments.practical_full_2x2.cache import atomic

    root = Path(rt["run_root"])
    root.mkdir(exist_ok=False)
    (root/"logs").mkdir()
    jobs = {j["id"]:j for j in prepare(rt)}
    gates = {key:dict(status="QUEUED") for key in jobs}
    state = dict(status="PREPARING", pid=os.getpid(), jobs=gates, code_commit=rt["code_commit"])
    path = root/"pipeline_state.json"
    atomic(path,state)
    gpu = next((g for g in rt["allowed_gpus"] if free_memory(g) >= 11000),None)
    if gpu is None:
        raise RuntimeError("No allocated GPU has memory for the initial checkpoint smoke")
    child, item = launch(rt,dict(id="smoke",kind="smoke",config=rt["runtime_file"]),gpu)
    state.update(status="SMOKE_RUNNING",smoke=item)
    atomic(path,state)
    if child.wait() or read(root/"smoke.json")["status"]!="PASS":
        state["status"]="FAILED_NO_QUERY_SMOKE";atomic(path,state)
        raise RuntimeError("Real checkpoint smoke did not pass")
    state["status"]="RUNNING"
    schedule(rt, jobs, state, {}, lambda memory, _: memory >= 11000, foreign_score=foreign_CPU_score)


def schedule(rt, jobs, state, active, admits, launch_job=launch, analyze=analysis, auxiliary=None,
             foreign_score=None):
    from experiments.cric_dura_comparison.pipeline import ready, done, block_failed_dependencies, alive
    from experiments.practical_full_2x2.cache import atomic

    root = Path(rt["run_root"])
    path = root/"pipeline_state.json"
    gates = state["jobs"]
    failures = []
    while True:
        for key,(proc,item) in list(active.items()):
            if (alive(item["pid"]) if proc is None else proc.poll() is None):
                continue
            exit_code = proc.returncode if proc is not None else None
            ok=exit_code in (None, 0) and done(jobs[key])
            gates[key]=dict(item,status="COMPLETE" if ok else "FAILED",exit_code=exit_code)
            if not ok: failures.append(key)
            del active[key]
        if auxiliary is not None and auxiliary.poll() is not None and "augmentation" not in failures:
            failures.append("augmentation")
        block_failed_dependencies(list(jobs.values()),state,[])
        queued=[job for key,job in jobs.items() if gates[key]["status"]=="QUEUED"]
        eligible=[j for j in queued if ready(j)] if not failures else []
        new=[]
        cpu=next((j for j in eligible if j["kind"] in ("score","reuse")),None)
        if (cpu and not any(item["gpu"] is None for _,item in active.values())
                and (cpu["kind"]=="reuse" or foreign_score is None
                     or not foreign_score({item["pid"] for _,item in active.values()}))):
            new.append((cpu,None))
        gpu_jobs=sorted((j for j in eligible if j["kind"] in ("train","predict")),
                        key=lambda j:(j["kind"]!="predict", j.get("priority",0)))
        for gpu in rt["allowed_gpus"]:
            if not gpu_jobs: break
            if any(item["gpu"]==gpu for _,item in active.values()):
                continue
            memory = free_memory(gpu)
            candidate = next((i for i, job in enumerate(gpu_jobs) if admits(memory, job)), None)
            if candidate is not None:
                new.append((gpu_jobs.pop(candidate),gpu))
        for job,gpu in new:
            active[job["id"]]=launch_job(rt,job,gpu)
            gates[job["id"]]=active[job["id"]][1]
        state.update(counts=dict(Counter(x["status"] for x in gates.values())),failed_jobs=failures,updated_at=time.time())
        atomic(path,state)
        if not active and (not queued or failures): break
        time.sleep(15)
    state["status"]="PARTIAL_TECHNICAL_FAILURE" if failures else "ANALYZING"
    atomic(path,state)
    if not failures:
        analyze(root)
        state["status"]="ANALYZED"
        atomic(path,state)


if __name__ == "__main__":
    import json

    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("dispatch","worker"))
    parser.add_argument("--runtime",required=True)
    parser.add_argument("--kind")
    parser.add_argument("--config")
    a=parser.parse_args()
    rt=json.loads(Path(a.runtime).read_text(encoding="utf-8"))
    sys.path.insert(0,str(Path(rt["project"])/"releases"/SOURCE_RELEASE))
    from experiments.cric_dura_comparison.io import read
    dispatch(rt) if a.action=="dispatch" else worker(read(a.config),a.kind)
