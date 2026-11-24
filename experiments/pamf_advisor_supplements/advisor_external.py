"""Run the four published Full+PL baselines on the advisor's exact data roles."""
import argparse
import json
import os
from pathlib import Path
import sys

from adaptive_queue import TASKS, launch, schedule
from plan import MODEL_SEED, matrix as advisor_matrix

RUN = "20261124-phase1-pamf-advisor-external-m96-r01"
FAMILIES = ("riei", "drift", "ibrff", "sdg")
MIN_FREE_MIB = 3400


def matrix():
    for row in advisor_matrix():
        if row["method"] != "ssl":
            continue
        for method in FAMILIES:
            yield dict(row, method=method, row_id=f"{row['setting']}-{method}-full-pl-s{row['model_seed']}")


def activate(rt):
    sys.path.insert(0, rt["reference_release"])
    from experiments.practical_full_2x2.runner import activate as native

    native(rt)


def source_cache(row, old, output, read, write):
    import numpy as np

    base = read(old/"source_contract.json")
    if (base["counts"] != row["counts"] or base["source_days"] != row["source_days"]
            or base["source_rxs"] != row["source_rxs"] or base["num_classes"] != 6
            or base["U_TX_truth_read"] or base["target_access_before_freeze"]
            or base["checkpoint_init"] != "scratch_only"):
        raise ValueError("Advisor source role mismatch")
    output.mkdir(parents=True, exist_ok=False)
    local = {rx:i for i,rx in enumerate(base["source_rxs"])}
    for role in ("L_s", "U_s", "V"):
        with np.load(old/f"{role}_metadata.npz", allow_pickle=False) as data:
            if data["ids"].tolist() != base["role_ids"][role] or ("y" in data.files) != (role != "U_s"):
                raise ValueError("Unexpected source-role IDs or labels")
            ix = dict(ids=data["ids"], rx=np.asarray([local[v] for v in data["rx_i"]], np.int64))
            if role != "U_s":
                ix["y"] = data["y"]
            np.savez(output/f"{role}_index.npz", **ix)
        for suffix in ("_x.npy", "_metadata.npz"):
            target = old/(role+suffix)
            if not target.is_file():
                raise FileNotFoundError(target)
            (output/(role+suffix)).symlink_to(target)
    contract = dict(base, U_s_use=True, U_s_labels_hidden=True)
    write(output/"source_contract.json", contract)
    write(output/"metadata.json", dict(source_contract=contract, classes=base["classes"], counts=row["counts"],
                                     target_constructed=False, U_s_exported=True, U_s_labels_exported=False))
    return contract


def prepare(rt, root):
    from experiments.cric_dura_comparison.io import read, write

    registered = read(Path(rt["release_root"])/"experiment.json")["rows"]
    rows = list(matrix())
    if [row["data_overrides"] for row in registered] != rows:
        raise ValueError("The registered and dispatched rows differ")
    original = Path(rt["advisor_root"])
    templates = {name:read(Path(rt["recipe_root"])/"configs"/f"F__F0-{name}-D-s{MODEL_SEED}.json")
                 for name in FAMILIES}
    jobs = {}
    contracts = {}

    def add(kind, row, config, artifact, status, dependencies=()):
        key = f"{kind}__{row['row_id']}"
        path = root/"configs"/f"{key}.json"
        write(path, config)
        jobs[key] = dict(id=key, kind=kind, config=str(path), completion=str(artifact),
                         successful_statuses=[status], dependencies=list(map(str, dependencies)))

    for row in rows:
        setting, key, method = row["setting"], row["row_id"], row["method"]
        cache = root/"data"/setting
        if setting not in contracts:
            contracts[setting] = source_cache(row, original/"data"/setting, cache, read, write)
        contract = contracts[setting]
        old = f"{setting}-ssl-s{MODEL_SEED}"
        p = read(original/"configs"/f"predict__{old}.json")
        s = read(original/"configs"/f"score__{old}.json")
        if (p["classes"] != contract["classes"] or list(p["tests"]) != list(TASKS)
                or any(s["tests"][task]["views_root"] != p["tests"][task]["views_root"] for task in TASKS)):
            raise ValueError("Advisor test prediction and scoring handles differ")
        src, out = root/"F"/key/"source", root/"readouts"/key
        c = dict(templates[method], run_id=RUN, row_id=key, protocol_id=setting, group=setting,
                 source_rxs=row["source_rxs"], source_days=row["source_days"], num_domains=row["n_source_rx"],
                 source_cache=str(cache), source_contract=str(cache/"source_contract.json"),
                 source_contract_path=str(cache/"source_contract.json"), output_root=str(src),
                 augmentation_cache=rt["augmentation_cache"], classes=contract["classes"], num_classes=6,
                 code_commit=rt["code_commit"])
        add("train", row, c, src/"freeze.json", "SOURCE_TRAINED_FROZEN")
        c = dict(run_id=RUN, row_id=key, views=p["views"], tests=p["tests"], classes=p["classes"],
                 source_output=str(src), source_contract=str(cache/"source_contract.json"),
                 output_root=str(out/"predictions"), readouts=["raw"], model_seed=MODEL_SEED,
                 source_days=row["source_days"])
        add("predict", row, c, out/"predictions"/"complete.json", "ALL_PREDICTIONS_COMPLETE", (src/"freeze.json",))
        s.update(run_id=RUN, row_id=key, prediction_output=c["output_root"], output_root=str(out/"scores"))
        add("score", row, s, out/"scores"/"complete.json", "SCORED_COMPLETE", (out/"predictions"/"complete.json",))
    write(root/"jobs.json", list(jobs.values()))
    return jobs


def adapt_native_source(rt):
    from experiments.cric_dura_comparison.io import read
    from experiments.practical_full_2x2 import plan

    root = Path(rt["run_root"])
    jobs = read(root/"jobs.json")
    configs = [read(job["config"]) for job in jobs if job["kind"] == "train"]
    plan.make_plan = lambda: dict(rows=configs, protocols={
        c["protocol_id"]:dict(source_rxs=c["source_rxs"], source_days=c["source_days"], dataset="ManySig")
        for c in configs})
    from experiments.practical_full_2x2 import source
    return source


def smoke(rt):
    import torch
    from experiments.cric_dura_comparison.io import read, write
    from experiments.practical_full_2x2.compat import tensor

    native = adapt_native_source(rt)
    root = Path(rt["run_root"])
    for row in matrix():
        if row["setting"] == "budget630":
            c = read(root/"configs"/f"train__{row['row_id']}.json")
            native.check_training(c)
            net = native.build(c).eval()
            x = native.array_loader(c, "L_s").dataset.tensors[0][:2]
            with torch.inference_mode():
                logits = native.forward(net, c, x)
            if logits.shape != (2,6) or not bool(torch.isfinite(logits).all()):
                raise ValueError(f"Source model failed for {row['method']}")
    old = Path(rt["recipe_root"])/"F"/f"F0-riei-D-s{MODEL_SEED}"/"source"
    d = read(old/"freeze.json")
    c = d["config"]
    contract = read(c["source_contract"])
    if d["status"] != "SOURCE_TRAINED_FROZEN" or d["target_access"] or d["target_score_used"]:
        raise ValueError("Historical no-query smoke reference is unverified")
    payload = native.native.load(d["checkpoint"], c, contract, "cpu")
    net = native.build(c).eval()
    net.load_state_dict(payload["state_dict"], strict=True)
    import numpy as np
    data = np.load(Path(c["source_cache"])/"L_s_x.npy", mmap_mode="r", allow_pickle=False)
    with torch.inference_mode():
        logits = native.forward(net, c, tensor(data[:2].copy()))
    if logits.shape != (2,6) or not bool(torch.isfinite(logits).all()):
        raise ValueError("Frozen source-only smoke failed")
    write(root/"smoke.json", dict(status="PASS", source_methods=list(FAMILIES),
                                 real_checkpoint=True, query_reads=0, inherited_models=0))
    print("ADVISOR_EXTERNAL_NO_QUERY_SMOKE_PASS", flush=True)


def predict(c, native):
    import numpy as np
    import torch
    from experiments.cric_dura_comparison.evaluation import VIEWS, handle, reject_truth
    from experiments.cric_dura_comparison.io import read, write
    from experiments.practical_full_2x2.compat import tensor

    reject_truth(c)
    if list(c["tests"]) != list(TASKS) or c["views"] != list(VIEWS) or c["readouts"] != ["raw"]:
        raise ValueError("Wrong frozen evaluator matrix")
    net, config = native.frozen_model(c["source_output"], read(c["source_contract"]), "cuda:0")
    if (config["run_id"],config["row_id"],config["model_seed"]) != (RUN,c["row_id"],MODEL_SEED):
        raise ValueError("Frozen external row identity differs")
    root = Path(c["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    counts = {}
    with torch.inference_mode():
        for task in TASKS:
            views = c["tests"][task]["views_root"]
            manifest, ids, positions = handle(views, c["classes"])
            outputs = {}
            for view in VIEWS:
                path = manifest["clean_ref"] if view == "clean" else Path(manifest.get("observations_root",views))/f"{view}.npy"
                data = np.load(path, mmap_mode="r", allow_pickle=False)
                labels = []
                for start in range(0,len(ids),128):
                    batch = slice(start,start+128)
                    logits = native.forward(net, config, tensor(np.ascontiguousarray(data[batch if positions is None else positions[batch]]),"cuda:0"))
                    if logits.shape[1] != len(c["classes"]) or not bool(torch.isfinite(logits).all()):
                        raise ValueError("Invalid external frozen logits")
                    labels.extend(logits.argmax(1).cpu().tolist())
                outputs[f"raw__{view}"] = np.asarray(labels, np.int64)
                print("FROZEN_PREDICTION", c["row_id"], task, view, len(ids), flush=True)
            folder = root/task
            folder.mkdir()
            np.savez(folder/"predictions.npz", ids=ids, **outputs)
            counts[task] = dict(count=len(ids), views=c["views"], readouts=["raw"])
    write(root/"complete.json", dict(status="ALL_PREDICTIONS_COMPLETE", run_id=RUN, row_id=c["row_id"],
                                     tests=counts, views=c["views"], readouts=["raw"], truth_read=False, query_fit=False))


def analyze(root):
    import csv
    from experiments.cric_dura_comparison.evaluation import DATES
    from experiments.cric_dura_comparison.io import read

    records = []
    for row in matrix():
        scores = read(root/"readouts"/row["row_id"]/"scores"/"scores.json")
        if not scores["truth_last"] or scores["row_id"] != row["row_id"]:
            raise ValueError("Independent final score missing")
        matching = [x for x in scores["results"] if (x["dimension"],x["stratum"]) ==
                    ("date",DATES[row["endpoint"]])] if row["endpoint"] is not None else [
                    x for x in scores["results"] if x["dimension"] == "overall"]
        if len(matching) != (7 if row["endpoint"] is not None else 14):
            raise ValueError("Registered scene or date missing")
        records.extend(dict(setting=row["setting"], method=row["method"], endpoint=row["endpoint"],
                            target_partition=r["test"], view=r["view"], count=r["count"],
                            accuracy=r["accuracy"], macro_f1=r["macro_f1"]) for r in matching)
    with (root/"external_results.csv").open("x", encoding="utf-8", newline="") as stream:
        out = csv.DictWriter(stream, fieldnames=list(records[0]))
        out.writeheader()
        out.writerows(records)


def dispatch(rt):
    from experiments.cric_dura_comparison.io import read
    from experiments.practical_full_2x2.cache import Reader, atomic

    root = Path(rt["run_root"])
    root.mkdir(exist_ok=False)
    (root/"logs").mkdir()
    jobs = prepare(rt, root)
    path = root/"pipeline_state.json"
    state = dict(pid=os.getpid(), status="SMOKE_RUNNING", code_commit=rt["code_commit"],
                 jobs={key:dict(status="QUEUED") for key in jobs})
    atomic(path, state)
    proc, _ = launch(rt, dict(id="smoke", kind="smoke", config=rt["runtime_file"]), None, Path(__file__).name)
    if proc.wait() or read(root/"smoke.json")["status"] != "PASS":
        state["status"] = "FAILED_NO_QUERY_SMOKE"
        atomic(path, state)
        raise RuntimeError("External source smoke failed")
    proc, item = launch(rt, dict(id="augmentation", kind="augmentation", config=rt["runtime_file"]),
                        None, Path(__file__).name)
    state.update(status="RUNNING", augmentation=item)
    atomic(path, state)
    Reader(read(jobs[f"train__budget630-riei-full-pl-s{MODEL_SEED}"]["config"])).request(1,1,.3)
    try:
        schedule(rt, jobs, state, {}, lambda memory, _:memory >= MIN_FREE_MIB,
                 launch_job=lambda cfg, job, gpu:launch(cfg,job,gpu,Path(__file__).name),
                 analyze=analyze, auxiliary=proc)
    finally:
        (Path(rt["augmentation_cache"])/"stop").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("dispatch", "worker"))
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--kind")
    args = parser.parse_args()
    rt = json.loads(args.runtime.read_text(encoding="utf-8"))
    activate(rt)
    from experiments.cric_dura_comparison.io import read

    if args.action == "dispatch":
        dispatch(rt)
    elif args.kind == "smoke":
        smoke(rt)
    elif args.kind == "augmentation":
        from experiments.practical_full_2x2.cache import service
        service(rt)
    elif args.kind == "score":
        from experiments.cric_dura_comparison import evaluation
        evaluation.tasks = lambda _:list(TASKS)
        evaluation.score(read(args.config))
    elif args.kind == "train":
        adapt_native_source(rt).train(read(args.config))
    elif args.kind == "predict":
        predict(read(args.config), adapt_native_source(rt))
    else:
        raise ValueError(f"Unknown worker kind {args.kind}")
