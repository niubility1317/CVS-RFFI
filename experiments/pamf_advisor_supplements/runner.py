"""Train fixed paper-recipe rows, freeze predictions, then score separately."""
import argparse
from contextlib import contextmanager

from pathlib import Path
import subprocess
import sys
import os
import time

from experiments.cric_dura_comparison.io import read, write
from .plan import RUN, matrix

ROWS = {row["row_id"]: row for row in matrix()}
TASKS = ("cross_rx", "cross_both")
READOUTS = ("raw",)


def checks(c, base_validate):
    row = ROWS[c["row_id"]]
    template = "labeled_1pct" if row["family"] == "label_budget" else "main_split"
    identity = f"{template}-factorial-{row['method']}-full-{'permuted' if row['method']=='ssl' else 'ordered'}-s{row['model_seed']}"
    from experiments.pamf_paper_completion.plan import RUN as original_run, make_plan
    reference = next(r for r in make_plan()["rows"] if r["row_id"] == identity)
    expected = dict(reference, row_id=row["row_id"], protocol_id=row["setting"])
    if (c["paper_row"] != expected or c["run_id"] != RUN or c["source_rxs"] != row["source_rxs"]
            or c["source_days"] != row["source_days"]):
        raise ValueError("Source row differs from the preregistered matrix")
    base_validate(dict(c, row_id=identity, run_id=original_run, paper_row=reference, source_days=[1, 2, 3]))
    for key in ("source_cache", "source_contract"):
        if Path(c[key]).name != (row["setting"] if key == "source_cache" else "source_contract.json"):
            raise ValueError("Wrong derived source cache")
    contract = read(c["source_contract"])
    if (contract["counts"] != row["counts"] or contract["source_days"] != row["source_days"]
            or contract["source_rxs"] != row["source_rxs"] or contract["checkpoint_init"] != "scratch_only"
            or contract["U_TX_truth_read"] or contract["target_access_before_freeze"]):
        raise ValueError("Source permissions or physical roles differ")
    return c


@contextmanager
def training_scope():
    from experiments.pamf_paper_completion import plan, source
    original = plan.validate_config
    wrapper = lambda config: checks(config, original)
    plan.validate_config = source.validate_config = wrapper
    try:
        yield
    finally:
        plan.validate_config = source.validate_config = original


def prepare(rt):
    from experiments.cric_dura_comparison.evaluation import VIEWS, handle
    from experiments.pamf_paper_completion.plan import make_plan, configuration
    from .roles import make_cache

    root = Path(rt["run_root"])
    ctx = read(rt["context"])
    old_rows = {r["row_id"]: r for r in make_plan()["rows"]}
    inputs = {k: h["source_cache"] for k, h in ctx["handles"].items()}
    inputs["chrono"] = rt["chrono_cache"]
    jobs = []
    settings = set()

    def add(kind, row, c, completion, status, deps=()):
        path = root / "configs" / f"{kind}__{row['row_id']}.json"
        write(path, c)
        jobs.append(dict(id=f"{kind}__{row['row_id']}", kind=kind, family=row["family"],
                         learning="ssl" if row["method"] == "ssl" else "pl", config=str(path),
                         completion=str(completion), successful_statuses=[status], dependencies=list(deps)))

    registered = read(Path(rt["release_root"]) / "automation_reports" / "CV-SincNet" / RUN / "experiment.json")
    if [r["data_overrides"] for r in registered["rows"]] != list(ROWS.values()):
        raise ValueError("Registration and executable matrix differ")

    for row in ROWS.values():
        cache = root / "data" / row["setting"]
        h = ctx["handles"][row["target_handle"]]
        classes = h["classes"]
        views = {task: h["tests"][task] for task in TASKS}
        if row["setting"] not in settings:
            make_cache(row, inputs, cache, classes)
            d = read(cache / "source_contract.json")
            source_ids = {i for ids in d["role_ids"].values() for i in ids}
            for item in views.values():
                mf, ids, _ = handle(item["views_root"], classes)
                if (mf["target_rxs"] != row["target_rxs"]
                        or set(d["source_rxs"]).intersection(mf["target_rxs"])
                        or source_ids.intersection(ids.tolist())):
                    raise ValueError("A target population intersects the source roles")
            settings.add(row["setting"])
        template = "labeled_1pct" if row["family"] == "label_budget" else "main_split"
        method = row["method"]
        key = f"{template}-factorial-{method}-full-{'permuted' if method=='ssl' else 'ordered'}-s{row['model_seed']}"
        original = old_rows[key]
        c = configuration(ctx, original, rt["project"], rt["code_commit"])
        source = root / "F" / row["row_id"] / "source"
        c.update(run_id=RUN, protocol_id=row["setting"], row_id=row["row_id"],
                 paper_row=dict(original, row_id=row["row_id"], protocol_id=row["setting"]),
                 source_rxs=row["source_rxs"], source_days=row["source_days"],
                 source_cache=str(cache), source_contract=str(cache / "source_contract.json"),
                 output_root=str(source), frozen_sources=rt["frozen_sources"])
        add("train", row, c, source / "complete.json", "SOURCE_FROZEN")
        out = root / "readouts" / row["row_id"]
        p = {k: c[k] for k in ("run_id", "protocol_id", "row_id", "paper_row", "model_seed",
                               "source_days", "source_contract", "frozen_sources")}
        p.update(classes=classes, source_output=str(source), restored_caches={}, views=list(VIEWS),
                 readouts=list(READOUTS), output_root=str(out / "predictions"),
                 tests={k: {"views_root": v["views_root"]} for k, v in views.items()})
        s = {k: p[k] for k in ("run_id", "row_id", "model_seed", "source_days", "classes", "views", "readouts")}
        s.update(tests=views, output_root=str(out / "scores"), prediction_output=p["output_root"])
        add("predict", row, p, out / "predictions/complete.json", "ALL_PREDICTIONS_COMPLETE",
            (str(source / "complete.json"),))
        add("score", row, s, out / "scores/complete.json", "SCORED_COMPLETE",
            (str(out / "predictions/complete.json"),))
    write(root / "jobs.json", jobs)
    return jobs


def smoke(rt):
    import torch
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.pamf_paper_completion.model import load_reader
    ctx = read(rt["context"])
    h = ctx["handles"]["F0"]
    old = ctx["current_views"]["3/2026092701"]
    reader, _ = load_reader(old["source_output"], read(h["source_contract"]),
                            dict(learning_regime="ssl", structure="full", signature_views=3,
                                 permuted=True, model_seed=2026092701), "cuda:0")
    row = next(iter(ROWS.values()))
    source_path = Path(rt["run_root"]) / "data" / row["setting"]
    source = read(source_path / "source_contract.json")
    x = SourceDataset(source_path, "L_s", source).x[:2].to("cuda:0")
    with torch.inference_mode():
        logits = reader.classify_features(reader.features(x))
    if logits.shape != (2, 6) or not bool(logits.isfinite().all()):
        raise ValueError("No-query frozen reader smoke failed")
    write(Path(rt["run_root"]) / "smoke.json", dict(status="PASS", real_checkpoint=True,
          source_samples=2, query_reads=0, U_truth_reads=0))


def worker(c, kind):
    if kind == "score":
        from experiments.cric_dura_comparison import evaluation as native
        native.tasks = lambda _: list(TASKS)
        return native.score(c)
    from experiments.pamf_paper_completion import runner as paper
    paper.activate(c)
    if kind == "smoke":
        return smoke(c)
    if kind == "train":
        from experiments.pamf_paper_completion.source import train
        with training_scope():
            return train(c)
    if kind == "predict":
        from experiments.pamf_single_view_paper import evaluation as native
        from experiments.pamf_paper_completion.model import load_reader
        native.TASKS, native.READOUTS, native.load_reader = TASKS, READOUTS, load_reader
        return native.predict(c)
    raise ValueError(f"Unknown worker kind {kind}")


def launch(rt, job, gpu):
    path = Path(rt["run_root"]) / "logs" / f"{job['id']}.log"
    argv = [rt["python"], "-u", "-m", __spec__.name, "worker",
            "--kind", job["kind"], "--config", job["config"]]
    cwd = rt["release_root"]
    env = dict(os.environ, PYTHONPATH=cwd, CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    with path.open("x") as stream:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
    return proc, dict(status="RUNNING", pid=proc.pid, gpu=gpu, cwd=cwd, argv=argv,
                      log=str(path), started_at=time.time())


def analyze(rt):
    import csv
    from experiments.cric_dura_comparison.evaluation import DATES
    root = Path(rt["run_root"])
    result = []
    for row in ROWS.values():
        score = read(root / "readouts" / row["row_id"] / "scores" / "scores.json")
        if not score["truth_last"] or score["row_id"] != row["row_id"]:
            raise ValueError("Missing independent score")
        for r in score["results"]:
            endpoint = row["endpoint"]
            if endpoint is None:
                include = r["dimension"] == "overall"
            else:
                include = r["dimension"] == "date" and r["stratum"] == DATES[endpoint]
            if include:
                result.append(dict(row_id=row["row_id"], family=row["family"], setting=row["setting"],
                    method=row["method"], source_rxs=" ".join(map(str,row["source_rxs"])),
                    source_days=" ".join(map(str,row["source_days"])), test=r["test"], date=r["stratum"],
                    view=r["view"], count=r["count"], accuracy=r["accuracy"], macro_f1=r["macro_f1"]))
    if len(result) != 7 * sum(2 if r["endpoint"] is None else 1 for r in ROWS.values()):
        raise ValueError("Missing scene or held-out date")
    with (root / "results.csv").open("x", encoding="utf-8", newline="") as stream:
        output = csv.DictWriter(stream, fieldnames=list(result[0]))
        output.writeheader()
        output.writerows(result)


def admits(gpu):
    from experiments.pamf_repl_completion.runner import capacity
    _, free = capacity(gpu)
    mem = next(line for line in Path("/proc/meminfo").read_text().splitlines()
               if line.startswith("MemAvailable:"))
    return free >= 11000 and int(mem.split()[1]) >= 32 * 1024**2


def dispatch(rt):
    from experiments.pamf_paper_completion import runner as paper, prepare as paper_prepare
    paper_prepare.prepare = prepare
    paper.launch, paper.admits, paper.analyze = launch, admits, analyze
    paper.dispatch(rt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("dispatch", "worker"))
    parser.add_argument("--kind", choices=("smoke", "train", "predict", "score"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = read(args.config)
    if args.action == "dispatch":
        dispatch(config)
    else:
        worker(config, args.kind)
