"""One GPU frozen-feature U-graph/predictor, followed by independent CPU scoring."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from .star_representation_launch import _read, _write, _env, _gpu_free, _job


RUN = "20261030-phase3-star-binary-graph-s2026092701-r01"
SOURCE_RUN = "20261029-phase3-star-neighborhood-s2026092701-r01"
FIELDS = {"run_id", "code_commit", "run_root", "release_root", "python", "launch_owner",
          "allowed_gpus", "minimum_free_mb", "max_added_workers_per_gpu", "predict_config",
          "U_truth", "Q_truth", "targets_path"}
PREDICT_FIELDS = {"run_id", "code_commit", "source_root", "U_public_root", "U_capsule_id", "U_split_id",
                  "Q_public_root", "Q_capsule_id", "Q_split_id", "output_root", "batch_size",
                  "dura_checkpoint", "dura_release"}
SCORE_STATUS = "STAR_BINARY_GRAPH_SCORES_COMPLETE"


def validate(r):
    from .star_binary_graph_run import validate_config
    if (not isinstance(r, dict) or set(r) != FIELDS or r.get("run_id") != RUN
            or not isinstance(r.get("code_commit"), str)
            or re.fullmatch("[0-9a-f]{40}", r["code_commit"]) is None
            or r.get("allowed_gpus") != list(range(8)) or r.get("minimum_free_mb") != 4096
            or r.get("max_added_workers_per_gpu") != 2
            or any(not isinstance(r[k], str) or not r[k].strip() for k in
                   ("run_root", "release_root", "python", "launch_owner", "U_truth", "Q_truth", "targets_path"))):
        raise ValueError("Only the fixed single-worker binary graph development launch is allowed")
    root, config = Path(r["run_root"]), r["predict_config"]
    if not isinstance(config, dict) or set(config) != PREDICT_FIELDS:
        raise ValueError("Predictor receives exactly the public thirteen-field configuration")
    validate_config(config)
    if (root.name != RUN or config["run_id"] != RUN or config["code_commit"] != r["code_commit"]
            or Path(config["output_root"]) != root / "predict"
            or Path(config["source_root"]) != root.parent / SOURCE_RUN / "source"):
        raise ValueError("Run/code identities and immutable prediction/old-source roots differ")
    private = {r["U_truth"], r["Q_truth"], r["targets_path"]}
    if any(isinstance(value, str) and value in private for value in config.values()):
        raise ValueError("Truth and scoring targets belong only to the independent scorer")
    for key in ("source_root", "U_public_root", "Q_public_root", "dura_release"):
        source = Path(config[key]).resolve()
        if root.resolve().is_relative_to(source) or source.is_relative_to(root.resolve()):
            raise ValueError("Run output must be isolated from frozen source/public inputs")
    if any(Path(r[k]).resolve().is_relative_to(root.resolve()) for k in ("U_truth", "Q_truth", "targets_path")):
        raise ValueError("Private/scoring inputs must remain outside the owned output root")
    return r


def start(runtime):
    r = validate(_read(runtime))
    root = Path(r["run_root"])
    if root.exists():
        raise FileExistsError("Observe the existing run; never overwrite or relaunch")
    root.mkdir()
    (root / "logs").mkdir()
    (root / "configs").mkdir()
    _write(root / "configs/predict.json", r["predict_config"])
    command = [r["python"], "-B", "-u", "-m", __package__ + ".star_binary_graph_launch",
               "driver", "--runtime", str(Path(runtime).resolve())]
    with (root / "logs/driver.log").open("xb") as log:
        child = subprocess.Popen(command, cwd=r["release_root"], env=_env(r), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    value = dict(run_id=RUN, code_commit=r["code_commit"], pid=child.pid, argv=command,
                 cwd=r["release_root"], time=time.time(), launch_owner=r["launch_owner"])
    _write(root / "launch.json", value)
    print(json.dumps(value), flush=True)


def driver(runtime):
    r = validate(_read(runtime))
    root = Path(r["run_root"])
    path = root / "pipeline_state.json"
    if path.exists():
        raise FileExistsError("Existing pipeline state is preserved; observe rather than restart")
    state = dict(run_id=RUN, code_commit=r["code_commit"], pid=os.getpid(), status="RUNNING",
                 started_at=time.time(), jobs={})
    _write(path, state)
    try:
        if _read(root / "configs/predict.json") != r["predict_config"]:
            raise ValueError("Copied prediction configuration differs from the fixed runtime")
        available = _gpu_free(r)
        if not available:
            raise RuntimeError("No allowed GPU has the actual preregistered free memory")
        command = [r["python"], "-B", "-u", "-m", __package__ + ".star_binary_graph_run", "run",
                   "--config", str(root / "configs/predict.json"), "--device", "cuda:0"]
        if not _job(r, state, path, "predict", command, available[0]):
            raise RuntimeError("Prediction/adaptation failed; preserve partial outputs without retry")
        from .star_binary_graph_run import STATUS
        predicted = _read(root / "predict/complete.json")
        if (predicted.get("status") != STATUS or predicted.get("run_id") != RUN
                or predicted.get("code_commit") != r["code_commit"]):
            raise ValueError("Complete prediction identity differs; independent truth remains unopened")
        command = [r["python"], "-B", "-u", "-m", __package__ + ".star_binary_graph_score",
                   "--predictions-root", str(root / "predict"), "--U-truth", r["U_truth"],
                   "--Q-truth", r["Q_truth"], "--targets", r["targets_path"], "--output", str(root / "score")]
        if not _job(r, state, path, "score", command, None):
            raise RuntimeError("Independent CPU scoring failed; preserve partial outputs without retry")
        scored = _read(root / "score/complete.json")
        if (scored.get("status") != SCORE_STATUS or scored.get("run_id") != RUN
                or scored.get("code_commit") != r["code_commit"]
                or scored.get("primary_state") != "DA1_REG1"
                or type(scored.get("primary_pass")) is not bool
                or not isinstance(scored.get("next_step"), str) or not scored["next_step"].strip()
                or (not scored["primary_pass"] and scored["next_step"] != "NO_PROMOTION")):
            raise ValueError("Independent score identity or fixed primary result differs")
        state.update(status="ANALYZED_DEVELOPMENT", primary_state="DA1_REG1",
                     primary_pass=scored["primary_pass"], next_step=scored["next_step"],
                     formal_target_reads=0, final_phase3_complete=False)
    except Exception as exc:
        state.update(status="FAILED_PRESERVED", error=str(exc))
    state["finished_at"] = time.time()
    _write(path, state)
    print(state["status"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["start", "driver"])
    parser.add_argument("--runtime", required=True)
    args = parser.parse_args()
    return dict(start=start, driver=driver)[args.mode](args.runtime)


if __name__ == "__main__":
    main()
