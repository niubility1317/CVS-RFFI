"""Run the single-seed frozen restoration matrix with independent CPU scoring."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from experiments.cric_dura_comparison.io import read, write
from .dura import COUNT, METHODS, PANELS, RUN, SEED, TASKS

MODULE = "experiments.pamf_advisor_supplements.dura"


def dispatch(rt):
    from experiments.pamf_repl_completion.runner import capacity
    from experiments.practical_full_2x2.cache import atomic
    root = Path(rt["run_root"])
    root.mkdir(exist_ok=False)
    state = dict(status="PREPARING", pid=os.getpid(), commit=rt["code_commit"], jobs={})
    atomic(root/"pipeline_state.json", state)
    ctx = read(rt["context"])
    h = ctx["handles"]["F0"]
    pred = dict(source_contract=h["source_contract"],
        source_output=ctx["current_views"]["3/2026092701"]["source_output"],
        source_smoke=str(Path(h["source_cache"])/"L_s_x.npy"),
        frozen_sources=rt["frozen_sources"], g=ctx["caches"]["F0"]["LU"]["G"],
        views={k:h["tests"][k]["views_root"] for k in TASKS}, output_root=str(root/"predictions"))
    score = dict(prediction_output=pred["output_root"], classes=h["classes"],
        tests={k:h["tests"][k] for k in TASKS}, output_root=str(root/"scores"))
    write(root/"predict.json", pred)
    write(root/"score.json", score)
    registered = read(Path(rt["release_root"])/"automation_reports"/"CV-SincNet"/RUN/"experiment.json")
    request=registered["rows"][0]["data_overrides"]
    if request != dict(routes=list(METHODS), panels=list(PANELS), samples_per_task=COUNT, sampling_seed=SEED):
        raise ValueError("Restoration registration differs from executable matrix")
    choices=rt["allowed_gpus"]
    for action in ("predict","score"):
        gpu = None
        if action == "predict":
            state["status"]="WAITING_FOR_GPU"
            atomic(root/"pipeline_state.json", state)
            while gpu is None:
                gpu = next((g for g in choices if capacity(g)[1]>=9000),None)
                if gpu is None:
                    time.sleep(20)
        command=[rt["python"],"-u","-m",MODULE,action,"--config",str(root/f"{action}.json")]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
            OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
        log=root/f"{action}.log"
        with log.open("x") as stream:
            child=subprocess.Popen(command,cwd=rt["release_root"],env=env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            job=dict(pid=child.pid,gpu=gpu,cwd=rt["release_root"],argv=command,log=str(log),status="RUNNING")
            state["jobs"][action]=job
            state["status"]="RUNNING"
            atomic(root/"pipeline_state.json",state)
            rc=child.wait()
        expected=("predictions","ALL_PREDICTIONS_COMPLETE") if gpu is not None else ("scores","SCORED_COMPLETE")
        path=root/expected[0]/"complete.json"
        success=not rc and path.exists() and read(path)["status"]==expected[1]
        job["status"]="COMPLETE" if success else "FAILED"
        job["exit_code"]=rc
        state["status"]="RUNNING" if success else "FAILED"
        atomic(root/"pipeline_state.json",state)
        if not success:
            raise RuntimeError(f"Restoration stage {action} did not close")
    state["status"]="ANALYZED"
    atomic(root/"pipeline_state.json",state)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("dispatch",))
    p.add_argument("--config",required=True)
    a=p.parse_args()
    dispatch(read(a.config))
