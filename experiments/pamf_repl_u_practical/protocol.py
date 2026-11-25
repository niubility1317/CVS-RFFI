"""Frozen Main split contract and the verified same-seed E20 prefix."""
from pathlib import Path, PurePosixPath

from experiments.cric_dura_comparison.io import read
from experiments.pamf_repl_view_shuffle.plan import PRACTICAL_RUN as RUN, RUN as BASE, SEEDS, validate_config

PROJECT = PurePosixPath("/home/szu2070436088/2510044040/CV-SincNet")
PYTHON = "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
PAPER = "20261027-phase1-pamf-paper-completion-m119-r03"
CONTEXT = PROJECT / "releases" / f"{PAPER}-release" / "context.json"
STAR_PAIRS = PROJECT / "runs/20261125-phase1-star-pamf-main-practical-m3-r01/source_pairs"
SEED = SEEDS[0]
ROW = f"main_split-repl-shuffled_time-3view-s{SEED}"
BASE_COMMIT = "3ff8f4b7e85b5608f5b45350aa18081499a7ef1d"
BASE_SOURCE = PROJECT / "runs" / BASE / "F" / ROW / "source"
PREFIX = BASE_SOURCE / "segment20"
BASE_READOUT = PROJECT / "runs" / PAPER / "readouts" / f"main_split-factorial-ssl-full-permuted-s{SEED}"
ROOT = PROJECT / "runs" / RUN
SOURCE_OUT = ROOT / "F" / ROW / "source"
RELEASE = PROJECT / "releases" / (RUN + "-release")
MODULE = "experiments.pamf_repl_u_practical.runner"
COMMAND = [PYTHON, "-u", "-m", MODULE, "dispatch", "--config", str(RELEASE / "runtime.json")]
GPUS = (2, 6, 7, 3, 5, 0, 1, 4)
MIN_MIB = 12000
ROUTES = ("raw", "DURA-LU")


def configuration(commit):
    old = read(Path(BASE_SOURCE) / "complete.json")
    c = old["config"]
    validate_config(c)
    if (c["code_commit"] != BASE_COMMIT or old["target_access"] or old["U_truth_read"]
            or old["epoch"] != 50 or old["updates"] != 11100 or c["model_seed"] != SEED):
        raise ValueError("Historical source recipe differs from the fixed main method")
    result = dict(c, run_id=RUN, code_commit=commit, output_root=str(SOURCE_OUT),
                  hard_pl_no_uas=False, hard_pl_practical_u_cache=str(STAR_PAIRS))
    return validate_config(result)


def prefix_record(c, prefix):
    if str(prefix) != str(PREFIX) or c["hard_pl_practical_u_cache"] != str(STAR_PAIRS):
        raise ValueError("Only the preregistered E20 source prefix and source-U views are allowed")
    prefix = Path(prefix)
    old = read(prefix / "complete.json")
    baseline = dict(c, run_id=BASE, code_commit=BASE_COMMIT, output_root=str(prefix),
                    hard_pl_no_uas=True, segment_epochs=20, epoch_offset=0,
                    step_offset=0, parent_source=None)
    baseline.pop("hard_pl_practical_u_cache")
    if (old["config"] != baseline or old["status"] != "SOURCE_FROZEN"
            or old["epoch"] != 20 or old["updates"] != 4440 or old["target_access"]
            or old["head_fit_updates"] or old["U_truth_read"] or old["initialization"] != "scratch"
            or old["ancestors"] != [] or old["source_contract"] != c["source_contract"]):
        raise ValueError("CHECKPOINT_DATA_CONTRACT_MISMATCH: fixed source-only E20 prefix")
    return dict(folder=str(prefix), **old)
