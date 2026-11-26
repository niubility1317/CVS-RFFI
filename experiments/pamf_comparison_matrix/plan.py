"""Match the paper panels and the advisor's additional source-role settings."""
from pathlib import PurePosixPath

from experiments.pamf_advisor_supplements.plan import matrix as advisor_matrix
from experiments.practical_full_2x2.plan import SEEDS

RUN = "20261126-phase1-pamf-seven-method-matrix-m111-r01"
PROJECT = PurePosixPath("/home/szu2070436088/2510044040/CV-SincNet")
PYTHON = "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
MODULE = "experiments.pamf_comparison_matrix.runner"
ROOT = PROJECT / "runs" / RUN
RELEASE = PROJECT / "releases" / (RUN + "-release")
COMMAND = [PYTHON, "-u", "-m", MODULE, "dispatch", "--config", str(RELEASE / "runtime.json")]
PAPER = "20261027-phase1-pamf-paper-completion-m119-r03"
CONTEXT = PROJECT / "releases" / (PAPER + "-release") / "context.json"
STAR = "20261125-phase1-star-pamf-main-practical-m3-r01"
SHEN = "20261125-phase1-shen-wisig-main-m3-r04"
EXTERNAL = "20261023-phase1-practical-residual-control-r03"
ADVISOR = "20261124-phase1-pamf-advisor-data-m32-r01"
ADVISOR_EXT = "20261124-phase1-pamf-advisor-external-m96-r01"
NEW_METHODS = ("shen", "star")
METHODS = dict(riei="RIEI", drift="DRIFT", ibrff="IB-RFF", sdg="SDG",
               shen="Receiver-Agnostic RFFI", star="STAR-RFFI", pamf="PAMF")
PAPER_SETTINGS = {
    "F0": "main_split", "F1_rx_repartition_r01": "receiver_split_1",
    "F2_rx_repartition_r01": "receiver_split_2", "ManyRx": "ten_class",
    "label01": "labeled_1pct", "label05": "labeled_5pct",
    **{f"m{n}-natural": f"source_rx_{n}" for n in (1, 2, 3, 4, 6)},
}
NATURAL = (1, 2, 3, 4, 5, 6)
GPUS = (2, 6, 7, 5, 3, 1, 0, 4)
GPU_WORKERS = 4
V = ("cross_rx", "cross_both")


def settings():
    result = {}
    for name, slug in PAPER_SETTINGS.items():
        result[name] = dict(kind="paper", slug=slug, seeds=SEEDS)
    for row in advisor_matrix():
        if row["method"] == "ssl":
            result[row["setting"]] = dict(row, kind="supplement", seeds=(row["model_seed"],))
    if len(result) != 35:
        raise ValueError("Paper and advisor subject matrices differ")
    return result


def row_name(setting, method, seed):
    return f"{setting}-{method}-s{seed}"


def published_scores(setting, info, method, seed):
    r = PROJECT / "runs"
    if method in NEW_METHODS:
        if setting == "F0" and method == "star":
            return r / STAR / "readouts" / f"main-star-s{seed}"
        return ROOT / "readouts" / row_name(setting, method, seed)
    if info["kind"] == "paper":
        if method == "pamf":
            return r / PAPER / "readouts" / f"{info['slug']}-factorial-ssl-full-permuted-s{seed}"
        return r / EXTERNAL / "readouts" / f"{setting}-{method}-D-s{seed}"
    if method == "pamf":
        return r / ADVISOR / "readouts" / f"{setting}-ssl-s{seed}"
    return r / ADVISOR_EXT / "readouts" / f"{setting}-{method}-full-pl-s{seed}"
