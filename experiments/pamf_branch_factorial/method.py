"""Bounded source-only branch experiment; no training starts on import."""
from contextlib import contextmanager
from copy import deepcopy

from .graph import SPECS

RUN = "20261109-phase1-pamf-branch-factorial-m36-r01"
SEEDS = (2026092701, 2026092702, 2026092703)
REUSED = ("base", "no_signature", "no_signature_no_pa")
ARMS = (*REUSED, *SPECS)
TASKS = ("cross_rx", "cross_both")
READOUTS = ["raw"]


def rows():
    return [dict(row_id=f"{arm}-s{seed}", arm=arm, model_seed=seed, reuse=arm in REUSED)
            for arm in ARMS for seed in SEEDS]


def configuration(context, row, project, commit):
    c = deepcopy(context["baselines"][str(row["model_seed"])]["complete"]["config"])
    c.update(run_id=RUN, row_id=row["row_id"], code_commit=commit,
             output_root=project + "/runs/" + RUN + "/F/" + row["row_id"] + "/source",
             algorithm_id="pamf_branch_factorial_ce", paper_method="PAMF CE structural ablation",
             display_method="PAMF CE structural ablation", G_source_roles="L_s",
             branch_factorial_arm=row["arm"], shared_prefix=None)
    c["repl_row"].update(row_id=row["row_id"], method="BF_" + row["arm"],
                         structure="BF_" + row["arm"], reuse=False, readouts=READOUTS)
    validate_config(c)
    return c


def validate_config(c):
    row = next((r for r in rows() if r["row_id"] == c.get("row_id")), None)
    if row is None or row["reuse"] or c.get("run_id") != RUN or c.get("model_seed") != row["model_seed"]:
        raise ValueError("Only preregistered new row and matching seed")
    if c.get("branch_factorial_arm") != row["arm"] or row["arm"] not in SPECS:
        raise ValueError("Unknown branch configuration")
    if any(c.get(k) for k in ("truth", "target_truth", "tests", "views_root", "dataset_path",
                              "checkpoint", "teacher_checkpoint", "parent_source", "shared_prefix")):
        raise ValueError("Only same-contract scratch source training is permitted")
    if (c["regime"] != "matched_sat_supervised" or c["ablation"] != "F15" or
            any(c.get(k) for k in ("U_use", "own_EMA", "native_U_gate", "native_unlabeled_loop", "posthoc_head_fit")) or
            c["epochs"] != 50 or c["optimizer_updates"] != 11100 or c["trajectory"] != [20, 10, 20] or
            c["satellite_concat"] is not True or c["satellite_CE_weight"] != .68 or
            c["channel_augmentation"] is not True or c["augmentation_seed"] != 2026102301 or
            c["source_rxs"] != [1, 3, 4, 6, 8] or c["source_days"] != [1, 2, 3] or c["num_classes"] != 6):
        raise ValueError("Fixed CE source recipe, split and budget required")
    r = c["repl_row"]
    if (r["structure"] != "BF_" + row["arm"] or r["group"] != "strict_structure" or
            r["U"] or r["L_A"] or r["L_B"] or any(r["blocks"].values()) or
            r["model_seed"] != c["model_seed"] or r["row_id"] != c["row_id"]):
        raise ValueError("No U or additional objectives permitted")


@contextmanager
def architecture_scope():
    from experiments.cric_dura_comparison import supervised_physics as physics
    from .graph import install as install_graph
    original = physics.install
    def install(model, arm, seed):
        return install_graph(model, arm[3:], seed) if arm.startswith("BF_") else original(model, arm, seed)
    physics.install = install
    try:
        yield
    finally:
        physics.install = original


@contextmanager
def optimizer_scope(no_time):
    if not no_time:
        yield
        return
    # There is no Sinc parameter in this architecture. Native CENTER_UNITS
    # requires two Sinc parameters; preserve the original AdamW arguments and
    # ordinary global clipping on all parameters that actually remain.
    import torch
    from experiments.cric_dura_comparison import f_optimizer
    original_scope = f_optimizer.native_scope

    @contextmanager
    def without_sinc(owner, mode):
        if mode != "CENTER_UNITS":
            raise ValueError("Unexpected source optimizer policy")
        original = torch.optim.AdamW
        class ScopedAdamW(original):
            def __init__(self, params, **options):
                if getattr(owner, "optimizer", None) is not None:
                    raise ValueError("Multiple optimizers")
                super().__init__(params, **options)
                owner.optimizer = self
        torch.optim.AdamW = ScopedAdamW
        try:
            yield
        finally:
            torch.optim.AdamW = original

    f_optimizer.native_scope = without_sinc
    try:
        yield
    finally:
        f_optimizer.native_scope = original_scope


def train(c):
    validate_config(c)
    from experiments.pamf_repl_completion import source
    original = source.validate_config
    source.validate_config = validate_config
    try:
        with architecture_scope(), optimizer_scope(not SPECS[c["branch_factorial_arm"]][0]):
            return source.train(c)
    finally:
        source.validate_config = original


def load_reader(folder, contract, device):
    from experiments.pamf_repl_completion.source import load_reader as load
    with architecture_scope():
        return load(folder, contract, device, require_no_unlabeled=True)
