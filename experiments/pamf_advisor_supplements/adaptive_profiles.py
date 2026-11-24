"""Transfer the validated adaptive Tri profiles to the advisor's fixed splits."""
from contextlib import contextmanager
from functools import partial
from pathlib import Path

RUN = "20261124-phase1-pamf-advisor-adaptive-m48-r01"
BASE = "20261124-phase1-pamf-advisor-data-m32-r01"
REFERENCE_COMMIT = "e479c466a6a0b1862a17407f6ad5f4d2ec6aedd2"
SOURCE_RELEASE = "20261121-phase1-pamf-tri-label01-earlyanchor-seedcheck-clean-m2-r01-release"
TASKS = ("cross_rx", "cross_both")


def matrix():
    from plan import matrix as original

    for row in original():
        if row["method"] != "ssl":
            continue
        setting = row["setting"]
        n = row["n_source_rx"]
        per_cell = row["counts"]["L_s"] / (n * len(row["source_days"]) * 6)
        profile = "original_fallback"
        if setting == "budget3150":
            profile = "fusion_rx_balance"
        elif row["family"] == "receiver_count" and n == 1:
            profile = "single_rx_hybrid"
        elif row["family"] == "receiver_count" and n == 6:
            profile = "six_rx_same_day"
        adaptive = dict(row, method="adaptive", profile=profile, early_anchor=per_cell < 14)
        adaptive["row_id"] = f"{setting}-adaptive-s{row['model_seed']}"
        adaptive["reuse"] = profile == "original_fallback" and not adaptive["early_anchor"]
        yield adaptive
        yield dict(row, row_id=f"{setting}-pl-s{row['model_seed']}", method="pl",
                   profile="ordinary_pl", early_anchor=False, reuse=row["family"] != "receiver_count")


def count_coverage(c):
    from experiments.pamf_repl_source_adapt.view_route import labeled_cell_counts
    table = labeled_cell_counts(c)
    return float(table[c["source_rxs"]][:, c["source_days"]].float().median())


def validate(c):
    from experiments.cric_dura_comparison.io import read

    by_id = {row["row_id"]: row for row in matrix()}
    row = by_id.get(c["row_id"])
    if (row is None or c["advisor_row"] != row or c["run_id"] != RUN
            or c["model_seed"] != row["model_seed"] or c["source_rxs"] != row["source_rxs"]
            or c["source_days"] != row["source_days"] or c["epochs"] != 50
            or c["optimizer_updates"] != 11100 or c["plain_PL_threshold"] != .95
            or c["plain_PL_weight"] != 1. or any(c.get(k) for k in (
                "truth", "target_truth", "target_inputs", "tests", "views_root", "checkpoint", "teacher_checkpoint"))):
        raise ValueError("Registered source-only advisor row required")
    contract = read(c["source_contract"])
    if (not contract["checkpoint_init"] == "scratch_only" or contract["target_access_before_freeze"]
            or contract["U_TX_truth_read"] or contract["counts"] != row["counts"]
            or contract["source_rxs"] != row["source_rxs"] or contract["source_days"] != row["source_days"]):
        raise ValueError("Advisor physical data role mismatch")
    if c["paper_row"]["permuted"] != (row["method"] == "adaptive"):
        raise ValueError("Incorrect Tri/PL deployment graph")
    return row


def active_config(c):
    """Apply the inherited variant to E21 onward only."""
    row = c["advisor_row"]
    profile = row["profile"]
    if row["early_anchor"]:
        c["source_adaptation"] = dict(mode="replace", floor=.55)
    if profile == "fusion_rx_balance":
        c["source_adaptation"] = dict(setting="labeled_5pct", route="joint")
    elif profile == "single_rx_hybrid":
        c.update(hybrid_contrast=dict(mode="fusion_branch", temperature=.2),
                 practical_gate=dict(mode="branch", strength=.25, schedule="taper"),
                 fused_physics=dict(weight="original_joint", raw_detached=True), day_relation_weight=.025)
    elif profile == "six_rx_same_day":
        from experiments.pamf_repl_component_guard.plan import apply_recipe
        from experiments.pamf_repl_two_setting.plan import RECIPES
        # The same frozen six-RX profile, transferred to the new source cache.
        from experiments.pamf_repl_component_guard import plan as component
        old = component.RECIPES
        component.RECIPES = dict(old, six_same_date=RECIPES["six_same_date"])
        try:
            c = apply_recipe(c, "six_same_date")
        finally:
            component.RECIPES = old
        c["tuning_recipe"] = RECIPES["six_same_date"]
    return c


@contextmanager
def without_fixed_validator(module, name="validate"):
    original = getattr(module, name)
    setattr(module, name, validate)
    try:
        yield
    finally:
        setattr(module, name, original)


@contextmanager
def scope(c, holder, ssl, bank=None):
    from experiments.pamf_paper_completion.model import installed as paper

    validate(c)
    row = c["advisor_row"]
    if not ssl:
        with paper(c, holder, ssl_enabled=False, validate=validate):
            yield
    elif row["profile"] == "single_rx_hybrid":
        from experiments.pamf_physics_tri import hybrid_contrast_plan, hybrid_contrast
        with without_fixed_validator(hybrid_contrast_plan), hybrid_contrast.installed(c, holder):
            yield
    elif row["profile"] == "fusion_rx_balance":
        from experiments.pamf_repl_source_adapt.receiver_balance import installed
        with installed(c, holder, validate_config=validate):
            yield
    elif row["profile"] == "six_rx_same_day":
        from experiments.pamf_repl_component_guard import plan as component
        from experiments.pamf_repl_day_pairs import objective as days
        from experiments.pamf_repl_source_adapt.early_label_anchor import Anchor

        original = days.DayPairsREPL
        if bank is not None:
            class AnchoredDayPairs(Anchor, original):
                def __init__(self, cfg):
                    super().__init__(cfg, bank)
            days.DayPairsREPL = AnchoredDayPairs
        try:
            with without_fixed_validator(component), days.installed(c, holder):
                yield
        finally:
            days.DayPairsREPL = original
    else:
        from experiments.pamf_repl_source_adapt.early_label_anchor import Anchor

        objective = partial(Anchor, bank=bank) if bank is not None else None
        with paper(c, holder, ssl_enabled=True, validate=validate, objective_factory=objective):
            yield


def train(c):
    import shutil
    from experiments.cric_dura_comparison.f_train import train as native
    from experiments.cric_dura_comparison.io import read, write
    from experiments.pamf_branch_training.source import checked_e20_parent
    from experiments.pamf_repl_source_adapt.early_label_anchor import source_teacher

    row = validate(c)
    coverage = count_coverage(c)
    if (row["method"] == "adaptive" and (coverage < 14) != row["early_anchor"]):
        raise ValueError("Measured label coverage differs from the registered trigger")
    root = Path(c["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    write(root/"resolved_config.json", dict(c, measured_label_coverage=coverage))
    segments = []
    if c.get("shared_prefix"):
        prefix = checked_e20_parent(c)
        prior = prefix["config"]
        if (prior["run_id"] != BASE or prior["row_id"] != row["row_id"].replace("-adaptive-", "-ssl-")
                or prior["source_cache"] != c["source_cache"] or prior["source_rxs"] != c["source_rxs"]
                or prior["source_days"] != c["source_days"] or not prior["paper_row"]["permuted"]):
            raise ValueError("Original advisor E20 does not match this row")
        segments.append(prefix)
    previous = segments[0]["folder"] if segments else None
    bank = None
    for n, start in ((20,0),(10,20),(20,30)):
        if not start and previous is not None:
            continue
        ssl = row["method"] == "adaptive" and start >= 20
        if ssl and row["early_anchor"] and bank is None:
            bank = source_teacher(c, segments[0])
        folder = root/f"segment{start+n}"
        cfg = dict(c, output_root=str(folder), segment_epochs=n, epoch_offset=start,
                   step_offset=start*222, parent_source=previous)
        if ssl:
            cfg = active_config(cfg)
        holder = {}
        with scope(cfg, holder, ssl, bank):
            native(cfg)
        if ssl:
            obj = holder["objective"].state_dict()
            if obj["U_truth_read"] or obj["calls"] != n*222 or obj["student_views"] != 6*obj["seen"]:
                raise ValueError("Incomplete or unauthorized adaptive U trajectory")
            write(folder/"adaptive_state.json", obj)
        done = read(folder/"complete.json")
        if (done["status"] != "SOURCE_FROZEN" or done["updates"] != n*222 or done["target_access"]
                or done["U_truth_read"] or done["head_fit_updates"]):
            raise ValueError("Incomplete, unauthorized source segment")
        segments.append(dict(folder=str(folder), **done))
        previous = str(folder)
        write(root/"progress.json", dict(status="SEGMENT_COMPLETE", epoch=start+n, updates=(start+n)*222),
              exclusive=False)
    shutil.copyfile(Path(previous)/"final_ssdg.pth", root/"final_ssdg.pth")
    write(root/"source_contract.json", read(c["source_contract"]))
    write(root/"complete.json", dict(status="SOURCE_FROZEN", config=c, epoch=50, updates=11100,
        segments=segments, initialization="same_contract_scratch20_own10_own20",
        ancestors=[s["folder"] for s in segments], target_access=False, U_truth_read=False,
        head_fit_updates=0, source_contract=c["source_contract"], label_coverage=coverage,
        adaptive_profile=row["profile"], early_anchor=row["early_anchor"]))
