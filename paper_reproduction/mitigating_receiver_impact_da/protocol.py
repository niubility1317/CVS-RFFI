from __future__ import annotations

from typing import Any


PAPER_UNSPECIFIED_FIELDS = [
    "batch size",
    "epoch count or convergence criterion",
    "optimizer apart from the reported learning rate",
    "ResNet18 1-D stem/kernel implementation details",
    "target validation policy without target-label leakage",
    "first-batch class-weight fallback",
    "empty pseudo-label target loss fallback",
    "zero-count CPL threshold floor",
]


PAPER_REPORTED_HYPERPARAMETERS = {
    "feature_extractor": "ResNet18 with 1-D convolutions",
    "classifier": "three-layer fully connected network",
    "estimate_network": "three-layer fully connected network",
    "learning_rate": 0.0006,
    "kl_weight_lambda": 0.005,
    "mu": 0.5,
    "estimate_update_frequency_m": 7,
    "initial_pseudo_label_threshold_tau": 0.7,
}


PAPER_TASKS = ["d01->d23", "14-7->3-19", "1-1->1-19", "1-1->8-8", "7-7->8-8"]
PAPER_COMPARE_METHOD_IDS = ["Source_only", "DANN", "MCD", "SHOT", "Proposed_GAD_DVKL_CPL_class_weighting"]
PAPER_DISPLAY_METHODS = ["Source only", "DANN", "MCD", "SHOT", "Proposed"]


def validate_paper_faithful_config(config: dict[str, Any]) -> dict[str, Any]:
    if bool(config.get("cvs_extension", False)):
        raise ValueError("paper-faithful Mitigating Receiver Impact DA config cannot set cvs_extension=true")
    if str(config.get("paper_scope", "")).strip() != "paper_faithful":
        raise ValueError("paper_scope must be paper_faithful")
    if "ManySig" not in str(config.get("dataset", "")):
        raise ValueError("paper-faithful reproduction expects WiSig ManySig")
    total_receivers = int(config.get("total_receivers", 0))
    tx_count = int(config.get("tx_count", 0))
    if total_receivers != 12:
        raise ValueError("IoTJ 2024 paper-faithful ManySig protocol expects 12 receivers")
    if tx_count != 6:
        raise ValueError("IoTJ 2024 paper-faithful ManySig protocol expects 6 transmitters")
    if not bool(config.get("target_unlabeled_allowed", False)):
        raise ValueError("target unlabeled data must be allowed for the UDA stages")
    if str(config.get("target_labels_scope", "evaluation_only")).strip() != "evaluation_only":
        raise ValueError("target_labels_scope must be evaluation_only for paper-faithful UDA")
    capture_days = int(config.get("capture_days", 4))
    if capture_days != 4:
        raise ValueError("IoTJ 2024 paper-faithful ManySig protocol expects 4 capture days")
    tasks = [str(v) for v in config.get("source_target_tasks", PAPER_TASKS)]
    if not tasks:
        raise ValueError("source_target_tasks cannot be empty")
    checked = dict(config)
    checked["source_target_tasks"] = tasks
    checked["claim_boundary"] = "paper-faithful closed-set cross-receiver DA with unlabeled target adaptation"
    checked["paper_unspecified_fields"] = PAPER_UNSPECIFIED_FIELDS
    checked["paper_reported_hyperparameters"] = PAPER_REPORTED_HYPERPARAMETERS
    checked["capture_days"] = capture_days
    checked["target_labels_scope"] = "evaluation_only"
    return checked


def _parse_task(task: str) -> tuple[str, str]:
    if "->" not in task:
        raise ValueError(f"paper task must use source->target form: {task}")
    source, target = [part.strip() for part in task.split("->", 1)]
    if not source or not target:
        raise ValueError(f"paper task must include source and target domains: {task}")
    return source, target


def build_paper_task_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in config["source_target_tasks"]:
        source, target = _parse_task(str(task))
        task_type = "cross_day" if source.startswith("d") or target.startswith("d") else "cross_receiver"
        rows.append(
            {
                "task": str(task),
                "task_type": task_type,
                "source_domain": source,
                "target_domain": target,
                "closed_set_tx_count": int(config["tx_count"]),
                "capture_days": int(config["capture_days"]),
                "target_data_role": "unlabeled_for_UDA_labels_evaluation_only",
                "compare_method_ids": list(PAPER_COMPARE_METHOD_IDS),
                "paper_display_methods": list(PAPER_DISPLAY_METHODS),
                "reported_hyperparameters": dict(PAPER_REPORTED_HYPERPARAMETERS),
                "primary_metric": "target receiver closed-set classification accuracy",
            }
        )
    return rows
