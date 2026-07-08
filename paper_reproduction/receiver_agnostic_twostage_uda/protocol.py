from __future__ import annotations

from typing import Any


PAPER_UNSPECIFIED_FIELDS = [
    "batch size",
    "epoch count or convergence criterion",
    "optimizer apart from the reported learning rate",
    "ResNet18 1-D stem/kernel implementation details",
    "target validation policy without target-label leakage",
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


PAPER_TASKS = ["d01->d23", "1-1->1-19", "1-1->8-8", "7-7->8-8", "14-7->3-19"]


def validate_paper_faithful_config(config: dict[str, Any]) -> dict[str, Any]:
    if bool(config.get("cvs_extension", False)):
        raise ValueError("paper-faithful Receiver-Agnostic UDA config cannot set cvs_extension=true")
    if str(config.get("paper_scope", "")).strip() != "paper_faithful":
        raise ValueError("paper_scope must be paper_faithful")
    if "ManySig" not in str(config.get("dataset", "")):
        raise ValueError("paper-faithful reproduction expects WiSig ManySig")
    total_receivers = int(config.get("total_receivers", 0))
    tx_count = int(config.get("tx_count", 0))
    if total_receivers != 12:
        raise ValueError("Bao et al. paper-faithful ManySig protocol expects 12 receivers")
    if tx_count != 6:
        raise ValueError("Bao et al. paper-faithful ManySig protocol expects 6 transmitters")
    if not bool(config.get("target_unlabeled_allowed", False)):
        raise ValueError("target unlabeled data must be allowed for the UDA stages")
    tasks = [str(v) for v in config.get("source_target_tasks", PAPER_TASKS)]
    if not tasks:
        raise ValueError("source_target_tasks cannot be empty")
    counts = [int(v) for v in config.get("source_receiver_counts", [])]
    if not counts:
        raise ValueError("source_receiver_counts cannot be empty")
    for count in counts:
        if count <= 0 or count >= total_receivers:
            raise ValueError("each source receiver count must be in [1,total_receivers)")
    checked = dict(config)
    checked["source_receiver_counts"] = counts
    checked["source_target_tasks"] = tasks
    checked["claim_boundary"] = "paper-faithful closed-set cross-receiver DA with unlabeled target adaptation"
    checked["paper_unspecified_fields"] = PAPER_UNSPECIFIED_FIELDS
    checked["paper_reported_hyperparameters"] = PAPER_REPORTED_HYPERPARAMETERS
    return checked


def build_receiver_ratio_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    total = int(config["total_receivers"])
    rows: list[dict[str, Any]] = []
    for source_count in config["source_receiver_counts"]:
        target_count = total - int(source_count)
        rows.append(
            {
                "ratio": f"{int(source_count)}:{target_count}",
                "source_receiver_count": int(source_count),
                "target_receiver_count": target_count,
                "closed_set_tx_count": int(config["tx_count"]),
                "target_data_role": "unlabeled_for_UDA_labeled_only_for_optional_finetune",
                "compare_methods": [
                    "source_only_lower_bound",
                    "target_labeled_retrain_upper_bound",
                    "DANN_baseline",
                    "MCD_baseline",
                    "SHOT_baseline",
                    "proposed_GAD_DVKL_CPL_class_weighting",
                ],
                "paper_tasks": list(config["source_target_tasks"]),
                "reported_hyperparameters": dict(PAPER_REPORTED_HYPERPARAMETERS),
                "primary_metric": "target receiver closed-set classification accuracy",
                "table_i_target_receiver_count": target_count if int(source_count) == 6 else None,
            }
        )
    return rows
