from __future__ import annotations

from typing import Any


PAPER_UNSPECIFIED_FIELDS = [
    "conv kernel/channel schedule beyond four Conv-BN-ReLU-MaxPool blocks",
    "optimizer",
    "learning rate",
    "batch size",
    "epoch count",
    "GRL lambda schedule",
    "LMMD lambda",
    "LMMD kernel and layer set",
    "target-domain train/evaluation split versus transductive UDA reporting pool",
    "fine-tuning freeze policy",
]

PAPER_FIG7_SOURCE_RECEIVER_COUNTS = [1, 2, 3, 4, 6]

TABLE_I_PAPER_REFERENCE_ACCURACY = [0.89, 0.94, 0.87, 0.91, 0.92, 0.92]

PAPER_PREPROCESSING_REQUIRED = {
    "equalized": 1,
    "normalize": True,
    "normalization": "RMS/power normalization",
    "crop_mode": "left",
    "out_len": 256,
    "cfo_policy": "preserved",
}


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
    if int(config.get("capture_days", 0)) != 4:
        raise ValueError("Bao et al. paper-faithful ManySig protocol expects 4 capture days")
    if not bool(config.get("target_unlabeled_allowed", False)):
        raise ValueError("target unlabeled data must be allowed for the UDA stages")
    preprocessing = dict(config.get("preprocessing", {}))
    for key, expected in PAPER_PREPROCESSING_REQUIRED.items():
        if preprocessing.get(key) != expected:
            raise ValueError(f"paper-faithful preprocessing requires {key}={expected!r}")
    counts = [int(v) for v in config.get("source_receiver_counts", [])]
    if not counts:
        raise ValueError("source_receiver_counts cannot be empty")
    if counts != PAPER_FIG7_SOURCE_RECEIVER_COUNTS:
        raise ValueError("source_receiver_counts must match the Fig.7 paper matrix [1,2,3,4,6]")
    for count in counts:
        if count <= 0 or count >= total_receivers:
            raise ValueError("each source receiver count must be in [1,total_receivers)")
    checked = dict(config)
    checked["source_receiver_counts"] = counts
    checked["claim_boundary"] = "paper-faithful closed-set cross-receiver UDA"
    checked["paper_unspecified_fields"] = PAPER_UNSPECIFIED_FIELDS
    checked["table_i_paper_reference_accuracy"] = TABLE_I_PAPER_REFERENCE_ACCURACY
    checked["preprocessing"] = {
        **PAPER_PREPROCESSING_REQUIRED,
        "raw_sync_preamble_extraction": preprocessing.get(
            "raw_sync_preamble_extraction",
            "upstream WiSig/ManySig compact pkl prerequisite",
        ),
        "execution_status": "requires_real_manysig_pkl_for_runtime_evidence",
    }
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
                "receiver_split_policy": config.get("receiver_split_policy", "paper-unspecified"),
                "receiver_split_ids": None,
                "seed": None,
                "seed_policy": config.get("seed_policy", "paper-unspecified"),
                "compare_methods": [
                    "source_only_lower_bound",
                    "target_labeled_retrain_upper_bound",
                    "DANN_global_domain_adaptation",
                    "DANN_plus_LMMD_subdomain_adaptation",
                    "optional_uncertainty_finetune",
                ],
                "table_i_target_receiver_count": target_count if int(source_count) == 6 else None,
                "table_i_paper_reference_accuracy": TABLE_I_PAPER_REFERENCE_ACCURACY if int(source_count) == 6 else None,
                "table_i_reference_only": int(source_count) == 6,
                "fine_tune_required_for_fig8": int(source_count) < 4,
                "fine_tune_iterations_to_report": [0, 25, 50, 75, 100] if int(source_count) < 4 else [],
            }
        )
    return rows
