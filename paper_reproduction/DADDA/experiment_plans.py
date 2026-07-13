from __future__ import annotations

from typing import Any


PAPER_TABLE2_METHODS = ["source_only", "dann", "dan", "dsan", "wd", "dcoral", "cdan", "dadda"]
IMPLEMENTED_TABLE2_METHODS = ["source_only", "dadda", "proposed"]
SNR_FIG5_TASKS = ["20-1->2-1", "1-19->2-19"]
SNR_FIG5_DB = [0, 5, 10, 15, 20]

DADDA_ABLATION_VARIANTS = [
    {
        "variant": "B",
        "use_multiscale": False,
        "use_mmd": False,
        "use_lmmd": False,
        "alpha_mode": "none",
    },
    {
        "variant": "B+M",
        "use_multiscale": True,
        "use_mmd": False,
        "use_lmmd": False,
        "alpha_mode": "none",
    },
    {
        "variant": "B+MF",
        "use_multiscale": False,
        "use_mmd": True,
        "use_lmmd": False,
        "alpha_mode": "none",
    },
    {
        "variant": "B+L",
        "use_multiscale": False,
        "use_mmd": False,
        "use_lmmd": True,
        "alpha_mode": "none",
    },
    {
        "variant": "B+M+L",
        "use_multiscale": True,
        "use_mmd": False,
        "use_lmmd": True,
        "alpha_mode": "none",
    },
    {
        "variant": "B+M+MF",
        "use_multiscale": True,
        "use_mmd": True,
        "use_lmmd": False,
        "alpha_mode": "none",
    },
    {
        "variant": "B+MF+L",
        "use_multiscale": False,
        "use_mmd": True,
        "use_lmmd": True,
        "alpha_mode": "fixed_0p5",
    },
    {
        "variant": "DADDA",
        "use_multiscale": True,
        "use_mmd": True,
        "use_lmmd": True,
        "alpha_mode": "dynamic",
    },
]


def build_snr_fig5_plan() -> dict[str, Any]:
    return {
        "paper_item": "Fig.5",
        "artifact_type": "snr_robustness_plan",
        "claim_status": "plan_only_not_formal_result",
        "formal_result": False,
        "noise": "AWGN",
        "snr_db": list(SNR_FIG5_DB),
        "tasks": [
            {"task": task, "snr_db": list(SNR_FIG5_DB), "methods": ["source_only", "dadda"]}
            for task in SNR_FIG5_TASKS
        ],
        "missing_for_formal_result": ["trained checkpoints", "real WiSig evaluation", "accuracy curves"],
    }


def build_table3_ablation_plan() -> dict[str, Any]:
    return {
        "paper_item": "Table III",
        "artifact_type": "module_ablation_plan",
        "claim_status": "plan_only_not_formal_result",
        "formal_result": False,
        "variants": [dict(item) for item in DADDA_ABLATION_VARIANTS],
        "missing_for_formal_result": ["variant training/evaluation runner", "real WiSig accuracy table"],
    }


def build_table4_alpha_plan() -> dict[str, Any]:
    return {
        "paper_item": "Table IV",
        "artifact_type": "dynamic_alpha_ablation_plan",
        "claim_status": "plan_only_not_formal_result",
        "formal_result": False,
        "variants": [
            {"variant": "fixed_0p5", "alpha_mode": "fixed", "fixed_alpha": 0.5},
            {"variant": "dynamic", "alpha_mode": "dynamic", "fixed_alpha": None},
        ],
        "missing_for_formal_result": ["fixed-alpha runner", "dynamic-alpha runner", "real WiSig accuracy table"],
    }


def build_analysis_artifact_plan() -> dict[str, Any]:
    common_fields = [
        "paper_item",
        "task",
        "method",
        "checkpoint_path",
        "dataset_sha256",
        "device",
        "seed",
        "claim_status",
        "formal_result",
        "source_receiver",
        "target_receiver",
        "sample_count",
        "output_files",
    ]
    artifacts = [
        {
            "paper_item": "Fig.6",
            "artifact_type": "a_distance_plan",
            "required_fields": common_fields + ["feature_layer", "split", "label_scope"],
        },
        {
            "paper_item": "Fig.7",
            "artifact_type": "tsne_plan",
            "required_fields": common_fields + ["feature_layer", "embedding_path", "label_scope"],
        },
        {
            "paper_item": "Fig.8",
            "artifact_type": "confusion_matrix_plan",
            "required_fields": common_fields + ["y_true", "y_pred", "label_scope"],
        },
        {
            "paper_item": "Table V",
            "artifact_type": "complexity_plan",
            "required_fields": common_fields + ["kernel_setting", "param_count", "macs_or_flops_estimate", "flop_count_scope"],
        },
        {
            "paper_item": "Table VI",
            "artifact_type": "timing_plan",
            "required_fields": common_fields + ["train_epoch_seconds", "test_epoch_seconds", "warmup", "iters", "hardware_note"],
        },
    ]
    return {
        "paper_item": "Fig.6-8/Table V/Table VI",
        "artifact_type": "analysis_artifact_plan",
        "claim_status": "plan_only_not_formal_result",
        "formal_result": False,
        "artifacts": artifacts,
        "missing_for_formal_result": ["checkpoint analysis", "feature export", "plot/table generation"],
    }


def build_pending_paper_artifacts() -> list[dict[str, Any]]:
    return [
        build_snr_fig5_plan(),
        build_table3_ablation_plan(),
        build_table4_alpha_plan(),
        build_analysis_artifact_plan(),
    ]


def build_paper_artifact_plan() -> dict[str, Any]:
    return {
        "method_id": "dadda_cross_receiver",
        "artifact_type": "paper_artifact_plan",
        "claim_status": "plan_only_not_formal_result",
        "formal_result": False,
        "table2_methods": {
            "required": list(PAPER_TABLE2_METHODS),
            "implemented": list(IMPLEMENTED_TABLE2_METHODS),
            "missing": [method for method in PAPER_TABLE2_METHODS if method not in {"source_only", "dadda"}],
        },
        "pending_paper_artifacts": build_pending_paper_artifacts(),
    }
