import json
import importlib.util
from pathlib import Path

import pytest

from cvsrffi.meta_phase1_entry import (
    parse_args_for_test,
    validate_meta_phase1_config,
)


def valid_config():
    return {
        "schema": "cvs.phase1.meta_adapter.tri_r4.v1",
        "run_id": "phase1_test_r1",
        "seed": 392002,
        "base_checkpoint": "runs/base/best.pth",
        "wisig_pkl": "Dataset_WigSig/ManySig.pkl",
        "source_receiver_ids": [0, 1, 2, 3, 4, 5, 6],
        "source_roles": {"L_s": 0.07, "U_s": 0.63, "V_cal": 0.15, "V_select": 0.15},
        "adapter": {
            "rank": 4,
            "sites": ["time", "freq", "fusion"],
            "inner_steps": 3,
            "deployment_max_steps": 5,
            "source_diagnostic_max_steps": 10,
        },
        "episode_weights": {
            "Q_SAME_DOMAIN": 0.40,
            "Q_RX_HOLDOUT": 0.20,
            "Q_DAY_CHANNEL_HOLDOUT": 0.15,
            "Q_CLEAN_TO_LEO": 0.15,
            "Q_LEO_CROSS": 0.10,
        },
        "k_choices": [1, 2, 5, 10],
        "meta_batch_size": 4,
        "phase1c_backbone_lr_ratio": 0.05,
        "evaluate_steps": [0, 1, 3, 5, 10],
    }


def test_meta_adapter_cli_defaults_are_v1_locked():
    args = parse_args_for_test(["--use_cvs_meta_adapter"])
    assert args.use_cvs_meta_adapter is True
    assert args.meta_adapter_rank == 4
    assert args.meta_adapter_sites == "time,freq,fusion"
    assert args.meta_inner_steps == 3
    assert args.meta_inner_max_steps == 5


def test_phase1_entry_rejects_noncanonical_source_ratios():
    config = valid_config()
    config["source_roles"]["L_s"] = 0.10
    with pytest.raises(ValueError, match=r"0\.07"):
        validate_meta_phase1_config(config)


def test_phase1_config_requires_explicit_source_receiver_ids():
    config = valid_config()
    del config["source_receiver_ids"]
    with pytest.raises(ValueError, match="source_receiver_ids"):
        validate_meta_phase1_config(config)


def test_phase1_config_rejects_target_receiver_fields():
    config = valid_config()
    config["target_receiver_ids"] = [7]
    with pytest.raises(ValueError, match="target receiver"):
        validate_meta_phase1_config(config)


def test_launcher_dry_run_does_not_create_output_root(tmp_path, capsys):
    config = valid_config()
    config_path = tmp_path / "meta.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_root = tmp_path / "run-root"

    launcher_path = Path(__file__).resolve().parents[1] / "scripts" / "launch_phase1_adv3b02_meta_adapter_tri_r4_v1.py"
    spec = importlib.util.spec_from_file_location("meta_phase1_launcher", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(launcher)
    main = launcher.main

    main(["--config", str(config_path), "--output-root", str(output_root), "--dry-run"])
    captured = capsys.readouterr().out
    assert "phase1_test_r1" in captured
    assert str(output_root) in captured
    assert not output_root.exists()

    output_root.mkdir()
    with pytest.raises(FileExistsError):
        main(["--config", str(config_path), "--output-root", str(output_root), "--dry-run"])
