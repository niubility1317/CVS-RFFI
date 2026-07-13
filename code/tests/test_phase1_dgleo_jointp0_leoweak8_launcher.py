from __future__ import annotations

import importlib.util
import shlex
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "code" / "scripts" / "launch_phase1_dgleo_jointp0_leoweak8_20260713.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("jointp0_launcher", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_jointp0_matrix_is_one_full_joint_candidate_per_gpu():
    module = _load_module()
    rows = module.build_matrix()
    assert len(rows) == 8
    assert {int(row["gpu"]) for row in rows} == set(range(8))
    assert all(float(row["config"]["source_w"]) >= 0.05 for row in rows)
    assert all(float(row["config"]["dm_lambda"]) >= 0.025 for row in rows)
    assert all(float(row["config"]["proxy_w"]) >= 0.015 for row in rows)


def test_jointp0_commands_start_open_losses_at_epoch_one_and_use_leo_weak_eval():
    module = _load_module()
    root = Path("/srv/CV-SincNet")
    commands = [
        module.build_command(
            row,
            root=root,
            python=Path("/opt/python"),
            run_id="dry",
            wisig_pkl=root / "Dataset_WigSig" / "ManySig.pkl",
            teacher_ckpt=root / "teacher.pth",
        )
        for row in module.build_matrix()
    ]
    assert len({tuple(command) for command in commands}) == 8
    for command in commands:
        joined = shlex.join(command)
        assert "--source_episode_start_epoch 1" in joined
        assert "--direct_metric_start_epoch 1" in joined
        assert "--proxy_unknown_start_epoch 1" in joined
        assert "--u_direct_metric_start_epoch 1" in joined
        assert "--use_tx_rx_balanced_sampler true" in joined
        assert "--use_proto_memory true" in joined
        assert "--eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak" in joined
        assert "--sat_protocol_disjoint_required false" in joined
        assert "--checkpoint_selection final_only" in joined


def test_all_jointp0_commands_parse_with_current_trainer():
    module = _load_module()
    from SSDG.train_ssdg import build_arg_parser

    root = Path("/srv/CV-SincNet")
    for row in module.build_matrix():
        command = module.build_command(
            row,
            root=root,
            python=Path("/opt/python"),
            run_id="dry",
            wisig_pkl=root / "Dataset_WigSig" / "ManySig.pkl",
            teacher_ckpt=root / "teacher.pth",
        )
        parsed = build_arg_parser().parse_args(command[3:])
        assert parsed.source_episode_start_epoch == 1
        assert parsed.direct_metric_start_epoch == 1
        assert parsed.eval_sat_scenarios == module.LEO_WEAK
