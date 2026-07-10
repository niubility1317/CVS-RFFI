from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPT = ROOT / "code" / "scripts" / "launch_phase1_dgleo_p1verify128_20260710.py"
SPEC = importlib.util.spec_from_file_location("p1verify128", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

from SSDG import train_ssdg  # noqa: E402


def _command(row):
    root = Path("/srv/CV-SincNet")
    return MODULE.build_command(
        row,
        root=root,
        python=Path("/env/bin/python"),
        run_id="p1verify128_test",
        wisig_pkl=root / "Dataset_WigSig" / "ManySig.pkl",
        teacher_ckpt=root / "teacher.pth",
    )


def test_matrix_has_32_cells_four_paired_seeds_and_balanced_gpu_totals():
    rows = MODULE.build_matrix()
    assert len(rows) == 128
    assert len({row["candidate_id"] for row in rows}) == 128
    assert set(Counter(row["cell"] for row in rows).values()) == {4}
    assert Counter(row["gpu"] for row in rows) == Counter({gpu: 16 for gpu in range(8)})
    for cell in {row["cell"] for row in rows}:
        assert {row["seed"] for row in rows if row["cell"] == cell} == set(MODULE.PAIRED_SEEDS)


def test_all_rows_preserve_source_only_disjoint_satellite_and_final_only_protocol():
    rows = MODULE.build_matrix()
    assert all(row["source_only"] is True for row in rows)
    assert all(row["phase1_proxy_only"] is True for row in rows)
    assert all(row["checkpoint_selection"] == "final_only" for row in rows)
    assert all(row["sat_train_family"] != row["sat_eval_family"] for row in rows)
    commands = [tuple(_command(row)) for row in rows]
    assert len(set(commands)) == 128
    joined = "\n".join(" ".join(command) for command in commands)
    assert "ManyTx.pkl" not in joined
    assert "--checkpoint_selection final_only" in joined
    assert "--tail_rollback_enabled false" in joined
    assert "--sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak" in joined
    assert "--eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,geo_clear,mixed_orbit" in joined
    assert "--sat_protocol_disjoint_required true" in joined
    assert "--zid_leakage_probe_required true" in joined
    assert "--test_eval_start_epoch 999999" in joined
    assert "--test_eval_interval 0" in joined


def test_all_128_commands_parse_with_current_trainer_contract():
    parser = train_ssdg.build_arg_parser()
    for row in MODULE.build_matrix():
        parsed = parser.parse_args(_command(row)[3:])
        assert parsed.candidate_id == row["candidate_id"]
        assert parsed.checkpoint_selection == "final_only"
        assert parsed.sat_protocol_disjoint_required is True


def test_matrix_contains_required_full_mechanisms_and_diagnostic_ablations():
    rows = MODULE.build_matrix()
    by_cell = {row["cell"]: row for row in rows}
    assert by_cell["G0_FULL_BALANCED"]["config"]["dm_require_local"] is True
    assert by_cell["G4_LOCAL_OFF"]["config"]["dm_require_local"] is False
    assert by_cell["G2_LINV_OFF"]["config"]["l_rx_inv"] == 0.0
    assert by_cell["G2_LINV_FULL"]["config"]["l_channel_inv"] > 0.0
    assert by_cell["G3_UINV_OFF"]["config"]["u_channel_inv"] == 0.0
    assert by_cell["G3_UINV_FULL"]["config"]["u_channel_inv"] > 0.0
    assert by_cell["G5_U_OFF"]["config"]["u_dm"] == 0.0
    assert by_cell["G5_U_FULL"]["config"]["u_dm"] > 0.0
    assert by_cell["G7_KD_HEAVY"]["config"]["os_surgery"] is False
    assert by_cell["G7_OS_HIGH_PROTECT"]["config"]["os_surgery"] is True


def test_scheduler_cli_hard_limits_each_gpu_to_two_active_experiments():
    parser = MODULE.build_parser()
    args = parser.parse_args([])
    assert args.max_active_per_gpu == 2
    assert parser.parse_args(["--max-active-per-gpu", "2"]).max_active_per_gpu == 2


def test_non_source_phase1_dataset_is_rejected():
    MODULE.validate_source_wisig_pkl(Path("/srv/Dataset_WigSig/ManySig.pkl"))
    try:
        MODULE.validate_source_wisig_pkl(Path("/srv/Dataset_WigSig/ManyTx.pkl"))
    except ValueError as exc:
        assert "source-only ManySig.pkl" in str(exc)
    else:
        raise AssertionError("ManyTx must be rejected for Phase1 source-only training")
