import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
CODE = ROOT / "code"
for path in (TOOLS, CODE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cen51_wisig_subset_kseg_transfer_matrix as matrix  # noqa: E402


def _arg_value(args, flag):
    return args[args.index(flag) + 1]


def test_subset_matrix_shape_and_gpu_balance():
    rows = matrix.assign_rows()

    assert len(rows) == 48
    assert len({row.run_name for row in rows}) == 48
    assert Counter(row.shot for row in rows) == {5: 8, 10: 8, 20: 8, 30: 8, 50: 8, 100: 8}
    assert Counter(row.gpu for row in rows) == {gpu: 6 for gpu in range(8)}

    by_gpu = defaultdict(set)
    for row in rows:
        by_gpu[row.gpu].add(row.shot)
    assert all(shots == set(matrix.SHOT_ORDER) for shots in by_gpu.values())


def test_subset_arguments_replace_original_split():
    rows = matrix.assign_rows()

    for row in rows:
        assert _arg_value(row.args, "--wisig_train_ratio") == "0.1"
        assert _arg_value(row.args, "--wisig_train_days") == "1,2"
        assert _arg_value(row.args, "--wisig_test_days") == "0,3"
        assert _arg_value(row.args, "--wisig_train_rxs") == "2,3,4,5,8,9,10"
        assert _arg_value(row.args, "--wisig_test_rxs") == "0,1,6,7,11"
        assert _arg_value(row.args, "--wisig_domain") == "rx_day"
        assert "0,1,2,3,4,5,6" not in row.args
        assert "7,8,9,10,11" not in row.args


def test_global_unified_control_present_for_every_k():
    rows = matrix.assign_rows()
    global_rows = [row for row in rows if row.axis == "global_unified"]

    assert Counter(row.shot for row in global_rows) == {5: 1, 10: 1, 20: 1, 30: 1, 50: 1, 100: 1}
    assert all(row.action == "same_pressure_sat_rx" for row in global_rows)
