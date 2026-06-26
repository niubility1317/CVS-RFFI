import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for path in (TOOLS,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cen51_highratio_kseg_controller_matrix as matrix  # noqa: E402


def _args(row):
    raw = matrix.arg_pairs(row.params)
    parsed = {}
    i = 0
    while i < len(raw):
        token = raw[i]
        if token.startswith("--"):
            if i + 1 < len(raw) and not raw[i + 1].startswith("--"):
                parsed[token] = raw[i + 1]
                i += 2
            else:
                parsed[token] = True
                i += 1
        else:
            i += 1
    return parsed


def test_highratio_matrix_shape_and_balance():
    rows = matrix.make_candidates()

    assert len(rows) == 36
    assert len({row.cid for row in rows}) == 36
    assert len({row.run_name for row in rows}) == 36
    assert Counter(row.strategy for row in rows) == {
        "KONLY_B03": 9,
        "NEFF_RXSAT": 9,
        "RATIO_STRICT": 9,
        "TOTAL_OVER": 9,
    }
    assert Counter(f"{row.ratio:.1f}" for row in rows) == {"0.2": 12, "0.3": 12, "0.5": 12}
    assert max(Counter(row.gpu for row in rows).values()) <= 5
    assert set(Counter(row.gpu for row in rows)) == set(range(8))


def test_every_ratio_k_point_has_four_strategies():
    rows = matrix.make_candidates()
    grouped = defaultdict(set)
    for row in rows:
        grouped[(row.ratio, row.k_cap)].add(row.strategy)

    expected_points = {
        (0.2, 100),
        (0.2, 150),
        (0.2, 200),
        (0.3, 150),
        (0.3, 225),
        (0.3, 300),
        (0.5, 250),
        (0.5, 375),
        (0.5, 500),
    }
    assert set(grouped) == expected_points
    assert all(strategies == {"KONLY_B03", "NEFF_RXSAT", "RATIO_STRICT", "TOTAL_OVER"} for strategies in grouped.values())


def test_subset_and_training_contract_args():
    rows = matrix.make_candidates()

    for row in rows:
        args = _args(row)
        assert args["--train_mode"] == "centralized"
        assert "--fl_rounds" not in args
        assert "--fl_client_key" not in args
        assert args["--wisig_protocol"] == "cvs_day_rx"
        assert args["--wisig_domain"] == "rx_day"
        assert args["--wisig_train_days"] == "1,2"
        assert args["--wisig_test_days"] == "0,3"
        assert args["--wisig_train_rxs"] == "2,3,4,5,8,9,10"
        assert args["--wisig_test_rxs"] == "0,1,6,7,11"
        assert args["--wisig_train_ratio"] == f"{row.ratio:.1f}"
        assert row.n_eff_nominal == row.k_cap * matrix.TRAIN_DOMAIN_COMBOS


def test_total_over_is_actual_negative_boundary():
    rows = matrix.make_candidates()
    total_over = [row for row in rows if row.strategy == "TOTAL_OVER"]
    neff = [row for row in rows if row.strategy == "NEFF_RXSAT"]

    assert all(float(_args(row)["--sat_view_prob"]) >= 0.88 for row in total_over)
    assert all(float(_args(row)["--concat_sat_ce_weight"]) >= 1.05 for row in total_over)
    assert min(float(_args(row)["--lambda_group_ce"]) for row in total_over) > max(float(_args(row)["--lambda_group_ce"]) for row in neff)
