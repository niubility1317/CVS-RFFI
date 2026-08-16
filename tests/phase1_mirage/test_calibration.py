"""Behavioral checks for source-only MIRAGE threshold calibration."""

from __future__ import annotations

import importlib
from dataclasses import FrozenInstanceError, replace

import pytest

from cvsrffi.phase1_mirage.protocol import ProxyRole, SourcePartition


def _api():
    """Import lazily so RED demonstrates the absent calibration feature."""

    return importlib.import_module("cvsrffi.phase1_mirage.calibration")


def _known_row(api, index: int, *, fold: int = 0, inside: bool = True, risk: float = 0.10):
    return api.KnownScoreRow(
        physical_id=f"known-physical-{fold}-{index}",
        query_id=f"known-query-{fold}-{index}",
        quality=0.90,
        unknown_risk=risk,
        inside_registered_support=inside,
        predicted_class=index % 2,
        true_class=index % 2,
        receiver=f"rx-{index % 2}",
        day=index % 2,
        scene="clear" if index % 2 == 0 else "rain",
        fold=fold,
    )


def _proxy_row(
    api,
    index: int,
    *,
    fold: int = 0,
    inside: bool = False,
    risk: float = 0.90,
    quality: float = 0.90,
):
    return api.ProxyScoreRow(
        physical_id=f"proxy-physical-{fold}-{index}",
        query_id=f"proxy-query-{fold}-{index}",
        quality=quality,
        unknown_risk=risk,
        inside_registered_support=inside,
        predicted_class=index % 2,
        fold=fold,
    )


def _known_table(api, *, role=SourcePartition.V_CAL, rows=None, fold: int = 0):
    return api.KnownScoreTable(
        role=role,
        rows=tuple(rows if rows is not None else (_known_row(api, 0, fold=fold), _known_row(api, 1, fold=fold))),
        update_count=0,
    )


def _proxy_table(api, *, role=ProxyRole.P_CAL, rows=None, fold: int = 0, update_count: int = 0):
    return api.ProxyScoreTable(
        role=role,
        rows=tuple(rows if rows is not None else (_proxy_row(api, 0, fold=fold), _proxy_row(api, 1, fold=fold))),
        update_count=update_count,
    )


def test_calibration_freezes_a_feasible_head_threshold_tuple_from_v_cal_and_p_cal_only():
    """Catch calibration that bypasses head.decide, known FRR, or proxy priority."""

    api = _api()
    thresholds = api.calibrate_thresholds(
        known_scores=_known_table(api),
        proxy_scores=_proxy_table(api),
        max_known_frr=0.10,
    )

    assert thresholds.tau_reg <= thresholds.tau_unk
    calibration_decisions = api.freeze_calibration_decisions(
        _known_table(api),
        _proxy_table(api),
        thresholds,
    )
    assert api.known_false_rejection_rate(calibration_decisions) <= 0.10
    assert api.proxy_explicit_rejection_rate(calibration_decisions) == 1.0


def test_calibration_enforces_known_frr_and_returns_no_deployable_separation():
    """Catch a candidate that silently deploys when every known row is rejected."""

    api = _api()
    inseparable_known = _known_table(
        api,
        rows=(
            _known_row(api, 0, inside=False, risk=0.90),
            _known_row(api, 1, inside=False, risk=0.90),
        ),
    )

    with pytest.raises(api.NoDeployableSeparation, match="NO_DEPLOYABLE_SEPARATION"):
        api.calibrate_thresholds(
            known_scores=inseparable_known,
            proxy_scores=_proxy_table(api),
            max_known_frr=0.10,
        )


def test_formal_calibration_rejects_any_relaxed_known_frr_limit():
    """Catch a formal caller relaxing the approved ten-percent known-FRR hard ceiling."""

    api = _api()

    with pytest.raises(api.CalibrationProtocolError, match="0.10"):
        api.calibrate_thresholds(
            known_scores=_known_table(api),
            proxy_scores=_proxy_table(api),
            max_known_frr=0.1000001,
        )


def test_calibration_rejects_selection_target_overlap_and_mutating_proxy_inputs():
    """Catch role leakage, reused physical/query IDs, or a proxy update during calibration."""

    api = _api()
    with pytest.raises(api.CalibrationProtocolError, match="V_cal"):
        api.calibrate_thresholds(
            known_scores=_known_table(api, role=SourcePartition.V_SELECT),
            proxy_scores=_proxy_table(api),
        )
    with pytest.raises(api.CalibrationProtocolError, match="P_cal"):
        api.calibrate_thresholds(
            known_scores=_known_table(api),
            proxy_scores=_proxy_table(api, role=ProxyRole.P_SELECT),
        )
    with pytest.raises(api.CalibrationProtocolError, match="update_count"):
        api.calibrate_thresholds(
            known_scores=_known_table(api),
            proxy_scores=_proxy_table(api, update_count=1),
        )

    overlapping = replace(_proxy_row(api, 0), physical_id="known-physical-0-0")
    with pytest.raises(api.CalibrationProtocolError, match="overlap"):
        api.calibrate_thresholds(
            known_scores=_known_table(api),
            proxy_scores=_proxy_table(api, rows=(overlapping, _proxy_row(api, 1))),
        )


def test_tables_are_immutable_and_selection_decisions_cannot_use_calibration_roles():
    """Catch mutable score tables or post-freeze scoring that reads V_cal/P_cal."""

    api = _api()
    v_cal = _known_table(api)
    p_cal = _proxy_table(api)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        v_cal.rows += (_known_row(api, 2),)

    thresholds = api.calibrate_thresholds(v_cal, p_cal)
    v_select = _known_table(api, role=SourcePartition.V_SELECT)
    p_select = _proxy_table(api, role=ProxyRole.P_SELECT)
    decisions = api.freeze_selection_decisions(
        v_select,
        p_select,
        thresholds,
        candidate_id="A",
    )
    assert decisions.known_role is SourcePartition.V_SELECT
    assert decisions.proxy_role is ProxyRole.P_SELECT

    with pytest.raises(api.CalibrationProtocolError, match="V_select"):
        api.freeze_selection_decisions(v_cal, p_select, thresholds, candidate_id="A")
    with pytest.raises(api.CalibrationProtocolError, match="P_select"):
        api.freeze_selection_decisions(v_select, p_cal, thresholds, candidate_id="A")


def test_frozen_decision_tables_are_factory_only_and_reject_tampered_cross_ids():
    """Catch a leaked factory token, public construction, or score-time cross-ID reuse."""

    api = _api()
    scoring = importlib.import_module("cvsrffi.phase1_mirage.scoring")
    thresholds = api.calibrate_thresholds(_known_table(api), _proxy_table(api))
    decisions = api.freeze_selection_decisions(
        _known_table(api, role=SourcePartition.V_SELECT),
        _proxy_table(api, role=ProxyRole.P_SELECT),
        thresholds,
        candidate_id="A",
    )

    construction_kwargs = {
        "candidate_id": "forged",
        "known_role": SourcePartition.V_SELECT,
        "proxy_role": ProxyRole.P_SELECT,
        "thresholds": thresholds,
        "known_rows": decisions.known_rows,
        "proxy_rows": decisions.proxy_rows,
        "proxy_update_count": 0,
    }
    leaked_seal = getattr(decisions, "_factory_seal", object())
    with pytest.raises((TypeError, api.CalibrationProtocolError)):
        api.FrozenDecisionTable(**construction_kwargs)
    with pytest.raises((TypeError, api.CalibrationProtocolError)):
        api.FrozenDecisionTable(**construction_kwargs, _factory_seal=leaked_seal)
    assert not hasattr(decisions, "_factory_seal")
    assert scoring.score_same_row(decisions).candidate_id == "A"

    crossed_proxy = replace(
        decisions.proxy_rows[0],
        row=replace(decisions.proxy_rows[0].row, physical_id=decisions.known_rows[0].row.physical_id),
    )
    object.__setattr__(decisions, "proxy_rows", (crossed_proxy,) + decisions.proxy_rows[1:])
    with pytest.raises(scoring.ScoringProtocolError, match="receipt|overlap|sealed"):
        scoring.score_same_row(decisions)


def test_frozen_decision_table_rejects_public_no_argument_construction():
    """Catch an init=False dataclass silently producing a half-initialized public table."""

    api = _api()

    with pytest.raises(TypeError, match="use validated factory"):
        api.FrozenDecisionTable()


def test_same_row_scoring_rejects_half_initialized_frozen_decision_table():
    """Catch a score entrypoint leaking AttributeError from a bypassed half-table instance."""

    api = _api()
    scoring = importlib.import_module("cvsrffi.phase1_mirage.scoring")
    half_initialized = object.__new__(api.FrozenDecisionTable)

    with pytest.raises(scoring.ScoringProtocolError, match="frozen decision table"):
        scoring.score_same_row(half_initialized)


def test_score_table_column_factory_fails_closed_for_shape_range_and_duplicate_ids():
    """Catch malformed table columns before a threshold or decision is emitted."""

    api = _api()
    with pytest.raises(api.CalibrationProtocolError, match="same length"):
        api.KnownScoreTable.from_columns(
            role=SourcePartition.V_CAL,
            physical_ids=("p0",),
            query_ids=("q0", "q1"),
            qualities=(0.9,),
            unknown_risks=(0.1,),
            inside_registered_support=(True,),
            predicted_classes=(0,),
            true_classes=(0,),
            receivers=("rx",),
            days=(0,),
            scenes=("clear",),
            folds=(0,),
        )
    with pytest.raises(api.CalibrationProtocolError, match="duplicate"):
        api.ProxyScoreTable(
            role=ProxyRole.P_CAL,
            rows=(
                _proxy_row(api, 0),
                replace(_proxy_row(api, 1), query_id="proxy-query-0-0"),
            ),
            update_count=0,
        )
    with pytest.raises(api.CalibrationProtocolError, match=r"\[0, 1\]"):
        api.ProxyScoreTable(
            role=ProxyRole.P_CAL,
            rows=(replace(_proxy_row(api, 0), unknown_risk=1.1),),
            update_count=0,
        )
