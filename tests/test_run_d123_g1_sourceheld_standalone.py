from __future__ import annotations

import inspect

from scripts import run_d123_g1_sourceheld_standalone as runner


def test_runner_identity_and_four_arm_surface_are_frozen() -> None:
    assert runner.CANDIDATE_ID == "D123_LOO_CRES_GROUND_HEAD"
    assert runner.PREDICTION_SCHEMA.startswith("cvs.d123.")
    assert runner.SCORE_SCHEMA.startswith("cvs.d123.")
    assert runner.ARMS == ("M0", "M_DA", "M_HEAD", "M_JOINT")
    parameters = set(inspect.signature(runner._build_four_arm_predictions).parameters)
    assert parameters == {
        "bundle",
        "rdce_asset",
        "support_signed",
        "labels",
        "query_signed",
        "registry",
        "k_shot",
        "package_sha256",
    }
    assert not parameters & {"held_class", "truth", "role", "quota", "selection"}


def test_overlay_changes_only_candidate_method_surface(monkeypatch) -> None:
    original_builder = runner.base._build_four_arm_predictions
    monkeypatch.setattr(runner.base, "CANDIDATE_ID", "old")
    monkeypatch.setattr(runner.base, "PREDICTION_SCHEMA", "old.prediction")
    monkeypatch.setattr(runner.base, "SCORE_SCHEMA", "old.score")
    monkeypatch.setattr(runner.base, "_build_four_arm_predictions", original_builder)
    runner._install_overlay()
    assert runner.base.CANDIDATE_ID == runner.CANDIDATE_ID
    assert runner.base.PREDICTION_SCHEMA == runner.PREDICTION_SCHEMA
    assert runner.base.SCORE_SCHEMA == runner.SCORE_SCHEMA
    assert runner.base._build_four_arm_predictions is runner._build_four_arm_predictions


def test_main_installs_overlay_before_reusing_sealed_lifecycle(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(runner.base, "CANDIDATE_ID", runner.base.CANDIDATE_ID)
    monkeypatch.setattr(runner.base, "PREDICTION_SCHEMA", runner.base.PREDICTION_SCHEMA)
    monkeypatch.setattr(runner.base, "SCORE_SCHEMA", runner.base.SCORE_SCHEMA)
    monkeypatch.setattr(
        runner.base,
        "_build_four_arm_predictions",
        runner.base._build_four_arm_predictions,
    )

    def fake_main(argv):
        captured.update(
            candidate=runner.base.CANDIDATE_ID,
            prediction=runner.base.PREDICTION_SCHEMA,
            score=runner.base.SCORE_SCHEMA,
            builder=runner.base._build_four_arm_predictions,
            argv=argv,
        )
        return 17

    monkeypatch.setattr(runner.base, "main", fake_main)
    assert runner.main(["predict"]) == 17
    assert captured == {
        "candidate": runner.CANDIDATE_ID,
        "prediction": runner.PREDICTION_SCHEMA,
        "score": runner.SCORE_SCHEMA,
        "builder": runner._build_four_arm_predictions,
        "argv": ["predict"],
    }
