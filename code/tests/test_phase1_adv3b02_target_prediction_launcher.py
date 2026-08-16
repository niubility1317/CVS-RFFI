from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
LAUNCHER = (
    CODE_ROOT
    / "scripts"
    / "launch_phase1_adv3b02_target_prediction6_v1_20260816.sh"
)
SMOKE_ENTRY = CODE_ROOT / "smoke_phase1_adv3b02_target_prediction_f1.py"
SMOKE_LAUNCHER = (
    CODE_ROOT
    / "scripts"
    / "smoke_phase1_adv3b02_target_prediction_f1_v2_20260816.sh"
)
V2_LAUNCHER = (
    CODE_ROOT
    / "scripts"
    / "launch_phase1_adv3b02_target_prediction6_v2_20260816.sh"
)
TARGET_REFERENCE_TEST = CODE_ROOT / "tests" / "test_phase1_adv3b02_target_reference.py"
RUN_ID = "phase1_adv3b02_target_prediction_20260816_v1"
V2_RUN_ID = "phase1_adv3b02_target_prediction_20260816_v2"
V2_SMOKE_ID = ".smoke_phase1_adv3b02_target_prediction_20260816_v2_F1"
PROJECT_ROOT = "/home/szu2070436088/2510044040/CV-SincNet"
ADV_RUN_ID = "phase1_adv3b02_clic6_20260816_v2"
CLEAN_RUN_ID = "phase1_clic_postfreeze_20260812_v4"
TARGET_PACKAGE_RUN_ID = "phase1_clic_target_prediction_20260812_v1"


def _git_bash_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    relative = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/{drive}/{relative}"


def _run_launcher(
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    assert LAUNCHER.is_file(), "ADV target prediction launcher is absent"
    return _run_script(LAUNCHER, *arguments, environment=environment)


def _run_script(
    script: Path,
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(GIT_BASH), _git_bash_path(script), *arguments],
        cwd=CODE_ROOT.parent,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )


def _run_v2_smoke_launcher(
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    assert SMOKE_LAUNCHER.is_file(), "ADV v2 file-backed smoke launcher is absent"
    return _run_script(SMOKE_LAUNCHER, *arguments, environment=environment)


def _run_v2_formal_launcher(
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    assert V2_LAUNCHER.is_file(), "ADV v2 formal prediction launcher is absent"
    return _run_script(V2_LAUNCHER, *arguments, environment=environment)


def _load_file_module(path: Path, *, name: str, missing: str):
    if not path.is_file():
        pytest.fail(missing)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dry_run_lines() -> list[str]:
    result = _run_launcher("--dry-run")
    assert result.returncode == 0, result.stderr
    return [line for line in result.stdout.splitlines() if line.strip()]


def _command_tokens(line: str) -> list[str]:
    tokens = shlex.split(line)
    python_index = next(
        index
        for index, token in enumerate(tokens)
        if Path(token).name in {"python", "python3"}
    )
    return tokens[python_index:]


def _project_environment(project_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PROJECT_ROOT"] = _git_bash_path(project_root)
    environment["CODE_ROOT"] = CODE_ROOT.as_posix()
    environment["PYTHON"] = Path(sys.executable).as_posix()
    environment.pop("BASH_ENV", None)
    return environment


def _write_path_shaped_invalid_inputs(project_root: Path) -> None:
    package = (
        project_root
        / "runs"
        / TARGET_PACKAGE_RUN_ID
        / "sealed_target"
        / "iq_only_package"
    )
    for member in ("manifest.json", "received_iq.npz"):
        path = package / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"invalid-test-input")
    for fold in range(1, 7):
        adv = (
            project_root
            / "runs"
            / ADV_RUN_ID
            / f"F{fold}_ADV3B02_CLIC"
        )
        for name in ("final_ssdg.pth", "phase1_training_completion_receipt.json"):
            path = adv / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"invalid-test-input")
        for arm in ("C", "G"):
            clic = (
                project_root
                / "runs"
                / "phase1_clic12_20260812_v5"
                / f"F{fold}{arm}_CLIC12"
            )
            clean = (
                project_root
                / "runs"
                / CLEAN_RUN_ID
                / f"F{fold}{arm}_CLIC12"
                / "source_clean_proxy.npz"
            )
            for path in (
                clic / "final_ssdg.pth",
                clic / "phase1_clic_terminal_receipt.json",
                clean,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"invalid-test-input")


def test_launcher_dry_run_emits_six_seals_then_six_publishes() -> None:
    """Break caught: a fold is skipped, duplicated, or published before all seals."""

    lines = _dry_run_lines()
    assert len(lines) == 12
    assert all("stage=ADV_TRAIN_CONFIG_SEAL" in line for line in lines[:6])
    assert all("stage=ADV_TARGET_PREDICTION" in line for line in lines[6:])
    for fold in range(1, 7):
        assert f"fold={fold}" in lines[fold - 1]
        assert f"fold={fold}" in lines[fold + 5]
        assert f"candidate=F{fold}_ADV3B02_CLIC" in lines[fold - 1]
        assert f"candidate=F{fold}_ADV3B02_CLIC" in lines[fold + 5]


def test_launcher_dry_run_binds_real_authorities_and_gpu_mapping() -> None:
    """Break caught: a fold uses the wrong checkpoint, receipt, clean arm, or GPU."""

    lines = _dry_run_lines()
    for fold, seal_line, publish_line in zip(
        range(1, 7), lines[:6], lines[6:], strict=True
    ):
        candidate = f"F{fold}_ADV3B02_CLIC"
        adv_root = f"{PROJECT_ROOT}/runs/{ADV_RUN_ID}/{candidate}"
        output_root = f"{PROJECT_ROOT}/runs/{RUN_ID}/{candidate}"
        canonical_clean = (
            f"{PROJECT_ROOT}/runs/{CLEAN_RUN_ID}/F{fold}C_CLIC12/"
            "source_clean_proxy.npz"
        )
        peer_clean = (
            f"{PROJECT_ROOT}/runs/{CLEAN_RUN_ID}/F{fold}G_CLIC12/"
            "source_clean_proxy.npz"
        )
        assert f"physical_gpu=CPU" in seal_line
        assert f"physical_gpu={fold - 1}" in publish_line
        assert f"{adv_root}/final_ssdg.pth" in seal_line
        assert f"{adv_root}/phase1_training_completion_receipt.json" in seal_line
        assert f"canonical_clean={canonical_clean}" in seal_line
        assert f"peer_clean={peer_clean}" in seal_line
        assert "clean_binding_proof=STRICT_C_G_METADATA_EQUAL" in seal_line
        assert f"{output_root}/train_data_config.json" in seal_line
        assert f"{adv_root}/final_ssdg.pth" in publish_line
        assert f"{adv_root}/phase1_training_completion_receipt.json" in publish_line
        assert f"{output_root}/train_data_config.json" in publish_line
        assert f"{output_root}/target_prediction.json" in publish_line
        assert (
            f"{PROJECT_ROOT}/runs/{TARGET_PACKAGE_RUN_ID}/sealed_target/"
            "iq_only_package"
        ) in publish_line


def test_launcher_dry_run_keeps_sealer_source_only_and_publisher_four_input_blind() -> None:
    """Break caught: clean enters publisher or truth/config/reference flags enter either stage."""

    lines = _dry_run_lines()
    seal_options = {
        "--seal-train-data-config",
        "--checkpoint",
        "--completion-receipt-json",
        "--clean-v4-npz",
        "--output",
    }
    publish_options = {
        "--publish-target-prediction",
        "--checkpoint",
        "--completion-receipt-json",
        "--train-config-manifest",
        "--iq-only-package",
        "--output",
    }
    for line in lines[:6]:
        options = {token for token in _command_tokens(line) if token.startswith("--")}
        assert options == seal_options
    for line in lines[6:]:
        options = {token for token in _command_tokens(line) if token.startswith("--")}
        assert options == publish_options
        assert "--clean-v4-npz" not in options
    all_options = {
        token
        for line in lines
        for token in _command_tokens(line)
        if token.startswith("--")
    }
    for forbidden in (
        "--truth",
        "--known",
        "--reference",
        "--metric",
        "--score",
        "--role",
        "--query",
        "--selection",
        "--retry",
    ):
        assert all(not option.startswith(forbidden) for option in all_options)


@pytest.mark.parametrize("root_kind", ("runs", "logs"))
def test_launcher_rejects_preexisting_run_or_log_root_before_inputs(
    tmp_path: Path, root_kind: str
) -> None:
    """Break caught: a second invocation resumes or overwrites a prior run."""

    project_root = tmp_path / "project"
    collision = project_root / root_kind / RUN_ID
    collision.mkdir(parents=True)
    result = _run_launcher(environment=_project_environment(project_root))
    assert result.returncode == 3
    assert "refusing to overwrite ADV target prediction run/log root" in result.stderr
    other_kind = "logs" if root_kind == "runs" else "runs"
    assert not (project_root / other_kind / RUN_ID).exists()


def test_launcher_rejects_clean_metadata_failure_before_any_output(tmp_path: Path) -> None:
    """Break caught: canonical C is selected without strict same-fold C/G metadata proof."""

    project_root = tmp_path / "project"
    _write_path_shaped_invalid_inputs(project_root)
    result = _run_launcher(environment=_project_environment(project_root))
    assert result.returncode != 0
    assert "clean-v4" in result.stderr or "metadata" in result.stderr
    assert not (project_root / "runs" / RUN_ID).exists()
    assert not (project_root / "logs" / RUN_ID).exists()


@pytest.mark.parametrize(
    "forbidden_argument",
    (
        "--truth-sidecar",
        "--known-test-config",
        "--adv-reference",
        "--retry",
    ),
)
def test_launcher_rejects_forbidden_or_unknown_arguments(
    forbidden_argument: str,
) -> None:
    """Break caught: a caller expands the frozen blind launcher input surface."""

    result = _run_launcher(forbidden_argument)
    assert result.returncode == 2
    assert f"invalid argument: {forbidden_argument}" in result.stderr


def test_v2_smoke_dry_run_is_one_file_backed_f1_strict_forward() -> None:
    """Break caught: detached smoke depends on SSH stdin or exposes target inputs."""

    assert SMOKE_ENTRY.is_file(), "ADV v2 file-backed smoke Python entry is absent"
    result = _run_v2_smoke_launcher("--dry-run")
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    line = lines[0]
    assert "stage=ADV_TARGET_PREDICTION_F1_TECHNICAL_SMOKE" in line
    assert f"run_id={V2_RUN_ID}" in line
    assert "fold=1" in line
    assert "physical_gpu=0" in line
    assert "SMOKE_INVOCATION=1" in line
    assert "FORMAL_INVOCATION=0" in line
    assert "retry=NO" in line
    tokens = _command_tokens(line)
    assert Path(tokens[2]).name == SMOKE_ENTRY.name
    options = {token for token in tokens if token.startswith("--")}
    assert options == {
        "--run-smoke",
        "--checkpoint",
        "--completion-receipt-json",
        "--clean-v4-npz",
        "--train-config-output",
        "--receipt-output",
    }
    f1_adv = f"{PROJECT_ROOT}/runs/{ADV_RUN_ID}/F1_ADV3B02_CLIC"
    assert f"{f1_adv}/final_ssdg.pth" in line
    assert f"{f1_adv}/phase1_training_completion_receipt.json" in line
    assert (
        f"{PROJECT_ROOT}/runs/{CLEAN_RUN_ID}/F1C_CLIC12/source_clean_proxy.npz"
        in line
    )
    assert f"{PROJECT_ROOT}/runs/{V2_SMOKE_ID}/F1_ADV3B02_CLIC" in line
    for forbidden in (
        "iq-only-package",
        "truth",
        "known",
        "reference",
        "metric",
        "score",
        "role",
        "query",
        "selection",
        "retry-count",
    ):
        assert forbidden not in line.lower()


@pytest.mark.parametrize("root_kind", ("runs", "logs"))
def test_v2_smoke_rejects_existing_root_before_opening_inputs(
    tmp_path: Path, root_kind: str
) -> None:
    """Break caught: the one-shot smoke resumes or overwrites partial evidence."""

    project_root = tmp_path / "project"
    collision = project_root / root_kind / V2_SMOKE_ID
    collision.mkdir(parents=True)
    result = _run_v2_smoke_launcher(
        environment=_project_environment(project_root)
    )
    assert result.returncode == 3
    assert "refusing to overwrite ADV target prediction v2 smoke run/log root" in result.stderr
    other_kind = "logs" if root_kind == "runs" else "runs"
    assert not (project_root / other_kind / V2_SMOKE_ID).exists()


def test_v2_smoke_entry_runs_real_strict_model_once_without_target_inputs(
    tmp_path: Path,
) -> None:
    """Break caught: the file entry skips strict reconstruction or opens target state."""

    helpers = _load_file_module(
        TARGET_REFERENCE_TEST,
        name="_adv3b02_target_reference_smoke_helpers",
        missing="Task2 target-reference fixture is absent",
    )
    smoke = _load_file_module(
        SMOKE_ENTRY,
        name="_adv3b02_target_prediction_f1_smoke_v2",
        missing="ADV v2 file-backed smoke Python entry is absent",
    )
    paths = helpers._write_training_authorities(tmp_path, input_len=256)
    helpers._install_real_ssdg_model_state(paths, input_len=256)
    output_root = tmp_path / "runs" / V2_SMOKE_ID / "F1_ADV3B02_CLIC"
    output_root.mkdir(parents=True)
    train_config = output_root / "train_data_config.json"
    receipt = output_root / "technical_smoke_receipt.json"

    result = smoke.run_f1_technical_smoke(
        checkpoint_path=paths["checkpoint"],
        completion_receipt_path=paths["completion"],
        clean_v4_npz_path=paths["clean"],
        train_config_output_path=train_config,
        receipt_output_path=receipt,
    )

    assert result == receipt.resolve()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    expected = {
        "schema": "cvs.phase1.adv3b02_target_prediction_technical_smoke.v2",
        "completed": True,
        "claim": "NO_PERFORMANCE_RESULT",
        "run_id": V2_RUN_ID,
        "adv_training_run_id": ADV_RUN_ID,
        "candidate_id": "F1_ADV3B02_CLIC",
        "fold": "F1",
        "scene": "leo_clear_weak",
        "input_shape": [2, 256],
        "source_class_count": 4,
        "finite_logit_count": 4,
        "strict_runtime_load": True,
        "synthetic_local_input_count": 1,
        "forward_count": 1,
        "target_fit_rows": 0,
        "target_update_rows": 0,
        "target_retry_count": 0,
        "target_selection_count": 0,
        "target_selection_feedback": False,
        "iq_only_package_opened": False,
        "truth_sidecar_opened": False,
        "known_test_config_opened": False,
        "reference_opened": False,
        "metrics_opened": False,
        "formal_invocation": 0,
        "retry_authorized": False,
    }
    for field, value in expected.items():
        assert type(payload.get(field)) is type(value)
        assert payload[field] == value
    sha256 = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert payload["checkpoint_sha256"] == sha256(paths["checkpoint"])
    assert payload["completion_receipt_sha256"] == sha256(paths["completion"])
    assert payload["clean_v4_npz_sha256"] == sha256(paths["clean"])
    assert payload["train_config_manifest_sha256"] == sha256(train_config)
    assert smoke.validate_f1_technical_smoke_receipt(
        checkpoint_path=paths["checkpoint"],
        completion_receipt_path=paths["completion"],
        train_config_manifest_path=train_config,
        receipt_path=receipt,
    ) == receipt.resolve()
    assert not any(
        marker in path.name.lower()
        for path in tmp_path.rglob("*")
        for marker in ("package", "truth", "known", "reference", "metric", "query")
    )

    second_config = output_root / "second_train_data_config.json"
    with pytest.raises(
        smoke.ADV3B02TargetSmokeError, match="receipt output already exists"
    ):
        smoke.run_f1_technical_smoke(
            checkpoint_path=paths["checkpoint"],
            completion_receipt_path=paths["completion"],
            clean_v4_npz_path=paths["clean"],
            train_config_output_path=second_config,
            receipt_output_path=receipt,
        )
    assert not second_config.exists()


def test_v2_smoke_entry_cli_excludes_target_truth_and_performance_inputs() -> None:
    """Break caught: the technical smoke accepts a target/package/scorer input."""

    smoke = _load_file_module(
        SMOKE_ENTRY,
        name="_adv3b02_target_prediction_f1_smoke_v2_cli",
        missing="ADV v2 file-backed smoke Python entry is absent",
    )
    parser = smoke.build_arg_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert option_strings == {
        "-h",
        "--help",
        "--run-smoke",
        "--validate-receipt",
        "--checkpoint",
        "--completion-receipt-json",
        "--clean-v4-npz",
        "--train-config-output",
        "--receipt-output",
    }


def test_v2_formal_dry_run_has_fresh_identity_and_twelve_commands() -> None:
    """Break caught: formal execution reuses/overwrites the stopped v1 run."""

    result = _run_v2_formal_launcher("--dry-run")
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 12
    assert all("stage=ADV_TRAIN_CONFIG_SEAL" in line for line in lines[:6])
    assert all("stage=ADV_TARGET_PREDICTION" in line for line in lines[6:])
    old_run_root = f"{PROJECT_ROOT}/runs/{RUN_ID}/F"
    for fold, seal_line, publish_line in zip(
        range(1, 7), lines[:6], lines[6:], strict=True
    ):
        new_output = f"{PROJECT_ROOT}/runs/{V2_RUN_ID}/F{fold}_ADV3B02_CLIC"
        assert new_output in seal_line
        assert new_output in publish_line
        assert old_run_root not in seal_line
        assert old_run_root not in publish_line


def test_v2_formal_requires_completed_smoke_before_any_output(tmp_path: Path) -> None:
    """Break caught: six-fold formal starts after a missing/failed one-shot smoke."""

    project_root = tmp_path / "project"
    result = _run_v2_formal_launcher(
        environment=_project_environment(project_root)
    )
    assert result.returncode == 2
    assert "formal v2 requires a complete F1 target-prediction smoke receipt" in result.stderr
    assert not (project_root / "runs" / V2_RUN_ID).exists()
    assert not (project_root / "logs" / V2_RUN_ID).exists()
