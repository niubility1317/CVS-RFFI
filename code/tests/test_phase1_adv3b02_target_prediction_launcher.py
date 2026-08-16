from __future__ import annotations

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
RUN_ID = "phase1_adv3b02_target_prediction_20260816_v1"
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
    return subprocess.run(
        [str(GIT_BASH), _git_bash_path(LAUNCHER), *arguments],
        cwd=CODE_ROOT.parent,
        env=environment,
        capture_output=True,
        text=True,
    )


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
