from __future__ import annotations

from argparse import Namespace

from scripts import run_d92_role_oracle_125 as launcher


def test_launcher_freezes_licensed_nonpromotable_contract(monkeypatch, tmp_path):
    captured = {}
    ground = tmp_path / "ground"
    ground.mkdir()
    args = Namespace(
        ground_component_dir=str(ground),
        ground_manifest_sha256="a" * 64,
        cpu_threads=2,
    )

    class Parser:
        @staticmethod
        def parse_args():
            return args

    monkeypatch.setattr(launcher, "parser", lambda: Parser())

    def fake_run(_args):
        captured["candidate"] = launcher.base.CANDIDATE
        captured["scope"] = launcher.base.CLAIM_SCOPE
        captured["authority"] = launcher.base.FORMAL_LAUNCH_AUTHORITY
        captured["contract"] = dict(launcher.base.PHASE2_CONTRACT)
        captured["manifest"] = launcher.base.build_manifest()
        return {"status": "MANIFEST_ONLY"}

    monkeypatch.setattr(launcher.base, "run", fake_run)
    monkeypatch.setattr(launcher.base, "build_manifest", lambda **_kwargs: {"jobs": [{}]})
    original_candidate = launcher.base.CANDIDATE
    original_job_command = launcher.base._job_command
    assert launcher.main() == 0
    assert captured["candidate"] == launcher.CANDIDATE_D92_ROLE_ORACLE
    assert captured["scope"] == launcher.LICENSE_STATUS
    assert captured["authority"] is False
    assert captured["contract"]["phase2_query_role_oracle_access"] is True
    assert captured["contract"]["formal_protocol_valid"] is False
    assert captured["contract"]["promotion_eligible"] is False
    assert captured["contract"]["licensed_protocol_deviation"] == (
        "query_old_new_role_oracle_only"
    )
    assert captured["manifest"]["ground_component_dir"] == str(ground.resolve())
    assert captured["manifest"]["ground_manifest_sha256"] == "a" * 64
    assert captured["manifest"]["jobs"][0]["ground_manifest_sha256"] == "a" * 64
    assert launcher.base.CANDIDATE == original_candidate
    assert launcher.base._job_command is original_job_command
    assert not hasattr(launcher.base, "_ORIGINAL_JOB_COMMAND")


def test_job_command_adds_only_locked_ground_component_arguments(monkeypatch):
    monkeypatch.setattr(
        launcher.base,
        "_ORIGINAL_JOB_COMMAND",
        lambda *_args, **_kwargs: ["python", "row.py", "--candidate", "locked"],
        raising=False,
    )
    monkeypatch.setattr(launcher, "_GROUND_COMPONENT_DIR", "ground")
    monkeypatch.setattr(launcher, "_GROUND_MANIFEST_SHA256", "b" * 64)
    command = launcher._job_command(
        {},
        phase1_checkpoint="checkpoint",
        sealed_runtime="runtime",
        method_lock="method",
        device="cuda:0",
    )
    assert command[-4:] == [
        "--ground-component-dir",
        "ground",
        "--ground-manifest-sha256",
        "b" * 64,
    ]


def test_cpu_threads_are_bounded(monkeypatch):
    configured = launcher._configure_child_cpu_threads(2)
    assert all(configured[name] == "2" for name in launcher._CPU_THREAD_ENV_VARS)
    assert configured["CVSRFFI_CPU_INTEROP_THREADS"] == "1"
