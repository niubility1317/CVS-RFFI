from __future__ import annotations

import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any
import zipfile

import pytest

from scripts import run_adv3b02_signed_source_diagnostic as runner


def _archive_bytes(members: dict[str, bytes], *, symlink: str | None = None) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            kind = stat.S_IFLNK if name == symlink else stat.S_IFREG
            info.external_attr = (kind | 0o444) << 16
            archive.writestr(info, payload)
    return stream.getvalue()


def _signed_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, bytes]]:
    members = {
        "code/cvsrffi/somph_runtime_trust.py": b"# fixed trust\n",
        "code/scripts/diagnose_adv3b02_runtime_numerics.py": b"# consumer\n",
    }
    archive = tmp_path / "adv3b02_numerical_source.zip"
    archive_bytes = _archive_bytes(members)
    archive.write_bytes(archive_bytes)
    rows = [
        {"path": name, "bytes": len(payload), "sha256": runner._sha256(payload)}
        for name, payload in sorted(members.items())
    ]
    receipt = {
        "schema": runner.SOURCE_SCHEMA,
        "issuer": runner.ISSUER,
        "key_id": runner.KEY_ID,
        "public_key_sha256": runner.PUBLIC_KEY_SHA256,
        "source_archive_path": str(archive.resolve()),
        "source_archive_sha256": runner._sha256(archive_bytes),
        "source_git_commit": "a" * 40,
        "source_members": rows,
        "source_manifest_root_sha256": runner._manifest_root(rows),
        "git_policy": {"mode": "signed_manifest_only_no_git"},
        "signature_hex": "00" * 64,
    }
    receipt_path = tmp_path / "source_release_receipt.json"
    receipt_path.write_bytes(runner._canonical_json(receipt))
    return archive, receipt_path, members


def _validate_fixture(archive: Path, receipt: Path) -> dict[str, Any]:
    return runner._validate_release_snapshots(
        archive_path=archive.resolve(),
        archive_bytes=archive.read_bytes(),
        receipt_path=receipt.resolve(),
        receipt_bytes=receipt.read_bytes(),
        verifier=lambda _message, _signature: None,
    )


def test_exact_release_validation_and_extraction(tmp_path: Path) -> None:
    archive, receipt, members = _signed_fixture(tmp_path)
    validated = _validate_fixture(archive, receipt)
    assert validated["member_payloads"] == members
    isolation = tmp_path / "isolation"
    isolation.mkdir()
    runner._extract_exact(isolation, validated["member_payloads"])
    assert (isolation / "code/scripts/diagnose_adv3b02_runtime_numerics.py").is_file()


@pytest.mark.parametrize(
    "attack", ["escape", "extra", "directory", "symlink", "duplicate"]
)
def test_archive_structure_attacks_fail_closed(tmp_path: Path, attack: str) -> None:
    archive, receipt, members = _signed_fixture(tmp_path)
    body = json.loads(receipt.read_text(encoding="utf-8"))
    attacked_members = dict(members)
    if attack == "escape":
        attacked_members["../escape.py"] = b"bad"
        archive_bytes = _archive_bytes(attacked_members)
    elif attack == "extra":
        attacked_members["code/extra.py"] = b"extra"
        archive_bytes = _archive_bytes(attacked_members)
    elif attack == "directory":
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as bundle:
            for name, payload in sorted(members.items()):
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o444) << 16
                bundle.writestr(info, payload)
            directory = zipfile.ZipInfo("code/extra/")
            directory.create_system = 3
            directory.external_attr = (stat.S_IFDIR | 0o555) << 16
            bundle.writestr(directory, b"")
        archive_bytes = stream.getvalue()
    elif attack == "symlink":
        archive_bytes = _archive_bytes(
            attacked_members,
            symlink="code/scripts/diagnose_adv3b02_runtime_numerics.py",
        )
    else:
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as bundle:
            for name, payload in sorted(members.items()):
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o444) << 16
                bundle.writestr(info, payload)
                if "diagnose" in name:
                    bundle.writestr(info, payload)
        archive_bytes = stream.getvalue()
    archive.write_bytes(archive_bytes)
    body["source_archive_sha256"] = runner._sha256(archive_bytes)
    body["signature_hex"] = "00" * 64
    receipt.write_bytes(runner._canonical_json(body))
    with pytest.raises(runner.SignedSourceRunnerError):
        _validate_fixture(archive, receipt)


def test_missing_receipt_boundary_is_exit2_and_zero_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "output"
    code = runner.main(
        [
            "--arm-id",
            "b202",
            "--checkpoint",
            "missing-checkpoint",
            "--runtime",
            "missing-runtime",
            "--lineage-evidence",
            "missing-lineage",
            "--isolation-parent",
            str(tmp_path),
            "--output-root",
            str(output),
            "--device",
            "cuda:0",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == runner.BLOCKED_STATUS
    assert payload["artifact_emitted"] is False
    assert not output.exists()


def test_release_receipt_path_and_signature_are_bound(tmp_path: Path) -> None:
    archive, receipt, _ = _signed_fixture(tmp_path)
    body = json.loads(receipt.read_text(encoding="utf-8"))
    body["source_archive_path"] = "/copied/elsewhere.zip"
    receipt.write_bytes(runner._canonical_json(body))
    with pytest.raises(runner.SignedSourceRunnerError, match="binding drift"):
        _validate_fixture(archive, receipt)

    archive, receipt, _ = _signed_fixture(tmp_path)
    with pytest.raises(runner.SignedSourceRunnerError, match="sentinel"):
        runner._validate_release_snapshots(
            archive_path=archive.resolve(),
            archive_bytes=archive.read_bytes(),
            receipt_path=receipt.resolve(),
            receipt_bytes=receipt.read_bytes(),
            verifier=lambda _message, _signature: (_ for _ in ()).throw(
                runner.SignedSourceRunnerError("sentinel")
            ),
        )


def test_git_parent_is_rejected_and_ceiling_probe_finds_no_repo(tmp_path: Path) -> None:
    git_parent = tmp_path / "with_git"
    (git_parent / ".git").mkdir(parents=True)
    child = git_parent / "child"
    child.mkdir()
    with pytest.raises(runner.SignedSourceRunnerError, match="Git parent"):
        runner._no_git_ancestor(child)

    clean = tmp_path / "clean"
    clean.mkdir()
    env = dict(os.environ)
    env["GIT_CEILING_DIRECTORIES"] = str(clean)
    audit = runner._git_unavailable_audit(clean, env)
    assert audit["repository_discovered"] is False


def test_isolated_runner_publishes_audit_and_removes_source_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_consumer = b"""\
import argparse, json, pathlib
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--artifact-out', required=True)
args,_=p.parse_known_args()
path=pathlib.Path(args.artifact_out)
path.write_text(json.dumps({'status':'NO_AUTHORITY'}), encoding='utf-8')
print(json.dumps({'status':'NO_AUTHORITY'}))
"""
    payloads = {
        "code/cvsrffi/somph_runtime_trust.py": b"# trust\n",
        "code/scripts/diagnose_adv3b02_runtime_numerics.py": fake_consumer,
    }
    archive = tmp_path / "archive.zip"
    receipt = tmp_path / "receipt.json"
    archive.write_bytes(b"archive")
    receipt.write_bytes(b"receipt")
    assets = []
    for name in ("checkpoint", "runtime", "lineage"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        assets.append(path)
    monkeypatch.setattr(
        runner,
        "_validate_release_snapshots",
        lambda **_kwargs: {
            "receipt": {"source_git_commit": "a" * 40},
            "receipt_sha256": runner._sha256(b"receipt"),
            "archive_sha256": runner._sha256(b"archive"),
            "archive_bytes": 7,
            "members": [
                {"path": key, "bytes": len(value), "sha256": runner._sha256(value)}
                for key, value in sorted(payloads.items())
            ],
            "member_payloads": payloads,
        },
    )
    monkeypatch.setattr(
        runner,
        "_assert_bound_asset",
        lambda snapshot, _expected_path, _expected_sha, _name: Path(
            snapshot["path"]
        ).resolve(),
    )
    monkeypatch.setattr(
        runner,
        "_production_platform_guard",
        lambda: {
            "platform": "unit-test",
            "os_name": "unit-test",
            "linux_only_production": False,
            "trust_helper_identity": "unit_test_non_authority",
            "trust_helper_sha256": runner.TRUST_HELPER_GIT_BLOB_SHA256,
        },
    )
    isolation_parent = tmp_path / "isolation_parent"
    isolation_parent.mkdir()
    output = tmp_path / "result"
    result = runner.run_signed_source_diagnostic(
        arm_id="b202",
        checkpoint=assets[0],
        runtime=assets[1],
        lineage_evidence=assets[2],
        source_archive=archive,
        source_release_receipt=receipt,
        isolation_parent=isolation_parent,
        output_root=output,
        device="cuda:0",
        timeout_seconds=30,
    )
    assert result["child_returncode"] == 0
    assert (output / "diagnostic.json").is_file()
    audit = json.loads((output / "runner_audit.json").read_text(encoding="utf-8"))
    assert audit["isolation"]["no_git_ancestor"] is True
    assert audit["isolation"]["git_probe"]["repository_discovered"] is False
    assert audit["asset_snapshot_binding"]["status"] == (
        "ALL_ASSETS_FD_HELD_AND_POSTFLIGHT_MATCHED"
    )
    assert len(audit["asset_snapshot_binding"]["assets"]) == 5
    assert not list(isolation_parent.glob("adv3b02_signed_source_*"))


def test_output_publication_refuses_overwrite(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    with pytest.raises(FileExistsError):
        runner._publish_directory(staging, output)


@pytest.mark.parametrize(
    "asset_flag",
    [
        "--source-archive",
        "--source-release-receipt",
        "--checkpoint",
        "--runtime",
        "--lineage-evidence",
    ],
)
def test_asset_mutation_during_child_is_rejected_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    asset_flag: str,
) -> None:
    fake_consumer = b"""\
import argparse, json, os, pathlib, sys
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--artifact-out', required=True)
args,_=p.parse_known_args()
path=pathlib.Path(args.artifact_out)
path.write_text(json.dumps({'status':'NO_AUTHORITY'}), encoding='utf-8')
flag=os.environ['ADV3B02_TEST_MUTATE_ASSET_FLAG']
target=pathlib.Path(sys.argv[sys.argv.index(flag)+1])
if os.environ.get('ADV3B02_TEST_RESTORE_ASSET') == '1':
    initial=target.stat()
    original=target.read_bytes()
    mutated=bytes(value ^ 0xff for value in original)
    target.write_bytes(mutated)
    assert target.read_bytes() == mutated
    target.write_bytes(original)
    os.utime(target, ns=(initial.st_atime_ns, initial.st_mtime_ns))
else:
    target.write_bytes(b'mutated-after-validation')
print(json.dumps({'status':'NO_AUTHORITY'}))
"""
    payloads = {
        "code/cvsrffi/somph_runtime_trust.py": b"# trust\n",
        "code/scripts/diagnose_adv3b02_runtime_numerics.py": fake_consumer,
    }
    archive = tmp_path / "archive.zip"
    receipt = tmp_path / "receipt.json"
    checkpoint = tmp_path / "checkpoint"
    runtime_path = tmp_path / "runtime"
    lineage = tmp_path / "lineage"
    for path, value in (
        (archive, b"archive"),
        (receipt, b"receipt"),
        (checkpoint, b"checkpoint"),
        (runtime_path, b"runtime"),
        (lineage, b"lineage"),
    ):
        path.write_bytes(value)
    monkeypatch.setenv("ADV3B02_TEST_MUTATE_ASSET_FLAG", asset_flag)
    monkeypatch.setattr(
        runner,
        "_validate_release_snapshots",
        lambda **_kwargs: {
            "receipt": {"source_git_commit": "a" * 40},
            "receipt_sha256": runner._sha256(b"receipt"),
            "archive_sha256": runner._sha256(b"archive"),
            "archive_bytes": 7,
            "members": [
                {"path": key, "bytes": len(value), "sha256": runner._sha256(value)}
                for key, value in sorted(payloads.items())
            ],
            "member_payloads": payloads,
        },
    )
    monkeypatch.setattr(
        runner,
        "_assert_bound_asset",
        lambda snapshot, _expected_path, _expected_sha, _name: Path(
            snapshot["path"]
        ).resolve(),
    )
    monkeypatch.setattr(
        runner,
        "_production_platform_guard",
        lambda: {"platform": "unit-test", "linux_only_production": False},
    )
    isolation_parent = tmp_path / "isolation_parent"
    isolation_parent.mkdir()
    output = tmp_path / "result"
    with pytest.raises(runner.SignedSourceRunnerError, match="changed or was replaced"):
        runner.run_signed_source_diagnostic(
            arm_id="b202",
            checkpoint=checkpoint,
            runtime=runtime_path,
            lineage_evidence=lineage,
            source_archive=archive,
            source_release_receipt=receipt,
            isolation_parent=isolation_parent,
            output_root=output,
            device="cuda:0",
            timeout_seconds=30,
        )
    assert not output.exists()


def test_same_inode_restore_window_is_rejected_before_publication_on_linux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name != "posix" or not Path("/proc/self/fd").is_dir():
        pytest.skip("production ctime restore-window check requires Linux /proc")
    capability_probe = tmp_path / "ctime-capability-probe"
    original = b"original"
    capability_probe.write_bytes(original)
    initial = capability_probe.stat()
    capability_probe.write_bytes(b"mutated")
    capability_probe.write_bytes(original)
    os.utime(
        capability_probe,
        ns=(initial.st_atime_ns, initial.st_mtime_ns),
    )
    restored = capability_probe.stat()
    if (
        restored.st_mtime_ns != initial.st_mtime_ns
        or restored.st_ctime_ns == initial.st_ctime_ns
    ):
        pytest.skip("filesystem cannot expose the ctime restore-window capability")
    monkeypatch.setenv("ADV3B02_TEST_RESTORE_ASSET", "1")
    test_asset_mutation_during_child_is_rejected_before_publication(
        tmp_path,
        monkeypatch,
        "--checkpoint",
    )


def test_production_platform_and_trust_sha_are_linux_git_blob_bound() -> None:
    assert runner.TRUST_HELPER_GIT_BLOB_SHA256 == (
        "4b1dee1d8ffdc793f48c46c21a11b0fdf8b6ef6e3b253807cc1138011dc1f9fc"
    )
    repo = Path(runner.__file__).resolve().parents[2]
    git_blob = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "cat-file",
            "blob",
            "HEAD:code/cvsrffi/somph_runtime_trust.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    assert runner._sha256(git_blob) == runner.TRUST_HELPER_GIT_BLOB_SHA256
    if os.name == "posix" and sys.platform.startswith("linux"):
        assert runner._production_platform_guard()["linux_only_production"] is True
    else:
        with pytest.raises(runner.SignedSourceRunnerError, match="Linux-only"):
            runner._production_platform_guard()


def test_production_cli_exposes_no_signature_fixture_switch() -> None:
    with pytest.raises(SystemExit):
        runner._parse_args(
            [
                "--arm-id",
                "b202",
                "--checkpoint",
                "c",
                "--runtime",
                "r",
                "--lineage-evidence",
                "l",
                "--isolation-parent",
                "i",
                "--output-root",
                "o",
                "--device",
                "cuda:0",
                "--unit-test-signature-fixture",
            ]
        )
