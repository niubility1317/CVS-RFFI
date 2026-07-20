from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import zipfile

import pytest

from scripts import diagnose_adv3b02_runtime_numerics as consumer
from scripts import sign_adv3b02_development_source_release as producer


def _git(repo: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        input=input_bytes,
        check=True,
        capture_output=True,
    ).stdout


def _repository(
    tmp_path: Path,
    *,
    members: list[str] | None = None,
    create_members: bool = True,
) -> tuple[Path, Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Source Release Test")
    _git(repo, "config", "user.email", "source-release-test@example.invalid")
    selected = members or ["code/a.py", "code/pkg/__init__.py", "code/pkg/b.py"]
    if create_members:
        for index, relative in enumerate(sorted(set(selected))):
            if ".." in Path(relative).parts or relative.startswith("/"):
                continue
            path = repo / Path(*relative.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"VALUE_{index} = {index}\n".encode("ascii"))
    lock = repo / "analysis" / "adv3b02_source_members.json"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(
        json.dumps(
            {
                "schema": producer.SOURCE_MEMBER_LOCK_SCHEMA,
                "members": selected,
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _git(repo, "add", "--all")
    _git(repo, "commit", "-m", "source release fixture")
    commit = _git(repo, "rev-parse", "HEAD").decode("ascii").strip()
    return repo, lock, commit


def _test_signature(message: bytes) -> bytes:
    return hashlib.sha512(b"adv3b02-source-release-test\0" + message).digest()


def _test_verify(message: bytes, signature: bytes) -> None:
    if signature != _test_signature(message):
        raise ValueError("test signature mismatch")


def _release_paths(tmp_path: Path, name: str = "release") -> tuple[Path, Path]:
    root = tmp_path / name
    return root / producer.SOURCE_ARCHIVE_NAME, root / producer.SOURCE_RECEIPT_NAME


def _test_release(
    tmp_path: Path,
    *,
    repo: Path | None = None,
    lock: Path | None = None,
    commit: str | None = None,
    name: str = "release",
) -> tuple[dict[str, Any], Path, Path, Path, Path, str]:
    if repo is None or lock is None or commit is None:
        repo, lock, commit = _repository(tmp_path)
    archive, receipt = _release_paths(tmp_path, name)
    summary = producer._sign_source_archive_release_impl(
        repo_root=repo,
        source_commit=commit,
        member_lock_path=lock,
        archive_output=archive,
        signed_archive_execution_path=str(archive.resolve()),
        receipt_output=receipt,
        issuer=producer.SOURCE_RELEASE_ISSUER,
        key_id=producer.SOURCE_RELEASE_KEY_ID,
        public_key_sha256=producer.SOURCE_RELEASE_PUBLIC_KEY_SHA256,
        sign_message=_test_signature,
        verify_signature=_test_verify,
        execution_path_normalizer=producer._native_test_execution_path,
    )
    return summary, archive, receipt, repo, lock, commit


def _make_writable(path: Path) -> None:
    if path.exists():
        os.chmod(path, 0o600)


def test_private_test_producer_emits_consumer_exact_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary, archive, receipt, _repo, _lock, commit = _test_release(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "signature_hex"}
    assert set(body) == {
        "schema",
        "issuer",
        "key_id",
        "public_key_sha256",
        "source_archive_path",
        "source_archive_sha256",
        "source_git_commit",
        "source_members",
        "source_manifest_root_sha256",
        "git_policy",
    }
    assert payload["schema"] == producer.SOURCE_RELEASE_SCHEMA
    assert payload["source_git_commit"] == commit
    assert payload["git_policy"] == {"mode": "signed_manifest_only_no_git"}
    assert payload["signature_hex"] == _test_signature(
        producer._canonical_json_bytes(body)
    ).hex()
    assert summary["signature_verified"] is True
    assert summary["formal_launch_authority"] is False
    assert summary["formal_metric_claim_allowed"] is False
    assert summary["target_access"] is False
    monkeypatch.setattr(consumer, "_verify_source_release_signature", _test_verify)
    validated = consumer._validate_source_release(
        source_archive_path=archive,
        source_release_receipt_path=receipt,
    )
    assert validated["signature_verified"] is True
    assert validated["acceptance"] == "EXTERNAL_SIGNED_SOURCE_RECEIPT_VERIFIED"


def test_archive_is_deterministic_exact_member_zip(tmp_path: Path) -> None:
    repo, lock, commit = _repository(tmp_path)
    first, first_archive, _receipt, *_ = _test_release(
        tmp_path, repo=repo, lock=lock, commit=commit, name="first"
    )
    second, second_archive, _receipt2, *_ = _test_release(
        tmp_path, repo=repo, lock=lock, commit=commit, name="second"
    )
    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert first["source_archive_sha256"] == second["source_archive_sha256"]
    with zipfile.ZipFile(first_archive, "r") as archive:
        infos = archive.infolist()
        assert [item.filename for item in infos] == [
            "code/a.py",
            "code/pkg/__init__.py",
            "code/pkg/b.py",
        ]
        assert all(not item.is_dir() for item in infos)
        assert all(item.compress_type == zipfile.ZIP_STORED for item in infos)
        assert all(item.date_time == (1980, 1, 1, 0, 0, 0) for item in infos)
        assert all(((item.external_attr >> 16) & 0o170000) == 0o100000 for item in infos)


@pytest.mark.parametrize("dirty_kind", ["tracked", "staged", "untracked"])
def test_dirty_git_state_fails_before_output(tmp_path: Path, dirty_kind: str) -> None:
    repo, lock, commit = _repository(tmp_path)
    if dirty_kind == "untracked":
        (repo / "untracked.py").write_text("x = 1\n", encoding="utf-8")
    else:
        member = repo / "code" / "a.py"
        member.write_text("VALUE = 999\n", encoding="utf-8")
        if dirty_kind == "staged":
            _git(repo, "add", "code/a.py")
    archive, receipt = _release_paths(tmp_path)
    with pytest.raises(
        producer.ADV3B02SourceReleaseSigningError,
        match="clean Git worktree",
    ):
        producer._sign_source_archive_release_impl(
            repo_root=repo,
            source_commit=commit,
            member_lock_path=lock,
            archive_output=archive,
            signed_archive_execution_path=str(archive.resolve()),
            receipt_output=receipt,
            issuer=producer.SOURCE_RELEASE_ISSUER,
            key_id=producer.SOURCE_RELEASE_KEY_ID,
            public_key_sha256=producer.SOURCE_RELEASE_PUBLIC_KEY_SHA256,
            sign_message=_test_signature,
            verify_signature=_test_verify,
            execution_path_normalizer=producer._native_test_execution_path,
        )
    assert not archive.parent.exists()


@pytest.mark.parametrize(
    "members,error",
    [
        (["code/a.py", "code/a.py"], "unique and sorted"),
        (["code/b.py", "code/a.py"], "unique and sorted"),
        (["../escape.py"], "normalized relative Python path"),
        (["code/missing.py"], "not one tracked Git object"),
        (["code/not_python.txt"], "normalized relative Python path"),
    ],
)
def test_member_lock_attacks_fail_closed(
    tmp_path: Path, members: list[str], error: str
) -> None:
    repo, lock, commit = _repository(
        tmp_path,
        members=members,
        create_members="missing" not in " ".join(members),
    )
    archive, receipt = _release_paths(tmp_path)
    with pytest.raises(producer.ADV3B02SourceReleaseSigningError, match=error):
        producer._sign_source_archive_release_impl(
            repo_root=repo,
            source_commit=commit,
            member_lock_path=lock,
            archive_output=archive,
            signed_archive_execution_path=str(archive.resolve()),
            receipt_output=receipt,
            issuer=producer.SOURCE_RELEASE_ISSUER,
            key_id=producer.SOURCE_RELEASE_KEY_ID,
            public_key_sha256=producer.SOURCE_RELEASE_PUBLIC_KEY_SHA256,
            sign_message=_test_signature,
            verify_signature=_test_verify,
            execution_path_normalizer=producer._native_test_execution_path,
        )
    assert not archive.parent.exists()


def test_wrong_signature_and_publish_failure_leave_no_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, lock, commit = _repository(tmp_path)
    archive, receipt = _release_paths(tmp_path)
    with pytest.raises(
        producer.ADV3B02SourceReleaseSigningError,
        match="not valid for the pinned identity",
    ):
        producer._sign_source_archive_release_impl(
            repo_root=repo,
            source_commit=commit,
            member_lock_path=lock,
            archive_output=archive,
            signed_archive_execution_path=str(archive.resolve()),
            receipt_output=receipt,
            issuer=producer.SOURCE_RELEASE_ISSUER,
            key_id=producer.SOURCE_RELEASE_KEY_ID,
            public_key_sha256=producer.SOURCE_RELEASE_PUBLIC_KEY_SHA256,
            sign_message=lambda _message: b"0" * 64,
            verify_signature=_test_verify,
            execution_path_normalizer=producer._native_test_execution_path,
        )
    assert not archive.parent.exists()

    calls = 0
    actual_write = producer._write_new_readonly

    def fail_second_write(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second-file failure")
        actual_write(path, payload)

    monkeypatch.setattr(producer, "_write_new_readonly", fail_second_write)
    with pytest.raises(OSError, match="second-file failure"):
        producer._sign_source_archive_release_impl(
            repo_root=repo,
            source_commit=commit,
            member_lock_path=lock,
            archive_output=archive,
            signed_archive_execution_path=str(archive.resolve()),
            receipt_output=receipt,
            issuer=producer.SOURCE_RELEASE_ISSUER,
            key_id=producer.SOURCE_RELEASE_KEY_ID,
            public_key_sha256=producer.SOURCE_RELEASE_PUBLIC_KEY_SHA256,
            sign_message=_test_signature,
            verify_signature=_test_verify,
            execution_path_normalizer=producer._native_test_execution_path,
        )
    assert not archive.parent.exists()
    assert not list(tmp_path.glob(".release.staging-*"))


def test_existing_output_root_is_never_overwritten(tmp_path: Path) -> None:
    summary, archive, receipt, repo, lock, commit = _test_release(tmp_path)
    archive_bytes = archive.read_bytes()
    receipt_bytes = receipt.read_bytes()
    assert summary["signature_verified"] is True
    with pytest.raises(FileExistsError, match="overwrite"):
        producer._sign_source_archive_release_impl(
            repo_root=repo,
            source_commit=commit,
            member_lock_path=lock,
            archive_output=archive,
            signed_archive_execution_path=str(archive.resolve()),
            receipt_output=receipt,
            issuer=producer.SOURCE_RELEASE_ISSUER,
            key_id=producer.SOURCE_RELEASE_KEY_ID,
            public_key_sha256=producer.SOURCE_RELEASE_PUBLIC_KEY_SHA256,
            sign_message=_test_signature,
            verify_signature=_test_verify,
            execution_path_normalizer=producer._native_test_execution_path,
        )
    assert archive.read_bytes() == archive_bytes
    assert receipt.read_bytes() == receipt_bytes


def test_publish_race_preserves_competing_root_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, lock, commit = _repository(tmp_path)
    archive, receipt = _release_paths(tmp_path)

    def lose_no_replace_race(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "competitor.txt").write_text("owned elsewhere", encoding="utf-8")
        raise FileExistsError("injected no-replace race")

    monkeypatch.setattr(producer, "_rename_directory_noreplace", lose_no_replace_race)
    with pytest.raises(FileExistsError, match="no-replace race"):
        producer._sign_source_archive_release_impl(
            repo_root=repo,
            source_commit=commit,
            member_lock_path=lock,
            archive_output=archive,
            signed_archive_execution_path=str(archive.resolve()),
            receipt_output=receipt,
            issuer=producer.SOURCE_RELEASE_ISSUER,
            key_id=producer.SOURCE_RELEASE_KEY_ID,
            public_key_sha256=producer.SOURCE_RELEASE_PUBLIC_KEY_SHA256,
            sign_message=_test_signature,
            verify_signature=_test_verify,
            execution_path_normalizer=producer._native_test_execution_path,
        )
    assert (archive.parent / "competitor.txt").read_text(encoding="utf-8") == "owned elsewhere"
    assert not archive.exists()
    assert not receipt.exists()
    assert not list(tmp_path.glob(".release.staging-*"))


def test_consumer_rejects_archive_relocation_and_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _summary, archive, receipt, *_ = _test_release(tmp_path)
    monkeypatch.setattr(consumer, "_verify_source_release_signature", _test_verify)
    relocated = tmp_path / "relocated.zip"
    relocated.write_bytes(archive.read_bytes())
    with pytest.raises(
        consumer.ADV3B02NumericalDiagnosticError,
        match="archive/issuer binding drift",
    ):
        consumer._validate_source_release(
            source_archive_path=relocated,
            source_release_receipt_path=receipt,
        )
    _make_writable(archive)
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(
        consumer.ADV3B02NumericalDiagnosticError,
        match="archive/issuer binding drift",
    ):
        consumer._validate_source_release(
            source_archive_path=archive,
            source_release_receipt_path=receipt,
        )


def test_no_git_execution_binding_accepts_exact_members_only(tmp_path: Path) -> None:
    _summary, _archive, receipt, *_ = _test_release(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    signed = payload["source_members"]
    contract = {
        "source_git_commit": payload["source_git_commit"],
        "source_members": signed,
        "source_manifest_root_sha256": payload["source_manifest_root_sha256"],
        "git_policy": payload["git_policy"],
        "contract_path": "unit-contract",
        "contract_sha256": "a" * 64,
    }
    dependencies = {"loaded_project_modules": signed}
    git = {"git_available": False}
    result = consumer._validate_execution_source_binding(
        dependencies=dependencies,
        git=git,
        source_git_commit=payload["source_git_commit"],
        contract=contract,
    )
    assert result["status"] == "SIGNED_MEMBER_MANIFEST_EXECUTION_CLOSED"
    attacked = {"loaded_project_modules": signed[:-1]}
    with pytest.raises(
        consumer.ADV3B02NumericalDiagnosticError,
        match="do not exactly match",
    ):
        consumer._validate_execution_source_binding(
            dependencies=attacked,
            git=git,
            source_git_commit=payload["source_git_commit"],
            contract=contract,
        )


def test_production_surface_has_no_identity_or_test_injection() -> None:
    parameters = set(inspect.signature(producer.sign_source_archive_release).parameters)
    assert parameters == {
        "repo_root",
        "source_commit",
        "member_lock_path",
        "archive_output",
        "signed_archive_execution_path",
        "receipt_output",
        "private_key_path",
        "openssl_bin",
    }
    parser = producer.parse_args(
        [
            "--repo-root",
            "repo",
            "--source-commit",
            "a" * 40,
            "--member-lock",
            "lock.json",
            "--archive-output",
            "release/adv3b02_numerical_source.zip",
            "--signed-archive-execution-path",
            "/release/adv3b02_numerical_source.zip",
            "--receipt-output",
            "release/source_release_receipt.json",
            "--private-key",
            "authority.pem",
        ]
    )
    assert not any("test" in key or "issuer" in key or "verifier" in key for key in vars(parser))


def test_summary_and_receipt_do_not_disclose_key_material(tmp_path: Path) -> None:
    summary, archive, receipt, *_ = _test_release(tmp_path)
    serialized = json.dumps(summary, sort_keys=True) + receipt.read_text(encoding="utf-8")
    assert "private_key" not in serialized
    assert "authority.pem" not in serialized
    with zipfile.ZipFile(archive, "r") as bundle:
        assert all("key" not in name.lower() for name in bundle.namelist())


def _pinned_openssl() -> Path:
    path = Path(producer.lock_signer.PINNED_OPENSSL_BINARY_PATH)
    if not path.is_file():
        pytest.skip("pinned OpenSSL is unavailable")
    return path.resolve()


def test_production_wrong_key_fails_without_output(tmp_path: Path) -> None:
    openssl = _pinned_openssl()
    repo, lock, commit = _repository(tmp_path)
    key = tmp_path / "wrong-authority.pem"
    subprocess.run(
        [str(openssl), "genpkey", "-algorithm", "ED25519", "-out", str(key)],
        check=True,
        capture_output=True,
    )
    archive, receipt = _release_paths(tmp_path)
    with pytest.raises(
        producer.ADV3B02SourceReleaseSigningError,
        match="not valid for the pinned identity",
    ):
        producer.sign_source_archive_release(
            repo_root=repo,
            source_commit=commit,
            member_lock_path=lock,
            archive_output=archive,
            signed_archive_execution_path=(
                "/home/szu2070436088/releases/"
                + producer.SOURCE_ARCHIVE_NAME
            ),
            receipt_output=receipt,
            private_key_path=key,
            openssl_bin=openssl,
        )
    assert not archive.parent.exists()


def test_production_execution_path_is_posix_absolute_and_filename_locked() -> None:
    valid = f"/home/source-release/{producer.SOURCE_ARCHIVE_NAME}"
    assert producer._production_execution_path(valid, producer.SOURCE_ARCHIVE_NAME) == valid
    for invalid in (
        producer.SOURCE_ARCHIVE_NAME,
        f"/home/../release/{producer.SOURCE_ARCHIVE_NAME}",
        "/home/release/other.zip",
        r"C:\release\adv3b02_numerical_source.zip",
    ):
        with pytest.raises(producer.ADV3B02SourceReleaseSigningError):
            producer._production_execution_path(invalid, producer.SOURCE_ARCHIVE_NAME)
