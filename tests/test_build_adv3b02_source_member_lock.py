from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from scripts import build_adv3b02_source_member_lock as builder


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True
    ).stdout


def _repo(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    files = {
        "code/app.py": "import pkg.a\nfrom pkg import b\n",
        "code/pkg/__init__.py": "",
        "code/pkg/a.py": "from pkg import c\n",
        "code/pkg/b.py": "VALUE = 2\n",
        "code/pkg/c.py": "VALUE = 3\n",
        "code/pkg/runtime_only.py": "VALUE = 4\n",
        "code/scripts/diagnose_adv3b02_runtime_numerics.py": (
            "class _Torch:\n"
            "    @staticmethod\n"
            "    def device(value): return value\n"
            "torch = _Torch()\n"
            "def _load_worker_dependencies():\n"
            "    import pkg.a\n"
            "    from pkg import b\n"
            "def _build_eager(checkpoint, device):\n"
            "    import pkg.runtime_only\n"
            "    return object(), {'loaded': True}, {}\n"
        ),
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(root, "init")
    _git(root, "config", "user.name", "Member Lock Test")
    _git(root, "config", "user.email", "member-lock@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "initial")
    return root, _git(root, "rev-parse", "HEAD").decode().strip()


def _probe(
    _root: Path, entries: tuple[str, ...] | list[str], _checkpoint: bytes
) -> list[str]:
    return sorted([*entries, "code/pkg/runtime_only.py"])


def _checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "checkpoint.pth"
    path.write_bytes(b"strict-adv3b02-test-checkpoint")
    monkeypatch.setattr(builder, "BASE_CHECKPOINT_SHA256", builder._sha256(path.read_bytes()))
    return path


def _review_from_proposal(
    proposal: dict[str, object], reviewed: str, lock_sha: str
) -> dict[str, object]:
    return {
        "schema": builder.REVIEW_SCHEMA,
        "decision": builder.REVIEW_DECISION,
        "scope": builder.REVIEW_SCOPE,
        "reviewer_id": builder.REVIEWER_IDENTITY,
        "reviewed_source_commit": reviewed,
        "member_lock_sha256": lock_sha,
        "checkpoint_sha256": proposal["checkpoint_sha256"],
        "static_member_rows": proposal["static_member_rows"],
        "runtime_member_rows": proposal["runtime_member_rows"],
        "static_member_root_sha256": proposal["static_member_root_sha256"],
        "runtime_member_root_sha256": proposal["runtime_member_root_sha256"],
        "closure_root_sha256": proposal["closure_root_sha256"],
    }


def test_static_and_runtime_probe_union_uses_tracked_blobs(tmp_path: Path) -> None:
    root, head = _repo(tmp_path)
    result = builder._analyze(
        root,
        head,
        checkpoint_bytes=b"checkpoint",
        entries=("code/app.py",),
        probe=_probe,
    )
    assert result["static_members"] == [
        "code/app.py",
        "code/pkg/__init__.py",
        "code/pkg/a.py",
        "code/pkg/b.py",
        "code/pkg/c.py",
    ]
    assert result["members"] == ["code/app.py", "code/pkg/runtime_only.py"]
    assert result["static_only_members"] == [
        "code/pkg/__init__.py",
        "code/pkg/a.py",
        "code/pkg/b.py",
        "code/pkg/c.py",
    ]
    assert json.loads(result["lock_bytes"])["schema"] == builder.LOCK_SCHEMA


def test_runtime_probe_cannot_add_untracked_fixture(tmp_path: Path) -> None:
    root, head = _repo(tmp_path)
    (root / "code/pkg/untracked.py").write_text("X=1\n", encoding="utf-8")
    with pytest.raises(builder.MemberLockBuildError, match="tracked blob"):
        builder._analyze(
            root,
            head,
            checkpoint_bytes=b"checkpoint",
            entries=("code/app.py",),
            probe=lambda _root, entries, _checkpoint: [
                *entries,
                "code/pkg/untracked.py",
            ],
        )


def test_real_child_import_probe_observes_repository_modules(tmp_path: Path) -> None:
    root, _ = _repo(tmp_path)
    observed = builder._runtime_import_probe(
        root,
        ("code/scripts/diagnose_adv3b02_runtime_numerics.py",),
        b"checkpoint",
    )
    assert "code/scripts/diagnose_adv3b02_runtime_numerics.py" in observed
    assert "code/pkg/a.py" in observed
    assert "code/pkg/b.py" in observed
    assert "code/pkg/c.py" in observed
    assert "code/pkg/runtime_only.py" in observed


def test_propose_then_human_review_commit_then_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _repo(tmp_path)
    checkpoint = _checkpoint(tmp_path, monkeypatch)
    monkeypatch.setattr(builder, "ENTRY_PATHS", ("code/app.py",))
    monkeypatch.setattr(builder, "_runtime_import_probe", _probe)
    outside = tmp_path / "outside"
    outside.mkdir()
    candidate = outside / "candidate.json"
    proposal = outside / "proposal.json"
    summary = builder.propose_member_lock(
        repo_root=root,
        checkpoint=checkpoint,
        candidate_out=candidate,
        evidence_out=proposal,
    )
    assert summary["formal_authority"] is False
    assert summary["status"].startswith("PROPOSED_REQUIRES_HUMAN")
    assert _git(root, "status", "--porcelain") == b""

    lock_path = root / "analysis/member_lock.json"
    lock_path.parent.mkdir()
    lock_path.write_bytes(candidate.read_bytes())
    _git(root, "add", "analysis/member_lock.json")
    _git(root, "commit", "-m", "track reviewed source member lock")
    reviewed_source_commit = _git(root, "rev-parse", "HEAD").decode().strip()
    proposal_payload = json.loads(proposal.read_text(encoding="utf-8"))
    review = _review_from_proposal(
        proposal_payload, reviewed_source_commit, summary["candidate_sha256"]
    )
    review_path = root / "analysis/member_lock_review.json"
    review_path.write_bytes(builder._canonical_json(review))
    _git(root, "add", "analysis/member_lock_review.json")
    _git(root, "commit", "-m", "record independent source member review")
    verified_path = outside / "verified.json"
    self_envelope = outside / "caller_self_envelope.json"
    self_envelope.write_text("{}", encoding="utf-8")
    verified = builder.verify_tracked_member_lock(
        repo_root=root,
        checkpoint=checkpoint,
        member_lock=lock_path,
        human_review=review_path,
        review_authority_envelope=self_envelope,
        evidence_out=verified_path,
    )
    assert verified["status"] == builder.BLOCKED_REVIEW_STATUS
    evidence = json.loads(verified_path.read_text(encoding="utf-8"))
    assert evidence["human_review_sha256"]
    assert evidence["independent_review_authority"]["verified"] is False
    assert evidence["independent_review_authority"]["status"] == (
        builder.BLOCKED_REVIEW_STATUS
    )
    assert evidence["offline_signature_emitted"] is False


def test_dirty_tree_and_review_drift_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _repo(tmp_path)
    checkpoint = _checkpoint(tmp_path, monkeypatch)
    monkeypatch.setattr(builder, "ENTRY_PATHS", ("code/app.py",))
    monkeypatch.setattr(builder, "_runtime_import_probe", _probe)
    (root / "untracked.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(builder.MemberLockBuildError, match="clean tree"):
        builder.propose_member_lock(
            repo_root=root,
            checkpoint=checkpoint,
            candidate_out=tmp_path / "candidate.json",
            evidence_out=tmp_path / "evidence.json",
        )


def test_source_change_after_reviewed_commit_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _repo(tmp_path)
    checkpoint = _checkpoint(tmp_path, monkeypatch)
    monkeypatch.setattr(builder, "ENTRY_PATHS", ("code/app.py",))
    monkeypatch.setattr(builder, "_runtime_import_probe", _probe)
    outside = tmp_path / "outside"
    outside.mkdir()
    candidate = outside / "candidate.json"
    proposal = outside / "proposal.json"
    proposed = builder.propose_member_lock(
        repo_root=root,
        checkpoint=checkpoint,
        candidate_out=candidate,
        evidence_out=proposal,
    )
    lock_path = root / "analysis/member_lock.json"
    lock_path.parent.mkdir()
    lock_path.write_bytes(candidate.read_bytes())
    _git(root, "add", "analysis/member_lock.json")
    _git(root, "commit", "-m", "track lock")
    reviewed = _git(root, "rev-parse", "HEAD").decode().strip()
    (root / "code/pkg/a.py").write_text(
        "from pkg import c\nCHANGED = True\n", encoding="utf-8"
    )
    proposal_payload = json.loads(proposal.read_text(encoding="utf-8"))
    review = _review_from_proposal(
        proposal_payload, reviewed, proposed["candidate_sha256"]
    )
    review_path = root / "analysis/member_lock_review.json"
    review_path.write_bytes(builder._canonical_json(review))
    _git(root, "add", "code/pkg/a.py", "analysis/member_lock_review.json")
    _git(root, "commit", "-m", "change source after review")
    with pytest.raises(
        builder.MemberLockBuildError,
        match="review rows/closure roots|changed after human review",
    ):
        builder.verify_tracked_member_lock(
            repo_root=root,
            checkpoint=checkpoint,
            member_lock=lock_path,
            human_review=review_path,
            review_authority_envelope=None,
            evidence_out=outside / "verified.json",
        )


def test_static_only_deletion_after_review_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _repo(tmp_path)
    checkpoint = _checkpoint(tmp_path, monkeypatch)
    monkeypatch.setattr(builder, "ENTRY_PATHS", ("code/app.py",))
    monkeypatch.setattr(builder, "_runtime_import_probe", _probe)
    outside = tmp_path / "outside"
    outside.mkdir()
    candidate = outside / "candidate.json"
    proposal_path = outside / "proposal.json"
    proposed = builder.propose_member_lock(
        repo_root=root,
        checkpoint=checkpoint,
        candidate_out=candidate,
        evidence_out=proposal_path,
    )
    lock_path = root / "analysis/member_lock.json"
    lock_path.parent.mkdir()
    lock_path.write_bytes(candidate.read_bytes())
    _git(root, "add", "analysis/member_lock.json")
    _git(root, "commit", "-m", "track lock")
    reviewed = _git(root, "rev-parse", "HEAD").decode().strip()
    review = _review_from_proposal(
        json.loads(proposal_path.read_text(encoding="utf-8")),
        reviewed,
        proposed["candidate_sha256"],
    )
    review_path = root / "analysis/member_lock_review.json"
    review_path.write_bytes(builder._canonical_json(review))
    (root / "code/pkg/c.py").unlink()
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "delete static-only source after review")
    with pytest.raises(builder.MemberLockBuildError, match="review rows/closure roots"):
        builder.verify_tracked_member_lock(
            repo_root=root,
            checkpoint=checkpoint,
            member_lock=lock_path,
            human_review=review_path,
            review_authority_envelope=None,
            evidence_out=outside / "verified.json",
        )


def test_arbitrary_reviewer_and_self_envelope_cannot_authorize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _repo(tmp_path)
    checkpoint = _checkpoint(tmp_path, monkeypatch)
    monkeypatch.setattr(builder, "ENTRY_PATHS", ("code/app.py",))
    monkeypatch.setattr(builder, "_runtime_import_probe", _probe)
    outside = tmp_path / "outside"
    outside.mkdir()
    candidate = outside / "candidate.json"
    proposal_path = outside / "proposal.json"
    proposed = builder.propose_member_lock(
        repo_root=root,
        checkpoint=checkpoint,
        candidate_out=candidate,
        evidence_out=proposal_path,
    )
    lock_path = root / "analysis/member_lock.json"
    lock_path.parent.mkdir()
    lock_path.write_bytes(candidate.read_bytes())
    _git(root, "add", "analysis/member_lock.json")
    _git(root, "commit", "-m", "track lock")
    reviewed = _git(root, "rev-parse", "HEAD").decode().strip()
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    review = _review_from_proposal(proposal, reviewed, proposed["candidate_sha256"])
    review["reviewer_id"] = "caller-self-certified-reviewer"
    review_path = root / "analysis/member_lock_review.json"
    review_path.write_bytes(builder._canonical_json(review))
    _git(root, "add", "analysis/member_lock_review.json")
    _git(root, "commit", "-m", "self certify review")
    fake_envelope = outside / "fake_authority.json"
    fake_envelope.write_text("{}", encoding="utf-8")
    with pytest.raises(builder.MemberLockBuildError, match="does not approve"):
        builder.verify_tracked_member_lock(
            repo_root=root,
            checkpoint=checkpoint,
            member_lock=lock_path,
            human_review=review_path,
            review_authority_envelope=fake_envelope,
            evidence_out=outside / "verified.json",
        )


def test_production_cli_has_no_fixture_or_signing_switch() -> None:
    with pytest.raises(SystemExit):
        builder._parse_args(
            [
                "propose",
                "--repo-root",
                "repo",
                "--checkpoint",
                "checkpoint",
                "--candidate-out",
                "candidate",
                "--evidence-out",
                "evidence",
                "--unit-test-fixture",
            ]
        )
