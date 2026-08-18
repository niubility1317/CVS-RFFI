from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.project_governance.collect_git import (
    CommandResult,
    GitOwnershipMapper,
    map_git_ownership,
    subprocess_git_runner,
)
from tools.project_governance.models import (
    AccessStatus,
    AssetKind,
    AssetRecord,
    HashStatus,
    GitOwnership,
    Location,
)
from tools.project_governance.paths import stable_asset_id


def _git(repo: Path, *args: str, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def _asset(root: Path, relative_path: str, *, location: Location = Location.LOCAL, root_id: str = "FIXTURE") -> AssetRecord:
    return AssetRecord(
        asset_id=stable_asset_id(location, root_id, relative_path),
        scan_id="GIT_TEST_SCAN",
        location=location,
        root_id=root_id,
        relative_path=relative_path,
        display_name=relative_path.rsplit("/", 1)[-1],
        escaped_name=relative_path.rsplit("/", 1)[-1],
        asset_kind=AssetKind.FILE,
        size_bytes=None,
        mtime_utc=None,
        access_status=AccessStatus.OK,
        hash_status=HashStatus.METADATA_ONLY,
        sha256=None,
    )


@pytest.fixture()
def git_fixture(tmp_path: Path) -> tuple[Path, tuple[AssetRecord, ...]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "Fixture User")
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    (repo / ".gitignore").write_text("cache.tmp\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt", ".gitignore")
    _git(repo, "commit", "-m", "fixture")
    (repo / "draft.txt").write_text("draft\n", encoding="utf-8")
    (repo / "cache.tmp").write_text("cache\n", encoding="utf-8")
    return repo, tuple(
        _asset(repo, name)
        for name in ("tracked.txt", "draft.txt", "cache.tmp")
    )


def test_maps_exact_git_ownership_without_modifying_fixture_worktree(git_fixture):
    repo, assets = git_fixture
    before = _git(repo, "status", "--porcelain=v2", "-z").stdout

    ownership = map_git_ownership(
        assets,
        repository_seeds=(repo,),
        root_paths={"FIXTURE": repo},
    )

    after = _git(repo, "status", "--porcelain=v2", "-z").stdout
    assert before == after
    assert ownership["tracked.txt"].ownership is GitOwnership.TRACKED_GIT
    assert ownership["draft.txt"].ownership is GitOwnership.UNTRACKED_IN_GIT_WORKTREE
    assert ownership["cache.tmp"].ownership is GitOwnership.IGNORED_REGENERABLE


def test_repository_metadata_contains_head_branch_status_and_worktree_evidence(git_fixture):
    repo, assets = git_fixture
    mapper = GitOwnershipMapper(
        repository_seeds=(repo,),
        root_paths={"FIXTURE": repo},
    )

    ownership = mapper.map(assets)
    tracked = ownership["tracked.txt"]
    head = _git(repo, "rev-parse", "HEAD").stdout.decode("ascii").strip()

    assert tracked.repository_root == str(repo.resolve())
    assert tracked.common_git_dir == str((repo / ".git").resolve())
    assert tracked.branch == "main"
    assert tracked.head_commit == head
    assert tracked.status_summary is not None
    assert tracked.linked_worktrees is not None
    assert str(repo.resolve()) in tracked.linked_worktrees
    assert tracked.error is None


def test_outside_and_remote_assets_are_not_promoted_to_local_git_states(git_fixture, tmp_path):
    repo, indexed = git_fixture
    outside = _asset(tmp_path, "outside.txt", root_id="OUTSIDE")
    remote = _asset(
        tmp_path,
        "remote.txt",
        location=Location.N607,
        root_id="N607_CVS_SINCNET",
    )

    ownership = map_git_ownership(
        (*indexed[:1], outside, remote),
        repository_seeds=(repo,),
        root_paths={"FIXTURE": repo, "OUTSIDE": tmp_path, "N607_CVS_SINCNET": tmp_path},
    )

    assert ownership["outside.txt"].ownership is GitOwnership.NON_GIT_EVIDENCE
    assert ownership["remote.txt"].ownership is GitOwnership.REMOTE_NON_GIT


def test_injected_runner_reports_git_failure_without_downgrading_to_untracked(git_fixture):
    repo, assets = git_fixture
    calls: list[tuple[Path, tuple[str, ...], bytes]] = []

    def failing_ls_files(cwd, args, *, input=b""):
        calls.append((Path(cwd), tuple(args), input))
        if tuple(args[:2]) == ("ls-files", "--stage"):
            return CommandResult(2, b"", b"fixture ls-files failure")
        return subprocess_git_runner(cwd, args, input=input)

    ownership = GitOwnershipMapper(
        repository_seeds=(repo,),
        root_paths={"FIXTURE": repo},
        runner=failing_ls_files,
    ).map(assets)

    assert all(record.ownership is GitOwnership.GIT_STATE_ERROR for record in ownership.values())
    assert all(record.error and "ls-files" in record.error for record in ownership.values())


def test_injected_runner_receives_only_indexed_path_batches_and_no_destructive_git_command(git_fixture):
    repo, assets = git_fixture
    calls: list[tuple[Path, tuple[str, ...], bytes]] = []
    banned = {"add", "commit", "clean", "reset", "checkout", "restore", "gc"}

    def recording_runner(cwd, args, *, input=b""):
        call = (Path(cwd), tuple(args), input)
        calls.append(call)
        assert not (args and args[0] in banned)
        assert not (args[:2] == ("ls-files", "--others"))
        return subprocess_git_runner(cwd, args, input=input)

    ownership = GitOwnershipMapper(
        repository_seeds=(repo,),
        root_paths={"FIXTURE": repo},
        runner=recording_runner,
        batch_size=2,
    ).map(assets)

    assert set(ownership) == {"tracked.txt", "draft.txt", "cache.tmp"}
    for _, args, _ in calls:
        assert not (args and args[0] in banned)
        assert not (args[:2] == ("ls-files", "--others"))
        if args[:2] == ("ls-files", "--stage"):
            separator = args.index("--")
            queried = args[separator + 1 :]
            assert queried
            assert set(queried) <= {asset.relative_path for asset in assets}
        if args[:2] == ("check-ignore", "-z"):
            assert args[2:] == ("--stdin",)


def test_mapper_can_classify_assets_from_an_indexed_directory_without_recursive_git_search(git_fixture):
    repo, assets = git_fixture
    nested = repo / "nested"
    nested.mkdir()
    (nested / "new.txt").write_text("new\n", encoding="utf-8")
    nested_asset = _asset(repo, "nested/new.txt")
    calls: list[tuple[str, ...]] = []

    def recording_runner(cwd, args, *, input=b""):
        calls.append(tuple(args))
        return subprocess_git_runner(cwd, args, input=input)

    ownership = map_git_ownership(
        (*assets, nested_asset),
        repository_seeds=(nested,),
        root_paths={"FIXTURE": repo},
        runner=recording_runner,
    )

    assert ownership["nested/new.txt"].ownership is GitOwnership.UNTRACKED_IN_GIT_WORKTREE
    assert not any(args[:2] == ("ls-files", "--others") for args in calls)


def test_does_not_promote_an_ancestor_repo_without_seed_or_indexed_git_directory(git_fixture):
    repo, _ = git_fixture
    nested = repo / "ancestor-only"
    nested.mkdir()
    asset = _asset(repo, "ancestor-only/not-a-repository.txt")

    mapper = GitOwnershipMapper(
        repository_seeds=(),
        root_paths={"FIXTURE": repo},
    )
    ownership = mapper.map((asset,))

    assert ownership["ancestor-only/not-a-repository.txt"].ownership is GitOwnership.NON_GIT_EVIDENCE
    assert mapper.repositories == ()


def test_linked_worktree_is_expanded_and_queried_from_its_own_root(git_fixture):
    repo, _ = git_fixture
    linked = repo.parent / "linked-worktree"
    _git(repo, "worktree", "add", "--detach", str(linked))
    (linked / "linked-draft.txt").write_text("linked\n", encoding="utf-8")
    linked_asset = _asset(linked, "linked-draft.txt", root_id="LINKED")
    main_asset = _asset(repo, "tracked.txt", root_id="MAIN")
    calls: list[tuple[Path, tuple[str, ...]]] = []

    def recording_runner(cwd, args, *, input=b""):
        calls.append((Path(cwd), tuple(args)))
        return subprocess_git_runner(cwd, args, input=input)

    ownership = map_git_ownership(
        (main_asset, linked_asset),
        repository_seeds=(repo,),
        root_paths={"MAIN": repo, "LINKED": linked},
        runner=recording_runner,
    )
    main_record = ownership["tracked.txt"]
    linked_record = ownership["linked-draft.txt"]
    linked_head = _git(linked, "rev-parse", "HEAD").stdout.decode("ascii").strip()

    assert main_record.repository_root == str(repo.resolve())
    assert main_record.branch == "main"
    assert linked_record.ownership is GitOwnership.UNTRACKED_IN_GIT_WORKTREE
    assert linked_record.repository_root == str(linked.resolve())
    assert linked_record.common_git_dir == str((repo / ".git").resolve())
    assert linked_record.branch == "DETACHED"
    assert linked_record.head_commit == linked_head
    assert linked_record.status_summary is not None
    assert "linked-draft.txt" in linked_record.status_summary
    assert "? draft.txt" not in linked_record.status_summary
    assert str(linked.resolve()) in (linked_record.linked_worktrees or ())
    assert any(
        cwd == linked.resolve() and args[:2] == ("ls-files", "--stage")
        for cwd, args in calls
    )


def test_linked_candidates_expand_their_common_repository_only_once(git_fixture):
    repo, _ = git_fixture
    linked = repo.parent / "linked-candidate"
    _git(repo, "worktree", "add", "--detach", str(linked))
    main_asset = _asset(repo, "tracked.txt", root_id="MAIN")
    linked_asset = _asset(linked, "tracked.txt", root_id="LINKED")
    calls: list[tuple[Path, tuple[str, ...]]] = []

    def recording_runner(cwd, args, *, input=b""):
        calls.append((Path(cwd).resolve(), tuple(args)))
        return subprocess_git_runner(cwd, args, input=input)

    ownership = map_git_ownership(
        (main_asset, linked_asset),
        repository_seeds=(repo, linked),
        root_paths={"MAIN": repo, "LINKED": linked},
        runner=recording_runner,
    )

    assert ownership[main_asset.asset_id].repository_root == str(repo.resolve())
    assert ownership[linked_asset.asset_id].repository_root == str(linked.resolve())
    worktree_list_calls = [args for _, args in calls if args == ("worktree", "list", "--porcelain")]
    status_roots = [
        cwd
        for cwd, args in calls
        if args == ("status", "--porcelain=v2", "-z")
    ]
    assert len(worktree_list_calls) == 1
    assert status_roots == [repo.resolve(), linked.resolve()]
