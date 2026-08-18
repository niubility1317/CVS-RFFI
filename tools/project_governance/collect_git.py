"""Read-only Git ownership mapping for already indexed project assets.

The mapper deliberately works from configured repository seeds and indexed
asset paths.  It never asks Git for an unbounded list of untracked files and
never invokes a worktree-mutating command.  The command runner is injectable
so tests can observe the exact, path-scoped Git calls without touching the
real project worktree.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .models import AssetKind, AssetRecord, GitOwnership, GitOwnershipRecord, Location
from .paths import normalize_relative_path


@dataclass(frozen=True)
class CommandResult:
    """The byte-preserving result returned by an injected Git command runner."""

    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


@dataclass(frozen=True)
class RepositoryRecord:
    """Read-only repository metadata used to enrich asset ownership records."""

    repository_root: str
    common_git_dir: str | None = None
    branch: str | None = None
    head_commit: str | None = None
    status_summary: str | None = None
    linked_worktrees: tuple[str, ...] | None = None
    error: str | None = None


GitCommandRunner = Callable[..., CommandResult]


def subprocess_git_runner(
    cwd: str | os.PathLike[str], args: Sequence[str], *, input: bytes = b""
) -> CommandResult:
    """Run one Git command with captured bytes and no shell or mutation."""

    completed = subprocess.run(
        ["git", *[str(argument) for argument in args]],
        cwd=os.fspath(cwd),
        input=input,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _coerce_result(result: Any) -> CommandResult:
    if isinstance(result, CommandResult):
        return result
    if isinstance(result, tuple) and len(result) == 3:
        return CommandResult(int(result[0]), bytes(result[1]), bytes(result[2]))
    return CommandResult(
        int(result.returncode),
        bytes(getattr(result, "stdout", b"") or b""),
        bytes(getattr(result, "stderr", b"") or b""),
    )


def _decode(value: bytes) -> str:
    return os.fsdecode(value)


def _clean_text(value: bytes) -> str:
    return _decode(value).rstrip("\x00\r\n")


def _path_key(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(path))))


def _resolved_path(path: str | os.PathLike[str]) -> Path:
    return Path(path).resolve(strict=False)


def _under(path: Path, root: Path) -> bool:
    """Return a Windows-safe lexical containment result without I/O."""

    try:
        path.relative_to(root)
        return True
    except ValueError:
        return _path_key(path).startswith(_path_key(root) + os.sep) or _path_key(path) == _path_key(root)


def _is_git_marker(path: Path) -> bool:
    marker = path / ".git"
    return marker.is_dir() or marker.is_file()


def _path_from_asset(asset: AssetRecord, root_paths: Mapping[Any, str | os.PathLike[str]]) -> Path | None:
    if asset.location is not Location.LOCAL:
        return None

    root = root_paths.get(asset.root_id)
    if root is None:
        root = root_paths.get(asset.location)
    if root is None:
        root = root_paths.get(asset.location.value)
    if root is None:
        return None

    try:
        relative = normalize_relative_path(asset.relative_path)
    except (TypeError, ValueError):
        return None
    return Path(root).joinpath(*relative.split("/"))


def _error_text(args: Sequence[str], result: CommandResult) -> str:
    detail = _clean_text(result.stderr) or _clean_text(result.stdout)
    if detail:
        return f"git {' '.join(args)} exited {result.returncode}: {detail}"
    return f"git {' '.join(args)} exited {result.returncode}"


def _parse_worktree_paths(output: bytes) -> tuple[str, ...]:
    paths: list[str] = []
    for line in _decode(output).splitlines():
        if line.startswith("worktree "):
            value = line[len("worktree ") :].strip()
            if value:
                paths.append(str(_resolved_path(value)))
    return tuple(dict.fromkeys(paths))


def _parse_ls_files(output: bytes) -> set[str]:
    tracked: set[str] = set()
    for entry in output.split(b"\x00"):
        if not entry:
            continue
        _, separator, path_bytes = entry.partition(b"\t")
        if not separator:
            continue
        try:
            normalized = normalize_relative_path(_decode(path_bytes))
        except (TypeError, ValueError):
            continue
        tracked.add(normalized)
    return tracked


def _parse_check_ignore(output: bytes) -> set[str]:
    ignored: set[str] = set()
    for path_bytes in output.split(b"\x00"):
        if not path_bytes:
            continue
        try:
            normalized = normalize_relative_path(_decode(path_bytes))
        except (TypeError, ValueError):
            continue
        ignored.add(normalized)
    return ignored


class OwnershipMap(dict[str, GitOwnershipRecord]):
    """Path-keyed records with an asset-id lookup fallback.

    Relative paths are the convenient public key for reports.  The fallback
    keeps the immutable asset identity available when two roots contain the
    same relative path.
    """

    def __init__(
        self,
        records: Mapping[str, GitOwnershipRecord],
        by_asset_id: Mapping[str, GitOwnershipRecord],
    ) -> None:
        super().__init__(records)
        self._by_asset_id = dict(by_asset_id)

    def __getitem__(self, key: str) -> GitOwnershipRecord:
        try:
            return super().__getitem__(key)
        except KeyError:
            return self._by_asset_id[key]

    def __contains__(self, key: object) -> bool:
        return super().__contains__(key) or key in self._by_asset_id


class GitOwnershipMapper:
    """Discover configured repositories and classify indexed assets read-only."""

    def __init__(
        self,
        repository_seeds: Iterable[str | os.PathLike[str]] = (),
        root_paths: Mapping[Any, str | os.PathLike[str]] | None = None,
        runner: GitCommandRunner | None = None,
        *,
        command_runner: GitCommandRunner | None = None,
        indexed_assets: Iterable[AssetRecord] = (),
        batch_size: int = 128,
    ) -> None:
        if runner is not None and command_runner is not None:
            raise ValueError("provide runner or command_runner, not both")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.repository_seeds = tuple(repository_seeds)
        self.root_paths = dict(root_paths or {})
        self.runner = command_runner or runner or subprocess_git_runner
        self.indexed_assets = tuple(indexed_assets)
        self.batch_size = batch_size
        self.repositories: tuple[RepositoryRecord, ...] = ()

    def _run(
        self,
        cwd: str | os.PathLike[str],
        args: Sequence[str],
        *,
        input: bytes = b"",
    ) -> CommandResult:
        try:
            result = self.runner(cwd, tuple(args), input=input)
        except TypeError as first_error:
            # Accept the two common test-double shapes as well as the native
            # ``(cwd, args, *, input=...)`` contract above.  The fallback is
            # only for argument binding; real command failures are returned as
            # CommandResult values and are never retried.
            try:
                result = self.runner(("git", *tuple(args)), cwd=cwd, input=input)
            except TypeError:
                try:
                    result = self.runner(("git", *tuple(args)), cwd, input)
                except TypeError:
                    raise first_error
        except OSError as exc:
            return CommandResult(127, b"", os.fsencode(str(exc)))
        return _coerce_result(result)

    def _asset_path(self, asset: AssetRecord) -> Path | None:
        path = _path_from_asset(asset, self.root_paths)
        if path is not None or asset.location is not Location.LOCAL:
            return path
        if len(self.repository_seeds) != 1:
            return None
        try:
            relative = normalize_relative_path(asset.relative_path)
        except (TypeError, ValueError):
            return None
        return Path(self.repository_seeds[0]).joinpath(*relative.split("/"))

    def _candidate_paths(self, assets: Iterable[AssetRecord]) -> tuple[Path, ...]:
        candidates: dict[str, Path] = {}

        for seed in self.repository_seeds:
            configured = Path(seed)
            if configured.name == ".git" or (configured.exists() and configured.is_file()):
                configured = configured.parent
            resolved = self._resolve_configured_seed(configured)
            if resolved is not None:
                candidates.setdefault(_path_key(resolved), resolved)

        for asset in assets:
            if asset.location is not Location.LOCAL or asset.asset_kind is not AssetKind.DIRECTORY:
                continue
            path = self._asset_path(asset)
            if path is None or not _is_git_marker(path):
                continue
            resolved = _resolved_path(path)
            candidates.setdefault(_path_key(resolved), resolved)
        return tuple(candidates.values())

    def _resolve_configured_seed(self, seed: Path) -> Path | None:
        result = self._run(seed, ("rev-parse", "--show-toplevel"))
        if result.returncode == 0 and _clean_text(result.stdout):
            return _resolved_path(_clean_text(result.stdout))
        if _is_git_marker(seed):
            return _resolved_path(seed)
        return None

    def _worktree_record(
        self,
        worktree_root: Path,
        linked_worktrees: tuple[str, ...],
        inherited_errors: Iterable[str] = (),
    ) -> RepositoryRecord:
        errors = list(inherited_errors)

        common_result = self._run(worktree_root, ("rev-parse", "--git-common-dir"))
        common_git_dir: str | None = None
        if common_result.returncode == 0 and _clean_text(common_result.stdout):
            common = Path(_clean_text(common_result.stdout))
            if not common.is_absolute():
                common = worktree_root / common
            common_git_dir = str(_resolved_path(common))
        else:
            errors.append(_error_text(("rev-parse", "--git-common-dir"), common_result))

        branch_result = self._run(worktree_root, ("symbolic-ref", "--quiet", "--short", "HEAD"))
        branch: str | None
        if branch_result.returncode == 0:
            branch = _clean_text(branch_result.stdout) or None
        elif branch_result.returncode == 1:
            branch = "DETACHED"
        else:
            branch = None
            errors.append(
                _error_text(("symbolic-ref", "--quiet", "--short", "HEAD"), branch_result)
            )

        head_result = self._run(worktree_root, ("rev-parse", "HEAD"))
        head_commit: str | None
        if head_result.returncode == 0:
            head_commit = _clean_text(head_result.stdout) or None
        else:
            head_commit = None
            errors.append(_error_text(("rev-parse", "HEAD"), head_result))

        status_args = ("status", "--porcelain=v2", "-z")
        status_result = self._run(worktree_root, status_args)
        status_summary: str | None
        if status_result.returncode == 0:
            status_summary = _decode(status_result.stdout)
        else:
            status_summary = None
            errors.append(_error_text(status_args, status_result))

        return RepositoryRecord(
            repository_root=str(worktree_root),
            common_git_dir=common_git_dir,
            branch=branch,
            head_commit=head_commit,
            status_summary=status_summary,
            linked_worktrees=linked_worktrees,
            error="; ".join(errors) if errors else None,
        )

    def _repository_record(self, candidate: Path) -> tuple[RepositoryRecord, ...]:
        errors: list[str] = []

        root_result = self._run(candidate, ("rev-parse", "--show-toplevel"))
        if root_result.returncode == 0 and _clean_text(root_result.stdout):
            repository_root = _resolved_path(_clean_text(root_result.stdout))
        else:
            errors.append(_error_text(("rev-parse", "--show-toplevel"), root_result))
            repository_root = _resolved_path(candidate)

        worktree_args = ("worktree", "list", "--porcelain")
        worktree_result = self._run(repository_root, worktree_args)
        if worktree_result.returncode == 0:
            parsed = _parse_worktree_paths(worktree_result.stdout)
            linked_worktrees = parsed or (str(repository_root),)
        else:
            linked_worktrees = (str(repository_root),)
            errors.append(_error_text(worktree_args, worktree_result))

        return tuple(
            self._worktree_record(
                _resolved_path(worktree),
                linked_worktrees,
                errors,
            )
            for worktree in linked_worktrees
        )

    def _common_git_key(self, candidate: Path) -> str:
        result = self._run(candidate, ("rev-parse", "--git-common-dir"))
        if result.returncode != 0 or not _clean_text(result.stdout):
            return _path_key(candidate)
        common = Path(_clean_text(result.stdout))
        if not common.is_absolute():
            common = candidate / common
        return _path_key(_resolved_path(common))

    def discover_repositories(
        self,
        indexed_assets: Iterable[AssetRecord] | None = None,
    ) -> tuple[RepositoryRecord, ...]:
        assets = self.indexed_assets if indexed_assets is None else tuple(indexed_assets)
        candidates = self._candidate_paths(assets)
        discovered: dict[str, RepositoryRecord] = {}
        expanded_common_dirs: set[str] = set()
        for candidate in candidates:
            common_key = self._common_git_key(candidate)
            if common_key in expanded_common_dirs:
                continue
            expanded_common_dirs.add(common_key)
            for record in self._repository_record(candidate):
                key = _path_key(record.repository_root)
                existing = discovered.get(key)
                if existing is None or (existing.error and not record.error):
                    discovered[key] = record
        self.repositories = tuple(discovered.values())
        return self.repositories

    def _repository_for_path(self, path: Path) -> tuple[RepositoryRecord, Path] | None:
        matches: list[tuple[RepositoryRecord, Path]] = []
        for repository in self.repositories:
            worktree_root = _resolved_path(repository.repository_root)
            if _under(path, worktree_root):
                matches.append((repository, worktree_root))
        if not matches:
            return None
        return max(matches, key=lambda item: len(str(item[1])))

    @staticmethod
    def _base_record(
        asset: AssetRecord,
        ownership: GitOwnership,
        repository: RepositoryRecord | None = None,
        error: str | None = None,
    ) -> GitOwnershipRecord:
        return GitOwnershipRecord(
            asset_id=asset.asset_id,
            ownership=ownership,
            repository_root=repository.repository_root if repository else None,
            common_git_dir=repository.common_git_dir if repository else None,
            branch=repository.branch if repository else None,
            head_commit=repository.head_commit if repository else None,
            status_summary=repository.status_summary if repository else None,
            linked_worktrees=repository.linked_worktrees if repository else None,
            error=error or (repository.error if repository else None),
        )

    def _map_repo_batch(
        self,
        repository: RepositoryRecord,
        assets: Sequence[AssetRecord],
        paths: Sequence[str],
        cwd: Path,
    ) -> list[GitOwnershipRecord]:
        if repository.error:
            return [
                self._base_record(asset, GitOwnership.GIT_STATE_ERROR, repository, repository.error)
                for asset in assets
            ]

        ls_args = ("ls-files", "--stage", "-z", "--", *paths)
        ls_result = self._run(cwd, ls_args)
        if ls_result.returncode != 0:
            error = _error_text(ls_args, ls_result)
            return [
                self._base_record(asset, GitOwnership.GIT_STATE_ERROR, repository, error)
                for asset in assets
            ]
        tracked = _parse_ls_files(ls_result.stdout)

        ignore_args = ("check-ignore", "-z", "--stdin")
        ignore_input = b"\x00".join(os.fsencode(path) for path in paths) + b"\x00"
        ignore_result = self._run(cwd, ignore_args, input=ignore_input)
        if ignore_result.returncode not in (0, 1):
            error = _error_text(ignore_args, ignore_result)
            return [
                self._base_record(asset, GitOwnership.GIT_STATE_ERROR, repository, error)
                for asset in assets
            ]
        ignored = _parse_check_ignore(ignore_result.stdout)

        records: list[GitOwnershipRecord] = []
        for asset, path in zip(assets, paths):
            if path in tracked:
                state = GitOwnership.TRACKED_GIT
            elif path in ignored:
                state = GitOwnership.IGNORED_REGENERABLE
            else:
                state = GitOwnership.UNTRACKED_IN_GIT_WORKTREE
            records.append(self._base_record(asset, state, repository))
        return records

    def map(self, assets: Iterable[AssetRecord] | None = None) -> OwnershipMap:
        selected = self.indexed_assets if assets is None else tuple(assets)
        self.discover_repositories(selected)

        records_by_path: dict[str, GitOwnershipRecord] = {}
        records_by_id: dict[str, GitOwnershipRecord] = {}
        grouped: dict[tuple[str, str], list[tuple[AssetRecord, str]]] = {}
        direct: list[GitOwnershipRecord] = []

        for asset in selected:
            if asset.location is Location.N607:
                direct.append(self._base_record(asset, GitOwnership.REMOTE_NON_GIT))
                continue

            path = self._asset_path(asset)
            match = self._repository_for_path(path) if path is not None else None
            if match is None or path is None:
                direct.append(self._base_record(asset, GitOwnership.NON_GIT_EVIDENCE))
                continue
            repository, worktree_root = match

            if repository.error:
                direct.append(
                    self._base_record(asset, GitOwnership.GIT_STATE_ERROR, repository, repository.error)
                )
                continue

            try:
                relative = path.relative_to(worktree_root).as_posix()
            except ValueError:
                direct.append(self._base_record(asset, GitOwnership.NON_GIT_EVIDENCE))
                continue
            group_key = (repository.repository_root, str(worktree_root))
            grouped.setdefault(group_key, []).append((asset, relative))

        mapped: list[GitOwnershipRecord] = list(direct)
        for repository in self.repositories:
            worktree_root = _resolved_path(repository.repository_root)
            entries = grouped.get((repository.repository_root, str(worktree_root)), [])
            for start in range(0, len(entries), self.batch_size):
                batch = entries[start : start + self.batch_size]
                batch_assets = tuple(item[0] for item in batch)
                batch_paths = tuple(item[1] for item in batch)
                mapped.extend(
                    self._map_repo_batch(
                        repository,
                        batch_assets,
                        batch_paths,
                        worktree_root,
                    )
                )

        for record in mapped:
            records_by_id[record.asset_id] = record
        for asset in selected:
            record = records_by_id[asset.asset_id]
            path_key = asset.relative_path
            records_by_path.setdefault(path_key, record)
        return OwnershipMap(records_by_path, records_by_id)

    def map_records(self, assets: Iterable[AssetRecord] | None = None) -> tuple[GitOwnershipRecord, ...]:
        return tuple(self.map(assets).values())

    def attach(self, assets: Iterable[AssetRecord] | None = None) -> tuple[AssetRecord, ...]:
        selected = self.indexed_assets if assets is None else tuple(assets)
        ownership = self.map(selected)
        return tuple(replace(asset, git_ownership=ownership[asset.asset_id].ownership) for asset in selected)


def discover_repositories(
    repository_seeds: Iterable[str | os.PathLike[str]],
    *,
    indexed_assets: Iterable[AssetRecord] = (),
    root_paths: Mapping[Any, str | os.PathLike[str]] | None = None,
    runner: GitCommandRunner | None = None,
    command_runner: GitCommandRunner | None = None,
) -> tuple[RepositoryRecord, ...]:
    selected_assets = tuple(indexed_assets)
    mapper = GitOwnershipMapper(
        repository_seeds,
        root_paths,
        runner,
        command_runner=command_runner,
        indexed_assets=selected_assets,
    )
    return mapper.discover_repositories(selected_assets)


def map_git_ownership(
    assets: Iterable[AssetRecord],
    repository_seeds: Iterable[str | os.PathLike[str]] = (),
    root_paths: Mapping[Any, str | os.PathLike[str]] | None = None,
    runner: GitCommandRunner | None = None,
    *,
    command_runner: GitCommandRunner | None = None,
    batch_size: int = 128,
) -> OwnershipMap:
    return GitOwnershipMapper(
        repository_seeds,
        root_paths,
        runner,
        command_runner=command_runner,
        batch_size=batch_size,
    ).map(assets)


collect_git_ownership = map_git_ownership
GitCollector = GitOwnershipMapper


__all__ = [
    "CommandResult",
    "GitCollector",
    "GitCommandRunner",
    "GitOwnershipMapper",
    "OwnershipMap",
    "RepositoryRecord",
    "collect_git_ownership",
    "discover_repositories",
    "map_git_ownership",
    "subprocess_git_runner",
]
