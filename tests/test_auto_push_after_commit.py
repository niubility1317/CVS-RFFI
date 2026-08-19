from __future__ import annotations

import os
import shlex
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTO_PUSH_SCRIPT = REPO_ROOT / "scripts" / "auto_push_after_commit.sh"


def _git(cwd: Path, *args: str) -> str:
    env = os.environ.copy()
    env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"})
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _configure_identity(repo: Path) -> None:
    _git(repo, "config", "user.name", "auto-push-test")
    _git(repo, "config", "user.email", "auto-push-test@example.invalid")


def _install_test_hook(repo: Path) -> None:
    assert AUTO_PUSH_SCRIPT.is_file(), "the automatic push implementation is missing"
    hook = repo / ".git" / "hooks" / "post-commit"
    hook.write_text(
        "#!/usr/bin/env bash\n"
        f"exec bash {shlex.quote(AUTO_PUSH_SCRIPT.as_posix())}\n",
        encoding="utf-8",
        newline="\n",
    )
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_post_commit_pushes_existing_and_new_branch_without_manual_push(tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    repo = tmp_path / "repo"

    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "init", "-b", "main", str(seed))
    _configure_identity(seed)
    (seed / "README.md").write_text("seed\n", encoding="utf-8")
    _git(seed, "add", "README.md")
    _git(seed, "commit", "-m", "seed")
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "origin", "HEAD:refs/heads/main")

    _git(tmp_path, "init", str(repo))
    _configure_identity(repo)
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "fetch", "origin", "main")
    _git(repo, "checkout", "-b", "main", "FETCH_HEAD")
    _install_test_hook(repo)

    (repo / "README.md").write_text("first automatic push\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "push existing branch")

    assert _git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}") == "origin/main"
    assert _git(repo, "ls-remote", "origin", "refs/heads/main").split()[0] == _git(repo, "rev-parse", "HEAD")

    _git(repo, "checkout", "-b", "topic")
    (repo / "topic.txt").write_text("new branch\n", encoding="utf-8")
    _git(repo, "add", "topic.txt")
    _git(repo, "commit", "-m", "push new branch")

    assert _git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}") == "origin/topic"
    assert _git(repo, "ls-remote", "origin", "refs/heads/topic").split()[0] == _git(repo, "rev-parse", "HEAD")
