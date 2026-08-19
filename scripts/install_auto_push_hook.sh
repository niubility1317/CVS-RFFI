#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
common_git_dir="$(git rev-parse --git-common-dir)"
if [[ "$common_git_dir" != /* && "$common_git_dir" != [A-Za-z]:/* ]]; then
    common_git_dir="$repo_root/$common_git_dir"
fi
hook_dir="$common_git_dir/hooks"
hook_path="$hook_dir/post-commit"
source_hook="$repo_root/scripts/auto_push_after_commit.sh"

if [[ ! -f "$source_hook" ]]; then
    echo "[auto-push] hook source is missing: $source_hook" >&2
    exit 1
fi
mkdir -p "$hook_dir"
if [[ -e "$hook_path" ]] && ! cmp -s "$source_hook" "$hook_path"; then
    echo "[auto-push] existing post-commit hook was not replaced: $hook_path" >&2
    exit 1
fi
cp "$source_hook" "$hook_path"
chmod +x "$hook_path"
git config --local push.autoSetupRemote true
git config --local push.default simple
printf '[auto-push] installed hook=%s\n' "$hook_path"
printf '[auto-push] push.autoSetupRemote=%s\n' "$(git config --local --get push.autoSetupRemote)"
