#!/usr/bin/env bash
set -euo pipefail

if [[ "${CVS_AUTO_PUSH_DISABLE:-0}" == "1" ]]; then
    exit 0
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

branch="$(git symbolic-ref --quiet --short HEAD || true)"
if [[ -z "$branch" ]]; then
    echo "[auto-push] cannot push a detached HEAD" >&2
    exit 1
fi

remote="${GIT_AUTO_PUSH_REMOTE:-$(git config --get "branch.${branch}.remote" || true)}"
if [[ -z "$remote" ]]; then
    remote="origin"
fi
if ! git remote get-url "$remote" >/dev/null 2>&1; then
    echo "[auto-push] remote is not configured: $remote" >&2
    exit 1
fi

upstream_ref="$(git rev-parse --symbolic-full-name '@{upstream}' 2>/dev/null || true)"
if [[ -n "$upstream_ref" ]]; then
    git push "$remote"
else
    git push --set-upstream "$remote" "HEAD:refs/heads/$branch"
    upstream_ref="refs/remotes/$remote/$branch"
fi

head_oid="$(git rev-parse HEAD)"
tracking_oid="$(git rev-parse "$upstream_ref")"
if [[ "$tracking_oid" != "$head_oid" ]]; then
    echo "[auto-push] tracking ref does not match HEAD: $tracking_oid != $head_oid" >&2
    exit 1
fi

upstream_name="${upstream_ref#refs/remotes/}"
upstream_remote="${upstream_name%%/*}"
upstream_branch="${upstream_name#*/}"
remote_oid="$(git ls-remote "$upstream_remote" "refs/heads/$upstream_branch" | awk 'NR == 1 {print $1}')"
if [[ -z "$remote_oid" || "$remote_oid" != "$head_oid" ]]; then
    echo "[auto-push] remote readback does not match HEAD: ${remote_oid:-<missing>} != $head_oid" >&2
    exit 1
fi

printf '[auto-push] VERIFIED %s -> %s/%s\n' "$head_oid" "$upstream_remote" "$upstream_branch"
