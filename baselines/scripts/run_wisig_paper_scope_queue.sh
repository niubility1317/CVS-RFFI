#!/usr/bin/env bash
set -euo pipefail

# Thin paper-audit wrapper. The repo-root launcher owns protocol defaults,
# manifests, logs, and satellite-view augmentation boundaries.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

exec bash "${REPO_ROOT}/scripts/launchers/run_cvs_baseline_queue.sh" "$@"
