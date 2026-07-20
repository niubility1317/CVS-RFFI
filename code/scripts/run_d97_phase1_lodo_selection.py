"""Run one immutable D97 Phase1 receiver-LODO selection from a sealed config."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for candidate in (str(REPO_ROOT), str(CODE_ROOT)):
    while candidate in sys.path:
        sys.path.remove(candidate)
for candidate in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, candidate)

from cvsrffi.phase2_candidate_capsule import BASE_CHECKPOINT_SHA256  # noqa: E402
from cvsrffi.stage2_d81_phase1_episode_scorer import (  # noqa: E402
    D81Phase1EpisodeScorer,
)
from cvsrffi.stage2_d96_d97_phase1_lodo import (  # noqa: E402
    canonical_sha256,
    run_phase1_lodo_selection,
    verify_receipt,
)


CONFIG_SCHEMA = "cvs.phase1.d97_lodo_release_config.v1"
RESULT_SCHEMA = "cvs.phase1.d97_lodo_release_result.v1"
REQUIRED_CONFIG_KEYS = {
    "schema",
    "run_id",
    "seed",
    "archive_path",
    "archive_manifest_path",
    "archive_manifest_sha256",
    "ground_component_dir",
    "ground_manifest_sha256",
    "phase1_checkpoint_sha256",
    "device",
    "candidate_grid",
    "expected_module_sha256",
    "output_dir",
}
REQUIRED_MODULES = {
    "stage2_d81_phase1_episode_scorer": (
        CODE_ROOT / "cvsrffi/stage2_d81_phase1_episode_scorer.py"
    ),
    "stage2_d96_d97_phase1_lodo": (
        CODE_ROOT / "cvsrffi/stage2_d96_d97_phase1_lodo.py"
    ),
}


class D97ReleaseRunnerError(ValueError):
    """Raised when the sealed D97 release config or output lifecycle drifts."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise D97ReleaseRunnerError(f"{name} must be lowercase SHA256")
    return normalized


def _read_bound_json(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    source = Path(path).resolve()
    expected = _require_sha256(expected_sha256, "config SHA256")
    if not source.is_file() or _sha256_file(source) != expected:
        raise D97ReleaseRunnerError("release config path/SHA256 drift")
    try:
        value = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise D97ReleaseRunnerError("release config is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise D97ReleaseRunnerError("release config must be a JSON object")
    return value


def _validate_grid(value: Any) -> dict[str, list[float]]:
    expected = {"beta", "temp_base", "temp_qk", "eta_max", "k1_eta_prior"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise D97ReleaseRunnerError("candidate_grid key closure drift")
    grid: dict[str, list[float]] = {}
    for name in sorted(expected):
        raw = value[name]
        if not isinstance(raw, list) or not raw:
            raise D97ReleaseRunnerError(f"candidate_grid.{name} must be nonempty")
        try:
            converted = [float(item) for item in raw]
        except (TypeError, ValueError) as exc:
            raise D97ReleaseRunnerError(
                f"candidate_grid.{name} must contain finite numbers"
            ) from exc
        if not all(item == item and abs(item) != float("inf") for item in converted):
            raise D97ReleaseRunnerError(
                f"candidate_grid.{name} must contain finite numbers"
            )
        grid[name] = converted
    return grid


def validate_release_config(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != REQUIRED_CONFIG_KEYS or value.get("schema") != CONFIG_SCHEMA:
        raise D97ReleaseRunnerError("release config exact schema drift")
    run_id = str(value["run_id"])
    if not run_id or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for ch in run_id):
        raise D97ReleaseRunnerError("run_id must be a lowercase immutable identifier")
    seed = value["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0x7FFFFFFF:
        raise D97ReleaseRunnerError("seed must be a nonnegative 31-bit integer")
    checkpoint_sha = _require_sha256(
        value["phase1_checkpoint_sha256"], "Phase1 checkpoint"
    )
    if checkpoint_sha != BASE_CHECKPOINT_SHA256:
        raise D97ReleaseRunnerError("Phase1 checkpoint identity drift")
    module_hashes = value["expected_module_sha256"]
    if not isinstance(module_hashes, Mapping) or set(module_hashes) != set(REQUIRED_MODULES):
        raise D97ReleaseRunnerError("expected_module_sha256 closure drift")
    verified_modules: dict[str, str] = {}
    for name, path in REQUIRED_MODULES.items():
        expected = _require_sha256(module_hashes[name], f"module {name}")
        if not path.is_file() or _sha256_file(path) != expected:
            raise D97ReleaseRunnerError(f"module source SHA256 drift: {name}")
        verified_modules[name] = expected
    output_dir = Path(str(value["output_dir"])).resolve()
    if output_dir.exists():
        raise D97ReleaseRunnerError("immutable output_dir already exists")
    validated = dict(value)
    validated["candidate_grid"] = _validate_grid(value["candidate_grid"])
    validated["archive_manifest_sha256"] = _require_sha256(
        value["archive_manifest_sha256"], "archive manifest"
    )
    validated["ground_manifest_sha256"] = _require_sha256(
        value["ground_manifest_sha256"], "ground manifest"
    )
    validated["phase1_checkpoint_sha256"] = checkpoint_sha
    validated["expected_module_sha256"] = verified_modules
    validated["output_dir"] = str(output_dir)
    return validated


def run_from_config(
    config_path: str | Path,
    config_sha256: str,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = validate_release_config(_read_bound_json(config_file, config_sha256))
    scorer = D81Phase1EpisodeScorer.from_component(
        config["ground_component_dir"],
        config["ground_manifest_sha256"],
        device=config["device"],
        metric_seed=config["seed"],
        phase1_checkpoint_sha256=config["phase1_checkpoint_sha256"],
    )
    receipt = run_phase1_lodo_selection(
        config["archive_path"],
        config["candidate_grid"],
        base_scorer=scorer,
        base_scorer_id=scorer.scorer_id,
        feature_archive_manifest_path=config["archive_manifest_path"],
        feature_archive_manifest_sha256=config["archive_manifest_sha256"],
        base_scorer_receipt_sha256=scorer.scorer_id,
        seed=config["seed"],
    )
    if not verify_receipt(receipt):
        raise D97ReleaseRunnerError("D97 selection receipt verification failed")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = output_dir / "d97_phase1_lodo_receipt.json"
    result_path = output_dir / "release_result.json"
    receipt_bytes = (
        json.dumps(
            receipt,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    receipt_path.write_bytes(receipt_bytes)
    result = {
        "schema": RESULT_SCHEMA,
        "run_id": config["run_id"],
        "config_path": str(config_file),
        "config_sha256": _require_sha256(config_sha256, "config SHA256"),
        "config_canonical_sha256": canonical_sha256(config),
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": _sha256_file(receipt_path),
        "receipt_content_sha256": receipt["receipt_sha256"],
        "development_lock_frozen": receipt["development_lock_frozen"],
        "full_phase1_lock": receipt["full_phase1_lock"],
        "formal_target_claim_allowed": receipt["formal_target_claim_allowed"],
        "selected_parameters": receipt["selected_parameters"],
        "outer_lodo_summary": receipt["outer_lodo_summary"],
        "final_lock_evaluation_summary": receipt["final_lock_evaluation_summary"],
        "int8_margin_audit": receipt["int8_margin_audit"]["aggregate"],
    }
    result_path.write_text(
        json.dumps(
            result,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {**result, "result_path": str(result_path), "result_sha256": _sha256_file(result_path)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        json.dumps(
            run_from_config(args.config, args.config_sha256),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

