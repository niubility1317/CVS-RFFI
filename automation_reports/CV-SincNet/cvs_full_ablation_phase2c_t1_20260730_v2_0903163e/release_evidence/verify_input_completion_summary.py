"""Fail closed unless an input-completion summary has exact expected counts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
METHOD_SEEDS = (7282101, 7282102, 7282103)
IDENTITY_PROFILES = {
    "stage2c-package": (
        ("receiver", "method_seed", "stage"),
        {
            (receiver, method_seed, stage)
            for receiver in RECEIVERS
            for method_seed in METHOD_SEEDS
            for stage in ("before", "new20", "new5")
        },
    ),
    "stage2c-feature": (
        ("receiver", "method_seed", "variant", "k_shot"),
        {
            (receiver, method_seed, variant, k_shot)
            for receiver in RECEIVERS
            for method_seed in METHOD_SEEDS
            for variant, k_values in {
                "new20": (1, 2, 5, 10),
                "new5": (10,),
            }.items()
            for k_shot in k_values
        },
    ),
    "stage2c-sidecar": (
        ("receiver", "method_seed", "variant"),
        {
            (receiver, method_seed, variant)
            for receiver in RECEIVERS
            for method_seed in METHOD_SEEDS
            for variant in ("new20", "new5")
        },
    ),
}


def _parse_scalar(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--equal", action="append", default=[])
    parser.add_argument("--result-identity-fields", default="")
    parser.add_argument(
        "--identity-profile",
        choices=sorted(IDENTITY_PROFILES),
    )
    parser.add_argument(
        "--artifact-profile",
        choices=sorted(IDENTITY_PROFILES),
    )
    parser.add_argument("--result-equal", action="append", default=[])
    parser.add_argument("--result-path-field", action="append", default=[])
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _equalities(items: list[str], *, option: str) -> dict[str, Any]:
    expected: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{option} must use field=JSON_VALUE")
        key, raw = item.split("=", 1)
        if not key or key in expected:
            raise ValueError(f"{option} field is empty or repeated")
        expected[key] = _parse_scalar(raw)
    return expected


def _verify_package_artifacts(result: dict[str, Any]) -> None:
    from cvsrffi.stage2_scoring_sidecar import (  # noqa: PLC0415
        load_verified_scoring_sidecar,
    )
    from cvsrffi.stage2_predictor_bundle import (  # noqa: PLC0415
        preflight_stage2_predictor_package,
    )

    output = Path(str(result["output"]))
    seal_path = output / "predictor.seal.json"
    manifest, _seal, _audit = preflight_stage2_predictor_package(
        output / "predictor",
        detached_seal_path=seal_path,
        expected_seal_sha256=_sha256(seal_path),
    )
    scoring_path = output / "scorer" / "scoring_manifest.json"
    truth, _scoring, _scoring_audit = load_verified_scoring_sidecar(
        scoring_path
    )
    expected_stage = (
        "stage2b" if result["stage"] == "before" else "stage2c"
    )
    if (
        manifest.get("stage") != expected_stage
        or manifest.get("receiver") != result["receiver"]
        or int(manifest.get("seed", -1)) != int(result["method_seed"])
        or truth.get("stage") != expected_stage
        or truth.get("receiver") != result["receiver"]
        or int(truth.get("seed", -1)) != int(result["method_seed"])
        or not list(truth.get("rows") or [])
    ):
        raise ValueError("package artifact identity drift")


def _verify_feature_artifacts(result: dict[str, Any]) -> None:
    from cvsrffi.stage2_ablation_feature_cache import (  # noqa: PLC0415
        load_feature_cache,
    )

    log_path = Path(str(result["log"]))
    values = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(values) != 1 or not isinstance(values[0], dict):
        raise ValueError("feature receipt log must contain one JSON object")
    receipt = values[0]
    caches = receipt.get("caches")
    if (
        receipt.get("receiver") != result["receiver"]
        or int(receipt.get("method_seed", -1)) != int(result["method_seed"])
        or int(receipt.get("k_shot", -1)) != int(result["k_shot"])
        or set(caches or {}) != {"stage2a", "stage2b", "stage2c"}
        or receipt.get("query_truth_opened") is not False
        or receipt.get("raw_dataset_opened") is not False
    ):
        raise ValueError("feature receipt identity drift")
    loaded: dict[str, dict[str, Any]] = {}
    for scope, item in caches.items():
        loaded[scope] = load_feature_cache(
            item["payload_path"],
            item["manifest_path"],
            expected_payload_sha256=item["payload_sha256"],
            expected_manifest_sha256=item["manifest_sha256"],
        )
    manifest = loaded["stage2c"]["manifest"]
    expected_new_count = 20 if result["variant"] == "new20" else 5
    if (
        manifest.get("stage_scope") != "stage2c"
        or manifest.get("receiver") != result["receiver"]
        or int(manifest.get("method_seed", -1)) != int(result["method_seed"])
        or int(manifest.get("k_shot", -1)) != int(result["k_shot"])
        or len(list(manifest.get("new_classes") or [])) != expected_new_count
        or manifest.get("query_truth_present") is not False
        or manifest.get("clean_source_samples_present") is not False
    ):
        raise ValueError("feature artifact identity drift")


def _verify_sidecar_artifacts(result: dict[str, Any]) -> None:
    from cvsrffi.stage2_metric_scorer import (  # noqa: PLC0415
        load_verified_scoring_sidecar,
    )

    output = Path(str(result["output"]))
    scoring_path = output / "scoring_manifest.json"
    truth, _manifest, _audit = load_verified_scoring_sidecar(
        scoring_path,
        expected_scoring_manifest_sha256=_sha256(scoring_path),
    )
    if (
        truth.get("stage") != "stage2c"
        or truth.get("receiver") != result["receiver"]
        or int(truth.get("seed", -1)) != int(result["method_seed"])
        or not list(truth.get("rows") or [])
    ):
        raise ValueError("sidecar artifact identity drift")


ARTIFACT_VERIFIERS = {
    "stage2c-package": _verify_package_artifacts,
    "stage2c-feature": _verify_feature_artifacts,
    "stage2c-sidecar": _verify_sidecar_artifacts,
}


def main() -> int:
    args = _parse_args()
    path = args.summary.absolute()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("completion summary is not a regular file")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_sha256):
        raise ValueError("expected SHA-256 must be lowercase hexadecimal")
    if _sha256(path) != args.expected_sha256:
        raise ValueError("completion summary SHA-256 drift")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("schema") != args.schema:
        raise ValueError("completion summary schema drift")
    expected = _equalities(args.equal, option="--equal")
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(
                f"completion summary field drift: {key}={summary.get(key)!r}"
            )
    results = summary.get("results")
    if not isinstance(results, list):
        raise ValueError("completion summary results must be a list")
    summary_expected = summary.get("expected")
    if (
        not isinstance(summary_expected, int)
        or isinstance(summary_expected, bool)
        or len(results) != summary_expected
    ):
        raise ValueError("completion summary result count drift")
    identity_fields = tuple(
        field.strip()
        for field in args.result_identity_fields.split(",")
        if field.strip()
    )
    if len(identity_fields) != len(set(identity_fields)):
        raise ValueError("result identity field is repeated")
    expected_identities: set[tuple[Any, ...]] | None = None
    if args.identity_profile:
        profile_fields, expected_identities = IDENTITY_PROFILES[
            args.identity_profile
        ]
        if identity_fields != profile_fields:
            raise ValueError("result identity fields do not match profile")
    if args.artifact_profile and args.artifact_profile != args.identity_profile:
        raise ValueError("artifact profile must match identity profile")
    result_expected = _equalities(args.result_equal, option="--result-equal")
    path_fields = tuple(args.result_path_field)
    if len(path_fields) != len(set(path_fields)) or any(
        not field for field in path_fields
    ):
        raise ValueError("result path field is empty or repeated")
    identities: set[tuple[Any, ...]] = set()
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"completion result {index} is not an object")
        if identity_fields:
            try:
                identity = tuple(result[field] for field in identity_fields)
            except KeyError as exc:
                raise ValueError(
                    f"completion result {index} lacks identity field"
                ) from exc
            if any(value is None or value == "" for value in identity):
                raise ValueError(f"completion result {index} has empty identity")
            if identity in identities:
                raise ValueError(f"completion result identity duplicated: {identity}")
            identities.add(identity)
        for key, value in result_expected.items():
            if result.get(key) != value:
                raise ValueError(
                    f"completion result {index} field drift: "
                    f"{key}={result.get(key)!r}"
                )
        for field in path_fields:
            raw_path = result.get(field)
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(
                    f"completion result {index} lacks path field {field}"
                )
            artifact_path = Path(raw_path)
            if (
                not artifact_path.is_absolute()
                or not artifact_path.exists()
                or artifact_path.is_symlink()
            ):
                raise FileNotFoundError(
                    f"completion result {index} artifact is unavailable: "
                    f"{artifact_path}"
                )
        if args.artifact_profile:
            ARTIFACT_VERIFIERS[args.artifact_profile](result)
    if expected_identities is not None and identities != expected_identities:
        missing = len(expected_identities - identities)
        extra = len(identities - expected_identities)
        raise ValueError(
            "completion result identity coverage drift: "
            f"missing={missing}, extra={extra}"
        )
    print(
        json.dumps(
            {
                "status": "INPUT_ARTIFACTS_COMPLETE",
                "schema": args.schema,
                "summary": str(path),
                "summary_sha256": args.expected_sha256,
                "verified_fields": expected,
                "result_count": len(results),
                "result_identity_fields": identity_fields,
                "result_identity_count": len(identities),
                "identity_profile": args.identity_profile,
                "artifact_profile": args.artifact_profile,
                "result_verified_fields": result_expected,
                "result_path_fields": path_fields,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
