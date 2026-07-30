"""Fail closed unless an input-completion summary has exact expected counts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


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
                "result_verified_fields": result_expected,
                "result_path_fields": path_fields,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
