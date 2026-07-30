"""Build the exact 75-entry Stage2-C cache-binding index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
METHOD_SEEDS = (7282101, 7282102, 7282103)
VARIANT_K = {
    "new20": (1, 2, 5, 10),
    "new5": (10,),
}
NEW_CLASS_DRAW_SEED = 7282401
EXPECTED_ENTRIES = 75
SCHEMA = "cvs.full_ablation.phase2.cache_binding_index.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _regular(path: Path) -> Path:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"binding artifact is not a regular file: {path}")
    return path


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(_regular(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"binding artifact must be a JSON object: {path}")
    return value


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o444,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing cache-binding index")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)


def main() -> int:
    args = _parse_args()
    if not all(
        path.is_absolute()
        for path in (
            args.package_root,
            args.feature_root,
            args.sidecar_root,
            args.output,
        )
    ):
        raise ValueError("all cache-binding paths must be absolute")
    entries: list[dict[str, Any]] = []
    candidate_lock_sha256 = ""
    identities: set[tuple[Any, ...]] = set()
    for receiver in RECEIVERS:
        rx = receiver.replace("-", "_")
        for method_seed in METHOD_SEEDS:
            support_seed = method_seed + 100
            query_seed = method_seed + 200
            for variant, k_values in VARIANT_K.items():
                new_count = 20 if variant == "new20" else 5
                package = (
                    args.package_root
                    / "artifacts"
                    / "packages"
                    / f"rx_{rx}"
                    / f"method_{method_seed}"
                    / variant
                )
                package_root = package / "predictor"
                package_manifest = _load(package_root / "package_manifest.json")
                seal_path = _regular(package / "predictor.seal.json")
                sidecar_manifest = _regular(
                    args.sidecar_root
                    / "artifacts"
                    / "sidecars"
                    / "stage2c"
                    / f"rx_{rx}"
                    / f"method_{method_seed}"
                    / variant
                    / "scoring_manifest.json"
                )
                current_lock = str(package_manifest.get("candidate_lock_sha256", ""))
                if candidate_lock_sha256 and current_lock != candidate_lock_sha256:
                    raise ValueError("candidate-lock drift across Stage2-C packages")
                candidate_lock_sha256 = current_lock
                for k_shot in k_values:
                    feature = (
                        args.feature_root
                        / "artifacts"
                        / "features"
                        / f"rx_{rx}"
                        / f"method_{method_seed}"
                        / variant
                        / f"k_{k_shot}"
                        / "stage2c"
                    )
                    payload = _regular(feature / "features.npz")
                    manifest_path = _regular(feature / "features.manifest.json")
                    manifest = _load(manifest_path)
                    identity = (
                        "stage2c",
                        receiver,
                        method_seed,
                        support_seed,
                        query_seed,
                        NEW_CLASS_DRAW_SEED,
                        k_shot,
                        new_count,
                    )
                    manifest_identity = (
                        str(manifest.get("stage_scope", "")),
                        str(manifest.get("receiver", "")),
                        int(manifest.get("method_seed", -1)),
                        int(manifest.get("support_seed", -1)),
                        int(manifest.get("query_seed", -1)),
                        int(manifest.get("new_class_draw_seed", -1)),
                        int(manifest.get("k_shot", -1)),
                        len(list(manifest.get("new_classes") or [])),
                    )
                    if manifest_identity != identity:
                        raise ValueError("Stage2-C feature identity drift")
                    if identity in identities:
                        raise ValueError("duplicate Stage2-C cache identity")
                    identities.add(identity)
                    entries.append(
                        {
                            "stage_scope": "stage2c",
                            "receiver": receiver,
                            "method_seed": method_seed,
                            "support_seed": support_seed,
                            "query_seed": query_seed,
                            "new_class_draw_seed": NEW_CLASS_DRAW_SEED,
                            "k_shot": k_shot,
                            "new_class_count": new_count,
                            "feature_cache_payload": str(payload),
                            "feature_cache_manifest": str(manifest_path),
                            "predictor_package_root": str(package_root),
                            "predictor_detached_seal": str(seal_path),
                            "predictor_detached_seal_sha256": _sha256(seal_path),
                            "scoring_manifest": str(sidecar_manifest),
                        }
                    )
    if (
        len(entries) != EXPECTED_ENTRIES
        or len(identities) != EXPECTED_ENTRIES
        or len(candidate_lock_sha256) != 64
        or any(value not in "0123456789abcdef" for value in candidate_lock_sha256)
    ):
        raise ValueError("Stage2-C cache-binding coverage is incomplete")
    payload = {
        "schema": SCHEMA,
        "candidate_lock_sha256": candidate_lock_sha256,
        "entries": entries,
    }
    _write_new(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "entry_count": len(entries),
                "candidate_lock_sha256": candidate_lock_sha256,
                "cross_launch_data_identity_required": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
