"""Sealed same-forward pre-ReLU160 runtime for D92-Lite-PR160.

The package seal and the original D92 runtime remain the source of received-IQ
and package identity.  This module loads a separately hashed graph-derived
extractor whose first output is ``joint_proj.0`` from that same forward.  It
does not read clean/source samples and it never exposes query labels or roles.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import stage2_d108_target125_runner as d108_runner
from .stage2_next_r1_tsl import normalize_signed_prerelu160


SOURCE_RUNTIME_SHA256 = "f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a"
FEATURE_WIDTH = 160
EXTRACTOR_RUNTIME_SHA256 = "56612c66b49c8167b3fbed0be5aaa25649a3246a178903618274048d541d80a3"


class PR160RuntimeError(d108_runner.D108Target125RunnerError):
    """Raised when the sealed PR160 runtime or package view drifts."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, name: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise PR160RuntimeError(f"{name} must be a regular non-symlink file")
    return source.resolve(strict=True)


def load_pr160_runtime(
    path: str | Path, *, expected_sha256: str, device: torch.device
) -> torch.jit.ScriptModule:
    """Load one immutable graph-derived runtime after hashing its open file."""

    if expected_sha256 != EXTRACTOR_RUNTIME_SHA256:
        raise PR160RuntimeError("PR160 extractor runtime identity drift")
    source = _regular_file(Path(path), "PR160 extractor runtime")
    if _sha256_file(source) != expected_sha256:
        raise PR160RuntimeError("PR160 extractor runtime SHA mismatch")
    try:
        with source.open("rb") as handle:
            actual = hashlib.sha256(handle.read()).hexdigest()
            if actual != expected_sha256:
                raise PR160RuntimeError("PR160 extractor runtime changed during load")
            handle.seek(0)
            model = torch.jit.load(handle, map_location=device)
    except PR160RuntimeError:
        raise
    except Exception as error:  # pragma: no cover - Torch backend detail
        raise PR160RuntimeError("PR160 extractor runtime load failed") from error
    model.eval()
    return model


def forward_signed_pr160(
    model: torch.jit.ScriptModule,
    rows: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Run the extractor and apply the one signed-totalization rule."""

    try:
        from .stage2_diag_cosine_exploration import forward_zid160

        pre = forward_zid160(
            model,
            np.ascontiguousarray(rows, dtype=np.float32),
            device=device,
            batch_size=batch_size,
        )
        return normalize_signed_prerelu160(pre)
    except PR160RuntimeError:
        raise
    except Exception as error:
        raise PR160RuntimeError("same-forward signed PR160 materialization failed") from error


class PR160StateMaterializer:
    """Typed D108 state provider returning only normalized PR160 features."""

    feature_width = FEATURE_WIDTH

    def __init__(
        self,
        *,
        plan: Mapping[str, Any],
        device: str,
        support_batch_size: int,
        extractor_runtime_path: str | Path,
        expected_extractor_runtime_sha256: str,
        expected_source_runtime_sha256: str = SOURCE_RUNTIME_SHA256,
    ) -> None:
        if type(support_batch_size) is not int or support_batch_size != 64:
            raise PR160RuntimeError("PR160 support_batch_size must equal 64")
        if expected_source_runtime_sha256 != SOURCE_RUNTIME_SHA256:
            raise PR160RuntimeError("PR160 source runtime identity drift")
        if expected_extractor_runtime_sha256 != EXTRACTOR_RUNTIME_SHA256:
            raise PR160RuntimeError("PR160 extractor runtime identity drift")
        identity = plan.get("identity")
        if not isinstance(identity, Mapping) or identity.get(
            "d92_sealed_runtime_sha256"
        ) != SOURCE_RUNTIME_SHA256:
            raise PR160RuntimeError("D92 package runtime is not the bound source runtime")
        try:
            from .stage2_diag_cosine_exploration import _device

            self.device = _device(device)
            self.model = load_pr160_runtime(
                extractor_runtime_path,
                expected_sha256=expected_extractor_runtime_sha256,
                device=self.device,
            )
        except PR160RuntimeError:
            raise
        except Exception as error:  # pragma: no cover - environment closure
            raise PR160RuntimeError("PR160 runtime/device setup failed") from error
        self.plan = plan
        self.support_batch_size = 64
        self.package_cache: dict[
            tuple[tuple[str, str], ...], tuple[Any, dict[str, Any], dict[str, Any]]
        ] = {}
        self.d92_fit = lambda *_args, **_kwargs: None

    def _package(self, reference: Mapping[str, Any]):
        key = tuple(sorted((str(name), str(value)) for name, value in reference.items()))
        if key not in self.package_cache:
            try:
                self.package_cache[key] = d108_runner._package_payloads(reference)
            except Exception as error:
                raise PR160RuntimeError("sealed D92 package verification failed") from error
        payloads, manifest, audit = self.package_cache[key]
        try:
            from .stage2_diag_cosine_exploration import _descriptor

            descriptor = _descriptor(manifest, "feature_runtime")
            if descriptor.get("sha256") != SOURCE_RUNTIME_SHA256:
                raise PR160RuntimeError("package/source runtime SHA drift")
        except PR160RuntimeError:
            raise
        except Exception as error:
            raise PR160RuntimeError("package feature runtime descriptor drift") from error
        return payloads, manifest, audit

    def _features(self, rows: np.ndarray, *, batch_size: int) -> np.ndarray:
        try:
            return forward_signed_pr160(
                self.model,
                rows,
                device=self.device,
                batch_size=batch_size,
            )
        except PR160RuntimeError:
            raise
        except Exception as error:
            raise PR160RuntimeError("PR160 feature materialization failed") from error

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        phase = request.get("phase")
        if phase not in d108_runner.PHASES:
            raise PR160RuntimeError("state request phase drift")
        packages = request.get("packages")
        if not isinstance(packages, Mapping):
            raise PR160RuntimeError("state request packages missing")
        support_ref = packages[f"{phase}_enrollment"]
        query_ref = packages[f"{phase}_apply"]
        support_payloads, support_manifest, _ = self._package(support_ref)
        query_payloads, query_manifest, _ = self._package(query_ref)
        try:
            from .stage2_diag_cosine_exploration import _validate_matched_packages

            _validate_matched_packages(support_manifest, query_manifest)
        except Exception as error:
            raise PR160RuntimeError("D92 support/query package pairing drift") from error
        scene = request["scene"]
        if scene not in support_payloads or scene not in query_payloads:
            raise PR160RuntimeError("D92 package scene is missing")
        registry = tuple(
            str(item.get("class_handle", ""))
            for item in support_manifest.get("registered_classes", [])
            if isinstance(item, Mapping)
        )
        if not registry or len(set(registry)) != len(registry):
            raise PR160RuntimeError("D92 registered-class contract drift")
        for manifest in (support_manifest, query_manifest):
            if (
                manifest.get("receiver") != request["receiver"]
                or manifest.get("seed") != request["seed"]
                or manifest.get("k_shot") != request["source_pool_k"]
            ):
                raise PR160RuntimeError("D92 package row binding drift")
        support_iq, labels, support_ids = d108_runner._support_rows(
            support_payloads[scene],
            registered_classes=registry,
            active_k=request["k_shot"],
        )
        query_iq, query_ids = d108_runner._query_rows(query_payloads[scene])
        if support_iq.shape[1:] != query_iq.shape[1:]:
            raise PR160RuntimeError("support/query IQ shape drift")
        return {
            "support_features": self._features(
                support_iq, batch_size=self.support_batch_size
            ),
            "support_labels": labels,
            "registered_classes": registry,
            "support_physical_ids": support_ids,
            "query_features": self._features(query_iq, batch_size=1),
            "query_physical_ids": query_ids,
        }


__all__ = [
    "FEATURE_WIDTH",
    "EXTRACTOR_RUNTIME_SHA256",
    "PR160RuntimeError",
    "PR160StateMaterializer",
    "SOURCE_RUNTIME_SHA256",
    "forward_signed_pr160",
    "load_pr160_runtime",
]
