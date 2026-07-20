#!/usr/bin/env python
"""Propose or verify the reviewed ADV3B02 diagnostic source member lock.

The production CLI never signs anything.  ``propose`` writes a candidate and
commit-bound evidence outside the repository.  A human must review both, add a
separate review record plus the lock to Git, and commit them.  ``verify`` then
requires a clean HEAD, rereads both files as Git blobs, and independently
recomputes the static/import closure before emitting external verification
evidence.  The offline signer remains the only source-release producer.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence


LOCK_SCHEMA = "cvs.development.adv3b02_source_member_lock.v1"
REVIEW_SCHEMA = "cvs.development.adv3b02_source_member_lock_human_review.v1"
EVIDENCE_SCHEMA = "cvs.development.adv3b02_source_member_lock_build.v1"
REVIEW_DECISION = "APPROVE_FOR_DEVELOPMENT_SOURCE_RELEASE"
REVIEW_SCOPE = "source_members_only_no_runtime_or_metric_authority"
BLOCKED_REVIEW_STATUS = "BLOCKED_INDEPENDENT_REVIEW_AUTHORITY"
BASE_CHECKPOINT_SHA256 = (
    "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
)
REVIEW_AUTHORITY_SCHEMA = (
    "cvs.development.adv3b02_source_member_review_authority.v1"
)
REVIEW_AUTHORITY_ISSUER = "qknnv42_stage2bc_extreme_light_route_20260716"
REVIEW_AUTHORITY_KEY_ID = "somph-authority-ed25519-20260716"
REVIEW_AUTHORITY_PUBLIC_KEY_HEX = (
    "ec301433b5a625f8e34f887f5aeea664e809236d1b871fcc0ffeb47cb540bdc1"
)
REVIEW_AUTHORITY_PUBLIC_KEY_SHA256 = (
    "52944e59ec99d360e227cbe78e84efeca6db3ebca3d9698f5d567270c37a9444"
)
REVIEWER_IDENTITY = "independent-adv3b02-source-reviewer-v1"
# Provision only after the external reviewer signs the exact tracked review
# record.  A caller-supplied envelope cannot populate this trust root.
TRUSTED_REVIEW_AUTHORITY_ENVELOPE_SHA256: str | None = None

# These are code roots explicitly audited by the signed-source consumer.  The
# static resolver follows their in-repository imports; the child probe records
# modules actually imported in the current production environment.
PRODUCTION_ENTRY_PATHS = (
    "code/SSDG/train_ssdg.py",
    "code/cvsrffi/checkpoint_loading.py",
    "code/cvsrffi/identity_only_forward.py",
    "code/cvsrffi/phase1_adv3b02_deployment_bundle.py",
    "code/cvsrffi/phase1_center_lowrank_prototype_bundle.py",
    "code/cvsrffi/phase2_candidate_capsule.py",
    "code/cvsrffi/somph_runtime_trust.py",
    "code/cvsrffi/stage2_predictor_bundle.py",
    "code/model_dual_cvsincnet.py",
    "code/scripts/diagnose_adv3b02_runtime_numerics.py",
    "code/scripts/export_adv3b02_effective8_torchscript.py",
    "paper_reproduction/scripts/benchmark_cvs_adaptive_rxlight_tta.py",
    "paper_reproduction/scripts/train_export_cvs_support_lora_adapter.py",
)
ENTRY_PATHS = PRODUCTION_ENTRY_PATHS


class MemberLockBuildError(RuntimeError):
    """Raised when the member-lock review boundary is not closed."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_stable_regular(path: str | Path, name: str) -> bytes:
    requested = Path(path)
    try:
        before = requested.lstat()
    except FileNotFoundError as exc:
        raise MemberLockBuildError(f"{name} is missing") from exc
    if requested.is_symlink() or not requested.is_file():
        raise MemberLockBuildError(f"{name} must be a regular non-symlink file")
    payload = requested.read_bytes()
    after = requested.lstat()
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(payload) != after.st_size
    ):
        raise MemberLockBuildError(f"{name} changed during snapshot")
    return payload


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _safe_path(value: Any) -> str:
    text = str(value)
    pure = PurePosixPath(text)
    if (
        not text
        or pure.is_absolute()
        or "\\" in text
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.suffix != ".py"
        or str(pure) != text
    ):
        raise MemberLockBuildError("member path must be normalized relative .py")
    return text


def _run_git(
    root: Path,
    args: Sequence[str],
    *,
    check: bool = True,
    timeout: int = 60,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=check,
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MemberLockBuildError("member-lock Git operation failed") from exc


def _clean_head(repo_root: str | Path) -> tuple[Path, str, str]:
    root = Path(repo_root).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise MemberLockBuildError("repository root must be a real directory")
    top = Path(
        _run_git(root, ["rev-parse", "--show-toplevel"]).stdout.decode().strip()
    ).resolve(strict=True)
    if os.path.normcase(str(top)) != os.path.normcase(str(root)):
        raise MemberLockBuildError("repo-root is not the Git top level")
    head = _run_git(root, ["rev-parse", "HEAD"]).stdout.decode("ascii").strip()
    if len(head) != 40 or any(ch not in "0123456789abcdef" for ch in head):
        raise MemberLockBuildError("HEAD is not a lowercase 40-hex commit")
    status = _run_git(root, ["status", "--porcelain=v1", "-z"]).stdout
    if status:
        raise MemberLockBuildError(
            "member-lock build requires a clean tree including no untracked files"
        )
    return root, head, _sha256(status)


def _tracked_blob(root: Path, commit: str, path: str) -> bytes:
    member = _safe_path(path) if path.endswith(".py") else str(PurePosixPath(path))
    row = _run_git(root, ["ls-tree", "-z", commit, "--", member]).stdout
    rows = [item for item in row.split(b"\0") if item]
    if len(rows) != 1:
        raise MemberLockBuildError(f"path is not one tracked blob: {member}")
    try:
        descriptor, encoded = rows[0].split(b"\t", 1)
        mode, kind, object_id = descriptor.split(b" ", 2)
    except ValueError as exc:
        raise MemberLockBuildError("invalid Git tree descriptor") from exc
    if (
        encoded.decode("utf-8") != member
        or kind != b"blob"
        or mode not in {b"100644", b"100755"}
    ):
        raise MemberLockBuildError(f"path is missing, symlink, or non-file: {member}")
    return _run_git(root, ["cat-file", "blob", object_id.decode("ascii")]).stdout


def _tracked_python_blobs(root: Path, commit: str) -> dict[str, bytes]:
    tree = _run_git(root, ["ls-tree", "-r", "-z", "--full-tree", commit]).stdout
    objects: list[tuple[str, str]] = []
    for raw in (row for row in tree.split(b"\0") if row):
        try:
            descriptor, encoded = raw.split(b"\t", 1)
            mode, kind, object_id = descriptor.split(b" ", 2)
            path = encoded.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise MemberLockBuildError("invalid recursive Git tree descriptor") from exc
        if not path.endswith(".py"):
            continue
        path = _safe_path(path)
        if kind != b"blob" or mode not in {b"100644", b"100755"}:
            raise MemberLockBuildError(f"tracked Python path is not a regular blob: {path}")
        objects.append((path, object_id.decode("ascii")))
    if not objects:
        raise MemberLockBuildError("source commit contains no tracked Python blobs")
    request = b"".join(object_id.encode("ascii") + b"\n" for _, object_id in objects)
    # _run_git has no stdin surface by design; run this one bounded batch directly.
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "cat-file", "--batch"],
            input=request,
            check=True,
            capture_output=True,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MemberLockBuildError("Git batch blob snapshot failed") from exc
    payload = bytes(result.stdout)
    offset = 0
    blobs: dict[str, bytes] = {}
    for path, object_id in objects:
        newline = payload.find(b"\n", offset)
        if newline < 0:
            raise MemberLockBuildError("Git batch blob header is truncated")
        header = payload[offset:newline].split(b" ")
        if len(header) != 3 or header[0].decode("ascii") != object_id or header[1] != b"blob":
            raise MemberLockBuildError("Git batch blob header drift")
        try:
            size = int(header[2])
        except ValueError as exc:
            raise MemberLockBuildError("Git batch blob size is invalid") from exc
        start = newline + 1
        end = start + size
        if end >= len(payload) or payload[end : end + 1] != b"\n":
            raise MemberLockBuildError("Git batch blob body is truncated")
        blobs[path] = payload[start:end]
        offset = end + 1
    if offset != len(payload):
        raise MemberLockBuildError("Git batch blob output has trailing bytes")
    return blobs


def _module_name(path: str) -> tuple[str, tuple[str, ...]]:
    pure = PurePosixPath(path)
    if pure.parts[0] == "code":
        parts = list(pure.parts[1:])
    else:
        parts = list(pure.parts)
    parts[-1] = PurePosixPath(parts[-1]).stem
    if parts[-1] == "__init__":
        parts.pop()
        package = tuple(parts)
    else:
        package = tuple(parts[:-1])
    return ".".join(parts), package


def _module_candidates(name: str) -> tuple[str, ...]:
    if not name:
        return ()
    stem = name.replace(".", "/")
    candidates = []
    for prefix in ("code", ""):
        base = f"{prefix}/{stem}" if prefix else stem
        candidates.extend((f"{base}.py", f"{base}/__init__.py"))
    return tuple(candidates)


def _imports_from_blob(path: str, payload: bytes) -> set[str]:
    try:
        tree = ast.parse(payload.decode("utf-8-sig"), filename=path)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise MemberLockBuildError(f"cannot parse tracked Python source: {path}") from exc
    _, package = _module_name(path)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package) - (node.level - 1)
                if keep < 0:
                    raise MemberLockBuildError(f"invalid relative import in {path}")
                base_parts = list(package[:keep])
            else:
                base_parts = []
            if node.module:
                base_parts.extend(node.module.split("."))
                names.add(".".join(base_parts))
            for alias in node.names:
                if alias.name != "*":
                    names.add(".".join([*base_parts, alias.name]))
    return names


def _static_closure(
    root: Path,
    commit: str,
    entries: Sequence[str],
    *,
    tracked_blobs: Mapping[str, bytes] | None = None,
) -> list[str]:
    blobs = dict(tracked_blobs) if tracked_blobs is not None else _tracked_python_blobs(root, commit)
    pending = [_safe_path(path) for path in entries]
    seen: set[str] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        try:
            payload = blobs[path]
        except KeyError as exc:
            raise MemberLockBuildError(f"path is not one tracked blob: {path}") from exc
        seen.add(path)
        for module in sorted(_imports_from_blob(path, payload)):
            for candidate in _module_candidates(module):
                if candidate not in blobs:
                    continue
                if candidate not in seen:
                    pending.append(candidate)
                break
    return sorted(seen)


_PROBE_PROGRAM = r"""
import gc, importlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve(strict=True)
output = pathlib.Path(sys.argv[2])
entries = json.loads(sys.argv[3])
checkpoint = pathlib.Path(sys.argv[4]).read_bytes()
for value in (str(root), str(root / 'code')):
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)
diagnostic_path = 'code/scripts/diagnose_adv3b02_runtime_numerics.py'
if diagnostic_path not in entries:
    raise RuntimeError('production consumer entry is not in the reviewed static roots')
diagnostic = importlib.import_module('scripts.diagnose_adv3b02_runtime_numerics')
diagnostic._load_worker_dependencies()
eager = diagnostic._build_eager(checkpoint, device=diagnostic.torch.device('cpu'))
del eager
gc.collect()
members = set()
for module in list(sys.modules.values()):
    source = getattr(module, '__file__', None)
    if not source:
        continue
    try:
        path = pathlib.Path(source).resolve(strict=True)
        relative = path.relative_to(root).as_posix()
    except (OSError, ValueError):
        continue
    if relative.endswith('.py'):
        members.add(relative)
output.write_text(json.dumps(sorted(members), separators=(',', ':')), encoding='utf-8')
"""


def _runtime_import_probe(
    root: Path, entries: Sequence[str], checkpoint_bytes: bytes
) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="adv3b02_member_probe_") as directory:
        output = Path(directory) / "members.json"
        checkpoint = Path(directory) / "checkpoint.pth"
        with checkpoint.open("xb") as handle:
            handle.write(checkpoint_bytes)
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    _PROBE_PROGRAM,
                    str(root),
                    str(output),
                    json.dumps(list(entries), separators=(",", ":")),
                    str(checkpoint),
                ],
                cwd=root,
                env=env,
                check=True,
                capture_output=True,
                timeout=180,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise MemberLockBuildError("real import probe failed") from exc
        if completed.stderr:
            # Warnings are evidence, not members; retain only their digest below.
            pass
        try:
            raw = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MemberLockBuildError("import probe did not emit valid members") from exc
    if not isinstance(raw, list) or not raw:
        raise MemberLockBuildError("import probe member list is empty")
    return sorted({_safe_path(item) for item in raw})


def _analyze(
    root: Path,
    commit: str,
    *,
    checkpoint_bytes: bytes,
    entries: Sequence[str] | None = None,
    probe: Callable[[Path, Sequence[str], bytes], list[str]] | None = None,
) -> dict[str, Any]:
    if entries is None:
        entries = ENTRY_PATHS
    if probe is None:
        probe = _runtime_import_probe
    normalized_entries = tuple(sorted({_safe_path(path) for path in entries}))
    tracked_blobs = _tracked_python_blobs(root, commit)
    static_members = _static_closure(
        root, commit, normalized_entries, tracked_blobs=tracked_blobs
    )
    runtime_members = probe(root, normalized_entries, checkpoint_bytes)
    for path in runtime_members:
        if path not in tracked_blobs:
            raise MemberLockBuildError(f"path is not one tracked blob: {path}")
    if set(normalized_entries) == set(PRODUCTION_ENTRY_PATHS):
        required = {
            "code/scripts/diagnose_adv3b02_runtime_numerics.py",
            "code/cvsrffi/somph_runtime_trust.py",
        }
        if not required.issubset(runtime_members):
            raise MemberLockBuildError(
                "real worker import probe lacks consumer or fixed trust helper"
            )
    # The consumer requires an exact match with actually loaded project modules.
    # Static reachability remains review evidence; adding static-only paths would
    # make the signed archive fail the consumer's exact-member gate.
    members = list(runtime_members)
    static_rows = _member_rows(static_members, tracked_blobs)
    runtime_rows = _member_rows(runtime_members, tracked_blobs)
    static_root = _member_row_root("static_members", static_rows)
    runtime_root = _member_row_root("runtime_members", runtime_rows)
    closure_root = _sha256(
        _canonical_json(
            {
                "static_member_root_sha256": static_root,
                "runtime_member_root_sha256": runtime_root,
            }
        )
    )
    lock = {"schema": LOCK_SCHEMA, "members": members}
    lock_bytes = _canonical_json(lock)
    return {
        "lock": lock,
        "lock_bytes": lock_bytes,
        "lock_sha256": _sha256(lock_bytes),
        "entry_paths": list(normalized_entries),
        "static_members": static_members,
        "runtime_members": runtime_members,
        "static_member_rows": static_rows,
        "runtime_member_rows": runtime_rows,
        "static_member_root_sha256": static_root,
        "runtime_member_root_sha256": runtime_root,
        "closure_root_sha256": closure_root,
        "static_only_members": sorted(set(static_members) - set(runtime_members)),
        "runtime_only_members": sorted(set(runtime_members) - set(static_members)),
        "members": members,
    }


def _member_rows(
    paths: Sequence[str], tracked_blobs: Mapping[str, bytes]
) -> list[dict[str, Any]]:
    return [
        {
            "path": path,
            "bytes": len(tracked_blobs[path]),
            "sha256": _sha256(tracked_blobs[path]),
        }
        for path in sorted(paths)
    ]


def _member_row_root(name: str, rows: Sequence[Mapping[str, Any]]) -> str:
    return _sha256(_canonical_json({name: list(rows)}))


def _outside_repo(path: Path, root: Path) -> None:
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError:
        return
    raise MemberLockBuildError("candidate/evidence outputs must be outside repository")


def _write_new(path: Path, value: bytes) -> None:
    path = path.resolve(strict=False)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.resolve(strict=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def propose_member_lock(
    *,
    repo_root: str | Path,
    checkpoint: str | Path,
    candidate_out: str | Path,
    evidence_out: str | Path,
) -> dict[str, Any]:
    root, head, status_root = _clean_head(repo_root)
    candidate = Path(candidate_out).resolve(strict=False)
    evidence_path = Path(evidence_out).resolve(strict=False)
    if candidate == evidence_path:
        raise MemberLockBuildError("candidate and evidence outputs must differ")
    _outside_repo(candidate, root)
    _outside_repo(evidence_path, root)
    checkpoint_bytes = _read_stable_regular(checkpoint, "ADV3B02 checkpoint")
    checkpoint_sha = _sha256(checkpoint_bytes)
    if checkpoint_sha != BASE_CHECKPOINT_SHA256:
        raise MemberLockBuildError("checkpoint is not strict ADV3B02")
    analysis = _analyze(root, head, checkpoint_bytes=checkpoint_bytes)
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "mode": "propose",
        "status": "PROPOSED_REQUIRES_HUMAN_REVIEW_AND_GIT_COMMIT",
        "formal_authority": False,
        "source_git_commit": head,
        "clean_status_root_sha256": status_root,
        "checkpoint_sha256": checkpoint_sha,
        "entry_paths": analysis["entry_paths"],
        "static_members": analysis["static_members"],
        "runtime_import_members": analysis["runtime_members"],
        "static_only_members": analysis["static_only_members"],
        "runtime_only_members": analysis["runtime_only_members"],
        "locked_runtime_members": analysis["members"],
        "static_member_rows": analysis["static_member_rows"],
        "runtime_member_rows": analysis["runtime_member_rows"],
        "static_member_root_sha256": analysis["static_member_root_sha256"],
        "runtime_member_root_sha256": analysis["runtime_member_root_sha256"],
        "closure_root_sha256": analysis["closure_root_sha256"],
        "member_lock_sha256": analysis["lock_sha256"],
        "human_review_required": True,
        "git_tracking_required": True,
        "offline_signature_emitted": False,
    }
    evidence_bytes = _canonical_json(evidence)
    _write_new(candidate, analysis["lock_bytes"])
    _write_new(evidence_path, evidence_bytes)
    return {
        "status": evidence["status"],
        "candidate": str(candidate),
        "candidate_sha256": analysis["lock_sha256"],
        "evidence": str(evidence_path),
        "evidence_sha256": _sha256(evidence_bytes),
        "member_count": len(analysis["members"]),
        "formal_authority": False,
    }


def _repo_relative(root: Path, path: str | Path, name: str) -> str:
    requested = Path(path)
    if requested.is_symlink():
        raise MemberLockBuildError(f"{name} must not be a symlink")
    resolved = requested.resolve(strict=True)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise MemberLockBuildError(f"{name} must be inside repository") from exc
    if not relative or any(part in {"", ".", ".."} for part in PurePosixPath(relative).parts):
        raise MemberLockBuildError(f"{name} path is unsafe")
    return relative


def _review_authority_status(
    *, envelope_path: str | Path | None, review_bytes: bytes
) -> dict[str, Any]:
    trusted_sha = TRUSTED_REVIEW_AUTHORITY_ENVELOPE_SHA256
    if trusted_sha is None:
        return {
            "status": BLOCKED_REVIEW_STATUS,
            "verified": False,
            "reason": "trusted independent-review authority envelope SHA is not provisioned",
            "envelope_path": str(Path(envelope_path).resolve(strict=False))
            if envelope_path is not None
            else None,
            "envelope_sha256": None,
        }
    if envelope_path is None:
        return {
            "status": BLOCKED_REVIEW_STATUS,
            "verified": False,
            "reason": "independent-review authority envelope is missing",
            "envelope_path": None,
            "envelope_sha256": None,
        }
    envelope_bytes = _read_stable_regular(
        envelope_path, "independent-review authority envelope"
    )
    envelope_sha = _sha256(envelope_bytes)
    if envelope_sha != trusted_sha:
        raise MemberLockBuildError("independent-review authority envelope SHA drift")
    try:
        envelope = json.loads(envelope_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MemberLockBuildError("review authority envelope is not UTF-8 JSON") from exc
    expected_keys = {
        "schema",
        "issuer",
        "key_id",
        "public_key_sha256",
        "reviewer_id",
        "review_record_sha256",
        "signature_hex",
    }
    if not isinstance(envelope, dict) or set(envelope) != expected_keys:
        raise MemberLockBuildError("review authority envelope exact schema drift")
    if (
        envelope.get("schema") != REVIEW_AUTHORITY_SCHEMA
        or envelope.get("issuer") != REVIEW_AUTHORITY_ISSUER
        or envelope.get("key_id") != REVIEW_AUTHORITY_KEY_ID
        or envelope.get("public_key_sha256")
        != REVIEW_AUTHORITY_PUBLIC_KEY_SHA256
        or envelope.get("reviewer_id") != REVIEWER_IDENTITY
        or envelope.get("review_record_sha256") != _sha256(review_bytes)
    ):
        raise MemberLockBuildError("review authority envelope binding drift")
    try:
        signature = bytes.fromhex(str(envelope.get("signature_hex", "")))
    except ValueError as exc:
        raise MemberLockBuildError("review authority signature is not hex") from exc
    if len(signature) != 64:
        raise MemberLockBuildError("review authority signature length drift")
    body = {key: value for key, value in envelope.items() if key != "signature_hex"}
    code_root = Path(__file__).resolve().parents[1]
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from cvsrffi import somph_runtime_trust  # local fixed verifier, no signer

    if (
        somph_runtime_trust.PINNED_AUTHORITY_ISSUER != REVIEW_AUTHORITY_ISSUER
        or somph_runtime_trust.PINNED_AUTHORITY_KEY_ID != REVIEW_AUTHORITY_KEY_ID
        or somph_runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_HEX
        != REVIEW_AUTHORITY_PUBLIC_KEY_HEX
        or somph_runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_SHA256
        != REVIEW_AUTHORITY_PUBLIC_KEY_SHA256
    ):
        raise MemberLockBuildError("review authority trust helper identity drift")
    try:
        somph_runtime_trust.verify_ed25519(
            bytes.fromhex(REVIEW_AUTHORITY_PUBLIC_KEY_HEX),
            _canonical_json(body),
            signature,
        )
    except (TypeError, ValueError) as exc:
        raise MemberLockBuildError("review authority signature is invalid") from exc
    return {
        "status": "INDEPENDENT_REVIEW_AUTHORITY_VERIFIED",
        "verified": True,
        "reason": None,
        "envelope_path": str(Path(envelope_path).resolve(strict=True)),
        "envelope_sha256": envelope_sha,
    }


def verify_tracked_member_lock(
    *,
    repo_root: str | Path,
    checkpoint: str | Path,
    member_lock: str | Path,
    human_review: str | Path,
    review_authority_envelope: str | Path | None,
    evidence_out: str | Path,
) -> dict[str, Any]:
    root, head, status_root = _clean_head(repo_root)
    evidence_path = Path(evidence_out).resolve(strict=False)
    _outside_repo(evidence_path, root)
    lock_relative = _repo_relative(root, member_lock, "member lock")
    review_relative = _repo_relative(root, human_review, "human review record")
    lock_bytes = _tracked_blob(root, head, lock_relative)
    review_bytes = _tracked_blob(root, head, review_relative)
    try:
        lock = json.loads(lock_bytes.decode("utf-8"))
        review = json.loads(review_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MemberLockBuildError("tracked lock/review is not UTF-8 JSON") from exc
    if not isinstance(lock, dict) or set(lock) != {"schema", "members"}:
        raise MemberLockBuildError("tracked member lock exact schema drift")
    if lock.get("schema") != LOCK_SCHEMA:
        raise MemberLockBuildError("tracked member lock schema drift")
    if not isinstance(review, dict) or set(review) != {
        "schema",
        "decision",
        "scope",
        "reviewer_id",
        "reviewed_source_commit",
        "member_lock_sha256",
        "checkpoint_sha256",
        "static_member_rows",
        "runtime_member_rows",
        "static_member_root_sha256",
        "runtime_member_root_sha256",
        "closure_root_sha256",
    }:
        raise MemberLockBuildError("tracked human review exact schema drift")
    lock_sha = _sha256(lock_bytes)
    if (
        review.get("schema") != REVIEW_SCHEMA
        or review.get("decision") != REVIEW_DECISION
        or review.get("scope") != REVIEW_SCOPE
        or review.get("reviewer_id") != REVIEWER_IDENTITY
        or review.get("member_lock_sha256") != lock_sha
    ):
        raise MemberLockBuildError("tracked human review does not approve this lock")
    reviewed = str(review.get("reviewed_source_commit", ""))
    if len(reviewed) != 40 or any(ch not in "0123456789abcdef" for ch in reviewed):
        raise MemberLockBuildError("reviewed source commit is invalid")
    ancestor = _run_git(root, ["merge-base", "--is-ancestor", reviewed, head], check=False)
    if ancestor.returncode != 0 or reviewed == head:
        raise MemberLockBuildError("reviewed source commit must be a strict ancestor of HEAD")
    if _tracked_blob(root, reviewed, lock_relative) != lock_bytes:
        raise MemberLockBuildError("member lock changed after the reviewed source commit")
    checkpoint_bytes = _read_stable_regular(checkpoint, "ADV3B02 checkpoint")
    checkpoint_sha = _sha256(checkpoint_bytes)
    if checkpoint_sha != BASE_CHECKPOINT_SHA256:
        raise MemberLockBuildError("checkpoint is not strict ADV3B02")
    analysis = _analyze(root, head, checkpoint_bytes=checkpoint_bytes)
    if lock_bytes != analysis["lock_bytes"]:
        raise MemberLockBuildError(
            "tracked reviewed lock does not equal recomputed static/import closure"
        )
    expected_review_closure = {
        "checkpoint_sha256": checkpoint_sha,
        "static_member_rows": analysis["static_member_rows"],
        "runtime_member_rows": analysis["runtime_member_rows"],
        "static_member_root_sha256": analysis["static_member_root_sha256"],
        "runtime_member_root_sha256": analysis["runtime_member_root_sha256"],
        "closure_root_sha256": analysis["closure_root_sha256"],
    }
    observed_review_closure = {
        key: review.get(key) for key in expected_review_closure
    }
    if observed_review_closure != expected_review_closure:
        raise MemberLockBuildError(
            "tracked review rows/closure roots do not equal current recomputation"
        )
    reviewed_python = _tracked_python_blobs(root, reviewed)
    current_python = _tracked_python_blobs(root, head)
    for path in sorted(set(analysis["static_members"]) | set(analysis["members"])):
        if reviewed_python.get(path) != current_python.get(path):
            raise MemberLockBuildError(
                f"reviewed execution/static source changed after human review: {path}"
            )
    authority = _review_authority_status(
        envelope_path=review_authority_envelope, review_bytes=review_bytes
    )
    status = (
        "VERIFIED_TRACKED_MEMBER_LOCK_READY_FOR_OFFLINE_SIGNER"
        if authority["verified"] is True
        else BLOCKED_REVIEW_STATUS
    )
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "mode": "verify",
        "status": status,
        "formal_authority": False,
        "source_git_commit": head,
        "clean_status_root_sha256": status_root,
        "member_lock_path": lock_relative,
        "member_lock_sha256": lock_sha,
        "human_review_path": review_relative,
        "human_review_sha256": _sha256(review_bytes),
        "reviewed_source_commit": reviewed,
        "reviewer_id": review["reviewer_id"],
        "independent_review_authority": authority,
        "checkpoint_sha256": checkpoint_sha,
        "entry_paths": analysis["entry_paths"],
        "static_members": analysis["static_members"],
        "runtime_import_members": analysis["runtime_members"],
        "static_only_members": analysis["static_only_members"],
        "runtime_only_members": analysis["runtime_only_members"],
        "locked_runtime_members": analysis["members"],
        "static_member_rows": analysis["static_member_rows"],
        "runtime_member_rows": analysis["runtime_member_rows"],
        "static_member_root_sha256": analysis["static_member_root_sha256"],
        "runtime_member_root_sha256": analysis["runtime_member_root_sha256"],
        "closure_root_sha256": analysis["closure_root_sha256"],
        "member_count": len(analysis["members"]),
        "offline_signature_emitted": False,
    }
    evidence_bytes = _canonical_json(evidence)
    _write_new(evidence_path, evidence_bytes)
    return {
        "status": evidence["status"],
        "source_git_commit": head,
        "member_lock_sha256": lock_sha,
        "evidence": str(evidence_path),
        "evidence_sha256": _sha256(evidence_bytes),
        "member_count": len(analysis["members"]),
        "formal_authority": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    propose = sub.add_parser("propose")
    propose.add_argument("--repo-root", type=Path, required=True)
    propose.add_argument("--checkpoint", type=Path, required=True)
    propose.add_argument("--candidate-out", type=Path, required=True)
    propose.add_argument("--evidence-out", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--repo-root", type=Path, required=True)
    verify.add_argument("--checkpoint", type=Path, required=True)
    verify.add_argument("--member-lock", type=Path, required=True)
    verify.add_argument("--human-review", type=Path, required=True)
    verify.add_argument("--review-authority-envelope", type=Path)
    verify.add_argument("--evidence-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mode == "propose":
        result = propose_member_lock(
            repo_root=args.repo_root,
            checkpoint=args.checkpoint,
            candidate_out=args.candidate_out,
            evidence_out=args.evidence_out,
        )
    else:
        result = verify_tracked_member_lock(
            repo_root=args.repo_root,
            checkpoint=args.checkpoint,
            member_lock=args.member_lock,
            human_review=args.human_review,
            review_authority_envelope=args.review_authority_envelope,
            evidence_out=args.evidence_out,
        )
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 2 if result["status"] == BLOCKED_REVIEW_STATUS else 0


if __name__ == "__main__":
    raise SystemExit(main())
