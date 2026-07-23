#!/usr/bin/env python3
"""Build an immutable r1f Phase-1 ground bundle; target/query inputs are absent."""
from __future__ import annotations
import argparse, io, json, os, stat, sys
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path: sys.path.insert(0, str(CODE_ROOT))
import numpy as np
from cvsrffi.stage2_dssc_zdom_jg_qknn_r4_bcrr import (
    PHASE1_ARCHIVE_MANIFEST_SHA256,
    PHASE1_ARCHIVE_SHA256,
    PHASE1_CHECKPOINT_SHA256,
    PHASE1_PARITY_RECEIPT_SHA256,
    build_ground_bundle_arrays,
    canonical_method_lock,
    sha256_file,
    validate_method_lock,
)

LOCK = canonical_method_lock()
def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="mode",required=True); w=s.add_parser("write-lock"); w.add_argument("--output",required=True); b=s.add_parser("build"); b.add_argument("--archive",required=True); b.add_argument("--archive-manifest",required=True); b.add_argument("--phase1-checkpoint",required=True); b.add_argument("--method-lock",required=True); b.add_argument("--output",required=True); return p
def write_lock(path: str|Path) -> dict[str,object]:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); raw=json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode("utf-8"); fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_BINARY",0),0o444)
    try: os.write(fd,raw); os.fsync(fd)
    finally: os.close(fd)
    return {"output":str(p),"sha256":__import__("hashlib").sha256(raw).hexdigest(),"status":"LOCKED"}

def build(args: argparse.Namespace) -> dict[str, object]:
    archive, manifest, checkpoint, lock, output = map(Path,(args.archive,args.archive_manifest,args.phase1_checkpoint,args.method_lock,args.output))
    if output.exists(): raise FileExistsError("DSSC bundle output must be a new path")
    archive_sha = sha256_file(archive)
    manifest_sha = sha256_file(manifest)
    checkpoint_sha = sha256_file(checkpoint)
    if archive_sha != PHASE1_ARCHIVE_SHA256 or manifest_sha != PHASE1_ARCHIVE_MANIFEST_SHA256:
        raise ValueError("DSSC bundle requires the frozen GEOFF/r8 archive and manifest")
    if checkpoint_sha != PHASE1_CHECKPOINT_SHA256:
        raise ValueError("DSSC bundle requires the frozen Phase1 checkpoint")
    document=json.loads(manifest.read_text(encoding="utf-8"))
    access = document.get("access_audit", {})
    if (
        document.get("schema") != "cvs.phase1.singleobs_dual_feature_archive.v2"
        or any(access.get(key) is not False for key in ("target_access", "query_access", "clean_iq_access"))
        or document.get("artifact", {}).get("sha256") != PHASE1_ARCHIVE_SHA256
        or document.get("inputs", {}).get("checkpoint_sha256") != PHASE1_CHECKPOINT_SHA256
        or document.get("inputs", {}).get("parity_receipt_sha256") != PHASE1_PARITY_RECEIPT_SHA256
    ):
        raise ValueError("archive is not the allowed Phase-1-only dual-feature archive")
    lock_doc, _lock_sha = validate_method_lock(
        json.loads(lock.read_text(encoding="utf-8"))
    )
    with np.load(archive,allow_pickle=False) as a:
        arrays=build_ground_bundle_arrays(z_id=a["z_id"],z_dom=a["z_dom"],labels=a["labels"],physical_ids=a["physical_ids"],
          archive_sha256=archive_sha,archive_manifest_sha256=manifest_sha,checkpoint_sha256=checkpoint_sha,method_lock=lock_doc)
    output.parent.mkdir(parents=True,exist_ok=True)
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    payload = stream.getvalue()
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_BINARY",0),0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(output,stat.S_IREAD)
    return {"output":str(output),"sha256":sha256_file(output),"prototype_count":int(len(arrays["prototype_class_indices"])),"basis_rank":int(arrays["u_id_codes"].shape[0]),"query_rows_used":0,"state_dtype":"int8_with_fp16_row_scale"}

def main() -> int:
    a=parser().parse_args(); print(json.dumps(write_lock(a.output) if a.mode=="write-lock" else build(a),sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
