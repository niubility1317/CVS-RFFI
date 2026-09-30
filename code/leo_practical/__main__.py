"""Explicit raw .npy conversion. Does not launch training or read class labels."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .batch import apply_leo_practical_channel_batch
from .channel import VERSION, Config


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="complex (N,T) or real (N,2,T) .npy")
    p.add_argument("--output-dir", type=Path, required=True, help="new directory; existing paths rejected")
    p.add_argument("--config", type=Path, required=True, help="JSON Config fields; fs_hz must be explicit")
    p.add_argument("--fs-hz", type=float, help="explicit sampling rate; overrides template null")
    p.add_argument("--receiver-profile", type=Path, help="optional ground/satellite compensation profile JSON")
    p.add_argument("--route", choices=["full", "residual"], help="override processing route")
    eq = p.add_mutually_exclusive_group()
    eq.add_argument("--equalization", dest="equalization", action="store_true")
    eq.add_argument("--no-equalization", dest="equalization", action="store_false")
    p.set_defaults(equalization=None)
    p.add_argument("--equalizer-method", choices=["mmse", "zf"])
    p.add_argument("--fs-source", required=True, help="sampling rate provenance, or explicitly declared assumption")
    p.add_argument("--fc-source", required=True, help="carrier-frequency provenance, or explicitly declared assumption")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--receiver-seed", type=int, default=0)
    p.add_argument("--namespace", required=True, help="e.g. source_train_v1 or target_eval_v1")
    p.add_argument("--ids", type=Path, required=True, help="JSON list of physical record IDs, no class labels")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--session-id", help="one virtual RX/session for this file")
    group.add_argument("--sessions", type=Path, help="JSON list of RX/session IDs")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--keep-reference-amplitude", action="store_true", help="do not normalize each input record")
    args = p.parse_args(argv)
    if args.batch_size <= 0 or not args.namespace.strip() or not args.fs_source.strip() or not args.fc_source.strip():
        p.error("batch size and provenance/namespace must be nonempty/positive")
    fields = json.loads(args.config.read_text(encoding="utf-8"))
    if args.receiver_profile:
        fields.update(json.loads(args.receiver_profile.read_text(encoding="utf-8")))
    if args.route is not None:
        fields["processing_route"] = args.route
    if args.equalization is not None:
        fields["equalization_enabled"] = args.equalization
    if args.equalizer_method:
        fields["equalizer_method"] = args.equalizer_method
    if args.fs_hz is not None:
        fields["fs_hz"] = args.fs_hz
    if fields.get("fs_hz") is None:
        p.error("set --fs-hz or a verified fs_hz in config; no implicit sampling rate")
    cfg = Config(**fields)
    x = np.load(args.input, mmap_mode="r", allow_pickle=False)
    valid_complex = x.ndim == 2 and np.iscomplexobj(x)
    valid_real = x.ndim == 3 and x.shape[1] == 2 and np.issubdtype(x.dtype, np.floating)
    if not (valid_complex or valid_real) or len(x) == 0 or x.shape[-1] < 4:
        raise ValueError("expected nonempty complex (N,T) or floating (N,2,T) raw IQ")
    ids = json.loads(args.ids.read_text(encoding="utf-8"))
    if not isinstance(ids, list) or len(ids) != len(x) or len(set(map(str, ids))) != len(x):
        raise ValueError("IDs must be a unique list matching all records")
    sessions = (json.loads(args.sessions.read_text(encoding="utf-8")) if args.sessions
                else [args.session_id]*len(x))
    if not isinstance(sessions, list) or len(sessions) != len(x):
        raise ValueError("sessions must match record count")
    # Fail before writing if boundary context would be required but unavailable.
    if cfg.boundary == "require_context":
        raise ValueError("this snapshot CLI has no preceding context; use ChannelStream API")
    input_hash = sha256(args.input)
    source_hashes = {v.name: sha256(v) for v in Path(__file__).parent.glob("*.py")}
    manifest = dict(version=VERSION, status="WRITING", config=asdict(cfg), config_hash=cfg.config_hash,
        fs_source=args.fs_source, fc_source=args.fc_source, input=str(args.input.resolve()),
        input_sha256=input_hash, ids_sha256=sha256(args.ids), sessions_sha256=sha256(args.sessions) if args.sessions else None,
        source_sha256=source_hashes, seed=args.seed, receiver_seed=args.receiver_seed,
        namespace=args.namespace, normalize_input_rms=not args.keep_reference_amplitude,
        shape=list(x.shape), rows_completed=0, raw_iq_only=True,
        physical_calibration="engineering_proxy_not_in_orbit_validation")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    def save_manifest():
        tmp = args.output_dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        tmp.replace(args.output_dir / "manifest.json")
    save_manifest()
    output = np.lib.format.open_memmap(args.output_dir / "iq.partial.npy", mode="w+",
        dtype=np.complex64 if valid_complex else np.float32, shape=x.shape)
    try:
        with (args.output_dir / "metadata.partial.jsonl").open("w", encoding="utf-8") as meta_file:
            for start in range(0, len(x), args.batch_size):
                stop = min(len(x), start + args.batch_size)
                y, metadata, _ = apply_leo_practical_channel_batch(x[start:stop], cfg,
                    seed=args.seed, sample_ids=ids[start:stop], session_ids=sessions[start:stop],
                    realization_namespace=args.namespace, receiver_seed=args.receiver_seed,
                    normalize_input_rms=not args.keep_reference_amplitude)
                output[start:stop] = y
                for row in metadata:
                    meta_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
                output.flush()
                meta_file.flush()
                manifest["rows_completed"] = stop
                save_manifest()
        del output
        (args.output_dir / "iq.partial.npy").rename(args.output_dir / "iq.npy")
        (args.output_dir / "metadata.partial.jsonl").rename(args.output_dir / "metadata.jsonl")
        manifest.update(status="COMPLETE", output_sha256=sha256(args.output_dir / "iq.npy"),
                        metadata_sha256=sha256(args.output_dir / "metadata.jsonl"))
        save_manifest()
    except BaseException as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        save_manifest()
        raise
    print(str(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
