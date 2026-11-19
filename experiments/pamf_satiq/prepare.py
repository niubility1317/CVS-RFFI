"""Build station-disjoint SatIQ inputs from the dorm computer's NumPy export."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path, PureWindowsPath

import numpy as np


FS = 25_000_000
STARTS = (4500, 5500, 6500, 7500)
WIDTH = 256
SOURCE_SEED = 2026111901
BLOCK_NS = 15_000_000_000
GUARD_NS = 500_000_000
MIN_CONFIDENCE = 90
MAX_CLASSES = 20
MIN_TRAIN = 30
MIN_VAL = 8
SCHEMA = "satiq_oxford_thun_v1"


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def inventory(root, station):
    manifest = json.loads((root / "metadata.json").read_text(encoding="utf-8"))["manifest"]
    shards = sorted(
        (PureWindowsPath(item["source"]).name, item)
        for item in manifest if item["country"] == station
    )
    if len(shards) != 6 or len({name for name, _ in shards}) != 6:
        raise ValueError(f"{station}: expected six distinct dorm export shards")
    result = []
    for name, item in shards:
        if item["records"] != 5000:
            raise ValueError(f"{station}/{name}: expected 5000 records")
        arrays = {
            key: np.load(root / Path(*PureWindowsPath(item["arrays"][key]).parts),
                         mmap_mode="r", allow_pickle=False)
            for key in ("samples", "satellite_id", "cell_id", "confidence", "timestamp")
        }
        if (arrays["samples"].shape != (5000, 11000, 2)
                or arrays["samples"].dtype != np.float32
                or any(v.shape != (5000,) or v.dtype.kind not in "iu"
                       for k, v in arrays.items() if k != "samples")):
            raise ValueError(f"{station}/{name}: invalid record layout")
        sat, beam = arrays["satellite_id"], arrays["cell_id"]
        if (np.any((sat < 0) | (sat > 127)) or np.any((beam < 0) | (beam > 63))
                or np.any((arrays["confidence"] < 0) | (arrays["confidence"] > 100))
                or np.any(arrays["timestamp"] < 10**18)):
            raise ValueError(f"{station}/{name}: invalid tuple, confidence or timestamp")
        result.append((name, item["split"], arrays))
    return result


def source_role(timestamp):
    block, position = divmod(int(timestamp), BLOCK_NS)
    if position < GUARD_NS or position >= BLOCK_NS - GUARD_NS:
        return "guard"
    digest = sha256(f"{SOURCE_SEED}:{block}".encode("ascii")).digest()
    return "val" if int.from_bytes(digest[:8], "big") % 5 == 0 else "train"


def source_selection(shards):
    records = {"train": [], "val": []}
    counts = {role: Counter() for role in records}
    for name, _, arrays in shards:
        for i in np.flatnonzero(arrays["confidence"] >= MIN_CONFIDENCE):
            role = source_role(arrays["timestamp"][i])
            if role in records:
                pair = int(arrays["satellite_id"][i]), int(arrays["cell_id"][i])
                records[role].append((name, int(i), pair))
                counts[role][pair] += 1
    eligible = sorted(
        (pair for pair, n in counts["train"].items()
         if n >= MIN_TRAIN and counts["val"][pair] >= MIN_VAL),
        key=lambda pair: (-counts["train"][pair], pair),
    )
    pairs = eligible[:MAX_CLASSES]
    if len(pairs) < 2:
        raise ValueError("UK-only class screening produced fewer than two classes")
    mapping = {pair: y for y, pair in enumerate(pairs)}
    selected = {role: [(name, i, mapping[pair]) for name, i, pair in entries if pair in mapping]
                for role, entries in records.items()}
    contract = {
        "schema": SCHEMA, "source_station": "Oxford, UK", "target_station": "Thun, Switzerland",
        "source_physical_records": 30_000, "target_physical_records": 30_000,
        "sample_rate_hz": FS, "header_samples": 11000, "window_starts": STARTS,
        "window_samples": WIDTH, "split_seed": SOURCE_SEED, "utc_block_ns": BLOCK_NS,
        "guard_ns_each_side": GUARD_NS, "confidence_source_threshold": MIN_CONFIDENCE,
        "max_classes": MAX_CLASSES, "min_train": MIN_TRAIN, "min_val": MIN_VAL,
        "eligible_classes": len(eligible),
        "classes": [{"class_id": y, "satellite_id": pair[0], "beam_id": pair[1],
                     "uk_train": counts["train"][pair], "uk_val": counts["val"][pair]}
                    for y, pair in enumerate(pairs)],
        "source_counts": {role: len(entries) for role, entries in selected.items()},
        "source_shards": [name for name, _, _ in shards],
    }
    return contract, selected


def record_id(station, name, i):
    return sha256(f"{SCHEMA}/{station}/{name}/{i}".encode("ascii")).hexdigest()[:32]


def cut(samples, indices):
    windows = np.stack([samples[indices, start:start + WIDTH, :].transpose(0, 2, 1)
                        for start in STARTS], axis=1)
    if not np.isfinite(windows).all() or (np.square(windows).sum((1, 2, 3)) == 0).any():
        raise ValueError("Non-finite or zero-energy header")
    return windows


def prepare_source(root, out):
    shards = inventory(root, "UK")
    contract, selected = source_selection(shards)
    out.mkdir(exist_ok=False)
    for role, records in selected.items():
        storage, ids, y = [], [], []
        for name, _, arrays in shards:
            entries = [(i, label) for member, i, label in records if member == name]
            indices = [i for i, _ in entries]
            storage.append(cut(arrays["samples"], indices))
            ids.extend(record_id("UK", name, i) for i in indices)
            y.extend(label for _, label in entries)
        np.savez(out / f"{role}.npz", ids=np.asarray(ids), iq=np.concatenate(storage),
                 y=np.asarray(y, dtype=np.int64))
    write_json(out / "contract.json", contract)
    print(json.dumps({"source_counts": contract["source_counts"], "classes": contract["classes"]}))


def prepare_target(root, out):
    shards = inventory(root, "Switzerland")
    out.mkdir(exist_ok=False)
    (out / "query").mkdir()
    (out / "scorer").mkdir()
    ids = np.asarray([record_id("Switzerland", name, i) for name, _, _ in shards
                      for i in range(5000)])
    iq = np.lib.format.open_memmap(out / "query/iq.npy", mode="w+", dtype=np.float32,
                                  shape=(len(ids), len(STARTS), 2, WIDTH))
    for k, (_, _, arrays) in enumerate(shards):
        iq[5000 * k:5000 * (k + 1)] = cut(arrays["samples"], np.arange(5000))
    iq.flush()
    np.save(out / "query/ids.npy", ids)
    np.savez(
        out / "scorer/truth.npz", ids=ids,
        satellite_id=np.concatenate([a["satellite_id"] for _, _, a in shards]),
        beam_id=np.concatenate([a["cell_id"] for _, _, a in shards]),
        confidence=np.concatenate([a["confidence"] for _, _, a in shards]),
        original_split=np.asarray([split for _, split, _ in shards for _ in range(5000)]),
    )
    print(f"Exported label-free query and separate scorer truth for {len(ids)} Swiss records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("source", "target"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (prepare_source if args.mode == "source" else prepare_target)(args.input, args.output)
