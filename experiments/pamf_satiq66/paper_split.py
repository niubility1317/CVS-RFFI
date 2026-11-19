"""Stage the labelled SatIQ CCS 2023 cohort using physical archive references."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from .data import SCHEMA, read_json, write_json


COHORT = 1_010_464
HOLDOUT = 50_000
SEED = 2026111905
LEGACY_QUERY_START = np.datetime64("2023-04-24")


def stage(archive_root, out):
    index = archive_root / "_download_control" / "full_scan_20261119"
    paths = sorted(index.glob("data_*.tar.gz.npz"))
    names = [path.name.removesuffix(".npz") for path in paths]
    if not names or not all((archive_root / name).is_file() for name in names):
        raise ValueError("SatIQ original archives and full-scan metadata are required")

    timestamps, sats, beams, archives, segments, rows = [], [], [], [], [], []
    total = 0
    for a, path in enumerate(paths):
        info = read_json(index / f"{names[a]}.json")
        with np.load(path, allow_pickle=False) as f:
            t = f["timestamp_global"].astype("datetime64[ns]")
            total += len(t)
            idx = np.flatnonzero(t < LEGACY_QUERY_START)
            if not len(idx):
                continue
            if (not np.all(f["sample_rate"][idx] == 25_000_000)
                    or not np.all(f["sample_count"][idx] == 11_000)
                    or not np.all(f["msg_type"][idx] == "RA")):
                raise ValueError(f"Invalid ring-alert sampling geometry in {names[a]}")
            timestamps.append(t[idx])
            sats.append(f["ra_sat"][idx])
            beams.append(f["ra_cell"][idx])
            archives.append(np.full(len(idx), a, dtype=np.uint8))
            segments.append(np.asarray(info["segments"], dtype=np.uint16)[idx // 10_000])
            rows.append((idx % 10_000).astype(np.uint16))
    t = np.concatenate(timestamps)
    if len(t) < COHORT or total != 1_706_556:
        raise ValueError(f"Unexpected public release size: {total}")
    cohort = np.argsort(t, kind="stable")[:COHORT]
    t = t[cohort]
    sat, beam, archive, segment, row = (
        np.concatenate(parts)[cohort] for parts in (sats, beams, archives, segments, rows)
    )
    pair = np.stack((sat, beam), axis=1)
    if (sat.min() < 0 or sat.max() >= 128 or beam.min() < 0 or beam.max() >= 64
            or len(np.unique(np.stack((archive, segment, row), axis=1), axis=0)) != COHORT):
        raise ValueError("Invalid labels or repeated archive positions in development cohort")
    opaque_ids = np.fromiter(
        (sha256(f"{SCHEMA}/{names[a]}/{s}/{r}".encode()).digest()[:16]
         for a, s, r in zip(archive, segment, row, strict=True)), dtype="V16", count=COHORT
    )
    shuffled = np.random.default_rng(SEED).permutation(COHORT)
    # The paper code holds out TFRecord shards 4 and 8 (zero-based, 50,000 each).
    v = shuffled[4 * HOLDOUT:5 * HOLDOUT]
    test = shuffled[8 * HOLDOUT:9 * HOLDOUT]
    l = np.concatenate((shuffled[:4 * HOLDOUT], shuffled[5 * HOLDOUT:8 * HOLDOUT],
                        shuffled[9 * HOLDOUT:]))
    classes, y, counts = np.unique(pair, axis=0, return_inverse=True, return_counts=True)
    counts_l = np.bincount(y[l], minlength=len(classes))
    visible_map = np.full(len(classes), -1, dtype=np.int32)
    visible_map[counts_l > 0] = np.arange((counts_l > 0).sum())
    labels = visible_map[y]

    out.mkdir(parents=True, exist_ok=False)
    source, heldout, scorer = (out / name for name in ("source", "heldout_iq", "scorer_only"))
    for folder in (source, heldout, scorer):
        folder.mkdir()

    def index_for(idx):
        return dict(archive=archive[idx], segment=segment[idx], row=row[idx], ids=opaque_ids[idx])

    for name, idx in (("L", l), ("V", v)):
        np.savez(source / f"{name}_index.npz", **index_for(idx), y=labels[idx], timestamp_ns=t[idx].astype(np.int64))
    np.savez(heldout / "T_index.npz", **index_for(test))
    np.savez(scorer / "T_truth.npz", ids=opaque_ids[test], y=labels[test], ra_sat=sat[test], ra_cell=beam[test])
    write_json(heldout / "archive_map.json", {"archives": names, "sample_rate_hz": 25_000_000, "samples": 11_000})
    train_classes = classes[counts_l > 0].tolist()
    stats = {
        "schema": "satiq_oxford_ccs2023_supervised_reindexed_v1",
        "preprocessing_reference": "SatIQ v1.0.2; original shuffle state unavailable",
        "cohort_selection": "First 1010464 by timestamp with release order as the tie-break",
        "cohort_range_utc": [str(t[0]), str(t[-1])],
        "release_records": total,
        "split_seed": SEED,
        "archive_map": names,
        "label_key": ["ra_sat", "ra_cell"],
        "class_pairs": train_classes,
        "training_class_counts": counts_l[counts_l > 0].tolist(),
        "development_classes": len(counts),
        "train_classes": len(train_classes),
        "train_classes_with_at_least_four": int((counts_l >= 4).sum()),
        "records": {"L": len(l), "V": len(v), "T": len(test)},
        "unseen_class_records": {"V": int((labels[v] < 0).sum()), "T": int((labels[test] < 0).sum())},
        "sample_rate_hz": 25_000_000,
        "samples": 11_000,
        "batch_recipe": {"classes": 8, "distinct_records_per_class": 4},
        "training_mode": "supervised_only",
    }
    write_json(source / "contract.json", stats)
    print({key: value for key, value in stats.items()
           if key not in {"archive_map", "class_pairs", "training_class_counts"}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stage(args.archives, args.output)
