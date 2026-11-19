"""Build source-only SatIQ IQ views for 66 physical (ra_sat, ra_cell) pairs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import io
from pathlib import Path
import re
import tarfile

import numpy as np

from .data import HEADER, SCHEMA, SOURCE_START, WIDTH, metadata, represent, write_json


BEAM_SCHEMA = "satiq_oxford_66_sat_cell_common_uw8000_v1"
SPLIT_SEED = 2026111901
L_PER_CLASS_DAY = 15
V_PER_CLASS = 36
COUNTS = {"L": 7920, "V": 2376}
RATES = [25000000, 800000]
RECORD_SHAPE = (11000, 2)
RECORD_BYTES = RECORD_SHAPE[0] * RECORD_SHAPE[1] * np.dtype("<f4").itemsize
VERIFY_FIELDS = {"ra_sat", "ra_cell", "timestamp_global", "sample_rate", "sample_count", "msg_type", "msg"}


def select_rows(rows):
    by_pair = {}
    for record in rows:
        pair, day = (record[3], record[5]), record[4]
        by_pair.setdefault(pair, [[] for _ in range(11)])[day].append(record)

    by_sat = {}
    for (sat, cell), days in by_pair.items():
        early = [len(group) for group in days[:8]]
        if min(early) >= L_PER_CLASS_DAY:
            by_sat.setdefault(sat, []).append((cell, sum(early)))
    ranked = [
        (sat, sorted(cells, key=lambda item: (-item[1], item[0]))[:3])
        for sat, cells in by_sat.items() if len(cells) >= 3
    ]
    ranked.sort(key=lambda item: (-sum(total for _, total in item[1]), item[0]))
    classes = [(sat, cell) for sat, cells in ranked[:22] for cell, _ in cells]
    if len(classes) != 66:
        raise ValueError(f"Only {len(classes)} cells fit the early UTC coverage requirement")
    for pair in classes:
        late = [len(group) for group in by_pair[pair][8:]]
        if sum(late) < V_PER_CLASS:
            raise ValueError(f"Frozen early-selected pair {pair} has insufficient V coverage: {late}")

    rng = np.random.default_rng(SPLIT_SEED)
    roles = {role: [] for role in COUNTS}
    for label, pair in enumerate(classes):
        days = by_pair[pair]
        for group in days[:8]:
            roles["L"].extend((label, group[i]) for i in rng.choice(len(group), L_PER_CLASS_DAY, replace=False))
        selected, remaining = [], []
        for group in days[8:]:
            order = rng.permutation(len(group))
            take = min(6, len(group))
            selected.extend(group[i] for i in order[:take])
            remaining.extend(group[i] for i in order[take:])
        selected.extend(remaining[i] for i in rng.choice(
            len(remaining), V_PER_CLASS - len(selected), replace=False,
        ))
        roles["V"].extend((label, record) for record in selected)
    return classes, roles


def read_prefixes(stream, indices, location):
    version = np.lib.format.read_magic(stream)
    shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
    if shape[1:] != RECORD_SHAPE or dtype != np.dtype("<f4") or fortran or indices[-1] >= shape[0]:
        raise ValueError(f"Invalid IQ array or physical row in {location}")
    packed = np.empty((len(indices), HEADER, 2), dtype=np.float32)
    for start in range(0, int(indices[-1]) + 1, 128):
        stop = min(start + 128, int(indices[-1]) + 1)
        chunk = stream.read((stop - start) * RECORD_BYTES)
        if len(chunk) != (stop - start) * RECORD_BYTES:
            raise EOFError(f"Truncated IQ array in {location}")
        iq = np.frombuffer(chunk, dtype=dtype).reshape((-1, *RECORD_SHAPE))
        first, last = indices.searchsorted([start, stop])
        packed[first:last] = iq[indices[first:last] - start, :HEADER]
    return packed


def verify_metadata(fields, requests, archive_name, segment):
    if set(fields) != VERIFY_FIELDS:
        raise ValueError(f"Missing RA metadata in {archive_name}/{segment}: {sorted(VERIFY_FIELDS - fields.keys())}")
    for i, (row, (_, _, sat, cell, day)) in enumerate(requests):
        timestamp = np.datetime64(int(fields["timestamp_global"][i]), "ns").astype("datetime64[D]")
        decoded = re.search(r"\bsat:(\d+)\s+beam:(\d+)\b", str(fields["msg"][i]))
        if ((int(fields["ra_sat"][i]), int(fields["ra_cell"][i])) != (sat, cell)
                or timestamp != SOURCE_START + day or fields["msg_type"][i] != "RA"
                or fields["sample_rate"][i] != RATES[0] or fields["sample_count"][i] != RECORD_SHAPE[0]
                or not decoded or tuple(map(int, decoded.groups())) != (sat, cell)):
            raise ValueError(f"Archive RA label or UTC date differs at {archive_name}/{segment}/{row}")


def export_archive(path, requested, views):
    indices = {seg: np.array(sorted(rows)) for seg, rows in requested.items()}
    fields = {seg: {} for seg in indices}
    complete_iq = set()
    with tarfile.open(path, "r|gz", bufsize=2**20) as archive:
        for member in archive:
            match = re.fullmatch(r"(.+)_(\d{3})\.npy", Path(member.name).name)
            if not match:
                continue
            key, seg = match[1], int(match[2])
            if seg not in requested or (key != "samples" and key not in VERIFY_FIELDS):
                continue
            with archive.extractfile(member) as stream:
                if key == "samples":
                    packed = read_prefixes(stream, indices[seg], f"{path.name}/{member.name}")
                    coarse = represent(packed)[:, 1]
                    for i, row in enumerate(indices[seg]):
                        role, pos, *_ = requested[seg][row]
                        views[role]["raw"][pos] = packed[i].T
                        views[role]["coarse"][pos] = coarse[i]
                    complete_iq.add(seg)
                else:
                    fields[seg][key] = np.load(io.BytesIO(stream.read()), allow_pickle=False)[indices[seg]]
            if complete_iq == requested.keys() and all(set(group) == VERIFY_FIELDS for group in fields.values()):
                break
    if complete_iq != requested.keys():
        raise ValueError(f"Missing IQ segments in {path.name}: {sorted(requested.keys() - complete_iq)}")
    for seg, rows in requested.items():
        verify_metadata(fields[seg], sorted(rows.items()), path.name, seg)
    count = sum(map(len, requested.values()))
    print(f"{path.name}: verified {count} IQ rows and archived RA labels", flush=True)
    return count


def export_source(archive_root, out, workers):
    classes, roles = select_rows(metadata(archive_root, False))
    out.mkdir(parents=True, exist_ok=False)
    seen, archives, metadata_by_role, views = set(), {}, {}, {}
    for role, records in roles.items():
        if len(records) != COUNTS[role]:
            raise ValueError(f"Unexpected record count for {role}: {len(records)}")
        ids, labels, days = [], [], []
        views[role] = {
            key: np.lib.format.open_memmap(out / f"{role}_{key}.npy", mode="w+", dtype=np.float32,
                                          shape=(len(records), 2, width))
            for key, width in (("raw", HEADER), ("coarse", WIDTH))
        }
        for pos, (label, (name, seg, row, sat, day, cell, _)) in enumerate(records):
            physical = f"{SCHEMA}/{name}/{seg}/{row}"
            if physical in seen:
                raise ValueError(f"Duplicate physical record {physical}")
            seen.add(physical)
            ids.append(sha256(physical.encode()).hexdigest()[:32])
            labels.append(label)
            days.append(day)
            archives.setdefault(name, {}).setdefault(seg, {})[row] = (role, pos, sat, cell, day)
        metadata_by_role[role] = {"ids": np.array(ids, dtype="U32"), "y": np.array(labels, dtype=np.int64),
                                  "day": np.array(days, dtype=np.int64)}

    print(f"Exporting {COUNTS} from {len(archives)} original archives", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(export_archive, archive_root / name, rows, views) for name, rows in archives.items()]
        if sum(job.result() for job in jobs) != sum(COUNTS.values()):
            raise ValueError("Exported record count differs from the frozen L/V split")
    names = ["contract.json"]
    for role, count in COUNTS.items():
        for key, width in (("raw", HEADER), ("coarse", WIDTH)):
            views[role].pop(key).flush()
            name = f"{role}_{key}.npy"
            names.append(name)
            saved = np.load(out / name, mmap_mode="r", allow_pickle=False)
            if saved.shape != (count, 2, width) or saved.dtype != np.float32:
                raise ValueError(f"Incorrect saved IQ layout for {name}")
            for start in range(0, count, 256):
                batch = saved[start:start + 256]
                if not np.isfinite(batch).all() or not np.any(batch, axis=(1, 2)).all():
                    raise ValueError(f"Non-finite or missing IQ in {name} near row {start}")
        name = f"{role}_meta.npz"
        names.append(name)
        np.savez(out / name, **metadata_by_role[role])
    write_json(out / "contract.json", {
        "schema": BEAM_SCHEMA, "prior_schema": SCHEMA, "classes": classes, "counts": COUNTS,
        "label_key": ["ra_sat", "ra_cell"], "sample_rates_hz": RATES, "prefix_samples": HEADER,
        "split_seed": SPLIT_SEED, "l_per_class_day": L_PER_CLASS_DAY, "v_per_class": V_PER_CLASS,
    })
    if {path.name for path in out.iterdir()} != set(names):
        raise ValueError(f"Unexpected files under {out}")
    print(f"Verified export: {out}", flush=True)
    return names


def pack_source(out, names, archive_path):
    with tarfile.open(archive_path, mode="x") as archive:
        for name in sorted(names):
            archive.add(out / name, arcname=name, recursive=False)
    with tarfile.open(archive_path, mode="r:") as archive:
        actual = {member.name: member.size for member in archive if member.isfile()}
    if actual != {name: (out / name).stat().st_size for name in names}:
        raise ValueError(f"Archive differs from the exported file list: {archive_path}")
    print(f"Verified uncompressed archive: {archive_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--tar", type=Path)
    args = parser.parse_args()
    if args.tar and args.tar.exists():
        parser.error(f"Archive output already exists: {args.tar}")
    names = export_source(args.archive_root, args.out, args.workers)
    if args.tar:
        pack_source(args.out, names, args.tar)
