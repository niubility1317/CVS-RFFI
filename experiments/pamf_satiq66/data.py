"""Prepare isolated SatIQ source and future views from the original archives."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
from shutil import copy2
import tarfile

import numpy as np
from scipy.signal import firwin, resample_poly


CLASSES = (78, 57, 36, 69, 44, 79, 73, 99, 110, 114, 88, 9, 16, 115, 7, 22, 104, 4,
           103, 26, 17, 112, 111, 24, 38, 92, 109, 93, 89, 28, 87, 51, 85, 13, 42, 46,
           71, 68, 96, 3, 18, 49, 2, 90, 94, 72, 48, 29, 23, 81, 50, 25, 67, 65, 33,
           107, 30, 8, 40, 77, 39, 74, 6, 5, 43, 82)
SCHEMA = "satiq_zenodo_8220494_oxford_66_v1"
SOURCE_START = np.datetime64("2023-03-24")
QUERY_START = np.datetime64("2023-04-24")
SOURCE_COUNTS = {"L": 10560, "U": 95040, "V": 19800}
SYMBOLS = np.array([1 + 1j, -1 - 1j, -1 - 1j, -1 - 1j, -1 - 1j, 1 + 1j, 1 + 1j, 1 + 1j],
                   dtype=np.complex64)
WINDOW_START = 384
HEADER = 8000
WIDTH = 256
FIR = firwin(3073, 225000, fs=100_000_000, window=("kaiser", 8.6)).astype(np.float32)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def represent(rows):
    if rows.ndim != 3 or rows.shape[1:] != (HEADER, 2):
        raise ValueError("Expected the 8000-sample common unique-word prefix")
    narrow = resample_poly(rows, 4, 125, axis=1, window=FIR, padtype="line")
    direct = rows[:, WINDOW_START:WINDOW_START + WIDTH]
    output = np.stack((direct.transpose(0, 2, 1), narrow.transpose(0, 2, 1)), axis=1)
    if not np.isfinite(output).all() or np.any(np.square(output).sum(axis=(2, 3)) <= 0):
        raise ValueError("Invalid common-word IQ")
    return output.astype(np.float32, copy=False)


def metadata(root, future):
    control = root / "_download_control" / "full_scan_20261119"
    for path in sorted(control.glob("data_*.tar.gz.npz")):
        name = path.name.removesuffix(".npz")
        info = read_json(control / (name + ".json"))
        with np.load(path, allow_pickle=False) as archive:
            dates = archive["timestamp_global"].astype("datetime64[ns]").astype("datetime64[D]")
            if future:
                mask = (dates >= QUERY_START) & (dates < QUERY_START + 7)
            else:
                mask = (dates >= SOURCE_START) & (dates < SOURCE_START + 11)
            indices = np.flatnonzero(mask)
            if not indices.size:
                continue
            start = QUERY_START if future else SOURCE_START
            sat = archive["ra_sat"][indices]
            rate = archive["sample_rate"][indices]
            if not np.all(rate == 25000000):
                raise ValueError(f"Unexpected sample rate in {name}")
            arr = (dates[indices] - start).astype(int)
            for index, satellite, day, beam, quality in zip(
                indices, sat, arr, archive["ra_cell"][indices], archive["magnitude"][indices], strict=True,
            ):
                segment = int(info["segments"][index // 10000])
                yield (name, segment, int(index % 10000), int(satellite), int(day), int(beam), float(quality))


def source_roles(rows, seed):
    rng = np.random.default_rng(seed)
    by_class_day = {}
    for row in rows:
        by_class_day.setdefault((row[3], row[4]), []).append(row)
    candidates = [sat for sat in {row[3] for row in rows}
                  if min(len(by_class_day.get((sat, day), ())) for day in range(8)) >= 200]
    candidates.sort(key=lambda sat: (-sum(len(by_class_day[sat, d]) for d in range(8)), sat))
    if tuple(candidates) != CLASSES:
        raise ValueError("Source-only eligibility does not match the frozen satellite mapping")
    result = {role: [] for role in SOURCE_COUNTS}
    for satellite in CLASSES:
        for day in range(11):
            group = by_class_day[satellite, day]
            if day < 8:
                selected = rng.choice(len(group), 200, replace=False)
                result["L"].extend(group[i] for i in selected[:20])
                result["U"].extend(group[i] for i in selected[20:])
            else:
                result["V"].extend(group[i] for i in rng.choice(len(group), 100, replace=False))
    return result


def read_archive(task):
    path, segments = task
    received = {}
    with tarfile.open(path, mode="r|gz", bufsize=2**20) as archive:
        for member in archive:
            match = re.fullmatch(r"samples_(\d{3})\.npy", Path(member.name).name)
            if not match or int(match[1]) not in segments:
                continue
            segment = int(match[1])
            requests = sorted(segments[segment])
            with archive.extractfile(member) as f:
                version = np.lib.format.read_magic(f)
                shape, fortran, dtype = np.lib.format._read_array_header(f, version)
                if shape[1:] != (11000, 2) or dtype != np.float32 or fortran:
                    raise ValueError(f"Invalid IQ layout in {path.name}/{member.name}")
                packed = np.empty((len(requests), HEADER, 2), dtype=np.float32)
                digests = []
                next_row = 0
                for pos, row in enumerate(requests):
                    if row >= shape[0]:
                        raise ValueError(f"Physical row outside {member.name}")
                    for offset in range(next_row, row + 1, 128):
                        count = min(row + 1 - offset, 128)
                        buffer = f.read(count * 11000 * 8)
                        if len(buffer) != count * 11000 * 8:
                            raise EOFError(f"Truncated IQ data in {member.name}")
                    raw = memoryview(buffer)[-11000 * 8:]
                    packed[pos] = np.frombuffer(raw, dtype=np.float32).reshape(11000, 2)[:HEADER]
                    digests.append(sha256(raw).hexdigest())
                    next_row = row + 1
            received[segment] = requests, represent(packed), digests
            if set(received) == set(segments):
                break
    if set(received) != set(segments):
        raise ValueError(f"Missing source segments in {path.name}")
    return path.name, received


def prepare(root, out, seed, future, workers, source=None):
    rows = list(metadata(root, future))
    mapping = {sat: i for i, sat in enumerate(CLASSES)}
    roles = {"Q": [row for row in rows if row[3] in mapping]} if future else source_roles(rows, seed)
    archives, indices = {}, {}
    out.mkdir(parents=True, exist_ok=False)
    public = out / "public"
    public.mkdir()
    private = out / "scorer"
    private.mkdir()
    storage, ids, hashes = {}, {}, {}
    for role, records in roles.items():
        storage[role] = np.lib.format.open_memmap(public / f"{role}_iq.npy", mode="w+", dtype=np.float32,
                                                   shape=(len(records), 2, 2, WIDTH))
        ids[role] = np.empty(len(records), dtype="U32")
        hashes[role] = np.empty(len(records), dtype="U64")
        for pos, row in enumerate(records):
            key = row[:3]
            if key in indices:
                raise ValueError(f"Duplicate physical record {key}")
            indices[key] = role, pos
            ids[role][pos] = sha256(f"{SCHEMA}/{key[0]}/{key[1]}/{key[2]}".encode()).hexdigest()[:32]
            archives.setdefault(key[0], {}).setdefault(key[1], []).append(key[2])
    tasks = [(root / name, segments) for name, segments in sorted(archives.items())]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, results in pool.map(read_archive, tasks):
            for segment, (rows, iq, digests) in results.items():
                for i, row in enumerate(rows):
                    role, pos = indices[name, segment, row]
                    storage[role][pos] = iq[i]
                    hashes[role][pos] = digests[i]
            print(f"Prepared {name}", flush=True)
    seen = set()
    if future:
        for role in SOURCE_COUNTS:
            seen.update(np.load(source / "scorer" / f"{role}_raw_hashes.npy", allow_pickle=False).tolist())
    for role, rows in roles.items():
        current = set(hashes[role].tolist())
        if "" in current or seen & current:
            raise ValueError("Unfilled row or identical full raw IQ crossing data roles")
        seen.update(current)
        storage[role].flush()
        np.save(private / f"{role}_raw_hashes.npy", hashes[role])
        y = np.array([mapping[row[3]] for row in rows], dtype=np.int64)
        day = np.array([row[4] for row in rows], dtype=np.int64)
        np.savez(public / f"{role}_meta.npz", ids=ids[role],
                 **({"day": day, "y": y} if role in ("L", "V") else {"day": day} if role == "U" else {}))
        if role == "Q":
            np.savez(private / "truth.npz", ids=ids[role], y=y, day=day,
                     beam=np.array([row[5] for row in rows]), magnitude=np.array([row[6] for row in rows]))
    contract = dict(schema=SCHEMA, classes=list(CLASSES), counts={k: len(v) for k, v in roles.items()},
                    split_seed=seed, iq_rate_hz=[25000000, 800000], width=WIDTH, header_samples=HEADER,
                    raw_window_start=WINDOW_START, true_receiver_count=1,
                    low_quality_threshold=None if future else float(np.quantile([r[6] for r in roles["L"]], .2)))
    write_json(public / "contract.json", contract)
    print(json.dumps(contract), flush=True)


def export_source(prepared, out):
    """Publish source views with the U row order independent of label blocks."""
    contract = read_json(prepared / "contract.json")
    if contract["counts"] != SOURCE_COUNTS or contract["schema"] != SCHEMA:
        raise ValueError("Expected the balanced 66-class source capsule")
    out.mkdir(parents=True, exist_ok=False)
    for name in ("contract.json", "L_iq.npy", "L_meta.npz", "V_iq.npy", "V_meta.npz"):
        copy2(prepared / name, out / name)
    with np.load(prepared / "U_meta.npz", allow_pickle=False) as u:
        if set(u.files) != {"ids", "day"}:
            raise ValueError("Hidden-U source payload must expose only ids and day")
        order = np.random.default_rng(contract["split_seed"] + 1).permutation(len(u["ids"]))
        np.savez(out / "U_meta.npz", ids=u["ids"][order], day=u["day"][order])
    source = np.load(prepared / "U_iq.npy", mmap_mode="r", allow_pickle=False)
    dest = np.lib.format.open_memmap(out / "U_iq.npy", mode="w+", dtype=np.float32, shape=source.shape)
    for start in range(0, len(order), 4096):
        dest[start:start + 4096] = source[order[start:start + 4096]]
    dest.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("source", "future", "export-source"))
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.role == "future" and args.source is None:
        parser.error("future requires --source for the cross-role overlap check")
    if args.role == "export-source":
        export_source(args.archive_root, args.out)
    else:
        prepare(args.archive_root, args.out, 2026111901, args.role == "future", args.workers, args.source)
