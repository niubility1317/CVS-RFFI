"""Build separate SatIQ CCS 2023 IQ caches from staged physical archive indices."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import io
from pathlib import Path, PurePosixPath
import re
import tarfile
import time

import numpy as np

from .data import SCHEMA, read_json


PAPER_SCHEMA = "satiq_oxford_ccs2023_supervised_reindexed_v1"
SAMPLES = 11_000
SAMPLE_RATE = 25_000_000
IQ_DTYPE = np.dtype("<f4")
RECORD_BYTES = SAMPLES * 2 * IQ_DTYPE.itemsize
POSITION_FIELDS = ("archive", "segment", "row")
ARCHIVE_METADATA = {"ra_sat", "ra_cell", "timestamp_global", "sample_rate", "sample_count"}
MEMBER = re.compile(r"(.+)_(\d{3})\.npy")


def load_group(role_group, manifest):
    if role_group == "source":
        contract = read_json(manifest / "contract.json")
        if contract["schema"] != PAPER_SCHEMA:
            raise ValueError(f"Unrecognized source manifest schema in {manifest}")
        names = contract["archive_map"]
        roles = ("L", "V")
        counts = contract["records"]
        classes = np.asarray(contract["class_pairs"], dtype=np.int64)
        if (classes.ndim != 2 or classes.shape[1] != 2 or len(classes) != contract["train_classes"]
                or np.any(classes < 0) or np.any(classes[:, 0] >= 128) or np.any(classes[:, 1] >= 64)
                or len(np.unique(classes, axis=0)) != len(classes)):
            raise ValueError(f"Invalid source class pairs in {manifest}")
        classes_by_pair = np.full((128, 64), -1, dtype=np.int32)
        classes_by_pair[classes[:, 0], classes[:, 1]] = np.arange(len(classes))
    else:
        contract = read_json(manifest / "archive_map.json")
        names = contract["archives"]
        roles = ("T",)
        counts = {}
        classes_by_pair = None
    if contract["sample_rate_hz"] != SAMPLE_RATE or contract["samples"] != SAMPLES:
        raise ValueError(f"Unexpected raw IQ geometry in {manifest}")
    if not names or len(set(names)) != len(names) or not all(
        re.fullmatch(r"data_\d+_\d+\.tar\.gz", name) for name in names
    ):
        raise ValueError(f"Invalid archive table in {manifest}")

    indices = {}
    for role in roles:
        fields = {*POSITION_FIELDS, "ids"}
        if role != "T":
            fields.update(("y", "timestamp_ns"))
        with np.load(manifest / f"{role}_index.npz", allow_pickle=False) as index:
            if set(index.files) != fields:
                raise ValueError(f"Unexpected index fields for {role}")
            arrays = {field: index[field] for field in fields}
        count = len(arrays["ids"])
        if not count or count != counts.get(role, count) or any(a.shape != (count,) for a in arrays.values()):
            raise ValueError(f"Unexpected index shape or count for {role}")
        if arrays["ids"].dtype != np.dtype("V16"):
            raise ValueError(f"Unexpected physical ID type for {role}")
        for field, limit in zip(POSITION_FIELDS, (len(names), 1000, 10_000), strict=True):
            values = arrays[field]
            if values.dtype.kind not in "iu" or np.any(values >= limit) or np.any(values < 0):
                raise ValueError(f"Invalid {field} index for {role}")
        if role != "T" and (arrays["timestamp_ns"].dtype != np.int64 or arrays["y"].dtype.kind not in "iu"
                            or np.any(arrays["y"] < (0 if role == "L" else -1))
                            or np.any(arrays["y"] >= contract["train_classes"])):
            raise ValueError(f"Invalid timestamps or labels for {role}")
        expected_ids = np.fromiter(
            (sha256(f"{SCHEMA}/{names[a]}/{s}/{r}".encode()).digest()[:16]
             for a, s, r in zip(*(arrays[key] for key in POSITION_FIELDS), strict=True)),
            dtype="V16", count=count,
        )
        if not np.array_equal(arrays["ids"], expected_ids):
            raise ValueError(f"Physical IDs do not match archive positions for {role}")
        indices[role] = arrays
    return names, indices, classes_by_pair, counts


def archive_requests(indices):
    """Group physical coordinates by archive segment without a Python object per row."""
    coordinates = []
    requests = {}
    for role, arrays in indices.items():
        archive, segment, row = (arrays[key] for key in POSITION_FIELDS)
        packed = archive.astype(np.uint64) * (1000 * 10_000) + segment.astype(np.uint64) * 10_000 + row
        coordinates.append(packed)
        order = np.argsort(packed, kind="stable")
        group = packed[order] // 10_000
        for positions in np.split(order, np.flatnonzero(group[1:] != group[:-1]) + 1):
            first = positions[0]
            roles = requests.setdefault(int(archive[first]), {}).setdefault(int(segment[first]), {})
            roles[role] = row[positions].astype(np.int32), positions
    all_positions = np.concatenate(coordinates)
    if len(np.unique(all_positions)) != len(all_positions):
        raise ValueError("A physical archive position was selected more than once")
    return requests


def stream_iq(stream, source, requests, outputs):
    version = np.lib.format.read_magic(stream)
    shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
    max_row = max(rows[-1] for rows, _ in requests.values())
    if len(shape) != 3 or shape[1:] != (SAMPLES, 2) or fortran or dtype != IQ_DTYPE or max_row >= shape[0]:
        raise ValueError(f"Invalid IQ layout or physical row in {source}")
    for start in range(0, int(max_row) + 1, 128):
        stop = min(start + 128, int(max_row) + 1)
        payload = stream.read((stop - start) * RECORD_BYTES)
        if len(payload) != (stop - start) * RECORD_BYTES:
            raise EOFError(f"Truncated IQ data in {source}")
        block = np.frombuffer(payload, dtype=dtype).reshape((-1, SAMPLES, 2))
        for role, (rows, positions) in requests.items():
            left, right = rows.searchsorted([start, stop])
            outputs[role][positions[left:right]] = block[rows[left:right] - start].transpose(0, 2, 1)


def verify_source_metadata(metadata, requests, indices, classes_by_pair, location):
    for role, (rows, positions) in requests.items():
        if role == "T":
            continue
        sat, cell = metadata["ra_sat"][rows], metadata["ra_cell"][rows]
        times = metadata["timestamp_global"][rows].astype("datetime64[ns]").astype(np.int64)
        if (np.any(sat < 0) or np.any(sat >= 128) or np.any(cell < 0) or np.any(cell >= 64)
                or not np.array_equal(classes_by_pair[sat, cell], indices[role]["y"][positions])
                or not np.array_equal(times, indices[role]["timestamp_ns"][positions])
                or not np.all(metadata["sample_rate"][rows] == SAMPLE_RATE)
                or not np.all(metadata["sample_count"][rows] == SAMPLES)):
            raise ValueError(f"Archived RA labels, timestamps or sample geometry differ from {role} in {location}")


def materialize_archive(path, segments, outputs, indices, classes_by_pair):
    metadata = {seg: {} for seg, roles in segments.items() if set(roles) != {"T"}}
    complete = set()
    with tarfile.open(path, mode="r|gz", bufsize=2**20) as archive:
        for member in archive:
            match = MEMBER.fullmatch(PurePosixPath(member.name).name)
            if not match:
                continue
            field, segment = match[1], int(match[2])
            location = f"{path.name}/{member.name}"
            if field == "samples" and segment in segments:
                if segment in complete:
                    raise ValueError(f"Repeated IQ member {location}")
                with archive.extractfile(member) as stream:
                    stream_iq(stream, location, segments[segment], outputs)
                complete.add(segment)
            elif field in ARCHIVE_METADATA and segment in metadata:
                with archive.extractfile(member) as stream:
                    metadata[segment][field] = np.load(io.BytesIO(stream.read()), allow_pickle=False)
            if complete == segments.keys() and all(group.keys() == ARCHIVE_METADATA for group in metadata.values()):
                break
    if complete != segments.keys():
        raise ValueError(f"Missing IQ segments in {path}: {sorted(segments.keys() - complete)}")
    for segment, group in metadata.items():
        if group.keys() != ARCHIVE_METADATA:
            raise ValueError(f"Missing source metadata in {path.name}/{segment}")
        verify_source_metadata(group, segments[segment], indices, classes_by_pair, f"{path.name}/{segment}")
    counts = {role: sum(len(group[role][0]) for group in segments.values() if role in group) for role in indices}
    print(f"{path.name}: {counts}", flush=True)
    return counts


def build_cache(role_group, archives, manifest, out, workers=4, heldout_manifest=None, heldout_out=None,
                wait_archive_seconds=0):
    if (heldout_manifest is None) != (heldout_out is None) or (heldout_out is not None and role_group != "source"):
        raise ValueError("Combined caching requires both heldout paths on the source command")
    groups = {role_group: out.resolve()}
    names, indices, classes_by_pair, counts = load_group(role_group, manifest)
    if heldout_out is not None:
        heldout_names, heldout_index, _, _ = load_group("heldout", heldout_manifest)
        if names != heldout_names or len(heldout_index["T"]["ids"]) != counts["T"]:
            raise ValueError("Heldout manifest disagrees with the source archive map or count")
        groups["heldout"] = heldout_out.resolve()
        if len(set(groups.values())) != 2 or any(a in b.parents for a in groups.values() for b in groups.values()):
            raise ValueError("Source and heldout caches must be physically separate directories")
        indices.update(heldout_index)
    requests = archive_requests(indices)
    missing = [names[archive] for archive in requests if not (archives / names[archive]).is_file()]
    if missing and not wait_archive_seconds:
        raise FileNotFoundError(f"Required staged archives are missing: {', '.join(missing)}")

    staging = {group: dest.with_name(dest.name + ".partial") for group, dest in groups.items()}
    for dest in (*groups.values(), *staging.values()):
        if dest.exists():
            raise FileExistsError(dest)
    for directory in staging.values():
        directory.mkdir(parents=True, exist_ok=False)
    expected = {role: len(index["ids"]) for role, index in indices.items()}
    paths = {role: staging["heldout" if role == "T" else "source"] / f"{role}_raw.npy" for role in indices}
    outputs = {role: np.lib.format.open_memmap(path, mode="w+", dtype=IQ_DTYPE, shape=(expected[role], 2, SAMPLES))
               for role, path in paths.items()}
    print(f"Building {expected} from {len(requests)} staged archives", flush=True)
    totals = dict.fromkeys(indices, 0)
    deadline = time.monotonic() + wait_archive_seconds

    def cache_archive(index, segments):
        path = archives / names[index]
        while not path.is_file():
            if time.monotonic() >= deadline:
                raise FileNotFoundError(f"Verified archive did not arrive: {path}")
            time.sleep(3)
        return materialize_archive(path, segments, outputs, indices, classes_by_pair)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(cache_archive, a, requests[a]) for a in sorted(requests)]
        for job in jobs:
            for role, count in job.result().items():
                totals[role] += count
    if totals != expected:
        raise ValueError(f"Cached row counts {totals} differ from the manifests {expected}")
    for role, path in paths.items():
        outputs.pop(role).flush()
        saved = np.load(path, mmap_mode="r", allow_pickle=False)
        if saved.shape != (expected[role], 2, SAMPLES) or saved.dtype != IQ_DTYPE:
            raise ValueError(f"Incorrect cached IQ layout in {path}")
        del saved
    for group, dest in groups.items():
        staging[group].rename(dest)
        print(f"Verified {group} cache: {dest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=("source", "heldout"))
    parser.add_argument("archives", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--heldout-manifest", type=Path)
    parser.add_argument("--heldout-out", type=Path)
    parser.add_argument("--wait-archive-seconds", type=int, default=0)
    args = parser.parse_args()
    build_cache(args.group, args.archives, args.manifest, args.out, args.workers,
                args.heldout_manifest, args.heldout_out, args.wait_archive_seconds)
