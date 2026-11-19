"""Validate and index the original Oxford and Thun full-header NumPy shards."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .prepare import FS, STARTS, WIDTH, record_id, write_json


HEADER = 11000
SHARD_SIZE = 5000
STATIONS = {"source": ("UK", "source"), "target": ("Switzerland", "target/query")}
FULL_STARTS = (*range(0, HEADER - WIDTH + 1, WIDTH), HEADER - WIDTH)
VIEWS = {"crop": STARTS, "full": FULL_STARTS}
SCHEMA = "satiq_oxford_thun_full_header_v1"


def array(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def install(root, stage, hashes, compact):
    """Use byte-identical original FP32 arrays and add separate station indexes."""
    station, sub = STATIONS[stage]
    base = root / sub
    manifest = read(hashes)
    if manifest["sample_rate_hz"] != FS:
        raise ValueError("Unexpected source asset sample rate")
    spec = sorted((e for e in manifest["shards"] if e["station"] == station),
                  key=lambda e: e["name"])
    if (len(spec) != 6 or len({e["name"] for e in spec}) != 6
            or any(e["records"] != SHARD_SIZE for e in spec)):
        raise ValueError(f"{stage}: invalid shard inventory")
    entries, ids = [], []
    for e in spec:
        name = e["name"]
        relative = Path("iq") / name / "iq.npy"
        path = base / relative
        if path.stat().st_size != e["size"]:
            raise ValueError(f"{path}: partial transfer")
        checksum = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 2**20), b""):
                checksum.update(block)
        data = array(path)
        if (checksum.hexdigest() != e["sha256"] or data.dtype != np.float32
                or data.shape != (SHARD_SIZE, HEADER, 2)):
            raise ValueError(f"{path}: the raw FP32 array differs from the dorm asset")
        entries.append(dict(path=relative.as_posix(), **e))
        ids.extend(record_id(station, name + ".tfrecord", row) for row in range(SHARD_SIZE))
    ids = np.asarray(ids)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Duplicate physical IDs")
    if stage == "target":
        previous = array(compact / "target/query/ids.npy")
        if not np.array_equal(ids, previous):
            raise ValueError("Full Swiss headers do not match the held-out manifest order")
    with (base / "ids.npy").open("xb") as f:
        np.save(f, ids)
    index = dict(schema=SCHEMA, station=station, role=stage, sample_rate_hz=FS,
                 records=len(ids), dtype="float32", axes=["packet", "sample", "iq"],
                 samples_per_packet=HEADER, shards=entries)
    marker = base / "index.json"
    if marker.exists():
        raise FileExistsError(marker)
    pending = base / ".index.pending.json"
    write_json(pending, index)
    pending.rename(marker)
    print(json.dumps({"verified": station, "shards": len(entries), "packets": len(ids),
                      "bytes": sum(e["size"] for e in entries)}), flush=True)


class Headers:
    """Select complete physical packet rows from read-only per-station memory maps."""

    def __init__(self, root, stage):
        station, sub = STATIONS[stage]
        base = root / sub
        index = read(base / "index.json")
        if (index["schema"] != SCHEMA or index["role"] != stage
                or index["station"] != station or index["records"] != 30000):
            raise ValueError("Wrong SatIQ full-header station index")
        self.ids = array(base / "ids.npy")
        self.shards = [array(base / e["path"]) for e in index["shards"]]
        if len(set(self.ids.tolist())) != 30000 or len(self.shards) != 6:
            raise ValueError("Full-header physical manifest differs")

    def rows(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        out = np.empty((len(indices), HEADER, 2), dtype=np.float32)
        for number in np.unique(indices // SHARD_SIZE):
            take = indices // SHARD_SIZE == number
            out[take] = self.shards[number][indices[take] % SHARD_SIZE]
        return out


def views(raw, mode):
    return np.stack([raw[:, start:start + WIDTH, :].transpose(0, 2, 1)
                     for start in VIEWS[mode]], axis=1)


def paired_uk(full, compact):
    source = Headers(full, "source")
    lookup = {identifier: i for i, identifier in enumerate(source.ids.tolist())}
    result, identities = {}, {}
    for role in ("train", "val"):
        with np.load(compact / "source" / f"{role}.npz", allow_pickle=False) as d:
            ids = d["ids"].tolist()
            raw = source.rows([lookup[identifier] for identifier in ids])
            if not np.array_equal(d["iq"], views(raw, "crop")):
                raise ValueError(f"UK {role}: full header does not reproduce original crops")
            if (not np.isfinite(raw).all()
                    or np.any(np.square(raw).sum(axis=(1, 2)) <= 0)):
                raise ValueError(f"UK {role}: invalid full-header waveform")
            result[role] = raw, d["y"]
        identities[role] = set(ids)
    if identities["train"] & identities["val"]:
        raise ValueError("UK train/validation ID overlap")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("stage", choices=STATIONS)
    parser.add_argument("hashes", type=Path)
    parser.add_argument("compact", type=Path)
    args = parser.parse_args()
    install(args.root, args.stage, args.hashes, args.compact)
