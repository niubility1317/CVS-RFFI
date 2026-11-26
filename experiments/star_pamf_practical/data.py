"""Load the existing source handles and produce paired practical-residual IQ."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.cric_dura_comparison.source_iq import aligned_source_iq
from experiments.practical_full_2x2.cache import transform

ROLES = ("L_s", "U_s", "V")
SOURCE = (1, 3, 4, 6, 8)
DAYS = (1, 2, 3)
COUNTS = dict(zip(ROLES, (6300, 56700, 27000)))
PROTOCOL = "F0"
CLASSES = ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"]


def load_source(folder, roles=ROLES, *, expected=None):
    folder = Path(folder)
    contract = read(folder / "source_contract.json")
    if expected is None:
        expected = dict(source_rxs=list(SOURCE), source_days=list(DAYS), counts=COUNTS, classes=CLASSES)
    if any(contract[k] != expected[k] for k in ("source_rxs", "source_days", "counts", "classes")) or (
            contract["split_seed"], contract["checkpoint_init"]) != (392005, "scratch_only"):
        raise ValueError("The registered PAMF source split is required")
    rxs, days, counts, classes = (contract[k] for k in ("source_rxs", "source_days", "counts", "classes"))
    seen = set()
    data = {}
    for role in roles:
        with np.load(folder / f"{role}_metadata.npz", allow_pickle=False) as arrays:
            if role == "U_s" and {"y", "tx_i", "label", "labels", "truth"} & set(arrays.files):
                raise ValueError("U_s cache exposes transmitter labels")
            names = ("ids", "rx_i", "day_i") + (() if role == "U_s" else ("y",))
            meta = {name: arrays[name] for name in names}
        ids = meta["ids"].tolist()
        if (ids != contract["role_ids"][role] or len(ids) != counts[role]
                or seen.intersection(ids) or set(meta["rx_i"]) != set(rxs)
                or set(meta["day_i"]) != set(days)):
            raise ValueError(f"Source role differs from the fixed contract: {role}")
        seen.update(ids)
        x, _ = aligned_source_iq(folder, role, ids)
        if x.dtype != np.float32:
            raise ValueError("Expected FP32 source IQ")
        if role != "U_s" and set(meta["y"]) != set(range(len(classes))):
            raise ValueError(f"Invalid source class map for {role}")
        meta["domain"] = np.searchsorted(rxs, meta["rx_i"]).astype(np.int64)
        data[role] = x, meta
    return data


_AUGMENT_SOURCE = None


def _initialize(folder, role, expected):
    global _AUGMENT_SOURCE
    _AUGMENT_SOURCE = load_source(folder, (role,), expected=expected)[role]


def _chunk(bounds):
    x, meta = _AUGMENT_SOURCE
    start, end = bounds
    values, scenes = [], []
    for index in range(start, end):
        y, scene, _ = transform(x[index], meta["ids"][index], int(meta["rx_i"][index]),
                                int(meta["day_i"][index]), 1, 1, 1.0)
        values.append(y)
        scenes.append(scene)
    return start, np.asarray(values, np.float32), np.asarray(scenes, np.int8)


def prepare_pairs(folder, output, workers, *, expected=None):
    source = load_source(folder, ("L_s", "U_s"), expected=expected)
    output = Path(output)
    output.mkdir(exist_ok=False)
    counts = {}
    for role in ("L_s", "U_s"):
        x, meta = source[role]
        np.save(output / f"{role}_ids.npy", meta["ids"], allow_pickle=False)
        result = np.lib.format.open_memmap(output / f"{role}_sg.npy", mode="w+",
                                           dtype=np.float32, shape=x.shape)
        scenes = np.empty(len(x), np.int8)
        spans = [(i, min(i + 128, len(x))) for i in range(0, len(x), 128)]
        with ProcessPoolExecutor(max_workers=workers, initializer=_initialize,
                                 initargs=(folder, role, expected)) as pool:
            for start, batch, indices in pool.map(_chunk, spans):
                result[start:start + len(batch)] = batch
                scenes[start:start + len(indices)] = indices
        result.flush()
        if set(scenes) != {0, 1, 2}:
            raise ValueError("Practical training mixture incomplete")
        np.save(output / f"{role}_scene.npy", scenes, allow_pickle=False)
        counts[role] = np.bincount(scenes, minlength=3).tolist()
        print("PRACTICAL_PAIRED_IQ_READY", role, counts[role], flush=True)
    write(output / "complete.json", dict(scene_counts=counts, U_TX_truth_read=False,
                                        validation_used=False, target_read=False))


def load_pairs(folder, source):
    output = Path(folder)
    read(output / "complete.json")
    pairs = {}
    for role in ("L_s", "U_s"):
        x, meta = source[role]
        ids = np.load(output / f"{role}_ids.npy", allow_pickle=False)
        sg = np.load(output / f"{role}_sg.npy", allow_pickle=False, mmap_mode="r")
        scene = np.load(output / f"{role}_scene.npy", allow_pickle=False)
        if not np.array_equal(ids, meta["ids"]) or sg.shape != x.shape or scene.shape != ids.shape:
            raise ValueError(f"Practical views and source physical IDs differ for {role}")
        pairs[role] = x, sg, scene
    return pairs
