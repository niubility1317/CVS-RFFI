"""Build smaller source-only role caches from the retained WiSig caches."""
from pathlib import Path
import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.cric_dura_comparison.source_iq import aligned_source_iq
from .plan import ROLES, SPLIT_SEED

LABELS = {"y"}
METADATA = {"rx_i", "day_i", "eq_i", "sig_i", "base_index", "domain", "ids"}


def load_role(root, role):
    root = Path(root)
    contract = read(root / "source_contract.json")
    with np.load(root / f"{role}_metadata.npz", allow_pickle=False) as archive:
        if set(archive.files) != METADATA | (LABELS if role != "U_s" else set()):
            raise ValueError(f"Unexpected metadata for {role}: {root}")
        meta = {key: archive[key] for key in archive.files}
    if meta["ids"].tolist() != contract["role_ids"][role]:
        raise ValueError("Source cache metadata order differs from its contract")
    x, _ = aligned_source_iq(root, role, meta["ids"])
    return x, meta


def groups(meta, with_label):
    names = ("rx_i", "day_i", "y") if with_label else ("rx_i", "day_i")
    distinct, membership = np.unique(np.stack([meta[key] for key in names], axis=1),
                                     axis=0, return_inverse=True)
    return [np.flatnonzero(membership == i) for i in range(len(distinct))]


def sample(meta, total, with_label):
    pools = groups(meta, with_label)
    per_group, rest = divmod(total, len(pools))
    if rest or per_group <= 0 or any(len(pool) < per_group for pool in pools):
        raise ValueError("The fixed count does not fit every stratum")
    rng = np.random.default_rng(SPLIT_SEED)
    return np.sort(np.concatenate([rng.choice(pool, per_group, replace=False) for pool in pools]))


def select(x, meta, index):
    return np.asarray(x[index], dtype=np.float32), {key: value[index] for key, value in meta.items()}


def get_roles(row, inputs):
    if row["input_cache"] == "by_date":
        result = {}
        for role in ROLES:
            chunks = []
            for day in row["source_days"]:
                cache = inputs["chrono" if day == 0 else "F0"]
                x, meta = load_role(cache, role)
                chunks.append(select(x, meta, meta["day_i"] == day))
            result[role] = (np.concatenate([x for x, _ in chunks]),
                            {k: np.concatenate([m[k] for _, m in chunks]) for k in chunks[0][1]})
        return result

    cache = inputs[row["input_cache"]]
    if row["family"] == "label_budget":
        x, meta = load_role(cache, "L_s")
        l_idx = sample(meta, row["counts"]["L_s"], with_label=True)
        v_idx = np.setdiff1d(np.arange(len(x)), l_idx, assume_unique=True)
        # The entire label budget comes from the original L. Never open the
        # old 27000-label validation set, or recode a hidden U label.
        return {"L_s": select(x, meta, l_idx), "V": select(x, meta, v_idx),
                "U_s": load_role(cache, "U_s")}

    result = {}
    for role in ROLES:
        x, meta = load_role(cache, role)
        mask = np.isin(meta["rx_i"], row["source_rxs"])
        x, meta = select(x, meta, mask)
        result[role] = select(x, meta, sample(meta, row["counts"][role], role != "U_s"))
    return result


def make_cache(row, inputs, output, classes):
    data = get_roles(row, inputs)
    domain_map = [(rx, day) for rx in row["source_rxs"] for day in row["source_days"]]
    unique = set()
    role_ids = {}
    for role, (x, meta) in data.items():
        if (x.shape != (row["counts"][role], 2, 256)
                or set(meta) != METADATA | (LABELS if role != "U_s" else set())
                or set(zip(meta["rx_i"], meta["day_i"])) != set(domain_map)):
            raise ValueError(f"Bad role coverage in {row['setting']}/{role}")
        ids = meta["ids"].tolist()
        if len(set(ids)) != len(ids) or unique.intersection(ids):
            raise ValueError("Duplicate physical IDs within or across source roles")
        unique.update(ids)
        role_ids[role] = ids
        lookup = {pair: i for i, pair in enumerate(domain_map)}
        meta["domain"] = np.asarray([lookup[pair] for pair in zip(meta["rx_i"], meta["day_i"])])
    output.mkdir(parents=True, exist_ok=False)
    for role, (x, meta) in data.items():
        np.save(output / f"{role}_x.npy", x, allow_pickle=False)
        np.savez(output / f"{role}_metadata.npz", **meta)
    contract = dict(schema="pamf_advisor_data_v1", dataset="ManySig", classes=classes,
                    num_classes=len(classes), source_rxs=row["source_rxs"], source_days=row["source_days"],
                    counts=row["counts"], role_ids=role_ids, equalized=1, out_len=256, normalize=True,
                    split_seed=SPLIT_SEED, checkpoint_init="scratch_only",
                    target_access_before_freeze=False, U_TX_truth_read=False,
                    domain_map=[[*pair, i] for i, pair in enumerate(domain_map)])
    write(output / "source_contract.json", contract)
