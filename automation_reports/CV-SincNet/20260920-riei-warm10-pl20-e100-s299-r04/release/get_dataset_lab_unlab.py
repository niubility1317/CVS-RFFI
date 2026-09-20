"""Dataset loading helpers for STAR-RFFI clean-train / sg-test experiments."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


DEFAULT_DATA_ROOT = str(Path(__file__).resolve().parent / "data" / "npy_tf2048_sg")


def _parse_rx_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    return [int(v) for v in value]


def _available_rxs(data_root):
    root = Path(data_root)
    rxs = []
    for path in root.glob("rx_*_y.npy"):
        parts = path.stem.split("_")
        if len(parts) >= 3:
            try:
                rxs.append(int(parts[1]))
            except ValueError:
                pass
    return sorted(set(rxs))


def _default_source_rxs(data_root, target_rx):
    available = _available_rxs(data_root)
    if available:
        candidates = [rx for rx in available if target_rx is None or rx != target_rx]
        if len(candidates) >= 4:
            return candidates[:4]
        return candidates
    defaults = [0, 1, 2, 3, 4, 5]
    return [rx for rx in defaults if target_rx is None or rx != target_rx][:4]


def _load_receiver(data_root, rx, variant, num=10):
    root = Path(data_root)
    x_path = root / f"rx_{rx}_{variant}_x.npy"
    y_path = root / f"rx_{rx}_y.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"missing data file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"missing label file: {y_path}")

    x = np.load(x_path).astype(np.float32, copy=False)
    y = np.load(y_path).astype(np.int64, copy=False)
    if x.ndim != 3 or x.shape[1:] != (2, 2048):
        raise ValueError(f"{x_path} must have shape (N, 2, 2048), got {x.shape}")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError(f"{y_path} must have shape (N,), got {y.shape} for X {x.shape}")

    mask = y < int(num)
    if not np.all(mask):
        print(
            f"WARNING: rx={rx} variant={variant} filtered "
            f"{np.count_nonzero(~mask)} samples with tx label >= {num}"
        )
        x = x[mask]
        y = y[mask]
    return x, y


def _parse_variants(value):
    if value is None:
        return ["clean", "sg"]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def ReadGeneratedNpyFiles(
    num=10,
    data_root=DEFAULT_DATA_ROOT,
    rx_indices=None,
    variants=("clean", "sg"),
    mmap_mode="r",
    require_pair_index=True,
    check_clean_sg_pair=True,
    finite_check=False,
):
    """Read generated npy files first and validate the fixed training inputs.

    This is a lightweight preflight loader for the files created by
    ``data_get.py``.  It opens the requested clean/sg arrays, labels, domain
    labels, and pair maps before training starts, so later calls use a fixed
    and traceable npy dataset.  Large X arrays default to memory mapping; set
    ``mmap_mode=None`` if a caller really wants to load them fully into RAM.
    """

    root = Path(data_root)
    rx_indices = _parse_rx_list(rx_indices)
    if rx_indices is None:
        rx_indices = _available_rxs(root)
    if not rx_indices:
        raise FileNotFoundError(f"no rx_*_y.npy files found in {root}")

    variants = _parse_variants(variants)
    for variant in variants:
        if variant not in {"clean", "sg"}:
            raise ValueError(f"variant must be 'clean' or 'sg', got {variant}")

    manifest = {
        "data_root": str(root),
        "rx_indices": [int(rx) for rx in rx_indices],
        "variants": variants,
        "receivers": {},
    }

    print("=" * 80)
    print("ReadGeneratedNpyFiles preflight")
    print(f"data_root={root}")
    print(f"rx_indices={rx_indices}")
    print(f"variants={variants}")
    print(f"mmap_mode={mmap_mode}")

    for rx in rx_indices:
        y_path = root / f"rx_{rx}_y.npy"
        domain_path = root / f"rx_{rx}_domain.npy"
        pair_path = root / f"rx_{rx}_pair_index.npy"
        if not y_path.exists():
            raise FileNotFoundError(f"missing label file: {y_path}")
        if not domain_path.exists():
            raise FileNotFoundError(f"missing domain file: {domain_path}")
        if require_pair_index and not pair_path.exists():
            raise FileNotFoundError(f"missing pair index file: {pair_path}")

        y = np.load(y_path).astype(np.int64, copy=False)
        domain = np.load(domain_path).astype(np.int64, copy=False)
        if y.ndim != 1:
            raise ValueError(f"{y_path} must have shape (N,), got {y.shape}")
        if domain.shape != y.shape:
            raise ValueError(
                f"{domain_path} shape {domain.shape} does not match y {y.shape}"
            )

        pair_index = None
        if pair_path.exists():
            pair_index = np.load(pair_path).astype(np.int64, copy=False)
            if pair_index.ndim != 2 or pair_index.shape[1] != 4:
                raise ValueError(
                    f"{pair_path} must have shape (N, 4), got {pair_index.shape}"
                )
            if pair_index.shape[0] != y.shape[0]:
                raise ValueError(
                    f"{pair_path} rows {pair_index.shape[0]} do not match y {y.shape[0]}"
                )
            if not np.array_equal(pair_index[:, 0], y):
                raise ValueError(
                    f"{pair_path} first column must match tx labels in {y_path}"
                )

        receiver_entry = {
            "y": y,
            "domain": domain,
            "pair_index": pair_index,
            "x": {},
            "paths": {
                "y": str(y_path),
                "domain": str(domain_path),
                "pair_index": str(pair_path) if pair_path.exists() else None,
            },
        }

        variant_shapes = {}
        for variant in variants:
            x_path = root / f"rx_{rx}_{variant}_x.npy"
            if not x_path.exists():
                raise FileNotFoundError(f"missing data file: {x_path}")
            x = np.load(x_path, mmap_mode=mmap_mode)
            if x.ndim != 3 or x.shape[1:] != (2, 2048):
                raise ValueError(f"{x_path} must have shape (N, 2, 2048), got {x.shape}")
            if x.shape[0] != y.shape[0]:
                raise ValueError(
                    f"{x_path} rows {x.shape[0]} do not match y rows {y.shape[0]}"
                )
            if finite_check and not np.isfinite(np.asarray(x)).all():
                raise FloatingPointError(f"{x_path} contains nan or inf")
            receiver_entry["x"][variant] = x
            receiver_entry["paths"][variant] = str(x_path)
            variant_shapes[variant] = tuple(x.shape)

        if check_clean_sg_pair and {"clean", "sg"}.issubset(set(variants)):
            if variant_shapes["clean"] != variant_shapes["sg"]:
                raise ValueError(
                    f"rx={rx} clean shape {variant_shapes['clean']} != "
                    f"sg shape {variant_shapes['sg']}"
                )

        manifest["receivers"][int(rx)] = receiver_entry
        print(
            f"rx={rx}: y={y.shape}, domain={domain.shape}, "
            f"pair={None if pair_index is None else pair_index.shape}, "
            f"tx counts={_count_by_tx(y, num)}, x shapes={variant_shapes}"
        )

    return manifest


read_generated_npy_files = ReadGeneratedNpyFiles


def _count_by_tx(y, num):
    return np.bincount(y.astype(int), minlength=int(num)).astype(int)


def _split_domain_indices(y, labeled_ratio, val_ratio, seed, num):
    """Stratify each source domain by tx label.

    Per class, the first split reserves the labeled portion (default 10%) and
    leaves the remaining samples as unlabeled.  Validation is then cut only from
    that labeled clean source portion, so target sg data never participates in
    training, validation, or early stopping.
    """

    rng = np.random.default_rng(seed)
    train_label_idx = []
    val_idx = []
    unlabel_idx = []

    for tx in range(int(num)):
        idx = np.flatnonzero(y == tx)
        if idx.size == 0:
            continue
        rng.shuffle(idx)
        labeled_count = int(round(idx.size * float(labeled_ratio)))
        if labeled_ratio > 0:
            labeled_count = max(1, labeled_count)
        labeled_count = min(labeled_count, idx.size)

        if labeled_count <= 1 or val_ratio <= 0:
            val_count = 0
        else:
            val_count = int(round(labeled_count * float(val_ratio)))
            val_count = max(1, val_count)
            val_count = min(val_count, labeled_count - 1)

        domain_label_idx = idx[: labeled_count - val_count]
        domain_val_idx = idx[labeled_count - val_count : labeled_count]
        domain_unlabel_idx = idx[labeled_count:]

        train_label_idx.extend(domain_label_idx.tolist())
        val_idx.extend(domain_val_idx.tolist())
        unlabel_idx.extend(domain_unlabel_idx.tolist())

    for bucket in (train_label_idx, val_idx, unlabel_idx):
        rng.shuffle(bucket)
    return (
        np.asarray(train_label_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(unlabel_idx, dtype=np.int64),
    )


def _make_y2(y, domain_label):
    domains = np.full_like(y, int(domain_label), dtype=np.int64)
    return np.column_stack([y.astype(np.int64), domains])


def _concat_or_empty(arrays, trailing_shape, dtype):
    if arrays:
        return np.concatenate(arrays, axis=0).astype(dtype, copy=False)
    return np.empty((0,) + tuple(trailing_shape), dtype=dtype)


def _print_counts(prefix, y2, num):
    if y2.size == 0:
        print(f"{prefix}: 0 samples")
        return
    print(
        f"{prefix}: {y2.shape[0]} samples, tx counts="
        f"{_count_by_tx(y2[:, 0], num)}"
    )


def TrainDataset(
    num=10,
    rand_num=299,
    data_root=DEFAULT_DATA_ROOT,
    source_rxs=None,
    target_rx=None,
    train_variant="clean_sg",
    val_variant=None,
    labeled_ratio=0.1,
    val_ratio=0.1,
    seed=299,
    split_dir=None,
):
    """Load source receivers for training and validation.

    Source receivers are the training domains and are remapped to 0..M-1.
    By default, source training/validation uses clean data.  The target
    receiver is printed for traceability only; target data is loaded by
    ``TestDataset`` and must not be used for early stopping.
    """

    if not 0 < labeled_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("labeled_ratio and val_ratio must be between zero and one")
    if seed is None:
        seed = rand_num
    source_rxs = _parse_rx_list(source_rxs)
    if target_rx is None:
        target_rx = 4
    if source_rxs is None:
        source_rxs = [rx for rx in range(6) if rx != target_rx]
    if not source_rxs or len(set(source_rxs)) != len(source_rxs) or target_rx in source_rxs:
        raise ValueError("source receivers must be unique, nonempty and exclude target")
    if val_variant is None:
        val_variant = "clean"
    if train_variant not in {"clean", "sg", "clean_sg"}:
        raise ValueError("train_variant must be clean, sg or clean_sg")
    if val_variant not in {"clean", "sg"}:
        raise ValueError("val_variant must be 'clean' or 'sg'")

    domain_mapping = {int(rx): domain for domain, rx in enumerate(source_rxs)}
    print("=" * 80)
    print("TrainDataset")
    print(f"data_root={data_root}")
    print(f"source_rxs={source_rxs}")
    print(f"target_rx={target_rx}")
    print(f"train_variant={train_variant}")
    print(f"val_variant={val_variant}")
    print(f"num_domains={len(source_rxs)}")
    print(f"domain mapping={domain_mapping}")

    x_label_parts, x_unlabel_parts, x_train_parts, x_val_parts = [], [], [], []
    y_label_parts, y_unlabel_parts, y_train_parts, y_val_parts = [], [], [], []

    for rx in source_rxs:
        domain_label = domain_mapping[int(rx)]
        base_variant = "clean" if train_variant == "clean_sg" else train_variant
        x, y = _load_receiver(data_root, rx, base_variant, num=num)
        x_val_source = x
        x_sg = None
        if train_variant == "clean_sg":
            x_sg, y_sg = _load_receiver(data_root, rx, "sg", num=num)
            if x.shape != x_sg.shape or not np.array_equal(y, y_sg):
                raise ValueError(f"rx={rx}: clean/sg shapes or labels differ")
            pairs = np.load(Path(data_root) / f"rx_{rx}_pair_index.npy")
            if pairs.shape != (len(y), 4) or not np.array_equal(pairs[:, 0], y):
                raise ValueError(f"rx={rx}: invalid pair map")
            identities = np.column_stack([np.repeat(pairs[:, 0], 2), pairs[:, 2:4].reshape(-1)])
            if len(np.unique(identities, axis=0)) != len(identities):
                raise ValueError(f"rx={rx}: raw IQ segments reused by multiple pairs")
        if val_variant != base_variant:
            x_val_source, y_val_source = _load_receiver(
                data_root, rx, val_variant, num=num
            )
            if not np.array_equal(y, y_val_source):
                raise ValueError(
                    f"rx={rx} train/val labels differ between "
                    f"{train_variant} and {val_variant}"
                )
        y2 = _make_y2(y, domain_label)
        # Stable audit identity; optimization still reads TX/domain columns only.
        y2 = np.column_stack([y2, np.full(len(y), rx, dtype=np.int64),
                              np.arange(len(y), dtype=np.int64)])
        label_idx, val_idx, unlabel_idx = _split_domain_indices(
            y=y,
            labeled_ratio=labeled_ratio,
            val_ratio=val_ratio,
            seed=int(seed) + int(rx),
            num=num,
        )

        if split_dir is not None:
            Path(split_dir).mkdir(parents=True, exist_ok=True)
            np.savez(Path(split_dir) / f"rx_{rx}_split.npz", labeled=label_idx,
                     validation=val_idx, unlabeled=unlabel_idx)
        if x_sg is None:
            x_label_parts.append(x[label_idx])
            x_unlabel_parts.append(x[unlabel_idx])
        else:
            # Keep each clean/sg pair together in the same minibatch. Split first.
            x_label_parts.append(np.stack([x[label_idx], x_sg[label_idx]], axis=1))
            x_unlabel_parts.append(np.stack([x[unlabel_idx], x_sg[unlabel_idx]], axis=1))
        # Compatibility-only whole-source return; main training does not consume it.
        x_train_parts.append(x)
        x_val_parts.append(x_val_source[val_idx])
        y_label_parts.append(y2[label_idx])
        y_unlabel_parts.append(y2[unlabel_idx])
        y_train_parts.append(y2)
        y_val_parts.append(y2[val_idx])

        print(
            f"source rx={rx} -> domain={domain_label}: "
            f"labeled={label_idx.size}, unlabeled={unlabel_idx.size}, "
            f"val={val_idx.size}, tx counts={_count_by_tx(y, num)}"
        )

    x_shape = (2, 2048)
    train_shape = (2,) + x_shape if train_variant == "clean_sg" else x_shape
    X_train_label = _concat_or_empty(x_label_parts, train_shape, np.float32)
    X_train_unlabel = _concat_or_empty(x_unlabel_parts, train_shape, np.float32)
    X_train = _concat_or_empty(x_train_parts, x_shape, np.float32)
    X_val = _concat_or_empty(x_val_parts, x_shape, np.float32)
    Y_train_label = _concat_or_empty(y_label_parts, (4,), np.int64)
    Y_train_unlabel = _concat_or_empty(y_unlabel_parts, (4,), np.int64)
    Y_train = _concat_or_empty(y_train_parts, (4,), np.int64)
    Y_val = _concat_or_empty(y_val_parts, (4,), np.int64)

    _print_counts("total labeled", Y_train_label, num)
    _print_counts("total unlabeled", Y_train_unlabel, num)
    _print_counts("total val", Y_val, num)

    return (
        X_train_label,
        X_train_unlabel,
        X_train,
        X_val,
        Y_train_label,
        Y_train_unlabel,
        Y_train,
        Y_val,
    )


def TestDataset(
    num=10,
    data_root=DEFAULT_DATA_ROOT,
    target_rx=None,
    test_variant="sg",
):
    """Load the held-out target receiver for final testing."""

    if target_rx is None:
        target_rx = 4
    if test_variant not in {"clean", "sg"}:
        raise ValueError("test_variant must be 'clean' or 'sg'")

    x, y = _load_receiver(data_root, int(target_rx), test_variant, num=num)
    target_domain = np.full_like(y, -1, dtype=np.int64)
    y2 = np.column_stack([y.astype(np.int64), target_domain,
                          np.full(len(y), target_rx, dtype=np.int64),
                          np.arange(len(y), dtype=np.int64)])

    print("=" * 80)
    print("TestDataset")
    print(f"data_root={data_root}")
    print(f"target_rx={target_rx}")
    print(f"test_variant={test_variant}")
    print(f"target test samples={x.shape[0]}")
    print(f"target tx counts={_count_by_tx(y, num)}")
    return x, y2


if __name__ == "__main__":
    TrainDataset()
    TestDataset()
