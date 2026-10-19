"""Materialize native, already-validated Phase1 source roles for local use.

No split changes, target/truth reads, checkpoint loading or model training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset


class SourceRoleDataset(Dataset):
    """Memory-mapped source IQ; U has no TX labels or TX metadata on disk."""

    def __init__(self, cache: str | Path, role: str):
        if role not in ("L_s", "U_s", "V"):
            raise ValueError("Only source L_s, U_s and V are permitted")
        self.role = role
        root = Path(cache)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "READY" or manifest["native_role_ids"] != "EXACT_MATCH":
            raise ValueError("Incomplete source cache")
        folder = root / role
        self.x = np.load(folder / "iq.npy", mmap_mode="r", allow_pickle=False)
        self.domain = np.load(folder / "domain.npy", mmap_mode="r", allow_pickle=False)
        self.rx = np.load(folder / "rx_i.npy", mmap_mode="r", allow_pickle=False)
        self.day = np.load(folder / "day_i.npy", mmap_mode="r", allow_pickle=False)
        self.ids = np.load(folder / "physical_ids.npy", mmap_mode="r", allow_pickle=False)
        self.y = None if role == "U_s" else np.load(folder / "labels.npy", mmap_mode="r", allow_pickle=False)
        if role == "U_s" and (folder / "labels.npy").exists():
            raise ValueError("U source cache must not contain TX labels")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return (torch.from_numpy(np.array(self.x[index], copy=True)),
                -1 if self.y is None else int(self.y[index]), int(self.domain[index]),
                dict(rx_i=int(self.rx[index]), day_i=int(self.day[index]),
                     physical_id=str(self.ids[index]), split_source=self.role,
                     tx_label_visible=self.role != "U_s"))


def prepare(dataset_root: Path, output: Path) -> dict:
    # Add the exact native implementation from the selected checkout.
    checkout = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(checkout), str(checkout / "code")]
    from experiments.cvs_f_supcon50 import design
    from experiments.cvs_phase1_stack.runtime import installed

    contract = json.loads((dataset_root / "source" / "source_contract.json").read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError("Preserve existing cache; use its manifest or a fresh output directory")
    config = design.config(next(r for r in design.training_rows() if r["arm"] == "CROSS_RX"))
    args = design.make_args(config, "cpu")
    args.wisig_pkl = str(dataset_root / "raw" / "ManySig.pkl")
    if not args.a1_source_screen_only or args.num_workers != 0:
        raise ValueError("Native source-only loader with zero subprocess workers required")
    torch.set_num_threads(2)
    with installed(config) as native:
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        from cvsrffi.game_tracking.data import opaque_id
        ctx = native._build_ssdg_wisig_data(args, torch.device("cpu"))
        if ctx["named_test_loaders"] or role_ids_from_native(ctx) != contract["role_ids"]:
            raise ValueError("Original physical roles differ or target was constructed")
        output.mkdir(parents=True)
        roles = {}
        for role, loader_name in (("L_s", "train_loader"), ("U_s", "unlabeled_loader"), ("V", "val_loader")):
            ds = ctx[loader_name].dataset
            if not hasattr(ds, "index"):
                ds = ds.base
            count = len(ds)
            folder = output / role
            folder.mkdir()
            arrays = {name: np.lib.format.open_memmap(folder / filename, mode="w+", dtype=dtype, shape=shape)
                      for name, filename, dtype, shape in
                      (("iq", "iq.npy", np.float32, (count, 2, 256)),
                       ("domain", "domain.npy", np.int64, (count,)),
                       ("rx", "rx_i.npy", np.int64, (count,)),
                       ("day", "day_i.npy", np.int64, (count,)),
                       ("ids", "physical_ids.npy", "U24", (count,)))}
            if role != "U_s":
                arrays["y"] = np.lib.format.open_memmap(folder / "labels.npy", mode="w+", dtype=np.int64, shape=(count,))
            for index in range(count):
                iq, label, domain, meta = ds[index]
                if tuple(iq.shape) != (2, 256) or not torch.isfinite(iq).all():
                    raise ValueError("Malformed native source IQ")
                if role == "U_s" and (label != -1 or any(k in meta for k in ("tx", "tx_i", "true_tx_i", "tx_label", "y_tx"))):
                    raise ValueError("Native U view exposed a TX label")
                arrays["iq"][index] = iq.numpy()
                arrays["domain"][index] = domain
                arrays["rx"][index] = meta["rx_i"]
                arrays["day"][index] = meta["day_i"]
                arrays["ids"][index] = opaque_id(ds.index[index])
                if role != "U_s":
                    arrays["y"][index] = label
            for array in arrays.values():
                array.flush()
            roles[role] = dict(count=count, labels_visible=role != "U_s", input_shape=[count, 2, 256])
            print(json.dumps(dict(event="SOURCE_ROLE_EXPORTED", role=role, count=count)), flush=True)
        manifest = dict(status="READY", schema="local_native_source_roles_v1", roles=roles,
                        native_role_ids="EXACT_MATCH", source_receivers=contract["source_rxs"],
                        source_days=contract["source_days"], class_id_to_tx=ctx["class_id_to_tx"],
                        num_domains=ctx["num_domains"], domain_label_map=ctx["domain_label_map"],
                        preprocessing="Original native WiSigCompactDataset center crop/RMS normalization, no training augmentation",
                        data_validation="Reuse existing validated contract; local role reconstruction matched exactly",
                        U_TX_labels_exported=False, target_loaders_constructed=False,
                        target_reads=0, truth_reads=0, checkpoint_reads=0, optimizer_updates=0)
        # JSON domain maps may contain tuple keys in the native implementation.
        manifest["domain_label_map"] = [[list(k) if isinstance(k, tuple) else k, v] for k, v in ctx["domain_label_map"].items()]
        with (output / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
    # Fresh readback through the public mmap dataset, not the export objects.
    for role in roles:
        ds = SourceRoleDataset(output, role)
        if len(ds) != contract["counts"][role] or sorted(ds.ids.tolist()) != contract["role_ids"][role]:
            raise ValueError("Readback source role mismatch")
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=128, num_workers=0)))
        if batch[0].shape != (128, 2, 256) or (role == "U_s" and (batch[1] != -1).any()):
            raise ValueError("Source cache cannot produce legal native batches")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset_root.resolve(), args.output.resolve()), ensure_ascii=False), flush=True)
