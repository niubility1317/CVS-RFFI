from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cvsrffi.eval import FCR_PREDICTION_SCENARIOS, apply_sat_channel_for_scenario, select_identity_logits
from cvsrffi.truth_last import build_truth_sidecar, stable_sample_id
from dataset_wisig import WiSigCompactDataset, load_wisig_compact_pkl
from post_stage_common import build_baseline_model, load_checkpoint


class _OpaqueTargetDataset(Dataset):
    """Predictor-only package view: IQ and opaque IDs, with no labels or TX metadata."""

    def __init__(self, package_root: Path):
        self.iq = np.load(package_root / "iq.npy", mmap_mode="r")
        manifest = json.loads((package_root / "manifest.json").read_text(encoding="utf-8"))
        self.sample_ids = [str(value) for value in manifest["sample_ids"]]
        if self.iq.shape != (168000, 2, 256) or len(self.sample_ids) != 168000:
            raise ValueError("predictor package must contain exactly 168000 IQ records")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        x = torch.from_numpy(np.asarray(self.iq[index]).copy())
        return x, -1, 0, {"physical_sample_id": self.sample_ids[index]}


def _ids_and_truth(base: WiSigCompactDataset, split_binding: str):
    for item in base.index:
        physical = f"tx{item.tx_i}:rx{item.rx_i}:day{item.day_i}:eq{item.eq_i}:sig{item.sig_i}"
        yield {"physical_id": physical, "label": int(item.tx_i)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wisig-pkl", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--input-package", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--target-rxs", default="0,2,5,7,9,10,11")
    parser.add_argument("--target-days", default="0,1,2,3")
    parser.add_argument("--split-binding", default="ManySig|tx_rx_day_1_7_2|392005")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=["prepare", "predict"], required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    input_package = Path(args.input_package)
    predictions_path = output_root / "predictions.json"
    truth_path = output_root / "truth_sidecar.json"
    if args.mode == "prepare" and (output_root.exists() or input_package.exists()):
        raise FileExistsError("refusing to overwrite target truth or predictor package")
    if args.mode == "predict" and predictions_path.exists():
        raise FileExistsError(f"refusing to overwrite predictions: {predictions_path}")

    if args.mode == "prepare":
        if not args.wisig_pkl:
            raise ValueError("--wisig-pkl is required only in prepare mode")
        ds = load_wisig_compact_pkl(args.wisig_pkl)
        base = WiSigCompactDataset(
            ds,
            out_len=256,
            crop_mode="center",
            normalize=True,
            equalized=1,
            tx_keep=[0, 1, 2, 3, 4, 5],
            rx_keep=[int(value) for value in args.target_rxs.split(",")],
            day_keep=[int(value) for value in args.target_days.split(",")],
            domain="rx_day",
            seed=392005,
            build_index=True,
        )
        if len(base) != 168000:
            raise ValueError(f"registered target size must be 168000, got {len(base)}")
        output_root.mkdir(parents=True)
        input_package.mkdir(parents=True)
        build_truth_sidecar(
            _ids_and_truth(base, args.split_binding),
            output_path=truth_path,
            split_binding=args.split_binding,
        )
        iq = np.lib.format.open_memmap(
            input_package / "iq.npy", mode="w+", dtype=np.float32, shape=(len(base), 2, 256)
        )
        sample_ids = []
        for index in range(len(base)):
            x, _label, _domain, meta = base[index]
            iq[index] = x.numpy()
            sample_ids.append(
                stable_sample_id(meta["physical_sample_id"], split_binding=args.split_binding)
            )
        iq.flush()
        (input_package / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "cvs.phase1.predictor_package.v1",
                    "record_count": len(sample_ids),
                    "sample_ids": sample_ids,
                    "contains_labels": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(
            f"TARGET_PREPARED records={len(base)} predictor_package={input_package} "
            f"truth_sidecar={truth_path}"
        )
        return
    if not input_package.is_dir():
        raise FileNotFoundError("independent predictor package is missing")
    output_root.mkdir(parents=True)

    checkpoint = load_checkpoint(args.checkpoint, torch.device("cpu"))
    model_args = dict(checkpoint.get("args") or {})
    model_args.update({"input_len": 256, "num_domains": 15, "num_classes": 6, "dataset": "wisig"})
    device = torch.device(args.device)
    model = build_baseline_model(SimpleNamespace(**model_args), device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    loader = DataLoader(
        _OpaqueTargetDataset(input_package),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    sat_args = SimpleNamespace(**model_args)
    records = []
    with torch.no_grad():
        for scenario_index, scenario in enumerate(FCR_PREDICTION_SCENARIOS):
            generator = torch.Generator(device=device)
            generator.manual_seed(int(model_args.get("sat_seed", 2027)) + scenario_index * 1009)
            for x, _masked_y, _domain, meta in loader:
                x = x.to(device, non_blocking=True)
                if scenario != "clean":
                    x, _ = apply_sat_channel_for_scenario(
                        x, scenario, sat_args, gen=generator, return_meta=False
                    )
                outputs = model(x, y_tx=None, grl_lambda=1.0, return_aux=True)
                predicted = select_identity_logits(outputs, model=model).argmax(dim=1).cpu().tolist()
                for sample_id, predicted_class in zip(meta["physical_sample_id"], predicted):
                    records.append(
                        {
                            "sample_id": str(sample_id),
                            "scenario": scenario,
                            "predicted_class": int(predicted_class),
                            "run_id": args.run_id,
                            "row_id": args.row_id,
                        }
                    )
    predictions_path.write_text(
        json.dumps(
            {
                "schema": "cvs.phase1.predictions.v1",
                "checkpoint": str(args.checkpoint),
                "record_count": len(records),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"PREDICTIONS_READY records={len(records)} path={predictions_path}")


if __name__ == "__main__":
    main()
