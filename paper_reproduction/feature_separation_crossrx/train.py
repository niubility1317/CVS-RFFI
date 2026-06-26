from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from paper_reproduction.common.config import contains_unresolved_placeholder, contains_unspecified, load_json_config
from paper_reproduction.common.wisig_runtime import (
    load_wisig_compact_pkl,
    make_loader,
    make_wisig_drift_day1_split,
    set_seed,
    tx_accuracy,
    write_json,
)
from paper_reproduction.feature_separation_crossrx.losses import feature_separation_loss
from paper_reproduction.feature_separation_crossrx.model import FeatureSeparationNet, build_wisig_fusion_representation


def _run(config: dict, args: argparse.Namespace) -> dict:
    seed = int(args.seed if args.seed is not None else config.get("seed", 1337))
    set_seed(seed)
    device = torch.device(args.device)
    wisig_pkl = str(args.wisig_pkl or config.get("wisig_pkl", ""))
    if not wisig_pkl:
        raise ValueError("wisig_pkl is required for Feature Separation training")
    ds = load_wisig_compact_pkl(wisig_pkl)
    train_ds, val_ds, test_ds, named_tests, named_meta, split_info = make_wisig_drift_day1_split(
        ds,
        equalized=int(config.get("equalized", 1)),
        out_len=256,
        domain="rx",
        day=config.get("day", 0),
        train_rxs=config.get("source_receivers"),
        test_rxs=config.get("target_receivers") or config.get("target_receiver"),
        train_samples_per_combo=int(config.get("train_samples_per_transmitter", 30)),
        val_samples_per_combo=int(config.get("val_samples_per_transmitter", 10)),
        test_samples_per_combo=int(config.get("test_samples_per_transmitter", 20)),
        seed=seed,
    )
    num_tx = len(ds.get("tx_list", []))
    num_rx = max(1, len(ds.get("rx_list", [])))
    model = FeatureSeparationNet(input_channels=3, input_length=256, num_tx=num_tx, num_rx=num_rx).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 0.005)))
    train_loader = make_loader(train_ds, batch_size=int(args.batch_size or config.get("batch_size", 256)), shuffle=True)
    val_loader = make_loader(val_ds, batch_size=int(args.eval_batch_size or config.get("eval_batch_size", 256)), shuffle=False)
    test_loader = make_loader(test_ds, batch_size=int(args.eval_batch_size or config.get("eval_batch_size", 256)), shuffle=False)
    epochs = int(args.epochs or config.get("epochs", 1))
    max_steps = int(args.max_steps or config.get("max_steps", 0))
    terms_last = {}
    global_step = 0
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            x = build_wisig_fusion_representation(batch["iq"].to(device))
            y = batch["label"].to(device)
            r = batch["domain"].to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss, terms = feature_separation_loss(
                out,
                y,
                r,
                lambda_similarity=float(config.get("lambda_similarity", 1.0)),
                lambda_tx_entropy=float(config.get("lambda_tx_entropy", 1.0)),
                lambda_rx_entropy=float(config.get("lambda_rx_entropy", 1.0)),
            )
            loss.backward()
            opt.step()
            terms_last = {k: float(v) for k, v in terms.items()}
            global_step += 1
            if max_steps and global_step >= max_steps:
                break
        if max_steps and global_step >= max_steps:
            break

    def evaluate(loader):
        model.eval()
        accs = []
        with torch.no_grad():
            for batch in loader:
                x = build_wisig_fusion_representation(batch["iq"].to(device))
                y = batch["label"].to(device)
                accs.append(tx_accuracy(model(x)["tx_logits"], y))
        return float(sum(accs) / max(1, len(accs)))

    result = {
        "baseline": "feature_separation_crossrx",
        "seed": seed,
        "steps": global_step,
        "last_terms": terms_last,
        "val_acc": evaluate(val_loader),
        "test_acc": evaluate(test_loader),
        "split_info": split_info,
        "named_test_meta": named_meta,
    }
    out_dir = Path(args.run_dir)
    write_json(out_dir / "metrics.json", result)
    write_json(out_dir / "resolved_config.json", config)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-original Feature Separation WiSig entrypoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--wisig-pkl", default="")
    parser.add_argument("--run-dir", default="runs/paper_reproduction_feature_separation")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_json_config(args.config)
    if args.formal and contains_unspecified(config):
        raise ValueError("formal Feature Separation config still contains paper-unspecified")
    if args.formal and contains_unresolved_placeholder(config):
        raise ValueError("formal Feature Separation config still contains unresolved placeholder")
    if args.dry_run:
        print(json.dumps({"baseline": "feature_separation_crossrx", "config": config}, ensure_ascii=False, sort_keys=True))
        return 0
    result = _run(config, args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
