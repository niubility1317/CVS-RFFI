from __future__ import annotations

import os

import torch

from baselines.common.io import default_arg_parser, ensure_dir, get_nested, load_config, save_json, set_seed
from baselines.common.train_utils import synthetic_iq_loader
from baselines.drift.model import DRIFTModel


def main() -> None:
    parser = default_arg_parser("DRIFT evaluation")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device(args.device or cfg.get("device", "cpu"))
    output_dir = ensure_dir(args.output_dir or get_nested(cfg, "output.result_dir", "outputs/drift_eval"))
    num_tx = int(get_nested(cfg, "data.num_tx", 4))
    num_rx = int(get_nested(cfg, "data.num_rx", 3))
    model = DRIFTModel(num_tx=num_tx, num_rx=num_rx, embedding_dim=128, split_dim=64).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
    loader = synthetic_iq_loader(num_tx=num_tx, num_rx=num_rx, batch_size=16, seed=seed, shuffle=False)
    model.eval()
    total = correct = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch["iq"].to(device), grl_lambda=0.0)
            pred = out["tx_logits"].argmax(dim=1).cpu()
            y = batch["label"]
            total += int(y.numel())
            correct += int((pred == y).sum().item())
    save_json({"overall_accuracy": correct / max(1, total)}, os.path.join(output_dir, "eval_results.json"))


if __name__ == "__main__":
    main()
