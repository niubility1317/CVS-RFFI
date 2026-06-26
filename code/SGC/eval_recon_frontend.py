from __future__ import annotations

import argparse
import time
from typing import Any, Mapping

import torch
import torch.nn as nn

from post_stage_cli import add_common_data_args, add_sat_eval_args, str2bool
from post_stage_common import build_standard_data, load_baseline_from_checkpoint, load_yaml_or_json, move_batch, resolve_device, set_seed
from cvsrffi.eval import apply_sat_channel_for_scenario
from cvsrffi.tensors import make_torch_generator
from training_controls import parse_sat_scenarios

from SGC.recon.complex_ops import residual_ratio
from SGC.recon.cx_consistency import CxConsistency
from SGC.recon.cx_resdiff import CxResDiff
from SGC.recon.cx_unet_1d import count_parameters
from SGC.recon.identity_losses import extract_feature


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PhyCon recon frontend before Frozen Base or SGC.")
    parser.add_argument("--config", type=str, default="SGC/configs/recon_cxconsistency_020m.yaml")
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--recon_ckpt", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_kind", type=str, default="consistency", choices=["consistency", "diffusion"])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--rho", type=float, default=0.15)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--amp", type=str2bool, default=True)
    add_common_data_args(parser)
    add_sat_eval_args(parser)
    return parser


class ReconBaseEvalAdapter(nn.Module):
    def __init__(self, recon: nn.Module, base_teacher: nn.Module, *, steps: int = 2, rho: float = 0.15) -> None:
        super().__init__()
        self.recon = recon
        self.base_teacher = base_teacher
        self.steps = int(steps)
        self.rho = float(rho)

    def forward(self, x, y_tx=None, grl_lambda: float = 1.0, return_aux: bool = True, domain_labels=None):
        if isinstance(self.recon, CxConsistency):
            x_hat = self.recon.correct(x, steps=self.steps, rho=self.rho)["x_hat"]
        else:
            x_hat = self.recon.correct(x, rho=self.rho)["x_hat"]
        out = self.base_teacher(x_hat, y_tx=y_tx, grl_lambda=grl_lambda, return_aux=return_aux, domain_labels=domain_labels)
        if torch.is_tensor(out):
            return {"tx_logits": out, "recon_x": x_hat}
        if isinstance(out, Mapping):
            result = dict(out)
            if "tx_logits" not in result:
                for key in ("logits", "base_logits"):
                    if torch.is_tensor(result.get(key)):
                        result["tx_logits"] = result[key]
                        break
            result["recon_x"] = x_hat
            return result
        raise TypeError("Unsupported base teacher output.")


def _build_recon(kind: str) -> nn.Module:
    return CxConsistency() if str(kind) == "consistency" else CxResDiff()


def _load_recon(path: str, kind: str, device) -> nn.Module:
    recon = _build_recon(kind).to(device)
    if path:
        ckpt = torch.load(path, map_location=device)
        recon.load_state_dict(ckpt.get("recon", ckpt), strict=False)
    recon.eval()
    return recon


@torch.no_grad()
def evaluate(args) -> int:
    cfg = load_yaml_or_json(args.config)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    recon = _load_recon(args.recon_ckpt, args.model_kind, device)
    print(f"[RECON-EVAL] kind={args.model_kind} params={count_parameters(recon):,} dry_run={int(bool(args.dry_run))}", flush=True)
    if args.dry_run:
        return 0
    data_ctx = build_standard_data(args, device)
    base_teacher, _, _ = load_baseline_from_checkpoint(args.teacher_ckpt, args, data_ctx, device, freeze=True)
    scenarios = parse_sat_scenarios(getattr(args, "eval_sat_scenarios", "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"))
    sat_gen = make_torch_generator(device, int(args.seed) + 3701)
    for scenario in scenarios:
        correct_raw = correct_recon = total = 0
        ratios = []
        id_cos = []
        t0 = time.time()
        for bi, batch in enumerate(data_ctx["val_loader"]):
            if int(args.eval_max_batches) > 0 and bi >= int(args.eval_max_batches):
                break
            x_clean, labels, _ = move_batch(batch, device)
            y_sat, meta = apply_sat_channel_for_scenario(x_clean, scenario, args, gen=sat_gen, return_meta=True)
            raw_logits = base_teacher(y_sat, y_tx=labels, return_aux=True)
            raw_tx = raw_logits.get("tx_logits", raw_logits.get("logits")) if isinstance(raw_logits, Mapping) else raw_logits
            if isinstance(recon, CxConsistency):
                rec = recon.correct(y_sat, meta=meta, steps=int(args.steps), rho=float(args.rho))
            else:
                rec = recon.correct(y_sat, meta=meta, rho=float(args.rho))
            x_hat = rec["x_hat"]
            recon_logits = base_teacher(x_hat, y_tx=labels, return_aux=True)
            recon_tx = recon_logits.get("tx_logits", recon_logits.get("logits")) if isinstance(recon_logits, Mapping) else recon_logits
            correct_raw += int(raw_tx.argmax(dim=-1).eq(labels).sum())
            correct_recon += int(recon_tx.argmax(dim=-1).eq(labels).sum())
            total += int(labels.numel())
            ratios.append(residual_ratio(x_hat, y_sat).detach())
            z_clean = extract_feature(base_teacher, x_clean)
            z_hat = extract_feature(base_teacher, x_hat)
            id_cos.append(torch.nn.functional.cosine_similarity(z_hat, z_clean, dim=-1).detach())
        ratio_all = torch.cat(ratios) if ratios else torch.zeros(1)
        cos_all = torch.cat(id_cos) if id_cos else torch.zeros(1)
        latency_ms = 1000.0 * (time.time() - t0) / max(1, total)
        print(
            f"[RECON-EVAL] scenario={scenario} raw_base={100.0 * correct_raw / max(1, total):.2f}% "
            f"recon_base={100.0 * correct_recon / max(1, total):.2f}% "
            f"res_mean={float(ratio_all.mean()):.4f} res_p95={float(torch.quantile(ratio_all.float(), 0.95)):.4f} "
            f"id_cos={float(cos_all.mean()):.4f} latency_ms_per_sample={latency_ms:.4f}",
            flush=True,
        )
    _ = cfg
    return 0


def main() -> int:
    return evaluate(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
