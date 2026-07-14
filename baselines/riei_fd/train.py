from __future__ import annotations

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from baselines.riei_fd.losses import mutual_independence_loss, entropy_from_logits, riei_total_loss


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(enabled)


def _feature_norm_penalty(z_e: torch.Tensor, z_r: torch.Tensor) -> torch.Tensor:
    return z_e.square().mean() + z_r.square().mean()


def _clip_grad(parameters, grad_clip_norm: float) -> torch.Tensor:
    if float(grad_clip_norm) <= 0.0:
        return torch.tensor(0.0)
    return torch.nn.utils.clip_grad_norm_(parameters, float(grad_clip_norm))


def alternating_training_step(
    model,
    batch,
    optimizer_all,
    optimizer_fed,
    lambda_mi=1.2,
    lambda_ie=1.2,
    device="cpu",
    mi_mode: str = "cosine_abs",
    ie_temperature: float = 1.0,
    ce_reduction: str = "mean",
    mi_reduction: str = "mean",
    ie_reduction: str = "mean",
    disentangle_steps: int = 1,
    grad_clip_norm: float = 0.0,
    lambda_feature_norm: float = 0.0,
):
    x, y = batch["iq"].to(device), batch["label"].to(device)
    d = batch["receiver_target"].to(device) if "receiver_target" in batch else batch["receiver"].to(device)
    set_requires_grad(model.ec, True)
    set_requires_grad(model.rc, True)
    out = model(x)
    loss_ce = riei_total_loss(
        out,
        y,
        d,
        lambda_mi=0.0,
        lambda_ie=0.0,
        ce_reduction=ce_reduction,
        mi_reduction=mi_reduction,
        ie_reduction=ie_reduction,
    )["loss_ce"]
    loss_feature_norm = _feature_norm_penalty(out["z_e"], out["z_r"])
    loss_all = loss_ce
    if float(lambda_feature_norm) > 0.0:
        loss_all = loss_all + float(lambda_feature_norm) * loss_feature_norm
    optimizer_all.zero_grad()
    loss_all.backward()
    grad_norm_all = _clip_grad(model.parameters(), grad_clip_norm)
    optimizer_all.step()

    set_requires_grad(model.ec, False)
    set_requires_grad(model.rc, False)
    dis_metrics = []
    grad_norm_fed = torch.tensor(0.0)
    for _ in range(max(1, int(disentangle_steps))):
        out = model(x)
        loss_mi = mutual_independence_loss(out["z_e"], out["z_r"], mode=mi_mode, reduction=mi_reduction)
        loss_ie = entropy_from_logits(
            out["cross_emitter_logits"],
            temperature=ie_temperature,
            reduction=ie_reduction,
        ) + entropy_from_logits(
            out["cross_receiver_logits"],
            temperature=ie_temperature,
            reduction=ie_reduction,
        )
        loss_dis = float(lambda_mi) * loss_mi - float(lambda_ie) * loss_ie
        loss_dis_total = loss_dis
        loss_feature_norm = _feature_norm_penalty(out["z_e"], out["z_r"])
        if float(lambda_feature_norm) > 0.0:
            loss_dis_total = loss_dis_total + float(lambda_feature_norm) * loss_feature_norm
        optimizer_fed.zero_grad()
        loss_dis_total.backward()
        grad_norm_fed = _clip_grad(model.fed.parameters(), grad_clip_norm)
        optimizer_fed.step()
        dis_metrics.append(
            {
                "loss_mi": loss_mi.detach(),
                "loss_ie": loss_ie.detach(),
                "loss_dis": loss_dis.detach(),
                "loss_feature_norm": loss_feature_norm.detach(),
            }
        )
    set_requires_grad(model.ec, True)
    set_requires_grad(model.rc, True)
    loss_mi = torch.stack([m["loss_mi"] for m in dis_metrics]).mean()
    loss_ie = torch.stack([m["loss_ie"] for m in dis_metrics]).mean()
    loss_dis = torch.stack([m["loss_dis"] for m in dis_metrics]).mean()
    loss_feature_norm = torch.stack([m["loss_feature_norm"] for m in dis_metrics]).mean()
    loss_total = loss_ce + loss_dis
    if float(lambda_feature_norm) > 0.0:
        loss_total = loss_total + float(lambda_feature_norm) * loss_feature_norm
    return {
        "loss": float(loss_total.detach().cpu()),
        "loss_ce": float(loss_ce.detach().cpu()),
        "loss_mi": float(loss_mi.detach().cpu()),
        "loss_ie": float(loss_ie.detach().cpu()),
        "loss_dis": float(loss_dis.detach().cpu()),
        "loss_feature_norm": float(loss_feature_norm.detach().cpu()),
        "grad_norm_all": float(grad_norm_all.detach().cpu()),
        "grad_norm_fed": float(grad_norm_fed.detach().cpu()),
    }


def main() -> None:
    from baselines.riei_fd.train_cvs import main as cvs_main

    cvs_main()


if __name__ == "__main__":
    main()
