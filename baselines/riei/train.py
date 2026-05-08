from __future__ import annotations

import torch

from baselines.riei.losses import mutual_independence_loss, entropy_from_logits, riei_total_loss


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(enabled)


def alternating_training_step(model, batch, optimizer_all, optimizer_fed, lambda_mi=0.1, lambda_ie=0.1, device="cpu"):
    x, y, d = batch["iq"].to(device), batch["label"].to(device), batch["receiver"].to(device)
    set_requires_grad(model.ec, True)
    set_requires_grad(model.rc, True)
    out = model(x)
    loss_ce = riei_total_loss(out, y, d, lambda_mi=0.0, lambda_ie=0.0)["loss_ce"]
    optimizer_all.zero_grad()
    loss_ce.backward()
    optimizer_all.step()

    set_requires_grad(model.ec, False)
    set_requires_grad(model.rc, False)
    out = model(x)
    loss_mi = mutual_independence_loss(out["z_e"], out["z_r"])
    loss_ie = entropy_from_logits(out["cross_emitter_logits"]) + entropy_from_logits(out["cross_receiver_logits"])
    loss_dis = float(lambda_mi) * loss_mi - float(lambda_ie) * loss_ie
    optimizer_fed.zero_grad()
    loss_dis.backward()
    optimizer_fed.step()
    set_requires_grad(model.ec, True)
    set_requires_grad(model.rc, True)
    return {"loss": float((loss_ce + loss_dis).detach().cpu()), "loss_ce": float(loss_ce.detach().cpu())}


def main() -> None:
    from baselines.riei.train_cvs import main as cvs_main

    cvs_main()


if __name__ == "__main__":
    main()
