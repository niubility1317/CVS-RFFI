"""RIEI's two alternating source-only updates, without satellite imports."""
import torch
import torch.nn.functional as F
from baselines.riei_fd.losses import mutual_independence_loss, entropy_from_logits


def alternating_training_step(model, batch, optimizer_all, optimizer_fed,
        lambda_mi=1.2, lambda_ie=1.2, device='cpu', grad_clip_norm=0., disentangle_steps=1):
    if grad_clip_norm != 0 or disentangle_steps != 1:
        raise ValueError('Registered RIEI uses no clipping and one disentangling step')
    x, y, rx = batch['iq'].to(device), batch['label'].to(device), batch['receiver_target'].to(device)
    for module in (model.ec, model.rc):
        for parameter in module.parameters(): parameter.requires_grad_(True)
    output=model(x)
    classification=F.cross_entropy(output['emitter_logits'],y)+F.cross_entropy(output['receiver_logits'],rx)
    optimizer_all.zero_grad();optimizer_fed.zero_grad();classification.backward()
    if not torch.isfinite(classification) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise FloatingPointError('Nonfinite RIEI classification update')
    optimizer_all.step();optimizer_fed.step()
    # Classifier parameters are fixed only inside this same minibatch update;
    # their original weights remain jointly trained through the CE step above.
    for module in (model.ec, model.rc):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    output=model(x)
    mi=mutual_independence_loss(output['z_e'],output['z_r'],mode='cosine_abs',reduction='mean')
    ie=entropy_from_logits(output['cross_emitter_logits'])+entropy_from_logits(output['cross_receiver_logits'])
    dis=lambda_mi*mi-lambda_ie*ie
    optimizer_fed.zero_grad();dis.backward()
    if not torch.isfinite(dis) or any(not torch.isfinite(p.grad).all() for p in model.fed.parameters() if p.grad is not None):
        raise FloatingPointError('Nonfinite RIEI disentanglement update')
    optimizer_fed.step()
    for module in (model.ec, model.rc):
        for parameter in module.parameters(): parameter.requires_grad_(True)
    return dict(loss=float((classification.detach()+dis.detach()).cpu()),
                loss_ce=float(classification.detach().cpu()),loss_mi=float(mi.detach().cpu()),loss_ie=float(ie.detach().cpu()))
