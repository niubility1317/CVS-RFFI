"""Execution-only acceleration of the existing ISSL target training recipe."""
import copy
import random
import torch
from torch.nn import functional as F

from . import external_additional as extra


def checked_step(loss, optimizer):
    """Retain finite-loss/gradient checks with one gradient synchronization."""
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite native training loss')
    loss.backward()
    grads = [p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
    if grads:
        norms = torch._foreach_norm(grads, float('inf'))
        if not torch.isfinite(torch.stack(norms)).all():
            raise FloatingPointError('Nonfinite native gradient')
    optimizer.step()


def issl_input_validated(x, crop=None):
    # Caller validates the complete immutable source/support tensor once.
    if crop is not None:
        starts = torch.randint(0, x.shape[-1] - crop + 1, (len(x),), device=x.device)
        x = x.gather(2, (starts[:, None] + torch.arange(crop, device=x.device))[..., None].transpose(1, 2).expand(-1, 2, -1))
    return (x-x.mean((1, 2), keepdim=True))/x.std((1, 2), correction=0, keepdim=True).clamp_min(1e-12)


def _mean_loss(values):
    # Same Python summation as the original, but a single device-to-host copy.
    return sum(torch.stack(values).detach().cpu().tolist())/len(values)


def issl_ssl(model, x, old_classes, spec, seed, logger):
    dev=extra._device(model); author=model.profile=='issl_author_behavior'
    key=copy.deepcopy(model).eval().requires_grad_(False)
    x=extra._iq(x).to(dev)
    gen=torch.Generator().manual_seed(seed)
    size=random.randint(1,min(1000,len(x)))
    first=torch.randperm(len(x),generator=gen)[:size].to(dev)
    with torch.no_grad():queue=key.raw(x[first],218)[0].detach()
    if not author:queue=F.normalize(queue,dim=1,eps=1e-12)
    opt=torch.optim.Adam(model.parameters(),lr=spec['C_SSL_lr'])
    scheduler=torch.optim.lr_scheduler.StepLR(opt,100,.8)
    key_params=list(key.parameters());model_params=list(model.parameters())
    for epoch in range(spec['C_SSL_epochs']):
        values=[];model.train()
        for ix in extra._indices(len(x),64,gen):
            xx=x[ix.to(dev)]
            qx=issl_input_validated(xx,218);kx=issl_input_validated(xx,218);q=model.network(qx)[0]
            with torch.no_grad():k=key.network(kx)[0];t=key.network(qx)[0]
            kd=extra._kd(q[:,:old_classes],t[:,:old_classes],20)
            if author:kd=kd.detach().requires_grad_(True)
            q=F.normalize(q,dim=1,eps=1e-12);k=F.normalize(k,dim=1,eps=1e-12)
            negative=q@(queue.reshape(queue.shape[1],queue.shape[0]) if author else queue.T)
            contrast=F.cross_entropy(torch.cat(((q*k).sum(1,keepdim=True),negative),1)/5,
                torch.zeros(len(q),dtype=torch.long,device=dev))
            loss=.5*kd+.5*contrast
            if not author:opt.zero_grad(set_to_none=True)
            checked_step(loss,opt);values.append(loss.detach())
            queue=torch.cat((queue,k.detach()),0)
            queue=queue[64:] if author and len(queue)>1000 else queue[-1000:]
            if not author:queue=F.normalize(queue,dim=1,eps=1e-12)
            with torch.no_grad():
                torch._foreach_mul_(key_params,.99)
                torch._foreach_add_(key_params,model_params,alpha=.01)
                if not author:
                    for kb,mb in zip(key.buffers(),model.buffers()):
                        if kb.is_floating_point():kb.mul_(.99).add_(mb,alpha=.01)
                        else:kb.copy_(mb)
        scheduler.step()
        extra._log(logger,stage='C_ssl',epoch=epoch+1,loss=_mean_loss(values),queue_size=len(queue),
            kd_student_gradient=not author,zero_grad=not author,query_fit=False)


def issl_transfer(model, x, y, epochs, batch, lr, seed, logger=None, crop=None):
    dev=extra._device(model)
    x=extra._iq(x).to(dev);y=y.to(dev)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,eps=1e-8)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,100,.5)
    gen=torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        values=[];model.train()
        for ix in extra._indices(len(y),batch,gen):
            ix=ix.to(dev);xx,yy=x[ix],y[ix]
            optimizer.zero_grad(set_to_none=True)
            scores=model.network(issl_input_validated(xx,crop))[0]
            loss=F.cross_entropy(scores,yy)
            checked_step(loss,optimizer);values.append(loss.detach())
        scheduler.step()
        extra._log(logger,stage='C_transfer',epoch=epoch+1,loss=_mean_loss(values),
                   lr=optimizer.param_groups[0]['lr'],query_fit=False)
    model.eval()
