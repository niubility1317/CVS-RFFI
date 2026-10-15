import math
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_sigtime_rdrop_envelope20 import mechanism as operators


def summary(a, b):
    a, b = a.detach().double().reshape(-1), b.detach().double().reshape(-1)
    na, nb = float(a.norm()), float(b.norm())
    if not math.isfinite(na + nb):
        raise ValueError('Finite gradients required')
    return dict(L2=na, reference_L2=nb, L2_ratio=na/nb if nb else None,
                cosine=float(a@b)/(na*nb) if na and nb else None)


def pair_gradients(model, x, y):
    if not model.training or not torch.is_grad_enabled() or x.shape != (144, 2, 256) or y.shape != (144,) or y.dtype != torch.long:
        raise ValueError('Fixed144 fit-L paired diagnostic,dropout on,autograd on required')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Diagnostic cannot inherit training gradients')
    first = model.forward_identity_only(x)
    second = model.forward_identity_only(x)
    p, q, z1, z2 = first['tx_logits'], second['tx_logits'], first['id_feat_joint'], second['id_feat_joint']
    if z1.shape != (144,160) or z2.shape != z1.shape:
        raise ValueError('Actual deployed joint160 required')
    named = dict(model.named_parameters())
    inputs = (p, q, z1, z2, *(named[n] for n in d.PARAMETERS))
    ce = .5*(F.cross_entropy(p,y,label_smoothing=.01)+F.cross_entropy(q,y,label_smoothing=.01))
    losses = dict(CE=ce)
    for arm in operators.ARMS:
        losses[arm], _ = operators.consistency(p,q,y,arm,operators.new_audit())
    gradients = {arm: torch.autograd.grad(loss,inputs,retain_graph=True) for arm,loss in losses.items()}
    grouped = {}
    for arm,g in gradients.items():
        grouped[arm] = dict(logits=torch.stack(g[:2]),joint160=torch.stack(g[2:4]),
            t1=g[4],t2=g[5],SIG_TIME=g[6],default_W=g[7])
    records=[]
    for arm in losses:
        for group,g in grouped[arm].items():
            reference=grouped['CE'][group]
            native=grouped['FULL'][group]
            record=dict(operator=arm,group=group,loss=float(losses[arm].detach()),
                against_CE=summary(g,reference),against_FULL=summary(g,native),
                total_against_CE=summary(g+reference,reference) if arm!='CE' else None)
            if group in ('joint160','logits'):
                # Orthogonal algebraic decomposition in the two output spaces.
                # Distinct view Jacobians prevent an encoder-centre invariance claim.
                common=(g[0].double()+g[1].double())/math.sqrt(2.)
                difference=(g[0].double()-g[1].double())/math.sqrt(2.)
                energy=float(g.double().square().sum())
                common_energy=float(common.square().sum())
                difference_energy=float(difference.square().sum())
                if abs(common_energy+difference_energy-energy)>1e-11*max(1.,energy):
                    raise ValueError('Paired gradient energy decomposition differs')
                pair_g=g.double().permute(1,0,2).flatten(1)
                pair_ce=reference.double().permute(1,0,2).flatten(1)
                norms=pair_g.norm(dim=1)*pair_ce.norm(dim=1)
                valid=norms>0
                cos=(pair_g*pair_ce).sum(1)[valid]/norms[valid]
                record.update(common_energy_fraction=common_energy/energy if energy else None,
                    difference_energy_fraction=difference_energy/energy if energy else None,
                    packet_CE_alignment_count=int(valid.sum()),
                    packet_CE_alignment_mean=float(cos.mean()) if len(cos) else None,
                    packet_CE_opposed_count=int((cos<0).sum()))
            records.append(record)
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('autograd.grad must leave model gradient slots empty')
    return records
