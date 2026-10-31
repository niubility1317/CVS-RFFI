"""Fixed-source common-class transport; no identity head or target labels.

Adaptive filling and statistical selection are adapted from
https://github.com/changwxx/UniOT-for-UniDA/blob/main/utils/lib.py .
The solver here uses fixed FP64 log-domain iterations, without POT.

MIT License

Copyright (c) 2022 changwxx

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F
from .star_target_prototype_core import _matrix

REG=.01
RHO=.5
ITERATIONS=400
GAMMA=.7
MU=.7
TEMPERATURE=.1
FILL_SEED=2026103120


def _sinkhorn(sim,beta):
    if (not torch.is_tensor(sim) or sim.ndim!=2 or not all(sim.shape) or sim.dtype!=torch.float64 or not bool(torch.isfinite(sim).all())
            or not torch.is_tensor(beta) or beta.dtype!=torch.float64 or beta.shape!=(sim.shape[1],) or beta.device!=sim.device
            or not bool(torch.isfinite(beta).all()) or bool(beta.le(0).any()) or abs(float(beta.sum())-1)>1e-10):
        raise ValueError('Finite FP64 similarity matrix and positive unit-mass beta required')
    with torch.no_grad():
        sim=sim.detach();beta=beta.detach();n,k=sim.shape;loga=sim.new_full((n,),-math.log(n));logb=beta.log();kernel=sim/REG
        u=sim.new_zeros(n);v=sim.new_zeros(k);theta=RHO/(RHO+REG)
        for _ in range(ITERATIONS):
            old_u,old_v=u,v
            u=theta*(loga-torch.logsumexp(kernel+v[None,:],dim=1))
            v=theta*(logb-torch.logsumexp(kernel+u[:,None],dim=0))
        logp=u[:,None]+kernel+v[None,:];logmass=torch.logsumexp(logp.flatten(),0)
        q=(logp-logmass).exp()
        derivative=REG*logp-sim+RHO*(torch.logsumexp(logp,1)-loga)[:,None]+RHO*(torch.logsumexp(logp,0)-logb)[None,:]
        if not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(derivative).all()):raise FloatingPointError('Nonfinite fixed-budget transport')
        return q,dict(iterations=ITERATIONS,reg=REG,rho=RHO,log_total_mass=float(logmass.cpu()),stationarity_residual=float(derivative.abs().max().cpu()),
            max_scaling_change=float(torch.maximum((u-old_u).abs().max(),(v-old_v).abs().max()).cpu()))


def _selection(q):
    mass=q.sum(0);best,labels=q.max(1)
    return labels,(best>1/len(q))&(mass[labels]>=1/q.shape[1]),mass


def assign(weak,centers,beta,generator):
    _matrix(weak,'current detached weak U');_matrix(centers,'fixed public old-six centers')
    if centers.shape!=(6,160) or centers.device!=weak.device or not isinstance(generator,torch.Generator) or generator.device.type!='cpu':raise ValueError('Six fixed same-device source centers and independent CPU filling RNG required')
    with torch.no_grad():
        x=F.normalize(weak.detach(),dim=1);p=F.normalize(centers.detach(),dim=1);sim=x.double()@p.double().T;n=len(x)
        positive=int(sim.max(1).values.gt(GAMMA).sum().cpu());branch='positive' if positive*2<=n else 'negative';requested=abs(n-2*positive)
        fake=0;initial_count=0;solves=[];newsim=sim
        if requested and branch=='positive':
            q,d=_sinkhorn(sim,beta);solves.append(d);_,selected,_=_selection(q);ids=selected.nonzero().flatten();initial_count=len(ids)
            if len(ids):
                draw=torch.randint(len(ids),(requested,),generator=generator).to(x.device);newsim=torch.cat((sim[ids[draw]],sim));fake=requested
        elif requested:
            farthest=sim.argmin(1);negative=F.normalize(.5*x+.5*p[farthest],dim=1)
            draw=torch.randint(n,(requested,),generator=generator).to(x.device)
            newsim=torch.cat((negative[draw].double()@p.double().T,sim));fake=requested
        q,d=_sinkhorn(newsim,beta);solves.append(d);labels,selected,mass=_selection(q);labels=labels[fake:];selected=selected[fake:]
        new_beta=MU*beta.detach()+(1-MU)*mass
        hist=torch.bincount(labels[selected],minlength=6)
        diagnostics=dict(iterations=ITERATIONS,reg=REG,rho=RHO,gamma=GAMMA,mu=MU,U_count=n,filled_count=len(newsim),fill_branch=branch,
            positive_count=positive,positive_rate=positive/n,requested_fake_size=requested,fake_size=fake,initial_selected_count=initial_count,
            selected_count=int(selected.sum().cpu()),selected_histogram=hist.cpu().tolist(),beta_before=beta.detach().cpu().tolist(),beta_after=new_beta.cpu().tolist(),
            column_mass=mass.cpu().tolist(),uot_calls=len(solves),transports=solves,assignments_detached=True,true_labels_used=False,true_class_count_used=False)
        return dict(labels=labels.detach(),mask=selected.detach(),beta=new_beta.detach(),diagnostics=diagnostics)


def classification_loss(strong0,strong1,centers,indices,assignment):
    _matrix(strong0,'strong zero');_matrix(strong1,'strong one');_matrix(centers,'fixed source centers')
    labels,mask=assignment['labels'],assignment['mask']
    if (strong1.shape!=strong0.shape or centers.shape!=(6,160) or any(a.device!=strong0.device for a in (strong1,centers,indices,labels,mask))
            or indices.dtype!=torch.long or indices.shape!=(len(strong0),) or labels.ndim!=1 or labels.dtype!=torch.long
            or mask.dtype!=torch.bool or mask.shape!=labels.shape or bool(((labels<0)|(labels>=6)).any()) or bool(((indices<0)|(indices>=len(labels))).any())):
        raise ValueError('Aligned strong batches, opaque U indices and detached six-class CCD assignments required')
    selected=mask.detach()[indices];count=int(selected.sum().cpu());p=F.normalize(centers.detach(),dim=1)
    if count:
        loss=.5*sum(F.cross_entropy((F.normalize(x,dim=1)@p.T/TEMPERATURE)[selected],labels.detach()[indices][selected]) for x in (strong0,strong1))
    else:loss=(strong0.sum()+strong1.sum())*0
    return dict(loss=loss,selected_batch_count=count)
