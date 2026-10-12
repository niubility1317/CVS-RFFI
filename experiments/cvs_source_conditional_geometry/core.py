"""Two native objectives from one forward, then exact finite-L ANOVA."""
import math,random
import numpy as np
import torch
from torch.nn import functional as F
from experiments.cvs_e20_boundary_style.multilevel import BoundarySensitivity,identity_margin,LEVELS
from experiments.cvs_e20_moment_audit.fishr_stats import exact_variances,penalty
from experiments.cvs_scratch_energy_sensitivity.moment_identifiability import decompose,interaction


def readout(model,L,seed,device,batch_size=128,progress=None):
    if L.get('role')!='L_s' or L['x'].ndim!=3 or len(L['x'])!=len(L['ids']) or len(set(L['ids']))!=len(L['ids']):raise ValueError('Unique source-L records required')
    if len(L['y'])!=len(L['ids']) or not bool(torch.isfinite(L['x']).all()):raise ValueError('Finite source L and visible labels required')
    helper=BoundarySensitivity(model.base,seed);chunks={name:{level:[] for level in LEVELS} for name in ('margin','ce')}
    before={k:v.detach().clone() for k,v in model.state_dict().items()};buffers={k:v.detach().clone() for k,v in model.named_buffers()};modes=[(m,m.training) for m in model.modules()]
    grads=[None if p.grad is None else p.grad.clone() for p in model.parameters()];py=random.getstate();nr=np.random.get_state()
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    hooks=sum(len(m._forward_hooks) for m in model.modules());calls=[];identity_error=0.;correct=0;ce_sum=0.
    try:
        model.eval()
        for start in range(0,len(L['ids']),batch_size):
            stop=min(start+batch_size,len(L['ids']));x=L['x'][start:stop].to(device);y=L['y'][start:stop].to(device)
            with helper.capture() as (probes,shapes):logits=model(x)
            calls.append(stop-start);margin=identity_margin(logits,y);ce=F.cross_entropy(logits,y,reduction='none')
            leaves=tuple(p for level in LEVELS for p in probes[level]);gm=torch.autograd.grad(margin.sum(),leaves,retain_graph=True,create_graph=False)
            gc=torch.autograd.grad(ce.sum(),leaves,create_graph=False);confidence=logits.detach().softmax(1).gather(1,y[:,None])
            for i,level in enumerate(LEVELS):
                a=torch.cat((gm[2*i].squeeze(-1),gm[2*i+1].squeeze(-1)),1).detach()
                b=torch.cat((gc[2*i].squeeze(-1),gc[2*i+1].squeeze(-1)),1).detach();expected=-(1-confidence)*a
                if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):raise FloatingPointError('Nonfinite raw sensitivity')
                torch.testing.assert_close(b,expected,rtol=2e-4,atol=2e-6)
                identity_error=max(identity_error,float((b-expected).abs().max()))
                chunks['margin'][level].append(a.cpu().double());chunks['ce'][level].append(b.cpu().double())
            correct+=int((logits.detach().argmax(1)==y).sum());ce_sum+=float(ce.detach().sum())
            if progress:progress(dict(batch=len(calls),packets=stop,total=len(L['ids'])))
        assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
        assert buffers.keys()==dict(model.named_buffers()).keys() and all(torch.equal(buffers[k],v) for k,v in model.named_buffers())
        assert all((p.grad is None and v is None) or (p.grad is not None and v is not None and torch.equal(p.grad,v)) for p,v in zip(model.parameters(),grads))
        assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
        now=np.random.get_state();assert nr[0]==now[0] and np.array_equal(nr[1],now[1]) and nr[2:]==now[2:] and py==random.getstate()
        assert sum(len(m._forward_hooks) for m in model.modules())==hooks
    finally:
        for module,mode in modes:module.training=mode
    values={name:{level:torch.cat(parts) for level,parts in levels.items()} for name,levels in chunks.items()}
    evidence=dict(model_forwards=len(calls),forward_batch_sizes=calls,L_occurrences=sum(calls),first_order_derivative_calls=2*len(calls),
        identity_updates=0,head_updates=0,parameters_buffers_gradients_RNG_modes_unchanged=True,no_saved_features=True,
        source_L_accuracy=correct/len(L['ids']),source_L_CE=ce_sum/len(L['ids']),CE_margin_identity_max_abs_error=identity_error)
    return values,evidence


def shuffled_controls(cells,seed):
    """Same finite samples, random balanced RX partition within each TX/day."""
    k,r,n,c=cells.shape;flat=cells.reshape(k,r*n,c);records=[]
    for i in range(32):
        gen=torch.Generator().manual_seed(seed+i*1597)
        shuffled=torch.stack([flat[y,torch.randperm(r*n,generator=gen)] for y in range(k)]).reshape(k,r,n,c)
        means=shuffled.mean(2);b=interaction(means)
        mixed=shuffled.permute(1,0,2,3).reshape(r,k*n,c).var(1,unbiased=False)
        records.append(dict(index=i,seed=seed+i*1597,interaction_energy=float(b.square().mean()),mixed_variance_penalty=float(penalty(mixed))))
    return records


def summarize(values,L,expected_cell_size=70,diagnostic_seed=0):
    if L.get('role')!='L_s':raise ValueError('Only source L can define conditional cells')
    domains=[1,3,4,6,8];days=[1,2,3];classes=list(range(6))
    if set(L['rx'].tolist())!=set(domains) or set(L['day'].tolist())!=set(days) or set(L['y'].tolist())!=set(classes):raise ValueError('Source cell grid differs')
    lookup={(day,y,rx):torch.where((L['day']==day)&(L['y']==y)&(L['rx']==rx))[0] for day in days for y in classes for rx in domains}
    if any(len(ix)!=expected_cell_size for ix in lookup.values()):raise ValueError('Incomplete/equally sized finite source cells required')
    records=[];scales={}
    for level in LEVELS:
        raw={name:values[name][level] for name in ('margin','ce')}
        if any(g.ndim!=2 or len(g)!=len(L['ids']) or not bool(torch.isfinite(g).all()) for g in raw.values()):raise ValueError('Missing finite source gradients')
        scales[level]={name:float(g.square().mean().sqrt()) for name,g in raw.items()}
        if any(not math.isfinite(v) or v<=0 for v in scales[level].values()):raise ValueError('Zero/nonfinite full-L scale; no fallback')
        statistics=dict(scaled_margin=raw['margin']/scales[level]['margin'],scaled_ce=raw['ce']/scales[level]['ce'],
            normalized_margin=raw['margin']/(raw['margin'].square().mean(1,keepdim=True)+1e-12).sqrt())
        for name,g in statistics.items():
            for day in days:
                cells=torch.stack([torch.stack([g[lookup[day,y,rx]] for rx in domains]) for y in classes])
                total,parts=decompose(cells);torch.testing.assert_close(total,sum(parts.values()),rtol=1e-10,atol=1e-12)
                means=cells.mean(2);b=interaction(means);torch.testing.assert_close(b.square().mean(),parts['interaction'],rtol=1e-12,atol=1e-14)
                ix=torch.where(L['day']==day)[0];variance=exact_variances(g[ix],L['y'][ix],L['rx'][ix]);fisher=penalty(variance)
                rawbase=raw['ce' if name=='scaled_ce' else 'margin']
                cell_metrics=[]
                for yi,y in enumerate(classes):
                    for ri,rx in enumerate(domains):
                        rawcell=rawbase[lookup[day,y,rx]]
                        cell_metrics.append(dict(TX=y,RX=rx,n=expected_cell_size,conditional_mean_RMS=float(means[yi,ri].square().mean().sqrt()),
                            interaction_RMS=float(b[yi,ri].square().mean().sqrt()),within_variance_mean=float(cells[yi,ri].var(0,unbiased=False).mean()),
                            raw_RMS=float(rawcell.square().mean().sqrt())))
                controls=shuffled_controls(cells,diagnostic_seed+day*100003)
                records.append(dict(statistic=name,layer=level,day=day,source_packets=len(ix),coordinates=g.shape[1],
                    total_centered_energy=float(total),components={k:float(v) for k,v in parts.items()},
                    fractions={k:float(v/total) if total>0 else None for k,v in parts.items()},
                    decomposition_abs_error=float((total-sum(parts.values())).abs()),mixed_diagonal_variance_penalty=float(fisher),
                    conditional_interaction_energy=float(parts['interaction']),cell_metrics=cell_metrics,RX_shuffled_controls=controls,
                    scale=scales[level]['ce' if name=='scaled_ce' else 'margin'] if name!='normalized_margin' else None))
    return dict(records=records,fixed_offline_scales=scales,cell_count=len(lookup),source_L_count=len(L['ids']),source_only=True,target_access=False,
        units='both scores dimensionless; variance penalty fourth-degree, interaction second-degree; magnitudes do not establish equal training strength',
        control_scope='32 balanced within-TX/day RX shuffles; descriptive finite-pool reference, not a population significance claim or unseen-RX test')
