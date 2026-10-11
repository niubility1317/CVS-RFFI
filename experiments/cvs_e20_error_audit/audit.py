"""Error-conditioned source readouts and immutable E20 gradient diagnostics."""
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F

BINS=(0.,.5,.8,.9,.95,.98,.99,1.)


@contextmanager
def modes(model,training=False):
    saved=[(m,m.training) for m in model.modules()]
    frozen=[]
    try:
        model.train(training)
        if training:
            for name,module in model.named_modules():
                if isinstance(module,nn.modules.batchnorm._BatchNorm) or any(k in module._buffers for k in ('running_mean','running_var')):
                    module.training=False;frozen.append(name)
        yield frozen
    finally:
        for module,value in saved:module.training=value


def _validate(cache,role):
    if role not in ('L','V','U'):raise ValueError('Explicit legal source role required')
    if role=='U' and 'y' in cache:raise ValueError('U truth must not be present')
    # load_source strips metadata from L/V but retains the native U_s cache.
    # Normalize only documented aliases, never accept another data role.
    permitted={'L':('L','L_s'),'V':('V',),'U':('U','U_s')}
    if cache.get('role',role) not in permitted[role]:raise ValueError('Source role mismatch')
    n=len(cache['ids'])
    if n==0 or len(set(cache['ids']))!=n:raise ValueError('Physical IDs must be nonempty and unique')
    if any(len(cache[key])!=n for key in ('x','rx','day')):raise ValueError('Source shape mismatch')
    if role!='U' and (len(cache['y'])!=n or cache['y'].dtype!=torch.long or not bool(((cache['y']>=0)&(cache['y']<6)).all())):
        raise ValueError('Six-class visible source labels required')


def collect_predictions(model,cache,device,batch=256,*,role):
    if role not in ('L','V'):raise ValueError('Readout labels only allowed for L/V; U prohibited')
    _validate(cache,role)
    if batch<1:raise ValueError('Positive batch required')
    values=[]
    with modes(model,False),torch.no_grad():
        for start in range(0,len(cache['ids']),batch):
            z=model(cache['x'][start:start+batch].to(device))
            if z.shape!=(min(batch,len(cache['ids'])-start),6) or not torch.isfinite(z).all():raise ValueError('Invalid six-class logits')
            values.append(z.detach().cpu())
    logits=torch.cat(values)
    return dict(role=role,logits=logits,probabilities=logits.softmax(1),ids=list(cache['ids']),
                **{k:cache[k].detach().cpu().clone() for k in ('y','rx','day')})


def _summary(old,new,y):
    oc,op=old.softmax(1).max(1);nc,np=new.softmax(1).max(1)
    correct=op==y;after=np==y;high=oc>=.95
    def margin(z):return z.gather(1,y[:,None]).flatten()-z.masked_fill(F.one_hot(y,6).bool(),-torch.inf).max(1).values
    om,nm=margin(old),margin(new)
    on,nnl=F.cross_entropy(old,y,reduction='none'),F.cross_entropy(new,y,reduction='none')
    def stats(mask):
        n=int(mask.sum())
        def mean(x):return float(x[mask].double().mean()) if n else None
        return dict(count=n,initial_correct=int((mask&correct).sum()),final_correct=int((mask&after).sum()),
            fixed=int((mask&~correct&after).sum()),broken=int((mask&correct&~after).sum()),
            persistent_error=int((mask&~correct&~after).sum()),persistent_correct=int((mask&correct&after).sum()),
            initial_accuracy=mean(correct),final_accuracy=mean(after),initial_margin=mean(om),final_margin=mean(nm),
            margin_change=mean(nm-om),initial_NLL=mean(on),final_NLL=mean(nnl),NLL_change=mean(nnl-on),
            final_confidence=mean(nc),initial_confidence=mean(oc),
            persistent_same_wrong_class=int((mask&~correct&~after&(op==np)).sum()))
    groups={name:stats(mask) for name,mask in dict(high_correct=high&correct,high_error=high&~correct,
        low_correct=~high&correct,low_error=~high&~correct).items()}
    calibration={}
    for label,conf,yes in [('initial',oc,correct),('final',nc,after)]:
        rows=[];ece=0.
        for index,(lo,hi) in enumerate(zip(BINS[:-1],BINS[1:])):
            mask=(conf>=lo)&((conf<=hi) if index==len(BINS)-2 else (conf<hi));n=int(mask.sum())
            acc=float(yes[mask].double().mean()) if n else None;avg=float(conf[mask].double().mean()) if n else None
            if n:ece+=n/len(y)*abs(acc-avg)
            rows.append(dict(lower=lo,upper=hi,upper_inclusive=index==len(BINS)-2,count=n,accuracy=acc,confidence=avg,error_count=int((mask&~yes).sum())))
        calibration[label]=dict(bins=rows,ECE=ece,temperature_fitted=False)
    return dict(overall=stats(torch.ones_like(correct)),initial_groups=groups,calibration=calibration)


def paired_readout(parent_result,end_result):
    a,b=parent_result,end_result
    if a['role'] not in ('L','V') or b['role']!=a['role'] or a['ids']!=b['ids'] or len(set(a['ids']))!=len(a['ids']):raise ValueError('Require same ordered unique PID and role')
    if any(not torch.equal(a[k],b[k]) for k in ('y','rx','day')):raise ValueError('Paired truth/strata mismatch')
    n=len(a['ids'])
    if any(z.shape!=(n,6) or not torch.isfinite(z).all() for z in (a['logits'],b['logits'])):raise ValueError('Invalid paired logits')
    result=_summary(a['logits'],b['logits'],a['y'])
    result.update(role=a['role'],same_PID=True,threshold=.95,by_stratum={})
    for key in ('rx','day','y'):
        result['by_stratum']['TX' if key=='y' else key]={}
        for label in a[key].unique().tolist():
            ix=a[key]==label
            result['by_stratum']['TX' if key=='y' else key][str(label)]=_summary(a['logits'][ix],b['logits'][ix],a['y'][ix])
    return result


def _groups(model):
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad and 'id_backbone' in n and not n.endswith(('low_hz_','band_hz_'))]
    result=dict(shallow=[p for n,p in named if '.time_fuse.' in n],middle=[p for n,p in named if '.t2.' in n],
        encoder=[p for n,p in named if '.cls_head.' not in n],classifier=[p for n,p in named if '.cls_head.' in n])
    if any(not values for values in result.values()):raise ValueError('Missing native gradient group')
    return result


def _grad(loss,params):
    if not loss.requires_grad:return [torch.zeros_like(p) for p in params]
    g=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return [(torch.zeros_like(p) if v is None else v).detach() for p,v in zip(params,g)]


def _alignment(values,reference,indices):
    norm=sum(values[i].double().square().sum() for i in indices).sqrt()
    ref=sum(reference[i].double().square().sum() for i in indices).sqrt()
    dot=sum((values[i].double()*reference[i].double()).sum() for i in indices)
    nonzero=bool(norm>0 and ref>0)
    return dict(norm=float(norm),reference_norm=float(ref),dot=float(dot),cosine=float(dot/(norm*ref)) if nonzero else None,
        norm_ratio=float(norm/ref) if ref>0 else None,negative=bool(dot<0) if nonzero else None,zero_gradient=bool(norm==0))


def gradient_audit(model,L,U,base_config,rounds=24,device='cuda:0'):
    from experiments.cvs_e20_budget_fishr import design as design
    from experiments.cvs_e20_budget_fishr.objective import objective
    from experiments.cvs_early_geometry.objective import weak
    from experiments.cvs_report_ssl.sampling import DomainStream
    _validate(L,'L');_validate(U,'U')
    if set(L['ids'])&set(U['ids']):raise ValueError('L/U physical roles overlap')
    if rounds<1:raise ValueError('Positive rounds required')
    groups=_groups(model);params=list({id(p):p for values in groups.values() for p in values}.values())
    positions={id(p):i for i,p in enumerate(params)};indices={k:[positions[id(p)] for p in ps] for k,ps in groups.items()}
    configs=[design.config(r) for r in design.rows() if r['model_seed']==base_config['model_seed']]
    if len(configs)!=8:raise ValueError('Require all eight fixed arms')
    before={n:t.detach().clone() for n,t in model.state_dict().items()};original_modes=[(m,m.training) for m in model.modules()]
    # manual_seed also seeds CUDA: preserve every CUDA generator, not just the
    # device hosting this model, so other owners' random streams cannot advance.
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    stream=DomainStream(U['rx'],U['day'],base_config['model_seed']+901337);rows=[]
    try:
        with torch.random.fork_rng(devices=devices):
            for i in range(rounds):
                step=1332+4*i
                main=model.multilevel.balanced_indices(L,step-4);aux=model.multilevel.balanced_indices(L,step);probe=model.multilevel.balanced_indices(L,step+4)
                sets=[{L['ids'][j] for j in ix.tolist()} for ix in (main,aux,probe)]
                if any(sets[a]&sets[b] for a,b in ((0,1),(0,2),(1,2))):raise ValueError('Main/aux/reference L overlap')
                ui,_=stream.batch(128)
                with modes(model,False):
                    reference_loss=F.cross_entropy(model(L['x'][probe].to(device)),L['y'][probe].to(device));reference=_grad(reference_loss,params)
                    with torch.no_grad():q,_=weak(model,L['x'][main].to(device))
                q=q.detach().cpu();weak_readout=_summary(q,q,L['y'][main].cpu())
                for c in configs:
                    torch.manual_seed(base_config['model_seed']+step*137)
                    if devices:torch.cuda.manual_seed_all(base_config['model_seed']+step*137)
                    with modes(model,True) as frozen:
                        total,parts,weighted,details=objective(c,model,None,L,main,U,ui if c['unlabeled'] else None,step,device)
                        objectives={key:weighted[key] for key in ('ce','KL','aux_ce','fishr')};objectives['total']=total
                        tasks={}
                        for name,loss in objectives.items():
                            values=_grad(loss,params)
                            tasks[name]=dict(loss=float(loss.detach()),groups={key:_alignment(values,reference,ix) for key,ix in indices.items()})
                    changed=[n for n,t in model.state_dict().items() if not torch.equal(t,before[n])]
                    if changed:raise ValueError('Diagnostic changed model state: '+','.join(changed))
                    rows.append(dict(round=i,step=step,arm=c['arm'],reference_loss=float(reference_loss.detach()),tasks=tasks,
                        main_L_ids=sorted(sets[0]),aux_L_ids=sorted(sets[1]),reference_L_ids=sorted(sets[2]),
                        main_day=L['day'][main].unique().tolist(),aux_day=L['day'][aux].unique().tolist(),reference_day=L['day'][probe].unique().tolist(),
                        U_occurrences=len(ui),weak_main_L_calibration=weak_readout,details=details,frozen_statistics_modules=list(frozen),state_unchanged=True))
    finally:
        with torch.no_grad():
            for name,t in model.state_dict().items():t.copy_(before[name])
        for module,training in original_modes:module.training=training
    return dict(rows=rows,rounds=rounds,arms=8,identity_updates=0,source_only=True,V_gradient_access=False,U_truth_access=False,
        state_unchanged=True,RNG_restored=True,main_forward='train dropout; only running-statistic modules frozen',
        reference='independent visible L CE in eval mode; not V; no optimizer replay',gradient_scope='dense native parameters excluding Hz; group overlap intentional')
