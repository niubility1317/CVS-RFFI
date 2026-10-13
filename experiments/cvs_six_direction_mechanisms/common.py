"""Shared source data and frozen feature interfaces. No target reader is called."""
import torch
from . import design as d
from experiments.cvs_source_pseudotarget_health.frozen import preserve_rng,snapshot,assert_unchanged
from experiments.cvs_source_pseudotarget_health.probes import capture,descriptors,partition
from experiments.cvs_source_pseudotarget_health.stats import validate_role

@torch.no_grad()
def state_q(x):
    if x.ndim!=3 or x.shape[1]!=2 or x.shape[2]!=256 or not torch.isfinite(x).all():raise ValueError('Finite received source IQ256 required')
    a=x.float().square().sum(1).clamp_min(1e-12).sqrt()
    a=a/a.square().mean(1,keepdim=True).clamp_min(1e-12).sqrt()
    blocks=a.reshape(len(a),4,64)
    return torch.cat((blocks.mean(2),blocks.std(2,unbiased=False),a.mean(1,keepdim=True),a.std(1,unbiased=False,keepdim=True),(a[:,1:]-a[:,:-1]).square().mean(1,keepdim=True),(a[:,4:]-a[:,:-4]).square().mean(1,keepdim=True)),1).detach().cpu()

def normalized_state(raw,fit):
    if raw.ndim!=2 or fit.dtype!=torch.bool or fit.shape!=(len(raw),) or not fit.any() or not torch.isfinite(raw).all():raise ValueError('Nonempty finite L-fit state required')
    mean=raw[fit].mean(0);scale=raw[fit].std(0,unbiased=False).clamp_min(1e-4)
    return ((raw-mean)/scale).detach(),mean,scale

def load_roles(c):
    from experiments.cvs_nocfo_foundation.source import load_source
    from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled
    L,unused,V,contract=load_source(c['source_loader_config'])
    if unused is not None:raise ValueError('L/V loader unexpectedly opened U')
    U=torch.load(d.parent.U_CACHE,map_location='cpu',weights_only=False)
    validate_unlabeled(U,contract['role_ids']['U_s'])
    roles={'L_s':L,'V':V,'U_s':U}
    for name,cache in roles.items():
        if validate_role(cache,name,contract['role_ids'][name])!=d.ROLE_COUNTS[name]:raise ValueError('Source roles differ')
    sets=[set(v['ids']) for v in roles.values()]
    if any(a&b for i,a in enumerate(sets) for b in sets[i+1:]):raise ValueError('Source physical roles overlap')
    return roles,contract

def load_model(row,device):
    from experiments.cvs_scratch30_baseline.predict import source_proof,own_model
    c=d.source_config(row);p=source_proof(c);model=own_model(c,device)
    if model.head.weight.shape!=(6,160) or model._identity_input_policy['explicit_CFO_identity_input']:raise ValueError('Ordinarycos/NoExplicitCFO interface differs')
    proof=dict(source_checkpoint=(p/'identity.pth').as_posix(),source_config=c,source_contract=d.read(p/'source_contract.json'),source_training=d.read(p/'completion.json'),identity_updates=0,target_access=False)
    return model,proof

@torch.no_grad()
def scan(model,cache,role,device):
    probs=[];features={k:[] for k in ('time_down','t2.pool','joint160')} if role!='U_s' else None
    for start in range(0,len(cache['ids']),d.BATCH_SIZE):
        x=cache['x'][start:start+d.BATCH_SIZE].to(device)
        if features is not None:
            with capture(model) as maps:out=model.forward_identity_only(x)
            z=descriptors(maps,out['id_feat_joint'])
            # Preserve the real middle time sequence for the directed-statistic diagnostic.
            if 'middle_sequence' not in features:features['middle_sequence']=[]
            features['middle_sequence'].append(maps['t2.pool'].detach().cpu())
            for k,v in z.items():features[k].append(v)
            logits=out['tx_logits']
        else:logits=model(x)
        if logits.shape!=(len(x),6) or not torch.isfinite(logits).all():raise ValueError('Frozen source logits invalid')
        probs.append(logits.softmax(1).cpu())
    return torch.cat(probs),{k:torch.cat(v) for k,v in features.items()} if features is not None else None

def context(row,model,roles,device):
    features={};probabilities={}
    for name,cache in roles.items():
        probabilities[name],z=scan(model,cache,name,device)
        if z is not None:features[name]=z
    folds,fold_info=partition(roles['L_s'])
    q=state_q(roles['L_s']['x']);standardized,mean,scale=normalized_state(q,folds!=2)
    return dict(model=model,config=d.source_config(row),seed=row['model_seed'],L=roles['L_s'],V=roles['V'],U=roles['U_s'],features=features,probabilities=probabilities,folds=folds,fold_info=fold_info,state_q_raw=q,state_q=standardized,state_mean=mean,state_scale=scale,device=device)
