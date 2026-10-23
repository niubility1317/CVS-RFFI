"""Read-only fixed-cohort feature geometry;no fitted probe or model-state update."""
import numpy as np
import torch
from torch.nn import functional as F

LEVELS=('time_down','t2.pool','joint160','core160','signature160','reference160')
SOURCE_RXS=(1,3,4,6,8)
SOURCE_DAYS=(1,2,3)


def fixed_indices(metadata,role):
    if role not in ('U_s','V'):raise ValueError('Only existing source U or diagnostic V permitted')
    if role=='U_s' and 'y' in metadata:raise ValueError('U TX truth must remain hidden')
    if role=='V' and 'y' not in metadata:raise ValueError('Source V diagnostic labels required')
    result=[];rng=np.random.default_rng(2026102307)
    for cls in ((None,) if role=='U_s' else range(6)):
        for rx in SOURCE_RXS:
            for day in SOURCE_DAYS:
                mask=(metadata['rx_i']==rx)&(metadata['day_i']==day)
                if cls is not None:mask=mask&(metadata['y']==cls)
                pool=np.flatnonzero(mask);n=64 if role=='U_s' else 8
                if len(pool)<n:raise ValueError('Fixed existing source cohort lacks requested RX/day/class cell')
                result.extend((rng.choice(pool,n,replace=False) if role=='U_s' else pool[:n]).tolist())
    expected=960 if role=='U_s' else 720
    if len(result)!=expected:raise ValueError('Fixed source diagnostic cohort differs')
    return np.asarray(result)[np.random.default_rng(2026102401).permutation(expected)].tolist()


def geometry(h,labels,rx,accepted,normalization):
    if (h.ndim!=2 or labels.shape!=(len(h),) or rx.shape!=labels.shape or accepted.shape!=labels.shape
        or labels.dtype!=torch.long or rx.dtype!=torch.long
        or normalization not in ('unit_only','cohort_native_cric') or not torch.isfinite(h).all()
        or not torch.isfinite(accepted).all() or not ((accepted==0)|(accepted==1)).all()
        or not ((labels>=0)&(labels<6)).all() or set(rx.tolist())-set(SOURCE_RXS)):
        raise ValueError('Finite aligned source descriptors,registered labels/RX and binary fixed mask required')
    selected=accepted.bool();n=int(selected.sum());z=h[selected].double();y=labels[selected];r=rx[selected]
    result=dict(samples_total=len(h),samples_selected=n,descriptor_dimension=h.shape[1],normalization=normalization,
        class_counts=[int((y==cls).sum()) for cls in range(6)])
    if not n:
        return dict(result,status='EMPTY_SELECTION',total_scatter=0.,class_between_scatter=0.,class_within_scatter=0.,
            conditional_RX_between_scatter=0.,class_fraction_of_total=None,
            conditional_RX_fraction_of_class_residual=None,conditional_RX_fraction_of_total=None,covered_class_RX_cells=0)
    if normalization=='cohort_native_cric':z=(z-z.mean(0))/z.std(0,unbiased=False).clamp_min(1e-4)
    z=F.normalize(z,dim=1,eps=1e-4);mean=z.mean(0);total=float((z-mean).square().sum())
    between=within=rx_between=0.;cells=0
    for cls in range(6):
        mask=y==cls;count=int(mask.sum())
        if not count:continue
        center=z[mask].mean(0);between+=count*float((center-mean).square().sum())
        within+=float((z[mask]-center).square().sum())
        for receiver in SOURCE_RXS:
            chosen=mask&(r==receiver);size=int(chosen.sum())
            if size:
                rx_between+=size*float((z[chosen].mean(0)-center).square().sum());cells+=1
    if abs(total-between-within)>1e-7*max(1.,total):raise ValueError('Source scatter decomposition differs')
    return dict(result,status='OK' if total>1e-12 else 'ZERO_SPREAD',total_scatter=total,
        class_between_scatter=between,class_within_scatter=within,conditional_RX_between_scatter=rx_between,
        class_fraction_of_total=between/total if total>1e-12 else None,
        conditional_RX_fraction_of_class_residual=rx_between/within if within>1e-12 else None,
        conditional_RX_fraction_of_total=rx_between/total if total>1e-12 else None,covered_class_RX_cells=cells)


def capture_deployed(reader,x):
    if reader.training or any(p.requires_grad for p in reader.parameters()):raise ValueError('Frozen eval reader required;no V gradient/state updates')
    if x.ndim!=3 or x.shape[1:]!=(2,256):raise ValueError('Existing source IQ2x256 required')
    captured={};handles=[]
    def capture(name,temporal=False):
        def hook(module,inputs,output):
            if name in captured:raise ValueError('Exactly one existing forward per branch required')
            value=torch.cat((output.mean(-1),output.std(-1,unbiased=False)),1) if temporal else output
            captured[name]=value.detach().cpu()
        return hook
    try:
        for name in LEVELS[:2]:handles.append(reader.get_submodule('core.id_backbone.'+name).register_forward_hook(capture(name,True)))
        handles.append(reader.sequence_pool.register_forward_hook(capture('signature160')))
        handles.append(reader.response_projection.register_forward_hook(capture('reference160')))
        with torch.inference_mode():
            joint=reader.features(x);logits=reader.classify_features(joint)
        if joint.shape!=(len(x),160) or logits.shape!=(len(x),6) or set(captured)!=set(LEVELS[:2]+LEVELS[4:]):
            raise ValueError('Actual deployed PAMF branch captures/default head missing')
        captured['joint160']=joint.detach().cpu()
        captured['reference160']=captured['reference160']*reader.response_gain.detach().cpu().tanh()
        captured['core160']=captured['joint160']-captured['signature160']-captured['reference160']
        if any(not torch.isfinite(h).all() for h in captured.values()) or not torch.isfinite(logits).all():
            raise ValueError('Nonfinite source forward')
        return captured,logits.detach().cpu()
    finally:
        for handle in handles:handle.remove()


def describe(features,labels,rx,accepted):
    if set(features)!=set(LEVELS):raise ValueError('All actual deployed feature branches required')
    values={mode:{level:geometry(features[level],labels,rx,accepted,mode) for level in LEVELS}
        for mode in ('unit_only','cohort_native_cric')}
    selected=accepted.bool();joint=features['joint160'].norm(dim=1).clamp_min(1e-12)
    contributions={}
    for level in LEVELS[3:]:
        h=features[level];ratios=h.norm(dim=1)/joint
        contributions[level]=dict(mean_norm=float(h.norm(dim=1).mean()),
            mean_norm_relative_to_joint=float(ratios.mean()),
            selected_mean_norm_relative_to_joint=float(ratios[selected].mean()) if selected.any() else None)
    return dict(geometry=values,branch_contributions=contributions,
        interpretation='Descriptive source geometry on the same fixed cohort and fixed reference acceptance/classes;unit-only and cohort-normalized summaries are not causal RX nuisance isolation or target gains. A zero-spread/collapsed branch is not evidence of useful invariance. U classes remain reference pseudo labels;no probe/projection fitted.')


def fixed_head_interventions(reader,features,labels,partner,valid):
    if reader.training or any(p.requires_grad for p in reader.parameters()):raise ValueError('Only frozen existing default head permitted')
    if set(features)!=set(LEVELS):raise ValueError('Actual deployed additive features required')
    h=features['joint160'];s=features['signature160'];r=features['reference160'];core=features['core160'];n=len(h)
    if (h.shape!=(n,160) or any(v.shape!=h.shape for v in (s,r,core)) or labels.shape!=(n,)
        or labels.dtype!=torch.long or partner.shape!=(n,) or partner.dtype!=torch.long
        or valid.shape!=(n,) or valid.dtype!=torch.bool or (n and (int(partner.min())<0 or int(partner.max())>=n))):
        raise ValueError('Aligned fixed same-pseudo/sourceRX partner indices and existing160 features required')
    cases=dict(joint=h,without_reference=h-r,without_signature=h-s,core_only=core,
        reference_swap=h-r+r[partner],signature_swap=h-s+s[partner],late_swap=core+s[partner]+r[partner])
    count=int(valid.sum());rows=[]
    with torch.inference_mode():
        for name,value in cases.items():
            logits=reader.classify_features(value)
            if logits.shape!=(n,6) or not torch.isfinite(logits).all():raise ValueError('Existing default-head intervention produced nonfinite logits')
            selected=logits[valid];y=labels[valid]
            rows.append(dict(intervention=name,samples_selected=count,
                label_agreements=int((selected.argmax(1)==y).sum()),
                label_agreement_percent=100*float((selected.argmax(1)==y).double().mean()) if count else None,
                mean_label_CE=float(F.cross_entropy(selected,y)) if count else None,
                mean_relative_feature_displacement=float(((value[valid]-h[valid]).norm(dim=1)/h[valid].norm(dim=1).clamp_min(1e-12)).mean()) if count else None))
    return dict(rows=rows,extra_backbone_forwards=0,existing_head_forward_calls=len(cases),head_fit_updates=0,
        interpretation='Frozen existing-head source interventions only;partners chosen by reference pseudo labels and RX,never sourceV truth.No fitted or replacement head,no model update.These synthetic latent combinations do not prove physical realizability or target gains.')
