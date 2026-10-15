"""Repair harmful actual evidence through its component edge, not its context."""
from collections.abc import Mapping
import torch
from torch.nn import functional as F
from experiments.cvs_identity_transport20.relationship import _bank, SOURCE_RX, SOURCE_DAYS

BRANCHES=('identity','physical','reference_raw','sequence_raw')
MODES=('ZERO','JOINT_GAP','ROUTED_GAP')
TOLERANCE=1e-6
COEFFICIENT=.1
EPS=1e-4

def positive(full_ce,without_ce):
    return torch.relu(full_ce-without_ce.detach()-TOLERANCE)

def compose(raw,gains,only=None,removed=None):
    parts={name:(value if only is None or name==only else value.detach()) for name,value in raw.items()}
    if removed is not None:parts[removed]=torch.zeros_like(parts[removed])
    return ((parts['identity']+gains[0].detach().tanh()*parts['physical'])
        +gains[1].detach().tanh()*parts['reference_raw'])+(.2*gains[2].detach().tanh())*parts['sequence_raw']

def ce(z,labels,weight):
    logits=30.*F.linear(F.normalize(z.double(),dim=1,eps=EPS),F.normalize(weight.detach().double(),dim=1,eps=EPS))
    return F.cross_entropy(logits,labels,reduction='none'),logits

def losses(components,bank,weight,gains,source_role='L_s'):
    # Permission precedes reading TX labels, even for ZERO/diagnostic controls.
    if source_role!='L_s':raise ValueError('Visible L_s only')
    if not isinstance(components,Mapping) or not all(k in components for k in (*BRANCHES,'joint')):
        raise ValueError('Direct same-call four deployment components required')
    labels,receivers,days,_=_bank(components['joint'],bank,source_role)
    joint=components['joint'];raw={name:components[name] for name in BRANCHES}
    for name,value in raw.items():
        if not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.shape!=joint.shape or value.dtype!=joint.dtype or value.device!=joint.device or not bool(torch.isfinite(value).all()):raise ValueError('Real component shape/dtype/device/finite differs: '+name)
    if not isinstance(weight,torch.Tensor) or weight.shape!=(6,160) or weight.dtype!=joint.dtype or weight.device!=joint.device or not bool(torch.isfinite(weight).all()):raise ValueError('Original finite same-device default W required')
    if not isinstance(gains,(tuple,list)) or len(gains)!=3:raise ValueError('Actual PA/reference/SIG three gain vectors required')
    for gain in gains:
        if not isinstance(gain,torch.Tensor) or gain.shape!=(160,) or gain.dtype!=joint.dtype or gain.device!=joint.device or not bool(torch.isfinite(gain).all()):raise ValueError('Actual finite gain vector differs')
    full=compose(raw,gains)
    if not torch.equal(full.detach(),joint.detach()):raise ValueError('Native four-term reconstruction must be exact')
    full_ce,full_logits=ce(full,labels,weight)
    without=[];terms=[];full_route=[];without_logits=[]
    for branch in BRANCHES:
        with torch.no_grad():
            ablated=compose({name:value.detach() for name,value in raw.items()},gains,removed=branch)
            factual,logits=ce(ablated,labels,weight)
        view=compose(raw,gains,only=branch)
        if not torch.equal(view.detach(),full.detach()):raise ValueError('Routed numeric fullview differs from deployed full')
        routed_ce,_=ce(view,labels,weight)
        if not torch.equal(routed_ce.detach(),full_ce.detach()):raise ValueError('Routing changes numeric full CE')
        terms.append(positive(routed_ce,factual));without.append(factual);full_route.append(routed_ce);without_logits.append(logits)
    without_ce=torch.stack(without);routed_terms=torch.stack(terms)
    joint_terms=positive(full_ce[None,:].expand(4,180),without_ce)
    values=dict(ZERO=sum((raw[name]*0.).sum() for name in BRANCHES),
        JOINT_GAP=(COEFFICIENT*joint_terms.sum()/720).to(joint.dtype),
        ROUTED_GAP=(COEFFICIENT*routed_terms.sum()/720).to(joint.dtype))
    if not torch.equal(values['JOINT_GAP'].detach(),values['ROUTED_GAP'].detach()):raise ValueError('Matched losses must have equal scalars and active samples')
    active=joint_terms.detach()>0
    with torch.no_grad():
        fixed32=F.normalize(weight.detach().float(),dim=1,eps=EPS)
        def formula32(z):return 30.*F.linear(F.normalize(z.detach().float(),dim=1,eps=EPS),fixed32)
        native32=formula32(full)
        fp32_error=float((full_logits.detach()-native32.double()).abs().max())
        ablated_error=max(float((logit-formula32(compose({name:value.detach() for name,value in raw.items()},gains,removed=branch))).double().abs().max()) for branch,logit in zip(BRANCHES,without_logits))
    def row(select):
        return dict(count=int(select.sum()),branch_decisions=int(select.sum())*4,
            active_decisions=int(active[:,select].sum()),active_origin_packet_count=int(active[:,select].any(0).sum()),
            mean_CE=float(full_ce[select].detach().mean()),errors_total=int((full_logits[select].detach().argmax(1)!=labels[select]).sum()),
            mean_positive_burden=float(joint_terms[:,select].detach().mean()),
            by_branch={name:dict(active=int(active[i,select].sum()),full_CE=float(full_ce[select].detach().mean()),without_CE=float(without_ce[i,select].mean()),mean_signed_burden=float((full_ce[select].detach()-without_ce[i,select]).mean()),mean_positive_burden=float(joint_terms[i,select].detach().mean()),without_errors=int((without_logits[i][select].argmax(1)!=labels[select]).sum())) for i,name in enumerate(BRANCHES)})
    shared=dict(row(torch.ones(180,dtype=torch.bool,device=joint.device)),source_role='L_s',
        by_RX={str(rx):row(receivers==rx) for rx in SOURCE_RX},by_TX={str(y):row(labels==y) for y in range(6)},
        by_day={str(day):row(days==day) for day in SOURCE_DAYS},
        fixed_denominator=720,unique_packets=180,actual_four_terms_exact=True,routed_numeric_CE_equal=True,
        loss_values_equal=True,default_W_stopped=True,auxiliary_all_three_gains_stopped=True,
        ablated_risk_stopped=True,context_stopped_for_ROUTED=True,shared_upstream_parameters_not_separate=True,
        stop_rule='actual evidence does not raise source trueclass risk relative to samepacket ablation',
        tolerance=TOLERANCE,coefficient=COEFFICIENT,extra_IQ_forwards=0,new_parameters=0,head_fit_updates=0,
        FP64_formula_decisions_this_call=1620,FP32_comparison_formula_decisions_this_call=900,
        FP64_FP32_full_max_abs_logit_difference=fp32_error,FP64_FP32_ablated_max_abs_logit_difference=ablated_error,
        feature_transform_at_deployment=False)
    summaries={mode:dict(shared,mode=mode,loss=float(values[mode].detach()),gradient_routing='all four attached' if mode=='JOINT_GAP' else 'only responsible component edge per term' if mode=='ROUTED_GAP' else 'connected zero') for mode in MODES}
    return values,summaries
