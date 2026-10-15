"""Matched-value, matched-auxiliary-gradient-norm source controls."""
from collections import Counter
import math
import torch
from experiments.cvs_evidence_responsibility20.objective import BRANCHES,compose,ce,positive

RX=(1,3,4,6,8)

def subset(bank,held_rx,training,source_role='L_s'):
    if source_role!='L_s':raise ValueError('Visible source L only')
    if held_rx not in RX or not isinstance(training,bool):raise ValueError('Fixed held RX/training selector required')
    if set(bank)!={'x','y','rx','day','side','ids'}:raise ValueError('Validated original L180 bank required')
    if any(value.shape[0]!=180 for key,value in bank.items() if key!='ids') or len(bank['ids'])!=180:
        raise ValueError('Complete L180 bank required')
    mask=bank['rx']!=held_rx if training else bank['rx']==held_rx
    indices=mask.nonzero().flatten().tolist()
    result={key:value[mask] for key,value in bank.items() if key!='ids'}
    result['ids']=[bank['ids'][i] for i in indices]
    result.update(role='L_s',held_rx=held_rx,training=training)
    validate_bank(result,source_role)
    return result

def validate_bank(bank,source_role='L_s'):
    if source_role!='L_s':raise ValueError('Visible source L only')
    if bank.get('role')!='L_s' or set(bank)!={'x','y','rx','day','side','ids','role','held_rx','training'}:
        raise ValueError('Only explicitly visible source packets are permitted')
    held=bank['held_rx'];training=bank['training']
    if held not in RX or not isinstance(training,bool):raise ValueError('Fixed RX-blocked bank required')
    receivers=tuple(rx for rx in RX if rx!=held) if training else (held,)
    n=144 if training else 36;x=bank['x']
    if x.shape!=(n,2,256) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()):
        raise ValueError('Finite FP32 IQ shape differs')
    for key in ('y','rx','day','side'):
        value=bank[key]
        if value.shape!=(n,) or value.dtype!=torch.long or value.device!=x.device:
            raise ValueError('Integer source grouping differs')
    expected={(rx,day,side,y):1 for rx in receivers for day in (1,2,3) for side in (0,1) for y in range(6)}
    if Counter(zip(bank['rx'].tolist(),bank['day'].tolist(),bank['side'].tolist(),bank['y'].tolist()))!=expected:
        raise ValueError('Held RX exclusion or balanced source cells differ')
    ids=bank['ids']
    if len(ids)!=n or any(len(identity)!=5 or any(type(v)is not int or v<0 for v in identity) for identity in ids):
        raise ValueError('Complete native physical IDs required')
    if len({tuple(identity) for identity in ids})!=n or len({identity[4] for identity in ids})!=n:
        raise ValueError('Duplicate source physical IDs')
    if any(identity[1]!=int(bank['rx'][i]) or identity[2]!=int(bank['day'][i]) for i,identity in enumerate(ids)):
        raise ValueError('Physical RX/day mismatch')
    return bank['y']

def losses(components,bank,weight,gains,source_role='L_s'):
    labels=validate_bank(bank,source_role)
    if not bank['training']:raise ValueError('Evaluation RX cannot construct training loss')
    if not set((*BRANCHES,'joint')).issubset(components):raise ValueError('Actual four components required')
    raw={key:components[key] for key in BRANCHES};joint=components['joint'];n=144
    if any(value.shape!=(n,160) or value.device!=joint.device or not bool(torch.isfinite(value).all()) for value in raw.values()):
        raise ValueError('Finite current raw components required')
    if weight.shape!=(6,160) or len(gains)!=3 or any(gain.shape!=(160,) for gain in gains):
        raise ValueError('Original default W/three gain shape differs')
    full=compose(raw,gains)
    if not torch.equal(full.detach(),joint.detach()):raise ValueError('Actual four-term value differs')
    full_ce,_=ce(full,labels,weight);joint_terms=[];routed_terms=[]
    for name in BRANCHES:
        with torch.no_grad():without,_=ce(compose(raw,gains,removed=name),labels,weight)
        view=compose(raw,gains,only=name);routed,_=ce(view,labels,weight)
        if not torch.equal(view.detach(),full.detach()) or not torch.equal(routed.detach(),full_ce.detach()):
            raise ValueError('Component route changes its numeric forward')
        joint_terms.append(positive(full_ce,without));routed_terms.append(positive(routed,without))
    values=dict(ZERO=sum((value*0).sum() for value in raw.values()),
        JOINT_GAP=(.1*torch.stack(joint_terms).sum()/576).float(),
        ROUTED_GAP=(.1*torch.stack(routed_terms).sum()/576).float())
    if not torch.equal(values['JOINT_GAP'].detach(),values['ROUTED_GAP'].detach()):raise ValueError('Loss values differ')
    return values,dict(active_decisions=int((torch.stack(joint_terms).detach()>0).sum()),
        active_packets=int((torch.stack(joint_terms).detach()>0).any(0).sum()),denominator=576,
        current_receivers=sorted(set(bank['rx'].tolist())),held_rx=bank['held_rx'],held_RX_used_for_gradient=False)

def matched_value(loss,scale):
    if not isinstance(scale,(float,int)) or not math.isfinite(scale) or scale<0:
        raise ValueError('Finite detached source-only norm factor required')
    return loss.detach()+float(scale)*(loss-loss.detach())
