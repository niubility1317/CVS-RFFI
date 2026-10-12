"""Retain main identity CE/SSL; add only fixed L relation or budget control."""
from torch.nn import functional as F
from .base_objective import objective as base,BASE_WEIGHTED_KEYS as OLD_KEYS,AUDIT_ONLY_KEYS
from .core import support,capture,reference,loss
from .update import rng_state,set_rng
BASE_WEIGHTED_KEYS=(*OLD_KEYS,'relation')

def objective(c,model,bank,L,li,U,ui,step,device):
    rc=c['relation'];active=step>c['semi']['warmup_updates'] and step%4==0
    levels=rc['levels'] if active else []
    with capture(model,levels,first_only=True) as features:
        total,parts,w,detail=base(c,model,bank,L,li,U,ui,step,device)
    joint=detail.pop('_query_joint');logits=detail.pop('_query_logits');zero=total*0
    parts['relation']=zero;w['relation']=zero;detail['relation']=dict(active=False,L_packets=0,forwards=0,extra_backward_calls=0)
    if active:
        si,evidence=support(L,li,c['model_seed'],step);x=L['x'][si].to(device);y=L['y'][li].to(device)
        if rc['mode']=='extra_CE':
            end=rng_state()
            try:extra=model(x)
            finally:set_rng(end)
            value=F.cross_entropy(extra,L['y'][si].to(device));records={};detail['relation'].update(control_L_accuracy=float((extra.argmax(1)==L['y'][si].to(device)).float().mean()),main_rng_restored=True)
        else:
            refs=reference(model,x,levels);features['id_feat_joint']=joint;values=[];records={}
            for level in levels:
                value,record=loss(features[level],refs[level],y,L['rx'][li].to(device),rc['mode'],rc['temperature']);values.append(value)
                pred=record.pop('predictions');head_correct=logits.detach().argmax(1)==y;rel_correct=pred==y
                record.update(head_wrong_relation_correct=int((~head_correct&rel_correct).sum()),head_correct_relation_wrong=int((head_correct&~rel_correct).sum()))
                records[level]=record
            value=sum(values)/len(values)
        parts['relation']=value;w['relation']=value*rc['weight']*detail['ramp'];total=sum(w[k] for k in BASE_WEIGHTED_KEYS)
        margin=logits.detach().gather(1,y[:,None]).squeeze(1)-logits.detach().masked_fill(F.one_hot(y,6).bool(),float('-inf')).max(1).values
        detail['relation'].update(evidence,active=True,forwards=1,extra_backward_calls=0,reference_has_gradient=rc['mode']=='extra_CE',prototype_grad=False,levels=records,raw_loss=float(value.detach()),weighted_loss=float(w['relation'].detach()),head_L_accuracy=detail['L_accuracy'],head_L_margin=float(margin.mean()),query_features_reused=True,level_average=len(levels),U_labels_used=False,V_access=False,target_access=False)
    return total,parts,w,detail
