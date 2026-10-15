"""Focused source-role, RX exclusion and matched-value gradient checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy,json
import torch
from .objective import subset,validate_bank,losses,matched_value
from experiments.cvs_evidence_responsibility20.objective import compose,BRANCHES

def bank():
    metadata=[(rx,day,side,y) for rx in (1,3,4,6,8) for day in (1,2,3) for side in (0,1) for y in range(6)]
    return dict(x=torch.randn(180,2,256),
        y=torch.tensor([v[3] for v in metadata]),rx=torch.tensor([v[0] for v in metadata]),
        day=torch.tensor([v[1] for v in metadata]),side=torch.tensor([v[2] for v in metadata]),
        ids=[(v[3],v[0],v[1],v[2],i) for i,v in enumerate(metadata)])

def check():
    torch.set_num_threads(2);torch.manual_seed(150015)
    original=bank();negatives=0
    def rejects(callback):
        nonlocal negatives
        try:callback()
        except ValueError:negatives+=1;return
        raise AssertionError('Illegal RX-blocked source input was accepted')
    for held in (1,3,4,6,8):
        train=subset(original,held,True);test=subset(original,held,False)
        assert len(train['ids'])==144 and len(test['ids'])==36
        assert held not in set(train['rx'].tolist()) and set(test['rx'].tolist())=={held}
        assert not set(map(tuple,train['ids'])) & set(map(tuple,test['ids']))
        rejects(lambda:subset(original,held,True,'U_s'))
        bad=copy.deepcopy(train);bad['role']='V';rejects(lambda:validate_bank(bad))
        bad=copy.deepcopy(train);bad['rx'][0]=held;rejects(lambda:validate_bank(bad))
        bad=copy.deepcopy(train);bad['y'][0]=-1;rejects(lambda:validate_bank(bad))
        bad=copy.deepcopy(train);bad['ids'][1]=bad['ids'][0];rejects(lambda:validate_bank(bad))
    current=subset(original,8,True)
    raw={name:torch.randn(144,160,requires_grad=True) for name in BRANCHES}
    gains=tuple(torch.full((160,),.2,requires_grad=True) for _ in range(3))
    weight=torch.randn(6,160,requires_grad=True)
    components=dict(raw,joint=compose(raw,gains))
    values,facts=losses(components,current,weight,gains)
    params=tuple(raw.values())+(weight,)+gains
    def gradient(value):return torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
    def norm(gs):return sum(float(g.double().square().sum()) for g in gs if g is not None)**.5
    joint=gradient(values['JOINT_GAP']);routed=gradient(values['ROUTED_GAP'])
    factor=norm(routed)/norm(joint);matched=matched_value(values['JOINT_GAP'],factor);gmatch=gradient(matched)
    assert torch.equal(matched.detach(),values['JOINT_GAP'].detach()) and torch.equal(matched.detach(),values['ROUTED_GAP'].detach())
    assert abs(norm(gmatch)-norm(routed))<=5e-6*norm(routed)
    assert all(g is None for gs in (joint,routed,gmatch) for g in gs[4:])
    assert all(torch.allclose(a,b*factor,atol=2e-7,rtol=1e-5) for a,b in zip(gmatch[:4],joint[:4]))
    rejects(lambda:losses(components,current,weight,gains,'U_s'))
    rejects(lambda:losses(components,subset(original,8,False),weight,gains))
    bad=dict(components);del bad['identity'];rejects(lambda:losses(bad,current,weight,gains))
    for scale in (-1.,float('nan'),float('inf')):rejects(lambda:matched_value(values['JOINT_GAP'],scale))
    class RolePoison(dict):
        def __getitem__(self,key):raise AssertionError('Labels/data read before role rejection')
    rejects(lambda:losses(components,RolePoison(),weight,gains,'target'))
    return dict(status='PASS',focused_negatives=negatives,all5_RX_exclusions=True,
        training144_held36=True,same_numeric_loss_3arms=True,matched_F_auxiliary_gradient_norm=True,
        norm_control_keeps_JOINT_direction=True,default_W_three_gains_auxiliary_None=True,
        U_V_target_rejected_before_labels=True,source_IQ_checkpoint_reads=0,actual_optimizer_updates=0,
        limitation='Synthetic raw-component tensors; real checkpoint smoke runs first in the source launcher')

if __name__=='__main__':print(json.dumps(check()))
