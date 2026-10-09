"""Focused real-CVS distribution R checks and negative split/day controls."""
import json
from unittest.mock import patch

import torch

from .receiver import (receiver_audit, make_relations, ReceiverDistributionAction,
    OnlineReceiver, packet_statistics, margin_vector, _fixed, intermediate)
from experiments.cvs_multi_action_audit.receiver_checks import run as legacy_invariants


def fixture(prefix, device='cpu'):
    y=torch.arange(3).repeat_interleave(24)
    rx=torch.arange(2).repeat_interleave(12).repeat(3)
    day=torch.arange(2).repeat_interleave(6).repeat(6)
    g=torch.Generator().manual_seed(182)
    x=torch.randn(72,2,256,generator=g)
    ids=[prefix+str(i) for i in range(len(x))]
    return dict(x=torch.cat((x,x+.001)).to(device),y=y.repeat(2).to(device),
        rx=rx.repeat(2).to(device),day=day.repeat(2).to(device),ids=ids*2,
        condition=['clean']*72+['source_practical_mid']*72)


def run(device='cpu'):
    torch.set_num_threads(1)
    from experiments.cvs_multi_disentangle.checks import identity
    from experiments.cvs_multi_disentangle import design
    config=design.config(design.rows()[0])
    model=identity(config,device).id_backbone.eval()
    fit,audit=fixture('fit-',device),fixture('audit-',device)
    relations,_=make_relations(fit)
    assert len(relations)==24
    for r in relations:
        assert r['y'] not in r['donor_tx']
        ids=[set(fit['ids'][i] for i in r[k]) for k in ('descriptor0','descriptor1','recipient','destination')]
        assert all(not ids[i]&ids[j] for i in range(4) for j in range(i))
        for k in ('descriptor0','descriptor1','recipient','destination'):
            assert {int(fit['day'][i]) for i in r[k]}=={r['day']}
    unmatched=fixture('day-',device)
    unmatched['day']=unmatched['rx'].clone()
    assert not make_relations(unmatched)[0]
    try:
        receiver_audit(model,fit,fit,torch.Generator().manual_seed(9),steps=1)
    except ValueError as e:assert 'overlap' in str(e)
    else:raise AssertionError('Fit audit physical leak accepted')
    original={k:v.clone() for k,v in model.state_dict().items()}
    result=receiver_audit(model,fit,audit,torch.Generator().manual_seed(9),steps=2)
    assert result['report']['status']=='VERIFIED'
    assert len(result['report']['predictive_metrics'])==8
    assert result['report']['anchor_supervision_active']
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())
    assert all(p.grad is None for p in model.parameters())
    branch=ReceiverDistributionAction().to(device)
    h=torch.randn(5,349,device=device);v=torch.randn(1,49,device=device)
    assert torch.equal(branch(h,v,v),torch.zeros_like(h))
    with patch('experiments.cvs_multi_state_action.receiver.identity_from_intermediate',lambda _,h:h.square()), \
         patch('experiments.cvs_multi_state_action.receiver.classify_intermediate',lambda _,h:h.square()):
        statistics=packet_statistics(None,torch.tensor([[1.,0.],[3.,0.]]),0)
    assert statistics['z'][0]==5 and statistics['margin'][0]==5
    assert margin_vector(torch.tensor([[3.,1.,2.]]),0).tolist()==[[2.,1.]]
    with _fixed(model):
        with torch.no_grad():h=intermediate(model,fit['x'])
        online=OnlineReceiver(max_age_steps=10).to(device)
        args=dict(x=fit['x'],h=h,y=fit['y'],rx=fit['rx'],day=fit['day'],
            ids=fit['ids'],condition=fit['condition'],reference_identity=model)
        online.update(**args,step=0,version='v1')
        online.update(**args,step=9,version='v1')
        assert all(e['step']==0 for e in online.entries.values())
        loss,stats=online.fit_loss(model,9,'v1')
        assert loss is not None and stats['active']
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in online.parameters())
        fit_ids={e['pid'] for bags in online._groups(9,'v1',0).values() for role in bags for e in role}
        held_ids={e['pid'] for bags in online._groups(9,'v1',1).values() for role in bags for e in role}
        assert not fit_ids&held_ids
        calibration=online.calibrate(model,9,'v1')
        assert calibration['active']
        # Nonheldout IDs cannot contribute even when a caller supplies them.
        import hashlib
        forbidden=[i for i,pid in enumerate(fit['ids']) if (int(hashlib.sha256(pid.encode()).hexdigest(),16)//3)%2==0]
        reject=online.calibrate_live(model,h[forbidden],fit['y'][forbidden],fit['rx'][forbidden],
            fit['day'][forbidden],[fit['ids'][i] for i in forbidden],
            [fit['condition'][i] for i in forbidden],9,'v1')
        assert not reject['active'] and not reject['counts']
        live=online.calibrate_live(model,h,fit['y'],fit['rx'],fit['day'],fit['ids'],fit['condition'],9,'v1')
        assert live['active'] and live['physical_holdout']
        again=online.calibrate_live(model,h,fit['y'],fit['rx'],fit['day'],fit['ids'],fit['condition'],9,'v1')
        assert again['counts']==live['counts']
        for c,count in live['counts'].items():
            if count<16:assert live['weights'][c]==0
        # Actual CVS identity group path differentiates current recipients;
        # auxiliary action and all cached destination statistics stay detached.
        online.zero_grad(set_to_none=True)
        live_h=h.detach().clone().requires_grad_(True)
        group_loss,group_stats=online.identity_loss(model,live_h,fit['y'],fit['rx'],fit['day'],
            fit['ids'],fit['condition'],9,'v1')
        assert group_stats['active'] and group_loss is not None
        group_loss.backward()
        assert live_h.grad is not None and live_h.grad.norm()>0
        assert all(p.grad is None for p in online.parameters())
        assert all(not e['h'].requires_grad and not e['z'].requires_grad for e in online.entries.values())
        recipient={i for i,pid in enumerate(fit['ids']) if int(hashlib.sha256(pid.encode()).hexdigest(),16)%3==1}
        assert all(float(live_h.grad[i].abs().sum())==0 for i in range(len(h)) if i not in recipient)
        online.expire(11,'v1');assert not online.entries
        online.update(**args,step=12,version='v1')
        online.expire(12,'v2');assert not online.entries and not online.live_calibration
    return dict(status='VERIFIED',device=device,real_cvs_steps=2,records=len(result['report']['records']),
        day_mismatch_rejected=True,physical_overlap_rejected=True,disjoint_bags=True,
        cross_tx_code=True,packetwise_nonlinearity=True,full_margin_vector=True,
        joint_conditions=True,identity_unchanged=True,online_fit_holdout_disjoint=True,
        live_identity_group_gradient=True,live_action_gradient_detached=True,
        live_calibration_fit_ids_rejected=True,live_calibration_repeated_ids_not_recounted=True,
        online_expiry_not_refreshed=True,online_reference_refresh_clears=True,
        legacy_statistics=legacy_invariants())


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    print(json.dumps(run(args.device),indent=2))
