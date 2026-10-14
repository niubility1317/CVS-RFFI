"""Focused synthetic individual-reference transport assembly, without a model."""
import copy
import json

import torch

from . import objective as api
from experiments.cvs_u_schedule50.counterfactual import _equal, _rng_snapshot


def _bank():
    cells=[(y,r,d,s) for s in (0,1) for y in range(6) for r in api.SOURCE_RX for d in api.SOURCE_DAYS]
    return dict(x=torch.zeros(180,2,256),y=torch.tensor([cell[0] for cell in cells]),
        rx=torch.tensor([cell[1] for cell in cells]),day=torch.tensor([cell[2] for cell in cells]),
        side=torch.tensor([cell[3] for cell in cells]),
        ids=[(1,r,d,2+s,930000+i) for i,(_,r,d,s) in enumerate(cells)])


def _summary(summary,mode):
    assert summary['matrix_count']==60 and summary['unique_matrices']==(15 if mode=='SAME_RX' else 60)
    assert summary['synthetic_decisions']==summary['fixed_decision_denominator']==3600
    assert summary['default_class_decisions_this_call']==7200 and summary['references_per_origin']==5
    assert summary['origin_packet_count']==180 and 0<=summary['active_origin_packet_count']<=180
    assert summary['active_decisions']>180 and summary['active_origin_packet_count']==180
    assert summary['active_origin_fraction']==1.
    assert summary['scale']==30. and summary['eps']==1e-4 and summary['stop']==.9 and summary['coefficient']==.1
    assert not summary['common_RX_additive_offset_loss_invariance_claimed']
    exposure=summary['exposures']
    assert exposure['per_packet_min']==exposure['per_packet_max']==4
    assert exposure['total_endpoint_appearances']==720 and exposure['total_triple_role_appearances']==10800
    assert exposure['per_packet_triple_role_min']==exposure['per_packet_triple_role_max']==60
    for role in ('origin','reference_source','reference_destination'):
        assert exposure[role+'_appearances']==3600
        assert exposure['per_packet_'+role+'_min']==exposure['per_packet_'+role+'_max']==20
    assert exposure['by_RX']=={str(r):dict(anchor=72,donor=72,origin=720,reference_source=720,reference_destination=720) for r in api.SOURCE_RX}
    assert exposure['by_day']=={str(d):dict(anchor=120,donor=120,origin=1200,reference_source=1200,reference_destination=1200) for d in api.SOURCE_DAYS}
    assert exposure['by_TX']=={str(y):dict(anchor=60,donor=60,origin=600,reference_source=600,reference_destination=600) for y in range(6)}
    assert exposure['by_side']=={str(s):dict(anchor=360 if s==0 else 0,donor=360 if s==1 else 0,
        origin=1800,reference_source=1800,reference_destination=1800) for s in (0,1)}
    assert all(row['decision_count']==720 for row in summary['by_anchor_RX'].values())
    assert all(row['origin_packet_count']==(36 if mode=='SAME_RX' else 144) for row in summary['by_anchor_RX'].values())
    assert all(row['decision_count']==600 and row['origin_packet_count']==150 for row in summary['by_reference_TX'].values())
    assert all(row['decision_count']==600 and row['origin_packet_count']==30 for row in summary['by_target_TX'].values())
    assert sum(row['active_origin_packet_count'] for row in summary['by_target_TX'].values())==summary['active_origin_packet_count']
    assert summary['transfer_difference_RMS']>0. and summary['to_actual_destination_identity_RMS']>0.
    json.dumps(summary,allow_nan=False)


def check():
    torch.set_num_threads(2)
    bank=_bank();gen=torch.Generator().manual_seed(153520)
    base=torch.randn(180,160,dtype=torch.float64,generator=gen,requires_grad=True)
    branches=[torch.randn(180,160,dtype=torch.float64,generator=gen) for _ in range(3)]
    gains=tuple(torch.nn.Parameter(torch.full((160,),value,dtype=torch.float64)) for value in (.1,.15,.2))
    joint=((base+gains[0].tanh()*branches[0])+gains[1].tanh()*branches[1])+.2*gains[2].tanh()*branches[2]
    W=torch.nn.Parameter(torch.randn(6,160,dtype=torch.float64,generator=gen))
    before=copy.deepcopy(bank);joint_before=joint.detach().clone();weight_before=W.detach().clone();rng=_rng_snapshot()
    captured=[];original=api._decisions
    def record(views,labels,weight):
        captured.append((views.detach().clone(),labels.detach().clone()))
        return original(views,labels,weight)
    api._decisions=record
    try:values,summaries=api.losses(joint,bank,W)
    finally:api._decisions=original
    assert tuple(values)==api.MODES and len(captured)==2
    for mode in api.MODES:
        _summary(summaries[mode],mode)
        gradients=torch.autograd.grad(values[mode],(base,*gains,W),allow_unused=True,retain_graph=True)
        assert gradients[-1] is None
        assert all(torch.isfinite(gradient).all() for gradient in gradients[:-1])
        if mode=='ZERO':
            assert float(values[mode].detach())==0. and all(torch.count_nonzero(gradient)==0 for gradient in gradients[:-1])
        else:
            assert float(values[mode].detach())>0. and all(float(gradient.norm())>0. for gradient in gradients[:-1])
            assert all(float(gradients[0][bank['side']==side].norm())>0. for side in (0,1))
    # Independently hand-construct every (y,c), both labels and both destinations.
    for control,mode in enumerate(('SAME_RX','CROSS_RX')):
        expected_views,expected_labels,expected_destination=[] ,[],[]
        for pair in summaries[mode]['actual_pairs']:
            def indices(receiver,side):
                return sorted([i for i in range(180) if int(bank['rx'][i])==receiver
                    and int(bank['day'][i])==pair['day'] and int(bank['side'][i])==side],key=lambda i:int(bank['y'][i]))
            ia,ib=indices(pair['anchor_RX'],0),indices(pair['donor_RX'],1)
            for forward in (True,False):
                origin,destination=(ia,ib) if forward else (ib,ia)
                for y in range(6):
                    for c in range(6):
                        if y==c:continue
                        expected_views.append(joint.detach()[origin[y]]-joint.detach()[origin[c]]+joint.detach()[destination[c]])
                        expected_labels.append(int(bank['y'][origin[y]]));expected_destination.append(joint.detach()[destination[y]])
            assert pair['label_source_sides']==[0,1] and pair['destination_RX']==[pair['donor_RX'],pair['anchor_RX']]
        expected_views=torch.stack(expected_views);expected_labels=torch.tensor(expected_labels)
        assert expected_views.shape==(3600,160)
        assert torch.equal(captured[control][0],expected_views) and torch.equal(captured[control][1],expected_labels)
        error=expected_views-torch.stack(expected_destination)
        assert summaries[mode]['to_actual_destination_identity_RMS']==float(error.square().mean().sqrt())
        independent=original(expected_views,expected_labels,W)
        assert torch.equal(values[mode].detach(),.1*independent['stop_terms'].sum()/3600)
    # Rename TX identities and corresponding default W rows together.
    rename=torch.tensor([3,5,0,2,1,4]);renamed=copy.deepcopy(bank);renamed['y']=rename[bank['y']]
    renamed_W=torch.empty_like(W.detach());renamed_W[rename]=W.detach()
    renamed_joint=joint.detach().clone().requires_grad_(True)
    renamed_values,_=api.losses(renamed_joint,renamed,renamed_W)
    permutation=torch.randperm(180,generator=gen)
    shuffled={key:value[permutation].clone() if isinstance(value,torch.Tensor)
              else [value[i] for i in permutation.tolist()] for key,value in bank.items()}
    shuffled_joint=joint.detach()[permutation].clone().requires_grad_(True)
    shuffled_values,_=api.losses(shuffled_joint,shuffled,W)
    plain=joint.detach().clone().requires_grad_(True);plain_values,_=api.losses(plain,bank,W)
    for mode in api.MODES:
        assert torch.allclose(values[mode].detach(),renamed_values[mode].detach(),atol=1e-12,rtol=1e-12)
        assert torch.equal(values[mode].detach(),shuffled_values[mode].detach())
        gp=torch.autograd.grad(plain_values[mode],plain,retain_graph=True)[0]
        gr=torch.autograd.grad(renamed_values[mode],renamed_joint,retain_graph=True)[0]
        gs=torch.autograd.grad(shuffled_values[mode],shuffled_joint,retain_graph=True)[0]
        assert torch.allclose(gp,gr,atol=1e-12,rtol=1e-10)
        assert torch.equal(gp,gs[torch.argsort(permutation)])
    # In the ideal TX+RX toy, every transferred view is the actual destination y.
    toy=torch.zeros(180,160,dtype=torch.float64)
    for i in range(180):
        toy[i,int(bank['y'][i])]=2.
        toy[i,6+api.SOURCE_RX.index(int(bank['rx'][i]))]=.125*(api.SOURCE_RX.index(int(bank['rx'][i]))+1)
    toy.requires_grad_(True);toy_W=torch.zeros(6,160,dtype=torch.float64);toy_W[:,:6]=torch.eye(6,dtype=torch.float64)
    toy_values,toy_summaries=api.losses(toy,bank,toy_W)
    for mode in api.MODES:
        assert toy_summaries[mode]['to_actual_destination_identity_RMS']==0.
        assert toy_summaries[mode]['active_decisions']==toy_summaries[mode]['active_origin_packet_count']==0
        assert float(toy_values[mode].detach())==0.
        assert torch.count_nonzero(torch.autograd.grad(toy_values[mode],toy,retain_graph=True)[0])==0
        assert not toy_summaries[mode]['common_RX_additive_offset_loss_invariance_claimed']
    negatives=0
    def reject(data=bank,weight=W,role='L_s'):
        nonlocal negatives
        try:api.losses(joint,data,weight,source_role=role)
        except ValueError:negatives+=1
        else:raise AssertionError('Invalid visible-L authority or default W input was accepted')
    for role in ('U_s','V','target'):reject(data=dict(bank,y=object()),role=role)
    for hint in ({'hide':True},{'source_role':'U_s'},{'tx_label_visible':False}):reject(data=dict(bank,y=object(),**hint))
    reject(weight=W[:5]);reject(weight=torch.full_like(W,float('nan')))
    assert _equal(before,bank) and torch.equal(joint_before,joint.detach()) and torch.equal(weight_before,W.detach())
    assert _equal(rng,_rng_snapshot()) and base.grad is None and W.grad is None and all(gain.grad is None for gain in gains)
    return dict(status='PASS',matrices60_decisions3600_per_control=True,all5_different_TX_references=True,
        both_direction_formulas_origin_labels_destination_comparison_verified=True,
        physical_origin_coverage_deduplicated180=True,group_endpoint4_each_triple_role20=True,
        RX_day_TX_side_exposure_matched=True,ZERO_exact_connected_zero=True,
        W_gradient_None_joint_both_sides_three_gains_attached=True,
        label_Wrow_rename_and_packetperm_loss_gradient_invariant=True,
        ideal_TX_plus_RX_destination_reconstruction_exact=True,stopped_loss_gradient_exact_zero=True,
        no_RX_offset_loss_invariance_claim=True,input_bank_features_weight_RNG_unchanged=True,
        focused_negative_checks=negatives,data_reads=0,checkpoint_reads=0,native_constructors=0,actual_updates=0)


if __name__=='__main__':
    print(json.dumps(check(),allow_nan=False))
