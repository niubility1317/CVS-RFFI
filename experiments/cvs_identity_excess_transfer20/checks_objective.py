"""Focused synthetic excess-risk assembly and stopped-factual checks."""
import copy
import json

import torch
from torch.nn import functional as F

from . import objective as api
from experiments.cvs_u_schedule50.counterfactual import _equal, _rng_snapshot


def _bank():
    cells = [(y,r,d,s) for s in (0,1) for y in range(6) for r in api.SOURCE_RX for d in api.SOURCE_DAYS]
    return dict(x=torch.zeros(180,2,256),y=torch.tensor([cell[0] for cell in cells]),
        rx=torch.tensor([cell[1] for cell in cells]),day=torch.tensor([cell[2] for cell in cells]),
        side=torch.tensor([cell[3] for cell in cells]),
        ids=[(1,r,d,2+s,940000+i) for i,(_,r,d,s) in enumerate(cells)])


def _summary(summary, mode):
    assert summary['matrix_count']==60 and summary['unique_matrices']==(15 if mode=='SAME_RX' else 60)
    assert summary['synthetic_decisions']==summary['fixed_decision_denominator']==3600
    assert summary['virtual_default_class_decisions']==summary['factual_default_class_decisions']==3600
    assert summary['total_default_class_decisions']==7200 and summary['default_class_decisions_this_call']==14400
    assert summary['FP64_formula_decisions_this_call']==summary['FP32_comparison_formula_decisions_this_call']==14400
    assert summary['unique_factual_packets']==180 and summary['factual_readout_reuse'] is False
    assert summary['references_per_origin']==5 and summary['origin_packet_count']==180
    assert 0<summary['active_origin_packet_count']<=180 and summary['active_decisions']>summary['active_origin_packet_count']
    assert summary['active_origin_fraction']==summary['active_origin_packet_count']/180
    assert summary['scale']==30. and summary['eps']==1e-4 and summary['coefficient']==.1
    assert summary['tolerance']==1e-6 and summary['stop_probability'] is None
    assert summary['stop_rule']==api.STOP_RULE and summary['factual_threshold_gradient_stopped']
    assert abs(summary['loss']-.1*summary['mean_positive_excess_risk'])<1e-15 if mode!='ZERO' else summary['loss']==0.
    exposure = summary['exposures']
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
    for rows in ('by_anchor_RX','by_reference_TX','by_target_TX'):
        assert all(all(key in row for key in ('mean_CE','mean_factual_CE','mean_excess_risk','mean_positive_excess_risk'))
                   for row in summary[rows].values())
    assert not summary['common_RX_additive_offset_loss_invariance_claimed']
    json.dumps(summary,allow_nan=False)


def check():
    torch.set_num_threads(2)
    bank = _bank(); gen = torch.Generator().manual_seed(153521)
    base = torch.randn(180,160,dtype=torch.float64,generator=gen,requires_grad=True)
    branches = [torch.randn(180,160,dtype=torch.float64,generator=gen,requires_grad=True) for _ in range(3)]
    gains = tuple(torch.nn.Parameter(torch.full((160,),value,dtype=torch.float64)) for value in (.1,.15,.2))
    joint = ((base+gains[0].tanh()*branches[0])+gains[1].tanh()*branches[1])+.2*gains[2].tanh()*branches[2]
    W = torch.nn.Parameter(torch.randn(6,160,dtype=torch.float64,generator=gen))
    before = copy.deepcopy(bank); joint_before = joint.detach().clone(); weight_before = W.detach().clone(); rng = _rng_snapshot()
    captured = []; original = api._decisions
    def record(virtual,factual,labels,weight):
        result = original(virtual,factual,labels,weight)
        captured.append((virtual.detach().clone(),factual.detach().clone(),labels.detach().clone(),result))
        return result
    api._decisions = record
    try: values,summaries = api.losses(joint,bank,W)
    finally: api._decisions = original
    assert tuple(values)==api.MODES and len(captured)==2
    for mode in api.MODES:
        _summary(summaries[mode],mode)
        gradients = torch.autograd.grad(values[mode],(base,*branches,*gains,W),allow_unused=True,retain_graph=True)
        assert gradients[-1] is None and all(torch.isfinite(gradient).all() for gradient in gradients[:-1])
        if mode=='ZERO':
            assert float(values[mode].detach())==0. and all(torch.count_nonzero(gradient)==0 for gradient in gradients[:-1])
        else:
            assert float(values[mode].detach())>0. and all(float(gradient.norm())>0. for gradient in gradients[:-1])
            assert all(float(gradients[0][bank['side']==side].norm())>0. for side in (0,1))
    # Independently enumerate all3600 origin labels and destination same-TX packets.
    for control,mode in enumerate(('SAME_RX','CROSS_RX')):
        virtuals,factuals,targets,origins = [],[],[],[]
        for pair in summaries[mode]['actual_pairs']:
            def indices(receiver,side):
                return sorted([i for i in range(180) if int(bank['rx'][i])==receiver
                    and int(bank['day'][i])==pair['day'] and int(bank['side'][i])==side],key=lambda i:int(bank['y'][i]))
            ia,ib = indices(pair['anchor_RX'],0),indices(pair['donor_RX'],1)
            for forward in (True,False):
                origin,destination = (ia,ib) if forward else (ib,ia)
                for y in range(6):
                    for c in range(6):
                        if y==c: continue
                        virtuals.append(joint.detach()[origin[y]]-joint.detach()[origin[c]]+joint.detach()[destination[c]])
                        factuals.append(joint.detach()[destination[y]])
                        targets.append(int(bank['y'][origin[y]])); origins.append(origin[y])
                        assert int(bank['y'][origin[y]])==int(bank['y'][destination[y]])
            assert pair['label_source_sides']==[0,1] and pair['destination_RX']==[pair['donor_RX'],pair['anchor_RX']]
            assert pair['factual']==['b_y','a_y']
        virtuals,factuals,targets,origins = torch.stack(virtuals),torch.stack(factuals),torch.tensor(targets),torch.tensor(origins)
        actual = captured[control]
        assert torch.equal(actual[0],virtuals) and torch.equal(actual[1],factuals) and torch.equal(actual[2],targets)
        normalized_W = F.normalize(W.detach(),dim=1,eps=1e-4)
        ce_virtual = F.cross_entropy(30.*F.linear(F.normalize(virtuals,dim=1,eps=1e-4),normalized_W),targets,reduction='none')
        ce_factual = F.cross_entropy(30.*F.linear(F.normalize(factuals,dim=1,eps=1e-4),normalized_W),targets,reduction='none')
        gap = ce_virtual-ce_factual; positive = torch.relu(gap-1e-6)
        assert torch.equal(values[mode].detach(),.1*positive.sum()/3600)
        assert summaries[mode]['mean_CE']==float(ce_virtual.mean())
        assert summaries[mode]['mean_factual_CE']==float(ce_factual.mean())
        assert summaries[mode]['mean_excess_risk']==float(gap.mean())
        assert summaries[mode]['mean_positive_excess_risk']==float(positive.mean())
        assert summaries[mode]['active_origin_packet_count']==int(torch.unique(origins[positive>0.]).numel())
        assert summaries[mode]['to_actual_destination_identity_RMS']==float((virtuals-factuals).square().mean().sqrt())
    # The factual leaf has no gradient, although its coordinates occur elsewhere
    # in full banks' virtual graphs. Test its isolated role directly.
    fixed_W = torch.nn.Parameter(torch.cat((torch.eye(6,dtype=torch.float64),torch.zeros(6,154,dtype=torch.float64)),dim=1))
    virtual = torch.zeros(1,160,dtype=torch.float64); virtual[0,1]=1.; virtual.requires_grad_(True)
    factual = torch.zeros(1,160,dtype=torch.float64); factual[0,0]=1.; factual.requires_grad_(True)
    label = torch.tensor([0])
    positive = original(virtual,factual,label,fixed_W)
    gradients = torch.autograd.grad(positive['stop_terms'].sum(),(virtual,factual,fixed_W),allow_unused=True)
    assert bool(positive['active'][0]) and float(gradients[0].norm())>0. and gradients[1] is None and gradients[2] is None
    negative = original(factual,virtual,label,fixed_W)
    assert float(negative['excess_risk'].detach())<0. and not bool(negative['active'][0])
    assert torch.count_nonzero(torch.autograd.grad(negative['stop_terms'].sum(),factual)[0])==0
    equal_virtual = virtual.detach().clone().requires_grad_(True)
    equal_factual = virtual.detach().clone().requires_grad_(True)
    equal = original(equal_virtual,equal_factual,label,fixed_W)
    equal_gradients = torch.autograd.grad(equal['stop_terms'].sum(),(equal_virtual,equal_factual),allow_unused=True)
    assert float(equal['excess_risk'].detach())==0. and float(equal['stop_terms'].detach())==0.
    assert torch.count_nonzero(equal_gradients[0])==0 and equal_gradients[1] is None
    # Exact post-tolerance boundary zero must have ReLU's zero derivative.
    boundary_virtual = torch.tensor([api.TOLERANCE],dtype=torch.float64,requires_grad=True)
    boundary_factual = torch.tensor([0.],dtype=torch.float64,requires_grad=True)
    boundary = api._excess(boundary_virtual,boundary_factual)
    boundary_gradients = torch.autograd.grad(boundary.sum(),(boundary_virtual,boundary_factual),allow_unused=True)
    assert float(boundary.detach())==0. and torch.count_nonzero(boundary_gradients[0])==0 and boundary_gradients[1] is None
    assert torch.equal(original(virtual.float(),factual.float(),label,fixed_W.float())['stop_terms'].detach(),positive['stop_terms'].detach())
    # Renaming true TX and the corresponding W rows preserves the same task.
    rename = torch.tensor([3,5,0,2,1,4]); renamed = copy.deepcopy(bank); renamed['y'] = rename[bank['y']]
    renamed_W = torch.empty_like(W.detach()); renamed_W[rename] = W.detach()
    renamed_joint = joint.detach().clone().requires_grad_(True)
    renamed_values,_ = api.losses(renamed_joint,renamed,renamed_W)
    permutation = torch.randperm(180,generator=gen)
    shuffled = {key:value[permutation].clone() if isinstance(value,torch.Tensor)
                else [value[i] for i in permutation.tolist()] for key,value in bank.items()}
    shuffled_joint = joint.detach()[permutation].clone().requires_grad_(True)
    shuffled_values,_ = api.losses(shuffled_joint,shuffled,W)
    plain = joint.detach().clone().requires_grad_(True); plain_values,_ = api.losses(plain,bank,W)
    for mode in api.MODES:
        assert torch.allclose(values[mode].detach(),renamed_values[mode].detach(),atol=1e-12,rtol=1e-12)
        assert torch.equal(values[mode].detach(),shuffled_values[mode].detach())
        gp = torch.autograd.grad(plain_values[mode],plain,retain_graph=True)[0]
        gr = torch.autograd.grad(renamed_values[mode],renamed_joint,retain_graph=True)[0]
        gs = torch.autograd.grad(shuffled_values[mode],shuffled_joint,retain_graph=True)[0]
        assert torch.allclose(gp,gr,atol=1e-12,rtol=1e-10) and torch.equal(gp,gs[torch.argsort(permutation)])
    # The real joint is FP32: analysis stays FP64 but returned loss and gradients
    # retain the input dtype instead of introducing a new training precision.
    joint32 = joint.detach().float().requires_grad_(True)
    values32,_ = api.losses(joint32,bank,W.detach().float())
    for mode in api.MODES:
        assert values32[mode].dtype==torch.float32
        grad32 = torch.autograd.grad(values32[mode],joint32,retain_graph=True)[0]
        assert grad32.dtype==torch.float32 and bool(torch.isfinite(grad32).all())
        assert torch.allclose(values32[mode].double(),values[mode].detach(),atol=1e-7,rtol=1e-6)
    # Ideal additive TX+RX reconstructs factual exactly, including a low-confidence
    # default head. Equal risk stops without imposing the previous .9 confidence.
    toy = torch.zeros(180,160,dtype=torch.float64)
    for i in range(180):
        toy[i,int(bank['y'][i])] = 2.
        toy[i,6+api.SOURCE_RX.index(int(bank['rx'][i]))] = .125*(api.SOURCE_RX.index(int(bank['rx'][i]))+1)
    toy.requires_grad_(True); toy_W = torch.zeros(6,160,dtype=torch.float64)
    toy_values,toy_summaries = api.losses(toy,bank,toy_W)
    for mode in api.MODES:
        assert toy_summaries[mode]['to_actual_destination_identity_RMS']==0.
        assert toy_summaries[mode]['mean_excess_risk']==toy_summaries[mode]['mean_positive_excess_risk']==0.
        assert toy_summaries[mode]['active_decisions']==toy_summaries[mode]['active_origin_packet_count']==0
        assert toy_summaries[mode]['mean_true_probability']<.9 and float(toy_values[mode].detach())==0.
        assert torch.count_nonzero(torch.autograd.grad(toy_values[mode],toy,retain_graph=True)[0])==0
    negatives = 0
    def reject(data=bank,weight=W,role='L_s'):
        nonlocal negatives
        try: api.losses(joint,data,weight,source_role=role)
        except ValueError: negatives+=1
        else: raise AssertionError('Invalid visible-L authority or default W input was accepted')
    for role in ('U_s','V','target'): reject(data=dict(bank,y=object()),role=role)
    for hint in ({'hide':True},{'source_role':'U_s'},{'tx_label_visible':False}): reject(data=dict(bank,y=object(),**hint))
    reject(weight=W[:5]); reject(weight=torch.full_like(W,float('nan')))
    assert _equal(before,bank) and torch.equal(joint_before,joint.detach()) and torch.equal(weight_before,W.detach())
    assert _equal(rng,_rng_snapshot()) and base.grad is None and W.grad is None
    assert all(gain.grad is None for gain in gains) and all(branch.grad is None for branch in branches)
    return dict(status='PASS',matrices60_virtual3600_factual3600_per_control=True,
        all5_references_both_direction_origin_labels_sameTX_destination_verified=True,
        hand_excess_default_cos_formula_and_full_denominator_verified=True,
        physical_origin_coverage_deduplicated180=True,group_endpoint4_each_triple_role20=True,
        isolated_factual_gradient_None=True,W_gradient_None_virtual_three_endpoints_gains_attached=True,
        positive_negative_equal_gap_and_exact_boundary_ReLU_verified=True,
        low_confidence_equal_factual_risk_loss_gradient_exact_zero=True,
        ZERO_exact_connected_zero=True,label_Wrow_rename_packetperm_loss_gradient_invariant=True,
        actual_FP32_joint_loss_gradient_dtype_preserved=True,
        virtual_factual_FP64_FP32_comparison_recorded=True,input_bank_features_weight_RNG_unchanged=True,
        focused_negative_checks=negatives,data_reads=0,checkpoint_reads=0,native_constructors=0,actual_updates=0)


if __name__=='__main__':
    print(json.dumps(check(),allow_nan=False))
