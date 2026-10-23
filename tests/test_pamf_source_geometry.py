from copy import deepcopy
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_ssl_research.source_geometry import fixed_indices,geometry,capture_deployed,describe,fixed_head_interventions,LEVELS


def test_fixed_source_cohort_has_no_U_truth_and_all_existing_cells():
    y=np.repeat(np.arange(6),5*3*64);rx=np.tile(np.repeat([1,3,4,6,8],3*64),6);day=np.tile(np.repeat([1,2,3],64),6*5)
    metadata=dict(y=y,rx_i=rx,day_i=day);v=fixed_indices(metadata,'V');u=fixed_indices(dict(rx_i=rx,day_i=day),'U_s')
    assert len(v)==720 and len(set(v))==720 and len(u)==960 and len(set(u))==960
    assert fixed_indices(dict(rx_i=rx,day_i=day),'U_s')==u
    assert all(sum(y[i]==c and rx[i]==r and day[i]==d for i in v)==8 for c in range(6) for r in (1,3,4,6,8) for d in (1,2,3))
    with pytest.raises(ValueError,match='truth'):fixed_indices(metadata,'U_s')
    with pytest.raises(ValueError,match='Only existing source'):fixed_indices(metadata,'target')


@pytest.mark.parametrize('mode',['unit_only','cohort_native_cric'])
def test_TX_and_conditional_RX_scatter_has_known_decomposition(mode):
    h=torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    y=torch.tensor([0,0,1,1]);rx=torch.tensor([1,3,1,3]);w=torch.ones(4)
    result=geometry(h,y,rx,w,mode)
    assert result['class_fraction_of_total']==pytest.approx(.5)
    assert result['conditional_RX_fraction_of_class_residual']==pytest.approx(1.)
    assert result['conditional_RX_fraction_of_total']==pytest.approx(.5)
    assert result['covered_class_RX_cells']==4
    collapsed=geometry(torch.zeros_like(h),y,rx,w,mode)
    assert collapsed['status']=='ZERO_SPREAD' and collapsed['class_fraction_of_total'] is None
    assert geometry(h,y,rx,torch.zeros(4),mode)['status']=='EMPTY_SELECTION'


def test_rejected_reference_samples_do_not_change_normalization_or_scatter():
    h=torch.randn(8,4);y=torch.tensor([0,0,1,1]*2);rx=torch.tensor([1,3,1,3]*2);mask=torch.ones(8)
    a=geometry(h,y,rx,mask,'cohort_native_cric')
    b=geometry(torch.cat((h,torch.full((1,4),1e6))),torch.cat((y,torch.tensor([5]))),torch.cat((rx,torch.tensor([8]))),
        torch.cat((mask,torch.zeros(1))),'cohort_native_cric')
    assert {k:v for k,v in a.items() if k!='samples_total'}=={k:v for k,v in b.items() if k!='samples_total'}


class Reference(nn.Module):
    def __init__(self):
        super().__init__();self.core=nn.Module();core=nn.Module();self.core.id_backbone=core
        core.time_down=nn.Conv1d(2,4,1);core.t2=nn.Module();core.t2.pool=nn.Conv1d(4,4,1);self.base=nn.Linear(4,160)
        self.sequence_pool=nn.Linear(4,160);self.response_projection=nn.Linear(4,160);self.response_gain=nn.Parameter(torch.ones(160)*.2)
        self.head=nn.Linear(160,6);self.feature_calls=0
    def features(self,x):
        self.feature_calls+=1;h=self.core.id_backbone.t2.pool(self.core.id_backbone.time_down(x)).mean(-1)
        return self.base(h)+self.sequence_pool(h)+self.response_gain.tanh()*self.response_projection(h)
    def classify_features(self,h):return self.head(h)


def test_one_frozen_forward_captures_true_additive_contributions_without_model_change():
    model=Reference().eval().requires_grad_(False);before=deepcopy(model.state_dict());x=torch.randn(6,2,256)
    features,logits=capture_deployed(model,x)
    assert set(features)==set(LEVELS) and logits.shape==(6,6) and model.feature_calls==1
    torch.testing.assert_close(features['joint160'],features['core160']+features['signature160']+features['reference160'])
    assert all(not m._forward_hooks for m in model.modules())
    for key,value in before.items():torch.testing.assert_close(value,model.state_dict()[key])
    summary=describe(features,torch.tensor([0,0,1,1,2,2]),torch.tensor([1,3]*3),torch.ones(6))
    assert set(summary['geometry'])=={'unit_only','cohort_native_cric'} and summary['branch_contributions']['signature160']['mean_norm']>0


def test_capture_rejects_trainable_or_training_reader_and_removes_hooks_on_failure():
    model=Reference();x=torch.randn(6,2,256)
    with pytest.raises(ValueError,match='Frozen eval'):capture_deployed(model,x)
    model.eval()
    with pytest.raises(ValueError,match='Frozen eval'):capture_deployed(model,x)
    model.requires_grad_(False)
    def fail(x):raise RuntimeError('source forward failed')
    model.features=fail
    with pytest.raises(RuntimeError,match='source forward failed'):capture_deployed(model,x)
    assert all(not m._forward_hooks for m in model.modules())


class FixedHead(nn.Module):
    def __init__(self):
        super().__init__();self.head=nn.Linear(160,6,bias=False);self.head_calls=0
        with torch.no_grad():self.head.weight.zero_();self.head.weight[:,:6].copy_(torch.eye(6))
    def classify_features(self,x):self.head_calls+=1;return self.head(x)


def intervention_fixture():
    model=FixedHead().eval().requires_grad_(False);core=torch.zeros(4,160);core[:2,0]=4;core[2:,1]=4
    reference=torch.zeros_like(core);reference[:,8]=torch.tensor([.8,-.8,.8,-.8])
    signature=torch.zeros_like(core);signature[:2,9]=.2;signature[2:,9]=-.2
    features=dict(core160=core,reference160=reference,signature160=signature,joint160=core+reference+signature,
        time_down=torch.zeros(4,8),**{'t2.pool':torch.zeros(4,8)})
    return model,features,torch.tensor([0,0,1,1]),torch.tensor([1,0,3,2]),torch.ones(4,dtype=torch.bool)


def test_frozen_head_interventions_change_RX_components_without_refitting_or_backbone_calls():
    model,features,y,partner,valid=intervention_fixture();before=deepcopy(model.state_dict())
    result=fixed_head_interventions(model,features,y,partner,valid)
    assert model.head_calls==7 and result['existing_head_forward_calls']==7 and result['extra_backbone_forwards']==0
    assert all(r['label_agreements']==4 and r['mean_label_CE']>0 for r in result['rows'])
    swap=next(r for r in result['rows'] if r['intervention']=='reference_swap')
    same=next(r for r in result['rows'] if r['intervention']=='joint')
    assert swap['mean_relative_feature_displacement']>0 and same['mean_relative_feature_displacement']==0
    for key,value in before.items():torch.testing.assert_close(value,model.state_dict()[key])


def test_head_intervention_rejects_bad_partner_or_trainable_state():
    model,features,y,partner,valid=intervention_fixture();bad=partner.clone();bad[0]=4
    with pytest.raises(ValueError,match='partner indices'):fixed_head_interventions(model,features,y,bad,valid)
    model.train()
    with pytest.raises(ValueError,match='frozen'):fixed_head_interventions(model,features,y,partner,valid)
