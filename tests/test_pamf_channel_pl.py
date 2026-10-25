from copy import deepcopy
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from experiments.pamf_channel_pl.policy import RECIPE,raw_predictions,channel_gate,dose_conserving_loss,validate_recipe
from experiments.pamf_channel_pl.objective import ChannelPL,installed


@pytest.mark.parametrize('classes',[6,10])
def test_masks_require_original_acceptance_same_pseudo_and_channel_confidence(classes):
    raw=torch.zeros(4,classes);raw[:3,0]=10.
    _,pseudo,accepted=raw_predictions(raw)
    candidate=torch.arange(4);aug=torch.zeros(4,classes);aug[0,0]=10.;aug[1,1]=10.;aug[3,0]=10.
    assert channel_gate(pseudo,accepted,candidate,aug).tolist()==[True,False,False,False]


def test_empty_transfer_is_exact_original_PL_and_has_original_gradient():
    z=torch.randn(7,6,requires_grad=True);pseudo=torch.tensor([0,1,2,3,4,5,0]);mask=torch.tensor([1,1,0,1,0,1,1],dtype=torch.bool)
    ce=F.cross_entropy(z,pseudo,reduction='none');old=(ce*mask).mean()
    new,delta=dose_conserving_loss(ce,mask,torch.empty(0,dtype=torch.long),None,pseudo)
    assert torch.equal(new,old) and delta.item()==0.
    assert torch.equal(torch.autograd.grad(new,z,retain_graph=True)[0],torch.autograd.grad(old,z)[0])


def test_per_sample_dose_and_gradient_are_convex_and_keep_rejected_samples_zero():
    raw=torch.tensor([1.,2.,3.,4.],requires_grad=True);mask=torch.tensor([1,1,0,1],dtype=torch.bool);pseudo=torch.zeros(4,dtype=torch.long)
    idx=torch.tensor([0,3]);aug=torch.tensor([[2.,0.],[0.,2.]],requires_grad=True)
    value,_=dose_conserving_loss(raw,mask,idx,aug,pseudo)
    asce=F.cross_entropy(aug,pseudo[idx],reduction='none')
    expected=(.8*raw[0]+.2*asce[0]+raw[1]+.8*raw[3]+.2*asce[1])/4
    torch.testing.assert_close(value,expected)
    torch.testing.assert_close(torch.autograd.grad(value,raw,retain_graph=True)[0],torch.tensor([.2,.25,0.,.2]))
    assert torch.autograd.grad(value,aug)[0].abs().sum()>0


@pytest.mark.parametrize('bad',['unaccepted','duplicate','shape','nonfinite'])
def test_rejects_invalid_transfers(bad):
    ce=torch.ones(4,requires_grad=True);mask=torch.tensor([1,1,0,1],dtype=torch.bool);pseudo=torch.zeros(4,dtype=torch.long)
    idx=torch.tensor([0,1]);aug=torch.ones(2,6,requires_grad=True)
    if bad=='unaccepted':idx=torch.tensor([0,2])
    elif bad=='duplicate':idx=torch.tensor([0,0])
    elif bad=='shape':aug=torch.ones(1,6)
    else:aug=torch.full((2,6),float('nan'))
    with pytest.raises(ValueError):dose_conserving_loss(ce,mask,idx,aug,pseudo)


@pytest.mark.parametrize('key,value',[('raw_threshold',.98),('channel_threshold',.9),('channel_to_raw_CE_ratio',1.),('U_weight',6.)])
def test_fixed_recipe_cannot_silently_be_retuned(key,value):
    p=deepcopy(RECIPE);p[key]=value
    with pytest.raises(ValueError):validate_recipe(p)


class Toy(nn.Module):
    def __init__(self,classes=6,scale=8.):
        super().__init__();self.w=nn.Parameter(torch.tensor(scale));self.classes=classes;self.train_packets=[]
        self._feature_l_steps=1;self._cd_U_seen=self._cd_U_accepted=0
    def forward(self,x):
        out=torch.zeros(len(x),self.classes,device=x.device);out[:,0]=self.w*x[:,0,0];return out
    def forward_identity_only(self,x,domain_labels):
        if self.training:self.train_packets.append(len(x))
        return dict(tx_logits=self(x))


def setup(classes=6,scale=8.,aug_scale=1.1,truth=False):
    model=Toy(classes,scale);x=torch.ones(8,2,256);meta={'tx_label_visible':torch.full((8,),truth)}
    batch=(x,torch.zeros(8,dtype=torch.long) if truth else torch.full((8,),-1,dtype=torch.long),torch.zeros(8,dtype=torch.long),meta)
    model._cd_U_loader=[batch];model._cd_U_iter=iter(model._cd_U_loader)
    c=dict(channel_pl=deepcopy(RECIPE),step_offset=4440,source_rxs=[1],source_days=[1,2,3],num_classes=classes)
    calls=[]
    def simulator(x,metadata,epoch,step):calls.append(len(x));return x*aug_scale
    return model,ChannelPL(c,simulator=simulator),calls


@pytest.mark.parametrize('classes',[6,10])
def test_real_objective_uses_only_accepted_half_and_original_head(classes):
    model,obj,calls=setup(classes);loss=obj.loss(model,'cpu');loss.backward();s=obj.state_dict()
    assert calls==[4] and model.train_packets==[8,4]
    assert s['seen']==s['accepted']==8 and s['generated']==s['channel_accepted']==4
    assert s['student_views']==12 and s['teacher_views']==12
    assert model._cd_U_seen==model._cd_U_accepted==8 and torch.isfinite(model.w.grad)
    assert s['EMA_updates']==s['U_truth_read']==s['new_inference_parameters']==0


def test_no_raw_acceptance_executes_no_channel_generator_or_student():
    model,obj,calls=setup(scale=0.);value=obj.loss(model,'cpu')
    assert value.item()==0 and calls==[] and model.train_packets==[8]
    assert obj.state_dict()['generated']==0


def test_rejected_channel_views_fall_back_exactly_to_original_raw_loss():
    model,obj,calls=setup(aug_scale=.5);original=F.cross_entropy(model(torch.ones(8,2,256)),torch.zeros(8,dtype=torch.long))
    value=obj.loss(model,'cpu')
    assert torch.equal(value,original) and calls==[4] and model.train_packets==[8]
    assert obj.state_dict()['channel_accepted']==0 and model.training


def test_U_truth_cannot_enter_the_objective():
    model,obj,calls=setup(truth=True)
    with pytest.raises(ValueError):obj.loss(model,'cpu')
    assert calls==[] and model.train_packets==[]


def test_scope_restores_native_plain_PL_hook_and_loader_on_error():
    from experiments.cric_dura_comparison import t1_model,f_train,f_data
    original=(t1_model.plain_pl,t1_model.hook,t1_model.install,f_train.arguments,f_data.context)
    c=dict(channel_pl=deepcopy(RECIPE),repl_row={'L_B':True,'group':'main','structure':'full'},U_use=True)
    with pytest.raises(RuntimeError):
        with installed(c,{}):raise RuntimeError('synthetic failure')
    assert original==(t1_model.plain_pl,t1_model.hook,t1_model.install,f_train.arguments,f_data.context)


def test_bound_native_hook_accepts_the_native_explicit_model_keyword(monkeypatch):
    from types import MethodType,SimpleNamespace
    from experiments.cric_dura_comparison import t1_model
    def native(self,*,model):
        assert self is model
        return torch.tensor(1.),{'T1_L_packets':768}
    native._accepts_labeled_inputs=True
    monkeypatch.setattr(t1_model,'hook',native)
    c=dict(channel_pl=deepcopy(RECIPE),repl_row={'L_B':True,'group':'main','structure':'full'},U_use=True)
    owner=object();holder={'objective':SimpleNamespace(metrics={'CPL_seen':256})}
    with installed(c,holder):
        bound=MethodType(t1_model.hook,owner)
        value,metrics=bound(model=owner)
        assert value==1. and metrics=={'T1_L_packets':768,'CPL_seen':256}
        assert bound._accepts_labeled_inputs
