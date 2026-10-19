import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_fullresolution_restore import design as d,runtime,evaluation as e
from experiments.pamf_fullresolution_restore.models import FullResolutionTCN,ResolutionRestorer
from experiments.pamf_fullresolution_restore.training import learning_rate
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import Core,ComplexTFUNet
from experiments.pamf_uncertainty_restore import design as old

torch.set_num_threads(2)


class Frozen(nn.Module):
    def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.tensor(1.))
    def estimate(self,x):
        main=torch.ones((len(x),256),dtype=torch.complex64,device=x.device)
        return .9*x,Action(torch.ones((len(x),1),dtype=torch.complex64,device=x.device),main,torch.zeros_like(main)),torch.full((len(x),),.01,device=x.device)


def parent():
    return dict(coarse_run=old.COARSE_RUN,coarse_commit=old.COARSE_COMMIT,action_run=old.ACTION_RUN,action_commit=old.ACTION_COMMIT,
        data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)


def test_fullresolution_layers_preserve_all_samples_and_have_close_cost():
    model=FullResolutionTCN();shapes=[]
    hooks=[module.register_forward_hook(lambda m,i,o:shapes.append(o.shape[-1])) for module in model.modules() if isinstance(module,nn.Conv1d)]
    value=model(torch.randn(2,4,256))
    for hook in hooks:hook.remove()
    assert value.shape==(2,2,256) and not value.any() and shapes and set(shapes)=={256}
    assert [block.net[0].dilation[0] for block in model.blocks]==d.configuration()['tcn']['dilations']
    ratio=sum(p.numel() for p in model.parameters())/sum(p.numel() for p in Core(incoming=4).parameters())
    assert .9<ratio<1.1 and not any(isinstance(m,(nn.AvgPool1d,nn.MaxPool1d,nn.Upsample,nn.BatchNorm1d)) for m in model.modules())


def test_TCN_is_per_packet_even_after_nonzero_output_head():
    torch.manual_seed(17);model=FullResolutionTCN().eval();nn.init.normal_(model.out.weight,std=.01)
    x=torch.randn(3,4,256)
    torch.testing.assert_close(model(x),torch.cat([model(row[None]) for row in x]),atol=2e-6,rtol=2e-5)


def test_same_initial_full_output_and_only_prior_gets_gradients():
    r=torch.randn(2,2,256)*.3;outputs=[]
    for candidate in d.ROWS:
        model=ResolutionRestorer(candidate,Frozen()).train();assert not model.frozen.training and all(not p.requires_grad for p in model.frozen.parameters())
        output=model(r);outputs.append(output.detach());output.square().mean().backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.prior.parameters())
        assert all(p.grad is None for p in model.frozen.parameters())
    assert torch.equal(*outputs)


def test_prior_seed_reset_is_after_parent_rng_consumption(monkeypatch):
    def load(args,kind,device):torch.randn(load.count);return Frozen(),parent()
    load.count=2;monkeypatch.setattr(runtime,'frozen_context',load)
    first,_=runtime.initialize('unet_control',None,'cpu');load.count=111
    second,_=runtime.initialize('unet_control',None,'cpu')
    assert all(torch.equal(v,second.prior.state_dict()[k]) for k,v in first.prior.state_dict().items())


def test_full_state_roundtrip_and_source_only_scratch_lineage(tmp_path):
    model=ResolutionRestorer('tcn_prior',ActionRestorer(ComplexTFUNet()));path=tmp_path/'final.pt'
    runtime.save_checkpoint(path,model,None,20,4940,parent());fresh,payload=runtime.load_checkpoint(path,'cpu')
    assert type(fresh.prior) is FullResolutionTCN and all(torch.equal(v,fresh.state_dict()[k]) for k,v in model.state_dict().items())
    for key,bad in [('training_F_access',True),('target_training',True),('inherited_prior_epochs',20),('prior_initialization','R10native'),('epoch',0)]:
        value=copy.deepcopy(payload);value[key]=bad
        with pytest.raises(ValueError):runtime.validate_payload(value)
    value=copy.deepcopy(payload);value['parent']['data_contract']='other'
    with pytest.raises(ValueError):runtime.validate_payload(value)
    value=copy.deepcopy(payload);value['parent']['action_commit']='unknown'
    with pytest.raises(ValueError):runtime.validate_payload(value)


def summary(low=80.,mid=89.,nmse=.03):
    value=dict(status='ANALYZED',kind=d.KIND,role='source',count=1800,F_run=e.REFERENCE_RUN,F_seed=d.MODEL_SEED,original_default_head=True,
        all_predictions_before_truth=True,training_TX_labels_read=0,training_F_access=False,head_fit_updates=0,target_access=False)
    value['views']={view:dict(restored=dict(waveform=dict(iq_nmse=.001 if view=='clean' else nmse))) for view in e.VIEWS}
    records=[]
    for view in e.VIEWS:
        accuracy=98.5 if view=='clean' else low if view=='practical_low_urban' else mid if view=='practical_mid_urban' else 97.
        records.append(dict(view=view,dimension='overall',value='all',readout='restored',accuracy=accuracy,baseline_accuracy=98.5))
        records += [dict(view=view,dimension='TX',value=str(i),readout='restored',accuracy=accuracy,baseline_accuracy=98.5) for i in range(6)]
    value['records']=records;return value


def test_fixed_source_gate_requires_both_identity_and_waveform():
    control=summary();reference=summary(low=80.2,mid=89.2,nmse=.029);candidate=summary(low=81.,mid=90.,nmse=.028)
    result=e.qualify_source(control,candidate,reference);assert result['qualified'] and result['target_rows']==list(d.ROWS)
    bad=summary(low=81.,mid=89.3,nmse=.028);assert not e.qualify_source(control,bad,reference)['qualified']
    bad=summary(low=81.,mid=90.,nmse=.04);assert not e.qualify_source(control,bad,reference)['qualified']
    bad=copy.deepcopy(candidate);bad['role']='target'
    with pytest.raises(ValueError):e.qualify_source(control,bad,reference)


def test_TX_guard_against_control_is_not_replaced_by_historical_reference():
    control=summary();reference=summary();candidate=summary(low=81.,mid=90.,nmse=.028)
    row=next(r for r in control['records'] if r['view']=='practical_low_urban' and r['dimension']=='TX');row['accuracy']=84.
    result=e.qualify_source(control,candidate,reference)
    assert not result['qualified'] and result['worst_lowurban_TX_vs_control_pp']==-3. and result['worst_lowurban_TX_vs_reference_pp']==1.


def test_clean_damage_and_other_scene_regression_block_promotion():
    control=summary();reference=summary();candidate=summary(low=81.,mid=90.,nmse=.028)
    for selected,accuracy in [('clean',98.),('practical_high',96.)]:
        value=copy.deepcopy(candidate)
        row=next(r for r in value['records'] if r['view']==selected and r['dimension']=='overall');row['accuracy']=accuracy
        assert not e.qualify_source(control,value,reference)['qualified']


def test_unqualified_or_unregistered_target_stops_before_input_IO(tmp_path,monkeypatch):
    args=SimpleNamespace(output=tmp_path,row='tcn_prior');path=tmp_path/'source_decision.json'
    monkeypatch.setattr(e,'original_predict',lambda *a,**k:pytest.fail('Forbidden target input IO'))
    for value in (dict(status='SOURCE_DECISION_FROZEN',qualified=False,target_rows=[]),dict(status='SOURCE_DECISION_FROZEN',qualified=True,target_rows=['unet_control'])):
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError):e.predict(args,'target')


def test_custom_checkpoint_loader_reaches_full_seven_view_prediction(tmp_path,monkeypatch):
    import experiments.pamf_uncertainty_restore.evaluation as original
    source=tmp_path/'input';source.mkdir()
    for view in e.VIEWS:np.save(source/(view+'.npy'),np.ones((16,2,256),np.float32))
    output=tmp_path/'output';(output/'tcn_prior'/d.KIND).mkdir(parents=True)
    args=SimpleNamespace(row='tcn_prior',output=output,reader_folder=tmp_path/'unused',source_cache=tmp_path/'unused')
    class Toy(nn.Module):
        def forward(self,x):return x
        def training_outputs(self,x):return x,torch.ones((len(x),256)),None,x
    class Reader(nn.Module):
        def __init__(self):super().__init__();self.coordinate=nn.Parameter(torch.ones(1))
        def features(self,x):assert self.coordinate._version>=0;return x.flatten(1)[:,:160]
        def classify_features(self,x):return x[:,:6]
    monkeypatch.setattr(original,'reference',lambda args,role:(source,np.arange(16),None,None))
    monkeypatch.setattr(original,'load_reader',lambda *a:Reader())
    monkeypatch.setattr(original,'load_checkpoint',lambda *a:pytest.fail('Oldschema loader forbidden'))
    monkeypatch.setattr(e,'load_checkpoint',lambda *a:(Toy(),dict(candidate='tcn_prior',kind=d.KIND,epoch=20,updates=4940)))
    monkeypatch.setattr(original,'array_tensor',lambda value,device:torch.tensor(value.tolist(),dtype=torch.float32))
    monkeypatch.setattr(original,'tensor_array',lambda value:np.asarray(value.tolist(),np.float32))
    for key in ('synchronize','reset_peak_memory_stats'):monkeypatch.setattr(torch.cuda,key,lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    e.predict(args,'source')
    values=original.prediction_arrays(output/'tcn_prior'/d.KIND/'prediction',np.arange(16),d.KIND,'source')
    assert tuple(values)==e.VIEWS


def test_fixed_full_budget_LR_and_no_extra_calibration():
    cfg=d.configuration();assert cfg['epochs']==20 and cfg['updates']==4940 and cfg['calibration_count']==0
    assert cfg['L_exposures']==126000 and cfg['U_exposures']==1134000 and not cfg['training_F_access']
    assert abs(learning_rate(4939)-2e-5)<1e-12
    with pytest.raises(ValueError):learning_rate(4940)
