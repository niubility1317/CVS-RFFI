import copy,json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_iqonly_restore import design as d,runtime,evaluation as e,training as t
from experiments.pamf_fullresolution_restore import runtime as paired
from experiments.pamf_fullresolution_restore.models import ResolutionRestorer
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_uncertainty_restore import design as ancestors

torch.set_num_threads(2)

def parent():
    return dict(coarse_run=ancestors.COARSE_RUN,coarse_commit=ancestors.COARSE_COMMIT,action_run=ancestors.ACTION_RUN,
        action_commit=ancestors.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)

def test_IQ_objective_and_gradient_match_complex_MSE_without_FFT(monkeypatch):
    monkeypatch.setattr(torch.fft,'fft',lambda *a,**k:pytest.fail('IQ-only training must not compute FFT'))
    x=torch.randn(3,2,256);y=torch.randn_like(x).requires_grad_()
    loss,metrics=t.iq_loss(y,x);manual=(y-x).square().mean((1,2))/x.square().mean((1,2)).clamp_min(1e-4)
    torch.testing.assert_close(loss,manual);assert set(metrics)=={'iq_nmse'}
    loss.mean().backward();expected=2*(y.detach()-x)/(3*512*x.square().mean((1,2)).clamp_min(1e-4)[:,None,None])
    torch.testing.assert_close(y.grad,expected)

def test_training_seam_passes_the_actual_loss_runtime_and_configuration(monkeypatch):
    capture={}
    monkeypatch.setattr(t,'shared_train',lambda args,**kwargs:capture.update(kwargs))
    t.train(SimpleNamespace());assert capture['loss_fn'] is t.iq_loss and capture['model_runtime'] is runtime
    assert capture['configuration']()==d.configuration()

def test_new_checkpoint_strictly_distinguishes_IQ_from_FFT_and_permissions(tmp_path):
    model=ResolutionRestorer(d.PRIMARY,ActionRestorer(ComplexTFUNet()));path=tmp_path/'new.pt'
    runtime.save_checkpoint(path,model,None,20,4940,parent());fresh,value=runtime.load_checkpoint(path,'cpu')
    assert all(torch.equal(v,fresh.state_dict()[k]) for k,v in model.state_dict().items())
    with pytest.raises(ValueError):paired.validate_payload(value)
    for key,bad in [('training_F_access',True),('target_training',True),('inherited_prior_epochs',20),('prior_initialization','R15trained')]:
        invalid=copy.deepcopy(value);invalid[key]=bad
        with pytest.raises(ValueError):runtime.validate_payload(invalid)
    invalid=copy.deepcopy(value);invalid['parent']['action_commit']='unknown'
    with pytest.raises(ValueError):runtime.validate_payload(invalid)
    paired.save_checkpoint(path,model,None,0,0,parent());old=torch.load(path,weights_only=False)
    with pytest.raises(ValueError):runtime.validate_payload(old)

def source_summary(low=80.,mid=89.,nmse=.03):
    value=dict(status='ANALYZED',kind=d.KIND,role='source',count=1800,F_run='20261016-phase1-F-supcon-e50-clean-manysig-m9-r02',
        F_seed=d.MODEL_SEED,original_default_head=True,all_predictions_before_truth=True,training_TX_labels_read=0,
        training_F_access=False,head_fit_updates=0,target_access=False)
    value['views']={v:dict(restored=dict(waveform=dict(iq_nmse=.001 if v=='clean' else nmse))) for v in e.VIEWS}
    records=[]
    for view in e.VIEWS:
        accuracy=98.5 if view=='clean' else low if view=='practical_low_urban' else mid if view=='practical_mid_urban' else 97.
        records.append(dict(view=view,dimension='overall',value='all',readout='restored',accuracy=accuracy,baseline_accuracy=98.5))
        records.extend(dict(view=view,dimension='TX',value=str(i),readout='restored',accuracy=accuracy,baseline_accuracy=98.5) for i in range(6))
    value['records']=records;return value

def test_fixed_IQ_primary_and_paired_fft_formal_control_have_both_guards():
    control=source_summary();reference=source_summary(low=80.2,mid=89.2);candidate=source_summary(low=81.,mid=90.,nmse=.028)
    result=e.qualify_source(control,candidate,reference)
    assert result['qualified'] and result['primary']==d.PRIMARY and result['target_rows']==list(d.TARGET_ROWS)
    assert 'tcn_prior' not in result['target_rows'] and 'mean_TCN' not in result and result['mean_IQ']==pytest.approx(.028)
    next(r for r in control['records'] if r['view']=='practical_low_urban' and r['dimension']=='TX')['accuracy']=84.
    assert not e.qualify_source(control,candidate,reference)['qualified']
    invalid=copy.deepcopy(candidate);invalid['target_access']=True
    with pytest.raises(ValueError):e.qualify_source(control,invalid,reference)

def test_pair_metadata_rejects_unverified_or_target_exposed_run(tmp_path):
    root=tmp_path/d.PAIRED;root.mkdir();(root/'launch.json').write_text(json.dumps(dict(commit=d.PAIRED_COMMIT)))
    good=dict(status='ANALYZED',source_only=True,target_rows=[],training_TX_F_access=False)
    (root/'complete.json').write_text(json.dumps(good));e.validate_pair(root)
    for key,value in [('source_only',False),('target_rows',['unet_control']),('training_TX_F_access',True)]:
        (root/'complete.json').write_text(json.dumps(dict(good,**{key:value})))
        with pytest.raises(ValueError):e.validate_pair(root)

def decision():return dict(status='SOURCE_DECISION_FROZEN',qualified=True,primary=d.PRIMARY,target_rows=list(d.TARGET_ROWS))

def test_secondary_and_unqualified_target_are_rejected_before_IO(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'original_predict',lambda *a,**k:pytest.fail('Forbidden target IO'))
    for row,value in [('tcn_prior',decision()),(d.PRIMARY,dict(decision(),qualified=False)),(d.PRIMARY,dict(decision(),primary='tcn_prior'))]:
        (tmp_path/'source_decision.json').write_text(json.dumps(value))
        with pytest.raises(ValueError):e.predict(SimpleNamespace(output=tmp_path,row=row),'target')
    with pytest.raises(ValueError):e.predict(SimpleNamespace(row=d.FFT_CONTROL),'source')

@pytest.mark.parametrize('row,role',[(d.PRIMARY,'source'),(d.FFT_CONTROL,'target')])
def test_both_checkpoint_routes_reach_all_seven_predictions(tmp_path,monkeypatch,row,role):
    import experiments.pamf_uncertainty_restore.evaluation as shared
    source=tmp_path/'input';source.mkdir()
    for view in e.VIEWS:np.save(source/(view+'.npy'),np.ones((16,2,256),np.float32))
    output=tmp_path/'output';(output/row/d.KIND).mkdir(parents=True)
    (output/'source_decision.json').write_text(json.dumps(decision()))
    args=SimpleNamespace(row=row,output=output,reader_folder=tmp_path/'unused',source_cache=tmp_path/'unused',paired_run=tmp_path/'pair')
    class Toy(nn.Module):
        def forward(self,x):return x
        def training_outputs(self,x):return x,torch.ones((len(x),256)),None,x
    class Reader(nn.Module):
        def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.ones(1))
        def features(self,x):assert self.anchor._version>=0;return x.flatten(1)[:,:160]
        def classify_features(self,x):return x[:,:6]
    calls=[]
    def loader(path,device):calls.append(str(path));return Toy(),dict(candidate=d.PRIMARY,kind=d.KIND,epoch=20,updates=4940)
    monkeypatch.setattr(e,'load_checkpoint',loader);monkeypatch.setattr(e,'load_fft',loader);monkeypatch.setattr(e,'validate_pair',lambda path:None)
    monkeypatch.setattr(shared,'load_checkpoint',lambda *a:pytest.fail('Default oldschema loader forbidden'))
    monkeypatch.setattr(shared,'reference',lambda *a:(source,np.arange(16),None,None));monkeypatch.setattr(shared,'load_reader',lambda *a:Reader())
    monkeypatch.setattr(shared,'array_tensor',lambda x,device:torch.tensor(x.tolist(),dtype=torch.float32))
    monkeypatch.setattr(shared,'tensor_array',lambda x:np.asarray(x.tolist(),np.float32))
    for key in ('synchronize','reset_peak_memory_stats'):monkeypatch.setattr(torch.cuda,key,lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    e.predict(args,role);folder=output/row
    if role=='target':folder=folder/'target_confirmation'
    values=shared.prediction_arrays(folder/d.KIND/'prediction',np.arange(16),d.KIND,role);assert tuple(values)==e.VIEWS
    assert len(calls)==1 and (str(args.paired_run) in calls[0])==(row==d.FFT_CONTROL)

def test_fixed_budget_pairing_and_no_spectral_training_metric():
    cfg=d.configuration();old=paired.d.configuration()
    for key in ('epochs','updates','batch_size','L_count','U_count','L_exposures','U_exposures','peak_lr','final_lr','warmup_updates','precision','data_seed','model_seed'):
        assert cfg[key]==old[key]
    assert cfg['calibration_count']==0 and cfg['primary']==d.PRIMARY and not cfg['training_F_access']
