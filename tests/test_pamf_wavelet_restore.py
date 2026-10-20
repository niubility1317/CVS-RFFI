import copy
import importlib
import pytest
import torch
from torch import nn

torch.set_num_threads(2)


def m():return importlib.import_module('experiments.pamf_wavelet_restore.models')


def inputs(n=2):return torch.randn((n,9,256),generator=torch.Generator().manual_seed(29001))*.2


def test_haar_roundtrip_energy_and_known_alternating_signal():
    x=inputs();bands=m().haar_split(x);assert bands.shape==(2,18,128)
    torch.testing.assert_close(m().haar_merge(bands),x,atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(bands.square().sum(),x.square().sum())
    constant=torch.ones(1,1,8);alternating=constant.clone();alternating[...,1::2]=-1
    torch.testing.assert_close(m().haar_split(constant)[:,:1],torch.full((1,1,4),2**.5))
    assert torch.equal(m().haar_split(constant)[:,1:],torch.zeros(1,1,4))
    assert torch.equal(m().haar_split(alternating)[:,:1],torch.zeros(1,1,4))
    torch.testing.assert_close(m().haar_split(alternating)[:,1:],torch.full((1,1,4),2**.5))


def test_retention_ablation_changes_only_high_bands():
    x=inputs();allbands=m().analysis_bands(x,True);low=m().analysis_bands(x,False)
    assert torch.equal(low[:,:9],allbands[:,:9]) and torch.count_nonzero(low[:,9:])==0
    torch.testing.assert_close(m().haar_merge(low)[...,0::2],(x[...,0::2]+x[...,1::2])/2)
    assert torch.equal(m().haar_merge(low)[...,0::2],m().haar_merge(low)[...,1::2])


@pytest.mark.parametrize('candidate',['lowpass_control','haar_allbands'])
def test_haar_packet_phase_mixing_commutes(candidate):
    x=inputs()[:,:2];phase=.7;c,s=torch.cos(torch.tensor(phase)),torch.sin(torch.tensor(phase))
    rotate=lambda z:torch.stack((c*z[:,0]-s*z[:,1],s*z[:,0]+c*z[:,1]),1)
    bands=m().analysis_bands(x,candidate=='haar_allbands')
    expected=torch.cat((rotate(bands[:,:2]),rotate(bands[:,2:])),1)
    torch.testing.assert_close(m().analysis_bands(rotate(x),candidate=='haar_allbands'),expected)


def opened(candidate):
    torch.manual_seed(29);net=m().WaveletCore(keep_high=candidate=='haar_allbands')
    with torch.no_grad():net.out.weight.normal_(0,.003)
    return net


def test_matched_core_state_capacity_and_allband_path_gradients():
    torch.manual_seed(29);a=m().WaveletCore(keep_high=True)
    torch.manual_seed(29);b=m().WaveletCore(keep_high=False)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    assert torch.equal(a(inputs()),b(inputs())) and torch.count_nonzero(a(inputs()))==0
    a=opened('haar_allbands');x=inputs().requires_grad_();y=a(x);y.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.norm()>0
    for layer in (*a.encode,*a.expand,*a.decode):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in layer.parameters())


@pytest.mark.parametrize('candidate',['lowpass_control','haar_allbands'])
def test_packet_independence_and_no_persistent_state_update(candidate):
    net=opened(candidate).eval();x=inputs(3);before=copy.deepcopy(net.state_dict())
    with torch.no_grad():
        result=net(x);single=torch.cat([net(v[None]) for v in x]);permuted=net(x[[2,0,1]])
    torch.testing.assert_close(result,single,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(result[[2,0,1]],permuted,atol=1e-6,rtol=1e-5)
    assert all(torch.equal(v,net.state_dict()[k]) for k,v in before.items())
    assert not any(isinstance(layer,(nn.Dropout,nn.BatchNorm1d)) for layer in net.modules())


@pytest.mark.parametrize('bad',[torch.ones(2,9,128),torch.ones(9,256),torch.ones(2,9,256,dtype=torch.int64),torch.full((2,9,256),float('nan'))])
def test_bad_prior_input_rejected(bad):
    with pytest.raises(ValueError):m().WaveletCore()(bad)


@pytest.mark.parametrize('bad',[torch.ones(1,2,7),torch.ones(2,8),torch.ones(1,2,8,dtype=torch.int64)])
def test_bad_analysis_input_rejected(bad):
    with pytest.raises(ValueError):m().haar_split(bad)


class Context(nn.Module):
    def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(.9))
    def estimate(self,x):
        from experiments.pamf_action_restore.operators import Action
        main=torch.full((len(x),256),.9+.1j,dtype=torch.complex64,device=x.device)
        return self.scale*x,Action(torch.ones((len(x),1),dtype=main.dtype,device=x.device),main,.02*main),torch.full((len(x),),.01,device=x.device)


def test_three_stage_matched_full_states_outputs_and_frozen_parent():
    from experiments.pamf_unrolled_restore.models import UnrolledRestorer
    x=inputs()[:,:2];old=UnrolledRestorer('unroll_unshared',Context()).eval()
    with torch.no_grad():original=old(x)
    first=None
    for name in ('lowpass_control','haar_allbands'):
        torch.manual_seed(29);net=m().WaveletRestorer(name,Context()).train()
        state=copy.deepcopy(net.state_dict())
        if first is None:first=state
        else:assert all(torch.equal(v,first[k]) for k,v in state.items())
        stages,*_=net.stage_outputs(x);assert len(stages)==3
        assert torch.equal(stages[-1],original)
        (stages[-1]-.95*x-.03).square().mean().backward()
        assert all(core.out.weight.grad is not None and torch.isfinite(core.out.weight.grad).all() and core.out.weight.grad.norm()>0 for core in net.prior)
        assert not net.frozen.training and not any(p.requires_grad or p.grad is not None for p in net.frozen.parameters())
        assert len({p.data_ptr() for core in net.prior for p in core.parameters()})==sum(len(list(core.parameters())) for core in net.prior)


def test_unknown_candidate_rejected():
    with pytest.raises(ValueError):m().WaveletRestorer('unknown',Context())


def implementation():return m()
def pipeline(name):return importlib.import_module('experiments.pamf_wavelet_restore.'+name)

def full_payload(candidate='haar_allbands'):
    d=pipeline('design')
    from experiments.pamf_uncertainty_restore import design as p
    return dict(schema='wavelet_prior_checkpoint_v1',config=d.configuration(),kind=d.KIND,candidate=candidate,
        prior_initialization='scratch',inherited_prior_epochs=0,training_TX_labels_read=0,training_F_access=False,target_training=False,
        epoch=20,updates=4940,parent=dict(coarse_run=p.COARSE_RUN,coarse_commit=p.COARSE_COMMIT,action_run=p.ACTION_RUN,
            action_commit=p.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False))


@pytest.mark.parametrize('mutation',['schema','TX','F','target','parent','budget','unregistered_control'])
def test_checkpoint_rejects_old_or_unauthorized_states(mutation):
    r=pipeline('runtime');value=full_payload();r.validate_payload(value)
    if mutation=='schema':value['schema']='joint_coarse_checkpoint_v1'
    elif mutation=='TX':value['training_TX_labels_read']=1
    elif mutation=='F':value['training_F_access']=True
    elif mutation=='target':value['target_training']=True
    elif mutation=='parent':value['parent']['action_commit']='unknown'
    elif mutation=='budget':value['updates']=4941
    else:value['candidate']='unet_control'
    with pytest.raises(ValueError):r.validate_payload(value)


@pytest.mark.parametrize('candidate',['lowpass_control','haar_allbands'])
def test_real_model_checkpoint_roundtrip(tmp_path,candidate):
    from experiments.pamf_action_restore.models import ActionRestorer
    from experiments.pamf_reception_restore.models import ComplexTFUNet
    model=implementation().WaveletRestorer(candidate,ActionRestorer(ComplexTFUNet())).eval()
    r=pipeline('runtime');path=tmp_path/'check.pt';r.save_checkpoint(path,model,None,0,0,full_payload()['parent'])
    fresh,value=r.load_checkpoint(path,'cpu');assert value['candidate']==candidate and value['epoch']==value['updates']==0
    with torch.no_grad():assert torch.equal(model(inputs()[:,:2]),fresh(inputs()[:,:2]))
    assert not any(p.requires_grad for p in fresh.frozen.parameters())


@pytest.mark.parametrize('qualified,row,targets',[(False,'haar_allbands',[]),(True,'lowpass_control',['haar_allbands']),(True,'haar_allbands',['haar_allbands','lowpass_control'])])
def test_unqualified_or_unselected_target_rejected_before_target_io(tmp_path,qualified,row,targets):
    import json
    from types import SimpleNamespace
    (tmp_path/'source_decision.json').write_text(json.dumps(dict(status='SOURCE_DECISION_FROZEN',qualified=qualified,selected='haar_allbands' if qualified else None,target_rows=targets)))
    with pytest.raises(ValueError):pipeline('evaluation').prepare_target(SimpleNamespace(output=tmp_path,row=row))
    assert not (tmp_path/row).exists()


def write_control(tmp_path):
    import json
    d=pipeline('design');root=tmp_path/d.CONTROL_RUN;row=root/d.CONTROL_CANDIDATE;(row/d.KIND).mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value),encoding='utf-8')
    write(root/'launch.json',dict(commit=d.CONTROL_COMMIT))
    write(root/'complete.json',dict(status='ANALYZED',source_only=True,target_rows=[],training_TX_F_access=False))
    write(row/'complete.json',dict(status='ANALYZED',role='source',all_predictions_before_truth=True,training_TX_F_access=False))
    train=dict(status='SOURCE_TRAINED',epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,prior_updates=4940,coarse_updates=0,action_estimator_updates=0,
        TX_labels_read=0,F_outputs_read=0,F_weights_read=0,target_reads=0,prior_initialization='scratch',inherited_prior_epochs=0)
    write(row/'training_complete.json',train)
    write(row/'resolved_config.json',dict(schema='joint_coarse_v1',candidate='prior_only',model_seed=2026092701,data_seed=2026102001,training_F_access=False,target_training=False,
        parameter_groups=dict(prior=2507526),steps=3,CG_total=6))
    summary=dict(status='ANALYZED',role='source',count=1800,F_seed=2026092701,original_default_head=True,training_TX_labels_read=0,training_F_access=False,target_access=False,
        all_predictions_before_truth=True,head_fit_updates=0)
    write(row/d.KIND/'summary.json',summary);(row/d.KIND/'final.pt').write_bytes(b'Existence only;real loader checked by G0')
    return root,write


@pytest.mark.parametrize('mutation',['commit','coarse','F','target_root','role','budget'])
def test_reused_control_requires_matching_source_provenance(tmp_path,mutation):
    import json
    e=pipeline('evaluation');d=pipeline('design');root,write=write_control(tmp_path);e.validate_control(root)
    row=root/d.CONTROL_CANDIDATE
    path=root/'launch.json' if mutation=='commit' else root/'complete.json' if mutation=='target_root' else row/d.KIND/'summary.json' if mutation=='role' else row/'training_complete.json'
    value=json.loads(path.read_text())
    if mutation=='commit':value['commit']='unknown'
    elif mutation=='coarse':value['coarse_updates']=1
    elif mutation=='F':value['F_weights_read']=1
    elif mutation=='target_root':value['source_only']=False;value['target_rows']=['joint_coarse']
    elif mutation=='role':value['role']='target'
    else:value['updates']=4939
    write(path,value)
    with pytest.raises(ValueError):e.validate_control(root)


def test_training_uses_only_IQ_loss_with_detached_clean():
    clean=inputs()[:,:2].requires_grad_();estimate=(clean.detach()+.03).requires_grad_()
    value,metrics=pipeline('training').iq_loss(estimate,clean);assert set(metrics)=={'iq_nmse'}
    value.mean().backward();assert clean.grad is None and torch.isfinite(estimate.grad).all()


def test_fullmatrix_has_two_new_rows_and_reused_fixed_control():
    d=pipeline('design');cfg=d.configuration()
    assert d.ROWS==('lowpass_control','haar_allbands') and cfg['training_F_access'] is False and cfg['target_training'] is False
    assert cfg['model_seed']==2026092701 and cfg['data_seed']==2026102001 and cfg['epochs']==20 and cfg['updates']==4940
    assert cfg['CG_total']==6 and cfg['calibration_count']==0
    assert d.CONTROL_CANDIDATE=='prior_only' and d.CONTROL_COMMIT=='d43dbab326d0b1e2cb6fb7faa23dc9a3ec25a025'


def source_summary(low=82.,mid=90.,nmse=.03):
    e=pipeline('evaluation');d=pipeline('design')
    value=dict(status='ANALYZED',kind=d.KIND,role='source',count=1800,F_run='20261016-phase1-F-supcon-e50-clean-manysig-m9-r02',
        F_seed=d.MODEL_SEED,original_default_head=True,all_predictions_before_truth=True,training_TX_labels_read=0,
        training_F_access=False,head_fit_updates=0,target_access=False)
    value['views']={v:dict(restored=dict(waveform=dict(iq_nmse=.001 if v=='clean' else nmse))) for v in e.VIEWS};records=[]
    for view in e.VIEWS:
        acc=98.5 if view=='clean' else low if view=='practical_low_urban' else mid if view=='practical_mid_urban' else 97.
        for dimension,groups in [('overall',['all']),('TX',range(6)),('RX',(1,3,4,6,8))]:
            records.extend(dict(view=view,dimension=dimension,value=str(i),readout='restored',accuracy=acc,baseline_accuracy=98.5) for i in groups)
    value['records']=records;return value


def test_six_source_references_and_matched_control_cannot_be_skipped():
    e=pipeline('evaluation');d=pipeline('design');refs=[source_summary() for _ in range(6)]
    candidates={'lowpass_control':source_summary(84.,92.,.028),'haar_allbands':source_summary(85.,93.,.028)}
    result=e.select_source(candidates,*refs)
    assert result['selected']=='haar_allbands' and result['target_rows']==['haar_allbands']
    candidates['lowpass_control']=source_summary(90.,95.,.028)
    result=e.select_source(candidates,*refs)
    assert not result['qualified'] and not result['target_rows']
    assert not result['checks']['haar_allbands']['vs_matched_lowpass_control']['qualified']
    candidates['lowpass_control']=source_summary(84.,92.,.028);refs[-1]=source_summary(90.,95.)
    assert not e.select_source(candidates,*refs)['qualified']
    with pytest.raises(ValueError):e.select_source({'haar_allbands':candidates['haar_allbands']},*refs)
    candidates['lowpass_control']['role']='target'
    with pytest.raises(ValueError):e.select_source(candidates,*refs)


def test_original_R23_owned_process_blocks_but_other_owner_or_exited_does_not(tmp_path):
    from types import SimpleNamespace
    c=pipeline('controller');root,write=write_control(tmp_path)
    write(root/'launch.json',dict(commit=pipeline('design').CONTROL_COMMIT,pid=123));write(root/'workers.json',dict(rows=[]))
    proc=tmp_path/'proc';process=proc/'123';process.mkdir(parents=True)
    (process/'status').write_text('State:\tS (sleeping)\n')
    (process/'cmdline').write_bytes(('python\0-m\0experiments.pamf_joint_coarse_restore.runner\0--output\0'+str(root)).encode())
    with pytest.raises(ValueError):c.require_preceding_finished(SimpleNamespace(preceding_run=root),proc_root=proc)
    (process/'cmdline').write_bytes(b'python\0other-owner-job');c.require_preceding_finished(SimpleNamespace(preceding_run=root),proc_root=proc)
    (process/'status').write_text('State:\tZ (zombie)\n');c.require_preceding_finished(SimpleNamespace(preceding_run=root),proc_root=proc)


def test_new_prediction_loader_emits_all_seven_before_truth(tmp_path,monkeypatch):
    import json
    import numpy as np
    from types import SimpleNamespace
    import experiments.pamf_uncertainty_restore.evaluation as shared
    e=pipeline('evaluation');d=pipeline('design');source=tmp_path/'input';source.mkdir()
    for view in e.VIEWS:np.save(source/(view+'.npy'),np.ones((16,2,256),np.float32))
    output=tmp_path/'output';(output/d.ROWS[0]/d.KIND).mkdir(parents=True)
    args=SimpleNamespace(row=d.ROWS[0],output=output,reader_folder=tmp_path/'unused',source_cache=tmp_path/'unused')
    class Toy(nn.Module):
        def forward(self,x):return x
        def training_outputs(self,x):return x,torch.ones((len(x),256)),None,x
    class Reader(nn.Module):
        def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.ones(1))
        def features(self,x):assert self.anchor._version>=0;return x.flatten(1)[:,:160]
        def classify_features(self,x):return x[:,:6]
    calls=[]
    def loader(path,device):calls.append(str(path));return Toy(),dict(candidate=d.ROWS[0],kind=d.KIND,epoch=20,updates=4940)
    monkeypatch.setattr(e,'load_checkpoint',loader)
    monkeypatch.setattr(shared,'reference',lambda *a:(source,np.arange(16),None,None));monkeypatch.setattr(shared,'load_reader',lambda *a:Reader())
    monkeypatch.setattr(shared,'array_tensor',lambda x,device:torch.tensor(x.tolist(),dtype=torch.float32));monkeypatch.setattr(shared,'tensor_array',lambda x:np.asarray(x.tolist(),np.float32))
    for key in ('synchronize','reset_peak_memory_stats'):monkeypatch.setattr(torch.cuda,key,lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    e.predict(args,'source');folder=output/d.ROWS[0]/d.KIND
    assert tuple(shared.prediction_arrays(folder/'prediction',np.arange(16),d.KIND,'source'))==e.VIEWS and len(calls)==1
    meta=json.loads((folder/'prediction/complete.json').read_text())
    assert meta['status']=='PREDICTIONS_COMPLETE' and meta['TX_truth_reads']==meta['state_updates']==meta['head_fit_updates']==0
