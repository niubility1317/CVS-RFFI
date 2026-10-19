import copy
import importlib
import importlib.util

import pytest
import torch
from torch import nn

torch.set_num_threads(2)


def implementation():
    assert importlib.util.find_spec('experiments.pamf_attention_restore') is not None,'Attention prototype missing'
    return importlib.import_module('experiments.pamf_attention_restore.models')


def inputs(count=2):
    return .2*torch.randn((count,9,256),generator=torch.Generator().manual_seed(24001))


def opened_prior(checkpoint_blocks=True):
    torch.manual_seed(24);model=implementation().TemporalAttentionPrior(checkpoint_blocks=checkpoint_blocks)
    with torch.no_grad():model.out.weight.normal_(0,.003)
    return model


def test_full_resolution_zero_output_and_fixed_capacity():
    model=implementation().TemporalAttentionPrior();x=inputs()
    assert torch.equal(model(x),torch.zeros(2,2,256))
    assert len(model.blocks)==6 and model.width==128 and model.heads==4
    assert model.position.shape==(1,256,128) and not model.position.requires_grad
    assert sum(p.numel() for p in model.parameters())==901890
    assert not any(isinstance(m,(nn.Dropout,nn.BatchNorm1d)) for m in model.modules())


def test_every_attention_local_and_MLP_block_has_finite_gradient():
    model=opened_prior().train();x=inputs().requires_grad_();model(x).square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.norm()>0
    for block in model.blocks:
        for part in (block.qkv,block.project,block.local,block.mix,block.mlp):
            gradients=[p.grad for p in part.parameters()]
            assert all(g is not None and torch.isfinite(g).all() and g.norm()>0 for g in gradients)


def test_checkpointed_and_plain_outputs_and_parameter_gradients_match():
    a=opened_prior(True).train();b=copy.deepcopy(a);b.checkpoint_blocks=False;x=inputs()
    aa=a(x);bb=b(x);assert torch.equal(aa,bb)
    aa.square().mean().backward();bb.square().mean().backward()
    for pa,pb in zip(a.parameters(),b.parameters()):assert torch.allclose(pa.grad,pb.grad,atol=1e-7,rtol=1e-5)


def test_packet_independence_order_and_state_preservation():
    model=opened_prior().eval();x=inputs(3);before=copy.deepcopy(model.state_dict())
    with torch.no_grad():
        whole=model(x);single=torch.cat([model(v[None]) for v in x]);permuted=model(x[[2,0,1]])
    assert torch.allclose(whole,single,atol=1e-6,rtol=1e-5)
    assert torch.allclose(whole[[2,0,1]],permuted,atol=1e-6,rtol=1e-5)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())


@pytest.mark.parametrize('bad',[torch.ones(2,9,128),torch.ones(9,256),torch.ones(2,9,256,dtype=torch.int64),torch.full((2,9,256),float('nan'))])
def test_invalid_input_rejected(bad):
    with pytest.raises(ValueError):implementation().TemporalAttentionPrior()(bad)


@pytest.mark.parametrize('amplitude',[0.,1e-10,1e3])
def test_silent_small_and_large_inputs_have_finite_gradients(amplitude):
    model=opened_prior().train();x=(inputs()*amplitude).requires_grad_();out=model(x)
    out.square().mean().backward();assert torch.isfinite(out).all() and torch.isfinite(x.grad).all()


class Context(nn.Module):
    def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(.9))
    def estimate(self,x):
        from experiments.pamf_action_restore.operators import Action
        main=torch.full((len(x),256),.9+.1j,dtype=torch.complex64,device=x.device)
        kernel=torch.ones((len(x),1),dtype=main.dtype,device=x.device)
        return self.scale*x,Action(kernel,main,.02*main),torch.full((len(x),),.01,device=x.device)


def test_composed_three_stage_models_start_with_same_output_and_frozen_context():
    from experiments.pamf_unrolled_restore.models import UnrolledRestorer
    m=implementation();x=inputs()[:,:2];initial=None
    for candidate in ('unet_control','tcn_prior','attention_prior'):
        torch.manual_seed(24);model=m.AttentionRestorer(candidate,Context()).train()
        stages,*_=model.stage_outputs(x)
        if initial is None:initial=stages[-1].detach()
        else:assert torch.equal(initial,stages[-1].detach())
        (stages[-1]-.95*x-.03).square().mean().backward()
        assert len(stages)==3 and all(core.out.weight.grad is not None and torch.isfinite(core.out.weight.grad).all() and core.out.weight.grad.norm()>0 for core in model.prior)
        assert not model.frozen.training and not any(p.requires_grad or p.grad is not None for p in model.frozen.parameters())
        assert len({p.data_ptr() for core in model.prior for p in core.parameters()})==sum(len(list(core.parameters())) for core in model.prior)


def test_unknown_restorer_candidate_rejected():
    with pytest.raises(ValueError):implementation().AttentionRestorer('unknown',Context())


def pipeline(name):return importlib.import_module('experiments.pamf_attention_restore.'+name)


def test_full_resolution_TCN_gradients_and_packet_independence():
    model=implementation().TemporalTCNPrior();x=inputs().requires_grad_()
    assert torch.equal(model(x),torch.zeros(2,2,256))
    with torch.no_grad():model.out.weight.normal_(0,.003)
    model(x).square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.norm()>0
    assert all(any(p.grad is not None and p.grad.norm()>0 for p in block.parameters()) for block in model.blocks)
    model.eval()
    with torch.no_grad():assert torch.allclose(model(x),torch.cat([model(v[None]) for v in x]),atol=1e-6,rtol=1e-5)
    with pytest.raises(ValueError):model(torch.ones(2,4,256))


def full_payload(candidate='attention_prior'):
    d=pipeline('design')
    from experiments.pamf_uncertainty_restore import design as p
    return dict(schema='fullresolution_priors_checkpoint_v1',config=d.configuration(),kind=d.KIND,candidate=candidate,
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


@pytest.mark.parametrize('candidate',['tcn_prior','attention_prior'])
def test_real_model_checkpoint_roundtrip(tmp_path,candidate):
    from experiments.pamf_action_restore.models import ActionRestorer
    from experiments.pamf_reception_restore.models import ComplexTFUNet
    model=implementation().AttentionRestorer(candidate,ActionRestorer(ComplexTFUNet())).eval()
    r=pipeline('runtime');path=tmp_path/'check.pt';r.save_checkpoint(path,model,None,0,0,full_payload()['parent'])
    fresh,value=r.load_checkpoint(path,'cpu');assert value['candidate']==candidate and value['epoch']==value['updates']==0
    with torch.no_grad():assert torch.equal(model(inputs()[:,:2]),fresh(inputs()[:,:2]))
    assert not any(p.requires_grad for p in fresh.frozen.parameters())


@pytest.mark.parametrize('qualified,row,targets',[(False,'attention_prior',[]),(True,'tcn_prior',['attention_prior']),(True,'attention_prior',['attention_prior','tcn_prior'])])
def test_unqualified_or_unselected_target_rejected_before_target_io(tmp_path,qualified,row,targets):
    import json
    from types import SimpleNamespace
    (tmp_path/'source_decision.json').write_text(json.dumps(dict(status='SOURCE_DECISION_FROZEN',qualified=qualified,selected='attention_prior' if qualified else None,target_rows=targets)))
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
    assert d.ROWS==('tcn_prior','attention_prior') and cfg['training_F_access'] is False and cfg['target_training'] is False
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


def test_six_source_references_and_ranking_cannot_skip_reused_control():
    e=pipeline('evaluation');d=pipeline('design')
    refs=[source_summary() for _ in range(6)]
    candidates={d.ROWS[0]:source_summary(84.,92.,.028),d.ROWS[1]:source_summary(85.,93.,.028)}
    result=e.select_source(candidates,*refs)
    assert result['selected']==d.ROWS[1] and result['target_rows']==[d.ROWS[1]]
    refs[-1]=source_summary(90.,95.)
    result=e.select_source(candidates,*refs)
    assert not result['qualified'] and not result['target_rows']
    assert all(not check['vs_reused_R23_control']['qualified'] for check in result['checks'].values())
    with pytest.raises(ValueError):e.select_source({d.ROWS[0]:candidates[d.ROWS[0]]},*refs)
    candidates[d.ROWS[0]]['role']='target'
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
