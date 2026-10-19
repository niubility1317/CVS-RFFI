import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_endpoint_restore import design as d,models as m,runtime,evaluation as e,training as t
from experiments.pamf_unrolled_restore import runtime as old_runtime,design as reference_design
from experiments.pamf_action_restore.models import SCALE,ActionRestorer
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_iterative_restore import evaluation as reference_e
from test_pamf_unrolled_restore import Frozen,parent,summary as base_summary
torch.set_num_threads(2)


def sample_meta():
    return dict(role='source',count=128,seeds=list(d.STABILITY_SEEDS),TX_truth_reads=0,
                primary_changed=False,disagreement_percent=0.)


def summary(**kwargs):return dict(base_summary(**kwargs),sampling_stability=sample_meta())


@pytest.mark.parametrize('candidate',(d.ROWS[0],))
def test_exact_training_path_velocity_target_and_matched_random_streams(candidate,monkeypatch):
    x=torch.randn(2,2,256);r=x+.01;model=m.EndpointRestorer(candidate,Frozen())
    generator=torch.Generator().manual_seed(17);reference=torch.Generator().manual_seed(17)
    noise=torch.randn(x.shape,generator=reference);time=torch.rand(len(x),generator=reference)
    endpoint=(x-.9*r)/SCALE;start=d.SIGMA[candidate]*noise;captured={}
    def field(point,condition,actual_time):
        captured.update(point=point,time=actual_time);return endpoint-start
    monkeypatch.setattr(model.prior,'field',field)
    loss,metrics=model.training_loss(x,r,generator)
    torch.testing.assert_close(captured['point'],(1-time[:,None,None])*start+time[:,None,None]*endpoint)
    torch.testing.assert_close(captured['time'],time)
    torch.testing.assert_close(loss,torch.zeros_like(loss),atol=1e-7,rtol=0)
    torch.testing.assert_close(metrics['teacher_path_endpoint_nmse'],torch.zeros_like(loss),atol=1e-7,rtol=0)
    assert torch.equal(generator.get_state(),reference.get_state())


@pytest.mark.parametrize('candidate',d.ROWS)
def test_full4Euler_context_once_softMAP_and_no_clean_path(candidate,monkeypatch):
    model=m.EndpointRestorer(candidate,Frozen());calls=[];conditions=[];solver=[]
    handle=model.prior.conditioner.register_forward_hook(lambda *a:conditions.append(1))
    def field(point,condition,time):calls.append(time.clone());return torch.full_like(point,2.)
    def solve(action,observed,prior,variance,**kwargs):solver.append(kwargs);return prior
    monkeypatch.setattr(model.prior,'field',field);monkeypatch.setattr(m,'weighted_map',solve)
    monkeypatch.setattr(model,'training_loss',lambda *a:pytest.fail('Teacher clean path forbidden in sampling'))
    r=torch.randn(2,2,256);noise=torch.randn_like(r);actual=model(r,noise);handle.remove()
    torch.testing.assert_close(actual,.9*r+SCALE*(d.SIGMA[candidate]*noise+2.))
    assert len(calls)==4 and len(conditions)==1 and len(solver)==1 and solver[0]['steps']==6
    assert [float(time[0]) for time in calls]==[k/4 for k in range(4)]
    torch.testing.assert_close(model(r),model(r,noise))


def test_same_full_initial_weights_and_seed_reset_after_parent(monkeypatch):
    def load(*a):torch.randn(load.draws);return Frozen(),parent()
    monkeypatch.setattr(runtime,'frozen_context',load)
    load.draws=3;first,_=runtime.initialize('path_four',None,'cpu')
    load.draws=99;second,_=runtime.initialize('endpoint_four',None,'cpu')
    assert all(torch.equal(v,second.state_dict()[k]) for k,v in first.state_dict().items())
    assert set(map(id,first.prior.parameters()))=={id(p) for p in first.parameters() if p.requires_grad}


@pytest.mark.parametrize('candidate',d.ROWS)
def test_real_velocity_gradient_frozen_context_and_fixednoise_packet_independence(candidate):
    torch.manual_seed(7);model=m.EndpointRestorer(candidate,Frozen()).train()
    x=torch.randn(2,2,256)*.3;r=x+.01
    loss,_=model.training_loss(x,r,torch.Generator().manual_seed(10));loss.mean().backward()
    gradients=[p.grad for p in model.prior.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients) and sum(float(g.abs().sum()) for g in gradients)>0
    assert all(p.grad is None and not p.requires_grad for p in model.frozen.parameters()) and not model.frozen.training
    model.eval();nn.init.normal_(model.prior.network.out.weight,std=.001);noise=torch.randn_like(r)
    with torch.no_grad():torch.testing.assert_close(model(r,noise),torch.cat([model(v[None],n[None]) for v,n in zip(r,noise)]),atol=3e-6,rtol=3e-5)


@pytest.mark.parametrize('candidate',d.ROWS)
def test_strict_full_checkpoint_roundtrip_and_oldschema_forbidden(tmp_path,candidate):
    model=m.EndpointRestorer(candidate,ActionRestorer(ComplexTFUNet()))
    path=tmp_path/'final.pt';runtime.save_checkpoint(path,model,None,20,4940,parent());fresh,value=runtime.load_checkpoint(path,'cpu')
    assert fresh.candidate==candidate and all(torch.equal(v,fresh.state_dict()[k]) for k,v in model.state_dict().items())
    with pytest.raises(ValueError):old_runtime.validate_payload(value)
    for key,bad in [('training_F_access',True),('target_training',True),('inherited_prior_epochs',20),('candidate','unroll_unshared')]:
        invalid=copy.deepcopy(value);invalid[key]=bad
        with pytest.raises(ValueError):runtime.validate_payload(invalid)
    invalid=copy.deepcopy(value);invalid['config']['candidate_designs'][candidate]['sigma0']=.5
    with pytest.raises(ValueError):runtime.validate_payload(invalid)


def test_sampling_stability_arithmetic_and_protocol_guard_before_truth(tmp_path,monkeypatch):
    logits=np.zeros((3,128,6),np.float32);logits[:,:,0]=1
    assert e.stability(logits)==0
    logits[1,0,:]=0;logits[1,0,1]=1;assert e.stability(logits)==100/128
    with pytest.raises(ValueError):e.stability(logits[:2])
    args=SimpleNamespace(row=d.ROWS[0],output=tmp_path)
    monkeypatch.setattr(e,'original_score',lambda *a:pytest.fail('Truth must stay closed before diagnostics complete'))
    with pytest.raises(FileNotFoundError):e.score(args,'source')
    folder=tmp_path/args.row/'sampling_stability';folder.mkdir(parents=True)
    meta=dict(sample_meta(),status='SOURCE_STABILITY_PREDICTIONS_COMPLETE',view=d.configuration()['stability_view'],state_updates=0,head_fit_updates=0,full_sampling_steps=4,disagreement_percent=100/128)
    (folder/'complete.json').write_text(json.dumps(meta));np.save(folder/'logits.npy',logits)
    assert e.validate_stability(folder)['qualified']
    meta['primary_changed']=True;(folder/'complete.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError):e.validate_stability(folder)


def test_R19_must_be_terminal_and_original_processes_absent(tmp_path):
    from experiments.pamf_endpoint_restore.controller import require_preceding_finished
    root=tmp_path/d.PRECEDING;root.mkdir();proc=tmp_path/'proc';proc.mkdir();args=SimpleNamespace(preceding_run=root)
    (root/'launch.json').write_text(json.dumps(dict(commit=d.PRECEDING_COMMIT,pid=100)))
    (root/'workers.json').write_text(json.dumps(dict(rows=[dict(pid=101)])))
    (root/'complete.json').write_text(json.dumps(dict(status='ANALYZED')))
    require_preceding_finished(args,proc)
    p=proc/'101';p.mkdir();(p/'status').write_text('State: S (sleeping)')
    (p/'cmdline').write_bytes(('python experiments.pamf_flow_start_restore.runner --output '+str(root)).encode())
    with pytest.raises(ValueError):require_preceding_finished(args,proc)
    (p/'status').write_text('State: Z (zombie)');require_preceding_finished(args,proc)
    (root/'complete.json').write_text(json.dumps(dict(status='RUNNING')))
    with pytest.raises(ValueError):require_preceding_finished(args,proc)


def decision():
    return dict(status='SOURCE_DECISION_FROZEN',qualified=True,
                selected=d.ROWS[0],target_rows=[d.ROWS[0]])


def test_unselected_target_and_extra_control_stop_before_target_IO(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'original_predict',lambda *a,**k:pytest.fail('Target IO forbidden'))
    for row,value in [(d.ROWS[1],decision()),(d.ROWS[0],dict(decision(),qualified=False)),
        (d.ROWS[0],dict(decision(),target_rows=[d.ROWS[0],'reference'])),('unroll_unshared',decision())]:
        (tmp_path/'source_decision.json').write_text(json.dumps(value))
        with pytest.raises(ValueError):e.predict(SimpleNamespace(row=row,output=tmp_path),'target')


@pytest.mark.parametrize('role',('source','target'))
def test_new_checkpoint_route_actual_all_seven_predictions(tmp_path,monkeypatch,role):
    import experiments.pamf_uncertainty_restore.evaluation as shared
    source=tmp_path/'input';source.mkdir()
    for view in e.VIEWS:np.save(source/(view+'.npy'),np.ones((16,2,256),np.float32))
    output=tmp_path/'output';row=d.ROWS[0];(output/row/d.KIND).mkdir(parents=True)
    (output/'source_decision.json').write_text(json.dumps(decision()))
    args=SimpleNamespace(row=row,output=output,reader_folder=tmp_path/'unused',source_cache=tmp_path/'unused')
    class Toy(nn.Module):
        def forward(self,x):return x
        def restore(self,x,noise):return x,torch.ones((len(x),256)),None,x
    class Reader(nn.Module):
        def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.ones(1))
        def features(self,x):assert self.anchor._version>=0;return x.flatten(1)[:,:160]
        def classify_features(self,x):return x[:,:6]
    calls=[]
    actual_proxy=e.SampledReader
    monkeypatch.setattr(e,'SampledReader',lambda model,device:actual_proxy(model,'cpu'))
    monkeypatch.setattr(e,'predict_stability',lambda args:None)
    def loader(path,device):
        calls.append(str(path));return Toy(),dict(candidate=row,kind=d.KIND,epoch=20,updates=4940)
    monkeypatch.setattr(e,'load_checkpoint',loader)
    monkeypatch.setattr(shared,'load_checkpoint',lambda *a:pytest.fail('Old default schema loader forbidden'))
    monkeypatch.setattr(shared,'reference',lambda *a:(source,np.arange(16),None,None))
    monkeypatch.setattr(shared,'load_reader',lambda *a:Reader())
    monkeypatch.setattr(shared,'array_tensor',lambda x,device:torch.tensor(x.tolist(),dtype=torch.float32))
    monkeypatch.setattr(shared,'tensor_array',lambda x:np.asarray(x.tolist(),np.float32))
    for key in ('synchronize','reset_peak_memory_stats'):monkeypatch.setattr(torch.cuda,key,lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    e.predict(args,role);path=output/row
    if role=='target':path=path/'target_confirmation'
    assert tuple(shared.prediction_arrays(path/d.KIND/'prediction',np.arange(16),d.KIND,role))==e.VIEWS
    assert len(calls)==1


def test_budget_rows_and_source_reference_wiring_before_GPU():
    from experiments.pamf_endpoint_restore.controller import PATH_ARGS,command
    with pytest.raises(ValueError):t.train(SimpleNamespace(row='unroll_unshared',parent_action=Path(d.ACTION)))
    cfg=d.configuration()
    assert cfg['epochs']==20 and cfg['updates']==4940 and cfg['L_exposures']==126000 and cfg['U_exposures']==1134000
    assert cfg['steps']==4 and cfg['CG_total']==6 and not cfg['training_F_access'] and cfg['calibration_count']==0
    assert cfg['candidate_designs']['path_four']['sigma0']==0 and cfg['candidate_designs']['endpoint_four']['sigma0']==0 and 'preceding_run' in PATH_ARGS
    assert 'unrolled_run' in PATH_ARGS and d.ALL_ROWS==d.ROWS
    args=SimpleNamespace(**{key:Path(key) for key in PATH_ARGS},expected_commit='method')
    cmd=command(args,'train',d.ROWS[0]);assert 'experiments.pamf_endpoint_restore.runner' in cmd and '--unrolled-run' in cmd


def test_target_scoring_saves_goal_assessment_from_completed_summary(tmp_path,monkeypatch):
    from experiments.pamf_native_formal.runner import REFERENCE_RUN
    args=SimpleNamespace(row=d.ROWS[0],output=tmp_path)
    scored=dict(status='ANALYZED',role='target',kind=d.KIND,count=168000,
        F_run=REFERENCE_RUN,F_seed=2026092701,original_default_head=True,
        all_predictions_before_truth=True,records=[dict(dimension='overall',readout='restored',
            view=view,accuracy=71. if view=='practical_low_urban' else 81.,baseline_accuracy=85.)
            for view in e.VIEWS])
    def score(row,role):
        assert row.output==tmp_path/args.row and role=='target'
        folder=row.output/'target_confirmation'/d.KIND;folder.mkdir(parents=True)
        (folder/'summary.json').write_text(json.dumps(scored),encoding='utf-8')
    monkeypatch.setattr(e,'original_score',score)
    e.score(args,'target')
    result=json.loads((tmp_path/args.row/'goal_assessment.json').read_text())
    assert result['status']=='PERFORMANCE_ASSESSED' and result['performance_targets_met']
    assert result['original_clean_accuracy']==85. and result['lowurban_accuracy']==71.
    assert len(result['other_five'])==5 and not result['inference_or_selection_use']


def test_primary_noise_is_explicit_and_independent_of_global_RNG():
    class Noise(nn.Module):
        def restore(self,x,noise):return noise,None,None,None
    r=torch.zeros(2,2,256);first=e.SampledReader(Noise(),'cpu')
    a=first(r);torch.randn(100)
    second=e.SampledReader(Noise(),'cpu');torch.testing.assert_close(a,second(r))
    assert not torch.equal(a,first(r))


def test_actual_three_draw_prediction_reads_only_received_source(tmp_path,monkeypatch):
    cache=tmp_path/'parent/cache/validation';cache.mkdir(parents=True)
    np.save(cache/'practical_low_urban.npy',np.ones((1800,2,256),np.float32))
    output=tmp_path/'output';row='endpoint_four';(output/row).mkdir(parents=True)
    args=SimpleNamespace(parent_action=tmp_path/'parent',output=output,row=row,reader_folder=tmp_path/'none',source_cache=tmp_path/'none')
    class Toy(nn.Module):
        def forward(self,x,noise):return x+.02*noise
    class Reader(nn.Module):
        def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.ones(1))
        def features(self,x):assert self.anchor._version>=0;return x.flatten(1)[:,:160]
        def classify_features(self,x):return x[:,:6]
    generator=torch.Generator
    monkeypatch.setattr(torch,'Generator',lambda **k:generator(device='cpu'))
    monkeypatch.setattr(e,'load_checkpoint',lambda *a:(Toy(),dict(candidate=row)))
    monkeypatch.setattr(e,'load_reader',lambda *a:Reader())
    monkeypatch.setattr(e,'array_tensor',lambda x,device:torch.tensor(x.tolist(),dtype=torch.float32))
    monkeypatch.setattr(e,'tensor_array',lambda x:np.asarray(x.detach().tolist(),np.float32))
    load=np.load;reads=[]
    def guarded(path,*a,**kw):
        reads.append(str(path));assert 'labels.npy' not in str(path) and 'clean.npy' not in str(path)
        return load(path,*a,**kw)
    monkeypatch.setattr(np,'load',guarded)
    e.predict_stability(args)
    result=e.validate_stability(output/row/'sampling_stability')
    assert result['count']==128 and result['seeds']==list(d.STABILITY_SEEDS) and result['TX_truth_reads']==0
    assert not result['primary_changed'] and len(reads)==2


def test_successful_source_score_preserves_original_summary_and_writes_flow_summary(tmp_path,monkeypatch):
    args=SimpleNamespace(output=tmp_path,row='path_four')
    folder=tmp_path/args.row/'sampling_stability';folder.mkdir(parents=True)
    logits=np.zeros((3,128,6),np.float32);np.save(folder/'logits.npy',logits)
    meta=dict(sample_meta(),status='SOURCE_STABILITY_PREDICTIONS_COMPLETE',view=d.configuration()['stability_view'],
        state_updates=0,head_fit_updates=0,full_sampling_steps=4)
    e.write_json(folder/'complete.json',meta)
    original=dict(status='ANALYZED',role='source',count=1800,records=[],all_predictions_before_truth=True)
    def score(row,role):
        assert role=='source' and row.output==tmp_path/args.row
        target=row.output/d.KIND;target.mkdir();e.write_json(target/'summary.json',original)
    monkeypatch.setattr(e,'original_score',score)
    e.score(args,'source')
    path=tmp_path/args.row/d.KIND
    assert e.read_json(path/'summary.json')==original
    enriched=e.read_json(path/'endpoint_source_summary.json')
    assert enriched['sampling_stability']['qualified'] and enriched['count']==1800


def test_endpoint_loss_is_actual_four_step_MAP_and_all_step_gradients(monkeypatch):
    torch.manual_seed(9);model=m.EndpointRestorer('endpoint_four',Frozen()).train();x=torch.randn(2,2,256)*.3;r=x+.02
    with torch.no_grad():expected=model(r)
    outputs=[];solver=[];solve=m.weighted_map
    def retain(module,inputs,value):value.retain_grad();outputs.append(value)
    handle=model.prior.network.register_forward_hook(retain)
    def capture(*args,**kwargs):solver.append(kwargs['steps']);return solve(*args,**kwargs)
    monkeypatch.setattr(m,'weighted_map',capture)
    gen=torch.Generator().manual_seed(31);reference=torch.Generator().manual_seed(31)
    torch.randn(x.shape,generator=reference);time=torch.rand(len(x),generator=reference)
    loss,metrics=model.training_loss(x,r,gen);handle.remove()
    torch.testing.assert_close(loss,(expected-x).square().mean((1,2))/x.square().mean((1,2)).clamp_min(1e-4))
    torch.testing.assert_close(metrics['endpoint_iq_nmse'],loss);torch.testing.assert_close(metrics['flow_time'],time)
    assert torch.equal(gen.get_state(),reference.get_state()) and len(outputs)==4 and solver==[6]
    loss.mean().backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in outputs)
    assert all(p.grad is None and not p.requires_grad for p in model.frozen.parameters())


def test_endpoint_training_never_feeds_clean_into_forward_states(monkeypatch):
    torch.manual_seed(4);model=m.EndpointRestorer('endpoint_four',Frozen()).eval();r=torch.randn(2,2,256)*.3
    field=model.prior.field;seen=[]
    def capture(point,condition,time):seen.append((point.detach().clone(),time.detach().clone()));return field(point,condition,time)
    monkeypatch.setattr(model.prior,'field',capture)
    model.training_loss(r+1,r,torch.Generator().manual_seed(3));first=seen[:];seen.clear()
    model.training_loss(r-2,r,torch.Generator().manual_seed(3))
    assert len(first)==len(seen)==4
    for (p,t),(q,u) in zip(first,seen):torch.testing.assert_close(p,q);torch.testing.assert_close(t,u)
    assert [float(t[0]) for _,t in seen]==[0.,.25,.5,.75]


def alternating_reference(root):
    from experiments.pamf_alternating_restore import design as alternate
    path=root/d.ALTERNATING_CANDIDATE;source=path/d.KIND;source.mkdir(parents=True)
    values={root/'launch.json':dict(commit=d.ALTERNATING_COMMIT),root/'resolved_config.json':alternate.configuration(),
        root/'source_decision.json':dict(qualified=True,selected=d.ALTERNATING_CANDIDATE,training_TX_F_access=False,target_reads_before_decision=0,target_feedback=False),
        path/'training_complete.json':dict(status='SOURCE_TRAINED',epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
            TX_labels_read=0,F_outputs_read=0,F_weights_read=0,target_reads=0,frozen_context_updates=0,prior_initialization='scratch',inherited_prior_epochs=0),
        source/'summary.json':summary(low=88.,mid=92.,nmse=.022)}
    for path,value in values.items():path.write_text(json.dumps(value),encoding='utf-8')
    return values


def test_fifth_reference_reads_only_R19_source_before_any_target(tmp_path,monkeypatch):
    root=tmp_path/d.ALTERNATING;alternating_reference(root);reads=[];read=e.read_json
    monkeypatch.setattr(e,'validate_old_references',lambda *a:None)
    def capture(path):
        reads.append(path);assert 'target_confirmation' not in str(path) and 'goal_assessment' not in str(path)
        return read(path)
    monkeypatch.setattr(e,'read_json',capture)
    result=e.validate_references(None,None,None,root)
    assert result['role']=='source' and root/'complete.json' not in reads


@pytest.mark.parametrize('filename,key,bad',[
    ('training_complete.json','F_weights_read',1),('training_complete.json','target_reads',1),
    ('training_complete.json','U_exposures',0),('summary.json','target_access',True),
    ('summary.json','role','target'),('source_decision.json','selected','time_frequency'),
    ('source_decision.json','target_feedback',True)])
def test_fifth_reference_permission_or_source_selection_mismatch_rejected(tmp_path,monkeypatch,filename,key,bad):
    root=tmp_path/d.ALTERNATING;values=alternating_reference(root);path=next(p for p in values if p.name==filename)
    values[path][key]=bad;path.write_text(json.dumps(values[path]),encoding='utf-8')
    monkeypatch.setattr(e,'validate_old_references',lambda *a:None)
    with pytest.raises(ValueError):e.validate_references(None,None,None,root)


def test_five_source_guards_and_fixed_one_winner_including_R19_TX_damage():
    single=summary(low=84.,mid=90.);native=summary(low=82.,mid=89.);fft=summary(low=83.,mid=89.5)
    unrolled=summary(low=86.,mid=91.,nmse=.025);alternating=summary(low=88.,mid=92.,nmse=.023)
    candidates={name:summary(low=89.+i,mid=93.+i,nmse=.022) for i,name in enumerate(d.ROWS)}
    result=e.select_source(candidates,single,native,fft,unrolled,alternating)
    assert result['selected']=='endpoint_four' and result['target_rows']==['endpoint_four']
    next(r for r in alternating['records'] if r['view']=='practical_low_urban' and r['dimension']=='TX')['accuracy']=98.
    result=e.select_source(candidates,single,native,fft,unrolled,alternating)
    assert not result['qualified'] and not result['target_rows']
    assert all(v['vs_R17_native']['qualified'] and not v['vs_R19_native']['qualified'] for v in result['checks'].values())


def test_foreign_R20_schema_rejected_and_training_compute_cost_explicit(tmp_path):
    from experiments.pamf_flow_start_restore.runtime import validate_payload as old_flow
    model=m.EndpointRestorer('endpoint_four',ActionRestorer(ComplexTFUNet()));path=tmp_path/'final.pt'
    runtime.save_checkpoint(path,model,None,20,4940,parent());_,value=runtime.load_checkpoint(path,'cpu')
    with pytest.raises(ValueError):old_flow(value)
    cfg=d.configuration();assert cfg['training_Core_calls']=={'path_four':1,'endpoint_four':4}
    assert cfg['training_CG_iterations']=={'path_four':0,'endpoint_four':6} and cfg['inference_Core_calls']==4
    assert cfg['alternating_source_run']==d.ALTERNATING
