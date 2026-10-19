import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_alternating_restore import design as d,models as m,runtime,evaluation as e,training as t
from experiments.pamf_unrolled_restore.models import UnrolledRestorer
from experiments.pamf_unrolled_restore import runtime as old_runtime,design as reference_design
from experiments.pamf_action_restore.models import SCALE,ActionRestorer
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_restoration.models import complex_iq,pair
from test_pamf_unrolled_restore import Frozen,parent,summary

torch.set_num_threads(2)


def test_exact_auxiliary_transforms_and_negative_frequency_mirror():
    signals=[torch.randn(2,2,256,requires_grad=True) for _ in range(4)]
    logvar=torch.randn(2,1,256)
    time=m.auxiliary_inputs(signals,logvar,False);freq=m.auxiliary_inputs(signals,logvar,True)
    assert time.shape==freq.shape==(2,17,256)
    torch.testing.assert_close(time[:,-1:],logvar);torch.testing.assert_close(freq[:,-1:],logvar)
    for k,signal in enumerate(signals):
        spectrum=torch.fft.fft(complex_iq(signal),norm='ortho')
        torch.testing.assert_close(time[:,2*k:2*k+2],signal)
        torch.testing.assert_close(time[:,8+2*k:10+2*k],pair(complex_iq(signal).conj()))
        torch.testing.assert_close(freq[:,2*k:2*k+2],pair(spectrum))
        torch.testing.assert_close(freq[:,8+2*k:10+2*k],pair(torch.fft.fft(complex_iq(signal).conj(),norm='ortho')))
        torch.testing.assert_close(spectrum.abs().square().sum(-1),signal.square().sum((1,2)))
    freq.square().mean().backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in signals)
    with pytest.raises(ValueError):m.auxiliary_inputs(signals[:3],logvar,True)


class ConstantCore(nn.Module):
    def __init__(self,value):super().__init__();self.value=nn.Parameter(torch.tensor(float(value)));self.calls=0
    def forward(self,x):self.calls+=1;return self.value*torch.ones_like(x[:,:2])


@pytest.mark.parametrize('candidate',d.ROWS)
def test_actual_IFFT_correction_coarse_anchor_CG_schedule_and_six_calls(candidate,monkeypatch):
    calls=[]
    def solve(action,observed,prior,variance,**kw):calls.append(kw['steps']);return prior
    monkeypatch.setattr(m,'weighted_map',solve)
    model=m.AlternatingRestorer(candidate,Frozen())
    model.prior=nn.ModuleList(m.Stage(ConstantCore(1),ConstantCore(2)) for _ in range(3))
    x=torch.randn(2,2,256)*.3;stages,*_=model.stage_outputs(x)
    correction=torch.full_like(x,2.)
    if candidate=='time_frequency':correction=pair(torch.fft.ifft(complex_iq(correction),norm='ortho'))
    torch.testing.assert_close(stages[-1],.9*x+SCALE+SCALE*correction)
    assert calls==[2,2,2] and len(stages)==3
    assert all(core.calls==1 for stage in model.prior for core in (stage.temporal,stage.auxiliary))
    # This analytic solver stub drops warm-start dependence; real final-loss
    # connectivity is checked below with the actual CG solver and real Cores.
    sum(stage.sum() for stage in stages).backward()
    assert all(core.value.grad is not None and core.value.grad>0 for stage in model.prior for core in (stage.temporal,stage.auxiliary))


def test_matching_full_new_initial_states_R17_temporal_state_outputs_and_parameters():
    torch.manual_seed(123);reference=UnrolledRestorer('unroll_unshared',Frozen()).eval()
    original=reference.state_dict();initial=None;counts=[];x=torch.randn(2,2,256)*.3
    with torch.no_grad():expected=reference(x)
    for candidate in d.ROWS:
        torch.manual_seed(123);model=m.AlternatingRestorer(candidate,Frozen()).eval();state=model.state_dict()
        if initial is None:initial={k:v.clone() for k,v in state.items()}
        assert set(initial)==set(state) and all(torch.equal(v,initial[k]) for k,v in state.items())
        for k,v in original.items():
            parts=k.split('.');key='.'.join(parts[:2]+['temporal']+parts[2:]) if parts[0]=='prior' else k
            assert torch.equal(v,state[key])
        with torch.no_grad():torch.testing.assert_close(model(x),expected,atol=1e-6,rtol=1e-5)
        assert len(model.prior)==3
        counts.append(sum(p.numel() for p in model.prior.parameters()))
    assert len(set(counts))==1 and counts[0]==d.TRAINABLE_PARAMETERS==3926220


@pytest.mark.parametrize('candidate',d.ROWS)
def test_real_solver_all_six_cores_trainable_and_receive_gradients_packet_independent(candidate):
    torch.manual_seed(7);model=m.AlternatingRestorer(candidate,Frozen()).train()
    assert not model.frozen.training and all(not p.requires_grad for p in model.frozen.parameters())
    optimized=list(model.prior.parameters())
    assert set(map(id,optimized))=={id(p) for p in model.parameters() if p.requires_grad}
    x=torch.randn(2,2,256)*.3;stages,*_=model.stage_outputs(x)
    (stages[-1]-x-.01).square().mean().backward()
    for stage in model.prior:
        for core in (stage.temporal,stage.auxiliary):
            gradients=[p.grad for p in core.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert sum(float(g.abs().sum()) for g in gradients)>0
    assert all(p.grad is None for p in model.frozen.parameters())
    model.eval()
    for stage in model.prior:
        for core in (stage.temporal,stage.auxiliary):nn.init.normal_(core.out.weight,std=.001)
    with torch.no_grad():torch.testing.assert_close(model(x),torch.cat([model(v[None]) for v in x]),atol=3e-6,rtol=3e-5)


@pytest.mark.parametrize('candidate',d.ROWS)
def test_strict_full_checkpoint_roundtrip_and_foreign_schema_forbidden(tmp_path,candidate):
    model=m.AlternatingRestorer(candidate,ActionRestorer(ComplexTFUNet()))
    path=tmp_path/'final.pt';runtime.save_checkpoint(path,model,None,20,4940,parent())
    fresh,value=runtime.load_checkpoint(path,'cpu')
    assert fresh.candidate==candidate and all(torch.equal(v,fresh.state_dict()[k]) for k,v in model.state_dict().items())
    with pytest.raises(ValueError):old_runtime.validate_payload(value)
    for key,bad in [('training_F_access',True),('target_training',True),('inherited_prior_epochs',20),('candidate','unroll_unshared')]:
        invalid=copy.deepcopy(value);invalid[key]=bad
        with pytest.raises(ValueError):runtime.validate_payload(invalid)
    invalid=copy.deepcopy(value);invalid['config']['FFT_normalization']='backward'
    with pytest.raises(ValueError):runtime.validate_payload(invalid)


def test_seed_reset_after_parent_loading(monkeypatch):
    def load(*a):torch.randn(load.draws);return Frozen(),parent()
    monkeypatch.setattr(runtime,'frozen_context',load)
    load.draws=3;first,_=runtime.initialize(d.ROWS[0],None,'cpu')
    load.draws=99;second,_=runtime.initialize(d.ROWS[1],None,'cpu')
    assert all(torch.equal(v,second.prior.state_dict()[k]) for k,v in first.prior.state_dict().items())


def write_reference(root):
    row=root/d.REFERENCE_CANDIDATE;source=row/d.KIND;source.mkdir(parents=True)
    values={root/'launch.json':dict(commit=d.UNROLLED_COMMIT),
        root/'resolved_config.json':reference_design.configuration(),
        root/'source_decision.json':dict(qualified=True,selected=d.REFERENCE_CANDIDATE,
            training_TX_F_access=False,target_reads_before_decision=0,target_feedback=False),
        root/'complete.json':dict(status='ANALYZED',source_only=False,target_rows=[d.REFERENCE_CANDIDATE]),
        row/'training_complete.json':dict(status='SOURCE_TRAINED',epochs=20,updates=4940,
            L_exposures=126000,U_exposures=1134000,TX_labels_read=0,F_outputs_read=0,
            F_weights_read=0,target_reads=0,frozen_context_updates=0,
            prior_initialization='scratch',inherited_prior_epochs=0),
        source/'summary.json':summary(low=86.,mid=91.,nmse=.025)}
    for path,value in values.items():path.write_text(json.dumps(value))
    return values


def test_R17_frozen_after_evaluation_reference_uses_only_source_records(tmp_path,monkeypatch):
    root=tmp_path/d.UNROLLED;write_reference(root);reads=[];read=e.read_json
    def capture(path):
        reads.append(path);assert 'target_confirmation' not in str(path) and 'goal_assessment' not in str(path)
        return read(path)
    monkeypatch.setattr(e,'read_json',capture);monkeypatch.setattr(e,'validate_old_references',lambda *a:None)
    result=e.validate_references(tmp_path/'single',tmp_path/'fft',root)
    assert result['role']=='source' and root/'complete.json' not in reads


@pytest.mark.parametrize('filename,key,bad',[
    ('training_complete.json','target_reads',1),('training_complete.json','F_weights_read',1),
    ('summary.json','role','target'),('summary.json','target_access',True),
    ('source_decision.json','target_feedback',True),('source_decision.json','selected','unroll_shared')])
def test_invalid_source_reference_rejected(tmp_path,monkeypatch,filename,key,bad):
    root=tmp_path/d.UNROLLED;values=write_reference(root)
    path=next(p for p in values if p.name==filename);values[path][key]=bad;path.write_text(json.dumps(values[path]))
    monkeypatch.setattr(e,'validate_old_references',lambda *a:None)
    with pytest.raises(ValueError):e.validate_references(tmp_path/'single',tmp_path/'fft',root)


def test_four_reference_guard_and_fixed_single_target_selection():
    single=summary(low=84.,mid=90.);native=summary(low=82.,mid=89.)
    fft=summary(low=83.,mid=89.5);latest=summary(low=86.,mid=91.,nmse=.025)
    candidates={d.ROWS[0]:summary(low=87.,mid=92.,nmse=.024),d.ROWS[1]:summary(low=88.,mid=93.,nmse=.024)}
    result=e.select_source(candidates,single,native,fft,latest)
    assert result['selected']==d.ROWS[1] and result['target_rows']==[d.ROWS[1]]
    next(r for r in latest['records'] if r['view']=='practical_low_urban' and r['dimension']=='TX')['accuracy']=94.
    result=e.select_source(candidates,single,native,fft,latest)
    assert not result['qualified'] and result['target_rows']==[]
    assert all(v['vs_previous']['qualified'] for v in result['checks'].values())
    candidates[d.ROWS[0]]['role']='target'
    with pytest.raises(ValueError):e.select_source(candidates,single,native,fft,latest)


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
        def training_outputs(self,x):return x,torch.ones((len(x),256)),None,x
    class Reader(nn.Module):
        def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.ones(1))
        def features(self,x):assert self.anchor._version>=0;return x.flatten(1)[:,:160]
        def classify_features(self,x):return x[:,:6]
    calls=[]
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
    from experiments.pamf_alternating_restore.controller import PATH_ARGS,command
    with pytest.raises(ValueError):t.train(SimpleNamespace(row='unroll_unshared',parent_action=Path(d.ACTION)))
    cfg=d.configuration()
    assert cfg['epochs']==20 and cfg['updates']==4940 and cfg['L_exposures']==126000 and cfg['U_exposures']==1134000
    assert cfg['steps']==3 and cfg['CG_total']==6 and not cfg['training_F_access'] and cfg['calibration_count']==0
    assert cfg['prior_Core_calls']==6 and all(sum(spec['CG_schedule'])==6 for spec in cfg['candidate_designs'].values())
    assert 'unrolled_run' in PATH_ARGS and d.ALL_ROWS==d.ROWS
    args=SimpleNamespace(**{key:Path(key) for key in PATH_ARGS},expected_commit='method')
    cmd=command(args,'train',d.ROWS[0]);assert 'experiments.pamf_alternating_restore.runner' in cmd and '--unrolled-run' in cmd


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
