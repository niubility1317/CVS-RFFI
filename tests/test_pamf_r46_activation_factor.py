"""Synthetic IQ and checkpoint tests; no project IQ, labels, F or remote access."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as ParentModel
from experiments.pamf_r46_activation_factor import design as d, models as m, runtime as r, calibration as c, evaluation as e, runner
from experiments.pamf_uncertainty_restore import design as ancestor
from tools.r46_activation_factorization import SharedInputMoments

torch.set_num_threads(2)


def lineage():
    return dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,
        action_run=ancestor.ACTION_RUN,action_commit=ancestor.ACTION_COMMIT,
        data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)


def parent_model():
    model=ParentModel('prune_refine').eval().requires_grad_(False)
    with torch.no_grad():
        for core in model.prior:core.out.weight.normal_(std=.001)
        model.frozen.coarse.temporal.out.weight.normal_(std=.001)
        model.frozen.estimator.core.out.weight.normal_(std=.001)
        model.frozen.estimator.global_head.weight.normal_(std=.001)
    return model


def calibration_payload(moments, groups=None):
    # Production metadata around synthetic moments exercises the loader contract.
    # This fixture is never persisted as an actual source-calibration result.
    return dict(schema=c.SCHEMA,run=d.RUN,config=d.calibration_configuration(),status='SOURCE_CALIBRATED',
        moments=moments,teacher_examples=d.CALIBRATION_IQ,unique_source_positions=d.CAL_UNIQUE,
        L_exposures=d.CAL_PER_ROLE*len(d.CAL_VIEWS),U_exposures=d.CAL_PER_ROLE*len(d.CAL_VIEWS),
        F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0,optimizer_updates=0,parent_unchanged=True,groups=groups or {})


@pytest.fixture(scope='module')
def synthetic_calibration():
    torch.manual_seed(53);parent=parent_model()
    with SharedInputMoments(parent,.5,d.CAL_POSITIONS) as collector:
        collector.observe(torch.randn(2,2,256)*.1)
    moments,report=collector.finish()
    return parent,calibration_payload(moments,report['groups'])


@pytest.mark.parametrize('arm',d.ARMS)
def test_full_model_joint_gradients_teacher_detach_extra_clean_and_strict_reload(tmp_path,arm,synthetic_calibration):
    parent,calibration=synthetic_calibration;before=copy.deepcopy(parent.state_dict())
    model=m.compact_from_parent(parent,arm,calibration);model.initialization_report['seconds']=0.
    assert sum(p.numel() for p in model.parameters())==d.PARAMETERS[arm]
    assert sum(p.numel() for p in model.frozen.parameters())==d.CONTEXT_PARAMETERS[arm]
    calls=dict(teacher=0,student=0)
    def hook(name):
        def record(module,inputs,output):calls[name]+=1
        return record
    handles=[v.register_forward_hook(hook('teacher')) for v in parent.prior]
    handles += [v.register_forward_hook(hook('student')) for v in model.prior]
    clean=(torch.randn(2,2,256)*.1).requires_grad_(True);received=clean.detach()+torch.randn_like(clean)*.005
    outputs,desired,identity=m.paired_outputs(model,parent,received,clean)
    for handle in handles:handle.remove()
    assert calls==dict(teacher=3,student=6)
    desired=[v.detach().requires_grad_(True) for v in desired]
    loss,terms=m.objective(outputs,desired,clean,1,arm,identity)
    torch.testing.assert_close(loss,terms['iq_nmse']+.05*terms['spectral_error']+terms['teacher_stage_nmse']+.02*terms['balanced_complex_nmse']+.25*terms['clean_identity_nmse'])
    loss.mean().backward()
    for group in (model.frozen.coarse,model.frozen.estimator,*model.prior):
        grads=[p.grad for p in group.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
    assert clean.grad is None and all(v.grad is None for v in desired)
    assert all(p.grad is None for p in parent.parameters()) and all(torch.equal(v,before[k]) for k,v in parent.state_dict().items())
    torch.optim.AdamW(model.parameters(),lr=2e-4).step();model.eval()
    path=tmp_path/'final.pt';r.save_checkpoint(path,model,None,20,19700,lineage());fresh,value=r.load_checkpoint(path,'cpu')
    assert value['optimizer'] is None and value['cumulative_component_epochs']==100 and not value['training_F_access']
    with torch.no_grad():assert torch.equal(model(received),fresh(received))
    with pytest.raises(FileExistsError):r.save_checkpoint(path,model,None,20,19700,lineage())


@pytest.fixture(scope='module')
def payload(synthetic_calibration):
    model=m.compact_from_parent(synthetic_calibration[0],d.ARMS[0],synthetic_calibration[1]);model.initialization_report['seconds']=0.
    return r.metadata(d.ARMS[0],20,19700,lineage(),model.initialization_report)


@pytest.mark.parametrize('field,bad',[('parent_R46_row','retain_refine'),('parent_R46_commit','unknown'),
    ('training_F_access',True),('F_outputs_read',1),('target_training',True),('V_training',True),('updates',19701),('epoch',21)])
def test_checkpoint_rejects_other_sources_or_permissions(payload,field,bad):
    value=copy.deepcopy(payload);r.validate_payload(value);value[field]=bad
    with pytest.raises(ValueError):r.validate_payload(value)


@pytest.mark.parametrize('field,bad',[('IQ_examples',0),('unique_source_positions',128),('calibration_L_exposures',0),
    ('calibration_U_exposures',0),('positions_per_conv',16),('ridge_fraction',.01),('rank_fraction',.5),('activation_calibrated',False)])
def test_checkpoint_rejects_data_free_or_changed_activation_initialization(payload,field,bad):
    value=copy.deepcopy(payload);value['initialization'][field]=bad
    with pytest.raises(ValueError):r.validate_payload(value)


def test_wrong_objective_config_and_alias_conflict_rejected(payload):
    value=copy.deepcopy(payload);value['config']['balanced_complex_weight']=.04
    with pytest.raises(ValueError):r.validate_payload(value)
    model=m.CompactSuccessor(d.ARMS[1]);state=copy.deepcopy(model.state_dict())
    key=next(k for k in state if k.startswith('prior.1.') and k.endswith('net.0.left.weight'));state[key]=state[key]+1
    with pytest.raises(ValueError,match='Conflicting'):r.validate_aliases(model,state)


def test_bad_initialization_arm_and_calibration_path_rejected_before_io(monkeypatch):
    monkeypatch.setattr(r,'validate_parent_root',lambda *a:pytest.fail('No parent IO'))
    with pytest.raises(ValueError):r.initialize('complex625','/runs','cpu','/source.pt')
    monkeypatch.setattr(c.torch,'load',lambda *a,**k:pytest.fail('No wrong calibration IO'))
    with pytest.raises(ValueError):c.load_calibration(Path('/other/source_moments.pt'))


@pytest.mark.parametrize('field,bad',[('teacher_examples',256),('TX_labels_read',1),('F_reads',1),('V_reads',1),
    ('target_reads',1),('optimizer_updates',1),('parent_unchanged',False)])
def test_calibration_permissions_and_exposure_rejected(synthetic_calibration,field,bad):
    value=dict(synthetic_calibration[1]);value[field]=bad
    with pytest.raises(ValueError):c.validate_calibration(value,synthetic_calibration[0])


def test_role_sampling_opaque_balanced_and_wrong_roles_rejected():
    roles=np.array(['L_s']*6300+['U_s']*56700)
    take=c.choose_positions(roles)
    assert len(take)==len(set(take))==256 and (roles[take]=='L_s').sum()==128
    bad=roles.copy();bad[0]='V'
    with pytest.raises(ValueError):c.choose_positions(bad)


def test_prepare_reads_only_permitted_source_views_and_preserves_parent(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'CAL_PER_ROLE',2);monkeypatch.setattr(d,'CAL_UNIQUE',4);monkeypatch.setattr(d,'CALIBRATION_IQ',12)
    parent=parent_model();before=copy.deepcopy(parent.state_dict());roles=np.array(['L_s']*6300+['U_s']*56700)
    args=SimpleNamespace(output=tmp_path/d.RUN,parent_action=tmp_path/d.ACTION_RUN,
        source_cache=tmp_path/'source_cache',compact_parent=tmp_path/d.COMPACT_PARENT_RUN)
    args.output.mkdir();seen=[]
    class SyntheticIQ:
        shape=(63000,2,256);dtype=np.dtype('float32')
        def __getitem__(self,indices):return np.full((len(indices),2,256),.1,dtype=np.float32)
    def load(path,**kwargs):
        rel=Path(path).relative_to(args.parent_action/'cache').as_posix();seen.append(rel)
        if rel=='train/roles.npy':return roles
        assert rel in ('train/clean.npy','epochs/01/received.npy','epochs/20/received.npy')
        return SyntheticIQ()
    monkeypatch.setattr(c.np,'load',load)
    monkeypatch.setattr(c,'validate_parent_root',lambda *a:None)
    monkeypatch.setattr(c,'read_json',lambda *a:dict(status='READY',native_role_ids='EXACT_MATCH'))
    monkeypatch.setattr(c.parent_runtime,'load_checkpoint',lambda *a:(parent,dict(arm=d.COMPACT_PARENT_ROW,epoch=20,updates=4940)))
    report=c.prepare(args)
    assert seen==['train/roles.npy','train/clean.npy','epochs/01/received.npy','epochs/20/received.npy']
    assert report['teacher_examples']==12 and report['L_exposures']==report['U_exposures']==6
    assert all(torch.equal(v,before[k]) for k,v in parent.state_dict().items())
    assert all(not module._forward_pre_hooks for module in parent.modules())
    value=c.load_calibration(args.output/'source_moments.pt',parent)
    assert value['teacher_examples']==12 and value['F_reads']==value['TX_labels_read']==value['target_reads']==0
    with pytest.raises(FileExistsError):c.prepare(args)


@pytest.mark.parametrize('arm',d.ARMS)
def test_actual_identity_branch_required_for_both_arms(arm):
    x=torch.randn(2,2,256)*.1
    with pytest.raises(ValueError):m.objective([x]*3,[x]*3,x,20,arm,None)
    identity=(x+.02).requires_grad_();loss,_=m.objective([x]*3,[x]*3,x,20,arm,identity)
    loss.sum().backward();assert identity.grad.abs().sum()>0


def own_run(tmp_path,arm):
    root=tmp_path/d.RUN;root.mkdir()
    (root/'launch.json').write_text(json.dumps(dict(commit='method')))
    (root/'comparison_plan.json').write_text(json.dumps(d.configuration()))
    return SimpleNamespace(output=root,arm=arm,expected_commit='method')


@pytest.mark.parametrize('arm',d.ARMS)
def test_full_formal_chain_has_no_source_gain_or_control_availability_gate(tmp_path,monkeypatch,arm):
    args=own_run(tmp_path,arm);seen=[]
    def train(a):(a.output/a.arm).mkdir();seen.append('train')
    monkeypatch.setattr(runner,'train',train)
    monkeypatch.setattr(runner,'predict',lambda a,role:seen.append('predict_'+role))
    monkeypatch.setattr(runner,'profile',lambda *a:seen.append('profile'))
    monkeypatch.setattr(runner,'score_subprocess',lambda a,role:seen.append('score_'+role))
    monkeypatch.setattr(runner.torch.cuda,'empty_cache',lambda:None);runner.worker(args)
    assert seen==['train','predict_source','profile','score_source','predict_target','score_target']
    assert (args.output/arm/'complete.json').exists() and len(d.VIEWS)==7


def test_existing_row_rejected_without_failure_marker(tmp_path,monkeypatch):
    args=own_run(tmp_path,d.ARMS[0]);row=args.output/args.arm;row.mkdir();marker=row/'complete.json';marker.write_text('{}')
    monkeypatch.setattr(runner,'train',lambda *a:pytest.fail('Existing row cannot train'))
    with pytest.raises(ValueError,match='existing outputs'):runner.worker(args)
    assert list(row.iterdir())==[marker]


def test_truth_waits_for_complete_prediction_artifacts(monkeypatch):
    args=SimpleNamespace(output=Path('/runs')/d.RUN,arm=d.ARMS[0]);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(e,'reference',lambda *a:(Path('/formal'),e.np.arange(2),None,None))
    class Incomplete(Exception):pass
    def stop(*a):raise Incomplete()
    monkeypatch.setattr(e.base,'prediction_arrays',stop)
    monkeypatch.setattr(e.base,'target_baseline',lambda *a:pytest.fail('Truth must stay closed'))
    with pytest.raises(Incomplete):e.score(args,'target')


def test_predecessor_terminal_file_is_insufficient_while_owned_process_alive(tmp_path,monkeypatch):
    root=tmp_path/'runs';prior=root/d.PREDECESSOR;prior.mkdir(parents=True);row=prior/'complex625';row.mkdir()
    (row/'complete.json').write_text('{}');(prior/'workers.json').write_text(json.dumps(dict(workers=[dict(GPU=2,arm='complex625',pid=12345)])))
    fake=tmp_path/'proc/12345';fake.mkdir(parents=True);(fake/'stat').write_text('12345 (python) S 1')
    (fake/'cmdline').write_bytes(('\0'.join(['python',d.PREDECESSOR_MODULE[d.PREDECESSOR],str(prior)])+'\0').encode())
    real_path=runner.Path
    monkeypatch.setattr(runner,'Path',lambda value:tmp_path/'proc' if str(value)=='/proc' else real_path(value))
    args=SimpleNamespace(output=root/d.RUN)
    assert not runner.predecessor_ready(args,d.ARMS[0])
    (fake/'stat').write_text('12345 (python) Z 1');assert runner.predecessor_ready(args,d.ARMS[0])


def test_fixed_configuration_is_source_calibrated_with_separate_exposure_and_complete_matrix():
    cfg=d.configuration();assert json.loads(json.dumps(cfg))==cfg
    assert d.EPOCHS==20 and d.UPDATES==19700 and d.BATCH==d.MICRO==64
    assert cfg['initialization_IQ_examples']==768 and cfg['calibration']['unique_source_positions']==256
    assert cfg['calibration']['L_exposures']==cfg['calibration']['U_exposures']==384
    assert cfg['extra_clean_student_inputs']=={a:1260000 for a in d.ARMS}
    assert cfg['F_weights_read']==cfg['F_outputs_read']==cfg['TX_labels_read']==0
    assert d.CLEAN_WEIGHT=={a:.25 for a in d.ARMS} and d.COMPLEX_WEIGHT==.02
    assert d.learning_rate(19699)==pytest.approx(2e-5) and d.teacher_weight(20)==pytest.approx(.1)
