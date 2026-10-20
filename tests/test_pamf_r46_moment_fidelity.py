import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as ParentModel
from experiments.pamf_r46_moment_fidelity import design as d, models as m, runtime as r, evaluation as e, runner
from experiments.pamf_uncertainty_restore import design as ancestor

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


@pytest.mark.parametrize('arm',d.ARMS)
def test_full_model_joint_gradients_frozen_teacher_branch_exposure_and_strict_reload(tmp_path,arm):
    torch.manual_seed(43);parent=parent_model();before=copy.deepcopy(parent.state_dict())
    model=m.compact_from_parent(parent,arm);model.initialization_report['seconds']=0.
    assert sum(p.numel() for p in model.parameters())==278180
    assert sum(p.numel() for p in model.frozen.parameters())==186703
    calls=dict(teacher=0,student=0)
    def hook(name):
        def record(module,inputs,output):calls[name]+=1
        return record
    handles=[v.register_forward_hook(hook('teacher')) for v in parent.prior]
    handles += [v.register_forward_hook(hook('student')) for v in model.prior]
    clean=(torch.randn(2,2,256)*.1).requires_grad_(True);received=clean.detach()+torch.randn_like(clean)*.005
    outputs,desired,identity=m.paired_outputs(model,parent,received,clean)
    for handle in handles:handle.remove()
    assert calls==dict(teacher=3,student=3+3*int(d.CLEAN_WEIGHT[arm]>0))
    desired=[v.detach().requires_grad_(True) for v in desired]
    loss,terms=m.objective(outputs,desired,clean,1,arm,identity)
    torch.testing.assert_close(loss,terms['iq_nmse']+.05*terms['spectral_error']+terms['teacher_stage_nmse']+.02*terms['balanced_complex_nmse']+d.CLEAN_WEIGHT[arm]*terms['clean_identity_nmse']+d.MOMENT_WEIGHT*terms['moment_objective'])
    loss.mean().backward()
    for group in (model.frozen.coarse,model.frozen.estimator,*model.prior):
        grads=[p.grad for p in group.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
    assert clean.grad is None and all(v.grad is None for v in desired)
    assert all(p.grad is None for p in parent.parameters()) and all(torch.equal(v,before[k]) for k,v in parent.state_dict().items())
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4);optimizer.step();model.eval()
    path=tmp_path/'final.pt';r.save_checkpoint(path,model,None,20,19700,lineage());fresh,value=r.load_checkpoint(path,'cpu')
    assert value['optimizer'] is None and value['cumulative_component_epochs']==100 and not value['training_F_access']
    with torch.no_grad():assert torch.equal(model(received),fresh(received))
    with pytest.raises(FileExistsError):r.save_checkpoint(path,model,None,20,19700,lineage())


@pytest.mark.parametrize('field,bad',[('parent_R46_row','retain_refine'),('parent_R46_commit','unknown'),
    ('training_F_access',True),('F_outputs_read',1),('target_training',True),('V_training',True),('updates',19701),('epoch',21)])
def test_checkpoint_contract_rejects_other_sources_or_permissions(field,bad):
    model=m.compact_from_parent(parent_model(),d.ARMS[0]);model.initialization_report['seconds']=0.
    value=r.metadata(d.ARMS[0],20,19700,lineage(),model.initialization_report);r.validate_payload(value);value[field]=bad
    with pytest.raises(ValueError):r.validate_payload(value)


def test_wrong_objective_config_rejected_in_checkpoint():
    model=m.compact_from_parent(parent_model(),d.ARMS[0]);model.initialization_report['seconds']=0.
    value=r.metadata(d.ARMS[0],20,19700,lineage(),model.initialization_report)
    value['config']['balanced_complex_weight']=.04
    with pytest.raises(ValueError):r.validate_payload(value)


def test_alias_conflict_and_invalid_arm_before_io(monkeypatch):
    model=m.CompactSuccessor(d.ARMS[0]);state=copy.deepcopy(model.state_dict())
    key=next(k for k in state if k.startswith('prior.1.') and k.endswith('net.0.left.weight'));state[key]=state[key]+1
    with pytest.raises(ValueError,match='Conflicting'):r.validate_aliases(model,state)
    monkeypatch.setattr(r,'validate_parent_root',lambda *a:pytest.fail('No parent IO'))
    with pytest.raises(ValueError):r.initialize('factor625_stage','/runs','cpu')


def test_complex_metric_detects_phase_hidden_by_power_loss():
    torch.manual_seed(28);x=torch.randn(3,2,256)*.1;y=torch.stack((-x[:,1],x[:,0]),1).requires_grad_()
    assert m.spectral_error(y,x).max()<1e-10
    torch.testing.assert_close(m.balanced_complex_error(y,x),torch.full((3,),2.))
    m.balanced_complex_error(y,x).mean().backward()
    assert torch.isfinite(y.grad).all() and y.grad.abs().sum()>0


def test_zero_identity_and_energy_floor_finite_gradients():
    z=torch.zeros(2,2,256,requires_grad=True);loss=m.balanced_complex_error(z,z)
    assert torch.equal(loss,torch.zeros(2));loss.sum().backward();assert torch.isfinite(z.grad).all()
    y=(torch.randn(2,2,256)*.1).requires_grad_();clean=torch.zeros_like(y,requires_grad=True)
    value=m.balanced_complex_error(y,clean)
    torch.testing.assert_close(value,y.square().sum(1).mean(1)/1e-4)
    value.sum().backward();assert clean.grad is None and torch.isfinite(y.grad).all()


@pytest.mark.parametrize('shape',[(2,256),(2,3,256),(2,2,255)])
def test_complex_metric_requires_paired_native_shape(shape):
    with pytest.raises(ValueError):m.balanced_complex_error(torch.zeros(shape),torch.zeros(2,2,256))


def test_identity_forward_required_for_both_arms():
    x=torch.randn(2,2,256)*.1;outputs=[x+.01]*3;teacher=[x]*3
    for arm in d.ARMS:
        with pytest.raises(ValueError):m.objective(outputs,teacher,x,1,arm,None)
        identity=(x+.02).requires_grad_();loss,terms=m.objective(outputs,teacher,x,20,arm,identity)
        loss.sum().backward();assert identity.grad.abs().sum()>0


@pytest.mark.parametrize('bad',['root','role','arm','kind','plan'])
def test_invalid_target_reference_stays_before_arrays(monkeypatch,bad):
    args=SimpleNamespace(output=Path('/runs')/d.RUN/d.ARMS[0],kind=d.KIND);role='target'
    if bad=='root':args.output=Path('/runs/other')/d.ARMS[0]
    elif bad=='role':role='query'
    elif bad=='arm':args.output=args.output.parent/'factor625_stage'
    elif bad=='kind':args.kind='other'
    plan=copy.deepcopy(d.configuration());plan['updates']=2470
    monkeypatch.setattr(e,'read_json',lambda *a:plan)
    monkeypatch.setattr(e.np,'load',lambda *a,**k:pytest.fail('Target arrays must stay closed'))
    with pytest.raises(ValueError):e.reference(args,role)


def own_run(tmp_path,arm):
    root=tmp_path/d.RUN;root.mkdir()
    (root/'launch.json').write_text(json.dumps(dict(commit='method')))
    (root/'comparison_plan.json').write_text(json.dumps(d.configuration()))
    return SimpleNamespace(output=root,arm=arm,expected_commit='method')


@pytest.mark.parametrize('arm',d.ARMS)
def test_every_row_runs_full_formal_chain_without_source_gain_gate(tmp_path,monkeypatch,arm):
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
    monkeypatch.setattr(runner,'train',lambda *a:pytest.fail('Existing run cannot train'))
    with pytest.raises(ValueError,match='existing outputs'):runner.worker(args)
    assert list(row.iterdir())==[marker]


def test_training_failure_preserved_without_target_or_retry(tmp_path,monkeypatch):
    args=own_run(tmp_path,d.ARMS[0])
    def fail(*a):raise FloatingPointError('source technical failure')
    monkeypatch.setattr(runner,'train',fail);monkeypatch.setattr(runner,'predict',lambda *a:pytest.fail('No prediction'))
    with pytest.raises(FloatingPointError):runner.worker(args)
    assert (args.output/args.arm/'failure.json').is_file()


def test_truth_waits_for_complete_prediction_artifacts(monkeypatch):
    args=SimpleNamespace(output=Path('/runs')/d.RUN,arm=d.ARMS[0]);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(e,'reference',lambda *a:(Path('/formal'),e.np.arange(2),None,None))
    class Incomplete(Exception):pass
    def stop(*a):raise Incomplete()
    monkeypatch.setattr(e.base,'prediction_arrays',stop)
    monkeypatch.setattr(e.base,'target_baseline',lambda *a:pytest.fail('Truth must stay closed'))
    with pytest.raises(Incomplete):e.score(args,'target')


@pytest.mark.parametrize('loss,accepted',[(1344,True),(1345,False)])
def test_exact_scene_tolerance_boundary(loss,accepted):
    def summary(correct):return dict(role='target',count=168000,views={v:dict(count=168000,correct=correct,accuracy=100*correct/168000) for v in d.VIEWS})
    assert e.compact_acceptance(summary(140000-loss),summary(140000))['accepted']==accepted


def test_both_arms_identical_data_free_initialization_and_fixed_budget():
    parent=parent_model();before=copy.deepcopy(parent.state_dict())
    a=m.compact_from_parent(parent,d.ARMS[0]);b=m.compact_from_parent(parent,d.ARMS[1])
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert all(torch.equal(v,before[k]) for k,v in parent.state_dict().items())
    assert d.CLEAN_WEIGHT==dict.fromkeys(d.ARMS,.25) and d.COMPLEX_WEIGHT==.02 and d.MOMENT_WEIGHT==1.
    cfg=d.configuration();assert json.loads(json.dumps(cfg))==cfg
    assert d.EPOCHS==20 and d.UPDATES==19700 and d.BATCH==d.MICRO==64
    assert d.learning_rate(19699)==pytest.approx(2e-5) and d.teacher_weight(20)==pytest.approx(.1)
    assert cfg['F_weights_read']==cfg['F_outputs_read']==cfg['TX_labels_read']==0
    assert cfg['extra_clean_student_inputs']==dict.fromkeys(d.ARMS,1260000)
    assert cfg['extra_teacher_G_inputs']==cfg['extra_student_G_inputs']==0
    assert d.PREDECESSOR.endswith('m2-r28')


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_descriptor_matches_frozen_R29_measurement(dtype):
    from experiments.pamf_r46_source_moments.metrics import descriptor
    torch.manual_seed(73);x=torch.randn(4,2,256,dtype=dtype)*.1
    actual=m.moment_descriptor(x)
    torch.testing.assert_close(actual,descriptor(x)[0],rtol=0,atol=0)
    phase=torch.tensor(.73,dtype=dtype)
    rotated=torch.stack((x[:,0]*phase.cos()-x[:,1]*phase.sin(),x[:,0]*phase.sin()+x[:,1]*phase.cos()),1)
    torch.testing.assert_close(m.moment_descriptor(2*rotated+.13),actual,rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('level',[0.,.2,1e-5])
def test_descriptor_constant_and_floor_have_finite_gradients(level):
    x=torch.full((2,2,256),level,requires_grad=True)
    value=m.moment_descriptor(x)
    torch.testing.assert_close(value,torch.zeros_like(value),rtol=0,atol=1e-20)
    value.square().sum().backward();assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('arm',d.ARMS)
def test_selected_moment_target_and_detached_references(arm):
    torch.manual_seed(6)
    clean=(torch.randn(3,2,256)*.1).requires_grad_()
    teacher=[(torch.randn_like(clean)*.1).requires_grad_() for _ in range(3)]
    student=[(torch.randn_like(clean)*.1).requires_grad_() for _ in range(3)]
    identity=(clean.detach()+.001).requires_grad_()
    loss,terms=m.objective(student,teacher,clean,10,arm,identity)
    which='clean_moment_mse' if arm=='moment_clean' else 'teacher_moment_mse'
    torch.testing.assert_close(terms['moment_objective'],terms[which])
    reference=clean if arm=='moment_clean' else teacher[-1]
    expected=(m.moment_descriptor(student[-1])-m.moment_descriptor(reference.detach())).square().mean(1)
    torch.testing.assert_close(terms['moment_objective'],expected)
    terms['moment_objective'].mean().backward(retain_graph=True)
    assert torch.isfinite(student[-1].grad).all() and student[-1].grad.abs().sum()>0
    loss.mean().backward()
    assert clean.grad is None and all(v.grad is None for v in teacher)


@pytest.mark.parametrize('shape',[(2,256),(2,3,256),(2,2,255)])
def test_moment_descriptor_requires_native_IQ(shape):
    with pytest.raises(ValueError):m.moment_descriptor(torch.zeros(shape))


def test_moment_contract_cannot_silently_change_weight_or_teacher():
    model=m.compact_from_parent(parent_model(),d.ARMS[0]);model.initialization_report['seconds']=0.
    value=r.metadata(d.ARMS[0],20,19700,lineage(),model.initialization_report)
    value['config']['moment_weight']=.5
    with pytest.raises(ValueError):r.validate_payload(value)


def test_queue_waits_for_unlaunched_predecessor_instead_of_failing(tmp_path):
    previous=tmp_path/d.PREDECESSOR;previous.mkdir();(previous/'workers.json').write_text('{"workers":[]}')
    args=SimpleNamespace(output=tmp_path/d.RUN)
    assert runner.predecessor_ready(args,d.ARMS[0]) is False


def test_queue_requires_terminal_row_and_real_owned_handle_exit(tmp_path,monkeypatch):
    previous=tmp_path/d.PREDECESSOR;previous.mkdir();row=previous/'prior';row.mkdir()
    entry=dict(arm='prior',GPU=2,pid=999876543,cwd='/release',argv=['python','-m',d.PREDECESSOR_MODULE[d.PREDECESSOR],'--output',str(previous)])
    (previous/'workers.json').write_text(json.dumps(dict(workers=[entry])))
    args=SimpleNamespace(output=tmp_path/d.RUN)
    assert runner.predecessor_ready(args,d.ARMS[0]) is False
    (row/'complete.json').write_text('{}')
    assert runner.predecessor_ready(args,d.ARMS[0]) is True


def test_both_arms_read_no_additional_F_or_moment_data_at_inference():
    for arm in d.ARMS:
        model=m.CompactSuccessor(arm)
        assert sum(p.numel() for p in model.parameters())==278180
        assert not any('moment' in key for key in model.state_dict())
    assert d.configuration()['moment_reference']==dict(moment_clean='paired_clean',moment_teacher='original_350521_restored_IQ')
