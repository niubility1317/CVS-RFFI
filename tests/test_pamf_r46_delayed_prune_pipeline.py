import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as Parent
from experiments.pamf_r46_delayed_prune import design as d,models as m,runtime as r,evaluation as e,runner
from experiments.pamf_uncertainty_restore import design as ancestor

torch.set_num_threads(2)


def lineage():
    return dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,
        action_run=ancestor.ACTION_RUN,action_commit=ancestor.ACTION_COMMIT,
        data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)


def models(arm=d.ARMS[0]):
    torch.manual_seed(65);parent=Parent('prune_refine').eval().requires_grad_(False)
    dense=m.dense_from_parent(parent,arm);dense.initialization_report['seconds']=0.
    return parent,dense,m.transition(dense,d.WARMUP_EPOCHS[arm])


@pytest.mark.parametrize('arm',d.ARMS)
@pytest.mark.parametrize('phase',['dense','compact'])
def test_portable_phase_smoke_and_final_with_gradient_and_detached_truth(tmp_path,arm,phase):
    parent,dense,compact=models(arm);model=dense if phase=='dense' else compact
    epoch=d.WARMUP_EPOCHS[arm] if phase=='dense' else 20
    clean=(torch.randn(2,2,256)*.1).requires_grad_(True);received=clean.detach()+torch.randn_like(clean)*.005
    with torch.no_grad():desired=parent.stage_outputs(received)[0]
    desired=[v.requires_grad_(True) for v in desired]
    loss,terms=m.objective(model.stage_outputs(received)[0],desired,clean,epoch)
    assert torch.allclose(loss,terms['iq_nmse']+.05*terms['spectral_error']+d.teacher_weight(epoch)*terms['teacher_stage_nmse'])
    loss.mean().backward()
    assert clean.grad is None and all(v.grad is None for v in desired) and all(p.grad is None for p in parent.parameters())
    for part in (model.frozen.coarse,model.frozen.estimator,model.prior):
        grads=[p.grad for p in part.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(v).all() for v in grads) and sum(float(v.abs().sum()) for v in grads)>0
    for checkpoint_epoch in (0,epoch):
        path=tmp_path/(str(checkpoint_epoch)+'.pt');r.save_checkpoint(path,model,None,checkpoint_epoch,checkpoint_epoch*247,lineage())
        fresh,payload=r.load_checkpoint(path,'cpu')
        assert payload['structural_smoke']==(checkpoint_epoch==0 and phase=='compact')
        with torch.no_grad():assert torch.equal(model(received),fresh(received))
        assert payload['optimizer'] is None
        with pytest.raises(FileExistsError):r.save_checkpoint(path,model,None,checkpoint_epoch,checkpoint_epoch*247,lineage())


@pytest.mark.parametrize('field,bad',[('training_F_access',True),('F_outputs_read',1),('target_training',True),('V_training',True),
    ('parent_R46_commit','unknown'),('parent_R46_row','retain_refine'),('updates',4939),('epoch',21),('new_calibration_IQ',256),('external_student_checkpoint_reads',1)])
def test_checkpoint_rejects_permissions_ancestry_and_incomplete_epoch(field,bad):
    _,_,model=models();value=r.metadata(d.ARMS[0],20,4940,lineage(),model.initialization_report)
    r.validate_payload(value);value[field]=bad
    with pytest.raises(ValueError):r.validate_payload(value)


@pytest.mark.parametrize('field,bad',[('phase','dense'),('optimizer_resets',0),('global_schedule_restarted',True),
    ('channel_maps_unchanged',False),('transition_completed_epoch',10),('teacher','warmup_student')])
def test_final_checkpoint_rejects_wrong_transition(field,bad):
    _,_,model=models();value=r.metadata(d.ARMS[0],20,4940,lineage(),model.initialization_report)
    value['initialization'][field]=bad
    with pytest.raises(ValueError):r.validate_payload(value)


def test_alias_conflict_and_actual_group_norm_partition():
    _,_,model=models();state=copy.deepcopy(model.state_dict())
    key=next(k for k in state if k.startswith('prior.1.') and k.endswith('net.0.weight'));state[key]=state[key]+1
    with pytest.raises(ValueError,match='Conflicting'):r.validate_aliases(model,state)
    model.frozen.coarse.spectral.encoder.blocks[0][0].net[1].num_groups=1
    with pytest.raises(ValueError,match='GN4'):r.validate_aliases(model)


def source_records(arm):
    trained=dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        parameters=273765,initialization_IQ_examples=0,teacher_parameter_updates=0,context_updates=4940,prior_updates=4940,
        cumulative_prior_epochs=100,cumulative_coarse_epochs=100,cumulative_action_epochs=100,TX_labels_read=0,V_reads=0,
        target_reads=0,training_F_access=False,F_weights_read=0,F_outputs_read=0,F_parameter_updates=0,head_fit_updates=0,
        paired_clean_optimized=True,dense_epochs=d.WARMUP_EPOCHS[arm],compact_epochs=20-d.WARMUP_EPOCHS[arm],optimizer_resets=1,
        channel_maps_unchanged=True,global_schedule_restarted=False,dense_IQ_examples=63000*d.WARMUP_EPOCHS[arm],
        compact_IQ_examples=63000*(20-d.WARMUP_EPOCHS[arm]),new_calibration_IQ=0,new_trainable_parameters=0,external_student_checkpoint_reads=0)
    transition=dict(status='SOURCE_TRANSITION_COMPLETE',completed_epoch=d.WARMUP_EPOCHS[arm],completed_updates=d.WARMUP_EPOCHS[arm]*247,
        L_exposures=6300*d.WARMUP_EPOCHS[arm],U_exposures=56700*d.WARMUP_EPOCHS[arm],optimizer_resets=1,channel_maps_unchanged=True,
        teacher_replaced=False,external_student_checkpoint_reads=0,new_calibration_IQ=0,global_schedule_restarted=False)
    return trained,transition


@pytest.mark.parametrize('field,bad',[('epochs',19),('updates',4939),('TX_labels_read',1),('F_outputs_read',1),
    ('target_reads',1),('V_reads',1),('dense_epochs',6),('compact_epochs',20),('optimizer_resets',0),('channel_maps_unchanged',False)])
def test_training_budget_rejected_before_formal_input(monkeypatch,field,bad):
    trained,transition=source_records(d.ARMS[0]);trained[field]=bad
    monkeypatch.setattr(e,'read_json',lambda p:trained if p.name=='training_complete.json' else transition if p.name=='transition.json' else dict(status='ANALYZED'))
    with pytest.raises(ValueError):e.validate_source_row(Path('/runs')/d.ARMS[0])


@pytest.mark.parametrize('field,bad',[('completed_updates',100),('L_exposures',0),('U_exposures',0),('teacher_replaced',True),('global_schedule_restarted',True)])
def test_transition_actual_exposure_required_before_formal(monkeypatch,field,bad):
    trained,transition=source_records(d.ARMS[0]);transition[field]=bad
    monkeypatch.setattr(e,'read_json',lambda p:trained if p.name=='training_complete.json' else transition if p.name=='transition.json' else dict(status='ANALYZED'))
    with pytest.raises(ValueError):e.validate_source_row(Path('/runs')/d.ARMS[0])


@pytest.mark.parametrize('arm',d.ARMS)
def test_each_row_automatically_completes_formal_even_without_source_gain(tmp_path,monkeypatch,arm):
    args=SimpleNamespace(output=tmp_path,arm=arm);(tmp_path/arm).mkdir();seen=[]
    monkeypatch.setattr(runner,'train',lambda *a:seen.append('train'))
    monkeypatch.setattr(runner,'predict',lambda a,role:seen.append('predict_'+role))
    monkeypatch.setattr(runner,'profile',lambda *a:seen.append('profile'))
    monkeypatch.setattr(runner,'score_subprocess',lambda a,role:seen.append('score_'+role))
    monkeypatch.setattr(runner.torch.cuda,'empty_cache',lambda:None);runner.worker(args)
    assert seen==['train','predict_source','profile','score_source','predict_target','score_target']
    assert (tmp_path/arm/'complete.json').is_file()


def test_truth_is_closed_before_all_predictions(monkeypatch):
    args=SimpleNamespace(output=Path('/runs')/d.RUN,arm=d.ARMS[0]);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(e,'reference',lambda *a:(Path('/formal'),e.np.arange(2),None,None))
    class Incomplete(Exception):pass
    monkeypatch.setattr(e.base,'prediction_arrays',lambda *a:(_ for _ in ()).throw(Incomplete()))
    monkeypatch.setattr(e.base,'target_baseline',lambda *a:pytest.fail('No truth'))
    with pytest.raises(Incomplete):e.score(args,'target')


def test_serializable_plan_and_no_early_target_access():
    assert json.loads(json.dumps(d.configuration()))==d.configuration()
    assert len(d.VIEWS)==7 and d.configuration()['max_own_GPU_workers']==2
    assert sorted(d.GPUS.values())==[2,3]
    assert d.configuration()['compact_parent_run']==d.COMPACT_PARENT_RUN
    assert d.configuration()['compact_parent_commit']=='bbb8daf817de050b958d217b70d4c33f4aefe341'
    assert d.teacher_weight(1)==1 and d.teacher_weight(20)==pytest.approx(.1)
    assert d.learning_rate(4939)==pytest.approx(2e-5)


def test_invalid_row_rejected_before_parent_IO(monkeypatch):
    monkeypatch.setattr(r,'validate_parent_root',lambda *a:pytest.fail('No parent IO'))
    with pytest.raises(ValueError):r.initialize('R64','/runs','cpu')


@pytest.mark.parametrize('kind',['live','zombie','missing','reused'])
def test_waits_for_actual_owned_predecessor_handle(tmp_path,kind):
    from experiments.pamf_r46_delayed_prune.training import handle_exited
    proc=tmp_path/'123';root=Path('/runs/R64')
    if kind!='missing':
        proc.mkdir();(proc/'stat').write_text('123 (python) '+('Z' if kind=='zombie' else 'S')+' 0 0',encoding='utf-8')
        argv=['python','-m','experiments.pamf_r46_group_fold.runner','worker','--output',str(root)] if kind!='reused' else ['unrelated','other-owner']
        (proc/'cmdline').write_bytes(chr(0).join(argv).encode('utf-8'))
    assert handle_exited(123,root,tmp_path)==(kind!='live')
