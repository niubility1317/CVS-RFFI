import copy,json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as Parent
from experiments.pamf_r46_reconstruction_tail import design as d,models as m,runtime as r,training as t,evaluation as e,runner
from experiments.pamf_uncertainty_restore import design as ancestor

torch.set_num_threads(2)


def lineage():
    return dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,action_run=ancestor.ACTION_RUN,
        action_commit=ancestor.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)


def initialized(arm):
    parent=Parent('prune_refine').eval().requires_grad_(False)
    with torch.no_grad():
        for part in (parent.frozen.coarse.temporal,parent.frozen.coarse.spectral,parent.frozen.estimator.core,*parent.prior):part.out.weight.normal_(std=.001)
    model=m.compact_from_parent(parent,arm);model.initialization_report['seconds']=0.
    return parent,model


@pytest.mark.parametrize('arm',d.ARMS)
@pytest.mark.parametrize('epoch',[0,20])
def test_full_compressed_graph_gradient_and_strict_portable_state(tmp_path,arm,epoch):
    parent,model=initialized(arm);received=torch.randn(2,2,256)*.1;clean=torch.randn(2,2,256,requires_grad=True)*.1
    clean=clean.detach().requires_grad_(True)
    with torch.no_grad():desired=parent.stage_outputs(received)[0]
    desired=[x.requires_grad_(True) for x in desired]
    loss,terms,focus=m.objective(model.stage_outputs(received)[0],desired,clean,1,arm);loss.mean().backward()
    assert clean.grad is None and all(x.grad is None for x in desired) and int(focus['selected'].sum())==1
    for part in (model.frozen.coarse,model.frozen.estimator,model.prior):
        gradients=[p.grad for p in part.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(x).all() for x in gradients) and sum(float(x.abs().sum()) for x in gradients)>0
    path=tmp_path/'final.pt';r.save_checkpoint(path,model,None,epoch,epoch*247,lineage())
    fresh,payload=r.load_checkpoint(path,'cpu')
    assert payload['optimizer'] is None and sum(p.numel() for p in fresh.parameters())==273765 and payload['extra_training_G_forwards']==0
    with torch.no_grad():assert torch.equal(model(received),fresh(received))
    assert all(p.grad is None for p in parent.parameters())
    with pytest.raises(FileExistsError):r.save_checkpoint(path,model,None,epoch,epoch*247,lineage())


@pytest.mark.parametrize('field,bad',[('training_F_access',True),('F_outputs_read',1),('training_TX_labels_read',1),('target_training',True),
    ('V_training',True),('parent_R46_commit','unknown'),('parent_R46_row','retain_refine'),('updates',4939),('epoch',19),
    ('external_student_checkpoint_reads',1),('new_calibration_IQ',1),('tail_fraction',.1),('tail_mix',.2),('extra_training_G_forwards',1)])
def test_lineage_full_budget_permissions_and_fixed_tail_formula(field,bad):
    _,model=initialized('tail_half');value=r.metadata('tail_half',20,4940,lineage(),model.initialization_report)
    r.validate_payload(value);value[field]=bad
    with pytest.raises(ValueError):r.validate_payload(value)


def trained(arm):
    return dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        parameters=273765,initialization_IQ_examples=0,teacher_parameter_updates=0,context_updates=4940,prior_updates=4940,
        cumulative_prior_epochs=100,cumulative_coarse_epochs=100,cumulative_action_epochs=100,TX_labels_read=0,V_reads=0,target_reads=0,
        training_F_access=False,F_weights_read=0,F_outputs_read=0,F_parameter_updates=0,head_fit_updates=0,paired_clean_optimized=True,
        focus_fraction=d.FRACTION[arm],focus_mix=.5,hard_exposures=int(1260000*d.FRACTION[arm]),extra_training_G_forwards=0,
        new_calibration_IQ=0,new_trainable_parameters=0,persistent_hardness_state=False,teacher_term_reweighted=False)


@pytest.mark.parametrize('field,bad',[('hard_exposures',1),('focus_fraction',.1),('focus_mix',1.),('teacher_term_reweighted',True),
    ('persistent_hardness_state',True),('F_outputs_read',1),('U_exposures',0),('extra_training_G_forwards',1)])
def test_actual_full_source_tail_budget_precedes_formal_access(monkeypatch,field,bad):
    value=trained('tail_half');value[field]=bad
    monkeypatch.setattr(e,'read_json',lambda p:value if p.name=='training_complete.json' else dict(status='ANALYZED'))
    with pytest.raises(ValueError):e.validate_source_row(Path('/runs')/d.RUN/'tail_half')


@pytest.mark.parametrize('arm',d.ARMS)
def test_all_versions_automatically_receive_full_formal_even_without_source_gain(tmp_path,monkeypatch,arm):
    args=SimpleNamespace(output=tmp_path,arm=arm);(tmp_path/arm).mkdir();seen=[]
    monkeypatch.setattr(runner,'train',lambda a:seen.append('train'))
    monkeypatch.setattr(runner,'predict',lambda a,role:seen.append('predict_'+role))
    monkeypatch.setattr(runner,'profile',lambda a:seen.append('profile'))
    monkeypatch.setattr(runner,'score_subprocess',lambda a,role:seen.append('score_'+role))
    monkeypatch.setattr(runner.torch.cuda,'empty_cache',lambda:None);runner.worker(args)
    assert seen==['train','predict_source','profile','score_source','predict_target','score_target'] and (tmp_path/arm/'complete.json').is_file()


def test_truth_closed_until_all_registered_predictions(monkeypatch):
    args=SimpleNamespace(output=Path('/runs')/d.RUN,arm='tail_half');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(e,'reference',lambda *a:(Path('/formal'),e.np.arange(2),None,None))
    monkeypatch.setattr(e,'validate_mean_control',lambda *a:True)
    class Incomplete(Exception):pass
    monkeypatch.setattr(e.base,'prediction_arrays',lambda *a:(_ for _ in ()).throw(Incomplete()))
    monkeypatch.setattr(e.base,'target_baseline',lambda *a:pytest.fail('No truth'))
    with pytest.raises(Incomplete):e.score(args,'target')


@pytest.mark.parametrize('damage',['queued_ema','live_final','failed','all_exited'])
def test_same_card_wait_includes_both_R66_final_and_queued_ema(tmp_path,monkeypatch,damage):
    root=tmp_path/d.base.RUN;root.mkdir();args=SimpleNamespace(output=tmp_path/d.RUN)
    def write(path,value):path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(value),encoding='utf-8')
    write(root/'launch.json',dict(commit=d.CONTROL['commit']))
    write(root/'shared_source_complete.json',dict(source_worker_exited=True))
    write(root/'workers.json',dict(workers=[dict(arm='final',GPU=3,pid=33),dict(arm='step_ema',GPU=3,pid=34)]))
    write(root/'final/complete.json',dict(status='ANALYZED'))
    if damage!='queued_ema':write(root/'step_ema/complete.json',dict(status='ANALYZED'))
    if damage=='failed':write(root/'failure.json',dict(status='FAILED_TECHNICAL'))
    fakeproc=tmp_path/'proc';fakeproc.mkdir()
    if damage=='live_final':
        proc=fakeproc/'33';proc.mkdir();(proc/'stat').write_text('33 (python) S 0',encoding='utf-8')
        (proc/'cmdline').write_bytes(('python\0experiments.pamf_r46_weight_mean.runner\0'+str(root)+'\0').encode())
    monkeypatch.setattr(t,'Path',lambda p:fakeproc if p=='/proc' else Path(p))
    if damage=='failed':
        with pytest.raises(ValueError):t.predecessor_exited(args,3)
    else:assert t.predecessor_exited(args,3)==(damage=='all_exited')


def test_runtime_plan_keeps_prototype_constants_but_no_false_observed_counters():
    cfg=d.runtime_configuration()
    assert cfg['parameters']=={a:273765 for a in d.ARMS} and cfg['GPU']=={'tail_half':3,'tail_quarter':2}
    assert cfg['fraction']==d.FRACTION and cfg['mix']==.5 and cfg['max_own_GPU_workers']==2
    assert all(k not in cfg for k in ('actual_source_epochs','actual_optimizer_updates','real_IQ_reads','GPU_launches'))
    assert cfg['predecessor_by_GPU']['3']==['final','step_ema'] and not cfg['target_feedback']


def test_source_exit_marker_before_worker_manifest_is_a_legal_wait(tmp_path):
    root=tmp_path/d.base.RUN;root.mkdir();args=SimpleNamespace(output=tmp_path/d.RUN)
    (root/'launch.json').write_text(json.dumps(dict(commit=d.CONTROL['commit'])),encoding='utf-8')
    (root/'shared_source_complete.json').write_text(json.dumps(dict(source_worker_exited=True)),encoding='utf-8')
    assert not (root/'workers.json').exists() and t.predecessor_exited(args,3) is False


def test_final_control_is_sufficient_without_whole_R66_completion(tmp_path):
    root=tmp_path/d.CONTROL['run'];row=root/'final';row.mkdir(parents=True);source=root/'source_path';source.mkdir()
    (root/'launch.json').write_text(json.dumps(dict(commit=d.CONTROL['commit'])),encoding='utf-8')
    (row/'complete.json').write_text(json.dumps(dict(status='ANALYZED',arm='final',role='target',all_predictions_before_truth=True,target_feedback=False)),encoding='utf-8')
    value=dict(status='SOURCE_TRAINED',epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,parameters=273765,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,training_F_access=False,V_reads=0,target_reads=0,shared_source_training_paths=1,training_counted_once_across_readouts=True)
    (source/'training_complete.json').write_text(json.dumps(value),encoding='utf-8')
    assert not (root/'complete.json').exists() and e.validate_mean_control(root)


def test_target_control_wait_happens_before_shared_score_lock(tmp_path,monkeypatch):
    import sys
    args=SimpleNamespace(output=tmp_path/d.RUN,arm='tail_quarter');(tmp_path/d.SCORING_RUN).mkdir()
    seen=[];ready=iter([False,True])
    def check(*a):seen.append('check_control');return next(ready)
    monkeypatch.setattr(runner,'validate_mean_control',check)
    monkeypatch.setattr(runner.time,'sleep',lambda *a:seen.append('wait'))
    monkeypatch.setitem(sys.modules,'fcntl',SimpleNamespace(LOCK_EX=1,flock=lambda *a:seen.append('lock')))
    monkeypatch.setattr(runner,'command',lambda *a:['score_target'])
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**k:seen.append('score'))
    runner.score_subprocess(args,'target')
    assert seen==['check_control','wait','check_control','lock','score']
