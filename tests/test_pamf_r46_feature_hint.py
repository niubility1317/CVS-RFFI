import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as HumanParent
from experiments.pamf_r46_feature_hint import design as d
from experiments.pamf_r46_feature_hint.models import CompactSuccessor,compact_from_parent,objective
from experiments.pamf_r46_feature_hint.hints import FeatureHints
from experiments.pamf_r46_feature_hint import runtime,evaluation,queueing
from experiments.pamf_uncertainty_restore import design as ancestor

torch.set_num_threads(2)

def pair(arm='all_hint'):
    torch.manual_seed(60)
    teacher=HumanParent('prune_refine').eval().requires_grad_(False)
    child=compact_from_parent(teacher,arm)
    maps={k:[torch.tensor(v,dtype=torch.int64) for v in values] for k,values in child.initialization_report['channel_maps'].items()}
    return teacher,child,maps

def forward_pair(hints,teacher,child,x):
    hints.begin()
    with torch.no_grad():wanted=teacher.stage_outputs(x)[0]
    hints.teacher_done();actual=child.stage_outputs(x)[0]
    return wanted,actual,hints.finish()

@pytest.mark.parametrize('arm',d.ARMS)
def test_existing_pruned_graph_and_joint_gradient_without_teacher_update(arm):
    teacher,child,maps=pair(arm);x=torch.randn(2,2,256)*.1
    original=copy.deepcopy(teacher.state_dict());keys=tuple(child.state_dict())
    with FeatureHints(teacher,child,maps,arm) as hints:
        assert len(hints.links)==(84 if arm=='context_hint' else 168)
        wanted,actual,features=forward_pair(hints,teacher,child,x)
        loss,terms=objective(actual,wanted,x,1,arm,features)
        assert loss.shape==(2,) and torch.isfinite(loss).all()
        assert terms['teacher_context_feature_nmse'].min()>0
        assert bool(terms['teacher_prior_feature_nmse'].count_nonzero())==(arm=='all_hint')
        loss.mean().backward()
        for module in (child.frozen.coarse,child.frozen.estimator,*child.prior):
            assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters())
        assert all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(v,original[k]) for k,v in teacher.state_dict().items())
    assert sum(p.numel() for p in child.parameters())==273765 and tuple(child.state_dict())==keys
    assert all(not m._forward_hooks for model in (teacher,child) for m in model.modules())
    assert child.initialization_report['IQ_examples']==child.initialization_report['new_calibration_IQ']==0

@pytest.mark.parametrize('arm',d.ARMS)
def test_per_IQ_losses_are_batch_independent_and_hooks_do_not_change_outputs(arm):
    teacher,child,maps=pair(arm);child.eval();x=torch.randn(2,2,256)*.1
    with torch.no_grad():plain=child(x)
    with FeatureHints(teacher,child,maps,arm) as hints:
        with torch.no_grad():
            _,actual,together=forward_pair(hints,teacher,child,x)
            separate=[forward_pair(hints,teacher,child,v[None])[2] for v in x]
        assert torch.equal(plain,actual[-1])
        for key in together:
            assert torch.allclose(together[key],torch.cat([v[key] for v in separate]),rtol=1e-5,atol=1e-7)

@pytest.mark.parametrize('fault',['student_first','repeat_teacher','missing_teacher','missing_student','unfinished_pair','closed'])
def test_incomplete_or_out_of_order_pairs_reject_and_close_hooks(fault):
    teacher,child,maps=pair();x=torch.randn(1,2,256)*.1
    hints=FeatureHints(teacher,child,maps,'all_hint')
    try:
        with pytest.raises(ValueError):
            hints.begin()
            if fault=='student_first':child(x)
            elif fault=='repeat_teacher':teacher(x);teacher(x)
            elif fault=='missing_teacher':hints.teacher_done()
            elif fault=='unfinished_pair':hints.begin()
            elif fault=='closed':hints.close();hints.finish()
            else:teacher(x);hints.teacher_done();hints.finish()
    finally:hints.close()
    assert all(not m._forward_hooks for model in (teacher,child) for m in model.modules())

@pytest.mark.parametrize('fault',['unknown','train_mode','trainable_teacher'])
def test_bad_scope_or_teacher_contract_reject_before_hook_registration(fault):
    teacher,child,maps=pair();arm='all_hint'
    if fault=='unknown':arm='target_selected'
    elif fault=='train_mode':teacher.train()
    else:next(teacher.parameters()).requires_grad_(True)
    with pytest.raises(ValueError):FeatureHints(teacher,child,maps,arm)
    assert all(not m._forward_hooks for model in (teacher,child) for m in model.modules())

def test_fixed_weight_closes_and_prior_addition_keeps_same_context_coefficient():
    x=torch.randn(2,2,256)*.1;outputs=[x+.01,x+.02,x+.03];wanted=[x,x,x]
    context=torch.tensor([2.,3.],requires_grad=True);prior=torch.tensor([5.,7.],requires_grad=True)
    zero=torch.zeros_like(context)
    a,ta=objective(outputs,wanted,x,10,'context_hint',dict(teacher_context_feature_nmse=context,teacher_prior_feature_nmse=zero))
    b,tb=objective(outputs,wanted,x,10,'all_hint',dict(teacher_context_feature_nmse=context,teacher_prior_feature_nmse=prior))
    assert torch.allclose(b-a,.02*prior)
    assert torch.allclose(a,ta['iq_nmse']+.05*ta['spectral_error']+d.teacher_weight(10,'context_hint')*ta['teacher_stage_nmse']+.02*context)
    b.sum().backward();assert torch.allclose(context.grad,torch.full_like(context,.02)) and torch.allclose(prior.grad,torch.full_like(prior,.02))

@pytest.mark.parametrize('fault',['scope','keys','shape','prior_in_context'])
def test_unregistered_loss_components_reject(fault):
    x=torch.zeros(2,2,256);features=dict(teacher_context_feature_nmse=torch.ones(2),teacher_prior_feature_nmse=torch.zeros(2));arm='context_hint'
    if fault=='scope':arm='target_selected'
    elif fault=='keys':features['F_embedding']=torch.ones(2)
    elif fault=='shape':features['teacher_context_feature_nmse']=torch.ones(1)
    else:features['teacher_prior_feature_nmse']=torch.ones(2)
    with pytest.raises(ValueError):objective([x]*3,[x]*3,x,1,arm,features)

def test_matching_widths_state_keys_and_fixed_source_only_plan():
    config=d.configuration();a=CompactSuccessor('context_hint');b=CompactSuccessor('all_hint')
    assert tuple(a.state_dict())==tuple(b.state_dict()) and sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())==273765
    assert config['hint']['context_modules']==config['hint']['prior_modules']==84
    assert config['hint']['new_parameters']==config['hint']['extra_teacher_or_student_G_forwards']==config['hint']['extra_calibration_IQ']==0
    assert config['training']['same_context_weight_both_rows'] and config['training']['source_clean_pairs']
    assert all(config['training'][k]==0 for k in ('TX_labels_read','F_weights_read','F_outputs_read','V_reads','target_reads'))
    assert config['evaluation']['target168000x7'] and config['evaluation']['all_rows_regardless_source']
    assert not config['prototype_only'] and config['full_pipeline_implemented']

def payload(arm='all_hint'):
    teacher,child,maps=pair(arm);child.initialization_report['seconds']=0.
    ancestry=dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,action_run=ancestor.ACTION_RUN,
        action_commit=ancestor.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)
    return teacher,child,maps,ancestry,runtime.metadata(arm,20,4940,ancestry,child.initialization_report)

@pytest.mark.parametrize('fault',['F_access','F_weight','target','V','hint_weight','hint_scope','new_parameters','calibration','parent_contract'])
def test_wrong_payload_contract_cannot_reach_model_state(fault):
    _,_,_,_,v=payload()
    if fault=='F_access':v['training_F_access']=True
    elif fault=='F_weight':v['F_weights_read']=1
    elif fault=='target':v['target_training']=True
    elif fault=='V':v['V_training']=True
    elif fault=='hint_weight':v['initialization']['feature_hint_weight']=.1
    elif fault=='hint_scope':v['initialization']['feature_hint_scope']='context_hint'
    elif fault=='new_parameters':v['initialization']['new_trainable_parameters']=1
    elif fault=='calibration':v['initialization']['new_calibration_IQ']=128
    else:v['parent']['data_contract']='target_selected'
    with pytest.raises(ValueError):runtime.validate_payload(v)

def test_strict_weight_reload_requires_removed_training_hooks(tmp_path):
    teacher,child,maps,ancestry,v=payload();path=tmp_path/'final.pt'
    with FeatureHints(teacher,child,maps,'all_hint'):
        with pytest.raises(ValueError):runtime.save_checkpoint(path,child,None,20,4940,ancestry)
        assert not path.exists()
    runtime.save_checkpoint(path,child,None,20,4940,ancestry);fresh,meta=runtime.load_checkpoint(path,'cpu')
    x=torch.randn(1,2,256)*.1;child.eval()
    with torch.no_grad():assert torch.equal(child(x),fresh(x))
    assert meta['optimizer'] is None and meta['cumulative_component_epochs']==100 and meta['initialization']['IQ_examples']==0
    assert all(not m._forward_hooks for m in fresh.modules())

@pytest.mark.parametrize('bad',['wrong_parent_root','unknown_arm'])
def test_parent_permissions_reject_before_any_checkpoint_IO(monkeypatch,bad):
    monkeypatch.setattr(runtime,'read_json',lambda *a:pytest.fail('No IO before root/arm validation'))
    with pytest.raises(ValueError):runtime.initialize('target_selected' if bad=='unknown_arm' else 'all_hint',Path('/wrong_parent'),'cpu')

@pytest.mark.parametrize('field,bad',[('epochs',19),('updates',4939),('F_outputs_read',1),('TX_labels_read',1),('target_reads',1),
    ('feature_hint_weight',.1),('train_only_hooks_removed',False),('extra_teacher_G_forwards',1)])
def test_incomplete_source_training_blocks_formal_access(monkeypatch,field,bad):
    v=dict(status='SOURCE_TRAINED',arm='all_hint',epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,parameters=273765,
        initialization_IQ_examples=0,teacher_parameter_updates=0,context_updates=4940,prior_updates=4940,cumulative_prior_epochs=100,
        cumulative_coarse_epochs=100,cumulative_action_epochs=100,TX_labels_read=0,V_reads=0,target_reads=0,training_F_access=False,
        F_weights_read=0,F_outputs_read=0,F_parameter_updates=0,head_fit_updates=0,paired_clean_optimized=True,
        feature_hint_scope='all_hint',feature_hint_weight=.02,context_hint_modules=84,prior_hint_modules=84,
        extra_teacher_G_forwards=0,extra_student_G_forwards=0,new_calibration_IQ=0,new_trainable_parameters=0,train_only_hooks_removed=True)
    v[field]=bad
    monkeypatch.setattr(evaluation,'read_json',lambda p:v if p.name=='training_complete.json' else dict(status='ANALYZED'))
    with pytest.raises(ValueError):evaluation.validate_source_row(Path('/runs')/d.RUN/'all_hint')

@pytest.mark.parametrize('state',['unfinished','live','zombie','exited','pid_reused','wrong_commit','missing_owner','failed'])
def test_actual_predecessor_PID_exit_not_marker_alone_controls_new_GPU_slot(tmp_path,state):
    args=SimpleNamespace(output=tmp_path/'runs'/d.RUN,arm='all_hint');root=args.output.parent/d.PREDECESSOR_RUN
    row=root/'identity_control';row.mkdir(parents=True);proc_root=tmp_path/'proc';proc_root.mkdir()
    def write(p,v):p.write_text(json.dumps(v),encoding='utf-8')
    if state!='unfinished':write(row/'complete.json',dict(status='ANALYZED'))
    write(root/'launch.json',dict(commit='wrong' if state=='wrong_commit' else d.PREDECESSOR_COMMIT))
    write(root/'workers.json',dict(workers=[] if state=='missing_owner' else [dict(arm='identity_control',GPU=2,pid=321)]))
    if state=='failed':write(root/'failure.json',dict(status='FAILED_TECHNICAL'))
    if state in ('live','zombie','pid_reused'):
        proc=proc_root/'321';proc.mkdir();(proc/'stat').write_text('321 (python) '+('Z' if state=='zombie' else 'R')+' 1 1',encoding='utf-8')
        argv=['python','-m',d.PREDECESSOR_MODULE,'worker','--output','../../runs/'+d.PREDECESSOR_RUN] if state!='pid_reused' else ['other_process']
        (proc/'cmdline').write_bytes('\0'.join(argv).encode('utf-8'))
    if state in ('wrong_commit','missing_owner','failed'):
        with pytest.raises(ValueError):queueing.predecessor_closed(args,proc_root=proc_root)
    else:assert queueing.predecessor_closed(args,proc_root=proc_root)==(state in ('zombie','exited','pid_reused'))
