import copy
import inspect
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from experiments.pamf_iq_physics_compact import design as d,runtime,evaluation,training,runner
from experiments.pamf_iq_physics_compact.models import JointCompactRestorer
from experiments.pamf_uncertainty_restore import design as ancestor
from experiments.pamf_restoration.data import write_json


def parent():
    return dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,action_run=ancestor.ACTION_RUN,
        action_commit=ancestor.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)


def initialization():
    return dict(mode='group_preserving_dense_channel_slices_shared_long',group_count=4,IQ_examples=0,TX_labels_read=0,
        F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,original_teacher_unchanged=True,
        exact_function_equivalence=False,context_trainable=True,seconds=0.)


@pytest.mark.parametrize('key,value',[('training_F_access',True),('F_outputs_read',1),('training_TX_labels_read',1),
    ('target_training',True),('V_training',True),('parent_R17_epoch',19),('joint_context_training',False),('updates',4939)])
def test_wrong_training_or_source_inheritance_contract_rejected(key,value):
    value_dict=runtime.metadata(d.ARMS[0],20,4940,parent(),initialization());value_dict[key]=value
    # E20 permits intermediate checkpoints, but final prediction rejects4939 separately.
    if key=='updates':value_dict[key]=5000
    with pytest.raises(ValueError):runtime.validate_payload(value_dict)


@pytest.mark.parametrize('arm',d.ARMS)
def test_real_checkpoint_roundtrip_preserves_dense_aliases_and_current_IQ(arm,tmp_path):
    model=JointCompactRestorer(arm).eval();model.initialization_report=initialization()
    path=tmp_path/'final.pt';runtime.save_checkpoint(path,model,None,20,4940,parent())
    fresh,value=runtime.load_checkpoint(path,'cpu')
    x=torch.randn(2,2,256)*.1
    with torch.no_grad():assert torch.equal(model(x),fresh(x))
    assert value['optimizer'] is None and value['training_F_access'] is False
    with pytest.raises(FileExistsError):runtime.save_checkpoint(path,model,None,20,4940,parent())


def test_shared_checkpoint_state_copies_cannot_silently_disagree():
    model=JointCompactRestorer(d.ARMS[0]);state={k:v.clone() for k,v in model.state_dict().items()}
    groups={}
    for name,p in model.named_parameters(remove_duplicate=False):groups.setdefault(id(p),[]).append(name)
    names=next(v for v in groups.values() if len(v)>1);state[names[1]].add_(1)
    with pytest.raises(ValueError):runtime.validate_aliases(model,state)


def test_training_path_has_no_trained_F_or_truth_access():
    source=inspect.getsource(training)
    assert 'load_reader' not in source and 'frozen_task_fidelity' not in source
    assert 'labels.npy' not in source and 'target_truth' not in source and 'source_cache' not in source
    assert 'parameters=list(model.parameters())' in source and 'loss.sum()/len(selected)' in source
    assert 'teacher.stage_outputs(r)' in source and 'objective(outputs,desired,x,epoch,context,target,native_target' in source
    assert 'epoch_arrays(cache,epoch)' in source and 'action_probe_vectors=2*(seen_l+seen_u)' in source


def complete_source(arm):
    return dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        parameters=d.PARAMETERS[arm],initialization_IQ_examples=0,teacher_parameter_updates=0,context_updates=4940,prior_updates=4940,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,training_F_access=False,head_fit_updates=0,V_reads=0,target_reads=0,paired_clean_optimized=True,simulated_physics_anchors=True,physics_target_examples=1260000,action_probe_vectors=2520000)


def test_source_diagnostic_completion_allows_formal_even_with_no_source_gain(tmp_path):
    row=tmp_path/d.ARMS[0];row.mkdir();value=complete_source(d.ARMS[0]);value['source_accuracy_gain_pp']=-5.
    write_json(row/'training_complete.json',value);write_json(row/'source_complete.json',dict(status='ANALYZED',qualified=False))
    evaluation.validate_source_row(row)


def test_formal_stays_closed_until_complete_source_diagnostic(tmp_path):
    row=tmp_path/d.ARMS[0];row.mkdir();write_json(row/'training_complete.json',complete_source(d.ARMS[0]))
    with pytest.raises(FileNotFoundError):evaluation.validate_source_row(row)


def test_worker_chains_training_source_and_all_formal_scoring(monkeypatch,tmp_path):
    args=SimpleNamespace(output=tmp_path,arm=d.ARMS[0]);calls=[]
    monkeypatch.setattr(runner,'train',lambda a:calls.append('train'))
    monkeypatch.setattr(runner,'predict',lambda a,role:calls.append('predict_'+role))
    monkeypatch.setattr(runner,'profile',lambda a:calls.append('source_profile'))
    monkeypatch.setattr(runner,'extract_context',lambda a:calls.append('source_context'))
    monkeypatch.setattr(runner,'score_subprocess',lambda a,role:calls.append('CPU_score_'+role))
    monkeypatch.setattr(runner.torch.cuda,'empty_cache',lambda:None)
    (tmp_path/d.ARMS[0]).mkdir();runner.worker(args)
    assert calls==['train','predict_source','source_context','source_profile','CPU_score_source','predict_target','CPU_score_target']
    assert (tmp_path/d.ARMS[0]/'complete.json').is_file()


def test_whole_frozen_F_is_used_only_in_noquery_smoke_after_IQ_loss():
    source=inspect.getsource(runner.smoke)
    assert source.index('loss.mean().backward()')<source.index('reader(actual)')
    assert 'frozen_task_fidelity' not in source and 'F_knowledge_in_loss=False' in source


def test_correct_row_and_immutable_root_in_subprocess_commands(tmp_path):
    args=SimpleNamespace(**{k:tmp_path/k for k in runner.PATH_ARGS},expected_commit='fixed')
    cmd=runner.command(args,'worker',d.ARMS[1])
    assert cmd[3]=='experiments.pamf_iq_physics_compact.runner' and cmd[6]==d.ARMS[1]
    assert d.GPUS=={d.ARMS[0]:0,d.ARMS[1]:1} and d.SCORING_RUN==d.RUN


@pytest.mark.parametrize('key,value',[('simulated_physics_anchors',False),('physics_target_examples',0),('action_probe_vectors',0)])
def test_missing_physics_training_contract_blocks_formal_access(tmp_path,key,value):
    row=tmp_path/d.ARMS[0];row.mkdir();trained=complete_source(d.ARMS[0]);trained[key]=value
    write_json(row/'training_complete.json',trained);write_json(row/'source_complete.json',dict(status='ANALYZED'))
    with pytest.raises(ValueError):evaluation.validate_source_row(row)


def test_omitted_source_context_cell_blocks_clean_oracle_access(monkeypatch,tmp_path):
    import numpy as np
    from experiments.pamf_iq_physics_compact import context_probe
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(context_probe,'source_reference',lambda *a:(np.arange(1800),None,None))
    def fail(*a):raise ValueError('Missing complete source context')
    monkeypatch.setattr(context_probe,'load_estimates',fail)
    monkeypatch.setattr(context_probe.np,'load',lambda *a,**k:pytest.fail('Oracle opened before allcontext cells'))
    args=SimpleNamespace(parent_action=tmp_path,source_cache=tmp_path,output=tmp_path,arm=d.ARMS[0])
    with pytest.raises(ValueError,match='Missing complete'):context_probe.score_context(args)


def test_physics_objective_config_change_cannot_be_loaded_as_same_checkpoint():
    value=runtime.metadata(d.ARMS[0],20,4940,parent(),initialization())
    value['config']['physics_weights']['action_probe_nmse']=.2
    with pytest.raises(ValueError):runtime.validate_payload(value)
