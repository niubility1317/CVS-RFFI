import json
from types import SimpleNamespace
import pytest
from experiments.pamf_r32_recovery_formal import design as d,runtime,runner
from experiments.pamf_restoration.data import write_json
from experiments.pamf_uncertainty_restore.evaluation import VIEWS


def failed_parent(tmp_path):
    root=tmp_path/d.PARENT;root.mkdir();(root/'logs').mkdir()
    write_json(root/'launch.json',dict(commit=d.PARENT_COMMIT,pid=999999999,cwd='/nonexistent-r32'))
    write_json(root/'workers.json',dict(rows=[]));write_json(root/'failure.json',dict(status='FAILED_TECHNICAL'))
    for candidate in d.ROWS:
        row=root/candidate;(row/d.KIND/'prediction').mkdir(parents=True);(row/'source_response').mkdir()
        (row/d.KIND/'final.pt').write_bytes(b'placeholder-not-loaded-by-unit-test')
        write_json(row/'training_complete.json',dict(status='SOURCE_TRAINED',candidate=candidate,epochs=20,updates=4940,
            L_exposures=126000,U_exposures=1134000,L_waveform_forwards=378000,U_waveform_forwards=3402000,
            TX_labels_read=0,F_outputs_read=0,F_weights_read=0,target_reads=0,frozen_context_updates=0,
            prior_initialization='scratch',inherited_prior_epochs=0,response_constraint_active=candidate=='observable_response'))
        write_json(row/d.KIND/'prediction/complete.json',dict(status='PREDICTIONS_COMPLETE',role='source',kind=d.KIND,count=1800,
            views=list(VIEWS),TX_truth_reads=0,state_updates=0,head_fit_updates=0))
        write_json(row/'source_response/complete.json',dict(status='SOURCE_RESPONSE_COMPLETE',run=d.PARENT,candidate=candidate,
            count=1800,cells=12,TX_truth_reads=0,F_access=0,target_reads=0,optimizer_updates=0,state_updates=0))
        (root/'logs'/('train-'+candidate+'.log')).write_text('ValueError: not enough values to unpack (expected 4, got 3)\n')
    return root


def replace(path,key,value):
    obj=json.loads(path.read_text());obj[key]=value;path.write_text(json.dumps(obj))


def test_only_saved_complete_training_and_predictions_are_recovered(tmp_path):
    p=failed_parent(tmp_path);matrix=runtime.parent_ready(SimpleNamespace(parent_run=p))
    assert matrix['rows']==list(d.ROWS) and not matrix['source_performance_gate']
    assert matrix['parent_state']=='SOURCE_SCORER_FAILED_TRAINING_SAVED'
    assert not (p/'complete.json').exists() # Never rewrite failed parent as successful.


@pytest.mark.parametrize('fault',['partial_training','training_F','partial_source','partial_response','wrong_failure','target_exists','wrong_parent'])
def test_recovery_rejects_wrong_or_incomplete_parent(fault,tmp_path):
    p=failed_parent(tmp_path);row=p/d.ROWS[0]
    if fault=='partial_training':replace(row/'training_complete.json','updates',4939)
    elif fault=='training_F':replace(row/'training_complete.json','F_weights_read',1)
    elif fault=='partial_source':replace(row/d.KIND/'prediction/complete.json','views',[])
    elif fault=='partial_response':replace(row/'source_response/complete.json','cells',11)
    elif fault=='wrong_failure':(p/'logs'/('train-'+d.ROWS[0]+'.log')).write_text('unknown technical failure')
    elif fault=='target_exists':(row/'target_confirmation').mkdir()
    else:replace(p/'launch.json','commit','wrong')
    with pytest.raises(ValueError):runtime.parent_ready(SimpleNamespace(parent_run=p))


def test_recovery_matrix_opens_formal_without_source_accuracy(tmp_path):
    matrix=runtime.parent_ready(SimpleNamespace(parent_run=failed_parent(tmp_path)))
    output=tmp_path/d.RUN;output.mkdir();(output/d.ROWS[0]).mkdir()
    write_json(output/'formal_matrix.json',matrix);write_json(output/'resolved_config.json',d.configuration('abc'))
    a=SimpleNamespace(output=output/d.ROWS[0],candidate=d.ROWS[0],kind=d.KIND,expected_commit='abc')
    runtime.audit_target(a)
    matrix['source_performance_gate']=True;(output/'formal_matrix.json').write_text(json.dumps(matrix))
    with pytest.raises(ValueError):runtime.audit_target(a)


def test_both_recovered_source_rows_proceed_to_formal_without_retraining(tmp_path,monkeypatch):
    p=failed_parent(tmp_path);output=tmp_path/'new';args=SimpleNamespace(parent_run=p,output=output,expected_commit='abc')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');calls=[]
    monkeypatch.setattr(runner,'smoke',lambda a:calls.append('actual-final-smoke'))
    monkeypatch.setattr(runner,'prepare_sources',lambda a:calls.append('all-source-arrays-before-truth'))
    monkeypatch.setattr(runner,'command',lambda a,mode,candidate:[mode,candidate])
    monkeypatch.setattr(runner.subprocess,'run',lambda cmd,**kw:calls.append(cmd))
    monkeypatch.setattr(runner,'queue',lambda a,rows:calls.append(('formal',list(rows))))
    runner.run(args)
    assert calls[1]=='all-source-arrays-before-truth'
    assert calls[2:4]==[['score_source',c] for c in d.ROWS]
    assert calls[-1]==('formal',list(d.ROWS))
    assert not json.loads((output/'complete.json').read_text())['source_performance_gate']


def test_source_truth_stays_closed_if_complete_validation_flag_absent(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    with pytest.raises(FileNotFoundError):runner.source_score(SimpleNamespace(output=tmp_path,candidate=d.ROWS[0]))


def test_no_new_optimizer_or_training_mode_is_added():
    config=d.configuration('abc')
    assert config['new_training_updates']==config['head_fit_updates']==config['state_updates']==0
    assert config['max_own_GPU_workers']==2 and config['max_own_workers_per_card']==1
