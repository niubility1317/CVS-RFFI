import json
from types import SimpleNamespace
import pytest
from experiments.pamf_r32_all_formal import design as d,runtime,runner
from experiments.pamf_restoration.data import write_json
from experiments.pamf_fcrossrx_readout.frozen import REFERENCE_RUN
from experiments.pamf_uncertainty_restore.evaluation import VIEWS


def replace_json(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')


def training(candidate):
    return dict(status='SOURCE_TRAINED',candidate=candidate,epochs=20,updates=4940,
        L_exposures=126000,U_exposures=1134000,L_waveform_forwards=378000,U_waveform_forwards=3402000,
        TX_labels_read=0,F_outputs_read=0,F_weights_read=0,target_reads=0,frozen_context_updates=0,
        prior_initialization='scratch',inherited_prior_epochs=0,response_constraint_active=candidate=='observable_response')


def parent(tmp_path):
    p=tmp_path/d.PARENT;p.mkdir()
    write_json(p/'launch.json',dict(commit=d.PARENT_COMMIT,pid=999999999,cwd='/nonexistent-r32'))
    write_json(p/'workers.json',dict(rows=[]))
    write_json(p/'complete.json',dict(status='ANALYZED',source_rows=list(d.ROWS),source_qualified=False,target_rows=[]))
    # Both historical source failures must still enter the new formal matrix.
    write_json(p/'source_decision.json',dict(qualified=False,target_rows=[]))
    for c in d.ROWS:
        row=p/c;(row/d.KIND).mkdir(parents=True)
        (row/d.KIND/'final.pt').write_bytes(b'placeholder-not-loaded-in-this-unit-test')
        write_json(row/'training_complete.json',training(c))
        write_json(row/'complete.json',dict(status='ANALYZED',candidate=c,role='source',all_predictions_before_truth=True,training_TX_F_access=False))
    return p


def row_args(tmp_path):
    output=tmp_path/d.RUN;output.mkdir();candidate=d.ROWS[0];(output/candidate).mkdir()
    write_json(output/'resolved_config.json',d.configuration('abc'))
    matrix=dict(status='FORMAL_MATRIX_FROZEN',rows=list(d.ROWS),kind=d.KIND,parent_run=d.PARENT,parent_commit=d.PARENT_COMMIT,
        policy=d.POLICY,source_performance_gate=False,new_training_updates=0,target_feedback=False,parent_decisions_modified=False)
    write_json(output/'formal_matrix.json',matrix)
    return SimpleNamespace(output=output/candidate,candidate=candidate,kind=d.KIND,expected_commit='abc',parent_run=tmp_path/d.PARENT)


def test_negative_source_results_do_not_remove_either_formal_row(tmp_path):
    matrix=runtime.parent_ready(SimpleNamespace(parent_run=parent(tmp_path)))
    assert matrix['rows']==list(d.ROWS) and matrix['source_performance_gate'] is False
    assert matrix['status']=='FORMAL_MATRIX_FROZEN' and 'qualified' not in matrix


@pytest.mark.parametrize('key,value',[('epochs',19),('updates',4939),('TX_labels_read',1),('F_weights_read',1),
    ('target_reads',1),('U_exposures',0),('inherited_prior_epochs',20)])
def test_illegal_or_incomplete_training_remains_blocked(key,value):
    t=training(d.ROWS[0]);t[key]=value
    with pytest.raises(ValueError):runtime.validate_training(t,d.ROWS[0])


def test_wrong_response_row_contract_rejected():
    t=training('observable_response');t['response_constraint_active']=False
    with pytest.raises(ValueError):runtime.validate_training(t,'observable_response')


@pytest.mark.parametrize('mutation',['unfinished','wrong_code','missing_row'])
def test_parent_must_finish_without_restarting_or_duplicate_testing(tmp_path,mutation):
    p=parent(tmp_path)
    if mutation=='unfinished':replace_json(p/'complete.json',dict(status='RUNNING',source_rows=list(d.ROWS)))
    elif mutation=='wrong_code':replace_json(p/'launch.json',dict(commit='wrong'))
    else:replace_json(p/'complete.json',dict(status='ANALYZED',source_rows=[d.ROWS[0]]))
    with pytest.raises(ValueError):runtime.parent_ready(SimpleNamespace(parent_run=p))


def test_explicit_fixed_matrix_allows_both_rows_without_source_decision(tmp_path):
    a=row_args(tmp_path)
    runtime.audit_target(a)
    a.output=a.output.parent/d.ROWS[1];a.candidate=d.ROWS[1]
    runtime.audit_target(a)


@pytest.mark.parametrize('mutation',['removed_row','qualified_forgery','gate','policy','wrong_candidate'])
def test_incomplete_or_forged_formal_permission_rejected(tmp_path,mutation):
    a=row_args(tmp_path);path=a.output.parent/'formal_matrix.json';matrix=runtime.read_json(path)
    if mutation=='removed_row':matrix['rows'].pop()
    elif mutation=='qualified_forgery':matrix=dict(status='SOURCE_DECISION_FROZEN',qualified=True,target_candidates=[d.KIND])
    elif mutation=='gate':matrix['source_performance_gate']=True
    elif mutation=='policy':matrix['policy']='automatic source selection'
    else:a.candidate='unregistered'
    replace_json(path,matrix)
    with pytest.raises(ValueError):runtime.audit_target(a)


def test_strict_checkpoint_row_and_budget_match(monkeypatch,tmp_path):
    path=tmp_path/d.ROWS[0]/d.KIND/'final.pt'
    for value in [dict(candidate=d.ROWS[1],epoch=20,updates=4940,kind=d.KIND),
                  dict(candidate=d.ROWS[0],epoch=19,updates=4693,kind=d.KIND)]:
        monkeypatch.setattr(runtime,'original_load',lambda *args:(None,value))
        with pytest.raises(ValueError):runtime.load_final(path,'cpu')


def test_partial_existing_predictions_cannot_trigger_duplicate_launch(tmp_path):
    a=row_args(tmp_path);(a.parent_run/a.candidate/'target_confirmation'/d.KIND).mkdir(parents=True)
    with pytest.raises(FileNotFoundError):runtime.reuse_complete(a)


def test_existing_complete_result_reused_even_if_accuracy_is_zero(tmp_path,monkeypatch):
    a=row_args(tmp_path);target=a.parent_run/a.candidate/'target_confirmation'/d.KIND;target.mkdir(parents=True)
    write_json(target.parent.parent/'target_complete.json',dict(status='ANALYZED',candidate=a.candidate,role='target',
        all_predictions_before_truth=True,training_TX_F_access=False))
    summary=dict(status='ANALYZED',kind=d.KIND,role='target',count=168000,F_run=REFERENCE_RUN,F_seed=2026092701,
        original_default_head=True,all_predictions_before_truth=True,training_TX_labels_read=0,training_F_access=False,
        head_fit_updates=0,target_access=True,views={v:{} for v in VIEWS},
        records=[dict(view=v,dimension='overall',readout='restored',accuracy=0.) for v in VIEWS])
    write_json(target/'summary.json',summary)
    calls=[]
    monkeypatch.setattr(runtime,'reference',lambda *args:(None,[],None,None))
    monkeypatch.setattr(runtime,'prediction_arrays',lambda *args:calls.append(args))
    result=runtime.reuse_complete(a)
    assert len(calls)==1 and result['status']=='REUSED_COMPLETE' and result['selection_uses_accuracy'] is False
    assert result['model_forwards']==0 and result['truth_reopened'] is False


def test_two_source_failures_both_reach_formal_queue(tmp_path,monkeypatch):
    p=parent(tmp_path);output=tmp_path/'new'
    args=SimpleNamespace(parent_run=p,output=output,expected_commit='abc')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(runner,'smoke',lambda a:None)
    monkeypatch.setattr(runner,'reuse_complete',lambda a:None)
    queued=[];monkeypatch.setattr(runner,'queue',lambda a,rows:queued.extend(rows))
    runner.run(args)
    assert queued==list(d.ROWS)
    result=runtime.read_json(output/'complete.json')
    assert result['target_rows']==list(d.ROWS) and result['source_performance_gate'] is False


def test_command_keeps_immutable_root_and_row_selection_separate(tmp_path):
    a=SimpleNamespace(**{k:tmp_path/k for k in runner.PATH_ARGS},expected_commit='abc')
    cmd=runner.command(a,'score',d.ROWS[1]);effective=runner.row_args(a,d.ROWS[1])
    assert cmd[3]=='experiments.pamf_r32_all_formal.runner' and cmd[6]==d.ROWS[1]
    assert effective.output==a.output/d.ROWS[1] and str(a.output) in cmd


def test_real_shared_predict_and_score_accept_reference_hook_before_any_model_or_truth(monkeypatch):
    from experiments.pamf_uncertainty_restore.evaluation import predict,score
    class BeforeAnyAccess(RuntimeError):pass
    calls=[]
    def frozen_reference(args,role):
        calls.append(role);raise BeforeAnyAccess('Registered reference consulted first')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    with pytest.raises(BeforeAnyAccess):predict(SimpleNamespace(),'target',checkpoint_loader=runtime.load_final,reference_loader=frozen_reference)
    with pytest.raises(BeforeAnyAccess):score(SimpleNamespace(),'target',reference_loader=frozen_reference)
    assert calls==['target','target']
