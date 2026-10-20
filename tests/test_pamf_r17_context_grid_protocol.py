import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from experiments.pamf_r17_context_grid import design as d,runtime as r,models as m,evaluation as e,runner
from experiments.pamf_uncertainty_restore import design as ancestor


def parent():
    return dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,action_run=ancestor.ACTION_RUN,
        action_commit=ancestor.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)


def info(arm):
    return dict(mode='data_free_shared_SVD',fraction=d.FRACTIONS[arm],layers=[{}]*28,seconds=1.,
        IQ_examples=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,pointwise_weights_shared=d.POINTWISE[arm],context_fraction=d.CONTEXT_FRACTION[arm],context_layers=[{}]*(0 if d.CONTEXT_FRACTION[arm]==1. else 84))


@pytest.mark.parametrize('arm',d.ARMS)
def test_checkpoint_reloads_shared_weights_and_independent_states_without_teacher(tmp_path,arm):
    model=m.SharedFactorRestorer(arm).eval();model.initialization_report=info(arm)
    path=tmp_path/'final.pt';r.save_checkpoint(path,model,None,20,4940,parent())
    fresh,value=r.load_checkpoint(path,'cpu')
    assert value['optimizer'] is None and value['training_F_access'] is True
    assert value['training_F_teacher']['head_fit_updates']==0
    assert all(not k.startswith(('reader.','teacher.')) for k in value['model'])
    assert sum(p.numel() for p in fresh.parameters())==d.PARAMETERS[arm]
    with torch.no_grad():
        x=torch.randn(2,2,256)*.1
        assert torch.equal(model(x),fresh(x))
    with pytest.raises(FileExistsError):r.save_checkpoint(path,model,None,20,4940,parent())


def test_conflicting_saved_aliases_are_rejected_instead_of_last_copy_winning():
    model=m.SharedFactorRestorer(d.ARMS[0]);state=copy.deepcopy(model.state_dict())
    keys=[key for key in state if key.startswith('prior.1.') and key.endswith('.left.weight')]
    assert keys
    state[keys[0]]=state[keys[0]]+1
    with pytest.raises(ValueError,match='Conflicting'):r.validate_aliases(model,state)


@pytest.mark.parametrize('key,bad',[('target_training',True),('training_TX_labels_read',1),('training_F_access',False),
    ('parent_R17_commit','unknown'),('parent_R17_epoch',19),('updates',4939)])
def test_checkpoint_permission_lineage_and_budget_cannot_change(key,bad):
    value=r.metadata(d.ARMS[0],20,4940,parent(),info(d.ARMS[0]));r.validate_payload(value);value[key]=bad
    with pytest.raises(ValueError):r.validate_payload(value)


def test_no_calibration_or_target_initialization_reads_allowed():
    value=r.metadata(d.ARMS[0],20,4940,parent(),info(d.ARMS[0]))
    value['initialization']['IQ_examples']=512
    with pytest.raises(ValueError,match='Data-free'):r.validate_payload(value)
    value=r.metadata(d.ARMS[0],20,4940,parent(),info(d.ARMS[0]));value['parent']['training_TX_F_reads']=1
    with pytest.raises(ValueError,match='ancestry'):r.validate_payload(value)


def test_unknown_arm_stops_before_teacher_IO(monkeypatch):
    monkeypatch.setattr(r,'validate_teacher_root',lambda *a:pytest.fail('Unknown arm must not open teacher'))
    with pytest.raises(ValueError):r.initialize('unknown',Path('/runs'),'cpu')


@pytest.mark.parametrize('bad',['role','row','root','kind'])
def test_reference_rejects_unregistered_roles_and_rows_before_IO(monkeypatch,bad):
    args=SimpleNamespace(output=Path('/runs')/d.RUN/d.ARMS[0],kind=d.KIND);role='target'
    if bad=='role':role='query'
    elif bad=='row':args.output=args.output.parent/'unknown'
    elif bad=='root':args.output=Path('/runs/unknown')/d.ARMS[0]
    else:args.kind='other'
    monkeypatch.setattr(e,'read_json',lambda *a:pytest.fail('No IO'))
    with pytest.raises(ValueError):e.reference(args,role)


@pytest.mark.parametrize('field,bad',[('arms',['shared_half']),('views',['clean']),('target_count_per_view',1),('target_acceptance_allowance_pp',0.)])
def test_changed_target_matrix_stops_before_arrays(monkeypatch,field,bad):
    plan=copy.deepcopy(d.configuration());plan[field]=bad
    args=SimpleNamespace(output=Path('/runs')/d.RUN/d.ARMS[0],kind=d.KIND)
    monkeypatch.setattr(e,'read_json',lambda *a:plan)
    monkeypatch.setattr(e.np,'load',lambda *a,**k:pytest.fail('Target arrays remain closed'))
    with pytest.raises(ValueError,match='comparison'):e.reference(args,'target')


def trained(arm):
    return dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        parameters=d.PARAMETERS[arm],initialization_IQ_examples=0,teacher_parameter_updates=0,context_updates=4940,context_trainable=True,waveform_teacher_weight=0.,R17_training_forward_examples=0,
        F_training_reference='source_paired_clean',clean_IQ_loss_weight=1.,
        TX_labels_read=0,V_reads=0,target_reads=0,training_F_access=True,F_parameter_updates=0,head_fit_updates=0)


@pytest.mark.parametrize('field,bad',[('updates',4939),('V_reads',1),('target_reads',1),('F_parameter_updates',1),('initialization_IQ_examples',1),('R17_training_forward_examples',1),('waveform_teacher_weight',10.),('F_training_reference','original_R17_restored')])
def test_incomplete_or_illegal_source_budget_stops_target(monkeypatch,field,bad):
    row=Path('/runs')/d.RUN/d.ARMS[0];value=trained(d.ARMS[0]);value[field]=bad
    monkeypatch.setattr(e,'read_json',lambda path:value if path.name=='training_complete.json' else dict(status='ANALYZED'))
    with pytest.raises(ValueError,match='Full source'):e.validate_source_row(row)


def test_source_accuracy_is_diagnostic_only_and_target_reference_opens_IDs_after_final(monkeypatch):
    args=SimpleNamespace(output=Path('/runs')/d.RUN/d.ARMS[0],kind=d.KIND,formal_reference=Path('/formal'))
    calls=[]
    def read(path):
        calls.append(path.name)
        if path.name=='comparison_plan.json':return d.configuration()
        if path.name=='training_complete.json':return trained(d.ARMS[0])
        if path.name=='source_complete.json':return dict(status='ANALYZED',diagnostic_only=True)
        return dict(status='FORMAL_REFERENCE_READY',count=168000,views=list(d.VIEWS),original_IQ_bytes_reused=True,TX_truth_reads=0)
    monkeypatch.setattr(e,'read_json',read)
    def load(path,**kwargs):
        assert calls==['comparison_plan.json','training_complete.json','source_complete.json','manifest.json']
        assert path.name=='physical_ids.npy';return e.np.arange(168000)
    monkeypatch.setattr(e.np,'load',load)
    assert len(e.reference(args,'target')[1])==168000
    assert not any('decision' in name or 'summary' in name or 'truth' in name for name in calls)


def test_scoring_remains_truth_last_and_CPU_only(monkeypatch):
    args=SimpleNamespace(output=Path('/runs')/d.RUN,arm=d.ARMS[0])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','4')
    with pytest.raises(ValueError,match='CPU'):e.score(args,'target')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(e,'reference',lambda *a:(Path('/formal'),e.np.arange(2),None,None))
    class MissingPredictions(Exception):pass
    def incomplete(*a):raise MissingPredictions()
    monkeypatch.setattr(e.base,'prediction_arrays',incomplete)
    monkeypatch.setattr(e.base,'target_baseline',lambda *a:pytest.fail('No truth/baseline before all predictions'))
    with pytest.raises(MissingPredictions):e.score(args,'target')


def test_worker_order_automatically_tests_each_fixed_row_without_score_selection(tmp_path,monkeypatch):
    args=SimpleNamespace(output=tmp_path,arm=d.ARMS[0]);(tmp_path/args.arm).mkdir();seen=[]
    monkeypatch.setattr(runner,'train',lambda *a:seen.append('train'))
    monkeypatch.setattr(runner,'predict',lambda a,role:seen.append('predict_'+role))
    monkeypatch.setattr(runner,'profile',lambda *a:seen.append('profile'))
    monkeypatch.setattr(runner,'score_subprocess',lambda a,role:seen.append('score_'+role))
    monkeypatch.setattr(runner.torch.cuda,'empty_cache',lambda:None)
    runner.worker(args)
    assert seen==['train','predict_source','profile','score_source','predict_target','score_target']
    assert (tmp_path/args.arm/'source_complete.json').exists() and (tmp_path/args.arm/'complete.json').exists()


def test_technical_failure_preserves_row_and_never_dispatches_target(tmp_path,monkeypatch):
    args=SimpleNamespace(output=tmp_path,arm=d.ARMS[0])
    def fail(*a):raise FloatingPointError('nonfinite')
    monkeypatch.setattr(runner,'train',fail)
    monkeypatch.setattr(runner,'predict',lambda *a:pytest.fail('No target on technical failure'))
    with pytest.raises(FloatingPointError):runner.worker(args)
    assert (tmp_path/args.arm/'failure.json').exists()
