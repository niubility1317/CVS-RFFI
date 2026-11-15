import pytest


def test_stabilized_mopc_is_versioned_and_requires_mopc_only():
    from experiments.pamf_phase2_practical.mopc_stabilization import policy, arm
    assert policy({}, 'native_cil') is None
    assert policy(dict(profile='mopc_hr',mopc_stabilization='clip1000_v1'), 'native_cil') == 1000.0
    assert policy(dict(profile='dadda_sda',methods=['mopc_hr'],mopc_stabilization='clip1000_v1'), 'hybrid') == 1000.0
    assert arm('mopc_hr', 'clip1000_v1') == 'mopc_hr_clip1000_v1'
    with pytest.raises(ValueError):
        policy(dict(profile='dadda_sda',methods=['mopc_hr','lora'],mopc_stabilization='clip1000_v1'), 'hybrid')
    with pytest.raises(ValueError):
        policy(dict(profile='mopc_hr',mopc_stabilization='unreviewed'), 'native_cil')


def test_stabilized_rows_remain_distinct_from_original_for_scoring():
    from experiments.pamf_phase2_practical.clean_score import expected_rows
    job=dict(profile='dadda_sda',methods=['mopc_hr'],points=[[1,20]],mopc_stabilization='clip1000_v1')
    rows=expected_rows('hybrid',job)
    assert rows == {('dadda_sda__mopc_hr_clip1000_v1','C',1,20,'P0')}
    native=expected_rows('native_cil',dict(profile='mopc_hr',points=[[1,20]],mopc_stabilization='clip1000_v1'))
    assert ('mopc_hr_clip1000_v1','C',1,20,'P0') in native
    assert ('mopc_hr','A0',1,0,'P0') in native


def test_only_real_mopc_numerical_exception_is_quarantined(tmp_path, monkeypatch):
    import json
    from experiments.pamf_phase2_practical import clean_external_worker as worker, group2_worker
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite.registration import MoPCNumericalError
    monkeypatch.setattr(worker, 'validate', lambda *args: None)
    job=dict(profile='dadda_sda',methods=['mopc_hr'],capsule='capsules/P01/MIX01/1-1',
             seed=2026092701,output=str(tmp_path/'out'))
    (tmp_path/'out').mkdir()
    def fail(*args):raise MoPCNumericalError('nonfinite loss')
    monkeypatch.setattr(group2_worker, 'predict', fail)
    with pytest.raises(MoPCNumericalError):worker.run({},job,'hybrid')
    marker=json.loads((tmp_path/'out/numerical_failure.json').read_text())
    assert marker['status']=='MOPC_NUMERICAL_FAILURE' and marker['truth_read'] is False
    (tmp_path/'out/numerical_failure.json').unlink()
    def bug(*args):raise RuntimeError('implementation error')
    monkeypatch.setattr(group2_worker, 'predict', bug)
    with pytest.raises(RuntimeError):worker.run({},job,'hybrid')
    assert not (tmp_path/'out/numerical_failure.json').exists()


def test_native_mopc_nonfinite_scores_are_classified_without_masking_shape_errors(monkeypatch):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from experiments.pamf_phase2_practical import external_execution as execution
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite.registration import MoPCNumericalError
    model=torch.nn.Linear(2,8)
    monkeypatch.setattr(execution,'tensor',lambda x,device:torch.as_tensor(x,device=device))
    native=SimpleNamespace(method='mopc_hr',old_count=6,classes=tuple(range(8)))
    state=dict(model=model,classes=tuple(range(8)),native_state=native,
               score_fn=lambda model,x:torch.full((len(x),8),float('nan')))
    with pytest.raises(MoPCNumericalError):execution.infer(state,np.zeros((2,2),np.float32),8)
    state['score_fn']=lambda model,x:torch.zeros((len(x),7))
    with pytest.raises(ValueError,match='shape'):execution.infer(state,np.zeros((2,2),np.float32),8)


def test_native_support_diagnostic_rejects_wrong_job_before_access():
    from experiments.pamf_phase2_practical.clean_mopc_native_support_diagnostic import run
    with pytest.raises(ValueError,match='diagnostic job'):
        run({},dict(profile='lora_triplet_knn',points=[[1,2]]),1,2,'cpu')


def test_native_diagnostic_excludes_target_handles_from_source_load(monkeypatch):
    from experiments.pamf_phase2_practical import clean_mopc_native_support_diagnostic as diag
    monkeypatch.setattr(diag,'activate',lambda _:None)
    def source(config,*args):
        assert 'data_root' not in config and 'clean_root' not in config
        assert config['run_root']=='source'
        raise RuntimeError('source boundary observed')
    monkeypatch.setattr(diag,'load_source',source)
    rt=dict(run_root='source',data_root='target',clean_root='clean')
    job=dict(profile='mopc_hr',points=[[1,2]],seed=2026092701)
    with pytest.raises(RuntimeError,match='source boundary observed'):
        diag.run(rt,job,1,2,'cpu')


def test_hybrid_nonfinite_scores_are_isolated_but_shape_errors_are_not():
    import numpy as np
    import torch
    from types import SimpleNamespace
    from experiments.pamf_phase2_practical.group2_worker import score_fixed
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite.registration import MoPCNumericalError
    state=SimpleNamespace(method='mopc_hr',model=torch.nn.Linear(2,2),
                          score=lambda x:torch.full((len(x),8),float('inf')))
    with pytest.raises(MoPCNumericalError):
        score_fixed(state,np.zeros((2,2),np.float32),count=8,old_count=6,batch_size=2,device='cpu')
    state.score=lambda x:torch.zeros((len(x),7))
    with pytest.raises(ValueError,match='scores'):
        score_fixed(state,np.zeros((2,2),np.float32),count=8,old_count=6,batch_size=2,device='cpu')
