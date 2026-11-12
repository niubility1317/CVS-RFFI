import numpy as np

from experiments.pamf_phase2_practical import clean_scalr_worker as clean
from experiments.pamf_phase2_practical import r42_suite_worker as common
from tests.test_r42_suite_worker import fake_runtime, setup_case


def test_scalr_clean_only_two_paths_and_stage_limited_queries(tmp_path, monkeypatch):
    rt,job,cap,bank=setup_case(tmp_path)
    calls=fake_runtime(monkeypatch,cap,bank)
    rt.update(clean_root=str(tmp_path/'clean'),arm='R42',kernel_method='adaptive_residual_ensemble')
    monkeypatch.setattr(clean,'source_smoke',lambda *args:None)
    seen=[]
    def task(cap, split_id, *, stage, clean_root, task_root, restored_root=None):
        spec=common.read(cap/'splits'/f'{split_id.removeprefix("split")}.json')
        support=np.asarray(spec['support_indices']); query=np.arange(5,len(spec['registered_classes'])*6,6)
        seen.append((stage,len(query)))
        value=dict(query_x=bank[query],input_condition='clean_ground_pre_satellite_normalized')
        if stage!='A':
            value.update(support_x=bank[support],support_y=np.repeat(np.arange(len(spec['registered_classes'])),5),
                         support_ids=list(map(str,support)))
        return value
    monkeypatch.setattr(clean,'load_task',task)
    monkeypatch.setattr(clean,'split_rows',common.split_rows)
    clean.run(rt,job)
    root=tmp_path/'run/predictions/rx'; rows=common.read(root/'predictions.json')['rows']
    assert len(rows)==6
    assert {(r['stage'],r['input_path']) for r in rows}=={
        ('A0','Q0'),('A0','Q1'),('B','S0Q0'),('B','S1Q1'),('C','S0Q0'),('C','S1Q1')}
    assert seen==[('A',6),('B',6),('C',26)]
    assert all(r['input_condition']=='clean_ground_pre_satellite_normalized' for r in rows)
    assert [n for arm,n,flags in calls['fits']]==[30,30,130,130]
    assert common.read(root/'complete.json')['truth_read'] is False


def test_clean_scalr_rejects_satellite_prediction_reuse(tmp_path):
    rt,job,cap,bank=setup_case(tmp_path)
    rt.update(clean_root=str(tmp_path/'clean'),arm='R42',kernel_method='adaptive_residual_ensemble')
    job['reuse_file']='satellite_predictions.json'
    import pytest
    with pytest.raises(ValueError,match='Do not reuse'):
        clean.run(rt,job)


def test_restored_cache_removes_repeated_g_calls(tmp_path,monkeypatch):
    import shutil
    rt,job,cap,bank=setup_case(tmp_path);calls=fake_runtime(monkeypatch,cap,bank)
    rt.update(clean_root=str(tmp_path/'clean'),arm='R42',kernel_method='adaptive_residual_ensemble')
    monkeypatch.setattr(clean,'source_smoke',lambda *args:None)
    def task(cap,split_id,*,stage,clean_root,task_root,restored_root=None):
        spec=common.read(cap/'splits'/f'{split_id.removeprefix("split")}.json')
        support=np.asarray(spec['support_indices']);query=np.arange(5,len(spec['registered_classes'])*6,6)
        restored=bank.copy();restored[:,0,1]=1
        result=dict(query_x=bank[query],input_condition='clean_ground_pre_satellite_normalized')
        if restored_root:result['restored_query_x']=restored[query]
        if stage!='A':
            result.update(support_x=bank[support],support_y=np.repeat(np.arange(len(spec['registered_classes'])),5),
                          support_ids=list(map(str,support)))
            if restored_root:result['restored_support_x']=restored[support]
        return result
    monkeypatch.setattr(clean,'load_task',task);monkeypatch.setattr(clean,'split_rows',common.split_rows)
    clean.run(rt,job)
    assert calls['G_rows']>0
    original=tmp_path/'run/predictions/rx';second=tmp_path/'second';second.mkdir()
    shutil.copy(tmp_path/'run/source_selection.json',second/'source_selection.json')
    rt.update(run_root=str(second),restored_root=str(tmp_path/'restored'));calls['G_rows']=0
    clean.run(rt,job)
    assert calls['G_rows']==0
    rows=common.read(second/'predictions/rx/predictions.json')['rows']
    assert len(rows)==6 and {r['input_path'] for r in rows}=={'Q0','Q1','S0Q0','S1Q1'}
    for row in rows:
        with np.load(original/row['prediction']) as a, np.load(second/'predictions/rx'/row['prediction']) as b:
            assert np.array_equal(a['predicted_indices'],b['predicted_indices'])
