import numpy as np
import pytest
from experiments.pamf_phase2_practical.episodic_dispatch import eligible,jobs
from experiments.pamf_phase2_practical.episodic_score import metrics
from experiments.pamf_phase2_practical.episodic_experiment import decide
from experiments.pamf_phase2_practical.milkr.d92_branch_local_ridge import fit_branch_local_ridge


def test_gpu_two_process_cap_counts_existing_and_pending_workers():
    assert eligible(dict(pids={11},free=8000,blocked=True),0)
    assert not eligible(dict(pids={11,12},free=18000),0)
    assert not eligible(dict(pids=set(),free=18000),2)
    assert not eligible(dict(pids=set(),free=4999),0)


def test_every_full_job_waits_for_source_smoke_data_and_same_row_pilot():
    rt=dict(run_root='/run',physical_pools=['P01','P02','P03'],episodes=[dict(id='MIX01')],target_receivers=['r1','r2'],model_seeds=[1,2])
    matrix=jobs(rt);by={j['id']:j for j in matrix}
    assert by['threepool-audit']['dependencies']==['source-smoke']
    pilot=by['predict-c0000-s1'];assert pilot['dependencies']==['threepool-audit']
    for j in matrix:
        if j['kind']=='predict' and j!=pilot:
            assert 'predict-c0000-s1' in j['dependencies'] and 'threepool-audit' in j['dependencies']
        if j['kind']=='score':assert j['dependencies']==[j['id'].removeprefix('score-')]
    assert len([j for j in matrix if j['kind']=='predict'])==12


def test_macro_does_not_treat_large_class_as_equal_to_all_other_classes():
    labels=np.r_[np.zeros(90,dtype=int),np.repeat(np.arange(1,8),10)]
    result=metrics(labels,np.zeros(len(labels),dtype=int),8)
    assert result['accuracy']==90/160 and result['macro_accuracy']==1/8
    assert result['old_macro_accuracy']==1/6 and result['new_macro_accuracy']==0 and result['H_macro']==0
    assert metrics(np.arange(6),np.arange(6),6)['H_macro'] is None
    with pytest.raises(ValueError,match='Prediction outside'):metrics(np.arange(6),np.full(6,6),6)


def test_chunked_original_milkr_score_preserves_per_record_all_class_decisions():
    rng=np.random.default_rng(20);classes=['z','b','x','a','f','k']
    sf={name:rng.normal(size=(12,width)) for name,width in [('z_id',160),('fft',96),('t_emb',160),('f_emb',160),('pa_local',160)]}
    qf={name:rng.normal(size=(9,value.shape[1])) for name,value in sf.items()}
    state=fit_branch_local_ridge(**sf,support_labels=np.repeat(np.arange(6),2),support_ids=[f's{i}' for i in range(12)],classes=classes,old_classes=classes)
    predicted,_=decide(state,qf,classes,3);scores=state.score(**qf);order=np.argsort(classes)
    assert np.array_equal(predicted,order[scores[:,order].argmax(1)])
