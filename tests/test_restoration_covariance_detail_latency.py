import copy
import json
from pathlib import Path
import numpy as np
import pytest


def test_latency_schedule_balances_route_order_and_counts():
    from experiments.pamf_phase3_restoration.covariance_detail_latency import schedule,ROUTES
    rows=schedule()
    for batch,count in [(1,100),(128,20)]:
        selected=[x for x in rows if x['batch']==batch and not x['warmup']]
        assert len(selected)==count*len(ROUTES)
        for route in ROUTES:
            r=[x for x in selected if x['route']==route]
            assert len(r)==count and sum(x['position']==0 for x in r)==count//5
        for i in range(count):assert len({x['source_block'] for x in selected if x['iteration']==i})==1
    assert sum(x['batch'] for x in rows if x['route']==ROUTES[0])==3047


def test_parent_Main_completion_only_not_scores_controls_release(tmp_path):
    from experiments.pamf_phase3_restoration.covariance_detail_latency import parent_ready
    from experiments.pamf_phase3_restoration.covariance_detail_trial import RUN
    root=tmp_path/'runs'/RUN;root.mkdir(parents=True)
    assert not parent_ready(tmp_path)
    (root/'complete.json').write_text(json.dumps(dict(status='ARTIFACTS_COMPLETE',goal_complete=False)))
    assert parent_ready(tmp_path)
    (root/'terminal-TF_CovarianceDetail-train.json').write_text(json.dumps(dict(exit_code=1)))
    with pytest.raises(RuntimeError):parent_ready(tmp_path)


def test_latency_rejects_wrong_final_model_identity_or_permissions():
    from experiments.pamf_phase3_restoration.covariance_detail_latency import validate_parent
    a=dict(status='SOURCE_TRAINED',code_commit='1fcbc3eb3cc16e7577ecc3cbedda99174b4c5cc6',arm='TF_CovarianceDetail',epoch=30,updates=7410,G_parameters=5481223,
        G_initialization_checkpoint_loads=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_training_reads=0,model_state_F0_updates=0,
        F0_identity=dict(source_code_commit='3ff8f4b7e85b5608f5b45350aa18081499a7ef1d'))
    validate_parent(a)
    for key,value in [('code_commit','other'),('epoch',29),('target_training_reads',1),('G_parameters',1230341)]:
        b=copy.deepcopy(a);b[key]=value
        with pytest.raises(ValueError):validate_parent(b)


def test_latency_summary_uses_only_fixed_timed_rows_and_all_routes():
    from experiments.pamf_phase3_restoration.covariance_detail_latency import summarize,schedule
    rows=[dict(x,wall_ms=2.,CUDA_event_ms=1.5,peak_allocated_bytes=30,baseline_allocated_bytes=10,incremental_bytes=20) for x in schedule()]
    result=summarize(rows)
    assert result['1']['new_G_F0']['timed_calls']==100
    assert result['128']['new_G_F0']['throughput_records_per_second']==64000
    with pytest.raises(ValueError):summarize(rows[:-1])
