from pathlib import Path
from types import SimpleNamespace
import pytest
from experiments.pamf_dura_direct_train import runner,head_guided,head_guided_capacity3,f_guided
from experiments.pamf_dura_direct_train.replace_pending import validate_pending,OLD_RUN,NEW_RUN,OLD_COMMIT


def test_total_process_cap_includes_the_worker_to_be_added_and_preserves_memory_limit():
    state={0:dict(free_mb=18000,pids={1}),1:dict(free_mb=22000,pids={2,3}),
        2:dict(free_mb=23000,pids={4,5,6}),3:dict(free_mb=11999,pids=set()),4:dict(free_mb=24000,pids=set())}
    assert runner.eligible_gpus(state,{4},2)==[0]
    assert runner.eligible_gpus(state,{4},3)==[1,0]
    assert runner.eligible_gpus(state,{4},3,[1,2])==[1]


def test_capacity3_is_a_separate_row_and_routes_children_without_changing_science():
    keys=('RUN','configuration','command','learning_rate','direct_objective','save_model','load_model','smoke','train')
    original={key:getattr(runner,key) for key in keys};objective=f_guided.objective;metric=f_guided.GUIDANCE_METRIC
    try:
        head_guided_capacity3.configure();config=runner.configuration()
        assert runner.RUN==config['run']==head_guided_capacity3.RUN and config['max_total_visible_GPU_processes']==3
        assert config['allowed_GPUs']==[4,6,7]
        assert config['epochs']==20 and config['updates']==4940 and config['F_feature_weight']==head_guided.FEATURE_WEIGHT
        assert config['training_TX_labels']==0 and not config['target_training'] and config['max_own_workers']==1
        assert config['max_own_workers_per_GPU']==1 and config['max_goal_GPU_workers']==2
        args=SimpleNamespace(root='root',commit='a'*40)
        assert all(runner.command(args,mode,'scratch_joint','target')[3]==head_guided_capacity3.MODULE for mode in ('worker','score'))
    finally:
        for key,value in original.items():setattr(runner,key,value)
        f_guided.objective=objective;f_guided.GUIDANCE_METRIC=metric


def test_pending_replacement_rejects_started_row_children_or_wrong_owner(tmp_path):
    old=tmp_path/OLD_RUN;new=tmp_path/NEW_RUN;old.mkdir()
    args=(old,new,{'commit':OLD_COMMIT},'/releases/20261030-DURA-source-headFeature-r01',
        'python -m experiments.pamf_dura_direct_train.head_guided run --commit '+OLD_COMMIT,'')
    validate_pending(*args)
    with pytest.raises(ValueError,match='started'):validate_pending(*args[:-1],'123')
    with pytest.raises(ValueError,match='command'):validate_pending(*args[:4],'python other.py','')
    (old/'submitted-scratch_joint.json').touch()
    with pytest.raises(ValueError,match='started'):validate_pending(*args)
