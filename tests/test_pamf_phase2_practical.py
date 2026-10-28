from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from experiments.pamf_phase2_practical.data import allocation
from experiments.pamf_phase2_practical.io import validate_split
from experiments.pamf_phase2_practical.runner import eligible,make_jobs
from experiments.pamf_phase2_practical.milkr.d92_branch_local_ridge import fit_branch_local_ridge

def split():
    return dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id='test',split_id='s',receiver='rx',scenario='MIX01',k=1,
                registered_classes=['old','new'],support_indices=[0,1],support_labels=[0,1],query_indices=[2,3],support_seed=1)

@pytest.mark.parametrize('mutation',[lambda s:s.update(query_truth=[0,1]),lambda s:s.update(query_indices=[0,2]),lambda s:s.update(capsule_id='wrong'),lambda s:s.update(support_labels=[0,0])])
def test_reject_illegal_inputs(mutation):
    s=split();mutation(s)
    with pytest.raises(ValueError):validate_split(s,dict(capsule_id='test'),4)

def test_capacity_failure_and_disjoint_allocation():
    days=[list(range(50)) for _ in range(4)]
    with pytest.raises(ValueError,match='Insufficient'):allocation(days,11,36,30,1)
    pools=allocation(days,11,10,8,1)
    ids=[i for roles in pools for role in roles for i in role]
    assert len(ids)==198 and len(set(ids))==198
    assert all(len(s)==10 and len(q)==8 and not set(s)&set(q) for s,q in pools)

def test_three_slot_gpu_limit_and_foreign_user():
    assert eligible(dict(blocked=False,pids={1,2},free=10000),2,3)
    assert not eligible(dict(blocked=False,pids={1,2,3},free=10000),2,3)
    assert not eligible(dict(blocked=True,pids={1},free=20000),0,1)

def test_connection_file_is_explicit_and_checked_before_packaging(tmp_path):
    from experiments.pamf_phase2_practical.publish import connection_options
    with pytest.raises(FileNotFoundError,match='--ssh-config'):connection_options(tmp_path)
    path=tmp_path/'known-project-config';path.write_text('Host N607\n',encoding='utf-8')
    options=connection_options(tmp_path,path)
    assert Path(options[1])==path.resolve() and options[-1]=='ConnectTimeout=10'

def test_pending_data_decision_has_no_target_jobs():
    rt=dict(run_root='run',preparation_only=True)
    assert [j['kind'] for j in make_jobs(rt)]==['smoke']

def test_current_support_refit_order_and_per_query_independence():
    rng=np.random.default_rng(40)
    values={k:rng.normal(size=(8,96 if k=='fft' else 160)) for k in ('z_id','fft','t_emb','f_emb','pa_local')}
    args=dict(support_labels=np.repeat(np.arange(4),2),support_ids=[f'physical-{i}' for i in range(8)],classes=['a','b','c','d'],old_classes=['a','b'])
    state=fit_branch_local_ridge(**values,**args)
    q={k:rng.normal(size=(3,v.shape[1])) for k,v in values.items()}
    before=state.alpha.copy();together=state.score(**q);apart=np.concatenate([state.score(**{k:v[i:i+1] for k,v in q.items()}) for i in range(3)])
    np.testing.assert_array_equal(together,apart);np.testing.assert_array_equal(before,state.alpha)
    order=rng.permutation(8);rev=fit_branch_local_ridge(**{k:v[order] for k,v in values.items()},
        support_labels=args['support_labels'][order],support_ids=[args['support_ids'][i] for i in order],classes=args['classes'],old_classes=args['old_classes'])
    np.testing.assert_allclose(state.score(**q),rev.score(**q),rtol=0,atol=1e-12)
    assert state.audit_dict()['query_rows_used']==0 and state.audit_dict()['optimizer_steps']==0
