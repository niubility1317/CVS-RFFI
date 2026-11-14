import json
import pytest

from experiments.pamf_phase2_practical.clean_affine_b_reuse import prior_b


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return str(path)


def test_prior_b_rejects_identity_mismatch_before_loading(tmp_path, monkeypatch):
    job = dict(key='one', capsule='capsules/P/M/R', profile='protonet_cda', seed=7,
               methods=['mopc_hr'], points=[[20, 20]], mopc_bridge='proto_affine_v1')
    oldjob = dict(job, seed=8, mopc_bridge='original', output=str(tmp_path/'old'), save_B_for_mopc=True)
    rt = dict(data_root='/data', clean_root='/clean', source_run_root='/source', source_contract='/contract')
    job['reuse_original_b'] = dict(job=put(tmp_path/'oldjob.json', oldjob), runtime=put(tmp_path/'rt.json', rt))
    monkeypatch.setattr('experiments.pamf_phase2_practical.clean_affine_b_reuse.load_b_state',
                        lambda *args: pytest.fail('Loaded a mismatched B'))
    with pytest.raises(ValueError, match='identity'):
        prior_b(job, rt, 10, 'split', 9, 'cpu')


def test_missing_b_may_only_be_fitted_when_no_old_rows_exist(tmp_path):
    job = dict(key='one', capsule='capsules/P/M/R', profile='protonet_cda', seed=7,
               methods=['mopc_hr'], points=[[20, 20]], mopc_bridge='proto_affine_v1')
    out = tmp_path/'old/tasks/one'; out.mkdir(parents=True)
    oldjob = dict(job, mopc_bridge='original', output=str(out), save_B_for_mopc=True)
    rt = dict(data_root='/data', clean_root='/clean', source_run_root='/source', source_contract='/contract')
    job['reuse_original_b'] = dict(job=put(tmp_path/'oldjob.json', oldjob), runtime=put(tmp_path/'rt.json', rt))
    state = tmp_path/'old/scheduler/state.json'
    put(state, dict(status='RUNNING', active=[]))
    with pytest.raises(ValueError, match='drained'):
        prior_b(job, rt, 20, 'split', 9, 'cpu')
    put(state, dict(status='FAILED', active=[]))
    assert prior_b(job, rt, 20, 'split', 9, 'cpu') is None
    (out/'C-K20-N2-mopc_hr.npz').touch()
    with pytest.raises(ValueError, match='missing saved B'):
        prior_b(job, rt, 20, 'split', 9, 'cpu')


def test_completed_original_b_is_loaded_and_its_cost_is_preserved(tmp_path):
    import torch
    from types import SimpleNamespace
    from experiments.pamf_phase2_practical.group2_b_state import _binding
    job = dict(key='one', capsule='capsules/P/M/R', profile='protonet_cda', seed=7,
               methods=['mopc_hr'], points=[[10, 20]], mopc_bridge='proto_affine_v1')
    out = tmp_path/'old/tasks/one'; out.mkdir(parents=True)
    oldjob = dict(job, mopc_bridge='original', output=str(out), save_B_for_mopc=True)
    rt = dict(data_root='/data', clean_root='/clean', source_run_root='/source', source_contract='/contract')
    job['reuse_original_b'] = dict(job=put(tmp_path/'oldjob.json', oldjob), runtime=put(tmp_path/'rt.json', rt))
    put(tmp_path/'old/scheduler/state.json', dict(status='FAILED', active=[]))
    anchor = SimpleNamespace(method='mopc_hr', model=torch.nn.Linear(2,2),
                             prototypes=torch.zeros(6,2), protocol_metadata={})
    bfile = out/'B-K10.pt'
    torch.save(dict(version=1, binding=_binding(oldjob, 10, 'split', 9), anchor=anchor,
                    model=torch.nn.Linear(2,2), old_prototypes=torch.zeros(6,2)), bfile)
    put(out/'complete.json', dict(status='PREDICTIONS_COMPLETE', truth_read=False, query_fit=False))
    put(out/'predictions.json', dict(capsule=job['capsule'], profile=job['profile'], model_seed=job['seed'],
        rows=[dict(k=10, resources=dict(B_state_ref=str(bfile.resolve()), B_seconds=3.5, B_native={'a':1}))]))
    found = prior_b(job, rt, 10, 'split', 9, 'cpu')
    assert found['path'] == str(bfile.resolve()) and found['seconds'] == 3.5 and found['native'] == {'a':1}
