import json
import pytest
import torch
from types import SimpleNamespace

from experiments.pamf_phase2_practical.group2_b_state import _binding


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return str(path)


def test_saved_anchor_requires_exact_binding_and_preserves_original(tmp_path):
    from experiments.pamf_phase2_practical.clean_saved_anchor_reuse import prior_anchor
    output=tmp_path/'old/tasks/example';output.mkdir(parents=True)
    old=dict(key='example',profile='dadda_sda',seed=2026092701,capsule='capsules/P/M/R',
             methods=['mopc_hr'],points=[[1,2]],output=str(output),save_B_for_mopc=True)
    rt=dict(data_root='/data',clean_root='/clean',source_run_root='/source',source_contract='/contract')
    job=dict(old,mopc_stabilization='clip1000_v1',output=str(tmp_path/'new'))
    job['reuse_saved_b']=dict(job=put(tmp_path/'oldjob.json',old),runtime=put(tmp_path/'oldrt.json',rt))
    put(tmp_path/'old/scheduler/state.json',dict(status='FAILED',active=[]))
    anchor=SimpleNamespace(method='mopc_hr',model=torch.nn.Linear(2,2),prototypes=torch.zeros(6,2),protocol_metadata={})
    path=output/'B-K1.pt'
    torch.save(dict(version=1,binding=_binding(old,1,'split',17),anchor=anchor,model=torch.nn.Linear(2,2),
                    old_prototypes=None),path)
    found=prior_anchor(job,rt,1,'split',17,'cpu')
    assert found['path']==str(path.resolve()) and found['payload']['anchor'].method=='mopc_hr'
    put(tmp_path/'old/scheduler/state.json',dict(status='RUNNING',active=[]))
    with pytest.raises(ValueError,match='drained'):
        prior_anchor(job,rt,1,'split',17,'cpu')
    put(tmp_path/'old/scheduler/predictions_complete.json',dict(status='SELECTED_PREDICTIONS_FIXED'))
    assert prior_anchor(job,rt,1,'split',17,'cpu')['path']==str(path.resolve())
    with pytest.raises(ValueError,match='binding'):
        prior_anchor(job,rt,1,'split',18,'cpu')
    assert path.is_file()
    put(output/'complete.json',dict(status='PREDICTIONS_COMPLETE',truth_read=False,query_fit=False))
    put(output/'predictions.json',dict(capsule=job['capsule'],profile=job['profile'],model_seed=job['seed'],
        rows=[dict(k=1,resources=dict(B_state_ref='wrong'))]))
    with pytest.raises(ValueError,match='row reference'):
        prior_anchor(job,rt,1,'split',17,'cpu')


def test_tweak_saved_b_rejects_different_source_checkpoint(tmp_path):
    from experiments.pamf_phase2_practical.clean_saved_anchor_reuse import prior_anchor
    old=dict(key='example',profile='tweak_m1',seed=2026092701,capsule='capsules/P/M/R',
             methods=['mopc_hr'],points=[[1,2]],output=str(tmp_path/'old/tasks/example'),save_B_for_mopc=True)
    rt=dict(data_root='/data',clean_root='/clean',source_run_root='/source',source_contract='/contract',
            additional_sources={'tweak_m1':{'2026092701':'/new-tweak'}})
    old_rt=dict(rt,additional_sources={'tweak_m1':{'2026092701':'/old-tweak'}})
    job=dict(old,mopc_stabilization='clip1000_v1',reuse_saved_b=dict(
        job=put(tmp_path/'oldjob.json',old),runtime=put(tmp_path/'oldrt.json',old_rt)))
    with pytest.raises(ValueError,match='Tweak source'):
        prior_anchor(job,rt,1,'split',17,'cpu')


def test_saved_anchor_missing_b_cannot_hide_existing_prediction(tmp_path):
    from experiments.pamf_phase2_practical.clean_saved_anchor_reuse import prior_anchor
    output=tmp_path/'old/tasks/example';output.mkdir(parents=True)
    old=dict(key='example',profile='dadda_sda',seed=2026092701,capsule='capsules/P/M/R',
             methods=['mopc_hr'],points=[[1,2]],output=str(output),save_B_for_mopc=True)
    rt=dict(data_root='/data',clean_root='/clean',source_run_root='/source',source_contract='/contract')
    job=dict(old,mopc_stabilization='clip1000_v1')
    job['reuse_saved_b']=dict(job=put(tmp_path/'oldjob.json',old),runtime=put(tmp_path/'oldrt.json',rt))
    put(tmp_path/'old/scheduler/state.json',dict(status='FAILED',active=[]))
    assert prior_anchor(job,rt,1,'split',17,'cpu') is None
    (output/'C-K1-N2-mopc_hr.npz').touch()
    with pytest.raises(ValueError,match='missing saved B'):
        prior_anchor(job,rt,1,'split',17,'cpu')
