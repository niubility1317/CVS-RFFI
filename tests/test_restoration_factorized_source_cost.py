import copy
import json
from pathlib import Path
import pytest
import torch


def test_cost_probe_rejects_wrong_parent_or_training_information():
    from experiments.pamf_phase3_restoration.factorized_source_cost import validate_parent_metadata
    path=Path(__file__).resolve().parents[1]/'automation_reports/CV-SincNet/20261102-phase1-Restore-CovarianceFeature-E30-s2026092701-r01/source_independent_check.json'
    done=json.loads(path.read_text(encoding='utf-8'))['training']['TF_CovarianceFeature']['complete']
    validate_parent_metadata(done)
    for field,value in [('code_commit','other'),('epoch',29),('updates',7409),('target_training_reads',1),('TX_labels_read',1)]:
        bad=copy.deepcopy(done);bad[field]=value
        with pytest.raises(ValueError):validate_parent_metadata(bad)
    bad=copy.deepcopy(done);bad['F0_identity']['source_code_commit']='other'
    with pytest.raises(ValueError):validate_parent_metadata(bad)


def test_cost_waits_for_last_parent_gpu_stage(tmp_path):
    from experiments.pamf_phase3_restoration.factorized_source_cost import spare_card
    from experiments.pamf_phase3_restoration import covariance_feature_trial as parent
    folder=tmp_path/'runs'/parent.RUN;folder.mkdir(parents=True)
    assert spare_card(tmp_path)==()
    (folder/f'terminal-{parent.ARM}-cross_rx.json').write_text(json.dumps(dict(pid=17,exit_code=0)))
    assert spare_card(tmp_path)==()
    (folder/f'terminal-{parent.ARM}-cross_both.json').write_text(json.dumps(dict(pid=19,exit_code=0)))
    assert spare_card(tmp_path)==(4,6)
    (folder/f'terminal-{parent.ARM}-train.json').write_text(json.dumps(dict(pid=13,exit_code=1)))
    with pytest.raises(RuntimeError):spare_card(tmp_path)


def test_source_logical_backward_preserves_gradients_and_all_model_state():
    from experiments.pamf_phase3_restoration.factorized_source_cost import logical_step
    from experiments.pamf_phase3_restoration.covariance_model import scratch_covariance
    class Reader(torch.nn.Module):
        def __init__(self):
            super().__init__();self.projection=torch.nn.Linear(512,160)
        def features(self,x):return self.projection(x.flatten(1))
    torch.manual_seed(21);G=scratch_covariance(31);F=Reader().eval().requires_grad_(False)
    clean=torch.randn(2,2,256);target=F.features(clean).detach().requires_grad_(True)
    v=dict(received=torch.randn_like(clean),kernel=torch.randn(2,2,64),coefficients=torch.randn(2,4,256),variance=torch.tensor([.02,.3]))
    before={k:x.clone() for k,x in G.state_dict().items()}
    dense=logical_step(G,F,v,clean,target,torch.ones(160),'dense',capture=True)
    factor=logical_step(G,F,v,clean,target,torch.ones(160),'factor',capture=True)
    torch.testing.assert_close(dense['loss'],factor['loss'],rtol=2e-6,atol=2e-7)
    assert set(dense['gradients'])==set(factor['gradients'])
    for key,value in dense['gradients'].items():torch.testing.assert_close(value,factor['gradients'][key],rtol=2e-4,atol=2e-6)
    assert all(torch.equal(value,G.state_dict()[key]) for key,value in before.items())
    assert all(p.grad is None for p in F.parameters()) and target.grad is None
    with pytest.raises(ValueError):logical_step(G,F,v,clean,target,torch.ones(160),'unregistered')
