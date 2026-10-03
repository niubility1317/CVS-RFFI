import copy
import pytest
import torch
from experiments.cvs_neural_readout_identity.model import build
from experiments.cvs_readout_attribution.run import components,validate_config,CONDITIONS,VARIANTS,PROJECT,SOURCE_RUN,SOURCE_COMMIT,RUN,OWNER

def config():
    rid='readout_attention-s2026092701'
    return dict(source_release=PROJECT+'/releases/cvs_neural_readout_identity_20261003_r01',source_commit=SOURCE_COMMIT,
        source_output=PROJECT+'/runs/'+SOURCE_RUN+'/'+rid+'/source',variant='readout_attention',model_seed=2026092701,
        output_root=PROJECT+'/runs/'+RUN+'/'+rid,conditions=list(CONDITIONS),role='V',launch_owner=OWNER)

@pytest.mark.parametrize('key,value',[('role','query'),('conditions',['all_on']),('source_commit','wrong'),
    ('model_seed',1),('source_release','/wrong'),('source_output','/wrong'),('output_root','/wrong'),
    ('variant','native'),('launch_owner','other')])
def test_reject_changed_contract(key,value):
    c=config();c[key]=value
    with pytest.raises(ValueError):validate_config(c)

def test_accept_exact_contract():assert validate_config(config())==config()

@pytest.mark.parametrize('variant',VARIANTS)
def test_interventions_match_only_selected_projection_removal(variant):
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.manual_seed(523)
        model=build(variant).eval()
        for _,block in model.readout_blocks():block.project.weight.normal_(0,.01)
        x=torch.randn(3,2,256)
        before={k:v.clone() for k,v in model.state_dict().items()}
        scores,features=components(model,x)
        assert set(scores)==set(CONDITIONS)
        assert set(features)=={'time_skip','time_delta','behavior_skip','behavior_delta'}
        assert all(value.shape==(3,320) for value in features.values())
        assert torch.equal(scores['all_on'],model(x))
        for condition in CONDITIONS[1:]:
            control=copy.deepcopy(model)
            for path in ('time','behavior'):
                if condition in (path+'_off','both_off'):getattr(control.core,path+'_readout').project.weight.zero_()
            assert torch.equal(scores[condition],control(x))
        assert any(not torch.equal(scores['all_on'],scores[c]) for c in CONDITIONS[1:])
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
        assert all(not m._forward_hooks for m in model.modules())

def test_hooks_removed_on_forward_error(monkeypatch):
    model=build(VARIANTS[0]).eval()
    def failed(x):raise RuntimeError('controlled forward failure')
    monkeypatch.setattr(model,'forward',failed)
    with pytest.raises(RuntimeError):components(model,torch.zeros(1,2,256))
    assert all(not m._forward_hooks for m in model.modules())
