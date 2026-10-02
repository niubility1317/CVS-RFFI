import pytest
import torch
from experiments.cvs_phase_curvature_identity.model import build,VARIANTS
from experiments.cvs_phase_curvature_attribution.run import intervention,CONDITIONS

@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('condition',CONDITIONS)
def test_registered_intervention_restores_exact_frozen_state(variant,condition):
    model=build(variant)
    with torch.no_grad():
        for i,p in enumerate(model.memory_parameters()):p.fill_(.05*(i+1))
        model.core.behavior[0].mix_raw.fill_(.2)
    before={k:v.clone() for k,v in model.state_dict().items()}
    with intervention(model,condition):
        for i,p in enumerate(model.memory_parameters()):
            off=condition=='curvature_all_off' or condition==f'curvature_off_{i}'
            assert float(p)==pytest.approx(0. if off else .05*(i+1))
        assert model.core.behavior[0].mix_raw.tolist()==pytest.approx([0.,0.] if condition=='input_mix_off' else [.2,.2])
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())

def test_exception_restores_and_unknown_condition_rejected():
    model=build(VARIANTS[0])
    with torch.no_grad():model.memory_parameters()[0].fill_(.3)
    with pytest.raises(RuntimeError):
        with intervention(model,'curvature_all_off'):raise RuntimeError('interrupted')
    assert float(model.memory_parameters()[0])==pytest.approx(.3)
    with pytest.raises(ValueError):
        with intervention(model,'unregistered'):pass


def test_inspection_does_not_replace_config_inside_log_marker(monkeypatch,tmp_path):
    import json
    from experiments.cvs_phase_curvature_attribution import publish
    scripts=[]
    def fake_ssh(script):
        compile(script,'inspect','exec')
        assert "text.split('RESOLVED_CONFIG',1)" in script
        scripts.append(script)
        return json.dumps({'pipeline':None,'rows':[]})
    monkeypatch.setattr(publish,'ssh',fake_ssh)
    assert publish.inspect(tmp_path)=={'pipeline':None,'rows':[]}
    assert len(scripts)==1
