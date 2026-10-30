import numpy as np
import pytest
import torch

from experiments.pamf_dura_efficiency.rf_geometry import analytic_terms, calibration_decision, gradient_pair, source_outputs
from experiments.pamf_restoration.views import PAMFViews


def test_fixed_analytic_features_have_input_gradients_without_checkpoint_or_classifier(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Trained checkpoint/classifier must not be used')
    monkeypatch.setattr(torch,'load',forbidden)
    extractor=PAMFViews().eval().requires_grad_(False)
    monkeypatch.setattr(extractor.probe,'forward',forbidden)
    monkeypatch.setattr(extractor.probe,'features',forbidden)
    before={k:v.clone() for k,v in extractor.state_dict().items()}
    x=torch.randn(2,2,256,generator=torch.Generator().manual_seed(31))
    iq,terms,g_iq,g_rf=gradient_pair(x+.03*torch.roll(x,1,-1),x,extractor)
    assert set(terms)=={'frequency','pa','sinc','highfreq','reference'}
    assert iq.shape==(2,) and g_iq.shape==g_rf.shape==x.shape
    assert torch.isfinite(g_rf).all() and g_rf.norm()>0
    assert all(p.grad is None and not p.requires_grad for p in extractor.parameters())
    assert all(torch.equal(v,before[k]) for k,v in extractor.state_dict().items())
    zero=analytic_terms(x,x,extractor)
    assert all(torch.equal(v,torch.zeros(2)) for v in zero.values())


def rows():return [dict(view=v,ratio=4.,cosine=.2) for v in (1,20) for _ in range(16)]


def test_calibration_is_one_fixed_scalar_and_needs_both_views():
    data=rows();value=calibration_decision(data)
    assert value['alpha']==.0625 and value['analytic_RF_study_eligible']
    data[16:]=[dict(view=20,ratio=4.,cosine=-.1) for _ in range(16)]
    assert not calibration_decision(data)['analytic_RF_study_eligible']
    data=rows();data[16:]=[dict(view=20,ratio=100.,cosine=.2) for _ in range(16)]
    assert not calibration_decision(data)['analytic_RF_study_eligible']
    with pytest.raises(ValueError):calibration_decision(rows()[:-1])
    data=rows();data[0]['ratio']=float('nan')
    with pytest.raises(ValueError):calibration_decision(data)


def test_saved_outputs_validate_source_metadata_without_opening_F_members(monkeypatch):
    indices=np.arange(512);ids=indices+1000;roles=np.array(['L_s']*128+['U_s']*384)
    x=np.ones((512,2,256),dtype=np.float32)
    values=dict(source_positions=indices,physical_ids=ids,roles=roles,clean=x,native=x)
    class Saved:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __getitem__(self,key):
            assert key in values,'Forbidden saved member opened'
            return values[key]
    monkeypatch.setattr(np,'load',lambda *a,**k:Saved())
    assert np.array_equal(source_outputs('unused',indices,ids,roles,x),x)
    values['physical_ids']=ids[::-1]
    with pytest.raises(ValueError,match='physical_ids'):source_outputs('unused',indices,ids,roles,x)
