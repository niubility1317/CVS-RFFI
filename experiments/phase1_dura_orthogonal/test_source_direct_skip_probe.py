import pytest
import torch

from experiments.phase1_dura_orthogonal import source_direct_skip_probe as probe
from experiments.phase1_dura_orthogonal.large_direct_timefreq import LargeDirectTimeFreq
from experiments.phase1_dura_orthogonal.large_direct_timefreq_stable import StableLargeDirectTimeFreq


class Reader(torch.nn.Module):
    def features(self, x):
        return x.mean(-1)
    def classify_features(self, x):
        return torch.cat((x, x, x), 1)


@pytest.mark.parametrize('architecture',[LargeDirectTimeFreq, StableLargeDirectTimeFreq])
def test_direct_timefreq_ignores_context_and_preserves_predictions(architecture, monkeypatch):
    torch.manual_seed(19)
    m=architecture(width=16,layers=1).eval()
    with torch.no_grad():m.out.weight.normal_(0, .05)
    x=torch.randn(3,2,256); reader=Reader().eval()
    monkeypatch.setattr(probe,'extract_context',lambda d,x:(x*.7,x*.2,torch.ones(len(x))))
    original=probe.run_batch(x,m,None,reader,True)
    optimized=probe.run_batch(x,m,None,reader,False)
    assert all(torch.equal(original[i],optimized[i]) for i in (0,1,3))


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError,match='direct'):
        probe.run_batch(torch.randn(2,2,256),torch.nn.Identity(),None,Reader(),False)


def test_reference_binds_source_checkpoint_and_prediction_version():
    from experiments.phase1_dura_orthogonal.direct_skip_full_parity import validate_reference_identity
    record = dict(source='src', source_commit='model', commit='predict', inference_mode='single')
    validate_reference_identity(record,'src','model','predict')
    for key in record:
        wrong=record.copy();wrong[key]='other'
        with pytest.raises(ValueError,match='identity mismatch'):
            validate_reference_identity(wrong,'src','model','predict')
