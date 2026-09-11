import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.anchored_calibration import FinalProbabilityCalibrator


def test_final_temperature_and_system_identity(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(4)
    lp=torch.randn(100,3).double().log_softmax(-1);y=torch.arange(100)%3
    cal=FinalProbabilityCalibrator().fit(lp,y,role='V',system_identity='frozen-demo')
    out=cal.predict(lp,system_identity='frozen-demo')
    assert torch.equal(out['log_probabilities'].argmax(-1),lp.argmax(-1))
    assert .05<=cal.temperature<=20.
    assert cal.attained_source_coverage>=.9
    with pytest.raises(ValueError,match='identity'):cal.predict(lp,system_identity='changed-expert')
    with pytest.raises(RuntimeError,match='frozen'):cal.fit(lp,y,role='V',system_identity='frozen-demo')
    with pytest.raises(ValueError,match='V'):FinalProbabilityCalibrator().fit(lp,y,role='L_s',system_identity='x')
    restored=FinalProbabilityCalibrator.from_state_dict(cal.state_dict())
    torch.testing.assert_close(restored.predict(lp,system_identity='frozen-demo')['log_probabilities'],out['log_probabilities'])


def test_tied_confidences_are_retained_together():
    lp=torch.tensor([[1.,0.]]).repeat(20,1).log_softmax(-1);y=torch.zeros(20,dtype=torch.long)
    cal=FinalProbabilityCalibrator().fit(lp,y,system_identity='ties')
    assert cal.attained_source_coverage==1.
    assert cal.predict(lp,system_identity='ties')['accepted'].all()


def test_bad_weights_and_nonfinite_inputs_rejected():
    lp=torch.randn(10,2);y=torch.zeros(10,dtype=torch.long)
    with pytest.raises(ValueError,match='weights'):
        FinalProbabilityCalibrator().fit(lp,y,weights=torch.full((10,),-1.),system_identity='x')
    with pytest.raises(ValueError,match='finite'):
        FinalProbabilityCalibrator().fit(lp*float('nan'),y,system_identity='x')
