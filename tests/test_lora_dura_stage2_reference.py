import pytest
import torch

from experiments.pamf_lora_main.lora_dura_stage2_reference import LoRaStage2
from experiments.pamf_r65_four_route.stage_polyphase import StageDistilled


def test_lora_reference_keeps_the_two_stage_r69_parameterization():
    parent_shape = StageDistilled()
    model = LoRaStage2()
    assert len(model.prior)==2
    assert sum(p.numel() for p in model.parameters())==266951
    assert {k:v.shape for k,v in parent_shape.state_dict().items()}=={k:v.shape for k,v in model.state_dict().items()}
    assert model.frozen.estimator.initial_kernel.argmax()==0
    assert parent_shape.frozen.estimator.initial_kernel.argmax()==24


def test_lora_reference_executes_two_native_stages_with_two_cg_updates_each():
    model = LoRaStage2().eval()
    received=torch.randn(1,2,8192)
    calls=[]

    def solve(action, observed, prior, variance, *, steps, initial):
        calls.append((observed.shape,variance.shape,steps,initial.shape))
        return prior

    with torch.no_grad():
        stages,_,_,_=model.stage_outputs(received,solve_map=solve)
        assert [t.shape for t in stages]==[received.shape,received.shape]
        assert all(torch.isfinite(t).all() for t in stages)
        assert calls==[(torch.Size((1,8192)),torch.Size((1,8192)),2,torch.Size((1,8192)))]*2
        with pytest.raises(ValueError,match='IQ2x256'):
            StageDistilled().stage_outputs(received)
