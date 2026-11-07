import json

import numpy as np
import pytest
import torch
from torch import nn

from experiments.pamf_phase3_restoration import phase_only_eval as evaluation
from experiments.pamf_phase3_restoration.phase_ramp_posterior import fixed_grid, ramp_posteriors


class FixedOperator(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('prior_mean',torch.arange(6).float()/7)
        self.register_buffer('prior_covariance',torch.eye(6))

    def estimate(self, received):
        return dict(action_matrix=torch.eye(6,dtype=torch.float64).expand(len(received),-1,-1),
                    variance=received.new_full((len(received),),.1))


def test_inference_only_readout_matches_full_pilot_phase_only():
    operator = FixedOperator()
    model = evaluation.FrozenPhaseOnly(operator)
    received = torch.randn(3,2,3,generator=torch.Generator().manual_seed(5))
    estimate = operator.estimate(received)
    expected,_ = ramp_posteriors(estimate['action_matrix'],received.flatten(1).double(),
        operator.prior_mean.double(),operator.prior_covariance.double(),estimate['variance'].double(),*fixed_grid('cpu'))
    result = model(received)
    torch.testing.assert_close(result,expected[:,1].float().reshape_as(received))
    torch.testing.assert_close(result,torch.cat([model(x[None]) for x in received]))
    assert (expected[:,1]-expected[:,0]).abs().max()>.01


def test_waits_for_c4_last_gpu_stage_or_recorded_failure(tmp_path):
    dependency = tmp_path/'runs'/evaluation.C4_RUN
    dependency.mkdir(parents=True)
    assert not evaluation.c4_gpu_finished(tmp_path)
    (dependency/f'terminal-{evaluation.C4_ARM}-train.json').write_text('{"exit_code":0}')
    assert not evaluation.c4_gpu_finished(tmp_path)
    (dependency/f'terminal-{evaluation.C4_ARM}-cross_both.json').write_text('{"exit_code":0}')
    assert evaluation.c4_gpu_finished(tmp_path)


def test_frozen_control_refuses_prediction_order_change(tmp_path):
    folder = tmp_path/'runs'/evaluation.parent.RUN/'rows'/evaluation.parent.ARM/'prediction'/'cross_rx'
    folder.mkdir(parents=True)
    metadata = dict(status='PREDICTIONS_COMPLETE',code_commit=evaluation.G_COMMIT,count=2,
        truth_read=False,query_fit=False,model_state_updates=0,F0_identity={'frozen':True},
        task='cross_rx',run_id=evaluation.parent.RUN,arm=evaluation.parent.ARM,
        readouts=['raw','DURA_v1','candidate'],views=list(evaluation.base.main_eval.VIEWS))
    (folder/'complete.json').write_text(json.dumps(metadata))
    np.savez(folder/'predictions.npz',ids=np.array(['B','A']))
    with pytest.raises(ValueError,match='order'):
        evaluation.frozen_controls(tmp_path,'cross_rx',np.array(['A','B']),metadata['F0_identity'])
