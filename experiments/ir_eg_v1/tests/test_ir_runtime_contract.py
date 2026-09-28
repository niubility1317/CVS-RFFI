"""Compatibility with the production N607 PyTorch 2.1 autocast API."""
from copy import deepcopy

import pytest
import torch

from ir_fixtures import make_case, run, assert_nested


@pytest.mark.parametrize('cpu_autocast', [False, True])
def test_legacy_autocast_api_real_ir_step(monkeypatch, cpu_autocast):
    monkeypatch.setenv('IR_ACCEPTANCE_DEVICE', 'cpu')
    case = make_case()
    before = deepcopy(case.model.state_dict())
    modern_query = torch.is_autocast_enabled
    attempted_device_queries = []

    def legacy_query(*args):
        if args:
            attempted_device_queries.append(args)
            raise TypeError('is_autocast_enabled() takes no arguments (1 given)')
        return modern_query()

    # Keep the real legacy CPU getter and actual autocast state. This exercises
    # the complete native IR step and transaction, not a coefficient-only mock.
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=cpu_autocast):
        with monkeypatch.context() as old_api:
            old_api.setattr(torch, 'is_autocast_enabled', legacy_query)
            if cpu_autocast:
                with pytest.raises(ValueError, match='FP32 without autocast'):
                    run(case)
            else:
                assert run(case).accepted
    assert ('cpu',) in attempted_device_queries
    if cpu_autocast:
        assert case.solver.steps == case.external.commits == 0
        assert not case.opt.state
        assert_nested(before, case.model.state_dict())
    else:
        assert case.solver.steps == case.external.commits == 1
