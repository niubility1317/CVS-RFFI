from types import SimpleNamespace

import pytest

from experiments.pamf_dura_direct_train import fast_lr


def test_fast_lr_keeps_budget_and_all_child_modes_in_the_new_recipe(monkeypatch):
    base=fast_lr.base
    original=(base.RUN,base.configuration,base.learning_rate,base.command)
    try:
        fast_lr.configure()
        assert base.learning_rate(246)==pytest.approx(1e-3)
        assert base.learning_rate(4939)==pytest.approx(2e-5)
        with pytest.raises(ValueError):base.learning_rate(4940)
        args=SimpleNamespace(root='root',commit='a'*40)
        for mode in ('worker','score'):
            command=base.command(args,mode,'scratch_joint','target' if mode=='score' else None)
            assert command[3]==fast_lr.MODULE
        # Checkpoints use the same recipe identity on training and score reload.
        config=base.configuration()
        assert config['run']==fast_lr.RUN and config['ancestor_checkpoints']==[]
    finally:
        base.RUN,base.configuration,base.learning_rate,base.command=original
