import numpy as np
import pytest

from experiments.phase1_dura_orthogonal import ground
from experiments.phase1_dura_orthogonal.conditional_iq_flow import ConditionalIQFlow
from experiments.phase1_dura_orthogonal.train_source import SCHEMA, WIDE_FIT_SIZE


def test_ground_loads_only_matching_source_flow_checkpoint(monkeypatch, tmp_path):
    model = ConditionalIQFlow(width=32, steps=4)
    common = dict(commit='source-commit', epochs=5, architecture='conditional_iq_flow',
                  objective='wide_conditional_flow', TX_labels_read=0, F_weights_read=0,
                  F_outputs_read=0, V_reads=0, target_reads=0, DURA_frozen=True,
                  source_fit_size=WIDE_FIT_SIZE, DURA_training_reads=1,
                  flow_steps=4, flow_time_seed=2026110713, flow_path='anchor_to_clean_linear')
    summary = dict(status='SOURCE_TRAINED', fit_ids=WIDE_FIT_SIZE, **common)
    state = dict(schema=SCHEMA, fit_ids=np.arange(WIDE_FIT_SIZE), width=32,
                 model=model.state_dict(), **common)
    monkeypatch.setattr(ground.core, 'read', lambda _: summary)
    monkeypatch.setattr(ground.torch, 'load', lambda *args, **kwargs: state)
    monkeypatch.setattr(ground, 'frozen_dura', lambda *args: model)
    monkeypatch.setattr(ground.core, 'teacher', lambda *args: (model, None))
    loaded, _, _ = ground.load_fixed(tmp_path, 'source-run', 'source-commit')
    assert isinstance(loaded, ConditionalIQFlow) and loaded.steps == 4
    state['flow_steps'] = 5
    with pytest.raises(ValueError, match='flow'):
        ground.load_fixed(tmp_path, 'source-run', 'source-commit')
