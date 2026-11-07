import numpy as np
import pytest
from experiments.phase1_dura_orthogonal.signal_gate_train import validate_parent


def test_parent_contract_and_fit_disjointness():
    state = dict(schema='phase1_dura_orthogonal_source_v1', commit='b873acec23758690e5484bd1c5f82162739c6655',
                 architecture='large_context_stable', objective='wide_large_context_stable', epochs=20,
                 TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0,
                 DURA_frozen=True, fit_ids=list(range(32768)))
    assert len(validate_parent(state, np.array([32768,32769]))) == 32768
    with pytest.raises(ValueError):
        validate_parent(state, np.array([4,32768]))
    state['F_outputs_read'] = 1
    with pytest.raises(ValueError):
        validate_parent(state, np.array([32768]))
