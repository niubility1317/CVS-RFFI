import numpy as np
import pytest
import torch

from experiments.phase1_dura_orthogonal import quartic_source_structure as diagnostic
from experiments.phase1_dura_orthogonal.quartic_recovery import DuraQuarticRecovery, quartic_loss
from experiments.phase1_dura_orthogonal.train_source import SCHEMA, WIDE_FIT_SIZE


def test_candidate_contract_rejects_label_access_and_pilot_overlap(monkeypatch, tmp_path):
    model = DuraQuarticRecovery(width=32)
    common = dict(commit=diagnostic.CANDIDATE_COMMIT, epochs=5, architecture='dura_quartic',
                  objective='wide_quartic_guard', source_fit_size=WIDE_FIT_SIZE, DURA_training_reads=1,
                  DURA_frozen=True, source_quartic_weight=.1, quartic_offsets=[1, 2, 4, 8],
                  quartic_branch_scale=.1, TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
                  V_reads=0, target_reads=0)
    fit = np.arange(WIDE_FIT_SIZE)
    summary = dict(status='SOURCE_TRAINED', fit_ids=WIDE_FIT_SIZE, **common)
    state = dict(schema=SCHEMA, fit_ids=fit, width=32, model=model.state_dict(), **common)
    roles = np.full(WIDE_FIT_SIZE+1, 'L_s')
    monkeypatch.setattr(diagnostic.core, 'read', lambda _: summary)
    monkeypatch.setattr(diagnostic.torch, 'load', lambda *args, **kwargs: state)
    loaded, loaded_fit = diagnostic.load_candidate(tmp_path, np.array([WIDE_FIT_SIZE]), roles)
    assert isinstance(loaded, DuraQuarticRecovery) and np.array_equal(loaded_fit, fit)
    state['TX_labels_read'] = 1
    with pytest.raises(ValueError, match='source-only'):
        diagnostic.load_candidate(tmp_path, np.array([WIDE_FIT_SIZE]), roles)
    state['TX_labels_read'] = 0
    with pytest.raises(ValueError, match='disjoint'):
        diagnostic.load_candidate(tmp_path, np.array([0]), roles)


def test_feature_shapes_and_finiteness():
    power, amplitude, quartic = diagnostic.structure(torch.randn(8, 2, 256))
    assert (power.shape, amplitude.shape, quartic.shape) == ((8, 256), (8, 16), (8, 256))
    assert all(torch.isfinite(value).all() for value in (power, amplitude, quartic))


def test_per_record_objective_matches_training_loss_and_identical_input():
    output, clean = torch.randn(8, 2, 256), torch.randn(8, 2, 256)
    values = diagnostic.per_record_quartic_error(output, clean)
    assert values.shape == (8,)
    assert torch.allclose(values.mean(), quartic_loss(output, clean), rtol=1e-6, atol=1e-8)
    assert torch.equal(diagnostic.per_record_quartic_error(clean, clean), torch.zeros(8))
