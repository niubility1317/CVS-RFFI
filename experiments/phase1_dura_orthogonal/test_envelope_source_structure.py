import torch
import numpy as np
import pytest

from experiments.phase1_dura_orthogonal import envelope_source_structure as diagnostic
from experiments.phase1_dura_orthogonal.envelope_source_structure import physical_ranks, summarize_ranks
from experiments.phase1_dura_orthogonal.monotone_envelope import DuraMonotoneEnvelope
from experiments.phase1_dura_orthogonal.train_source import SCHEMA, WIDE_FIT_SIZE


def test_physical_ranks_match_unique_records_and_penalize_ties():
    identity = torch.eye(8)
    ranks = physical_ranks(identity, identity, torch.zeros(8), torch.ones(8))
    assert torch.equal(ranks, torch.ones(8, dtype=torch.int64))
    tied = physical_ranks(torch.zeros(8, 2), torch.zeros(8, 2), torch.zeros(2), torch.ones(2))
    assert torch.equal(tied, torch.full((8,), 8, dtype=torch.int64))
    result = summarize_ranks(tied, torch.ones(8, dtype=torch.bool))
    assert result['count'] == 8 and result['top1'] == 0 and result['top5'] == 0


def test_candidate_contract_rejects_label_reads_and_fit_overlap(monkeypatch, tmp_path):
    model = DuraMonotoneEnvelope(width=32)
    common = dict(commit=diagnostic.CANDIDATE_COMMIT, epochs=5, architecture='dura_envelope',
                  objective='wide_envelope_guard', source_fit_size=WIDE_FIT_SIZE, DURA_training_reads=1,
                  DURA_frozen=True, source_envelope_weight=.5, source_envelope_quantiles=16,
                  envelope_segments=8, envelope_segment_width=.5,
                  TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0)
    fit = np.arange(WIDE_FIT_SIZE)
    summary = dict(status='SOURCE_TRAINED', fit_ids=WIDE_FIT_SIZE, **common)
    state = dict(schema=SCHEMA, fit_ids=fit, width=32, model=model.state_dict(), **common)
    roles = np.full(WIDE_FIT_SIZE+1, 'L_s')
    monkeypatch.setattr(diagnostic.core, 'read', lambda _: summary)
    monkeypatch.setattr(diagnostic.torch, 'load', lambda *args, **kwargs: state)
    loaded, loaded_fit = diagnostic.load_candidate(tmp_path, np.array([WIDE_FIT_SIZE]), roles)
    assert isinstance(loaded, DuraMonotoneEnvelope) and np.array_equal(loaded_fit, fit)
    state['TX_labels_read'] = 1
    with pytest.raises(ValueError, match='source-only'):
        diagnostic.load_candidate(tmp_path, np.array([WIDE_FIT_SIZE]), roles)
    state['TX_labels_read'] = 0
    with pytest.raises(ValueError, match='disjoint'):
        diagnostic.load_candidate(tmp_path, np.array([0]), roles)
