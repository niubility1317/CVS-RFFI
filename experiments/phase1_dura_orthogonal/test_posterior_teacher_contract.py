import numpy as np
import pytest
import torch

from experiments.phase1_dura_orthogonal.posterior_teacher_contract import (
    TEACHER_COMMIT, TEACHER_SCHEMA, COVARIANCE_SOURCE, COVARIANCE_COMMIT, CONTEXT_SOURCE,
    validate_posterior_teacher, select_teacher_views)
from experiments.phase1_dura_orthogonal.train_source import posterior_teacher_loss


def test_teacher_provenance_and_fit_are_required():
    fit = np.arange(8192)
    result = dict(status='SOURCE_POSTERIOR_TEACHER_COMPLETE', commit=TEACHER_COMMIT,
                  exposure=24576, model_updates=0)
    contract = dict(schema=TEACHER_SCHEMA, fit_ids=fit.tolist(), epochs=[21, 22, 23],
        covariance_source=COVARIANCE_SOURCE, covariance_commit=COVARIANCE_COMMIT,
        context_source=CONTEXT_SOURCE, cg_steps=160, positive_limit=.5,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0)
    validate_posterior_teacher(result, contract, fit)
    validate_posterior_teacher(result, contract, fit[:64], smoke=True)
    with pytest.raises(ValueError):
        validate_posterior_teacher(result, contract, fit[::-1])
    with pytest.raises(ValueError):
        validate_posterior_teacher(result, dict(contract, target_reads=1), fit)
    with pytest.raises(ValueError):
        validate_posterior_teacher(dict(result, commit='unverified'), contract, fit)


def test_smoke_view_subset_preserves_view_major_order():
    values = torch.arange(12)
    assert select_teacher_views(values, full_count=4, selected_count=2).tolist() == [0, 1, 4, 5, 8, 9]


def test_teacher_iq_loss_is_normalized_by_clean_energy():
    clean = 2*torch.ones(2, 2, 256)
    teacher = 3*torch.ones_like(clean)
    output = 5*torch.ones_like(clean)
    torch.testing.assert_close(posterior_teacher_loss(output, teacher, clean), torch.tensor(1.))
