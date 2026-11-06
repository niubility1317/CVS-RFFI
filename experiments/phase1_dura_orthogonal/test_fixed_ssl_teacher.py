import numpy as np
import pytest

from experiments.phase1_dura_orthogonal.train_source import (
    FIT_SIZE, SSL_TEACHER_COMMIT, validate_ssl_teacher)


def test_teacher_must_match_source_contract_and_fit_order():
    common = dict(commit=SSL_TEACHER_COMMIT, objective='ssl_joint_guard', architecture='dura_multiscale',
                  TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0,
                  target_reads=0, DURA_frozen=True, source_ssl_seed=2026110615)
    fit = np.arange(FIT_SIZE)
    summary = dict(common, status='SOURCE_TRAINED')
    state = dict(common, fit_ids=fit, source_ssl_aux_state={})
    validate_ssl_teacher(summary, state, fit)
    validate_ssl_teacher(summary, state, fit[:64], smoke=True)
    with pytest.raises(ValueError):
        validate_ssl_teacher(summary, state, fit[:64])
    with pytest.raises(ValueError):
        validate_ssl_teacher(summary, dict(state, target_reads=1), fit)
    with pytest.raises(ValueError):
        validate_ssl_teacher(summary, state, fit[::-1])
