import numpy as np
import pytest

from experiments.phase1_dura_orthogonal.expert_teacher_contract import (
    EXPERT_COMMIT, FIT_SIZE, validate_expert_checkpoint)


def fixture():
    common = dict(commit=EXPERT_COMMIT, architecture='dura_experts', objective='wide_mixture_guard',
        epochs=5, source_fit_size=FIT_SIZE, source_expert_count=3, deterministic_algorithms=True,
        DURA_training_reads=1, DURA_frozen=True, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_reads=0)
    fit = np.arange(FIT_SIZE)
    return dict(common, status='SOURCE_TRAINED', fit_ids=FIT_SIZE), dict(
        common, schema='phase1_dura_orthogonal_source_v1', fit_ids=fit), fit


def test_teacher_rejects_target_ancestry_and_wrong_source_ids():
    summary, state, fit = fixture()
    validate_expert_checkpoint(summary, state, fit)
    validate_expert_checkpoint(summary, state, fit[:64])
    contaminated = dict(state, target_reads=1)
    with pytest.raises(ValueError):
        validate_expert_checkpoint(summary, contaminated, fit)
    with pytest.raises(ValueError):
        validate_expert_checkpoint(summary, state, fit[::-1])
