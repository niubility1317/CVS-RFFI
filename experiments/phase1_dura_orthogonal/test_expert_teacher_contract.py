import numpy as np
import pytest
import torch

from experiments.phase1_dura_orthogonal.expert_teacher_contract import (
    EXPERT_COMMIT, EXPERT_SOURCE, TEACHER_COMMIT, TEACHER_SCHEMA, FIT_SIZE,
    validate_expert_checkpoint, validate_teacher_cache, select_teacher_views)


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


def test_smoke_teacher_preserves_view_major_pairing():
    values = torch.arange(15)
    assert select_teacher_views(values, 5, 2).tolist() == [0, 1, 5, 6, 10, 11]
    with pytest.raises(ValueError):
        select_teacher_views(values, 5, 6)


def test_teacher_cache_rejects_target_data_and_wrong_view_contract():
    fit = np.arange(FIT_SIZE)
    flags = dict(TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0)
    result = dict(flags, status='SOURCE_EXPERT_TEACHER_COMPLETE', commit=TEACHER_COMMIT,
                  exposure=3*FIT_SIZE, fit_ids=FIT_SIZE, model_updates=0)
    contract = dict(flags, schema=TEACHER_SCHEMA, source=EXPERT_SOURCE, source_commit=EXPERT_COMMIT,
                    epochs=[21, 22, 23], context_cache='fixed-cache', deterministic_algorithms=True,
                    fit_ids=fit.tolist())
    validate_teacher_cache(result, contract, fit, 'fixed-cache')
    validate_teacher_cache(result, contract, fit[:64], 'fixed-cache', smoke=True)
    with pytest.raises(ValueError):
        validate_teacher_cache(result, dict(contract, target_reads=1), fit, 'fixed-cache')
    with pytest.raises(ValueError):
        validate_teacher_cache(result, dict(contract, epochs=[23, 22, 21]), fit, 'fixed-cache')
    with pytest.raises(ValueError):
        validate_teacher_cache(result, contract, fit[::-1], 'fixed-cache')
