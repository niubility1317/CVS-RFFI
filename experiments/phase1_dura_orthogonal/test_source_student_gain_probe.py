import numpy as np
import pytest
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_orthogonal.direct_spectral_student import DirectSpectralStudent
from experiments.phase1_dura_orthogonal.source_student_gain_probe import load_student
from experiments.phase1_dura_orthogonal.train_source import SCHEMA


def test_student_loader_rejects_target_ancestry_and_pilot_overlap(monkeypatch, tmp_path):
    expected = dict(commit='source-commit', architecture='direct_spectral_student', objective='distill_iq',
        epochs=20, DURA_training_reads=1, source_iq_teacher='dura_v1_R65_E20_4940',
        source_iq_teacher_weight=.5, inference_received_iq_only=True, TX_labels_read=0,
        F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0, DURA_frozen=True)
    summary = dict(expected, status='SOURCE_TRAINED', fit_ids=8192)
    state = dict(expected, schema=SCHEMA, fit_ids=np.arange(8192), width=32,
                 model=DirectSpectralStudent().state_dict())
    monkeypatch.setattr(core, 'read', lambda _: summary)
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: state)
    load_student(tmp_path, 'run', 'source-commit', 'distill_iq', 8192, np.array([9000]))
    summary['target_reads'] = 1
    with pytest.raises(ValueError):
        load_student(tmp_path, 'run', 'source-commit', 'distill_iq', 8192, np.array([9000]))
    summary['target_reads'] = 0
    with pytest.raises(ValueError):
        load_student(tmp_path, 'run', 'source-commit', 'distill_iq', 8192, np.array([12]))
