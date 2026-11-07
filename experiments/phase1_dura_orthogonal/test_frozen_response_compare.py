import numpy as np
import pytest
import torch

from experiments.phase1_dura_orthogonal import frozen_response_compare as probe


def test_exact_identity_response_through_estimated_action_and_three_methods():
    class Action:
        def __call__(self, value):
            return value

        def adjoint(self, value):
            return value

    class Dura:
        def estimate_context(self, x):
            return None, Action(), torch.ones(len(x))

        def stage_outputs(self, x, context):
            return [x], None, context[1], None

    class Model:
        def __call__(self, x, anchor, innovation, variance):
            return anchor

    torch.manual_seed(3)
    x = torch.randn(3, 2, 256)
    metrics, nmse = probe.evaluate_batch(Dura(), [Model(), Model()], x, x)
    assert metrics.shape == (3, 4, 3, 3)
    torch.testing.assert_close(metrics[..., 0], torch.ones(3, 4, 3), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(metrics[..., 1], torch.zeros(3, 4, 3), rtol=0, atol=1e-6)
    torch.testing.assert_close(nmse, torch.zeros(3, 3))


def test_fixed_models_rejects_source_that_read_classifier(tmp_path, monkeypatch):
    from experiments.pamf_phase1_flow import experiment as core
    from experiments.phase1_dura_orthogonal.train_source import SCHEMA
    common = dict(commit=probe.SOURCES[0][1], architecture='dura_multiscale',
                  objective='wide_source_guard', epochs=5, source_fit_size=32768,
                  DURA_training_reads=1, DURA_frozen=True, TX_labels_read=0,
                  F_weights_read=1, F_outputs_read=0, V_reads=0, target_reads=0)
    summary = dict(common, status='SOURCE_TRAINED', fit_ids=32768)
    state = dict(common, schema=SCHEMA, fit_ids=np.arange(32768))
    monkeypatch.setattr(core, 'read', lambda path: summary)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: state)
    with pytest.raises(ValueError, match='provenance'):
        probe.fixed_models(tmp_path, np.array([40000]), 'cpu')
