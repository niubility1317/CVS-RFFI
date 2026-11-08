import pytest
import torch
from experiments.phase1_residual_diffusion.source_probe import (normalized_error, check_metadata, evaluate,
    JOINT_SCHEMA, JOINT_COMMIT)


def test_normalized_error_and_nonfinite_rejection():
    clean = torch.ones(2, 2, 256)
    error = normalized_error(clean*2, clean)
    torch.testing.assert_close(error, torch.ones(2))
    clean[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='Finite'):
        normalized_error(clean, clean)


def test_metadata_rejects_contaminated_or_wrong_checkpoint():
    good = dict(schema=JOINT_SCHEMA, commit=JOINT_COMMIT, epochs=20, fit_ids=32768,
        actual_source_view_count=98304, views_per_record=3, width=192, steps=3,
        inference_particles=4, TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_reads=0, DURA_frozen=True)
    summary = dict(good, status='SOURCE_TRAINED')
    check_metadata(summary, good, schema=JOINT_SCHEMA, commit=JOINT_COMMIT, steps=3)
    with pytest.raises(ValueError, match='checkpoint'):
        check_metadata(summary, dict(good, target_reads=1), schema=JOINT_SCHEMA, commit=JOINT_COMMIT, steps=3)
    with pytest.raises(ValueError, match='checkpoint'):
        check_metadata(summary, dict(good, steps=8), schema=JOINT_SCHEMA, commit=JOINT_COMMIT, steps=3)


def test_same_anchor_and_input_used_for_both_candidates():
    class Action:
        def __call__(self, z): return z
        def adjoint(self, z): return z
    class Dura:
        def estimate_context(self, x): return (None, Action(), torch.ones(len(x)))
        def stage_outputs(self, x, context): return ([x], None, None, None)
    seen = []
    def joint(y, anchor, direction, variance, action):
        seen.append(('joint', y.clone(), anchor.clone()))
        return anchor+1
    def diffusion(y, anchor, direction, variance):
        seen.append(('diffusion', y.clone(), anchor.clone()))
        return anchor-1
    received = torch.ones(2, 2, 256)
    values = evaluate(Dura(), joint, diffusion, received, received)
    assert torch.equal(values['dura'], torch.zeros(2))
    assert torch.equal(values['joint'], torch.ones(2))
    assert torch.equal(values['diffusion'], torch.ones(2))
    assert [x[0] for x in seen] == ['joint', 'diffusion']
    assert all(torch.equal(y, received) and torch.equal(anchor, received) for _, y, anchor in seen)
