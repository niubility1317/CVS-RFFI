"""Real-constructor synthetic capture checks; no checkpoint or data access."""
import copy
import json

import torch

from .capture import forward, HEAD_PATH, SEQUENCE_PATH, REFERENCE_PATH
from experiments.cvs_sigtime_heldcredit.held_credit import CACHE_ATTRIBUTES
from experiments.cvs_u_schedule50.counterfactual import _rng_snapshot, _equal


def hooks(model):
    return [(module, tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
            for module in model.modules()]


def caches(model):
    return [(module, key, getattr(module, key)) for module in model.modules()
            for key in CACHE_ATTRIBUTES if hasattr(module, key)]


def check():
    from experiments.cvs_crossrx_sharpness20 import design as d, model as api
    torch.set_num_threads(2)
    c = d.config(next(row for row in d.rows() if row['arm'] == 'BASE_SSL'
                     and row['model_seed'] == d.SEEDS[0]))
    model = api.build(c, 'cpu').eval().requires_grad_(True)
    # Synthetic nonzero reference gain detects omission of the fourth real term.
    with torch.no_grad():
        model.id_backbone.encoder.response_gain.fill_(.1)
    generator = torch.Generator().manual_seed(151072)
    x = torch.randn(4, 2, 256, generator=generator)
    head = model.get_submodule(HEAD_PATH)
    pool = model.get_submodule(SEQUENCE_PATH)
    norm_values = {}
    h1 = head.base_norm.register_forward_hook(lambda m, i, o: norm_values.update(identity=o))
    h2 = head.pa_norm.register_forward_hook(lambda m, i, o: norm_values.update(physical=o))
    h3 = pool.output_norm.register_forward_hook(lambda m, i, o: norm_values.update(sequence_raw=o))
    h4 = model.get_submodule(REFERENCE_PATH).register_forward_hook(
        lambda m, i, o: norm_values.update(reference_raw=o))
    before = copy.deepcopy(model.state_dict())
    before_rng = _rng_snapshot()
    before_hooks, before_caches = hooks(model), caches(model)
    try:
        result = forward(model, x)
        assert set(result) == {'identity', 'physical', 'sequence_raw', 'reference_raw', 'joint', 'logits'}
        assert all(result[name] is value for name, value in norm_values.items())
        assert all(value.requires_grad for value in result.values())
        assert _equal(before, model.state_dict()) and _equal(before_rng, _rng_snapshot())
        assert all(tuple(m._forward_hooks) == ordinary and tuple(m._forward_pre_hooks) == pre
                   for m, ordinary, pre in before_hooks)
        assert all(getattr(m, key) is value for m, key, value in before_caches)
        reconstruction = ((result['identity'] + head.gain.tanh() * result['physical'])
                          + model.id_backbone.encoder.response_gain.tanh() * result['reference_raw'])
        reconstruction = reconstruction + (.2 * pool.gain_raw.tanh()) * result['sequence_raw']
        assert torch.equal(reconstruction, result['joint'])
        wrong_three_terms = result['identity'] + head.gain.tanh() * result['physical']
        wrong_three_terms = wrong_three_terms + (.2 * pool.gain_raw.tanh()) * result['sequence_raw']
        assert not torch.equal(wrong_three_terms, result['joint'])
        assert torch.autograd.grad(result['joint'].square().sum(), head.weight,
                                   retain_graph=True, allow_unused=True)[0] is None
        feature = pool.output_projection.weight
        gradient = torch.autograd.grad(result['sequence_raw'].square().sum(), feature,
                                       retain_graph=True)[0]
        assert bool(torch.isfinite(gradient).all()) and gradient.abs().sum() > 0
    finally:
        for handle in (h1, h2, h3, h4):
            handle.remove()
    # Remove W's influence on logits explicitly; the real joint remains identical.
    with torch.no_grad():
        head.weight.zero_()
    changed_w = forward(model, x)
    assert torch.equal(changed_w['joint'], result['joint'])
    assert torch.count_nonzero(changed_w['logits']) == 0
    assert not torch.equal(changed_w['logits'], result['logits'])
    assert all(p.grad is None for p in model.parameters())
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in teacher.modules())
    teacher_before = copy.deepcopy(teacher.state_dict())
    other = forward(teacher, x[:2])
    assert other['sequence_raw'].shape == (2, 160) and not other['joint'].requires_grad
    assert _equal(teacher_before, teacher.state_dict())
    assert result['sequence_raw'].shape == (4, 160)
    # Existing gradients are preserved by ordinary forward-only capture.
    head.weight.grad = torch.ones_like(head.weight)
    original_grad = head.weight.grad
    forward(model, torch.zeros(2, 2, 256))
    assert head.weight.grad is original_grad and torch.equal(original_grad, torch.ones_like(original_grad))
    head.weight.grad = None
    # A failing real branch must leave no call-local hook or cached sample behind.
    fail = model.get_submodule(REFERENCE_PATH)
    def injected(module, inputs, output):
        raise RuntimeError('synthetic branch failure')
    handle = fail.register_forward_hook(injected)
    before_hooks, before_caches = hooks(model), caches(model)
    try:
        try:
            forward(model, x)
        except RuntimeError as error:
            assert str(error) == 'synthetic branch failure'
        else:
            raise AssertionError('Expected branch failure')
        assert all(tuple(m._forward_hooks) == ordinary and tuple(m._forward_pre_hooks) == pre
                   for m, ordinary, pre in before_hooks)
        assert all(getattr(m, key) is value for m, key, value in before_caches)
    finally:
        handle.remove()
    model.train()
    try:
        forward(model, x)
    except ValueError:
        pass
    else:
        raise AssertionError('Training-mode caller was silently changed')
    assert model.training
    return dict(status='PASS', actual_SIG_TIME_CENTER_constructor=True,
                normalized_branches_direct_output=True, sequence_pre_gain_direct_hook=True,
                reference_fourth_term_exact=True, deployed_joint_bitwise_reconstruction=True,
                joint_W_dependency_None=True, changed_W_only_changes_logits=True,
                attached_feature_graph=True, parameters_buffers_gradients_RNG_caches_preserved=True,
                hooks_removed_on_success_and_exception=True, deepcopy_teacher_has_no_capture_hook=True,
                teacher_and_other_forward_isolated=True, training_flags_caller_owned=True,
                actual_data_or_checkpoint_reads=0, actual_optimizer_updates=0)


if __name__ == '__main__':
    print(json.dumps(check(), allow_nan=False, indent=2))
