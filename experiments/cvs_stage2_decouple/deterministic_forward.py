"""Dropout-only deterministic identity forwards for differentiable mechanisms.

Use one context around both clean and perturbed identity forwards. The context
never calls model.eval(), changes parameter flags, or disables autograd. The
ordinary CE forward remains outside this context in its original train mode.
This helper supports the registered reference_response identity path; it does
not silently normalize a new stochastic architecture into that contract.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn


DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
                 nn.AlphaDropout, nn.FeatureAlphaDropout)
COUNTERS = ('last_mixstyle_calls', 'last_mixstyle_changed_samples')


class DeterministicIdentityContractError(RuntimeError):
    """The active identity graph differs from the dropout-only contract."""


def _fail(message):
    raise DeterministicIdentityContractError('DETERMINISTIC_IDENTITY_CONTRACT_MISMATCH: ' + message)


def _identity(model):
    identity = getattr(model, 'id_backbone', None)
    if not isinstance(identity, nn.Module):
        _fail('expected model.id_backbone module')
    # The native forward_identity_only method has these optional paths outside
    # id_backbone. They are absent in the registered stage2 parent. Reject them
    # instead of leaving an uninspected stochastic correction head active.
    if getattr(model, 'representation_mode', None) == 'single_parameter_matched':
        _fail('unregistered identity_capacity correction is active')
    for name in ('sat_anchor_identity_adapter', 'a1_r3', 'a1_response'):
        if getattr(model, name, None) is not None:
            _fail('unregistered identity correction is active: ' + name)
    return identity


def identity_mode_contract(model, require_dropout_disabled=False):
    """Inspect actual module state; dormant MixStyle objects are not execution.

    Production evidence: Phase1IdentityAdapter.forward gates MixStyle on its
    training/mixstyle_on flags; native CVSCore._apply_mixstyle gates its child
    on mixstyle_on; branch_drop is gated on its own training flag/probability.
    ReferenceCVS/ResidualPhysicalHead use LayerNorm and the native backbone
    uses GroupNorm. The running-stat check catches later BN/IN introductions.
    """
    identity = _identity(model)
    modules = dict(identity.named_modules())
    dropout, dormant_mixstyle, running_statistics = [], [], []
    for name, module in modules.items():
        label = 'id_backbone' + ('.' + name if name else '')
        if isinstance(module, DROPOUT_TYPES):
            if require_dropout_disabled and module.training:
                _fail('dropout re-enabled inside deterministic scope: ' + label)
            dropout.append(dict(name=label, type=type(module).__name__, p=float(module.p),
                                original_training=bool(module.training)))
        if hasattr(module, 'branch_drop_p'):
            probability = float(module.branch_drop_p)
            if not math.isfinite(probability) or not 0. <= probability <= 1.:
                _fail('invalid branch_drop_p at ' + label)
            if module.training and probability > 0:
                _fail('active branch_drop at ' + label)
        if module.training and bool(getattr(module, 'mixstyle_on', False)):
            _fail('enabled train-mode MixStyle at ' + label)
        if 'mixstyle' in type(module).__name__.lower():
            parent_name = name.rpartition('.')[0]
            parent = modules.get(parent_name) if name else None
            # Only the native named child has an explicit owner-side gate.
            gated_off = (parent is not None and getattr(parent, 'mixstyle', None) is module
                         and hasattr(parent, 'mixstyle_on') and not parent.mixstyle_on)
            enabled = (module.training and float(getattr(module, 'p', 1.)) > 0
                       and float(getattr(module, 'alpha', 1.)) > 0
                       and float(getattr(module, 'strength', 1.)) > 0)
            if enabled and not gated_off:
                _fail('active or unregistered MixStyle module at ' + label)
            dormant_mixstyle.append(label)
        is_statistical = isinstance(module, (nn.modules.batchnorm._BatchNorm,
                                             nn.modules.instancenorm._InstanceNorm))
        if is_statistical:
            if module.training and module.track_running_stats:
                _fail('active running-stat update at ' + label)
            running_statistics.append(dict(name=label, type=type(module).__name__,
                                            training=bool(module.training),
                                            track_running_stats=bool(module.track_running_stats)))
        # These built-in modules implement functional dropout rather than child
        # Dropout modules; a later architecture must not silently bypass scope.
        if (isinstance(module, (nn.RNNBase, nn.MultiheadAttention)) and module.training
                and float(getattr(module, 'dropout', 0.)) > 0):
            _fail('functional attention/recurrent dropout at ' + label)
    return dict(status='IDENTITY_MODE_CONTRACT_VALID', identity_scope='model.id_backbone',
                dropout_modules=dropout, dormant_mixstyle_modules=dormant_mixstyle,
                statistical_modules=running_statistics, active_branch_drop=False,
                active_mixstyle=False, running_statistics_updated=False,
                changes='Dropout-family training flags only; all other module modes retained')


@contextmanager
def deterministic_identity(model):
    """Keep clean/aug autograd while disabling only identity Dropout modules.

    Nested scopes restore to their entry states. Exceptions restore the original
    per-module mixed modes and adapter counters. Caller-created autograd graphs,
    parameter requires_grad flags, buffers, domain modes and RNG are untouched.
    Autograd must already be enabled: accepting an accidental no_grad caller
    would hide a disconnected auxiliary loss.
    """
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        _fail('mechanism forward requires a grad-enabled caller')
    contract = identity_mode_contract(model)
    identity = _identity(model)
    modes = [(module, module.training) for module in identity.modules()]
    counters = [(module, key, getattr(module, key)) for module in identity.modules()
                for key in COUNTERS if hasattr(module, key)]
    handles = []
    try:
        for module, _ in modes:
            if isinstance(module, DROPOUT_TYPES):
                module.training = False

        def check_entry(module, arguments):
            if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
                _fail('identity forward entered no_grad/inference mode inside deterministic scope')
            identity_mode_contract(model, require_dropout_disabled=True)

        def check_exit(module, arguments, output):
            if any(getattr(owner, key) != 0 for owner, key, _ in counters):
                _fail('adapter reports a MixStyle call during deterministic identity forward')

        handles.append(identity.register_forward_pre_hook(check_entry))
        handles.append(identity.register_forward_hook(check_exit))
        yield contract
    finally:
        for handle in handles:
            handle.remove()
        # Assign individual flags. Calling train(flag) recursively would destroy
        # a caller's intentionally mixed train/eval configuration.
        for module, training in modes:
            module.training = training
        for module, key, value in counters:
            setattr(module, key, value)
