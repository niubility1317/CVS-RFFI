"""Adapt the user-fixed reference_response architecture to the native dual API.

Only the identity network is replaced. Domain heads and the nuisance network
remain native. The adapter never loads a checkpoint or consumes target data.
"""
from contextlib import contextmanager
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / 'experiments/adv3b02_xuc/code'
VARIANT = 'reference_response'


def native_modules():
    if str(NATIVE) not in sys.path:
        sys.path.insert(0, str(NATIVE))
    from SSDG import train_ssdg
    return train_ssdg


class Phase1IdentityAdapter(nn.Module):
    """Real branch features, unchanged inference, original MixStyle locations."""
    emb_dim = 160

    def __init__(self, model_args):
        super().__init__()
        if int(model_args.num_classes) != 6 or int(model_args.input_len) != 256:
            raise ValueError('Selected research network requires six classes and 256 IQ samples')
        if getattr(model_args, 'use_crra', False) or getattr(model_args, 'use_a1_r3', False):
            raise ValueError('CRRA/R3 are not registered for this replacement')
        from experiments.cvs_reference_identity.model import build
        self.encoder = build(VARIANT)
        from model import MixStyle1D
        self.mixstyle_on = bool(getattr(model_args, 'use_mixstyle', False))
        self.mixstyle_layers = set(str(getattr(model_args, 'mixstyle_layers', 'time_down,t1')).split(','))
        if self.mixstyle_layers - {'time_down', 't1'}:
            raise ValueError('Unregistered MixStyle location')
        self.mixstyle = MixStyle1D(
            p=float(getattr(model_args, 'mixstyle_p', .18)),
            alpha=float(getattr(model_args, 'mixstyle_alpha', .1)),
            eps=float(getattr(model_args, 'mixstyle_eps', 1e-6)),
            mix=str(getattr(model_args, 'mixstyle_mix', 'same_tx_crossdomain')),
            strength=float(getattr(model_args, 'mixstyle_strength', .7)),
            fallback=str(getattr(model_args, 'mixstyle_fallback', 'skip')))
        self.mixstyle_use_domain_label = bool(getattr(model_args, 'mixstyle_use_domain_label', True))
        self.last_mixstyle_calls = 0
        self.last_mixstyle_changed_samples = 0

    def contract(self):
        return dict(variant=VARIANT, architecture=self.encoder.contract(),
                    identity_parameters=sum(p.numel() for p in self.parameters()),
                    mixstyle_enabled=self.mixstyle_on, mixstyle_layers=sorted(self.mixstyle_layers),
                    mixstyle_p=self.mixstyle.p, mixstyle_alpha=self.mixstyle.alpha,
                    mixstyle_eps=self.mixstyle.eps, mixstyle_strength=self.mixstyle.strength,
                    mixstyle_mix=self.mixstyle.mix, mixstyle_fallback=self.mixstyle.fallback)

    def forward(self, x, y=None, return_aux=False, domain_labels=None):
        # All hooks are call-local, removed even if the forward fails. Deepcopy
        # for the EMA therefore cannot retain closures bound to another model.
        hooks, branches = [], {}
        self.last_mixstyle_calls = 0
        self.last_mixstyle_changed_samples = 0
        head = self.encoder.core.id_backbone.cls_head
        if return_aux:
            hooks.append(head.base_norm.register_forward_hook(
                lambda module, inputs, output: branches.update(identity=output)))
            hooks.append(head.pa_norm.register_forward_hook(
                lambda module, inputs, output: branches.update(physical=output)))
        if self.training and self.mixstyle_on:
            def mix(module, inputs, output):
                self.last_mixstyle_calls += 1
                mixed = self.mixstyle(output,
                    domain_labels if self.mixstyle_use_domain_label else None,
                    tx_labels=y)
                self.last_mixstyle_changed_samples += int((mixed.detach() != output.detach()).flatten(1).any(1).sum())
                return mixed
            for label in ('time_down', 't1'):
                if label in self.mixstyle_layers:
                    hooks.append(getattr(self.encoder.core.id_backbone, label).register_forward_hook(mix))
        try:
            # FFT/correlation must retain the architecture's FP32 path
            # even if the native training loop autocasts its other branches.
            with torch.autocast(device_type=x.device.type, enabled=False):
                z = self.encoder.features(x.float())
                logits = self.encoder.classify_features(z)
        finally:
            for hook in hooks:
                hook.remove()
        if not return_aux:
            return logits
        if set(branches) != {'identity', 'physical'}:
            raise RuntimeError('Selected architecture bypassed branch-feature capture')
        return dict(logits=logits, feat_joint=z, feat_cls=branches['identity'],
                    feat_imp=branches['physical'], feat_pa=branches['physical'],
                    feat_dac=torch.zeros_like(z), feat_con=z, base=branches['identity'],
                    arch_family=VARIANT)


@contextmanager
def installed_identity(variant, seed):
    """Scope the replacement to one worker, including strict checkpoint reload."""
    if variant not in ('native', VARIANT):
        raise ValueError('Unregistered identity architecture')
    native = native_modules()
    original = native.build_baseline_model

    def build(args, device):
        model = original(args, device)
        if variant != 'native':
            # Keep nuisance-network initialization paired across mechanisms and
            # construct the identity with the historical architecture RNG seed.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                model.id_backbone = Phase1IdentityAdapter(args).to(device)
            if model.emb_dim != 160:
                raise ValueError('Native domain/identity head dimension mismatch')
        return model

    native.build_baseline_model = build
    try:
        yield native
    finally:
        native.build_baseline_model = original
