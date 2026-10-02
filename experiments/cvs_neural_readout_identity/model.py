"""CE-learned, packet-local readouts on a fresh shallow neural CVS backbone.

The original invariant observations remain the complete skip path. Attention
uses the full received packet offline; this is not a causal streaming model or
an arbitrary receiver/channel invariant representation.
"""
from contextlib import contextmanager

import torch
from torch import nn

from experiments.cvs_equivariant_identity.model import ComplexConv, InvariantReadout
from experiments.cvs_neural_residual_identity.model import NeuralResidualCVS, neural_contract


VARIANTS = ('readout_attention', 'readout_complex_attention')
BASE_VARIANT = 'neural_residual_shallow'
BASE_PARAMETERS = 220987


class LearnedInvariantReadout(nn.Module):
    """Independent temporal attention heads over one packet's log powers."""

    def __init__(self, channel_mixing=False):
        super().__init__()
        self.skip = InvariantReadout()
        if channel_mixing:
            # Do not shift paired score/project initialization across variants.
            with torch.random.fork_rng(devices=[]):
                self.mixer = ComplexConv(32, 32, 1)
            with torch.no_grad():
                self.mixer.weight_real.copy_(torch.eye(32).unsqueeze(-1))
                self.mixer.weight_imag.zero_()
        else:
            self.mixer = nn.Identity()
        self.scores = nn.Sequential(
            nn.Conv1d(32, 16, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv1d(16, 4, 1, bias=True),
        )
        self.attention = nn.Softmax(dim=-1)
        self.project = nn.Linear(128, 320, bias=False)
        with torch.no_grad():
            self.project.weight.zero_()

    def forward(self, z):
        if z.ndim != 4 or z.shape[1:] != (2, 32, 64):
            raise ValueError('Expected complex readout input [B,2,32,64]')
        values = torch.log1p(self.mixer(z).square().sum(1))
        weights = self.attention(self.scores(values))
        pooled = torch.einsum('bht,bct->bhc', weights, values).flatten(1)
        return self.skip(z) + self.project(pooled)


def readout_contract(variant):
    if variant not in VARIANTS:
        raise ValueError('Unregistered neural readout architecture')
    mixing = variant == 'readout_complex_attention'
    d = neural_contract(BASE_VARIANT)
    d.update(
        mode=variant,
        identity_core='neural_readout_identity',
        base_variant=BASE_VARIANT,
        base_trainable_parameters=BASE_PARAMETERS,
        new_trainable_parameters=89256 if mixing else 85160,
        learned_readout_active=True,
        readout_paths=['time', 'behavior'],
        attention_heads=4,
        attention_hidden=16,
        attention_kernel=3,
        attention_normalization='softmax over 64 temporal positions within each packet only',
        attention_values='log1p(complex power)',
        output_dimension=320,
        readout_channel_mixing=mixing,
        readout_channel_mixer_initialization='complex identity' if mixing else 'none',
        readout_skip='complete original InvariantReadout with 320 outputs',
        readout_rule='InvariantReadout(z)+Linear(concat_heads(sum_time(attention*logpower)))',
        readout_exit_initialization='zero',
        readout_scope='full packet offline aggregation; no cross-packet fit or state',
        initial_function='exact own scratch neural_residual_shallow; isolated new initialization RNG',
    )
    return d


def _matches_readout(block, mixing):
    """Check the instantiated operators, not just the variant label."""
    if not isinstance(block, LearnedInvariantReadout) or not isinstance(block.skip, InvariantReadout):
        return False
    if not isinstance(block.scores, nn.Sequential) or len(block.scores) != 3:
        return False
    first, activation, last = block.scores
    if not isinstance(first, nn.Conv1d) or not isinstance(last, nn.Conv1d):
        return False
    if not (first.in_channels == 32 and first.out_channels == 16 and first.kernel_size == (3,)
            and first.padding == (1,) and first.stride == (1,) and first.dilation == (1,)
            and first.groups == 1 and first.bias is not None and first.padding_mode == 'zeros'
            and isinstance(activation, nn.GELU) and activation.approximate == 'none'
            and last.in_channels == 16 and last.out_channels == 4 and last.kernel_size == (1,)
            and last.padding == (0,) and last.stride == (1,) and last.dilation == (1,)
            and last.groups == 1 and last.bias is not None
            and isinstance(block.attention, nn.Softmax) and block.attention.dim == -1
            and isinstance(block.project, nn.Linear) and block.project.in_features == 128
            and block.project.out_features == 320 and block.project.bias is None):
        return False
    if not mixing:
        return isinstance(block.mixer, nn.Identity)
    return (isinstance(block.mixer, ComplexConv) and block.mixer.cin == 32 and block.mixer.cout == 32
            and block.mixer.k == 1 and block.mixer.stride == 1 and block.mixer.dilation == 1
            and not block.mixer.causal)


@contextmanager
def _diagnostic_mode(model, x):
    """Read diagnostics without changing training flags, RNG, or running state."""
    flags = [(module, module.training) for module in model.modules()]
    try:
        with torch.random.fork_rng(devices=[x.get_device()] if x.is_cuda else []):
            model.eval()
            yield
    finally:
        for module, training in flags:
            module.training = training


class NeuralReadoutCVS(NeuralResidualCVS):
    def __init__(self, variant):
        contract = readout_contract(variant)
        super().__init__(BASE_VARIANT)
        self.readout_variant = variant
        # Existing scratch tensors and subsequent CPU randomness stay identical.
        with torch.random.fork_rng(devices=[]):
            for path in ('time', 'behavior'):
                setattr(self.core, path + '_readout',
                        LearnedInvariantReadout(contract['readout_channel_mixing']))

    def readout_blocks(self):
        return [(path + '.readout', getattr(self.core, path + '_readout'))
                for path in ('time', 'behavior')]

    def readout_parameters(self):
        return [parameter for _, block in self.readout_blocks() for parameter in block.parameters()]

    def contract(self):
        d = readout_contract(self.readout_variant)
        blocks = [block for _, block in self.readout_blocks()]
        new_parameters = self.readout_parameters()
        new_count = sum(p.numel() for p in new_parameters if p.requires_grad)
        new_ids = {id(p) for p in new_parameters}
        d.update(
            neural_residual_active=super().contract()['neural_residual_active'],
            learned_readout_active=(blocks[0] is not blocks[1]
                                    and all(_matches_readout(b, d['readout_channel_mixing']) for b in blocks)
                                    and not ({id(p) for p in blocks[0].parameters()}
                                             & {id(p) for p in blocks[1].parameters()})),
            base_trainable_parameters=sum(p.numel() for p in self.parameters()
                                          if p.requires_grad and id(p) not in new_ids),
            new_trainable_parameters=new_count,
            actual_phase_lag=self.core.behavior[0].phase_lag,
            actual_envelope_lag=self.core.behavior[0].envelope_lag,
            actual_mixture_parameter_shape=list(self.core.behavior[0].mix_raw.shape),
        )
        return d

    @torch.no_grad()
    def readout_diagnostics(self, x):
        records, hooks, captured = [], [], {}

        def capture_attention(name):
            def hook(module, inputs, output):
                captured[name] = output.detach()
            return hook

        def capture_output(name):
            def hook(module, inputs, output):
                before = module.skip(inputs[0])
                difference = (output - before).norm(dim=1)
                relative = difference / before.norm(dim=1).clamp_min(1e-12)
                weights = captured.pop(name)
                entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
                records.append(dict(
                    block=name,
                    packets=len(output),
                    input_shape=list(inputs[0].shape),
                    output_dimension=output.shape[-1],
                    relative_output_change_mean=float(relative.mean()),
                    projection_norm=float(module.project.weight.norm()),
                    attention_entropy_mean=float(entropy.mean()),
                    attention_effective_tokens_mean=float(entropy.exp().mean()),
                    attention_sum_max_abs_error=float((weights.sum(-1) - 1).abs().max()),
                    attention_heads=weights.shape[1],
                    attention_tokens=weights.shape[2],
                    channel_mixing=isinstance(module.mixer, ComplexConv),
                ))
            return hook

        try:
            for name, block in self.readout_blocks():
                hooks.append(block.attention.register_forward_hook(capture_attention(name)))
                hooks.append(block.register_forward_hook(capture_output(name)))
            with _diagnostic_mode(self, x):
                self.features(x)
        finally:
            for hook in hooks:
                hook.remove()
        return dict(active=self.contract()['learned_readout_active'], records=records,
                    scope='last source batch; actual readout hooks; no additional loss or updates')

    @torch.no_grad()
    def diagnostics(self, x):
        with _diagnostic_mode(self, x):
            return dict(super().diagnostics(x), learned_readout=self.readout_diagnostics(x))


def build(variant):
    return NeuralReadoutCVS(variant)
