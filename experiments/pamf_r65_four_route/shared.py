"""One received-IQ context decoder for both coarse IQ and physical action.

All weights come from the designated R65 ``prune_after10`` student.  Copying
the temporal trunk into the action path changes its context features, so this
initialization is deliberately not claimed to preserve the parent function.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.pamf_action_restore.models import KERNEL_LENGTH
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_r17_compact.models import stages
from experiments.pamf_r46_delayed_prune.models import CompactSuccessor
from experiments.pamf_reception_restore.models import ComplexTFUNet


class _SharedContext(nn.Module):
    """The parent's temporal encoder and decoder, with no unused output head."""

    def __init__(self, temporal):
        super().__init__()
        self.encoder = temporal.encoder
        self.decode = temporal.decode

    def forward(self, value):
        features = self.encoder(value)
        deepest = features[-1]
        value = deepest
        for block, skip in zip(self.decode, features[-2::-1]):
            value = block(torch.cat((F.interpolate(
                value, size=skip.shape[-1], mode='linear', align_corners=False), skip), 1))
        return value, deepest


class SharedContextRestorer(nn.Module):
    """Trainable shared context plus the original three physical prior stages.

    ``parent=None`` only constructs the matching random shape skeleton used
    for strict standalone state loading.  It does not open a checkpoint or IQ.
    """

    def __init__(self, parent=None):
        super().__init__()
        supplied_parent = parent is not None
        if parent is None:
            parent = CompactSuccessor('prune_after10')
        if (not isinstance(parent, CompactSuccessor)
                or parent.arm != 'prune_after10'
                or sum(p.numel() for p in parent.parameters()) != 273765):
            raise ValueError('Only the designated compact R65 prune_after10 parent is accepted')

        # A single deepcopy preserves all existing cross-prior weight aliases
        # while ensuring no student parameter or buffer belongs to the teacher.
        spectral, temporal, action_head, global_head, prior, impulse = copy.deepcopy((
            parent.frozen.coarse.spectral,
            parent.frozen.coarse.temporal,
            parent.frozen.estimator.core.out,
            parent.frozen.estimator.global_head,
            parent.prior,
            parent.frozen.estimator.initial_kernel,
        ))
        self.arm = 'shared_context'
        self.spectral = spectral
        self.shared = _SharedContext(temporal)
        self.coarse_head = temporal.out
        self.action_head = action_head
        self.global_head = global_head
        self.prior = prior
        self.register_buffer('initial_kernel', impulse)
        self.requires_grad_(True)
        self.train()

        parameters = sum(p.numel() for p in self.parameters())
        self.initialization_report = dict(
            mode=('R65_weight_copy_shared_context' if supplied_parent
                  else 'random_shape_skeleton_for_strict_state_load'),
            parent_arm='prune_after10', parent_parameters=273765,
            initialized_from_supplied_parent=supplied_parent,
            parameters=parameters,
            copied_parameters=parameters if supplied_parent else 0,
            new_parameters=0 if supplied_parent else parameters,
            new_architecture_parameters=0,
            discarded_parameters=273765-parameters,
            shared_trunk_source='parent.frozen.coarse.temporal.encoder_and_decode',
            coarse_head_source='parent.frozen.coarse.temporal.out',
            action_head_source='parent.frozen.estimator.core.out',
            global_head_source='parent.frozen.estimator.global_head',
            spectral_source='parent.frozen.coarse.spectral',
            priors_source='three_parent_priors_copied_together_with_long_weight_aliases',
            context_input='received_and_spectral_observation',
            parent_action_input='received_and_coarse',
            context_encoder_calls=1, context_decoder_calls=1,
            exact_initial_function_equivalence=False,
            initialization_limitation='Action heads now receive coarse temporal features; source retraining is required',
            parent_unchanged=True, all_student_parameters_trainable=True,
            IQ_examples=0, TX_labels_read=0, F_reads=0, V_reads=0,
            target_reads=0, optimizer_updates=0, new_calibration_IQ=0,
        )

    def spectral_observation(self, received):
        # This existing implementation depends only on self.spectral.  Keeping
        # it avoids changing the parent's widely-linear spectral observation.
        return ComplexTFUNet.spectral_observation(self, received)

    def estimate_context(self, received):
        if received.ndim != 3 or received.shape[1:] != (2, 256):
            raise ValueError('Only current received IQ2x256 is accepted')
        spectral = self.spectral_observation(received)
        decoded, deepest = self.shared(torch.cat((received, spectral), 1))
        base = received+self.coarse_head(decoded)
        coefficients = self.action_head(decoded)
        global_value = self.global_head(deepest.mean(-1))
        kernel = torch.complex(
            global_value[:, :KERNEL_LENGTH]+self.initial_kernel,
            global_value[:, KERNEL_LENGTH:2*KERNEL_LENGTH])
        kernel = kernel/kernel.abs().square().sum(-1, keepdim=True).sqrt().clamp_min(1e-6)
        main = 1+torch.complex(coefficients[:, 0], coefficients[:, 1])
        image = torch.complex(coefficients[:, 2], coefficients[:, 3])
        log_variance = math.log(1e-8)+(math.log(1.)-math.log(1e-8))*torch.sigmoid(
            global_value[:, -1]+.351)
        return base, Action(kernel, main, image), log_variance.exp()

    def stage_outputs(self, received):
        return stages(self.prior, received, self.estimate_context(received))

    def training_outputs(self, received):
        output, variance, action, base = self.stage_outputs(received)
        return output[-1], variance, action, base

    def forward(self, received):
        return self.training_outputs(received)[0]
