"""Three original R65 priors with one analytic block-PCG step per stage.

The I/Q blocks below are the exact diagonal blocks of the real-linear normal
matrix, up to floating-point FFT error.  They are not a diagonal approximation
of the complex-linear part alone: the image term and reflected-tap collisions
both contribute.  Off-time normal blocks are omitted by block Jacobi.
"""
import copy
import math
from typing import NamedTuple

import torch
from torch import nn

from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_action_restore.operators import real_dot
from experiments.pamf_r46_delayed_prune.models import CompactSuccessor
from experiments.pamf_restoration.models import complex_iq, pair


class IQBlockInverse(NamedTuple):
    """Analytic symmetric 2x2 inverse at each current-window time point."""

    ii: torch.Tensor
    iq: torch.Tensor
    qq: torch.Tensor

    def apply(self, value):
        if value.shape != self.ii.shape or not value.is_complex():
            raise ValueError('Matched complex current-window residual required')
        return torch.complex(self.ii*value.real+self.iq*value.imag,
                             self.iq*value.real+self.qq*value.imag)


def _weight(action, variance, prior_variance):
    if variance.shape == (len(action.main),):
        variance = variance[:, None]
    if (variance.shape != (len(action.main), 1) or variance.is_complex()
            or not math.isfinite(prior_variance) or prior_variance <= 0):
        raise ValueError('One positive real variance per window and positive prior required')
    # The estimator guarantees finite positive native variance.  These public
    # helper checks also reject invalid direct calls rather than silently
    # changing the physical noise model.
    if not torch.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError('One positive real variance per window and positive prior required')
    return variance.reciprocal()


def _reflect_tap_sum(action, coefficient, taps):
    """Sum coefficient[n]*taps[k] into j=abs(n-k), using three FFT transforms.

    This is the reflect-filter transpose with *unconjugated* taps.  Its FFT
    construction is the original true reflect adjoint, including scatter-add.
    """
    padded = torch.zeros((len(coefficient), action.nfft),
                         device=coefficient.device, dtype=taps.dtype)
    padded[:, action.length-1:action.length-1+action.count] = coefficient
    response = torch.fft.fft(taps.conj(), n=action.nfft).conj()
    extended = torch.fft.ifft(torch.fft.fft(padded)*response)[:, :len(action.index)]
    return torch.zeros_like(action.main).scatter_add(
        1, action.index.expand(len(coefficient), -1), extended)


def _normal_coefficients(action, weight):
    """Return s,t for blocks [[s+2Re(t),-2Im(t)],[-2Im(t),s-2Re(t)]].

    At output n, c[n,j]=sum_{k:abs(n-k)=j}h[k].  For a=main[n]*c
    and b=image[n]*conj(c), the real map a*x+b*conj(x) has Gram block
    with s=w*(|main|^2+|image|^2)*|c|^2 and
    t=w*main*conj(image)*c^2.  Transpose-filtering squared taps gives
    the single-tap terms.  Each k<l with k+l even collides once at
    n=(k+l)/2,j=(l-k)/2; its two cross terms are added explicitly.

    Cost is O(B*Nfft*log(Nfft)+B*K^2), storage O(B*(Nfft+K^2)); no
    dense 2N-by-2N normal matrix or N-by-K bank is constructed.
    """
    kernel = action.kernel
    source_s = weight*(action.main.abs().square()+action.image.abs().square())
    source_t = weight*action.main*action.image.conj()
    s = _reflect_tap_sum(action, source_s, kernel.abs().square().to(kernel.dtype)).real
    t = _reflect_tap_sum(action, source_t, kernel.square())
    if action.length > 2:
        left, right = torch.triu_indices(action.length, action.length, offset=1,
                                        device=kernel.device)
        # Keeping the zero-weight opposite-parity pairs avoids a GPU-to-host
        # synchronization from dynamic boolean indexing.  All pair counts
        # depend only on the fixed kernel length.
        same_parity = ((left+right).remainder(2) == 0).to(weight.dtype)
        output_index = (left+right)//2
        input_index = ((right-left)//2).expand(len(kernel), -1)
        hleft, hright = kernel[:, left], kernel[:, right]
        energy_cross = 2*(hleft*hright.conj()).real*same_parity
        phase_cross = 2*hleft*hright*same_parity
        s = s.scatter_add(1, input_index, source_s[:, output_index]*energy_cross)
        t = t.scatter_add(1, input_index, source_t[:, output_index]*phase_cross)
    return s, t


def normal_blocks(action, variance, prior_variance=SCALE**2):
    """Exact real I/Q diagonal blocks of A*WA+I/tau (FFT roundoff aside)."""
    s, t = _normal_coefficients(action, _weight(action, variance, prior_variance))
    scalar = s+1/prior_variance
    return scalar+2*t.real, -2*t.imag, scalar-2*t.real


def build_preconditioner(action, variance, prior_variance=SCALE**2):
    """Build differentiable analytic block Jacobi once for the current input.

    Mathematically s>=2|t|.  Clamp only its roundoff-sized violation, then
    add the original 1/tau prior precision.  The analytic eigenvalues avoid
    subtracting two large products in a 2x2 determinant at low variance.
    No learned calibration, extra regularizer, or persistent cache is used.
    """
    s, t = _normal_coefficients(action, _weight(action, variance, prior_variance))
    radius = 2*t.abs()
    lower = (s-radius).clamp_min(0)+1/prior_variance
    upper = torch.maximum(s, radius)+radius+1/prior_variance
    inverse_determinant = lower.reciprocal()*upper.reciprocal()
    # Compute the nonnegative radius +/- 2Re(t) before adding lower; this
    # preserves the prior term for a nearly singular I/Q block.  Unlike a
    # polar eigenvector formula, this also has the right derivative at t=0.
    return IQBlockInverse((lower+(radius-2*t.real))*inverse_determinant,
                          2*t.imag*inverse_determinant,
                          (lower+(radius+2*t.real))*inverse_determinant)


def preconditioned_map(action, observed, prior, variance, preconditioner,
                       prior_variance=SCALE**2, *, initial):
    """Exactly one warm-start PCG update; retain the original rhs equations."""
    weight = _weight(action, variance, prior_variance)
    if (observed.shape != action.main.shape or prior.shape != observed.shape
            or initial.shape != prior.shape or not all(
                value.is_complex() for value in (observed, prior, initial))):
        raise ValueError('Matched complex observed, prior and warm start required')
    rhs = action.adjoint(weight*(observed-action(prior)))
    delta = initial-prior
    residual = rhs-action.adjoint(weight*action(delta))-delta/prior_variance
    direction = preconditioner.apply(residual)
    normal = action.adjoint(weight*action(direction))+direction/prior_variance
    alpha = real_dot(residual, direction)/real_dot(direction, normal).clamp_min(1e-20)
    delta = delta+alpha[:, None]*direction
    return prior+delta


class PreconditionedRestorer(nn.Module):
    """Same trainable R65 graph and context, with a changed physical solver."""

    def __init__(self, parent=None):
        super().__init__()
        from_checkpoint = parent is not None
        if parent is None:
            parent = CompactSuccessor('prune_after10')
        if (not isinstance(parent, CompactSuccessor) or parent.arm != 'prune_after10'
                or sum(p.numel() for p in parent.parameters()) != 273765):
            raise ValueError('Only the designated compact R65 prune_after10 parent is accepted')
        self.frozen, self.prior = copy.deepcopy((parent.frozen, parent.prior))
        self.arm = 'pcg1'
        self.requires_grad_(True)
        self.initialization_report = dict(
            mode=('exact_R65_weight_copy_solver_changed' if from_checkpoint
                  else 'random_shape_skeleton_for_strict_state_load'),
            initialized_from_parent_checkpoint=from_checkpoint, parent_arm='prune_after10',
            parent_parameters=273765, parameters=273765, copied_parameters=273765,
            new_parameters=0, discarded_parameters=0,
            priors_source='three_parent_priors_copied_together_with_long_weight_aliases',
            exact_initial_weight_equivalence=True, exact_initial_context_equivalence=True,
            exact_initial_function_equivalence=False, solver_changed=True,
            CG_steps_per_stage=1, stages=3,
            preconditioner='exact_real_IQ_normal_diagonal_blocks_with_reflect_collision_correction',
            preconditioner_builds_per_input=1, preconditioner_cached_across_inputs=False,
            preconditioner_roundoff_repair='clamp_negative_unregularized_minimum_eigenvalue_to_zero',
            parent_unchanged=True, all_student_parameters_trainable=True,
            IQ_examples=0, TX_labels_read=0, F_reads=0, V_reads=0, target_reads=0,
            optimizer_updates=0, new_calibration_IQ=0)

    def estimate_context(self, received):
        # Call the same two trainable modules as the joint R65 parent.  Its
        # ActionRestorer.estimate would incorrectly suppress coarse gradients.
        base = self.frozen.coarse(received)
        action, native = self.frozen.estimator(received, base)
        return base, action, native

    def stage_outputs(self, received):
        if received.ndim != 3 or received.shape[1:] != (2, 256):
            raise ValueError('Only current received IQ2x256 is accepted')
        base, action, native = self.estimate_context(received)
        variance = native[:, None].expand(-1, 256)
        preconditioner = build_preconditioner(action, native[:, None])
        observed, current, outputs = complex_iq(received), base, []
        logvar = native.log().clamp(-16, 4)[:, None, None].expand(-1, 1, 256)/8
        for core in self.prior:
            innovation = pair(action.adjoint(observed-action(complex_iq(current))))
            innovation = innovation/(innovation.square().mean((1, 2), keepdim=True)+1e-6).sqrt()
            prior = base+SCALE*core(torch.cat((received, base, current, innovation, logvar), 1))
            current = pair(preconditioned_map(
                action, observed, complex_iq(prior), native[:, None], preconditioner,
                initial=complex_iq(current)))
            outputs.append(current)
        return outputs, variance, action, base

    def training_outputs(self, received):
        outputs, variance, action, base = self.stage_outputs(received)
        return outputs[-1], variance, action, base

    def forward(self, received):
        return self.training_outputs(received)[0]
