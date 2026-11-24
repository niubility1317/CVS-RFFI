"""Estimate a widely linear LoRa prior over the non-carrier spectrum."""

from math import log

import torch
from torch import nn

from .variance_transport import helmert, physical_norm, posterior_gain


class CrossFrequencyPrior(nn.Module):
    """Use identical common/private moments for the coupled and per-bin filters."""

    def __init__(self):
        super().__init__()
        dimensions = 2 * 255
        self.register_buffer('basis', helmert(), persistent=False)
        bins = torch.arange(1, 256)
        self.register_buffer('outer', (bins > 64) & (bins < 192), persistent=False)
        self.register_buffer('center', torch.zeros(6, dimensions))
        self.register_buffer('joint_axes', torch.empty(2, dimensions, dimensions))
        self.register_buffer('joint_variance', torch.empty(2, dimensions))
        self.register_buffer('bin_axes', torch.empty(2, 255, 2, 2))
        self.register_buffer('bin_variance', torch.empty(2, 255, 2))

    def spectrum(self, observed):
        fourier = torch.fft.fft(observed, norm='ortho')[..., 1:]
        return self.basis.to(fourier) @ fourier

    @torch.no_grad()
    def calibrate(self, frames):
        dimensions = self.center.shape[-1]
        total = self.center.new_zeros(self.center.shape)
        second = self.center.new_zeros(2, dimensions, dimensions)
        n = 0
        for observed in frames:
            parts = torch.view_as_real(self.spectrum(observed)).flatten(-2)
            common = parts[:, 0]
            private = parts[:, 1:].flatten(0, 1)
            total += parts.sum(0)
            second[0].addmm_(common.T, common)
            second[1].addmm_(private.T, private)
            n += len(observed)
        if n == 0:
            raise ValueError('Expected source U frames for prior calibration')
        self.center.copy_(total / n)
        common_center = self.center[:1]
        private_center = self.center[1:]
        covariance = torch.stack((
            second[0] / n - common_center.T @ common_center,
            second[1] / (5 * n) - (private_center.T @ private_center) / 5,
        ))
        covariance = (covariance + covariance.mT) * .5
        bin_index = torch.arange(255, device=covariance.device)
        per_bin = covariance.reshape(2, 255, 2, 255, 2).diagonal(dim1=1, dim2=3).movedim(-1, 1)
        same_bin = bin_index.repeat_interleave(2)
        full = covariance * (.9 + .1 * (same_bin[:, None] == same_bin))
        for name, blocks in (('joint', full), ('bin', per_bin)):
            values, axes = torch.linalg.eigh(blocks)
            getattr(self, f'{name}_axes').copy_(axes)
            getattr(self, f'{name}_variance').copy_(values.clamp_min(1e-8))

    def noise(self, spectrum):
        power = spectrum[..., self.outer].abs().square()
        shared = power[:, 0].median(-1).values
        private = power[:, 1:].flatten(1).median(-1).values
        return torch.stack((shared, private), dim=-1) / (2 * log(2))

    def estimate(self, value, noise, component):
        by_bin = value.unflatten(-1, (255, 2))
        joint_axes = self.joint_axes[component]
        bin_axes = self.bin_axes[component]
        joint_rotated = value @ joint_axes
        bin_rotated = torch.einsum('bcfq,fqr->bcfr', by_bin, bin_axes)
        joint_gain = posterior_gain(self.joint_variance[component], noise[:, None, None])
        bin_gain = posterior_gain(self.bin_variance[component], noise[:, None, None, None])
        joint = torch.stack((joint_gain - 1., joint_gain.sqrt() - 1.)) * joint_rotated
        separate = torch.stack((bin_gain - 1., bin_gain.sqrt() - 1.)) * bin_rotated
        joint = joint @ joint_axes.T
        separate = torch.einsum('kbcfr,fqr->kbcfq', separate, bin_axes).flatten(-2)
        return torch.stack((joint[0], separate[0], joint[1], separate[1]))

    def transport(self, spectrum, noise):
        value = torch.view_as_real(spectrum).flatten(-2) - self.center
        edits = []
        for index, part in enumerate((value[:, :1], value[:, 1:])):
            axes = self.joint_axes[index]
            gain = posterior_gain(self.joint_variance[index], noise[:, index, None, None])
            edits.append(((part @ axes) * (gain.sqrt() - 1.)) @ axes.T)
        packed = torch.view_as_complex(torch.cat(edits, dim=1).unflatten(-1, (255, 2)).contiguous())
        full = torch.cat((torch.zeros_like(packed[..., :1]), packed), dim=-1)
        return self.basis.mT.to(full) @ torch.fft.ifft(full, norm='ortho')

    def changes(self, observed, rms):
        spectrum = self.spectrum(observed)
        noise = self.noise(spectrum)
        values = torch.view_as_real(spectrum).flatten(-2) - self.center
        common = self.estimate(values[:, :1], noise[:, 0], 0)
        private = self.estimate(values[:, 1:], noise[:, 1], 1)
        packed = torch.cat((common, private), dim=2)
        complex_edit = torch.view_as_complex(packed.unflatten(-1, (255, 2)).contiguous())
        spectrum_edit = torch.cat((torch.zeros_like(complex_edit[..., :1]), complex_edit), dim=-1)
        edit = self.basis.mT.to(spectrum_edit) @ torch.fft.ifft(spectrum_edit, norm='ortho')
        observed_power = physical_norm(observed, rms)
        dose = physical_norm(edit, rms)
        edit = edit * (.30 * observed_power / dose.clamp_min(1e-10)).clamp_max(1.)
        both = edit[2:]
        norms = physical_norm(both, rms)
        equal = both * (norms.amin(0) / norms.clamp_min(1e-10)).clamp_max(1.)
        return torch.cat((edit, equal)), noise
