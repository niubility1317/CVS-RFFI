"""Estimate LoRa private chirp spectra from a packet's coherent structure."""

from dataclasses import dataclass

import torch

from .spectral_covariance import real_coordinates


def private_spectrum(spectrum):
    return spectrum - spectrum.mean(dim=1, keepdim=True)


def ordered_context(spectrum):
    """Predict an ordered contrast using the straight-line fit on other chirps."""
    n = spectrum.shape[1]
    steps = torch.arange(n, dtype=spectrum.real.dtype, device=spectrum.device)
    position = steps - steps.mean()
    available = ~torch.eye(n, dtype=torch.bool, device=spectrum.device)
    offset = (available * position[None, :]).sum(-1, keepdim=True) / (n - 1)
    regressors = available * (position[None, :] - offset)
    weights = position[:, None] * regressors / regressors.square().sum(-1, keepdim=True)
    return weights.to(spectrum.dtype) @ spectrum


def component_features(spectrum, shared_bins, private_bins, *, order_spectrum=None):
    pooled = real_coordinates(spectrum[..., shared_bins].mean(1, keepdim=True)).expand(-1,spectrum.shape[1],-1)
    order_spectrum = spectrum if order_spectrum is None else order_spectrum
    trend = real_coordinates(ordered_context(order_spectrum[..., private_bins]))
    return pooled, torch.cat((pooled,trend),dim=-1)


@dataclass(frozen=True)
class Regression:
    center: torch.Tensor
    scale: torch.Tensor
    target_mean: torch.Tensor
    coefficients: torch.Tensor

    def predict(self, x):
        z = ((x-self.center) / self.scale).transpose(0,1)
        return (z @ self.coefficients).transpose(0,1) + self.target_mean

    def bytes(self):
        return sum(t.numel()*t.element_size() for t in
                   (self.center,self.scale,self.target_mean,self.coefficients))


def fit_regression(x, y, *, ridge=.10):
    center = x.mean(0)
    target_mean = y.mean(0)
    scale = x.std(dim=0,unbiased=False).clamp_min(1e-5)
    normalized = ((x-center)/scale).transpose(0,1).double()
    targets = (y-target_mean).transpose(0,1).double()
    basis = torch.eye(x.shape[-1],dtype=normalized.dtype,device=x.device)
    coeff = torch.linalg.solve(normalized.mT @ normalized / len(x) + ridge*basis,
                               normalized.mT @ targets / len(x))
    return Regression(center,scale,target_mean,coeff.to(x.dtype))


def context_center(predictor, features):
    re, im = predictor.predict(features).chunk(2,dim=-1)
    # Center each held-out prior independently. The recovered packet is
    # projected to zero-sum only at the final joint reconstruction.
    return torch.complex(re,im)


def contextual_private(measured, base, bins, prior_mean, gain):
    private = private_spectrum(measured[..., bins])
    updated = prior_mean + gain * (private - prior_mean)
    shared = base[..., bins].mean(1,keepdim=True)
    result = base.clone()
    result[..., bins] = shared + private_spectrum(updated)
    return result
