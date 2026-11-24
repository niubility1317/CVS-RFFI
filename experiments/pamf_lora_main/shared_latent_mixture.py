"""Moment-matched mixture prior in the frozen LoRa shared rank-four coordinates."""

import torch

from .spectral_covariance import real_coordinates


WITHIN_VARIANCE = .25


def empirical_centers(common, mean, diagonal, factors):
    """Project and whiten a balanced source cloud; keep Gaussian prior moments."""
    q = factors.double()
    precision = (q / diagonal.double().clamp_min(1e-8)[:, None]).T
    identity = torch.eye(q.shape[-1],device=q.device,dtype=q.dtype)
    scores = torch.linalg.solve(precision @ q + 1e-4 * identity,
                                precision @ (common-mean).double().T).T
    scores = scores-scores.mean(0)
    cov = scores.T @ scores / len(scores)
    chol = torch.linalg.cholesky(cov)
    standardized = torch.linalg.solve_triangular(chol,scores.T,upper=False).T
    return ((1-WITHIN_VARIANCE)**.5 * standardized).to(factors.dtype)


def latent_posteriors(precision, natural, centers):
    """Posterior means with the same prior covariance but different latent laws."""
    d = precision.shape[-1]
    eye = torch.eye(d,device=precision.device,dtype=precision.dtype)
    gaussian = torch.linalg.solve(eye + precision,natural.unsqueeze(-1)).squeeze(-1)
    system = eye + WITHIN_VARIANCE * precision
    solved = torch.linalg.solve(system,torch.cat((natural.unsqueeze(-1),precision),dim=-1))
    projected_natural, projected_precision = solved[...,0],solved[...,1:]
    projected_centers = centers @ projected_precision
    votes = projected_natural @ centers.T - .5 * (projected_centers * centers).sum(-1)
    weights = votes.softmax(-1)
    center = weights @ centers
    mixture = WITHIN_VARIANCE * projected_natural + torch.linalg.solve(system,center.unsqueeze(-1)).squeeze(-1)
    return gaussian,mixture


def shared_posteriors(observed, noise, prior, centers):
    """Integrate the individual chirps before resolving the public latent."""
    bins = prior.correlated_bins
    measures = real_coordinates(observed.index_select(-1,bins))
    variance = noise.index_select(-1,bins) * .5
    noise_iq = torch.cat((variance,variance),dim=-1)
    precision = (prior.private_real+noise_iq).clamp_min(1e-8).reciprocal()
    posterior_var = precision.sum(1).reciprocal()
    mean = real_coordinates(prior.mean[bins])
    average = (precision * (measures-mean)).sum(1)*posterior_var
    factors = prior.factors
    diagonal = (prior.shared_real-factors.square().sum(-1)).clamp_min(1e-8)
    total = diagonal+posterior_var
    weighted = factors / total.unsqueeze(-1)
    a = factors.T @ weighted
    b = ((average/total).unsqueeze(1) @ factors).squeeze(1)
    gaussian, mixture = latent_posteriors(a,b,centers)
    constant = mean + diagonal/total*average
    result = []
    for latent in (gaussian,mixture):
        value = constant + posterior_var/total*(latent @ factors.T)
        i,q = value.chunk(2,dim=-1)
        result.append(torch.complex(i,q))
    return tuple(result)


def replace_common(spectrum,bins,common):
    original = spectrum.index_select(-1,bins)
    corrected = original + (common-original.mean(1)).unsqueeze(1)
    return spectrum.index_copy(-1,bins,corrected)
