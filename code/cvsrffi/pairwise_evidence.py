"""Shared-covariance diagnostics and label-free pair residual projection."""
import torch
from torch import nn
from .partial_gaussian_head import _chol


def shared_pair_distance(means, covariance, observed):
    """Exact squared separation [B,C,C], ONLY for shared covariance [B,D,D]."""
    if means.ndim != 3:
        raise ValueError('means must be [B,C,D]')
    b, c, d = means.shape
    if covariance.shape != (b, d, d) or observed.shape != (b, d) or observed.dtype != torch.bool:
        raise ValueError('requires shared [B,D,D] covariance and boolean [B,D] mask')
    result = []
    for i in range(b):
        ix = observed[i].nonzero().flatten()
        m = means[i].index_select(-1, ix)
        delta = m[:, None] - m[None, :]
        cov = covariance[i].index_select(0, ix).index_select(1, ix)
        if not ix.numel():
            result.append(delta.sum(-1) + cov.sum()); continue
        if not torch.isfinite(delta).all():
            raise ValueError('observed means must be finite')
        white = torch.linalg.solve_triangular(_chol(cov), delta.reshape(c*c, -1).T, upper=False)
        result.append(white.square().sum(0).reshape(c, c))
    return torch.stack(result)


def schur_incremental_information(delta, covariance, existing, added):
    """Unbatched shared-covariance decomposition for disjoint index sets J and K."""
    j = torch.as_tensor(existing, device=delta.device, dtype=torch.long)
    k = torch.as_tensor(added, device=delta.device, dtype=torch.long)
    all_ix = torch.cat((j, k))
    if delta.ndim != 1 or covariance.shape != (delta.numel(), delta.numel()) or all_ix.unique().numel() != all_ix.numel():
        raise ValueError('requires vector delta, shared square covariance, disjoint unique indices')
    vjj = covariance[j[:, None], j[None, :]]
    vkk = covariance[k[:, None], k[None, :]]
    vkj = covariance[k[:, None], j[None, :]]
    base = delta[j].sum() * 0
    conditional_delta, conditional_cov = delta[k], vkk
    if j.numel():
        chol = _chol(vjj)
        solved = torch.cholesky_solve(delta[j, None], chol).squeeze(-1)
        base = delta[j] @ solved
        conditional_delta = delta[k] - vkj @ solved
        conditional_cov = vkk - vkj @ torch.cholesky_solve(vkj.T, chol)
    incremental = conditional_delta.sum() * 0
    if k.numel():
        incremental = conditional_delta @ torch.cholesky_solve(conditional_delta[:, None], _chol(conditional_cov)).squeeze(-1)
    return dict(base=base, incremental=incremental, total=base+incremental)


def effective_fisher_information(identity, cross, nuisance, rtol=None):
    """Schur complement on identifiable nuisance eigenmodes; reject invalid blocks.

    Inputs are unbatched Fisher blocks I_tt, I_te, I_ee. Rank is a diagnostic;
    differentiability holds within a fixed rank region, not across rank changes.
    """
    if identity.ndim != 2 or identity.shape[0] != identity.shape[1] or nuisance.ndim != 2 or nuisance.shape[0] != nuisance.shape[1] or cross.shape != (identity.shape[0], nuisance.shape[0]):
        raise ValueError('incompatible Fisher blocks')
    whole = torch.cat((torch.cat((identity, cross), 1), torch.cat((cross.T, nuisance), 1)), 0)
    if not torch.isfinite(whole).all() or not torch.allclose(whole, whole.T):
        raise ValueError('Fisher blocks must be finite symmetric')
    tol = (max(whole.shape) * torch.finfo(whole.dtype).eps if rtol is None else rtol)
    if tol < 0:
        raise ValueError('rtol must be nonnegative')
    spectrum = torch.linalg.eigvalsh(whole)
    threshold = tol * spectrum.abs().max().clamp_min(torch.finfo(whole.dtype).tiny)
    if (spectrum < -threshold).any():
        raise ValueError('joint Fisher matrix must be positive semidefinite')
    eig, vec = torch.linalg.eigh(nuisance)
    eig_scale = eig.abs().max() if eig.numel() else eig.new_zeros(())
    keep = eig > tol * eig_scale.clamp_min(torch.finfo(eig.dtype).tiny)
    basis = vec[:, keep]
    projected = cross @ basis
    info = identity - (projected / eig[keep]) @ projected.T
    info = (info + info.T) * 0.5
    info_eig = torch.linalg.eigvalsh(info)
    rank = (info_eig > threshold).sum()
    return dict(information=info, nuisance_rank=keep.sum(), identifiable_rank=rank)


class SharedPairResidual(nn.Module):
    """One shared descriptor network; bounded antisymmetry, zero initial residual."""
    def __init__(self, feature_dim, state_dim, hidden=64, bound=1.0):
        super().__init__()
        if bound <= 0:
            raise ValueError('bound must be positive')
        self.bound = float(bound)
        self.network = nn.Sequential(nn.Linear(7*feature_dim+state_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1, bias=False))
        nn.init.zeros_(self.network[-1].weight)

    def forward(self, z, means, state, quality, covariance_diagonal, observed=None):
        b, c, d = means.shape
        if z.shape != (b,d) or quality.shape != (b,d) or covariance_diagonal.shape != means.shape or state.shape[0] != b:
            raise ValueError('incompatible pair descriptors')
        observed = torch.ones_like(z,dtype=torch.bool) if observed is None else observed
        if observed.dtype != torch.bool or observed.shape != z.shape:
            raise ValueError('observed must be bool [B,D]')
        z = torch.where(observed,z,0.)
        means = torch.where(observed[:,None,:],means,0.)
        quality = torch.where(observed,quality,0.)
        covariance_diagonal = torch.where(observed[:,None,:],covariance_diagonal,0.)
        def pair(t):
            return t[:, :, None].expand(-1, -1, c, -1)
        ma, va = pair(means), pair(covariance_diagonal)
        features = torch.cat((z[:,None,None].expand(-1,c,c,-1), ma, ma.transpose(1,2),
                              quality[:,None,None].expand(-1,c,c,-1), va, va.transpose(1,2),
                              observed[:,None,None].expand(-1,c,c,-1).to(z.dtype),
                              state[:,None,None].expand(-1,c,c,-1)), -1)
        raw = self.network(features).squeeze(-1)
        return self.bound * torch.tanh((raw - raw.transpose(1,2)) * 0.5)


def project_pairwise(base_scores, differences, weights, anchor=1.0):
    """Exact minimizer of sum_{a<b} w_ab(u_a-u_b-d_ab)^2+anchor||u-s||^2."""
    b, c = base_scores.shape
    if anchor <= 0 or differences.shape != (b,c,c) or weights.shape != (b,c,c):
        raise ValueError('positive anchor and matching pair matrices required')
    if not torch.isfinite(differences).all() or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('finite differences and nonnegative finite weights required')
    if not torch.allclose(weights, weights.transpose(1,2)) or not torch.allclose(differences, -differences.transpose(1,2), atol=1e-6):
        raise ValueError('weights must be symmetric and differences antisymmetric')
    w = weights - torch.diag_embed(weights.diagonal(dim1=-2,dim2=-1))
    lap = torch.diag_embed(w.sum(-1)) - w
    matrix = lap + anchor * torch.eye(c, dtype=base_scores.dtype, device=base_scores.device)
    rhs = anchor * base_scores + (w*differences).sum(-1)
    return torch.linalg.solve(matrix, rhs[...,None]).squeeze(-1)
