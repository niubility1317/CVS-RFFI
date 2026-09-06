"""ECRS V2 fixed physical estimator and independently trainable response branch.

No transmitter labels or dataset templates enter this module. The default reference
is explicitly estimated, not a verified protocol waveform. Cross-fit is diagnostic.
"""
from __future__ import annotations

import hashlib
import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

BASIS_NAMES = ("direct_cubic_m0", "direct_quintic_m0", "direct_cubic_m1",
               "direct_quintic_m1", "conjugate_linear_m0", "conjugate_cubic_m0",
               "conjugate_linear_m1", "conjugate_cubic_m1")
ESTIMATOR_VARIANTS = ("legacy28_old_reference", "legacy28_estimated_reference", "compact8")


def _complex(x: Tensor) -> Tensor:
    if x.is_complex():
        return x
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError("IQ must have shape [B,2,T] or complex [B,T]")
    x = x.float() if x.dtype != torch.float64 else x
    return torch.complex(x[:, 0], x[:, 1])


def real_stack(x: Tensor) -> Tensor:
    return torch.cat((x.real, x.imag), dim=-1)


def complex_design(phi: Tensor) -> Tensor:
    """[B,T,K] complex to [B,2T,2K], coefficient order real then imag."""
    return torch.cat((torch.cat((phi.real, -phi.imag), -1),
                      torch.cat((phi.imag, phi.real), -1)), -2)


def build_response_basis(s: Tensor, history: Optional[Tensor] = None) -> Tensor:
    """Eight complex candidates; history is explicit and never wraps in time."""
    if history is None:
        history = F.pad(s[..., :-1], (1, 0))
    a, b = s.abs().square(), history.abs().square()
    return torch.stack((s*a, s*a*a, history*b, history*b*b,
                        s.conj(), s.conj()*a, history.conj(), history.conj()*b), -1)


def build_nuisance_basis(s: Tensor) -> Tensor:
    """Real gain, phase and normalized-time CFO Jacobians: three real columns."""
    t = torch.linspace(-1, 1, s.shape[-1], device=s.device, dtype=s.real.dtype)
    n = torch.stack((s, 1j*s, 1j*s*t), -1)
    return torch.cat((n.real, n.imag), -2)


def make_anchor_states(mode: str = "complex24", *, device=None,
                       dtype=torch.complex64) -> tuple[Tensor, Tensor]:
    if mode == "complex24":
        amp = torch.tensor([.3, .6, .9], device=device)
        phase = torch.arange(4, device=device) * (math.pi/2)
        s = (amp[:, None] * torch.exp(1j*phase)[None]).flatten().to(dtype)
        return torch.cat((s, s)), torch.cat((s, torch.zeros_like(s)))
    if mode == "real8":
        s = torch.tensor([-.9, -.3, .3, .9], device=device, dtype=torch.float64).to(dtype)
        return torch.cat((s, s)), torch.cat((s, torch.zeros_like(s)))
    raise ValueError("anchor_mode must be complex24 or real8")


def probe_response(theta: Tensor, mode: str = "complex24") -> Tensor:
    s, h = make_anchor_states(mode, device=theta.device, dtype=theta.dtype)
    return torch.einsum("qk,bk->bq", build_response_basis(s, h), theta)


def _chol_solve(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """One batched factorization and one bounded fallback for its failed subset."""
    factor, info = torch.linalg.cholesky_ex(a)
    failed = info.ne(0) | ~torch.isfinite(factor).all(dim=(-2, -1))
    status = failed.to(torch.int64)
    if failed.any():
        sub = a[failed]
        boost = sub.diagonal(dim1=-2, dim2=-1).abs().mean(-1).clamp_min(1) * 1e-3
        retry, retry_info = torch.linalg.cholesky_ex(
            sub + boost[:, None, None]*torch.eye(a.shape[-1], device=a.device, dtype=a.dtype))
        factor = factor.clone()
        factor[failed] = retry
        status[failed] = torch.where(retry_info.eq(0), 1, 2)
    bad = status.eq(2)
    if bad.any():
        factor = factor.clone()
        factor[bad] = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
    result = torch.cholesky_solve(b, factor)
    result = torch.where(bad[:, None, None], torch.zeros_like(result), result)
    return result, status


def schur_ridge(y: Tensor, nuisance: Tensor, response: Tensor, *,
                alpha_eta: float = .01, alpha_theta: float = .01,
                weights: Optional[Tensor] = None, fit_mask: Optional[Tensor] = None,
                normalize_columns: bool = True, diagnostics: bool = False,
                noise_variance: Optional[float] = None) -> dict:
    """Batched real Schur ridge with fixed positive penalties in normalized space.

    Returned coefficients and sensitivity are mapped back to original column units.
    With an explicitly supplied iid Gaussian variance sigma2, the reported posterior
    uses prior precision diag(alpha)/sigma2; otherwise this is only ridge sensitivity.
    """
    if alpha_eta <= 0 or alpha_theta <= 0:
        raise ValueError("ridge penalties must be strictly positive")
    if noise_variance is not None and (not math.isfinite(noise_variance) or noise_variance <= 0):
        raise ValueError("noise_variance must be finite and positive")
    dtype = torch.float64 if y.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=y.device.type, enabled=False):
        y, n, p = y.to(dtype), nuisance.to(dtype), response.to(dtype)
        if y.ndim != 2 or n.shape[:2] != y.shape or p.shape[:2] != y.shape:
            raise ValueError("expected y[B,M], nuisance[B,M,J], response[B,M,K]")
        w = torch.ones_like(y) if weights is None else weights.to(dtype).expand_as(y)
        if (w < 0).any():
            raise ValueError("weights must be nonnegative")
        mask = torch.ones_like(y, dtype=torch.bool) if fit_mask is None else fit_mask.bool().expand_as(y)
        # Nonfinite held-out observations do not invalidate a fit-side estimate.
        valid = ((torch.isfinite(y) | ~mask).all(-1) &
                 (torch.isfinite(n).all(-1) | ~mask).all(-1) &
                 (torch.isfinite(p).all(-1) | ~mask).all(-1) &
                 (torch.isfinite(w) | ~mask).all(-1) & mask.any(-1))
        y, n, p = (torch.nan_to_num(v, nan=0., posinf=0., neginf=0.) for v in (y, n, p))
        w = torch.where(mask & valid[:, None], torch.nan_to_num(w, nan=0., posinf=0., neginf=0.), torch.zeros_like(w))
        count = w.sum(-1).clamp_min(1)
        def scaled(d):
            scale = ((d.square()*w[..., None]).sum(-2)/count[:, None]).sqrt()
            scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale)) if normalize_columns else torch.ones_like(scale)
            return d/scale[:, None], scale
        n, ns = scaled(n)
        p, ps = scaled(p)
        ntw, ptw = n.transpose(-2, -1)*w[:, None], p.transpose(-2, -1)*w[:, None]
        j, k = n.shape[-1], p.shape[-1]
        gn = ntw@n + alpha_eta*torch.eye(j, device=y.device, dtype=dtype)
        c, ny = ntw@p, ntw@y[..., None]
        solved, status_n = _chol_solve(gn, torch.cat((c, ny), -1))
        nc, nny = solved[..., :k], solved[..., k:]
        ae = ptw@p + alpha_theta*torch.eye(k, device=y.device, dtype=dtype) - c.transpose(-2, -1)@nc
        ae = .5*(ae + ae.transpose(-2, -1))
        rhs = ptw@y[..., None] - c.transpose(-2, -1)@nny
        theta, status_p = _chol_solve(ae, rhs)
        eta = nny - nc@theta
        status = torch.maximum(status_n, status_p)
        valid = valid & status.lt(2) & torch.isfinite(theta).all(dim=(-2, -1))
        theta = torch.where(valid[:, None], theta.squeeze(-1)/ps, 0)
        eta = torch.where(valid[:, None], eta.squeeze(-1)/ns, 0)
        out = {"theta": theta, "eta": eta, "valid": valid,
               "solver_status": torch.where(valid, status, 2), "effective_count": (w > 0).sum(-1)}
        if diagnostics:
            inv, _ = _chol_solve(ae, torch.eye(k, device=y.device, dtype=dtype).expand(y.shape[0], -1, -1))
            sensitivity = inv/(ps[:, :, None]*ps[:, None, :])
            out["response_sensitivity"] = sensitivity
            out["uncertainty_kind"] = "ridge_sensitivity"
            if noise_variance is not None:
                out["response_posterior_covariance"] = noise_variance*sensitivity
                out["uncertainty_kind"] = "iid_gaussian_conditional_posterior"
            design = torch.cat((n, p), -1)*w.sqrt()[..., None]
            singular = torch.linalg.svdvals(design)
            out["effective_rank"] = (singular > singular[:, :1]*1e-5).sum(-1)
            out["condition_number"] = (singular[:, 0]/singular[:, -1].clamp_min(1e-12)).clamp_max(1e12)
            out["zero_columns"] = (design.square().sum(-2) < 1e-12).sum(-1)
            # Normalized correlation is a diagnostic, not a claim of identifiability.
            out["nuisance_response_overlap"] = (c.abs()/((ntw@n).diagonal(dim1=-2, dim2=-1).sqrt()[:, :, None]*
                (ptw@p).diagonal(dim1=-2, dim2=-1).sqrt()[:, None, :]).clamp_min(1e-12)).amax(dim=(-2, -1))
        return out


class FixedOperatorCache:
    """One-entry cache: every operator-defining value participates in invalidation.

    Content comparison deliberately handles in-place changes and version reuse.
    This exact check copies small fixed dictionaries to CPU; dynamic references
    must bypass the cache. Runtime cache is never serialized in a bundle.
    """
    def __init__(self):
        self.key = None
        self.operator = None
        self.misses = 0
        self.hits = 0

    def clear(self):
        self.key = self.operator = None

    def solve(self, y, n, p, *, reference_version, **kwargs):
        if n.shape[0] != 1 or p.shape[0] != 1:
            raise ValueError("fixed cache requires one shared operator, not per-packet dictionaries")
        def digest(t):
            if t is None:
                return None
            return (tuple(t.shape), str(t.dtype), str(t.device),
                    hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest())
        if any(isinstance(v, Tensor) and v.ndim > 1 and v.shape[0] != 1 for v in kwargs.values()):
            raise ValueError("fixed cache requires shared weights and masks")
        key = (reference_version, str(y.dtype), str(y.device), digest(n), digest(p),
               tuple((name, digest(val) if isinstance(val, Tensor) else val) for name, val in sorted(kwargs.items())))
        if key != self.key:
            # Construct all RHS together as a batch, once per immutable operator.
            m = y.shape[-1]
            eye = torch.eye(m, device=y.device, dtype=y.dtype)
            nn_, pp = n[:1].expand(m, -1, -1), p[:1].expand(m, -1, -1)
            args = {k: v[:1].expand(m, -1) if isinstance(v, Tensor) else v for k, v in kwargs.items()}
            fit = schur_ridge(eye, nn_, pp, **args)
            self.operator = (fit["theta"].T.detach(), fit["eta"].T.detach(), fit["solver_status"].max().detach())
            self.key = key
            self.misses += 1
        else:
            self.hits += 1
        tm, em, status = self.operator
        valid = torch.isfinite(y).all(-1)
        clean = torch.nan_to_num(y)
        return {"theta": torch.where(valid[:, None], clean@tm.T, 0),
                "eta": torch.where(valid[:, None], clean@em.T, 0),
                "valid": valid & status.lt(2), "solver_status": torch.where(valid, status, 2)}


class ECRSV2PhysicalEstimator(nn.Module):
    def __init__(self, *, reference_mode="estimated_reference", public_reference=None,
                 reference_version="ecrs_v2_ref_v1", alpha_eta=.01, alpha_theta=.01,
                 fir_kernel=(.25, .5, .25), anchor_mode="complex24", cache_fixed_operator=True,
                 estimator_variant="compact8", legacy_reference_provider=None, legacy_basis=None):
        super().__init__()
        if estimator_variant not in ESTIMATOR_VARIANTS:
            raise ValueError("unsupported estimator_variant")
        if estimator_variant != "compact8" and legacy_basis is None:
            raise ValueError("legacy28 bridges require an injected legacy ResponseBasis")
        if estimator_variant == "legacy28_old_reference" and legacy_reference_provider is None:
            raise ValueError("old-reference bridge requires an injected frozen reference provider")
        if estimator_variant != "compact8" and reference_mode != "estimated_reference":
            raise ValueError("legacy28 bridges use their explicitly declared estimated reference")
        self.estimator_variant = estimator_variant
        self.basis_dim = 8 if estimator_variant == "compact8" else 28
        self.legacy_reference_provider = legacy_reference_provider
        self.legacy_basis = legacy_basis
        for component in (legacy_reference_provider, legacy_basis):
            if component is not None:
                component.requires_grad_(False)
        if reference_mode not in ("estimated_reference", "public_reference"):
            raise ValueError("unsupported reference mode")
        if reference_mode == "public_reference" and public_reference is None:
            raise ValueError("public_reference requires an explicit verified public waveform")
        kernel = torch.as_tensor(fir_kernel, dtype=torch.float32)
        if kernel.ndim != 1 or len(kernel) % 2 != 1 or not torch.isfinite(kernel).all() or (kernel < 0).any() or kernel.sum() <= 0:
            raise ValueError("FIR must be finite, odd length, nonnegative, and nonzero")
        self.register_buffer("fir_kernel", kernel/kernel.sum())
        reference = None if public_reference is None else torch.as_tensor(public_reference)
        if reference is not None:
            if not reference.is_complex() and reference.ndim == 2 and reference.shape[0] == 2:
                reference = torch.complex(reference[0].float(), reference[1].float())
            reference = reference.flatten().to(torch.complex64)
            if not torch.isfinite(reference).all():
                raise ValueError("public reference must be finite")
        self.register_buffer("public_reference", reference)
        self.reference_mode, self.reference_version = reference_mode, reference_version
        self.alpha_eta, self.alpha_theta = float(alpha_eta), float(alpha_theta)
        if self.alpha_eta <= 0 or self.alpha_theta <= 0:
            raise ValueError("ridge penalties must be positive")
        self.anchor_mode = anchor_mode
        make_anchor_states(anchor_mode)
        self.cache_fixed_operator = cache_fixed_operator
        self.operator_cache = FixedOperatorCache()

    @staticmethod
    def _fit_extension(x, fit_mask):
        if fit_mask is None:
            return x
        t = torch.arange(x.shape[-1], device=x.device)
        distance = (t[:, None]-t[None, :]).abs().expand(x.shape[0], -1, -1)
        distance = distance.masked_fill(~fit_mask[:, None, :], x.shape[-1]+1)
        return x.gather(-1, distance.argmin(-1))

    def _reference(self, x, fit_mask=None):
        if self.reference_mode == "public_reference":
            if self.public_reference.numel() != x.shape[-1]:
                raise ValueError("public reference length must match IQ; no implicit resampling")
            return self.public_reference.to(x).expand_as(x)
        # Fit-first reference: unavailable positions are extended from nearest fit
        # observation before shared-real FIR. This is an explicit estimated-reference
        # extrapolation, never a claim of a known excitation on the held-out side.
        x = self._fit_extension(x, fit_mask)
        iq = torch.stack((x.real, x.imag), 1)
        k = self.fir_kernel.to(iq).view(1, 1, -1).expand(2, 1, -1)
        radius = self.fir_kernel.numel()//2
        smooth = F.conv1d(F.pad(iq, (radius, radius), mode="replicate"), k, groups=2)
        return torch.complex(smooth[:, 0], smooth[:, 1])

    def _legacy_scale(self, s, fit_mask):
        values = s.abs() if fit_mask is None else s.abs().masked_fill(~fit_mask, float('nan'))
        return torch.nan_to_num(torch.nanquantile(values, .95, dim=-1, keepdim=True), nan=1e-4).clamp_min(1e-4)

    def _basis(self, s, scale=None, history=None, history2=None):
        if self.estimator_variant == "compact8":
            return build_response_basis(s, history)
        return self.legacy_basis(s, amplitude_scale=scale, history=history, history2=history2)

    def _probe_design(self, theta, scale):
        s, h = make_anchor_states(self.anchor_mode, device=theta.device, dtype=theta.dtype)
        s, h = s.expand(theta.shape[0], -1), h.expand(theta.shape[0], -1)
        return self._basis(s, scale, h, h)

    def forward(self, x, *, fit_mask=None, return_diagnostics=False, use_cache=True):
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            x = _complex(x).to(torch.complex128 if x.dtype in (torch.float64, torch.complex128) else torch.complex64)
            b, t = x.shape
            if fit_mask is not None:
                fit_mask = fit_mask.bool().expand(b, t)
            input_valid = (torch.isfinite(x) if fit_mask is None else (torch.isfinite(x) | ~fit_mask)).all(-1)
            # Sanitize arithmetic only; invalidity remains explicit in all fit outputs.
            x = torch.complex(torch.nan_to_num(x.real, nan=0., posinf=0., neginf=0.),
                              torch.nan_to_num(x.imag, nan=0., posinf=0., neginf=0.))
            observation = x
            if self.estimator_variant == "legacy28_old_reference":
                # The provider estimates parameters/reference from fit extension,
                # then applies only that frozen transform to the evaluation target.
                reference_data = self.legacy_reference_provider(x, self._fit_extension(x, fit_mask))
                s, observation = reference_data["reference"], reference_data["observation"]
            else:
                s = self._reference(x, fit_mask)
            basis_scale = None if self.estimator_variant == "compact8" else self._legacy_scale(s, fit_mask)
            n, p = build_nuisance_basis(s), complex_design(self._basis(s, basis_scale))
            y = real_stack(observation-s)
            mask = None if fit_mask is None else torch.cat((fit_mask, fit_mask), -1)
            # Remove columns wholly unexcited or contained in real nuisance span.
            # Preserve the public eight-complex coefficient schema with explicit masks;
            # zero slots are never counted as effective identifiable dimensions.
            active_rows = torch.ones_like(y) if mask is None else mask.to(y)
            nf, pf = n*active_rows[..., None], p*active_rows[..., None]
            un, sn, _ = torch.linalg.svd(nf, full_matrices=False)
            un = un * (sn > sn[:, :1]*1e-6)[:, None]
            orthogonal = pf - un@(un.transpose(-2, -1)@pf)
            power = pf.square().sum(-2)
            # Deterministic modified Gram-Schmidt in candidate order also deletes
            # exact response-response duplicates on the actual fit excitation.
            accepted = []
            column_masks = []
            for col in range(p.shape[-1]):
                v = orthogonal[..., col]
                for previous in accepted:
                    v = v - (v*previous).sum(-1, keepdim=True)*previous
                remaining = v.square().sum(-1)
                active_col = (power[:, col] > 1e-12) & (remaining > power[:, col]*1e-10)
                accepted.append(torch.where(active_col[:, None], v/remaining.sqrt().clamp_min(1e-12)[:, None], 0))
                column_masks.append(active_col)
            column_active = torch.stack(column_masks, -1)
            p = p*column_active[:, None]
            args = dict(alpha_eta=self.alpha_eta, alpha_theta=self.alpha_theta, fit_mask=mask)
            cached = (use_cache and self.cache_fixed_operator and self.reference_mode == "public_reference"
                      and fit_mask is None and not return_diagnostics)
            if cached:
                fit = self.operator_cache.solve(y, n[:1], p[:1], reference_version=self.reference_version, **args)
            else:
                fit = schur_ridge(y, n, p, diagnostics=return_diagnostics, **args)
            fit["valid"] = fit["valid"] & input_valid
            fit["solver_status"] = torch.where(input_valid, fit["solver_status"], 2)
            fit["theta"] = torch.where(input_valid[:, None], fit["theta"], 0)
            fit["theta"] = fit["theta"]*column_active
            fit["eta"] = torch.where(input_valid[:, None], fit["eta"], 0)
            theta = torch.complex(fit["theta"][:, :self.basis_dim], fit["theta"][:, self.basis_dim:])
            anchor_design = self._probe_design(theta, basis_scale)
            response = torch.einsum("bqk,bk->bq", anchor_design, theta)
            reconstruction = (n@fit["eta"][..., None] + p@fit["theta"][..., None]).squeeze(-1)
            active = torch.ones_like(y) if mask is None else mask.to(y)
            energy = (torch.nan_to_num(real_stack(observation)).square()*active).sum(-1)
            residual = ((torch.nan_to_num(y)-reconstruction).square()*active).sum(-1)
            nmse = residual/energy.clamp_min(1e-12)
            valid = fit["valid"] & energy.gt(1e-12) & response.abs().square().sum(-1).gt(1e-20)
            response = torch.where(valid[:, None], response, 0)
            # Coverage is a separate diagnostic, not an identity weighting.
            anchors, _ = make_anchor_states(self.anchor_mode, device=x.device, dtype=x.dtype)
            observed_amp = s.abs() if fit_mask is None else s.abs().masked_fill(~fit_mask, 0)
            extrapolated = anchors.abs()[None] > observed_amp.amax(-1)[:, None]
            coverage = 1-extrapolated.float().mean(-1)
            residual_score = -10*nmse.clamp_min(1e-12).log10()
            audit_design = torch.cat((n, p), -1)*active_rows[..., None]
            audit_design = audit_design/audit_design.square().sum(-2, keepdim=True).sqrt().clamp_min(1e-12)
            eigenvalues = torch.linalg.eigvalsh(audit_design.transpose(-2, -1)@audit_design).clamp_min(0)
            log_condition = (eigenvalues[:, -1].clamp_min(1e-12)/eigenvalues[:, 0].clamp_min(1e-12)).log()
            quality = {"coverage": coverage, "log_condition": log_condition,
                "effective_rank": column_active.sum(-1), "effective_sample_size": active_rows.sum(-1)/2,
                "nmse": nmse, "fit_residual_score_db": residual_score, "quality_valid": valid,
                "gram_eigenvalues": eigenvalues}
            out = {"resp_coef": theta, "resp_anchor": torch.view_as_real(response),
                   "resp_quality": quality,
                   "fusion_quality_features": torch.stack((valid.float(), coverage, residual_score), -1),
                   "quality_valid": valid, "solver_status": fit["solver_status"],
                   "reference_version": self.reference_version,
                   "reference_mode": self.reference_mode,
                   "estimator_variant": self.estimator_variant}
            if return_diagnostics:
                probe = complex_design(anchor_design)
                probe = probe*column_active[:, None]
                anchor_sensitivity = (probe@fit["response_sensitivity"]*probe).sum(-1)
                out["diagnostics"] = {**fit, "fit_nmse": nmse, "fit_residual_score_db": -10*nmse.clamp_min(1e-12).log10(),
                    "denominator_energy": energy, "anchor_extrapolated": extrapolated, "snr": "unknown",
                    "response_active_real_columns": column_active, "anchor_sensitivity_diagonal": anchor_sensitivity,
                    "reference": s, "observation": observation, "basis_amplitude_scale": basis_scale,
                    "nuisance_design": n, "response_design": p}
            return out

    @torch.no_grad()
    def cross_fit(self, x, *, guard=None):
        """Blocked two-way diagnostic; all observation-dependent quantities are fit-first."""
        x = _complex(x)
        b, t = x.shape
        dependency = self.fir_kernel.numel()//2 + (1 if self.estimator_variant == "compact8" else 2)
        if self.estimator_variant == "legacy28_old_reference":
            dependency = int(self.legacy_reference_provider.dependency_radius) + 2
        guard = dependency if guard is None else int(guard)
        if guard < dependency or t//2 <= guard+1:
            raise ValueError("cross-fit guard must cover FIR radius plus one memory sample")
        pos = torch.arange(t, device=x.device)
        a, bb = pos < t//2-guard, pos >= t//2+guard
        outputs = []
        for fit_mask, evaluation in ((a, bb), (bb, a)):
            mask = fit_mask.expand(b, -1)
            fit = self.forward(x, fit_mask=mask, return_diagnostics=True, use_cache=False)
            d = fit["diagnostics"]
            y = real_stack(d["observation"]-d["reference"])
            n, p = d["nuisance_design"], d["response_design"]
            # Same solver and penalties, with zero response columns for nuisance-only.
            null = schur_ridge(y, n, torch.zeros_like(p), alpha_eta=self.alpha_eta,
                alpha_theta=self.alpha_theta, fit_mask=torch.cat((mask, mask), -1))
            full_pred = real_stack(d["reference"]) + (n@d["eta"][..., None] + p@d["theta"][..., None]).squeeze(-1)
            null_pred = real_stack(d["reference"]) + (n@null["eta"][..., None]).squeeze(-1)
            ev = torch.cat((evaluation, evaluation)).to(y)
            obs = real_stack(d["observation"])
            denom = (obs.square()*ev).sum(-1)
            full_nmse = ((obs-full_pred).square()*ev).sum(-1)/denom.clamp_min(1e-12)
            null_nmse = ((obs-null_pred).square()*ev).sum(-1)/denom.clamp_min(1e-12)
            outputs.append({"fit": fit, "nmse_full": full_nmse, "nmse_nuisance_only": null_nmse,
                "delta_pred": null_nmse-full_nmse, "denominator_energy": denom,
                "evaluation_valid": torch.isfinite(full_nmse) & denom.gt(1e-12),
                "effective_points": int(evaluation.sum()), "guard": guard,
                "discarded_boundary_points": 2*guard})
        return outputs

    def bundle_metadata(self):
        return {"version": "ecrs_v2", "reference_mode": self.reference_mode,
            "estimator_variant": self.estimator_variant, "complex_basis_dim": self.basis_dim,
            "reference_version": self.reference_version, "public_reference_verified_by_module": False,
            "fir_kernel": self.fir_kernel.detach().cpu().tolist(), "scale": "fixed_input_units_fit_only_column_rms",
            "nuisance_parameters": ["real_gain", "real_phase", "real_normalized_cfo"],
            "complex_response_basis": list(BASIS_NAMES) if self.basis_dim == 8 else
                [f"legacy_{block}_{i}" for block, count in (("pa",8),("iq",8),("cross",4),("slew",8)) for i in range(count)],
            "basis_gate_scale": "fixed_input_units" if self.basis_dim == 8 else "fit_only_p95_amplitude; propagated_to_public_anchor_probe",
            "anchor_mode": self.anchor_mode,
            "alpha_eta": self.alpha_eta, "alpha_theta": self.alpha_theta,
            "uncertainty_kind": "ridge_sensitivity", "cross_fit": "diagnostic_only",
            "alignment": "frozen_legacy_canonicalizer_fit_only" if self.estimator_variant == "legacy28_old_reference" else
                "no_packet_alignment_estimator; nuisance_linearization_only",
            "bridge_shared_contract": "frozen_physics; W=I; real3_nuisance; column_RMS_ridge; explicit_history_anchors; common_neural_readout",
            "bridge_claim_boundary": "B3a vs historical B2 is a frozen estimator/readout package; B3b changes reference and canonical frame; B3c changes basis family and gate scaling, not only dimension"}


def tangent_fusion(z_raw: Tensor, projected: Tensor, rho: float = .05) -> Tensor:
    if not math.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and nonnegative")
    z = F.normalize(z_raw, dim=-1)
    u = projected - (projected*z).sum(-1, keepdim=True)*z
    v = u/u.norm(dim=-1, keepdim=True).clamp_min(1)
    fused = F.normalize(z+rho*v, dim=-1)
    return torch.where(z_raw.norm(dim=-1, keepdim=True) > 1e-12, fused, torch.zeros_like(fused))


class ECRSV2Branch(nn.Module):
    def __init__(self, *, raw_dim=160, resp_dim=64, reference_mode="estimated_reference",
                 anchor_mode="complex24", alpha_eta=.01, alpha_theta=.01, fixed_rho=.05,
                 fusion_mode="off", update_resp_from_fusion=False, public_reference=None,
                 reference_version="ecrs_v2_ref_v1", fir_kernel=(.25, .5, .25), cache_fixed_operator=True,
                 estimator_variant="compact8", legacy_reference_provider=None, legacy_basis=None):
        super().__init__()
        if fusion_mode not in ("off", "fixed"):
            raise ValueError("V2 currently supports off/fixed fusion; dynamic gate is deferred")
        if not math.isfinite(fixed_rho) or not 0 <= fixed_rho <= .25:
            raise ValueError("fixed_rho must be in [0,.25]")
        self.physical = ECRSV2PhysicalEstimator(reference_mode=reference_mode, public_reference=public_reference,
            reference_version=reference_version, alpha_eta=alpha_eta, alpha_theta=alpha_theta,
            fir_kernel=fir_kernel, anchor_mode=anchor_mode, cache_fixed_operator=cache_fixed_operator,
            estimator_variant=estimator_variant, legacy_reference_provider=legacy_reference_provider, legacy_basis=legacy_basis)
        dim = 48 if anchor_mode == "complex24" else 16
        self.response_encoder = nn.Sequential(nn.Linear(dim, 96), nn.SiLU(), nn.Linear(96, resp_dim))
        self.resp_to_id = nn.Linear(resp_dim, raw_dim, bias=False)
        nn.init.zeros_(self.resp_to_id.weight)
        self.fixed_rho, self.fusion_mode = fixed_rho, fusion_mode
        self.register_buffer("active_rho", torch.tensor(fixed_rho if fusion_mode == "fixed" else 0.))
        self.update_resp_from_fusion = update_resp_from_fusion
        self.resp_dim, self.raw_dim = resp_dim, raw_dim

    @property
    def encoder(self):
        return self.response_encoder

    @property
    def response_projection(self):
        return self.resp_to_id

    def set_active_fusion(self, rho, mode=None):
        if mode is not None:
            if mode not in ("off", "fixed"):
                raise ValueError("only off/fixed fusion is implemented")
            self.fusion_mode = mode
        if not 0 <= float(rho) <= .25:
            raise ValueError("active rho must be in [0,.25]")
        self.active_rho.fill_(float(rho) if self.fusion_mode == "fixed" else 0.)

    def get_extra_state(self):
        return {"fusion_mode": self.fusion_mode}

    def set_extra_state(self, state):
        self.fusion_mode = state["fusion_mode"]

    def forward(self, x, z_raw=None, return_diagnostics=False):
        out = self.physical(x, return_diagnostics=return_diagnostics)
        r = out["resp_anchor"].flatten(1).detach()
        out["encoder_input"] = r
        encoded = F.normalize(self.response_encoder(r), dim=-1)
        z_resp = torch.where(out["quality_valid"][:, None], encoded, torch.zeros_like(encoded))
        out["z_resp"] = z_resp
        if z_raw is None:
            return out
        out["z_id_raw"] = z_raw
        if self.fusion_mode == "off":
            out["z_id_fused"] = z_raw
        else:
            response = z_resp if self.update_resp_from_fusion else z_resp.detach()
            out["z_id_fused"] = tangent_fusion(z_raw.detach(), self.resp_to_id(response), float(self.active_rho))
        out["rho"] = self.active_rho.to(z_raw).expand(z_raw.shape[0], 1)
        out["rho_resp"] = out["rho"].squeeze(-1)
        return out

    def bundle_metadata(self):
        return {**self.physical.bundle_metadata(), "raw_dim": self.raw_dim, "resp_dim": self.resp_dim,
            "fusion_mode": self.fusion_mode, "fixed_rho": self.fixed_rho,
            "update_resp_from_fusion": self.update_resp_from_fusion,
            "physical_gradient": "detached_before_encoder", "fusion_raw_gradient": "detached"}

    def export_bundle(self):
        return {"schema": "ecrs_v2_bundle_v1", "metadata": self.bundle_metadata(),
                "state_dict": self.state_dict()}
