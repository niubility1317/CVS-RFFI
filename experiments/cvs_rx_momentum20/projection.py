"""FP64 Euclidean filtering of historical momentum by centered RX contrasts.

No data, model, parameter state or optimizer is read or changed. The coordinates
are supplied by the caller; this projection does not establish RX nuisance,
identity preservation, reparameterization invariance or a finite-step CE bound.
"""
import torch


def project_history(history, contrasts):
    """Return ``(u - P_H u, scalar_summary)`` for five centered RX rows.

    history is a nonempty real FP64 Tensor[D]; contrasts is a real FP64
    Tensor[5,D] on the same device. Inputs are detached locally. H is computed
    only through its5x5 Gram eigendecomposition, keeping eigenvalues strictly
    above max(1e-30,1e-10*max_eigenvalue); its rank is at most4. No threshold
    is chosen from outcome metrics. Common contrast scale preserves the result
    while it preserves this fixed numerical rank (the absolute floor matters).

    Inner products in the summary use the *recentered* contrast units times
    history units. Their normalized residual is abs(h_r dot filtered) divided
    by norm(h_r)*norm(original history), with zero denominator recorded as0.
    Discarded subthreshold directions need not have zero row residual. The
    retained-basis residual separately measures numerical orthogonality within
    the kept subspace. These are ideal FP64 vector residuals, not errors after
    FP32 optimizer-state conversion or actual parameter addition.
    """
    if (not isinstance(history, torch.Tensor) or not isinstance(contrasts, torch.Tensor)
            or history.dtype != torch.float64 or contrasts.dtype != torch.float64
            or history.layout != torch.strided or contrasts.layout != torch.strided
            or history.ndim != 1 or history.numel() == 0
            or contrasts.shape != (5, history.numel())
            or history.device != contrasts.device or history.device.type == 'meta'):
        raise ValueError('Finite dense FP64 history[D] and contrasts[5,D] on the same real device are required')
    history = history.detach()
    contrasts = contrasts.detach()
    if not bool(torch.isfinite(history).all()) or not bool(torch.isfinite(contrasts).all()):
        raise ValueError('Historical vector and RX contrasts must be finite')
    centered = contrasts - contrasts.mean(dim=0, keepdim=True)
    if not bool(torch.isfinite(centered).all()):
        raise ValueError('FP64 RX recentering overflowed')
    gram = centered @ centered.T
    if not bool(torch.isfinite(gram).all()):
        raise ValueError('FP64 contrast Gram overflowed')
    if bool((centered == 0).all()):
        eigenvalues = torch.zeros(5, dtype=torch.float64, device=history.device)
        eigenvectors = torch.eye(5, dtype=torch.float64, device=history.device)
    else:
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    if not bool(torch.isfinite(eigenvalues).all()) or not bool(torch.isfinite(eigenvectors).all()):
        raise ValueError('FP64 Gram eigendecomposition is nonfinite')
    maximum = max(0., float(eigenvalues[-1]))
    cutoff = max(1e-30, 1e-10 * maximum)
    keep = eigenvalues > cutoff
    rank = int(keep.sum())
    if rank > 4:
        raise ValueError('Recentered five RX rows unexpectedly exceed numerical rank4')
    if rank:
        basis = (eigenvectors[:, keep].T @ centered) / eigenvalues[keep].sqrt()[:, None]
        removed = basis.T @ (basis @ history)
        filtered = history - removed
        retained_products = basis @ filtered
    else:
        filtered = history.clone()
        removed = torch.zeros_like(history)
        retained_products = torch.empty(0, dtype=torch.float64, device=history.device)
    history_norm = torch.linalg.vector_norm(history)
    removed_norm = torch.linalg.vector_norm(removed)
    row_norms = torch.linalg.vector_norm(centered, dim=1)
    products = centered @ filtered
    intermediate = (filtered, removed, history_norm, removed_norm, row_norms, products, retained_products)
    if any(not bool(torch.isfinite(value).all()) for value in intermediate):
        raise ValueError('FP64 projection or residual summary overflowed')
    normalized = torch.zeros_like(products)
    nonzero_rows = row_norms > 0
    if float(history_norm) > 0:
        normalized[nonzero_rows] = products[nonzero_rows].abs() / row_norms[nonzero_rows] / history_norm
    retained_max = float(retained_products.abs().max()) if rank else 0.
    retained_relative = retained_max / float(history_norm) if float(history_norm) > 0 else 0.
    summary = dict(rank=rank, cutoff=cutoff, max_eigenvalue=maximum,
                   history_norm=float(history_norm), removed_norm=float(removed_norm),
                   removed_ratio=float(removed_norm / history_norm) if float(history_norm) > 0 else 0.,
                   centered_row_norms=row_norms.tolist(),
                   filtered_inner_products=products.tolist(),
                   normalized_inner_product_residuals=normalized.tolist(),
                   max_absolute_inner_product_residual=float(products.abs().max()),
                   max_normalized_inner_product_residual=float(normalized.max()),
                   retained_basis_max_absolute_inner_product_residual=retained_max,
                   retained_basis_max_normalized_inner_product_residual=retained_relative,
                   rank_rule='eigenvalue > max(1e-30,1e-10*max_eigenvalue)',
                   residual_units='recentered contrast times original history; normalized by row norm and original history norm',
                   ideal_FP64_only=True, actual_FP32_state_rounding_measured=False,
                   caller_establishes_source_and_coordinate_scope=True)
    return filtered, summary
