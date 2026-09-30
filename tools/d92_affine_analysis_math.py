"""Independent analysis mathematics for the draft affine candidate.

This module neither fits a head nor imports the candidate core.  Its sole
operation is the complete fixed-reference centering VJP in rank-one form.
"""

import numpy as np


def center_kernel_vjp(bar_K, bar_L, q, gamma):
    """Return the symmetric raw-train and raw-cross kernel adjoints.

    Let ``P = I - ones(n, 1) @ q[None, :]``, with fixed q, and define
    ``K = gamma * P @ R @ P.T`` and
    ``L = gamma * (Q - ones(h, 1) @ (q @ R)[None, :]) @ P.T``.
    R is a symmetric n-by-n raw kernel; Q is h-by-n.  For incoming adjoints
    bar_K[n,n] and bar_L[h,n], the result is

      gamma * sym(P.T @ bar_K @ P - q[:,None] * (ones(h) @ bar_L @ P)),
      gamma * bar_L @ P.

    The implementation uses only axis reductions, broadcast outer products
    and elementwise operations: O(n*n + h*n) time and extra memory.  It does
    not form P or perform any dense matrix multiplication.  Both reference
    endpoint terms are retained; affine saddle cancellation is a separate
    caller check, never a shortcut used here.

    Inputs are converted to float64 without mutation.  n must be positive;
    h may be zero.  gamma must be a finite scalar.  q is used exactly as
    supplied; the caller certifies its physical reference measure, and this
    function does not renormalize it.  The algebra also applies to general
    incoming bar_K; its raw symmetric-kernel adjoint is symmetrized here.
    Shape or nonfinite input raises ValueError; unrepresentable arithmetic
    raises FloatingPointError.  Reassociation is not bitwise equivalent to
    the dense-P oracle in floating point.
    """
    A = np.asarray(bar_K, dtype=np.float64)
    B = np.asarray(bar_L, dtype=np.float64)
    weights = np.asarray(q, dtype=np.float64)
    scale = np.asarray(gamma, dtype=np.float64)
    if (A.ndim != 2 or A.shape[0] == 0 or A.shape[1] != A.shape[0]
            or B.ndim != 2 or B.shape[1] != A.shape[0]
            or weights.shape != (A.shape[0],) or scale.shape != ()):
        raise ValueError('Centering VJP shape mismatch')
    if not all(np.isfinite(x).all() for x in (A, B, weights, scale)):
        raise ValueError('Centering VJP requires finite numeric inputs')
    scalar = float(scale)
    if scalar == 0.:
        return np.zeros_like(A), np.zeros_like(B)

    with np.errstate(over='raise', invalid='raise'):
        # B @ P.  Its column sum is ones(h) @ B @ P, including the
        # moving-reference contribution in the train-kernel adjoint.
        cross = B - B.sum(axis=1, keepdims=True)*weights[None, :]
        column_sum = A.sum(axis=0)
        row_sum = A.sum(axis=1)
        total = float(A.sum())
        full = (A - weights[:, None]*column_sum[None, :]
                - row_sum[:, None]*weights[None, :]
                + total*weights[:, None]*weights[None, :])
        full -= weights[:, None]*cross.sum(axis=0)[None, :]
        # Scale the summands before adding, avoiding needless overflow in
        # full + full.T.  This remains the full symmetric raw-R adjoint.
        bar_R = scalar*(.5*full + .5*full.T)
        bar_Q = scalar*cross
    if not np.isfinite(bar_R).all() or not np.isfinite(bar_Q).all():
        raise FloatingPointError('Centering VJP result is not representable')
    return bar_R, bar_Q
