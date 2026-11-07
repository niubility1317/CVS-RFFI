"""Small diagnostic projection for two first-order source-loss constraints."""
import torch


def project_two_descent_halfspaces(direction, first_gradient, second_gradient):
    """Project a proposed descent direction onto both nonnegative gradient halfspaces.

    The returned vector only satisfies a first-order condition. It does not
    guarantee finite-step loss improvement or preserve an optimizer trajectory.
    """
    if (direction.ndim != 1 or direction.shape != first_gradient.shape or direction.shape != second_gradient.shape
            or direction.dtype != first_gradient.dtype or direction.dtype != second_gradient.dtype
            or direction.device != first_gradient.device or direction.device != second_gradient.device
            or not direction.is_floating_point()
            or not all(torch.isfinite(t).all() for t in (direction, first_gradient, second_gradient))):
        raise ValueError('Finite, matching floating-point gradient vectors required')
    d, a, b = (x.to(torch.float64) for x in (direction, first_gradient, second_gradient))
    aa, bb, ab, ad, bd = a@a, b@b, a@b, a@d, b@d
    candidates = [torch.zeros_like(d), d]
    if aa > 0:
        candidates.append(d - (ad/aa)*a)
    if bb > 0:
        candidates.append(d - (bd/bb)*b)
    determinant = aa*bb-ab*ab
    if determinant > 1e-12*aa*bb:
        coefficients = torch.linalg.solve(torch.stack((torch.stack((aa, ab)), torch.stack((ab, bb)))),
                                          torch.stack((ad, bd)))
        candidates.append(d-coefficients[0]*a-coefficients[1]*b)
    tolerance = 1e-12*(1+torch.linalg.vector_norm(d))
    feasible = [x for x in candidates if a@x >= -tolerance*torch.linalg.vector_norm(a)
                and b@x >= -tolerance*torch.linalg.vector_norm(b)]
    best = min(feasible, key=lambda x: float(torch.sum((x-d).square())))
    return best.to(direction.dtype)
