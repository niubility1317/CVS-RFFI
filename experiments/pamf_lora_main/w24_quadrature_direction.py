"""Symmetrize a frozen LoRa direction estimate in its input I/Q coordinates."""

import torch


def rotated_proposals(head, observed, base):
    """Return the original, sign-symmetrized, and quadrature-symmetrized edits."""
    original = head(observed, base)
    sign, plus, minus = head(torch.cat((-observed, 1j*observed, -1j*observed)),
                             torch.cat((-base, 1j*base, -1j*base))).chunk(3)
    sign_average = .5 * (original - sign)
    quadrature_average = .5 * (sign_average + .5j * (minus - plus))
    return original, sign_average, quadrature_average
