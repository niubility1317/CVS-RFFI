"""Exact zero-bias score primitive from the existing matched CSIL port.

Only this inference primitive is needed; the historical deployment/receipt
machinery in the original module is deliberately outside this snapshot.
"""
import torch


def zero_bias_logits(features, fingerprints, *, norm_mag=5.0, epsilon=1e-9):
    feature_norm = torch.sqrt(torch.sum(features.square(), dim=1, keepdim=True) + float(epsilon) ** 2)
    fingerprint_norm = torch.sqrt(torch.sum(fingerprints.square(), dim=1, keepdim=True) + float(epsilon) ** 2)
    return (features / feature_norm) @ (fingerprints / fingerprint_norm).t() * float(norm_mag) + float(norm_mag)
