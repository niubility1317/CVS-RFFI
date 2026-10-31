"""A fixed, weak source-clean anchor for the existing coarse IQ output."""
import torch

CALIBRATION_RUN='20261031-diagnostic-DURA-source-module-gradients-s2026092701-r01'
CALIBRATION_RATIO=12.068895139571893
CALIBRATION_PRESSURE=.25
ALPHA=CALIBRATION_PRESSURE/CALIBRATION_RATIO


def coarse_anchor(coarse,clean):
    if coarse.ndim!=3 or coarse.shape[1:]!=(2,256) or clean.shape!=coarse.shape:
        raise ValueError('Matched source Bx2x256 coarse and clean IQ required')
    target=clean.detach()
    energy=target.square().mean((1,2)).clamp_min(1e-4)
    error=(coarse-target).square().mean((1,2))/energy
    return ALPHA*error,error
