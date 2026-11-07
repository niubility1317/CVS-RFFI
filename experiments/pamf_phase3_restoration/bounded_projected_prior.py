"""Limit posterior prior corrections using only the current received IQ."""
import torch

from .projected_prior_detail import ProjectedPriorDetail

MAX_RELATIVE_CORRECTION=2.


def bound_correction(received,correction):
    if received.ndim!=3 or received.shape[1:]!=(2,256) or correction.shape!=received.shape:
        raise ValueError('Matched per-record IQ and posterior correction required')
    if not torch.isfinite(received).all() or not torch.isfinite(correction).all():
        raise ValueError('Finite received IQ and correction required')
    input_norm=received.detach().flatten(1).double().norm(dim=1)
    correction_norm=correction.flatten(1).double().norm(dim=1)
    cap=MAX_RELATIVE_CORRECTION*input_norm
    scale=torch.where(correction_norm<=cap,torch.ones_like(cap),cap/correction_norm.clamp_min(1e-12)).to(correction.dtype)
    return correction*scale[:,None,None],scale


class BoundedProjectedPriorDetail(ProjectedPriorDetail):
    def _readout(self,received,source_clean,*,prior=None):
        restored,aux=super()._readout(received,source_clean,prior=prior)
        base=aux['base_iq']
        raw=aux['projected_prior_update'] if source_clean is not None else restored-base
        bounded,scale=bound_correction(received,raw)
        clipped=scale<1.
        restored=torch.where(clipped[:,None,None],base+bounded,restored)
        if source_clean is not None:
            corrected=source_clean.detach()-aux['source_shrinkage']+bounded
            aux['source_noiseless_corrected_iq']=torch.where(clipped[:,None,None],corrected,aux['source_noiseless_corrected_iq'])
            aux['unbounded_projected_update']=raw
            aux['projected_prior_update']=bounded
        aux['correction_scale']=scale
        return restored,aux
