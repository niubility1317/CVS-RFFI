"""Train-only source SIG regression at the actual G temporal encoder level 2."""
import torch
from torch import nn

WEIGHT=.01
CHANNELS=16
BINS=8
PARAMETERS=20640
METRIC='mid_SIG_relative_error'
GUIDE='mid_SIG_weighted_loss'


class MidSignatureProjection(nn.Module):
    def __init__(self,seed):
        super().__init__()
        # Only CPU modules are initialized here; do not reseed the CUDA probe
        # stream or advance the matched base G CPU stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed)+2026103108)
            self.projection=nn.Linear(CHANNELS*BINS,160)
            nn.init.normal_(self.projection.weight,std=1e-3)
            nn.init.zeros_(self.projection.bias)
    def forward(self,h):
        if h.ndim!=3 or h.shape[1:]!=(CHANNELS,128):
            raise ValueError('Actual G temporal encoder level2 Bx16x128 required')
        return self.projection(torch.nn.functional.adaptive_avg_pool1d(h,BINS).flatten(1))


def attach(model,seed):
    layer=model.frozen.coarse.temporal.encoder.blocks[1]
    if layer[0].net[0].out_channels!=CHANNELS or hasattr(model,'mid_signature_aux'):
        raise ValueError('Original G level2 and one new projection required')
    model.add_module('mid_signature_aux',MidSignatureProjection(seed))
    model.initialization_report=dict(model.initialization_report,mid_signature_aux_parameters=PARAMETERS,
        mid_signature_layer='frozen.coarse.temporal.encoder.blocks.1',mid_signature_shape=[16,128],
        mid_signature_training_only=True,mid_signature_init_std=.001,mid_signature_init_CPU_rng_preserved=True)
    return model


def capture(model,compute):
    observed=[]
    handle=model.frozen.coarse.temporal.encoder.blocks[1].register_forward_hook(lambda m,a,v:observed.append(v))
    try:
        result=compute()
    finally:
        handle.remove()
    if len(observed)!=1 or observed[0].ndim!=3 or observed[0].shape[1:]!=(CHANNELS,128):
        raise ValueError('Exactly one actual coarse temporal level2 feature required')
    return result,observed[0]


def auxiliary_loss(model,h,target):
    if target.shape!=(len(h),160):raise ValueError('Matched paired source clean SIG target required')
    prediction=model.mid_signature_aux(h)
    error=(prediction-target.detach()).square().sum(1)/target.detach().square().sum(1).clamp_min(1e-4)
    if not torch.isfinite(error).all():raise FloatingPointError('Nonfinite G middle SIG objective')
    return error


def auxiliary_gradient_parameters(model):
    return {
        # The existing spectral output is zero-initialized. Its encoder's
        # initial derivative is therefore zero; the output is the first
        # trainable spectral parameter that receives the auxiliary signal.
        'coarse_spectral_output':model.frozen.coarse.spectral.out.weight,
        'temporal_shallow':model.frozen.coarse.temporal.encoder.blocks[0][0].net[0].weight,
        'temporal_middle':model.frozen.coarse.temporal.encoder.blocks[1][0].net[0].weight,
        'aux_projection':model.mid_signature_aux.projection.weight}
