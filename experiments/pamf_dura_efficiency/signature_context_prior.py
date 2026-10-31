"""Predict source SIG inside each prior and use it in its IQ decoder."""
import torch
from torch import nn
from torch.nn import functional as F

WIDTHS=(12,16,28,40)
BINS=8
DIMENSION=160
BOUND=.1
WEIGHT=.01
SEED_OFFSET=2026103113
ADDED_PARAMETERS=77344
TOTAL_PARAMETERS=344295
CONV_LINEAR_MAC=25144872
AUX_ONLY_ADDED_PARAMETERS=41280
AUX_ONLY_TOTAL_PARAMETERS=308231
METRICS=('prior_context_SIG_relative_error','prior_context_SIG_weighted_loss','prior0_context_SIG_relative_error','prior1_context_SIG_relative_error')


class SignatureContextPrior(nn.Module):
    def __init__(self,core,seed,*,decoder_context=True):
        super().__init__()
        widths=tuple(b[0].net[0].out_channels for b in core.encoder.blocks)
        if getattr(core,'conditional',False) or widths!=WIDTHS:raise ValueError('Original unconditional prior Core widths required')
        self.encoder,self.decode,self.out=core.encoder,core.decode,core.out
        self.conditional=False
        self.decoder_context=bool(decoder_context)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            self.signature_projection=nn.Linear(WIDTHS[1]*BINS,DIMENSION)
            nn.init.normal_(self.signature_projection.weight,std=1e-3);nn.init.zeros_(self.signature_projection.bias)
            self.signature_conditioner=nn.Linear(DIMENSION,2*sum(WIDTHS[:-1])) if self.decoder_context else None
            if self.signature_conditioner is not None:
                nn.init.zeros_(self.signature_conditioner.weight);nn.init.zeros_(self.signature_conditioner.bias)

    def _decode(self,features,signature=None):
        modulation=self.signature_conditioner(signature) if self.decoder_context else None
        offset=0;value=features[-1]
        for block,skip,width in zip(self.decode,features[-2::-1],WIDTHS[-2::-1]):
            value=block(torch.cat((F.interpolate(value,size=skip.shape[-1],mode='linear',align_corners=False),skip),1))
            if modulation is not None:
                scale=1+BOUND*modulation[:,offset:offset+width].tanh();offset+=width
                shift=BOUND*modulation[:,offset:offset+width].tanh();offset+=width
                value=value*scale[:,:,None]+shift[:,:,None]
        return self.out(value)

    def forward_with_signature(self,value):
        if value.ndim!=3 or value.shape[1:]!=(9,256):raise ValueError('Actual prior Bx9x256 input required')
        features=self.encoder(value)
        signature=self.signature_projection(F.adaptive_avg_pool1d(features[1],BINS).flatten(1))
        return self._decode(features,signature),signature

    def forward(self,value):
        if self.decoder_context:return self.forward_with_signature(value)[0]
        return self._decode(self.encoder(value))


def attach(model,seed,*,decoder_context=True):
    if len(model.prior)!=2 or any(isinstance(p,SignatureContextPrior) for p in model.prior) or sum(p.numel() for p in model.parameters())!=266951:
        raise ValueError('One original complete two-prior G required')
    model.prior=nn.ModuleList(SignatureContextPrior(core,int(seed)+SEED_OFFSET+i,decoder_context=decoder_context) for i,core in enumerate(model.prior))
    added=sum(p.numel() for core in model.prior for module in (core.signature_projection,core.signature_conditioner) if module is not None for p in module.parameters())
    expected_added=ADDED_PARAMETERS if decoder_context else AUX_ONLY_ADDED_PARAMETERS
    expected_total=TOTAL_PARAMETERS if decoder_context else AUX_ONLY_TOTAL_PARAMETERS
    if added!=expected_added or sum(p.numel() for p in model.parameters())!=expected_total:raise ValueError('Complete RF context parameter accounting differs')
    model.initialization_report=dict(model.initialization_report,prior_SIG_context_parameters=added,prior_SIG_context_heads=2,
        prior_SIG_context_dimension=DIMENSION,prior_SIG_context_bins=BINS,prior_SIG_context_decoder_levels=3,
        prior_SIG_context_bound=BOUND,prior_SIG_context_conditioner_zero_init=True,prior_SIG_context_CPU_CUDA_rng_preserved=True,
        prior_SIG_context_seed_offset=SEED_OFFSET,prior_SIG_context_inference_active=bool(decoder_context),
        prior_SIG_context_decoder_enabled=bool(decoder_context),F_RX_scene_inputs=False)
    return model


def signature_errors(predictions,target):
    if target.ndim!=2 or target.shape[1]!=DIMENSION or len(predictions)!=2 or any(p.shape!=target.shape for p in predictions):
        raise ValueError('Two matched prior Bx160 predictions and source SIG target required')
    reference=target.detach();energy=reference.square().sum(1).clamp_min(1e-4)
    return torch.stack([(p-reference).square().sum(1)/energy for p in predictions])
