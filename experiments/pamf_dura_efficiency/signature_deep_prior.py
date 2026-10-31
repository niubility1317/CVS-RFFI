"""Improve each prior SIG predictor with already computed deeper features."""
import torch
from torch import nn
from torch.nn import functional as F
from .signature_context_prior import SignatureContextPrior

DEEP_DIMENSION=192
SEED_OFFSET=2026103121
ADDED={'linear':61760,'mlp':113280}
PARAMETERS={'linear':406055,'mlp':457575}
MACS={'linear':25206312,'mlp':25257512}


class DeepSignaturePrior(SignatureContextPrior):
    def __init__(self,prior,seed,mode):
        nn.Module.__init__(self)
        if mode not in ADDED or not prior.decoder_context:raise ValueError('Main SIG context prior and fixed deep mode required')
        self.encoder,self.decode,self.out=prior.encoder,prior.decode,prior.out
        self.signature_projection,self.signature_conditioner=prior.signature_projection,prior.signature_conditioner
        self.decoder_context=True;self.conditional=False;self.deep_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            self.deep_signature_projection=nn.Linear(DEEP_DIMENSION,160) if mode=='linear' else nn.Sequential(nn.Linear(DEEP_DIMENSION,160),nn.SiLU(),nn.Linear(160,160))
            last=self.deep_signature_projection if mode=='linear' else self.deep_signature_projection[-1]
            nn.init.zeros_(last.weight);nn.init.zeros_(last.bias)

    def forward_with_signature(self,value):
        if value.ndim!=3 or value.shape[1:]!=(9,256):raise ValueError('Actual prior Bx9x256 input required')
        features=self.encoder(value)
        shallow=self.signature_projection(F.adaptive_avg_pool1d(features[1],8).flatten(1))
        deep=torch.cat((F.adaptive_avg_pool1d(features[2],4).flatten(1),F.adaptive_avg_pool1d(features[3],2).flatten(1)),1)
        signature=shallow+self.deep_signature_projection(deep)
        return self._decode(features,signature),signature


def attach(model,seed,mode):
    if mode not in ADDED or len(model.prior)!=2 or sum(p.numel() for p in model.parameters())!=344295 or any(type(p) is not SignatureContextPrior for p in model.prior):
        raise ValueError('One original main two-prior SIG-context G required')
    model.prior=nn.ModuleList(DeepSignaturePrior(prior,int(seed)+SEED_OFFSET+i,mode) for i,prior in enumerate(model.prior))
    if sum(p.numel() for p in model.parameters())!=PARAMETERS[mode]:raise ValueError('Full deep-SIG parameter count differs')
    model.initialization_report=dict(model.initialization_report,deep_SIG_parameters=ADDED[mode],deep_SIG_mode=mode,
        deep_SIG_dimension=DEEP_DIMENSION,deep_SIG_pool_bins=[4,2],deep_SIG_encoder_levels=[2,3],deep_SIG_zero_output_init=True,
        deep_SIG_seed_offset=SEED_OFFSET,deep_SIG_CPU_CUDA_rng_preserved=True)
    return model
