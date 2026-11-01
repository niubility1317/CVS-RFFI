"""Scratch IQ restorer with a continuous feature condition and clean-power support."""
import torch
from torch import nn
from torch.nn import functional as F
from .multiscale import MultiscaleAttentionRestorer


def unit_power(value):
    power=value.square().sum(1).mean(1)
    if not (torch.isfinite(power)&(power>0)).all():
        raise ValueError('Finite nonzero predicted IQ required for clean unit-power projection')
    return value*power.rsqrt()[:,None,None]


class FeatureConditionedRestorer(MultiscaleAttentionRestorer):
    def __init__(self):
        super().__init__()
        self.feature_regressor=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,512),nn.SiLU(),nn.Linear(512,160))
        self.conditions=nn.ModuleList(nn.Linear(160,2*width) for width in (256,192,128,64))
        for layer in self.conditions:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    @staticmethod
    def condition(value,feature,layer):
        gamma,beta=layer(feature).chunk(2,dim=1)
        return value*(1+gamma[:,:,None])+beta[:,:,None]

    def forward(self,received,*,return_aux=False):
        if received.ndim!=3 or received.shape[1:]!=(2,256) or received.dtype!=torch.float32:
            raise ValueError('Current per-record FP32 IQ2x256 required')
        value=self.stem(torch.cat((received,self.coordinates.expand(len(received),-1,-1)),1));skips=[]
        for encoder,down in zip(self.encoders,self.down):
            value=encoder(value);skips.append(value);value=down(value)
        tokens=self.bottleneck(value.transpose(1,2)+self.position)
        feature=self.feature_regressor(tokens.mean(1))
        value=self.condition(tokens.transpose(1,2),feature,self.conditions[0])
        for up,decoder,skip,layer in zip(self.up,self.decoders,reversed(skips),self.conditions[1:]):
            value=F.interpolate(value,size=skip.shape[-1],mode='linear',align_corners=False)
            value=self.condition(up(torch.cat((value,skip),1)),feature,layer)
            value=decoder(value)
        raw=received+self.output(value);restored=unit_power(raw)
        return (restored,dict(predicted_features=feature,raw_restored=raw)) if return_aux else restored


def latent_feature_loss(predicted,target,scale):
    if predicted.ndim!=2 or predicted.shape[1]!=160 or target.shape!=predicted.shape or scale.shape!=(160,):
        raise ValueError('Matched continuous source fused160 features required')
    target=target.detach()
    cosine=1-F.cosine_similarity(predicted,target,dim=1,eps=1e-8)
    standardized=((predicted-target).square()/scale).mean(1)
    return .1*(cosine+standardized),dict(latent_cosine_error=cosine,latent_standardized_error=standardized)


def scratch_feature_conditioned(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=FeatureConditionedRestorer()
    model.initialization_report=dict(mode='scratch_joint_single30',model_seed=seed,ancestor_checkpoints=[],
        training_stages=1,continuous_feature_dimension=160,teacher_features_are_loss_only=True)
    return model
