"""Feature-conditioned IQ restoration with learned power, not a unit projection."""
import torch
from torch.nn import functional as F
from .feature_conditioned import FeatureConditionedRestorer


class SoftPowerRestorer(FeatureConditionedRestorer):
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
        raw=received+self.output(value)
        return (raw,dict(predicted_features=feature,raw_restored=raw)) if return_aux else raw


def scratch_soft_power(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=SoftPowerRestorer()
    model.initialization_report=dict(mode='scratch_joint_single30',model_seed=seed,ancestor_checkpoints=[],
        training_stages=1,continuous_feature_dimension=160,teacher_features_are_loss_only=True,unit_power_projection=False)
    return model


def soft_power_anchor(aux,clean):
    raw=aux['raw_restored'];reference=clean.detach()
    # Same RMS ratio as PowerAnchor for nonzero IQ. The norm has a finite zero subgradient.
    ratio=torch.linalg.vector_norm(raw,dim=(1,2))/torch.linalg.vector_norm(reference,dim=(1,2))
    error=(ratio-1).square();power=raw.square().sum(1).mean(1)
    return .1*error,dict(raw_complex_power=power,raw_clean_RMS_ratio=ratio,raw_power_anchor_error=error)
