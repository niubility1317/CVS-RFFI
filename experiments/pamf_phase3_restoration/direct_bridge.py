"""Four deterministic paired-IQ restoration steps with one shared learned prior.

The update follows the direct-iteration form in Delbracio and Milanfar (2023),
Eq.5; extra observed-IQ conditioning, latent guidance and power support are ours.
This is not an implementation or performance claim for the original image method.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .feature_conditioned import FeatureConditionedRestorer,unit_power


def training_times(count,device):
    # Endpoint gets half the training exposure; the three later states get 1/6 each.
    values=torch.tensor([1.,1.,1.,.25,.5,.75],device=device,dtype=torch.float32)
    return values[torch.randint(6,(count,),device=device)]


def training_state(clean,received,t):
    if (clean.shape!=received.shape or clean.ndim!=3 or clean.shape[1:]!=(2,256)
            or t.shape!=(len(clean),) or clean.dtype!=torch.float32 or received.dtype!=clean.dtype
            or t.dtype!=clean.dtype or not torch.isfinite(t).all() or not ((t>0)&(t<=1)).all()):
        raise ValueError('Aligned source IQ and finite bridge times in (0,1] required')
    return (1-t[:,None,None])*clean.detach()+t[:,None,None]*received.detach()


def iterate_four(estimate,received):
    state=received
    for remaining in (4,3,2,1):
        t=torch.full((len(received),),remaining/4,device=received.device,dtype=received.dtype)
        predicted=estimate(state,received,t)
        state=predicted if remaining==1 else state+(predicted-state)/remaining
    return state


class DirectBridgeRestorer(FeatureConditionedRestorer):
    def __init__(self):
        super().__init__()
        self.stem=nn.Conv1d(6,64,5,padding=2)
        self.time_conditions=nn.ModuleList(nn.Linear(4,2*w) for w in (256,192,128,64))
        for layer in self.time_conditions:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def estimate(self,state,received,t,*,return_aux=False):
        if (state.ndim!=3 or state.shape[1:]!=(2,256) or received.shape!=state.shape
                or state.dtype!=torch.float32 or received.dtype!=state.dtype or t.shape!=(len(state),)
                or t.dtype!=state.dtype or not torch.isfinite(t).all() or not ((t>0)&(t<=1)).all()):
            raise ValueError('Current/state FP32 IQ2x256 and bridge time in (0,1] required')
        time_features=torch.stack([t,t.square(),torch.sin(math.pi*t),torch.cos(math.pi*t)],1)
        value=self.stem(torch.cat([state,received,t[:,None,None].expand(-1,1,256),self.coordinates.expand(len(state),-1,-1)],1))
        skips=[]
        for encoder,down in zip(self.encoders,self.down):
            value=encoder(value);skips.append(value);value=down(value)
        tokens=self.bottleneck(value.transpose(1,2)+self.position)
        feature=self.feature_regressor(tokens.mean(1))
        value=self.condition(tokens.transpose(1,2),feature,self.conditions[0])
        value=self.condition(value,time_features,self.time_conditions[0])
        for up,decoder,skip,layer,time_layer in zip(self.up,self.decoders,reversed(skips),self.conditions[1:],self.time_conditions[1:]):
            value=F.interpolate(value,size=skip.shape[-1],mode='linear',align_corners=False)
            value=self.condition(up(torch.cat([value,skip],1)),feature,layer)
            value=decoder(self.condition(value,time_features,time_layer))
        raw=state+self.output(value);restored=unit_power(raw)
        return (restored,dict(predicted_features=feature,raw_restored=raw)) if return_aux else restored

    def forward(self,received):
        if received.ndim!=3 or received.shape[1:]!=(2,256) or received.dtype!=torch.float32:
            raise ValueError('Current per-record FP32 IQ2x256 required')
        return iterate_four(self.estimate,received)


def scratch_direct_bridge(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=DirectBridgeRestorer()
    model.initialization_report=dict(mode='scratch_joint_single30',model_seed=seed,ancestor_checkpoints=[],
        training_stages=1,continuous_feature_dimension=160,teacher_features_are_loss_only=True,inference_core_calls=4)
    return model
