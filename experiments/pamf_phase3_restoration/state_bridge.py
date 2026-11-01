"""Direct IQ iteration whose estimator cannot algebraically undo a clean mixture."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .direct_bridge import DirectBridgeRestorer
from .feature_conditioned import unit_power


class StateBridgeRestorer(DirectBridgeRestorer):
    def __init__(self):
        super().__init__()
        original=self.stem
        self.stem=nn.Conv1d(4,64,5,padding=2)
        # Both stems are created from this scratch seed; no trained state is loaded.
        with torch.no_grad():
            self.stem.weight.copy_(original.weight[:,[0,1,4,5]])
            self.stem.bias.copy_(original.bias)

    def estimate(self,state,received_unused,t,*,return_aux=False):
        if (state.ndim!=3 or state.shape[1:]!=(2,256) or state.dtype!=torch.float32 or t.shape!=(len(state),)
                or t.dtype!=state.dtype or not torch.isfinite(t).all() or not ((t>0)&(t<=1)).all()):
            raise ValueError('Current intermediate FP32 IQ2x256 and time in (0,1] required')
        time_features=torch.stack([t,t.square(),torch.sin(math.pi*t),torch.cos(math.pi*t)],1)
        # No original received values enter this estimator. It is only the rollout initial state.
        value=self.stem(torch.cat([state,t[:,None,None].expand(-1,1,256),self.coordinates.expand(len(state),-1,-1)],1))
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


def scratch_state_bridge(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=StateBridgeRestorer()
    model.initialization_report=dict(mode='scratch_joint_single30',model_seed=seed,ancestor_checkpoints=[],training_stages=1,
        continuous_feature_dimension=160,teacher_features_are_loss_only=True,inference_core_calls=4,original_received_condition=False)
    return model
