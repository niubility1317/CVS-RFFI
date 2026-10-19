import math
import torch
from torch import nn
from torch.nn import functional as F
from . import design as d
from .losses import posterior_distance
from experiments.pamf_feature_refine.models import feature_distance
from experiments.pamf_reception_restore.models import Core,ComplexTFUNet
from experiments.pamf_residual_restore.models import ResidualDeterministic
from experiments.pamf_change_preserve.runner import load_restorer


class TemporalAttention(nn.Module):
    def __init__(self,arm):
        super().__init__()
        if arm not in d.ARMS:raise ValueError('Registered attention window required')
        self.norm=nn.LayerNorm(d.WIDTH);self.qkv=nn.Linear(d.WIDTH,3*d.WIDTH);self.out=nn.Linear(d.WIDTH,d.WIDTH)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
        position=torch.arange(d.POSITIONS,dtype=torch.float32)[:,None]
        scale=torch.exp(torch.arange(0,d.WIDTH,2,dtype=torch.float32)*(-math.log(10000.)/d.WIDTH))
        pe=torch.empty(d.POSITIONS,d.WIDTH);pe[:,0::2]=(position*scale).sin();pe[:,1::2]=(position*scale).cos()
        index=torch.arange(d.POSITIONS);mask=(index[:,None]-index[None,:]).abs()>d.RADIUS
        if arm==d.ARMS[1]:mask.zero_()
        self.register_buffer('position',pe,persistent=False);self.register_buffer('mask',mask,persistent=False)
        self.allowed_pairs=int((~mask).sum());self.training_calls=0;self.training_IQ_examples=0;self.training_context_pairs=0

    def forward(self,value):
        if value.ndim!=3 or value.shape[1:]!=(d.WIDTH,d.POSITIONS) or value.dtype!=torch.float32:
            raise ValueError('Current-frame FP32 corrector bottleneck128x32 required')
        tokens=value.transpose(1,2);normalized=self.norm(tokens)+self.position
        q,k,v=self.qkv(normalized).reshape(len(tokens),d.POSITIONS,3,d.HEADS,d.WIDTH//d.HEADS).permute(2,0,3,1,4).unbind(0)
        score=(q@k.transpose(-1,-2))/math.sqrt(d.WIDTH//d.HEADS)
        weights=score.masked_fill(self.mask,-torch.inf).softmax(-1)
        context=(weights@v).transpose(1,2).reshape(len(tokens),d.POSITIONS,d.WIDTH)
        if self.training:
            self.training_calls+=1;self.training_IQ_examples+=len(tokens);self.training_context_pairs+=len(tokens)*self.allowed_pairs
        return (tokens+self.out(context)).transpose(1,2)


class ContextCore(Core):
    def __init__(self,arm):
        super().__init__(incoming=4);self.attention=TemporalAttention(arm)

    def forward(self,value):
        features=self.encoder(value);value=self.attention(features[-1])
        for block,skip in zip(self.decode,features[-2::-1]):
            value=block(torch.cat((F.interpolate(value,size=skip.shape[-1],mode='linear',align_corners=False),skip),1))
        return self.out(value)


class JointContext(ResidualDeterministic):
    def __init__(self,coarse,scale,arm):
        super().__init__(coarse,scale);self.correction=ContextCore(arm);self.requires_grad_(True)

    def context(self,received):
        base=self.coarse(received)
        return base,torch.cat((received,base),1)

    def train(self,mode=True):return nn.Module.train(self,mode)


def initialize(parent_checkpoint,device,arm):
    parent=load_restorer(parent_checkpoint);model=JointContext(ComplexTFUNet(),float(parent.residual_scale),arm)
    state=model.state_dict();old=parent.state_dict()
    if set(state)-set(old)!={'correction.attention.'+k for k in model.correction.attention.state_dict()} or not set(old).issubset(state):
        raise ValueError('All original R03 state plus exactly the new attention required')
    state.update(old);model.load_state_dict(state,strict=True)
    return model.to(device).eval()


def save_checkpoint(path,model,optimizer,arm,epoch,updates,attention_state=None):
    torch.save(dict(schema='temporal_context_checkpoint_v1',config=d.configuration(arm),arm=arm,epoch=epoch,updates=updates,
        model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict(),scale=float(model.residual_scale),
        parent_run=d.PARENT_RUN,parent_commit=d.PARENT_COMMIT,parent_corrector_epochs=20,parent_coarse_epochs=20,
        TX_labels_read=0,target_training=False,F_optimizer_updates=0,head_fit_updates=0,coarse_updates=updates,
        attention_updates=updates,attention_epochs=epoch,attention_state=attention_state),path)


def load_checkpoint(path,arm,device):
    value=torch.load(path,map_location='cpu',weights_only=False)
    if (value['schema']!='temporal_context_checkpoint_v1' or value['config']!=d.configuration(arm) or value['arm']!=arm or
        value['parent_run']!=d.PARENT_RUN or value['parent_commit']!=d.PARENT_COMMIT or value['parent_corrector_epochs']!=20 or
        value['parent_coarse_epochs']!=20 or value['TX_labels_read']!=0 or value['target_training'] or value['F_optimizer_updates']!=0 or
        value['head_fit_updates']!=0 or value['coarse_updates']!=value['updates'] or value['attention_updates']!=value['updates'] or
        value['attention_epochs']!=value['epoch'] or not 0<=value['epoch']<=d.EPOCHS or not 0<=value['updates']<=d.UPDATES or
        not math.isfinite(value['scale']) or value['scale']<=0):raise ValueError('Attention checkpoint budget/permissions/provenance mismatch')
    model=JointContext(ComplexTFUNet(),value['scale'],arm).to(device);model.load_state_dict(value['model'],strict=True)
    return model.eval().requires_grad_(False),value
