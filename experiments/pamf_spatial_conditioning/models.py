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
from experiments.pamf_layerwise_refine.guidance import capture
from experiments.pamf_fcrossrx_readout.frozen import from_state


class ConditionedField(Core):
    def __init__(self,arm):
        super().__init__(incoming=4,outgoing=2,widths=d.WIDTHS);d.configuration(arm);self.arm=arm
        channels=(d.MAP_CHANNELS[0],d.MAP_CHANNELS[0],d.MAP_CHANNELS[1],d.MAP_CHANNELS[1])
        self.projections=nn.ModuleList(nn.Conv1d(c,w,1) for c,w in zip(channels,d.WIDTHS))
        self.fusions=nn.ModuleList(nn.Conv1d(2*w,w,1) for w in d.WIDTHS)

    def forward(self,context,maps):
        if context.ndim!=3 or context.shape[1:]!=(4,256) or set(maps)!={'t1','t2'}:
            raise ValueError('Current IQ context and native t1/t2 maps required')
        normalized={}
        for name,channels in zip(('t1','t2'),d.MAP_CHANNELS):
            value=maps[name].detach()
            if value.ndim!=3 or value.shape[:2]!=(len(context),channels) or value.shape[2]<2 or not torch.isfinite(value).all():
                raise ValueError('Finite native per-frame BCT feature map required')
            value=value/value.square().mean((1,2),keepdim=True).clamp_min(1e-4).sqrt()
            normalized[name]=value.mean(2,keepdim=True) if self.arm==d.ARMS[0] else value
        features=self.encoder(context);fused=[]
        for value,name,projection,fusion in zip(features,('t1','t1','t2','t2'),self.projections,self.fusions):
            condition=F.interpolate(projection(normalized[name]),size=value.shape[2],mode='linear',align_corners=False)
            fused.append(fusion(torch.cat((value,condition),1)))
        value=fused[-1]
        for block,skip in zip(self.decode,fused[-2::-1]):
            value=block(torch.cat((F.interpolate(value,size=skip.shape[2],mode='linear',align_corners=False),skip),1))
        return self.out(value)


class SpatialConditionedRestorer(nn.Module):
    def __init__(self,parent,runtime_reader,arm):
        super().__init__();d.configuration(arm);self.arm=arm
        self.parent=parent.eval().requires_grad_(False);self.runtime_reader=runtime_reader.eval().requires_grad_(False)
        if tuple(getattr(self.runtime_reader.core.id_backbone,name).pw.out_channels for name in ('t1','t2'))!=d.MAP_CHANNELS:
            raise ValueError('Original F t1/t2 channels required')
        self.field=ConditionedField(arm)

    def train(self,mode=True):
        super().train(mode);self.parent.eval();self.runtime_reader.eval();return self

    def forward(self,received):
        if received.ndim!=3 or received.shape[1:]!=(2,256):raise ValueError('Current receivedIQ N2x256 required')
        with torch.no_grad():
            anchor=self.parent(received);_,maps=capture(self.runtime_reader,anchor)
        scale=anchor.square().mean((1,2),keepdim=True).clamp_min(1e-4).sqrt()
        return anchor+scale*self.field(torch.cat((received/scale,anchor/scale),1),maps)


def reader_from_state(state):
    return from_state({'id_backbone.encoder.'+k:v for k,v in state.items()},d.RUNTIME_F_SEED)


def initialize(path,device,arm,runtime_reader):
    return SpatialConditionedRestorer(load_restorer(path),runtime_reader,arm).to(device).eval()


def save_checkpoint(path,model,optimizer,arm,epoch,updates,gradient_state=None):
    torch.save(dict(schema='spatial_condition_checkpoint_v1',config=d.configuration(arm),arm=arm,epoch=epoch,updates=updates,
        model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict(),scale=float(model.parent.residual_scale),
        parent_run=d.PARENT_RUN,parent_commit=d.PARENT_COMMIT,parent_corrector_epochs=20,parent_coarse_epochs=20,
        runtime_F_seed=d.RUNTIME_F_SEED,runtime_F_commit=d.RUNTIME_F_COMMIT,coarse_updates=0,correction_updates=0,field_updates=updates,
        gradient_updates=updates,gradient_epochs=epoch,gradient_state=gradient_state,TX_labels_read=0,target_training=False,F_optimizer_updates=0,head_fit_updates=0),path)


def load_checkpoint(path,arm,device):
    item=torch.load(path,map_location='cpu',weights_only=False)
    if (item['schema']!='spatial_condition_checkpoint_v1' or item['config']!=d.configuration(arm) or item['arm']!=arm or
        item['parent_run']!=d.PARENT_RUN or item['parent_commit']!=d.PARENT_COMMIT or item['parent_coarse_epochs']!=20 or item['parent_corrector_epochs']!=20 or
        item['runtime_F_seed']!=d.RUNTIME_F_SEED or item['runtime_F_commit']!=d.RUNTIME_F_COMMIT or
        item['coarse_updates']!=0 or item['correction_updates']!=0 or item['field_updates']!=item['updates'] or item['gradient_updates']!=item['updates'] or item['gradient_epochs']!=item['epoch'] or
        any(item[k]!=0 for k in ('TX_labels_read','F_optimizer_updates','head_fit_updates')) or item['target_training'] or
        not 0<=item['epoch']<=d.EPOCHS or not 0<=item['updates']<=d.UPDATES or not math.isfinite(item['scale']) or item['scale']<=0):
        raise ValueError('Original frozenR03/F1,new-field budget and legal source checkpoint required')
    parent=ResidualDeterministic(ComplexTFUNet(),item['scale'])
    state={k[len('runtime_reader.'):]:v for k,v in item['model'].items() if k.startswith('runtime_reader.')}
    model=SpatialConditionedRestorer(parent,reader_from_state(state),arm).to(device);model.load_state_dict(item['model'],strict=True)
    return model.eval().requires_grad_(False),item
