import math
import torch
from torch import nn
from . import design as d
from experiments.pamf_reception_restore.models import Core,ComplexTFUNet
from experiments.pamf_residual_restore.models import ResidualDeterministic
from experiments.pamf_change_preserve.runner import load_restorer
from experiments.pamf_feature_refine.models import feature_distance
from experiments.pamf_gradient_balance.losses import posterior_distance


def paired_state(anchor,clean,time):
    if (anchor.shape!=clean.shape or anchor.ndim!=3 or anchor.shape[1:]!=(2,256) or
        time.shape!=(len(anchor),) or not torch.isfinite(time).all() or (time<0).any() or (time>1).any()):
        raise ValueError('Matched source IQ pair and finite frame time0..1 required')
    return anchor+time[:,None,None]*(clean-anchor)


class AnchoredFlow(nn.Module):
    def __init__(self,parent,arm):
        super().__init__();d.configuration(arm);self.arm=arm
        self.parent=parent.requires_grad_(False).eval()
        self.field=Core(incoming=7,outgoing=2,widths=d.WIDTHS)

    def train(self,mode=True):
        super().train(mode);self.parent.eval();return self

    def anchor(self,received):
        with torch.no_grad():return self.parent(received)

    def velocity(self,received,anchor,point,time):
        if (received.ndim!=3 or received.shape[1:]!=(2,256) or anchor.shape!=received.shape or point.shape!=received.shape or
            time.shape!=(len(received),) or not torch.isfinite(time).all() or (time<0).any() or (time>1).any()):
            raise ValueError('Current-frame IQ/state and time0..1 required')
        scale=anchor.square().mean((1,2),keepdim=True).clamp_min(1e-4).sqrt()
        inputs=torch.cat((received/scale,anchor/scale,(point-anchor)/scale,time[:,None,None].expand(-1,1,256)),1)
        return scale*self.field(inputs)

    def training_prediction(self,received,clean,time):
        if self.arm==d.ARMS[0] and torch.count_nonzero(time):raise ValueError('Direct residual control requires time0')
        anchor=self.anchor(received);point=paired_state(anchor,clean,time)
        velocity=self.velocity(received,anchor,point,time);target=(clean-anchor).detach()
        energy=clean.square().mean((1,2)).clamp_min(1e-4)
        error=(velocity-target).square().mean((1,2))/energy
        # One velocity predicts the full clean-minus-anchor displacement. This
        # source proxy is not the four-step sampled output used in evaluation.
        return anchor+velocity,error

    def forward(self,received):
        anchor=self.anchor(received);point=anchor
        steps=d.configuration(self.arm)['inference_steps']
        for i in range(steps):
            time=received.new_full((len(received),),i/steps)
            point=point+self.velocity(received,anchor,point,time)/steps
        return point


def initialize(path,device,arm):
    return AnchoredFlow(load_restorer(path),arm).to(device).eval()


def save_checkpoint(path,model,optimizer,arm,epoch,updates,field_state=None):
    torch.save(dict(schema='R03_anchored_flow_checkpoint_v1',config=d.configuration(arm),arm=arm,epoch=epoch,updates=updates,
        model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict(),
        scale=float(model.parent.residual_scale),parent_run=d.PARENT_RUN,parent_commit=d.PARENT_COMMIT,
        parent_corrector_epochs=20,parent_coarse_epochs=20,coarse_updates=0,correction_updates=0,
        TX_labels_read=0,target_training=False,F_optimizer_updates=0,head_fit_updates=0,field_state=field_state),path)


def load_checkpoint(path,arm,device):
    value=torch.load(path,map_location='cpu',weights_only=False)
    if (value['schema']!='R03_anchored_flow_checkpoint_v1' or value['config']!=d.configuration(arm) or value['arm']!=arm or
        value['parent_run']!=d.PARENT_RUN or value['parent_commit']!=d.PARENT_COMMIT or
        value['parent_coarse_epochs']!=20 or value['parent_corrector_epochs']!=20 or value['coarse_updates']!=0 or value['correction_updates']!=0 or
        value['TX_labels_read']!=0 or value['target_training'] or value['F_optimizer_updates']!=0 or value['head_fit_updates']!=0 or
        not 0<=value['epoch']<=d.EPOCHS or not 0<=value['updates']<=d.UPDATES or not math.isfinite(value['scale']) or value['scale']<=0):
        raise ValueError('Fixed source-only anchored-flow checkpoint required')
    parent=ResidualDeterministic(ComplexTFUNet(),value['scale'])
    model=AnchoredFlow(parent,arm).to(device);model.load_state_dict(value['model'],strict=True)
    return model.eval().requires_grad_(False),value
