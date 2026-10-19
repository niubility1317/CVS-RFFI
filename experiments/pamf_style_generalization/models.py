import math
import torch
from torch import nn
from torch.nn import functional as F
from . import design as d
from .mixing import LAYERS,mix_statistics
from .losses import posterior_distance
from experiments.pamf_feature_refine.models import feature_distance
from experiments.pamf_reception_restore.models import Core,ComplexTFUNet
from experiments.pamf_residual_restore.models import ResidualDeterministic
from experiments.pamf_change_preserve.runner import load_restorer


class StyleCore(Core):
    def __init__(self):
        super().__init__(incoming=4)
        self.style_applications={str(i):0 for i in LAYERS}
        self.style_calls={str(i):0 for i in LAYERS}

    def forward(self,value,plan=None):
        if not self.training:return super().forward(value)
        if plan is None:raise ValueError('Source training requires registered paired style plan')
        features=[]
        for i,block in enumerate(self.encoder.blocks):
            value=block(F.avg_pool1d(value,2) if i else value)
            if i in LAYERS:
                value=mix_statistics(value,plan)
                self.style_applications[str(i)]+=2*int(plan.physical_active.sum())
                self.style_calls[str(i)]+=1
            features.append(value)
        value=features[-1]
        for block,skip in zip(self.decode,features[-2::-1]):
            value=block(torch.cat((F.interpolate(value,size=skip.shape[-1],mode='linear',align_corners=False),skip),1))
        return self.out(value)


class JointStyle(ResidualDeterministic):
    def __init__(self,coarse,scale):
        super().__init__(coarse,scale);self.correction=StyleCore();self.requires_grad_(True)

    def context(self,received):
        base=self.coarse(received)
        return base,torch.cat((received,base),1)

    def train(self,mode=True):return nn.Module.train(self,mode)

    def forward(self,received,plan=None):
        base,condition=self.context(received)
        return base+self.residual_scale*self.correction(condition,plan)


def initialize(parent_checkpoint,device):
    parent=load_restorer(parent_checkpoint)
    model=JointStyle(ComplexTFUNet(),float(parent.residual_scale))
    model.load_state_dict(parent.state_dict(),strict=True)
    return model.to(device).eval()


def save_checkpoint(path,model,optimizer,arm,epoch,updates,augmentation_state=None):
    torch.save(dict(schema='style_generalization_checkpoint_v1',config=d.configuration(arm),arm=arm,epoch=epoch,updates=updates,
        model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict(),scale=float(model.residual_scale),
        parent_run=d.PARENT_RUN,parent_commit=d.PARENT_COMMIT,parent_corrector_epochs=20,parent_coarse_epochs=20,
        TX_labels_read=0,target_training=False,F_optimizer_updates=0,head_fit_updates=0,coarse_updates=updates,
        augmentation_state=augmentation_state),path)


def load_checkpoint(path,arm,device):
    value=torch.load(path,map_location='cpu',weights_only=False)
    if (value['schema']!='style_generalization_checkpoint_v1' or value['config']!=d.configuration(arm) or value['arm']!=arm or
        value['parent_run']!=d.PARENT_RUN or value['parent_commit']!=d.PARENT_COMMIT or value['parent_corrector_epochs']!=20 or
        value['parent_coarse_epochs']!=20 or value['TX_labels_read']!=0 or value['target_training'] or value['F_optimizer_updates']!=0 or
        value['head_fit_updates']!=0 or value['coarse_updates']!=value['updates'] or not 0<=value['epoch']<=d.EPOCHS or
        not 0<=value['updates']<=d.UPDATES or not math.isfinite(value['scale']) or value['scale']<=0):
        raise ValueError('Style checkpoint budget/permissions/provenance mismatch')
    model=JointStyle(ComplexTFUNet(),value['scale']).to(device);model.load_state_dict(value['model'],strict=True)
    return model.eval().requires_grad_(False),value
