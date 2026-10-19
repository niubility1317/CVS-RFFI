import math
import torch
from torch import nn
from . import design as d
from .losses import posterior_distance
from experiments.pamf_feature_refine.models import feature_distance
from experiments.pamf_residual_restore.models import ResidualDeterministic
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_change_preserve.runner import load_restorer


class JointResidual(ResidualDeterministic):
    def __init__(self,coarse,scale):
        super().__init__(coarse,scale);self.requires_grad_(True)

    def context(self,received):
        base=self.coarse(received)
        return base,torch.cat((received,base),1)

    def train(self,mode=True):return nn.Module.train(self,mode)


def initialize(parent_checkpoint,device):
    parent=load_restorer(parent_checkpoint)
    model=JointResidual(ComplexTFUNet(),float(parent.residual_scale));model.load_state_dict(parent.state_dict(),strict=True)
    return model.to(device).eval()


def save_checkpoint(path,model,optimizer,arm,epoch,updates,weighting_state=None):
    torch.save(dict(schema='hardness_frozenF_checkpoint_v1',config=d.configuration(arm),arm=arm,epoch=epoch,updates=updates,
        model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict(),scale=float(model.residual_scale),
        parent_run=d.PARENT_RUN,parent_commit=d.PARENT_COMMIT,parent_corrector_epochs=20,parent_coarse_epochs=20,
        TX_labels_read=0,target_training=False,F_optimizer_updates=0,head_fit_updates=0,coarse_updates=updates,weighting_state=weighting_state),path)


def load_checkpoint(path,arm,device):
    value=torch.load(path,map_location='cpu',weights_only=False)
    if (value['schema']!='hardness_frozenF_checkpoint_v1' or value['config']!=d.configuration(arm) or value['arm']!=arm or
        value['parent_run']!=d.PARENT_RUN or value['parent_commit']!=d.PARENT_COMMIT or value['parent_corrector_epochs']!=20 or
        value['parent_coarse_epochs']!=20 or value['TX_labels_read']!=0 or value['target_training'] or value['F_optimizer_updates']!=0 or
        value['head_fit_updates']!=0 or value['coarse_updates']!=value['updates'] or not 0<=value['epoch']<=d.EPOCHS or
        not 0<=value['updates']<=d.UPDATES or not math.isfinite(value['scale']) or value['scale']<=0):raise ValueError('Joint checkpoint budget/permissions/provenance mismatch')
    model=JointResidual(ComplexTFUNet(),value['scale']).to(device);model.load_state_dict(value['model'],strict=True)
    return model.eval().requires_grad_(False),value
