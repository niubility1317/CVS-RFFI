import math
import torch
from . import design as d
from .losses import posterior_distance
from experiments.pamf_feature_refine.models import feature_distance
from experiments.pamf_identity_priority.models import JointResidual
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_change_preserve.runner import load_restorer


def initialize(parent_checkpoint,device,arm):
    d.configuration(arm);parent=load_restorer(parent_checkpoint)
    model=JointResidual(ComplexTFUNet(),float(parent.residual_scale));model.load_state_dict(parent.state_dict(),strict=True)
    return model.to(device).eval()


def save_checkpoint(path,model,optimizer,arm,epoch,updates,gradient_state=None):
    torch.save(dict(schema='gradient_projection_checkpoint_v1',config=d.configuration(arm),arm=arm,epoch=epoch,updates=updates,
        model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict(),scale=float(model.residual_scale),
        parent_run=d.PARENT_RUN,parent_commit=d.PARENT_COMMIT,parent_corrector_epochs=20,parent_coarse_epochs=20,
        TX_labels_read=0,target_training=False,F_optimizer_updates=0,head_fit_updates=0,coarse_updates=updates,
        gradient_updates=updates,gradient_epochs=epoch,gradient_state=gradient_state),path)


def load_checkpoint(path,arm,device):
    value=torch.load(path,map_location='cpu',weights_only=False)
    if (value['schema']!='gradient_projection_checkpoint_v1' or value['config']!=d.configuration(arm) or value['arm']!=arm or
        value['parent_run']!=d.PARENT_RUN or value['parent_commit']!=d.PARENT_COMMIT or value['parent_corrector_epochs']!=20 or
        value['parent_coarse_epochs']!=20 or value['TX_labels_read']!=0 or value['target_training'] or value['F_optimizer_updates']!=0 or
        value['head_fit_updates']!=0 or value['coarse_updates']!=value['updates'] or value['gradient_updates']!=value['updates'] or
        value['gradient_epochs']!=value['epoch'] or not 0<=value['epoch']<=d.EPOCHS or not 0<=value['updates']<=d.UPDATES or
        not math.isfinite(value['scale']) or value['scale']<=0):raise ValueError('Gradient checkpoint budget/permissions/provenance mismatch')
    model=JointResidual(ComplexTFUNet(),value['scale']).to(device);model.load_state_dict(value['model'],strict=True)
    return model.eval().requires_grad_(False),value
