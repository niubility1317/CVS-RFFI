import math
import torch
from torch.nn import functional as F
from . import design as d
from experiments.pamf_residual_restore.models import ResidualDeterministic
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_change_preserve.runner import load_restorer


def feature_distance(estimated,reference):
    if estimated.ndim!=2 or estimated.shape!=reference.shape or estimated.shape[1]!=160:
        raise ValueError('Matched original F50 embeddings required')
    if not torch.isfinite(estimated).all() or not torch.isfinite(reference).all():
        raise FloatingPointError('Finite F50 embeddings required')
    return 1-(F.normalize(estimated,dim=1,eps=1e-8)*F.normalize(reference.detach(),dim=1,eps=1e-8)).sum(1)


def enable_correction(model):
    model.requires_grad_(False);model.correction.requires_grad_(True)
    return model


def initialize(parent_checkpoint,device):
    return enable_correction(load_restorer(parent_checkpoint).to(device))


def save_checkpoint(path,model,optimizer,arm,epoch,updates):
    torch.save(dict(schema='frozenF_feature_refinement_checkpoint_v1',config=d.configuration(arm),
        model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict(),
        arm=arm,epoch=epoch,updates=updates,scale=float(model.residual_scale),
        parent_run=d.PARENT_RUN,parent_commit=d.PARENT_COMMIT,parent_corrector_epochs=20,parent_coarse_epochs=20,
        TX_labels_read=0,target_training=False,F_optimizer_updates=0,head_fit_updates=0,coarse_updates=0),path)


def load_checkpoint(path,arm,device):
    value=torch.load(path,map_location='cpu',weights_only=False)
    if (value['schema']!='frozenF_feature_refinement_checkpoint_v1' or value['config']!=d.configuration(arm) or
        value['arm']!=arm or value['parent_run']!=d.PARENT_RUN or value['parent_commit']!=d.PARENT_COMMIT or
        value['parent_corrector_epochs']!=20 or value['parent_coarse_epochs']!=20 or
        value['TX_labels_read']!=0 or value['target_training'] or value['F_optimizer_updates']!=0 or
        value['head_fit_updates']!=0 or value['coarse_updates']!=0 or
        not 0<=value['epoch']<=d.EPOCHS or not 0<=value['updates']<=d.UPDATES or
        not math.isfinite(value['scale']) or value['scale']<=0):
        raise ValueError('Refinement checkpoint permissions/provenance/budget mismatch')
    model=ResidualDeterministic(ComplexTFUNet(),value['scale']).to(device)
    model.load_state_dict(value['model'],strict=True)
    return model.requires_grad_(False).eval(),value
