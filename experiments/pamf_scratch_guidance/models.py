import torch
from . import design as d
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_feature_refine.models import feature_distance
from experiments.pamf_spatial_conditioning.losses import posterior_distance


def initialize(device,arm):
    d.configuration(arm)
    return ComplexTFUNet().to(device).eval()


def save_checkpoint(path,model,optimizer,arm,epoch,updates,gradient_state=None):
    torch.save(dict(schema='scratch_guidance_checkpoint_v1',config=d.configuration(arm),arm=arm,
        model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict(),
        epoch=epoch,updates=updates,gradient_state=gradient_state,initialization='scratch',restoration_parent=None,
        F_commit=d.F_COMMIT,F_seeds=list(d.TEACHER_SEEDS),F_optimizer_updates=0,head_fit_updates=0,
        TX_labels_read=0,target_training=False),path)


def load_checkpoint(path,arm,device):
    item=torch.load(path,map_location='cpu',weights_only=False)
    if (item['schema']!='scratch_guidance_checkpoint_v1' or item['config']!=d.configuration(arm) or item['arm']!=arm or
        item['initialization']!='scratch' or item['restoration_parent'] is not None or item['F_commit']!=d.F_COMMIT or
        item['F_seeds']!=list(d.TEACHER_SEEDS) or any(item[k]!=0 for k in ('F_optimizer_updates','head_fit_updates','TX_labels_read')) or
        item['target_training'] or not 0<=item['epoch']<=d.EPOCHS or item['updates']!=d.UPDATES_PER_EPOCH*item['epoch']):
        raise ValueError('Scratch restoration checkpoint contract/budget/permissions mismatch')
    model=initialize(device,arm);model.load_state_dict(item['model'],strict=True)
    return model.eval().requires_grad_(False),item
