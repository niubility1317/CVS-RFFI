import json,math
import torch
from . import design as d
from .losses import posterior_distance
from experiments.pamf_feature_refine.models import feature_distance
from experiments.pamf_joint_refine.models import JointResidual
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_residual_restore.runtime import load_coarse,parent_metadata


def source_scale(path,metadata):
    value=json.loads(path.read_text())
    if (path.parent.name!=d.CALIBRATION_RUN or value.get('status')!='SOURCE_SCALE_FROZEN' or value.get('count')!=8192 or
        value.get('source_training_only') is not True or value.get('parent')!=metadata or
        any(value.get(k)!=0 for k in ('source_V_reads','TX_labels_read','target_reads','optimizer_updates')) or
        not math.isfinite(value.get('scale',float('nan'))) or value['scale']<=0):
        raise ValueError('Matching source-only scalar from original coarse required')
    return value


def initialization_metadata(parent,scale_path):
    metadata=parent_metadata(parent);scale=source_scale(scale_path,metadata)
    return dict(coarse=metadata,corrector_initialization='scratch zero output',calibration=scale,
        coarse_epochs=20,corrector_epochs=0,trained_R03_correction_inherited=False)


def initialize(parent,scale_path,device):
    coarse,metadata=load_coarse(parent,device)
    scale=source_scale(scale_path,metadata)
    return JointResidual(coarse,scale['scale']).to(device).eval()


def save_checkpoint(path,model,optimizer,arm,epoch,updates):
    torch.save(dict(schema='layerwise_frozenF_checkpoint_v1',config=d.configuration(arm),arm=arm,epoch=epoch,updates=updates,
        model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict(),scale=float(model.residual_scale),
        parent_run=d.PARENT_RUN,parent_commit=d.PARENT_COMMIT,parent_corrector_epochs=0,parent_coarse_epochs=20,
        corrector_initialization='scratch_zero_output',calibration_run=d.CALIBRATION_RUN,
        TX_labels_read=0,target_training=False,F_optimizer_updates=0,head_fit_updates=0,coarse_updates=updates),path)


def load_checkpoint(path,arm,device):
    value=torch.load(path,map_location='cpu',weights_only=False)
    if (value['schema']!='layerwise_frozenF_checkpoint_v1' or value['config']!=d.configuration(arm) or value['arm']!=arm or
        value['parent_run']!=d.PARENT_RUN or value['parent_commit']!=d.PARENT_COMMIT or value['parent_corrector_epochs']!=0 or
        value['parent_coarse_epochs']!=20 or value['corrector_initialization']!='scratch_zero_output' or value['calibration_run']!=d.CALIBRATION_RUN or
        value['TX_labels_read']!=0 or value['target_training'] or value['F_optimizer_updates']!=0 or value['head_fit_updates']!=0 or
        value['coarse_updates']!=value['updates'] or not 0<=value['epoch']<=d.EPOCHS or not 0<=value['updates']<=d.UPDATES or
        not math.isfinite(value['scale']) or value['scale']<=0):raise ValueError('Layerwise checkpoint budget/permissions/provenance mismatch')
    model=JointResidual(ComplexTFUNet(),value['scale']).to(device);model.load_state_dict(value['model'],strict=True)
    return model.eval().requires_grad_(False),value
