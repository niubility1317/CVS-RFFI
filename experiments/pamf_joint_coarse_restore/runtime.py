from types import SimpleNamespace
import torch
from . import design as d
from .models import JointCoarseRestorer
from experiments.pamf_fullresolution_restore import runtime as shared
from experiments.pamf_uncertainty_restore.runtime import frozen_context
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import ComplexTFUNet

ROOT=shared.ROOT
SCHEMA='joint_coarse_checkpoint_v1'


def initialize(candidate,parent_action,device):
    context,parent=frozen_context(SimpleNamespace(parent_action=parent_action),d.KIND,device)
    torch.manual_seed(d.MODEL_SEED);torch.cuda.manual_seed_all(d.MODEL_SEED)
    return JointCoarseRestorer(candidate,context).to(device),parent


def update_counts(candidate,updates):
    if candidate not in d.ROWS:raise ValueError('Registered coarse-update arm required')
    return dict(prior_updates=updates,coarse_updates=updates if candidate=='joint_coarse' else 0,action_estimator_updates=0)


def save_checkpoint(path,model,optimizer,epoch,updates,parent):
    value=dict(schema=SCHEMA,config=d.configuration(),kind=d.KIND,candidate=model.candidate,
        model=model.state_dict(),optimizer=optimizer.state_dict() if optimizer is not None else None,
        epoch=epoch,updates=updates,parent=parent,prior_initialization='scratch',inherited_prior_epochs=0,
        training_TX_labels_read=0,training_F_access=False,target_training=False,
        component_updates=update_counts(model.candidate,updates))
    validate_payload(value);torch.save(value,path)


def validate_payload(value):
    shared.validate_payload(value,configuration=d.configuration(),schema=SCHEMA,allowed_candidates=d.ROWS)
    if value.get('component_updates')!=update_counts(value['candidate'],value['updates']):
        raise ValueError('Actual coarse/prior/estimator update contract differs')


def load_checkpoint(path,device):
    value=torch.load(path,map_location=device,weights_only=False);validate_payload(value)
    model=JointCoarseRestorer(value['candidate'],ActionRestorer(ComplexTFUNet())).to(device)
    model.load_state_dict(value['model'],strict=True)
    return model.eval(),value
