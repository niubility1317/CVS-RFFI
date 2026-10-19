from types import SimpleNamespace
import torch
from . import design as d
from .models import AnalyticRestorer
from .calibration import validate_metadata
from experiments.pamf_fullresolution_restore import runtime as shared
from experiments.pamf_uncertainty_restore.runtime import frozen_context
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import ComplexTFUNet

ROOT=shared.ROOT
SCHEMA='analytic_unrolled_checkpoint_v1'


def initialize(candidate,parent_action,device):
    frozen,parent=frozen_context(SimpleNamespace(parent_action=parent_action),d.KIND,device)
    torch.manual_seed(d.MODEL_SEED);torch.cuda.manual_seed_all(d.MODEL_SEED)
    return AnalyticRestorer(candidate,frozen).to(device),parent


def save_checkpoint(path,model,optimizer,epoch,updates,parent,*,calibration=None):
    if epoch:validate_metadata(calibration)
    value=dict(schema=SCHEMA,config=d.configuration(),kind=d.KIND,candidate=model.candidate,
        model=model.state_dict(),optimizer=optimizer.state_dict() if optimizer is not None else None,
        epoch=epoch,updates=updates,parent=parent,prior_initialization='scratch',inherited_prior_epochs=0,
        training_TX_labels_read=0,training_F_access=False,target_training=False,source_scaling=calibration)
    validate_payload(value);torch.save(value,path)


def validate_payload(value):
    shared.validate_payload(value,configuration=d.configuration(),schema=SCHEMA,allowed_candidates=d.ROWS)
    if value['epoch']:validate_metadata(value.get('source_scaling'))
    elif value.get('source_scaling') is not None:validate_metadata(value['source_scaling'])


def load_checkpoint(path,device):
    value=torch.load(path,map_location=device,weights_only=False);validate_payload(value)
    model=AnalyticRestorer(value['candidate'],ActionRestorer(ComplexTFUNet())).to(device)
    model.load_state_dict(value['model'],strict=True)
    return model.eval(),value
