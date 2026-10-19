from types import SimpleNamespace
import torch
from . import design as d
from .models import UnrolledRestorer
from experiments.pamf_fullresolution_restore import runtime as shared
from experiments.pamf_uncertainty_restore.runtime import frozen_context
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import ComplexTFUNet
ROOT=shared.ROOT
SCHEMA='unrolled_prior_checkpoint_v1'

def initialize(candidate,parent_action,device):
    frozen,parent=frozen_context(SimpleNamespace(parent_action=parent_action),d.KIND,device)
    torch.manual_seed(d.MODEL_SEED);torch.cuda.manual_seed_all(d.MODEL_SEED)
    return UnrolledRestorer(candidate,frozen).to(device),parent

def save_checkpoint(path,model,optimizer,epoch,updates,parent):
    shared.save_checkpoint(path,model,optimizer,epoch,updates,parent,configuration=d.configuration(),schema=SCHEMA)

def validate_payload(value):shared.validate_payload(value,configuration=d.configuration(),schema=SCHEMA,allowed_candidates=d.ROWS)

def load_checkpoint(path,device):
    value=torch.load(path,map_location=device,weights_only=False);validate_payload(value)
    model=UnrolledRestorer(value['candidate'],ActionRestorer(ComplexTFUNet())).to(device);model.load_state_dict(value['model'],strict=True)
    return model.eval(),value
