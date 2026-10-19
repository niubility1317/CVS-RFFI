from experiments.pamf_fullresolution_restore import runtime as shared
from . import design as d
ROOT=shared.ROOT
initialize=shared.initialize
SCHEMA='iqonly_prior_checkpoint_v1'

def save_checkpoint(path,model,optimizer,epoch,updates,parent):
    shared.save_checkpoint(path,model,optimizer,epoch,updates,parent,configuration=d.configuration(),schema=SCHEMA)

def validate_payload(value):shared.validate_payload(value,configuration=d.configuration(),schema=SCHEMA)

def load_checkpoint(path,device):return shared.load_checkpoint(path,device,configuration=d.configuration(),schema=SCHEMA)
