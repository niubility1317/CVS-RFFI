from pathlib import Path
from types import SimpleNamespace
import torch
from . import design as d
from .models import ResolutionRestorer
from experiments.pamf_uncertainty_restore.runtime import frozen_context
from experiments.pamf_uncertainty_restore import design as parent_design
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import ComplexTFUNet

ROOT=Path(__file__).resolve().parents[2]


def initialize(candidate,parent_action,device):
    frozen,parent=frozen_context(SimpleNamespace(parent_action=parent_action),d.KIND,device)
    torch.manual_seed(d.MODEL_SEED);torch.cuda.manual_seed_all(d.MODEL_SEED)
    return ResolutionRestorer(candidate,frozen).to(device),parent


def save_checkpoint(path,model,optimizer,epoch,updates,parent,*,configuration=None,schema='fullresolution_prior_checkpoint_v1'):
    torch.save(dict(schema=schema,config=d.configuration() if configuration is None else configuration,kind=d.KIND,candidate=model.candidate,
        model=model.state_dict(),optimizer=optimizer.state_dict() if optimizer is not None else None,epoch=epoch,updates=updates,
        parent=parent,prior_initialization='scratch',inherited_prior_epochs=0,training_TX_labels_read=0,training_F_access=False,
        target_training=False),path)


def validate_payload(value,*,configuration=None,schema='fullresolution_prior_checkpoint_v1'):
    if (value.get('schema')!=schema or value.get('config')!=(d.configuration() if configuration is None else configuration)
            or value.get('kind')!=d.KIND or value.get('candidate') not in d.ROWS or value.get('prior_initialization')!='scratch'
            or value.get('inherited_prior_epochs')!=0 or value.get('training_TX_labels_read')!=0
            or value.get('training_F_access') is not False or value.get('target_training') is not False):
        raise ValueError('Scratch fullresolution prior contract/permissions differ')
    parent=value['parent']
    required=dict(coarse_run=parent_design.COARSE_RUN,coarse_commit=parent_design.COARSE_COMMIT,
        action_run=parent_design.ACTION_RUN,action_commit=parent_design.ACTION_COMMIT,data_contract='original7/63/30',
        training_TX_F_reads=0,target_training=False)
    if any(parent.get(key)!=item for key,item in required.items()):raise ValueError('Original source-only R02/R07 full parent required')
    epoch,updates=value.get('epoch',-1),value.get('updates',-1)
    if not 0<=epoch<=20 or not 0<=updates<=4940 or (epoch==0)!=(updates==0):raise ValueError('Fixed budget or initial checkpoint required')
    if epoch and not (epoch-1)*247<updates<=epoch*247:raise ValueError('Actual epoch/update exposure differs')


def load_checkpoint(path,device,*,configuration=None,schema='fullresolution_prior_checkpoint_v1'):
    value=torch.load(path,map_location=device,weights_only=False);validate_payload(value,configuration=configuration,schema=schema)
    model=ResolutionRestorer(value['candidate'],ActionRestorer(ComplexTFUNet())).to(device)
    model.load_state_dict(value['model'],strict=True)
    return model.eval(),value
