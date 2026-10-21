import math
import time
from pathlib import Path
import torch
from . import design as d
from .models import CompactSuccessor,dense_from_parent,validate_maps
from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as DenseSkeleton
from experiments.pamf_r44_pruned_refine.runtime import load_checkpoint as load_parent
from experiments.pamf_r46_anchor_refine.runtime import validate_parent_root
from experiments.pamf_uncertainty_restore import design as ancestor

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='R46_delayed_prune_checkpoint_v1'


def metadata(arm,epoch,updates,parent,initialization):
    return dict(schema=SCHEMA,config=d.configuration(),kind=d.KIND,arm=arm,epoch=epoch,updates=updates,
        parent=parent,initialization=initialization,parent_R46_run=d.COMPACT_PARENT_RUN,
        parent_R46_commit=d.COMPACT_PARENT_COMMIT,parent_R46_row=d.COMPACT_PARENT_ROW,
        parent_R46_epoch=20,parent_R46_updates=4940,parent_R17_run=d.PARENT_RUN,parent_R17_commit=d.PARENT_COMMIT,
        parent_training_TX_F_target_reads=0,inherited_component_epochs=80,cumulative_component_epochs=80+epoch,
        training_TX_labels_read=0,training_F_access=False,F_weights_read=0,F_outputs_read=0,
        target_training=False,V_training=False,teacher_parameter_updates=0,F_parameter_updates=0,head_fit_updates=0,
        initialization_IQ_examples=0,new_calibration_IQ=0,external_student_checkpoint_reads=0,
        structural_smoke=(epoch==0 and initialization.get('phase')=='compact'))


def validate_payload(value):
    arm=value.get('arm');epoch=value.get('epoch');updates=value.get('updates');initial=value.get('initialization')
    if arm not in d.ARMS or type(epoch) is not int or type(updates) is not int or not 0<=epoch<=20:
        raise ValueError('Registered row and source training budget required')
    if (epoch==0 and updates!=0) or (epoch>0 and updates!=epoch*247):
        raise ValueError('Only complete fixed source epochs or zero-update smoke checkpoints allowed')
    if not isinstance(initial,dict):raise ValueError('Dense initialization and fixed transition plan required')
    expected=metadata(arm,epoch,updates,value.get('parent'),initial)
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Strict M35 source-only checkpoint contract required')
    required=dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,action_run=ancestor.ACTION_RUN,
        action_commit=ancestor.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)
    parent=value['parent']
    if not isinstance(parent,dict) or any(parent.get(k)!=v for k,v in required.items()):
        raise ValueError('Matching complete original source ancestry required')
    required=dict(mode='dense_M35_then_fixed_map_pruning',warmup_epochs=d.WARMUP_EPOCHS[arm],initial_parameters=350521,
        IQ_examples=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,
        parent_unchanged=True,exact_initial_function_equivalence=True,map_recomputed_after_initialization=False,
        teacher='original_M35_only')
    if any(initial.get(k)!=v for k,v in required.items()) or not math.isfinite(initial.get('seconds',-1)) or initial.get('seconds',-1)<0:
        raise ValueError('Direct exact dense M35 initialization without IQ or reader knowledge required')
    phase=initial.get('phase')
    if phase not in ('dense','compact') or (epoch>0 and phase!=d.phase(epoch,arm)):
        raise ValueError('Checkpoint graph does not match the fixed source epoch')
    if phase=='compact':
        required=dict(transition_completed_epoch=d.WARMUP_EPOCHS[arm],transition_completed_updates=d.WARMUP_EPOCHS[arm]*247,
            optimizer_resets=1,global_schedule_restarted=False,channel_maps_unchanged=True,new_calibration_IQ=0,
            external_student_checkpoint_reads=0)
        if any(initial.get(k)!=v for k,v in required.items()):raise ValueError('Fixed once-only source transition required')
    dense=DenseSkeleton('prune_refine');compact=CompactSuccessor(arm)
    validate_maps(dense,compact,initial.get('channel_maps'))


def validate_aliases(model,state=None):
    if not isinstance(model.initialization_report,dict):raise ValueError('Source phase metadata required')
    norms=[m for m in model.modules() if isinstance(m,torch.nn.GroupNorm)]
    if len(norms)!=168 or any(m.num_groups!=4 for m in norms):raise ValueError('Original GN4 partition required in both phases')
    phase=model.initialization_report['phase']
    expected=350521 if phase=='dense' else d.PARAMETERS[model.arm]
    if sum(p.numel() for p in model.parameters())!=expected:raise ValueError('Graph parameters do not match the training phase')
    groups={}
    for name,p in model.named_parameters(remove_duplicate=False):groups.setdefault(id(p),[]).append(name)
    if state is not None:
        for names in groups.values():
            if len(names)>1 and any(not torch.equal(state[names[0]],state[name]) for name in names[1:]):
                raise ValueError('Conflicting shared prior aliases')
    return model


def initialize(arm,root,device):
    if arm not in d.ARMS:raise ValueError('Fixed row before any parent IO required')
    validate_parent_root(root)
    teacher,payload=load_parent(Path(root)/d.COMPACT_PARENT_ROW/d.KIND/'final.pt','cpu')
    if (payload['arm'],payload['epoch'],payload['updates'])!=(d.COMPACT_PARENT_ROW,20,4940):
        raise ValueError('User-designated source-final M35 required')
    parent=payload['parent'];del payload;teacher.eval().requires_grad_(False);started=time.perf_counter()
    model=dense_from_parent(teacher,arm);model.initialization_report['seconds']=time.perf_counter()-started
    validate_payload(metadata(arm,0,0,parent,model.initialization_report))
    return validate_aliases(model.to(device)),teacher.to(device),parent


def save_checkpoint(path,model,optimizer,epoch,updates,parent):
    validate_aliases(model);value=metadata(model.arm,epoch,updates,parent,model.initialization_report)
    value.update(model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict())
    validate_payload(value)
    with Path(path).open('xb') as stream:torch.save(value,stream)


def load_checkpoint(path,device):
    value=torch.load(path,map_location='cpu',weights_only=False);validate_payload(value)
    model=DenseSkeleton('prune_refine') if value['initialization']['phase']=='dense' else CompactSuccessor(value['arm'])
    model.arm=value['arm'];model.initialization_report=value['initialization']
    validate_aliases(model,value['model']);model.load_state_dict(value['model'],strict=True)
    return validate_aliases(model.to(device)).eval(),value
