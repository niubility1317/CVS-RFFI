import math
from pathlib import Path
import time
import torch
from . import design as d
from .models import JointCompactRestorer,compact_from_teacher
from experiments.pamf_r17_task.runtime import validate_teacher_root,load_original
from experiments.pamf_uncertainty_restore import design as ancestor

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='IQ_physics_dense_compact_checkpoint_v1'


def metadata(arm,epoch,updates,parent,initialization):
    return dict(schema=SCHEMA,config=d.configuration(),kind=d.KIND,arm=arm,epoch=epoch,updates=updates,
        parent=parent,initialization=initialization,parent_R17_run=d.PARENT_RUN,parent_R17_commit=d.PARENT_COMMIT,
        parent_R17_row='unroll_unshared',parent_R17_epoch=20,parent_R17_updates=4940,inherited_prior_epochs=20,
        training_TX_labels_read=0,training_F_access=False,F_weights_read=0,F_outputs_read=0,
        target_training=False,V_training=False,teacher_parameter_updates=0,joint_context_training=True,simulated_physics_anchors=True)


def validate_payload(value):
    arm=value.get('arm')
    if arm not in d.ARMS:raise ValueError('Registered joint compact arm required')
    expected=metadata(arm,value.get('epoch'),value.get('updates'),value.get('parent'),value.get('initialization'))
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Strict F-free joint compact contract required')
    parent=value['parent'];required=dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,
        action_run=ancestor.ACTION_RUN,action_commit=ancestor.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)
    if not isinstance(parent,dict) or any(parent.get(k)!=v for k,v in required.items()):raise ValueError('Original legal R02/R07 ancestry required')
    init=value['initialization'];required=dict(mode='group_preserving_dense_channel_slices_shared_long',group_count=4,
        IQ_examples=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,
        original_teacher_unchanged=True,exact_function_equivalence=False,context_trainable=True)
    if not isinstance(init,dict) or any(init.get(k)!=v for k,v in required.items()) or not math.isfinite(init.get('seconds',-1)) or init.get('seconds',-1)<0:
        raise ValueError('Data-free original teacher narrowing required')
    epoch,updates=value['epoch'],value['updates']
    if type(epoch) is not int or type(updates) is not int or not 0<=epoch<=20 or (epoch==0 and updates!=0) or (epoch and not (epoch-1)*247<updates<=epoch*247):
        raise ValueError('Actual fixed source epoch/update budget required')


def validate_aliases(model,state=None):
    if sum(p.numel() for p in model.parameters())!=d.PARAMETERS[model.arm]:raise ValueError('Complete dense architecture parameter count differs')
    groups={}
    for name,parameter in model.named_parameters(remove_duplicate=False):groups.setdefault(id(parameter),[]).append(name)
    if state is not None:
        for names in groups.values():
            if len(names)>1 and any(not torch.equal(state[names[0]],state[name]) for name in names[1:]):
                raise ValueError('Conflicting serialized copies of shared long weights')
    return model


def initialize(arm,root,device):
    if arm not in d.ARMS:raise ValueError('Registered arm required before checkpoint access')
    validate_teacher_root(root)
    teacher,value=load_original(root/'unroll_unshared'/d.KIND/'final.pt','cpu')
    if (value['candidate'],value['epoch'],value['updates'])!=('unroll_unshared',20,4940):raise ValueError('Original fixed R17 final required')
    parent=value['parent'];del value
    teacher.eval().requires_grad_(False);started=time.perf_counter()
    model=compact_from_teacher(teacher,arm);model.initialization_report['seconds']=time.perf_counter()-started
    validate_payload(metadata(arm,0,0,parent,model.initialization_report))
    return validate_aliases(model.to(device)),teacher.to(device),parent


def save_checkpoint(path,model,optimizer,epoch,updates,parent):
    validate_aliases(model)
    value=metadata(model.arm,epoch,updates,parent,model.initialization_report)
    value.update(model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict())
    validate_payload(value)
    with Path(path).open('xb') as stream:torch.save(value,stream)


def load_checkpoint(path,device):
    value=torch.load(path,map_location='cpu',weights_only=False);validate_payload(value)
    model=JointCompactRestorer(value['arm']);validate_aliases(model,value['model'])
    model.load_state_dict(value['model'],strict=True);model.initialization_report=value['initialization']
    return validate_aliases(model.to(device)).eval(),value
