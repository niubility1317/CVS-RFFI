from pathlib import Path
import math
import time
import torch

from . import design as d
from .models import SharedFactorRestorer,compact_from_teacher
from experiments.pamf_r17_task.runtime import validate_teacher_root,load_original
from experiments.pamf_uncertainty_restore import design as ancestor

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='R17_context_compact_checkpoint_v1'


def metadata(arm,epoch,updates,parent,initialization):
    return dict(schema=SCHEMA,config=d.configuration(),kind=d.KIND,arm=arm,epoch=epoch,updates=updates,
        initialization=initialization,parent=parent,parent_R17_run=d.PARENT_RUN,parent_R17_commit=d.PARENT_COMMIT,
        parent_R17_row='unroll_unshared',parent_R17_epoch=20,parent_R17_updates=4940,inherited_prior_epochs=20,
        training_F_teacher=dict(run=d.F_RUN,commit=d.F_COMMIT,seed=d.F_SEED,parameter_updates=0,head_fit_updates=0),
        training_TX_labels_read=0,training_F_access=True,target_training=False,teacher_parameter_updates=0)


def validate_payload(value):
    arm=value.get('arm')
    expected=metadata(arm,value.get('epoch'),value.get('updates'),value.get('parent'),value.get('initialization'))
    if arm not in d.ARMS or any(value.get(k)!=v for k,v in expected.items()):
        raise ValueError('Registered original-R17 shared factor contract required')
    parent=value['parent'];required=dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,
        action_run=ancestor.ACTION_RUN,action_commit=ancestor.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)
    if not isinstance(parent,dict) or any(parent.get(k)!=v for k,v in required.items()):
        raise ValueError('Matching original source-only coarse/action ancestry required')
    report=value['initialization'];required=dict(mode='data_free_shared_SVD',fraction=d.FRACTIONS[arm],
        IQ_examples=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,pointwise_weights_shared=d.POINTWISE[arm],context_fraction=.5)
    if (not isinstance(report,dict) or any(report.get(k)!=v for k,v in required.items())
            or len(report.get('layers',[]))!=28 or len(report.get('context_layers',[]))!=84 or not math.isfinite(report.get('seconds',-1)) or report.get('seconds',-1)<0):
        raise ValueError('Data-free original teacher initialization required')
    epoch,updates=value['epoch'],value['updates']
    if type(epoch) is not int or type(updates) is not int or (epoch,updates) not in ((0,0),(20,4940)):
        raise ValueError('Fixed E20/4940 source exposure required')


def validate_aliases(model,state=None):
    if sum(p.numel() for p in model.parameters())!=d.PARAMETERS[model.arm]:
        raise ValueError('Shared parameter aliases/full inference count changed')
    if sum(p.numel() for p in model.frozen.parameters())!=d.CONTEXT_PARAMETERS or any(p.requires_grad for p in model.frozen.parameters()):
        raise ValueError('Factored source-only context must stay frozen')
    groups={}
    for name,parameter in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(parameter),[]).append(name)
    if state is not None:
        for names in groups.values():
            if len(names)>1 and any(not torch.equal(state[names[0]],state[name]) for name in names[1:]):
                raise ValueError('Conflicting checkpoint copies of one shared parameter')
    return model


def initialize(arm,root,device):
    if arm not in d.ARMS:raise ValueError('Registered compact arm required before IO')
    validate_teacher_root(root)
    teacher,value=load_original(root/'unroll_unshared'/d.KIND/'final.pt','cpu')
    if (value['candidate'],value['epoch'],value['updates'])!=('unroll_unshared',20,4940):
        raise ValueError('Original final source-selected R17 required')
    parent=value['parent'];del value
    teacher.eval().requires_grad_(False);started=time.perf_counter()
    student=compact_from_teacher(teacher,arm)
    student.initialization_report['seconds']=time.perf_counter()-started
    validate_payload(metadata(arm,0,0,parent,student.initialization_report))
    return validate_aliases(student.to(device)),teacher.to(device),parent


def save_checkpoint(path,model,optimizer,epoch,updates,parent):
    validate_aliases(model)
    value=metadata(model.arm,epoch,updates,parent,model.initialization_report)
    value.update(model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict())
    validate_payload(value)
    with Path(path).open('xb') as stream:torch.save(value,stream)


def load_checkpoint(path,device):
    # Keep serialized state on CPU, including any optimizer, out of GPU profiling.
    value=torch.load(path,map_location='cpu',weights_only=False);validate_payload(value)
    model=SharedFactorRestorer(value['arm']);validate_aliases(model,value['model'])
    model.load_state_dict(value['model'],strict=True);model.initialization_report=value['initialization']
    return validate_aliases(model.to(device)).eval(),value
