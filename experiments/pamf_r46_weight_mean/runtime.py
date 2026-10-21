import math
import time
from pathlib import Path
import torch
from . import design as d
from .models import CompactSuccessor,source_from_parent
from .averaging import SourceParameterMeans
from experiments.pamf_r44_pruned_refine.runtime import load_checkpoint as load_parent
from experiments.pamf_r46_anchor_refine.runtime import validate_parent_root
from experiments.pamf_uncertainty_restore import design as ancestor

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='R46_shared_source_parameter_readout_checkpoint_v1'


def validate_aliases(model,state=None):
    if sum(p.numel() for p in model.parameters())!=273765:raise ValueError('Exact existing compressed graph required')
    norms=[m for m in model.modules() if isinstance(m,torch.nn.GroupNorm)]
    if len(norms)!=168 or any(m.num_groups!=4 for m in norms):raise ValueError('Original ordinary GN4 partitions required')
    groups={}
    for name,p in model.named_parameters(remove_duplicate=False):groups.setdefault(id(p),[]).append(name)
    if state is not None:
        for names in groups.values():
            if len(names)>1 and any(not torch.equal(state[names[0]],state[name]) for name in names[1:]):
                raise ValueError('Conflicting shared prior parameter aliases')
    return model


def metadata(arm,epoch,updates,parent,initialization,average):
    return dict(schema=SCHEMA,config=d.configuration(),kind=d.KIND,arm=arm,epoch=epoch,updates=updates,
        parent=parent,initialization=initialization,parameter_readout=average,structural_smoke=epoch==0,
        source_path=d.SOURCE_ARM,source_path_training_counted_once=True,
        parent_R46_run=d.COMPACT_PARENT_RUN,parent_R46_commit=d.COMPACT_PARENT_COMMIT,parent_R46_row=d.COMPACT_PARENT_ROW,
        parent_R46_epoch=20,parent_R46_updates=4940,parent_R17_run=d.PARENT_RUN,parent_R17_commit=d.PARENT_COMMIT,
        inherited_component_epochs=80,cumulative_component_epochs=80+epoch,parent_training_TX_F_target_reads=0,
        training_TX_labels_read=0,training_F_access=False,F_weights_read=0,F_outputs_read=0,
        target_training=False,V_training=False,teacher_parameter_updates=0,F_parameter_updates=0,head_fit_updates=0,
        initialization_IQ_examples=0,new_calibration_IQ=0,external_student_checkpoint_reads=0,
        new_trainable_parameters=0,extra_training_neural_forwards=0)


def validate_payload(value):
    arm=value.get('arm');epoch=value.get('epoch');updates=value.get('updates')
    init=value.get('initialization');average=value.get('parameter_readout')
    if arm not in d.ARMS or type(epoch) is not int or type(updates) is not int or (epoch,updates) not in ((0,0),(20,4940)):
        raise ValueError('Only structural zero-update smoke or fixed complete source readouts allowed')
    if not isinstance(init,dict) or not isinstance(average,dict):raise ValueError('Original source initialization and averaging metadata required')
    expected=metadata(arm,epoch,updates,value.get('parent'),init,average)
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Strict direct M35 source-only contract required')
    parent=value['parent'];required=dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,
        action_run=ancestor.ACTION_RUN,action_commit=ancestor.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)
    if not isinstance(parent,dict) or any(parent.get(k)!=v for k,v in required.items()):raise ValueError('Matching original source ancestry required')
    required=dict(mode='original_M35_static_prune_shared_source_path',group_count=4,IQ_examples=0,TX_labels_read=0,
        F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,parent_unchanged=True,exact_function_equivalence=False,
        fixed_original_maps=True,parameter_mean_readouts=list(d.ARMS),external_student_checkpoint_reads=0,
        new_calibration_IQ=0,new_trainable_parameters=0)
    if any(init.get(k)!=v for k,v in required.items()) or not math.isfinite(init.get('seconds',-1)) or init.get('seconds',-1)<0:
        raise ValueError('Direct weight-only static M35 compression required')
    required=dict(readout=arm,rule=d.configuration()['readouts'][arm],shared_source_epochs=epoch,shared_source_updates=updates,
        ema_updates=2470 if epoch else 0,late_epoch_snapshots=5 if epoch else 0,
        unique_parameters_averaged=714,parameter_aliases=770,buffers_averaged=False,teacher_changed=False,
        extra_neural_forwards=0,new_trainable_parameters=0,TX_reads=0,F_reads=0,V_reads=0,target_reads=0)
    if any(average.get(k)!=v for k,v in required.items()):raise ValueError('Fixed terminal,late-mean or step-EMA readout contract differs')


def initialize(root,device):
    validate_parent_root(root)
    teacher,payload=load_parent(Path(root)/d.COMPACT_PARENT_ROW/d.KIND/'final.pt','cpu')
    if (payload['arm'],payload['epoch'],payload['updates'])!=(d.COMPACT_PARENT_ROW,20,4940):raise ValueError('User-designated source-final M35 required')
    parent=payload['parent'];del payload;teacher.eval().requires_grad_(False);started=time.perf_counter()
    model=source_from_parent(teacher);model.initialization_report['seconds']=time.perf_counter()-started
    empty=SourceParameterMeans(model)
    validate_payload(metadata('final',0,0,parent,model.initialization_report,empty.metadata('final')))
    return validate_aliases(model.to(device)),teacher.to(device),parent


def save_checkpoint(path,state,arm,epoch,updates,parent,initialization,average):
    value=metadata(arm,epoch,updates,parent,initialization,average);validate_payload(value)
    skeleton=CompactSuccessor(arm);validate_aliases(skeleton,state);skeleton.load_state_dict(state,strict=True)
    value.update(model=state,optimizer=None)
    with Path(path).open('xb') as stream:torch.save(value,stream)


def load_checkpoint(path,device):
    value=torch.load(path,map_location='cpu',weights_only=False);validate_payload(value)
    if value.get('optimizer') is not None:raise ValueError('Ordinary inference readout must not include optimizer')
    model=CompactSuccessor(value['arm']);validate_aliases(model,value['model']);model.load_state_dict(value['model'],strict=True)
    model.initialization_report=value['initialization'];model.parameter_readout=value['parameter_readout']
    return validate_aliases(model.to(device)).eval(),value
