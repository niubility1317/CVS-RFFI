import math
from pathlib import Path
import time
import torch
from . import design as d
from .models import CompactSuccessor,compact_from_parent
from experiments.pamf_r44_pruned_refine import runtime as parent_runtime
from experiments.pamf_uncertainty_restore import design as ancestor
from experiments.pamf_restoration.runner import read_json

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='R46_gn_response_checkpoint_v1'


def validate_parent_root(root):
    root=Path(root)
    if root.name!=d.COMPACT_PARENT_RUN:raise ValueError('User-designated compact parent root required before IO')
    if read_json(root/'launch.json')['commit']!=d.COMPACT_PARENT_COMMIT:raise ValueError('Fixed R46 parent method required')
    plan=read_json(root/'comparison_plan.json')
    if plan!=d.parent.configuration():raise ValueError('Source-fixed R46 parent plan required')
    trained=read_json(root/d.COMPACT_PARENT_ROW/'training_complete.json')
    required=dict(status='SOURCE_TRAINED',arm=d.COMPACT_PARENT_ROW,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        parameters=350521,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,training_F_access=False,V_reads=0,target_reads=0,
        cumulative_prior_epochs=80,cumulative_coarse_epochs=80,cumulative_action_epochs=80)
    if any(trained.get(k)!=v for k,v in required.items()):raise ValueError('Matching source-only full R46 parent budget required')


def metadata(arm,epoch,updates,parent,initialization):
    return dict(schema=SCHEMA,config=d.configuration(),kind=d.KIND,arm=arm,epoch=epoch,updates=updates,parent=parent,initialization=initialization,
        parent_R46_run=d.COMPACT_PARENT_RUN,parent_R46_commit=d.COMPACT_PARENT_COMMIT,parent_R46_row=d.COMPACT_PARENT_ROW,parent_R46_epoch=20,parent_R46_updates=4940,
        parent_R17_run=d.PARENT_RUN,parent_R17_commit=d.PARENT_COMMIT,parent_training_TX_F_target_reads=0,
        inherited_component_epochs=80,cumulative_component_epochs=80+epoch,training_TX_labels_read=0,training_F_access=False,
        F_weights_read=0,F_outputs_read=0,target_training=False,V_training=False,teacher_parameter_updates=0,F_parameter_updates=0,head_fit_updates=0)


def validate_payload(value):
    arm=value.get('arm');epoch=value.get('epoch');updates=value.get('updates')
    if arm not in d.ARMS or type(epoch) is not int or type(updates) is not int or not 0<=epoch<=d.EPOCHS or (epoch==0 and updates!=0) or (epoch and not (epoch-1)*247<updates<=epoch*247):
        raise ValueError('Fixed successor epoch and arm required')
    expected=metadata(arm,epoch,updates,value.get('parent'),value.get('initialization'))
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Strict successor source and F knowledge contract required')
    required=dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,action_run=ancestor.ACTION_RUN,
        action_commit=ancestor.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)
    parent=value['parent']
    if not isinstance(parent,dict) or any(parent.get(k)!=v for k,v in required.items()):raise ValueError('Matching original source ancestry required')
    required=dict(mode=d.MODE[arm],group_count=4,IQ_examples=512,source_unique_IQ=128,calibration_L_exposures=256,calibration_U_exposures=256,
        shared_prior_maps=True,calibration_seed=d.CAL_SEED,ridge_fraction=d.RIDGE_FRACTION,fold_strength=1.,GN_strength=d.GN_STRENGTH[arm],GN_ridge_fraction=d.GN_RIDGE_FRACTION,GN_slope_bounds=list(d.GN_SLOPE_BOUNDS),no_intercept=True,
        GN_modules=d.GN_MODULES,GN_affine_has_intercept=True,new_parent_G_inputs=256,new_student_G_inputs=256,reused_fold_parent_G_inputs=256,GN_fit_shared_once=True,
        fold_calibration_run=d.FOLD_CAL_RUN,fold_calibration_commit=d.FOLD_CAL_COMMIT,
        TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,parent_unchanged=True,exact_function_equivalence=False)
    init=value['initialization']
    if not isinstance(init,dict) or any(init.get(k)!=v for k,v in required.items()) or not math.isfinite(init.get('seconds',-1)) or init.get('seconds',-1)<0:
        raise ValueError('Fixed source-calibrated R46 initialization required')


def validate_aliases(model,state=None):
    if sum(p.numel() for p in model.parameters())!=d.PARAMETERS[model.arm]:raise ValueError('Whole successor parameter count differs')
    groups={}
    for name,parameter in model.named_parameters(remove_duplicate=False):groups.setdefault(id(parameter),[]).append(name)
    if state is not None:
        for names in groups.values():
            if len(names)>1 and any(not torch.equal(state[names[0]],state[name]) for name in names[1:]):raise ValueError('Conflicting shared-weight aliases')
    return model


def initialize(arm,root,device,calibration_path):
    if arm not in d.ARMS:raise ValueError('Registered arm required before parent IO')
    validate_parent_root(root)
    teacher,value=parent_runtime.load_checkpoint(Path(root)/d.COMPACT_PARENT_ROW/d.KIND/'final.pt','cpu')
    if (value['arm'],value['epoch'],value['updates'])!=(d.COMPACT_PARENT_ROW,20,4940):raise ValueError('Fixed source-final parent required')
    parent=value['parent'];del value;teacher.eval().requires_grad_(False);started=time.perf_counter()
    from .calibration import validate_calibration
    calibration=validate_calibration(read_json(calibration_path))
    model=compact_from_parent(teacher,arm,calibration);model.initialization_report['seconds']=time.perf_counter()-started
    validate_payload(metadata(arm,0,0,parent,model.initialization_report))
    return validate_aliases(model.to(device)),teacher.to(device),parent


def save_checkpoint(path,model,optimizer,epoch,updates,parent):
    validate_aliases(model);value=metadata(model.arm,epoch,updates,parent,model.initialization_report)
    value.update(model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict());validate_payload(value)
    with Path(path).open('xb') as stream:torch.save(value,stream)


def load_checkpoint(path,device):
    value=torch.load(path,map_location='cpu',weights_only=False);validate_payload(value)
    model=CompactSuccessor(value['arm']);validate_aliases(model,value['model']);model.load_state_dict(value['model'],strict=True)
    model.initialization_report=value['initialization'];return validate_aliases(model.to(device)).eval(),value
