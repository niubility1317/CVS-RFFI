from pathlib import Path
import time
from .calibration import collect_moments,validate_calibration,none_info
import torch

from . import design as d
from .models import TaskRestorer, compact_from_teacher
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_uncertainty_restore import design as ancestor
from experiments.pamf_unrolled_restore.runtime import load_checkpoint as load_original
from experiments.pamf_restoration.runner import read_json

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'R17_task_checkpoint_v4'


def metadata(arm, epoch, updates, parent, calibration):
    return dict(schema=SCHEMA, config=d.configuration(), kind=d.KIND, arm=arm, epoch=epoch, updates=updates,
                calibration=calibration, training_F_teacher=dict(run=d.F_RUN,commit=d.F_COMMIT,seed=d.F_SEED,parameter_updates=0,head_fit_updates=0), parent=parent, parent_R17_run=d.PARENT_RUN, parent_R17_commit=d.PARENT_COMMIT,
                parent_R17_row='unroll_unshared', parent_R17_epoch=20, parent_R17_updates=4940,
                inherited_prior_epochs=20, training_TX_labels_read=0, training_F_access=True,
                target_training=False, teacher_parameter_updates=0)


def validate_payload(value):
    expected=metadata(value.get('arm'),value.get('epoch'),value.get('updates'),value.get('parent'),value.get('calibration'))
    if any(value.get(k) != v for k, v in expected.items()) or value.get('arm') not in d.ARMS:
        raise ValueError('Fixed R17 compression contract and source-only teacher required')
    validate_calibration(value.get('calibration'),value['arm'],value['epoch'])
    parent=value['parent']
    required=dict(coarse_run=ancestor.COARSE_RUN, coarse_commit=ancestor.COARSE_COMMIT,
                  action_run=ancestor.ACTION_RUN, action_commit=ancestor.ACTION_COMMIT,
                  data_contract='original7/63/30', training_TX_F_reads=0, target_training=False)
    if not isinstance(parent,dict) or any(parent.get(k)!=v for k,v in required.items()):
        raise ValueError('Original source-only R02/R07 ancestors required')
    epoch, updates=value['epoch'],value['updates']
    if not isinstance(epoch,int) or not isinstance(updates,int) or not 0<=epoch<=d.EPOCHS or not 0<=updates<=d.UPDATES or (epoch==0)!=(updates==0):
        raise ValueError('Actual fixed compression budget required')
    if epoch and not (epoch-1)*247<updates<=epoch*247:
        raise ValueError('Epoch/update exposure differs')


def validate_teacher_root(root):
    if root.name!=d.PARENT_RUN or read_json(root/'launch.json')['commit']!=d.PARENT_COMMIT:
        raise ValueError('Original R17 source-selected baseline required')
    decision=read_json(root/'source_decision.json')
    if decision['status']!='SOURCE_DECISION_FROZEN' or not decision['qualified'] or decision['selected']!='unroll_unshared' or decision['target_reads_before_decision']!=0:
        raise ValueError('R17 must have been selected on source before target access')
    trained=read_json(root/'unroll_unshared/training_complete.json')
    if (trained['epochs'],trained['updates'],trained['L_exposures'],trained['U_exposures'])!=(20,4940,126000,1134000):
        raise ValueError('Full original R17 training exposure required')
    if any(trained[k]!=0 for k in ('TX_labels_read','F_outputs_read','F_weights_read','target_reads','frozen_context_updates')):
        raise ValueError('R17 training must retain original source-only permissions')
    return trained


def initialize(arm, root, device, *, calibration_batches, calibration_info):
    if arm not in d.ARMS:raise ValueError('Registered calibrated R17 arm required before checkpoint IO')
    validate_teacher_root(root)
    teacher,value=load_original(root/'unroll_unshared'/d.KIND/'final.pt','cpu')
    if (value['candidate'],value['epoch'],value['updates'])!=('unroll_unshared',20,4940):
        raise ValueError('Original final R17 teacher required')
    teacher.eval().requires_grad_(False).to(device)
    started=time.perf_counter()
    if arm=='activation_task':moments,observed=collect_moments(teacher,calibration_batches,.5)
    else:
        moments=None;calibration_info=none_info();observed=dict(observed_examples=0,eligible_layers=0,sampled_patches_min=0,sampled_patches_max=0)
    student,details=compact_from_teacher(teacher,arm,moments)
    student.calibration_report=dict(calibration_info,**observed,factor_reports=details,seconds=time.perf_counter()-started)
    validate_calibration(student.calibration_report,arm,0 if calibration_info['mode']=='smoke2Ls' else d.EPOCHS)
    if sum(p.numel() for p in student.parameters())!=d.PARAMETERS[arm]:
        raise ValueError('Actual complete inference parameter count differs')
    return student.to(device),teacher.to(device),value['parent']


def save_checkpoint(path,model,optimizer,epoch,updates,parent):
    value=metadata(model.arm,epoch,updates,parent,model.calibration_report)
    value.update(model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict())
    validate_payload(value)
    with Path(path).open('xb') as stream:
        torch.save(value,stream)


def load_checkpoint(path,device):
    value=torch.load(path,map_location=device,weights_only=False);validate_payload(value)
    model=TaskRestorer(value['arm'],ActionRestorer(ComplexTFUNet())).to(device)
    model.load_state_dict(value['model'],strict=True)
    model.calibration_report=value['calibration']
    return model.eval(),value
