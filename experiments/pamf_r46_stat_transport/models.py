from . import design as d
from .normalization import convert_normalization
from experiments.pamf_r46_fold_compensate.models import CompactSuccessor as Existing
from experiments.pamf_r46_fold_compensate.models import compact_from_parent as full_fold
from experiments.pamf_r46_multipoint_fast.models import objective as original_objective
import torch

def CompactSuccessor(arm):
    if arm not in d.ARMS:raise ValueError('Registered statistic-transport arm required')
    model=convert_normalization(Existing('fold_full'));model.arm=arm
    if sum(p.numel() for p in model.parameters())!=d.PARAMETERS[arm]:raise ValueError('Full statistic-transport parameter count differs')
    return model

@torch.no_grad()
def compact_from_parent(parent,arm,calibration):
    if arm not in d.ARMS:raise ValueError('Registered statistic-transport initializer required')
    from .calibration import validate_calibration,aligned_links
    validate_calibration(calibration);model=convert_normalization(full_fold(parent,'fold_full',calibration['fold_calibration']))
    maps={k:[torch.tensor(v,dtype=torch.int64) for v in levels] for k,levels in model.initialization_report['channel_maps'].items()}
    links=aligned_links(parent,model,maps)
    if {v.name for v in links}!=set(calibration['stat_fits']):raise ValueError('Every fixed statistic module fit required')
    for link in links:
        item=calibration['stat_fits'][link.name]
        if item['outputs']!=link.outputs.tolist():raise ValueError('Fixed source group alignment differs')
        if arm=='calibrated':link.target.set_calibration(item['mean'],item['logvariance'])
    model.arm=arm;model.requires_grad_(True)
    model.initialization_report.update(mode=d.MODE[arm],IQ_examples=512,source_unique_IQ=128,calibration_L_exposures=256,calibration_U_exposures=256,
        statistic_initializer=arm,statistic_modules=168,new_stat_parameters=2688,stat_slope_bounds=list(d.SLOPE_BOUNDS),
        new_parent_G_inputs=256,new_identity_child_G_inputs=256,reused_fold_parent_G_inputs=256,stat_fit_shared_once=True,
        fold_calibration_run=d.FOLD_CAL_RUN,fold_calibration_commit=d.FOLD_CAL_COMMIT,calibration_states_not_runtime_inputs=True)
    return model

def objective(student,teacher,clean,epoch,arm):
    if arm not in d.ARMS:raise ValueError('Registered statistic-transport arm required')
    return original_objective(student,teacher,clean,epoch)
