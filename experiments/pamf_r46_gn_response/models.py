import torch
from . import design as d
from .normalization import norm_links,apply_affine
from experiments.pamf_r46_fold_compensate.models import CompactSuccessor as Existing,compact_from_parent as full_fold
from experiments.pamf_r46_multipoint_fast.models import objective as original_objective,spectral_error

def CompactSuccessor(arm):
    if arm not in d.ARMS:raise ValueError('Registered GN-response arm required')
    model=Existing('fold_full');model.arm=arm;return model

@torch.no_grad()
def compact_from_parent(parent,arm,calibration):
    if arm not in d.ARMS:raise ValueError('Registered GN-response arm required before initialization')
    from .calibration import validate_calibration
    validate_calibration(calibration);model=full_fold(parent,'fold_full',calibration['fold_calibration'])
    maps={k:[torch.tensor(v,dtype=torch.int64) for v in levels] for k,levels in model.initialization_report['channel_maps'].items()}
    links=norm_links(parent,model,maps)
    if {v.name for v in links}!=set(calibration['GN_fits']):raise ValueError('Complete fixed GN fitting map required')
    for link in links:
        item=calibration['GN_fits'][link.name]
        if item['outputs']!=link.outputs.tolist():raise ValueError('Fitted GN output channels differ')
        apply_affine(link.target,item,d.GN_STRENGTH[arm])
    model.arm=arm;model.requires_grad_(True)
    model.initialization_report.update(mode=d.MODE[arm],IQ_examples=512,source_unique_IQ=128,calibration_L_exposures=256,calibration_U_exposures=256,
        GN_strength=d.GN_STRENGTH[arm],GN_ridge_fraction=d.GN_RIDGE_FRACTION,GN_slope_bounds=list(d.GN_SLOPE_BOUNDS),GN_affine_has_intercept=True,
        GN_modules=len(links),new_parent_G_inputs=256,new_student_G_inputs=256,reused_fold_parent_G_inputs=256,GN_fit_shared_once=True,
        fold_calibration_run=d.FOLD_CAL_RUN,fold_calibration_commit=d.FOLD_CAL_COMMIT,
        calibration_states_not_runtime_inputs=True)
    return model

def objective(student,teacher,clean,epoch,arm):
    if arm not in d.ARMS:raise ValueError('Registered GN-response arm required')
    return original_objective(student,teacher,clean,epoch)
