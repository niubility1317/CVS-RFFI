from experiments.pamf_r17_calibrated.calibration import collect_moments,source_batches,smoke_info
from experiments.pamf_r17_calibrated.calibration import validate_calibration as validate_activation


def none_info():
    return dict(mode='none',examples=0,L_s=0,U_s=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,
        optimizer_updates=0,teacher_parameter_updates=0)


def validate_calibration(value,arm,epoch):
    if arm=='activation_task':return validate_activation(value,'activation_half',epoch)
    if arm!='step_norm_task' or not isinstance(value,dict) or any(value.get(k)!=v for k,v in none_info().items()):
        raise ValueError('Registered initialization and actual zero-calibration metadata required')
    if any(value.get(k)!=0 for k in ('observed_examples','eligible_layers','sampled_patches_min','sampled_patches_max')):
        raise ValueError('Shared-step initialization must not silently calibrate data')
