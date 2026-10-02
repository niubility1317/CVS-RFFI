"""Measured six scalar gradients and actual feature outputs; no data fitting."""
import math
import statistics
from experiments.cvs_phase_curvature_identity.model import PATHS

PREFIXES=tuple('memory_'+p.replace('.','_') for p in PATHS)

def memory_state(model):
    return [(float(p.detach()),float(p.detach().tanh())) for p in model.memory_parameters()]

def memory_gradients(model):
    gradients=[]
    for p in model.memory_parameters():
        if p.grad is None or not p.grad.isfinite().all():
            raise ValueError('Missing/nonfinite feature-curvature scalar CE gradient')
        gradients.append(float(p.grad.detach()))
    if len(gradients)!=6:raise ValueError('Exactly six feature-curvature gradients required')
    return gradients

def memory_step_metrics(before,after,gradients):
    if not len(before)==len(after)==len(gradients)==6:raise ValueError('Incomplete memory measurement')
    result={'memory_gradient_used_parameters':6}
    for prefix,old,new,gradient in zip(PREFIXES,before,after,gradients):
        result.update({prefix+'_raw_before':old[0],prefix+'_coefficient_before':old[1],
            prefix+'_raw_after':new[0],prefix+'_coefficient_after':new[1],prefix+'_gradient':gradient})
    return result

def memory_epoch_metrics(state,gradient_sums,gradient_abs_sums,steps):
    result={'memory_gradient_used_parameters':6}
    for prefix,current,gradient,absolute in zip(PREFIXES,state,gradient_sums,gradient_abs_sums):
        result.update({prefix+'_raw':current[0],prefix+'_coefficient':current[1],
            prefix+'_gradient':gradient/steps,prefix+'_gradient_abs_mean':absolute/steps})
    return result

def validate_memory_diagnostics(diag,packets,expected_delays,epoch=None):
    records=diag.get('records',[])
    if diag.get('active') is not True or len(records)!=6 or [r.get('block') for r in records]!=list(PATHS):
        raise ValueError('Actual six FIR-curvature output measurements missing')
    for prefix,r in zip(PREFIXES,records):
        if r.get('packets')!=packets or r.get('actual_delays')!=expected_delays:
            raise ValueError('Actual feature grid delays/packet scope differ')
        raw=r.get('raw_parameter');coefficient=r.get('coefficient')
        values=[r.get(k) for k in ('input_formula_max_abs_error','prefix_correction_max_abs_error',
            'delta_complex_abs_max','actual_relative_output_change_mean')]
        if any(v is None or not math.isfinite(v) or v<0 for v in values):raise ValueError('Invalid feature output measurements')
        if values[0]!=0 or values[1]!=0 or values[2]>4.00001:raise ValueError('Feature formula, prefix mask or analytic bound differs')
        if raw is None or coefficient is None or not math.isfinite(raw) or not math.isfinite(coefficient) or not math.isclose(coefficient,math.tanh(raw),abs_tol=1e-7):
            raise ValueError('Actual feature coefficient formula differs')
        if r.get('eligible_grid_positions')!=max(0,r['grid_length']-expected_delays[2]):raise ValueError('Feature grid eligibility differs')
        if epoch is not None and (raw!=epoch[prefix+'_raw'] or coefficient!=epoch[prefix+'_coefficient']):
            raise ValueError('Actual output/optimizer memory state differs')

def audit_memory_logs(epochs,steps,expected_delays):
    previous=[0.]*6
    for epoch in epochs:
        batch_steps=steps[(epoch['epoch']-1)*50:epoch['epoch']*50]
        for step in batch_steps:
            if step.get('memory_gradient_used_parameters')!=6:raise ValueError('Missing six feature CE gradients')
            for i,prefix in enumerate(PREFIXES):
                before=step[prefix+'_raw_before'];after=step[prefix+'_raw_after']
                cb=step[prefix+'_coefficient_before'];ca=step[prefix+'_coefficient_after'];g=step[prefix+'_gradient']
                if any(not math.isfinite(v) for v in (before,after,cb,ca,g)) or before!=previous[i]:raise ValueError('Broken/nonfinite feature scalar continuity')
                if not math.isclose(cb,math.tanh(before),abs_tol=1e-7) or not math.isclose(ca,math.tanh(after),abs_tol=1e-7):raise ValueError('Feature tanh coefficient differs')
                previous[i]=after
        if epoch.get('memory_gradient_used_parameters')!=6:raise ValueError('Missing six epoch feature CE gradients')
        for prefix in PREFIXES:
            if epoch[prefix+'_raw']!=batch_steps[-1][prefix+'_raw_after'] or epoch[prefix+'_coefficient']!=batch_steps[-1][prefix+'_coefficient_after']:raise ValueError('Epoch/step feature scalar state differs')
            if not math.isclose(epoch[prefix+'_gradient'],statistics.mean(s[prefix+'_gradient'] for s in batch_steps),abs_tol=1e-12,rel_tol=1e-12):raise ValueError('Epoch feature gradient average differs')
            if not math.isclose(epoch[prefix+'_gradient_abs_mean'],statistics.mean(abs(s[prefix+'_gradient']) for s in batch_steps),abs_tol=1e-12,rel_tol=1e-12):raise ValueError('Epoch feature absolute gradient average differs')
        validate_memory_diagnostics(epoch['curvature_diagnostics']['feature_curvature'],28,expected_delays,epoch)
