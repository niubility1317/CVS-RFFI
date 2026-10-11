"""One fixed source-L margin calibration at the registered student update."""
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from experiments.cvs_e20_boundary_style.multilevel import BoundarySensitivity
from experiments.cvs_e20_budget_fishr.multilevel import LEVELS, _parameter_groups, _norm


def scheduled_after(c):
    value=c['multilevel']['calibration_after_updates']
    if (type(value) is not int or value not in (0,444)
            or c['semi']['warmup_updates']!=444
            or not c['multilevel']['enabled'] or c['multilevel']['statistic']!='margin'):
        raise ValueError('Registered margin calibration must follow exactly 0 or 444 identity updates')
    return value


def begin_calibration(c,model,L,device,out):
    after=scheduled_after(c)
    if not after:return calibrate(c,model,L,device,out,after_identity_updates=0)
    if (Path(out)/'calibration.json').exists():raise FileExistsError('Fixed calibration already exists')
    if hasattr(model.multilevel,'fixed_coefficients'):raise ValueError('Delayed calibration inherited coefficients')
    return dict(status='PENDING_CALIBRATION',scheduled_after_identity_updates=after,
        after_identity_updates=None,fixed_coefficients={},pairs=0,extra_source_readouts=0,
        model_forwards=0,seconds=0.,source_role='L_s only',U_access=False,V_access=False,
        target_access=False,identity_updates=0,head_updates=0,auxiliary_heads=0,
        calibration_reference='own_margin_term')


def advance_calibration(c,model,L,device,out,current,completed_updates):
    """Call after optimizer.step/counting and before the next stream.batch."""
    after=scheduled_after(c)
    if current['status']=='FIXED_CALIBRATION_COMPLETE':return current,False
    if current['status']!='PENDING_CALIBRATION':raise ValueError('Unknown calibration lifecycle state')
    if completed_updates<after:return current,False
    if completed_updates!=after:raise ValueError('Missed exact delayed calibration update boundary')
    return calibrate(c,model,L,device,out,after_identity_updates=completed_updates),True


def require_calibration(c,model,current,step):
    """Fail before an auxiliary graph is constructed with absent coefficients."""
    mc=c['multilevel']
    if step<=c['semi']['warmup_updates'] or step%mc['interval']:return
    coefficients=current.get('fixed_coefficients',{})
    if (current.get('status')!='FIXED_CALIBRATION_COMPLETE'
            or current.get('after_identity_updates')!=scheduled_after(c)
            or set(coefficients)!=set(LEVELS)
            or coefficients!=getattr(model.multilevel,'fixed_coefficients',None)
            or any(not math.isfinite(v) or v<=0 for v in coefficients.values())):
        raise ValueError('Auxiliary identity update requires the completed fixed calibration')


def calibrate(c,model,L,device,out,after_identity_updates=0):
    """Read the same twelve private L pairs; never update optimizer or model."""
    if type(after_identity_updates) is not int or after_identity_updates!=scheduled_after(c):
        raise ValueError('Calibration invoked at an unregistered identity update')
    path=Path(out)/'calibration.json'
    if path.exists():raise FileExistsError('Fixed calibration already exists')
    mc=c['multilevel']
    if L.get('role','L_s') not in ('L_s','L') or any(k not in L for k in ('x','y','rx','day','ids')):
        raise ValueError('Calibration requires visible source L only')
    if not isinstance(model.multilevel,BoundarySensitivity):raise ValueError('Own margin calibration needs zero-head margin module')
    if hasattr(model.multilevel,'fixed_coefficients'):raise ValueError('Fixed calibration coefficients already exist')
    if mc['fishr_calibration_pairs']!=12 or mc['fishr_ratio']!=.2 or mc['fishr_cap']!=10000.:
        raise ValueError('Fixed margin calibration protocol differs')
    started=time.perf_counter()
    before={n:v.detach().clone() for n,v in model.state_dict().items()}
    modes=[(module,module.training) for module in model.modules()]
    py_rng=random.getstate();np_rng=np.random.get_state();cpu_rng=torch.get_rng_state().clone()
    devices=list(range(torch.cuda.device_count())) if str(device).startswith('cuda') else []
    cuda_rng=[torch.cuda.get_rng_state(index).clone() for index in devices]
    groups=_parameter_groups(model);selected=dict(mc,fishr_levels=list(LEVELS));rows=[]
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            for i in range(12):
                tick=time.perf_counter()
                _,terms,detail,ce=model.multilevel._pair(model,L,4*(i+1),device,selected,calibration=True)
                values={level:dict(main_ce_gradient_norm=_norm(ce,groups[level]),
                    term_gradient_norm=_norm(terms[level],groups[level]),raw_term=float(terms[level].detach())) for level in LEVELS}
                if devices:torch.cuda.synchronize(device)
                rows.append(dict(pair=i,day=detail['day'],cross_batch_PID_overlap=detail['cross_batch_PID_overlap'],
                    levels=values,main_ce=float(ce.detach()),seconds=time.perf_counter()-tick))
        if any(not torch.equal(v,before[n]) for n,v in model.state_dict().items()):
            raise AssertionError('Margin calibration changed model parameters or buffers')
    finally:
        # Restore only changed tensors, never bump versions on the healthy path.
        with torch.no_grad():
            for name,value in model.state_dict().items():
                if not torch.equal(value,before[name]):value.copy_(before[name])
        for module,mode in modes:module.training=mode
        random.setstate(py_rng);np.random.set_state(np_rng)
        torch.set_rng_state(cpu_rng)
        for index,state in zip(devices,cuda_rng):torch.cuda.set_rng_state(state,index)
    coefficients={};summary={}
    for level in LEVELS:
        ce=sum(r['levels'][level]['main_ce_gradient_norm'] for r in rows)/12
        norm=sum(r['levels'][level]['term_gradient_norm'] for r in rows)/12
        if not math.isfinite(ce+norm) or min(ce,norm)<=mc['fishr_eps']:
            raise ValueError('Invalid fixed margin calibration gradient norm')
        uncapped=.2*ce/(norm+mc['fishr_eps']);value=min(mc['fishr_cap'],uncapped)
        coefficients[level]=value
        summary[level]=dict(mean_CE_gradient_norm=ce,mean_term_gradient_norm=norm,
            uncapped_coefficient=uncapped,coefficient=value,cap_active=uncapped>mc['fishr_cap'])
    result=dict(status='FIXED_CALIBRATION_COMPLETE',calibration_reference='own_margin_term',
        fixed_coefficients=coefficients,levels=summary,pairs=12,records=rows,extra_source_readouts=1440,
        model_forwards=24,seconds=time.perf_counter()-started,state_unchanged=True,RNG_preserved=True,
        parameters_buffers_unchanged=True,modes_preserved=True,python_numpy_RNG_preserved=True,
        source_role='L_s only',U_access=False,V_access=False,target_access=False,
        identity_updates=0,head_updates=0,auxiliary_heads=0,all_layers_calibrated=True,
        after_identity_updates=after_identity_updates,scheduled_after_identity_updates=scheduled_after(c),
        coefficient_formula='min(cap,0.2*mean(pair_main_CE_gradient_norm)/(mean(own_margin_term_gradient_norm)+eps))',
        sampling_namespace='same twelve source-private calibration pairs as scratch boundary style; own-margin statistic')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    model.multilevel.fixed_coefficients=coefficients
    return result
