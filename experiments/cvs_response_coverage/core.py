"""Separate bounded response capacity from relaxed shape coverage."""
import numpy as np


def basis():
    return np.cos(np.pi*(np.arange(32)[:,None]+.5)*np.arange(1,9)[None,:]/32)*np.sqrt(2/32)


def dictionary(radius,radii,angles):
    if not 0<radius<=.06 or radii<2 or angles<4:raise ValueError('Invalid fixed response grid')
    cosine=np.cos(2*np.pi*(np.arange(256)-128)/256)
    phase=np.arange(angles)*2*np.pi/angles
    profiles=[np.zeros((1,8))];parameters=[np.zeros((1,2))];band_range=0.
    for r in np.linspace(0,radius,radii)[1:]:
        a=r*np.exp(1j*phase);power=np.abs(1+a[:,None]*cosine[None,:])**2
        if power.min()<(1-radius)**2-1e-12 or power.max()>(1+radius)**2+1e-12:raise ValueError('Response bound differs')
        pooled=power.reshape(-1,32,8);logs=np.log(pooled.mean(2));logs-=logs.mean(1,keepdims=True)
        profiles.append(logs@basis());parameters.append(np.column_stack((a.real,a.imag)))
        band_range=max(band_range,float(np.ptp(np.log(pooled),axis=2).max()))
    return np.concatenate(profiles),np.concatenate(parameters),band_range


def project(predicted,atoms):
    p=np.asarray(predicted,dtype=float);q=np.asarray(atoms,dtype=float)
    if p.ndim!=2 or q.ndim!=2 or p.shape[1]!=q.shape[1] or not np.isfinite(p).all() or not np.isfinite(q).all():raise ValueError('Finite compatible profile arrays required')
    p2=(p*p).sum(1);q2=(q*q).sum(1);dot=p@q.T
    distances=np.maximum(p2[:,None]+q2[None,:]-2*dot,0)
    bounded_index=distances.argmin(1)
    energy=np.divide(np.maximum(dot,0)**2,q2[None,:],out=np.zeros_like(dot),where=q2[None,:]>1e-24)
    index=energy.argmax(1)
    scalar=np.divide(np.maximum(dot[np.arange(len(p)),index],0),q2[index],out=np.zeros(len(p)),where=q2[index]>1e-24)
    return dict(bounded=q[bounded_index],bounded_index=bounded_index,
        relaxed=scalar[:,None]*q[index],relaxed_index=index,relaxed_scalar=scalar,
        direction_energy=energy[np.arange(len(p)),index],predicted_energy=p2)


def source(evidence):
    complete=evidence['row']['completion.json'];s=evidence['row']['spectral_crossfit.json'];p=evidence['row']['provenance.json']
    if (complete.get('status')!='SOURCE_DIAGNOSTIC_COMPLETE' or complete.get('identity_updates')!=0
        or complete.get('checkpoint_loaded') or complete.get('target_access') is not False or complete.get('L_count')!=6300
        or any(s.get(k) is not False for k in ('U_used','V_used','target_access'))
        or s.get('primary')!='rectangular/DCT1to8'):
        raise ValueError('Only completed fixed L-derived source statistics permitted')
    ids=p['L_ids'];roles=p['source_contract']['role_ids']
    if len(ids)!=6300 or len(set(ids))!=6300 or set(ids)!=set(roles['L_s']) or set(ids)&(set(roles['V'])|set(roles['U_s'])):
        raise ValueError('Source aggregate physical roles differ')
    observation=s['observations'][s['primary']]
    if observation['labels']!=dict(TX=list(range(6)),day=[1,2,3],RX=[1,3,4,6,8],cell_count=70):raise ValueError('Source coordinate labels differ')
    expected=[(t,d) for t in range(6) for d in range(3)]
    if [(f['TX_index'],f['day_index']) for f in observation['folds']]!=expected:raise ValueError('Fixed exclusion folds differ')
    predicted=np.asarray(observation['predicted_contrasts'],dtype=float);actual=np.asarray(observation['observed_contrasts'],dtype=float)
    if predicted.shape!=(18,5,8) or actual.shape!=predicted.shape or not np.isfinite(predicted).all() or not np.isfinite(actual).all():raise ValueError('Source aggregate dimensions differ')
    if not np.allclose(predicted.sum(1),0,atol=1e-12) or not np.allclose(actual.sum(1),0,atol=1e-12):raise ValueError('RX contrasts must have zero RX mean')
    zero=float((actual*actual).sum());ratio=float(((predicted-actual)**2).sum()/zero) if zero>0 else None
    if ratio is not None and abs(ratio-observation['relative_MSE']['prediction'])>1e-10:raise ValueError('Existing source result arithmetic differs')
    return predicted,actual,ratio


def audit(predicted,actual,grid,radius):
    q,params,band_range=dictionary(radius,grid['radii'],grid['angles'])
    p=predicted.reshape(-1,8);a=actual.reshape(-1,8);fit=project(p,q)
    pe=float((p*p).sum());ae=float((a*a).sum());per=[]
    for i in range(len(p)):
        per.append(dict(TX_index=i//15,day_index=(i//5)%3,RX_index=i%5,predicted_energy=float(fit['predicted_energy'][i]),
            direction_coverage=float(np.clip(fit['direction_energy'][i]/fit['predicted_energy'][i],0,1)) if fit['predicted_energy'][i]>1e-24 else None,
            bounded_parameter=params[fit['bounded_index'][i]].tolist(),relaxed_parameter=params[fit['relaxed_index'][i]].tolist(),
            relaxed_scalar=float(fit['relaxed_scalar'][i]),observed_zero_SSE=float(a[i]@a[i]),
            observed_full_prediction_SSE=float(((p[i]-a[i])**2).sum()),
            observed_bounded_prediction_SSE=float(((fit['bounded'][i]-a[i])**2).sum()),
            observed_relaxed_prediction_SSE=float(((fit['relaxed'][i]-a[i])**2).sum())))
    return dict(grid=grid,atoms=len(q),weighted_direction_coverage=float(np.clip(fit['direction_energy'].sum()/pe,0,1)) if pe>1e-24 else None,
        bounded_mismatch_to_predicted=float(((fit['bounded']-p)**2).sum()/pe) if pe>1e-24 else None,
        relative_observed_MSE={k:float(((values-a)**2).sum()/ae) if ae>1e-24 else None for k,values in
            [('full_empirical',p),('bounded_current_family',fit['bounded']),('relaxed_shape_only',fit['relaxed'])]},
        within_bin_proxy_log_power_range=band_range,profiles=per,
        fitting_scope='Dictionary choice uses excluded-TX/day prediction only; held-out observations only score the diagnostic',
        relaxed_limit='Unbounded scalar is a diagnostic relaxation; not an allowed IQ operator or chosen training amplitude')


def decision(primary,control,c):
    a=primary['weighted_direction_coverage'];b=control['weighted_direction_coverage']
    if a is None or b is None:return 'INSUFFICIENT_SOURCE_PROFILE_ENERGY'
    if abs(a-b)>c['resolution_tolerance']:return 'INCONCLUSIVE_GRID_RESOLUTION'
    if a<c['direction_gap_upper']:return 'IDEAL_MAGNITUDE_DIRECTION_GAP'
    if a>=c['substantial_direction_lower']:return 'SUBSTANTIAL_IDEAL_DIRECTION_COVERAGE'
    return 'MIXED_IDEAL_DIRECTION_COVERAGE'
