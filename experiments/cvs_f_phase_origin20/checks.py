import copy
import numpy as np
import torch
from . import design as d
from .core import factor_views,phase_from_kernel
from experiments.cvs_f_practical_source20.checks import reject
from experiments.cvs_urban_sigtime.channel import imports,configurations,tensor_iq

def serialization_regression():
    import json
    from experiments.cvs_urban_sigtime.channel import audit
    from .core import require_same_practical
    current=audit()['actual_configs'];saved=json.loads(json.dumps(current))
    assert current!=saved;require_same_practical(current,saved)
    for key,value in [('reference_snr_db',31.),('boundary','edge'),('equalization_enabled',True)]:
        changed=copy.deepcopy(saved);changed['practical_low_urban'][key]=value;reject(lambda:require_same_practical(current,changed))
    return dict(status='PASS',tuple_list_semantic_equality=True,real_scalar_changes_rejected=3,model_forwards=0,IQ_reads=0,CP_reads=0)

def run():
    torch.set_num_threads(2);negative=0
    for key,value in [('target_access',True),('V_used',True),('U_used',True),('delay_samples',25),('boundary','wrap'),
        ('checkpoint','alternate'),('optimizer_updates',1),('draw_seed',0)]:
        c=d.config(d.rows()[0]);c[key]=value;reject(lambda:d.validate(c));negative+=1
    ch,residual=imports();rng=np.random.default_rng(1234);wave=rng.normal(size=256)+1j*rng.normal(size=256);wave,_=ch.normalize_input(wave)
    views=factor_views(wave,.7);expected=np.r_[wave[24:0:-1],wave[:-24]]
    assert np.allclose(views['delay'],expected,atol=1e-7,rtol=1e-6)
    assert np.allclose(views['phase'],wave*np.exp(.7j),atol=1e-7,rtol=1e-6)
    assert np.allclose(views['phase_delay'],expected*np.exp(.7j),atol=1e-7,rtol=1e-6)
    zero=factor_views(wave,0.,0);assert all(np.allclose(w,wave,atol=1e-7,rtol=1e-6) for w in zero.values())
    for delay in (-1,1,23,25):reject(lambda:factor_views(wave,0.,delay));negative+=1
    reject(lambda:factor_views(wave,float('nan')));negative+=1
    reject(lambda:factor_views(wave,.7,boundary='wrap'));negative+=1
    kernels=[]
    for scene,cfg in configurations().items():
        seed=ch.stable_seed(d.DRAW_SEED,'synthetic-only',scene);hardware=ch.receiver_for_session(cfg,d.SEED,'rx1:day1:eq1')
        phase,meta=phase_from_kernel(cfg,seed,hardware);_,_,traces=residual.ResidualChannel(cfg,seed,hardware).process(wave,return_traces=True)
        h=traces['propagation_kernel'];assert abs(phase-np.angle(h[np.argmax(abs(h))]))<1e-12
        assert len(h)==meta['kernel_length'] and meta['numerical_delay']==24;kernels.append(scene)
    from experiments.cvs_raw_anchor20 import model as api,design as parent
    from experiments.cvs_f_practical_source20.trace import ActualTrace,feature_summary
    model=api.build(parent.config(parent.rows()[0]),'cpu').eval().requires_grad_(False);snapshot={k:v.clone() for k,v in model.state_dict().items()};trace=ActualTrace(model)
    bank=[wave,*views.values()];x=tensor_iq(np.stack([np.stack([w.real,w.imag]) for w in bank]),'cpu')
    with torch.no_grad(),trace.installed():
        out,z=trace.forward(x);fs=feature_summary(z,z)
        assert out['tx_logits'].shape==(4,6) and torch.isfinite(out['tx_logits']).all() and len(fs)==15
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in snapshot.items()) and all(p.grad is None for p in model.parameters())
    from .harvest import Group
    g=Group()
    for i,p in enumerate([0,1,2,4,3,5]):g.add(dict(TX=i,prediction=p,margin=.2,full_prediction=i,clean_correct=True,A0_clean_correct=True,common_clean_correct=True))
    assert abs(g.result()['accuracy']-4/6)<1e-12 and abs(g.result()['macro_f1']-4/6)<1e-12
    return dict(status='PASS',protocol_negatives=negative,synthetic_model_packet_forwards=4,synthetic_derivative_calls=0,optimizer_updates=0,
        synthetic_original_kernel_checks=kernels,explicit_factor_formulas=True,zero_factors_identity=True,default_head_actual_trace=True,
        reconstruction_error=trace.maximum_reconstruction_error,parameters_buffers_unchanged=True,independent_metrics_fixture=True,real_IQ=0,real_checkpoint=False)
