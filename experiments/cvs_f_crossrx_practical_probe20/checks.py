"""Bounded synthetic checks; real checkpoint smoke is worker-first."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_f_practical_source20.trace import ActualTrace,gradient,gradient_comparison,feature_summary


def reject(fn):
    try:fn()
    except (ValueError,KeyError):return
    raise AssertionError('Protocol-negative was accepted')


def run():
    torch.set_num_threads(2);negative=0
    for key,value in [('target_access',True),('source_only',False),('V_used',True),('U_used',True),
        ('optimizer_updates',1),('checkpoint','alternate.pth'),('source_count',1259),('conditions',['full'])]:
        c=d.config(d.rows()[0]);c[key]=value;reject(lambda:d.validate(c));negative+=1
    # Metadata-only lawful synthetic role carrier; never physical data.
    rx=torch.tensor([r for r in (1,3,4,6,8) for _ in range(1260)])
    y=torch.tensor([t for r in range(5) for t in range(6) for day in range(3) for _ in range(70)])
    day=torch.tensor([day for r in range(5) for t in range(6) for day in (1,2,3) for _ in range(70)])
    L=dict(ids=[f'synthetic-{i}' for i in range(6300)],x=torch.zeros(6300,2,256),rx=rx,y=y,day=day)
    contract=dict(role_ids=dict(L_s=L['ids'][1260:],U_s=['unlabeled-synthetic'],V=['validation-synthetic']))
    assert len(d.legal_held_indices(L,[contract,contract]))==1260
    for role in ('L_s','U_s','V'):
        c=copy.deepcopy(contract);c['role_ids'][role].append(L['ids'][0]);reject(lambda:d.legal_held_indices(L,[c]));negative+=1
    bad=dict(L,day=torch.zeros_like(day));reject(lambda:d.legal_held_indices(bad,[contract]));negative+=1
    from experiments.cvs_f_crossrx_practical20 import model as api
    model=api.build(d.parent_model_config('C0'),'cpu').eval();before={k:v.clone() for k,v in model.state_dict().items()}
    x=torch.randn(4,2,256);labels=torch.arange(4)%6
    plain=model.forward_identity_only(x);trace=ActualTrace(model)
    with trace.installed():
        out,values=trace.forward(x);assert torch.allclose(plain['tx_logits'],out['tx_logits'],atol=1e-6,rtol=1e-6)
        clean={k:v.detach() for k,v in values.items()};cg=gradient(F.cross_entropy(out['tx_logits'],labels,label_smoothing=.01),model)
        out,values=trace.forward(x+.01*torch.randn_like(x));sg=gradient(F.cross_entropy(out['tx_logits'],labels,label_smoothing=.01),model)
        fs=feature_summary(values,clean);assert len(gradient_comparison(cg,sg))==len(cg)==14
        from .run import decisions
        z=decisions(out['tx_logits'],labels);assert len(z['margin'])==4
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()) and all(p.grad is None for p in model.parameters())
    from experiments.cvs_urban_sigtime.channel import imports,configurations,audit
    from experiments.cvs_urban_sigtime.controls import compare
    ch,_=imports();configs=configurations();physical=[]
    t=np.arange(256);wave,_=ch.normalize_input(np.exp(1j*.17*t)+.23*np.exp(-1j*.073*t))
    for scene,cfg in configs.items():
        receiver=ch.receiver_for_session(cfg,d.RECEIVER_SEED,'rx1:day1:eq1');seed=ch.stable_seed(d.DRAW_SEED,'synthetic-only',scene)
        views,meta,canonical=compare(wave,cfg,seed,receiver)
        assert tuple(views)==d.CONDITIONS and all(np.isfinite(v).all() for v in views.values())
        full=meta['full']
        for condition in ('no_noise','flat_H','no_carrier','no_RX_IQ'):
            fixed=meta[condition+'/fixed']
            for k in ('receiver_lock_status','residual_scale','agc_gain_db'):
                assert fixed[k]==full[k]
        assert meta['no_noise/fixed']['signal_to_added_noise_db'] is None
        assert full['noise_coloration_dynamic_range_db']==0. and full['boundary_interior_relative_error']<1e-6
        assert cfg.equalization_enabled is False and full['kernel_length']>=full['numerical_delay']
        physical.append(dict(scene=scene,state=full['state'],status=full['receiver_lock_status'],canonical_full_match=True))
    return dict(status='PASS',protocol_negatives=negative,synthetic_model_packet_forwards=12,derivative_calls=2,
        synthetic_physical_draws=len(physical),synthetic_counterfactual_waveforms=6*11,actual_graph_reconstruction_error=trace.maximum_reconstruction_error,
        features=list(fs),gradient_parameters=list(cg),parameters_buffers_unchanged=True,optimizer_updates=0,head_fit_updates=0,
        physical_checks=physical,channel=audit(),real_IQ_packets=0,real_checkpoint_loaded=False)
