"""Scratch joint E30 with phase-canonical priors, no F or learned teacher loss."""
import json
import numpy as np
import torch
from . import no_variance
from .coarse_hint import learning_rate
from experiments.pamf_dura_efficiency.phase_prior import wrap_priors,rotate_inputs,rotate_iq

base=no_variance.base
RUN='20261031-phase1-DURA-phasePrior-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.phase_prior'
CONTROL='20261031-phase1-DURA-repeatViews-E30-s2026092701-r01'


def smoke(args,p):
    clean,roles=base.source_training(p)
    from .f_guided import source_ids
    source_ids(p,roles)
    take=np.flatnonzero(roles=='L_s')[:2];folder=p['cache']/'epochs/01'
    values={key:base.tensor(np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu') for key in ['received','kernel','coefficients','variance']}
    x=base.tensor(clean[take],'cpu');y=values['received'];student=base.scratch_model(2026092701)
    with torch.no_grad():expected=student(y).clone()
    rng=torch.Generator().manual_seed(2026103004)
    probes=[torch.complex(torch.randn(2,256,generator=rng),torch.randn(2,256,generator=rng)) for _ in range(2)]
    loss,_=base.direct_objective(student,y,x,base.supervised_action(values['kernel'],values['coefficients']),values['variance'],probes)
    loss.mean().backward()
    for prefix in ['frozen.coarse.','frozen.estimator.','prior.']:
        grads=[v.grad for k,v in student.named_parameters() if k.startswith(prefix) and v.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum()>0 for g in grads):
            raise ValueError('Finite nonzero joint phase-prior G gradients required: '+prefix)
    checkpoint=p['output']/'smoke-scratch_joint.pt';base.save_model(checkpoint,student,'scratch_joint',0,0,args.commit)
    loaded=base.load_model(checkpoint,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    from experiments.pamf_stage2_inference.engine import TensorPipeline
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    with torch.no_grad():
        native=loaded(y)
        if not torch.equal(native,expected):raise ValueError('Strict phase-prior reload changed IQ')
        optimized,valid=TensorPipeline(loaded,None,reuse=True,native=False)(y)
        if not valid:raise ValueError('Actual phase-prior deployment arithmetic failed')
        torch.testing.assert_close(optimized,native,rtol=3e-5,atol=3e-5)
        prior_input=torch.cat((y,x,x,y,torch.zeros_like(x[:,:1])),dim=1)
        angle=torch.tensor([.37,-1.13]);unit=torch.stack((angle.cos(),angle.sin()),dim=1)[:,:,None]
        errors=[]
        for core in loaded.prior:
            actual=core(rotate_inputs(prior_input,unit));reference=rotate_iq(core(prior_input),unit)
            errors.append(float((actual-reference).abs().max()))
            torch.testing.assert_close(actual,reference,rtol=3e-5,atol=3e-5)
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    with torch.no_grad():scores=reader(optimized)
    if scores.shape!=(2,6) or not torch.isfinite(scores).all():raise ValueError('Original frozen F/defaulthead no-query smoke failed')
    base.write_json(p['output']/'smoke.json',dict(status='PASS',real_Ls=2,G_inputs=8,prior_only_inputs=8,F_inputs=2,prior_equivariance_max_errors=errors,
        joint_coarse_action_prior_gradients=True,strict_reload=True,native_optimized_close=True,
        initialization_checkpoint_loads=0,TX_labels_read=0,V_reads=0,target_reads=0,F_updates=0,training_F_access=False))
    print(json.dumps(dict(event='REAL_PHASE_PRIOR_E30_SMOKE_PASS',prior_equivariance_max_errors=errors)),flush=True)


def configure():
    no_variance.configure();previous_configuration=base.configuration;previous_command=base.command;original_factory=base.scratch_model
    def factory(seed):return wrap_priors(original_factory(seed))
    def configuration():
        value=previous_configuration()
        value.update(run=RUN,recipe='scratch_E30_phase_canonical_prior',epochs=30,updates=7410,new_cumulative_epochs=30,
            L_exposures=189000,U_exposures=1701000,corruption_epochs=20,source_corruption_schedule='01..20then01..10',
            source_freeze='fixedfinalE30/7410;noearlystop/source-target-ranking',CUDA_probe_seed=2026092701,numpy_shuffle_seed=2026103004,
            teacher_model=None,training_F_access=False,F_guided_training=False,source_F_informed_method_research=True,
            source_supervision='Originalsourcepairedclean/action only;no teacher/F/TX/V/targetoptimization',
            prior_phase_coordinate='strongest_coarse_base_point;fallback_to_strongest_IQ_pair_only_for_zero_base',
            prior_phase_extra_parameters=0,prior_phase_scalar_channel_unchanged=True,prior_phase_core_calls=2,
            whole_G_equivariance_claim=False,extra_coordinate_ops_not_in_ConvLinear_MAC=True,
            matched_control=CONTROL,matched_control_commit='8cf8e22c9ffb54d271b22e621401de8e34c69068',
            allowed_GPUs=[4,6,7],max_total_visible_GPU_processes=3,host_numpy_conversion='DLPack_owned_FP32')
        return value
    def command(args,mode,arm=None,role=None):
        return [MODULE if v==no_variance.MODULE else v for v in previous_command(args,mode,arm,role)]
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.scratch_model=factory;base.learning_rate=learning_rate;base.smoke=smoke


if __name__=='__main__':configure();base.main()
