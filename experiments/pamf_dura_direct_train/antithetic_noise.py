"""Scratch single E30 with source residual sign diversity and original SIG loss."""
import json,time
import numpy as np
import torch
from . import signature_guided as parent,f_guided
from experiments.pamf_dura_efficiency.antithetic_noise import augment

base=parent.base
RUN='20261031-phase1-DURA-antitheticNoise-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.antithetic_noise'
CACHE_RUN=parent.RUN
AUGMENTATION_SEED=2026103107
METRICS=('source_noise_flip_requested','source_noise_flip_effective','source_noise_gain','source_noise_power_ratio','source_noise_action_inputs')
_OBJECTIVE=parent.objective
_rng=None
_smoke_mode=False


def objective(student,reader,received,clean,action,variance,probes,target):
    if _smoke_mode:
        requested=torch.zeros(len(received),dtype=torch.bool,device=received.device);requested[0]=True
    else:
        if _rng is None:raise ValueError('Dedicated source augmentation RNG must be configured')
        requested=base.tensor((_rng.random(len(received))<.5).astype(np.float32),received.device).bool()
    observed,actual,new_variance,info=augment(received,clean,action,variance,requested)
    loss,terms,restored=_OBJECTIVE(student,reader,observed,clean,actual,new_variance,probes,target)
    extra=dict(zip(METRICS,(info['requested'],info['effective'],info['gain'],info['power_ratio'],torch.ones_like(info['gain']))))
    return loss,dict(terms,**extra),restored


def reuse_targets(p,reader,clean,roles):
    started=time.perf_counter();ids=f_guided.source_ids(p,roles);folder=p['output'].parent/CACHE_RUN/'source_signature_cache'
    meta=base.read_json(folder/'complete.json')
    expected=dict(status='SOURCE_ACTUAL_SIG_COMPLETE',count=63000,shape=[63000,160],numeric_bytes=40320000,
        source_contract='original7/63/30',F_teacher=f_guided.teacher_reference(),actual_weighted_TripleSIG=True,
        full_F_inputs=0,F_prefix_inputs=63000,F_input_VJP_inputs=0,F_updates=0,model_optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0)
    if any(meta.get(k)!=v for k,v in expected.items()) or not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids):
        raise ValueError('Same original source-only F50 SIG cache and physical order required')
    targets=np.load(folder/'targets.npy',mmap_mode='r',allow_pickle=False)
    if targets.shape!=(63000,160) or targets.dtype!=np.float32:raise ValueError('Matching source SIG shape/dtype required')
    return targets,dict(status='SOURCE_ACTUAL_SIG_REUSED',count=63000,shape=[63000,160],numeric_bytes=40320000,source_contract='original7/63/30',
        F_teacher=f_guided.teacher_reference(),reused_from_run=CACHE_RUN,original_metadata=meta,new_F_prefix_inputs=0,new_F_input_VJP_inputs=0,new_numeric_bytes=0,
        TX_labels_read=0,V_reads=0,target_reads=0,seconds=time.perf_counter()-started)


def smoke(args,p):
    global _smoke_mode
    clean,roles=base.source_training(p);targets,reuse=reuse_targets(p,None,clean,roles);folder=p['cache']/'epochs/01'
    variances=np.load(folder/'variance.npy',mmap_mode='r',allow_pickle=False)
    take=np.flatnonzero((roles=='L_s')&(variances>np.float32(1e-8)))[:2]
    if len(take)!=2:raise ValueError('Two actual nonclean source L_s smoke records required')
    values={key:base.tensor(np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu') for key in ['received','kernel','coefficients','variance']}
    x=base.tensor(clean[take],'cpu');target=base.tensor(targets[take],'cpu');action=base.supervised_action(values['kernel'],values['coefficients'])
    requested=torch.tensor([True,False]);augmented,adjusted,new_variance,info=augment(values['received'],x,action,values['variance'],requested)
    if not torch.equal(info['effective'],requested.float()):raise ValueError('Real smoke must exercise one flipped and one unchanged source IQ')
    torch.testing.assert_close(augmented[1],values['received'][1],atol=0,rtol=0)
    torch.testing.assert_close(new_variance,values['variance']*info['gain'].square(),atol=1e-12,rtol=1e-6)
    student=base.scratch_model(2026092701)
    if sum(v.numel() for v in student.parameters())!=266951 or hasattr(student,'mid_signature_aux'):raise ValueError('Original complete scratch G required')
    reader,rng_info=parent.load_teacher_preserving_rng(p,'cpu');before={k:v.clone() for k,v in reader.state_dict().items()}
    with torch.no_grad():
        _,parts=parent.components(reader,x);torch.testing.assert_close(parts['signature'],target,atol=3e-5,rtol=2e-5)
    generator=torch.Generator().manual_seed(2026103004)
    probes=[torch.complex(torch.randn(2,256,generator=generator),torch.randn(2,256,generator=generator)) for _ in range(2)]
    # Known-source action probes must be scaled with the augmented observation.
    torch.testing.assert_close(adjusted(probes[0]),info['gain'][:,None]*action(probes[0]),atol=0,rtol=0)
    _smoke_mode=True
    try:loss,terms,restored=objective(student,reader,values['received'],x,action,values['variance'],probes,target)
    finally:_smoke_mode=False
    torch.testing.assert_close(terms[METRICS[1]],requested.float(),atol=0,rtol=0)
    actual=torch.autograd.grad(terms[parent.GUIDE].sum(),restored,retain_graph=True)[0]
    iqnorm=2*(restored.detach()-x).flatten(1).norm(dim=1)/(512*x.square().mean((1,2)).clamp_min(1e-4))
    if not actual.abs().sum()>0 or not torch.isfinite(actual).all() or (actual.flatten(1).norm(dim=1)>parent.CAP*iqnorm*(1+2e-5)).any():
        raise ValueError('Actual augmented SIG guidance cap failed')
    loss.mean().backward()
    for prefix in ['frozen.coarse.','frozen.estimator.','prior.']:
        grads=[v.grad for k,v in student.named_parameters() if k.startswith(prefix) and v.grad is not None]
        if not grads or not all(torch.isfinite(v).all() for v in grads) or not any(v.abs().sum()>0 for v in grads):
            raise ValueError('Actual source augmented joint G gradient missing: '+prefix)
    with torch.no_grad():expected=student(values['received']).clone()
    checkpoint=p['output']/'smoke-scratch_joint.pt';f_guided.save_model(checkpoint,student,'scratch_joint',0,0,args.commit)
    loaded=parent.load_model(checkpoint,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    from experiments.pamf_stage2_inference.engine import TensorPipeline
    with torch.no_grad():
        if not torch.equal(loaded(values['received']),expected):raise ValueError('Strict augmented-G E0 reload changed raw IQ')
        optimized,valid=TensorPipeline(loaded,None,reuse=True,native=False)(values['received'])
        if not valid:raise ValueError('Optimized native G inference invalid')
        torch.testing.assert_close(optimized,expected,atol=3e-5,rtol=3e-5);scores=reader(optimized)
    if scores.shape!=(2,6) or not torch.isfinite(scores).all() or any(v.grad is not None for v in reader.parameters()) or any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()):
        raise ValueError('Original frozen F/defaulthead changed')
    base.write_json(p['output']/'smoke.json',dict(status='PASS',real_nonclean_Ls=2,forced_mask=[True,False],RMS_preserved=True,
        source_action_and_variance_scaled=True,clean_reference_unchanged=True,G_parameters=266951,G_inputs=8,
        F_full_inputs=4,F_prefix_inputs=2,F_input_VJP_inputs=2,source_action_clean_inputs=4,source_action_probe_inputs=8,
        all_G_modules_gradients=True,strict_E0_reload=True,native_optimized_match=True,source_only_augmentation_not_in_G_forward=True,
        SIG_IQ_gradient_cap=.5,F_unchanged=True,F_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,G_initialization_checkpoint_loads=0,
        source_cache_reuse=reuse,teacher_load=rng_info))
    print(json.dumps(dict(event='REAL_ANTITHETIC_SOURCE_NOISE_E30_SMOKE_PASS')),flush=True)


def configure():
    global _rng
    parent.configure();_rng=np.random.default_rng(AUGMENTATION_SEED)
    previous_configuration=base.configuration;previous_command=base.command
    def configuration():
        value=previous_configuration();value.update(run=RUN,recipe='scratch_E30_SIG_with_source_residual_sign_augmentation',
            source_noise_augmentation='Bernoulli.5 residualsign;preserve originalreceivedRMS;scale actualA and variance;skip numericalzeroresidue',
            source_noise_augmentation_seed=AUGMENTATION_SEED,source_noise_requested_probability=.5,source_noise_zero_energy_relative=1e-12,
            source_noise_action_clean_inputs=1890000,source_noise_action_probe_inputs=3780000,source_noise_inference_calls=0,
            new_F_clean_cache_inputs=0,reused_F_clean_cache_run=CACHE_RUN,source_noise_new_full_simulator_draws=0,
            matched_control=CACHE_RUN,matched_control_commit='47591cdfe2e6e6dd7534cdea846badd942321b8f',
            training_regime='source_F_guided_SIG_and_source_residual_sign_augmentation_single_joint_E30',
            source_augmentation_scope='Fixedrealizedsourceaction/residualsign with renormalization;not new fullsimulator draw or exact conditional-on-AGC distribution claim',
            old_F_antithetic_variance_FAIL_preserved=True,
            objective='unchangedIQ1+priorIQ.1+STFT.05+adjustedsourceactionprobe.05+actualSIGcap.5;varianceaux0')
        return value
    def command(args,mode,arm=None,role=None):return [MODULE if x==parent.MODULE else x for x in previous_command(args,mode,arm,role)]
    parent.EXTRA=(*parent.EXTRA,*METRICS);parent.objective=objective;parent.build_targets=reuse_targets
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.smoke=smoke;base.__doc__=__doc__


if __name__=='__main__':configure();base.main()
