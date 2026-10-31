"""Scratch single E30 SIG-guided G with received-derived multiscale prior FiLM."""
import json
import numpy as np
import torch
from . import signature_guided as parent,f_guided
from .antithetic_noise import reuse_targets
from experiments.pamf_dura_efficiency.model import scratch_model as original_factory
from experiments.pamf_dura_efficiency.context_film import attach,BOUND,HIDDEN

base=parent.base
RUN='20261031-phase1-DURA-contextFiLM-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.context_film'


def factory(seed):return attach(original_factory(seed),seed)


def smoke(args,p):
    clean,roles=base.source_training(p);targets,reuse=reuse_targets(p,None,clean,roles);folder=p['cache']/'epochs/01'
    variances=np.load(folder/'variance.npy',mmap_mode='r',allow_pickle=False)
    take=np.flatnonzero((roles=='L_s')&(variances>np.float32(1e-8)))[:2]
    if len(take)!=2:raise ValueError('Two real nonclean source L_s smoke records required')
    values={k:base.tensor(np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu') for k in ['received','kernel','coefficients','variance']}
    x=base.tensor(clean[take],'cpu');target=base.tensor(targets[take],'cpu');student=factory(2026092701);original=original_factory(2026092701)
    if any(not torch.equal(v,student.state_dict()[k]) for k,v in original.state_dict().items()):raise ValueError('Base G initial state differs from matched SIG control')
    with torch.no_grad():torch.testing.assert_close(student(values['received']),original(values['received']),atol=0,rtol=0)
    reader,rng_info=parent.load_teacher_preserving_rng(p,'cpu');before={k:v.clone() for k,v in reader.state_dict().items()}
    with torch.no_grad():torch.testing.assert_close(parent.signature_features(reader,x),target,atol=3e-5,rtol=2e-5)
    generator=torch.Generator().manual_seed(2026103004)
    probes=[torch.complex(torch.randn(2,256,generator=generator),torch.randn(2,256,generator=generator)) for _ in range(2)]
    loss,terms,restored=parent.objective(student,reader,values['received'],x,base.supervised_action(values['kernel'],values['coefficients']),values['variance'],probes,target)
    actual=torch.autograd.grad(terms[parent.GUIDE].sum(),restored,retain_graph=True)[0]
    iqnorm=2*(restored.detach()-x).flatten(1).norm(dim=1)/(512*x.square().mean((1,2)).clamp_min(1e-4))
    if not actual.abs().sum()>0 or not torch.isfinite(actual).all() or (actual.flatten(1).norm(dim=1)>parent.CAP*iqnorm*(1+2e-5)).any():raise ValueError('Actual SIG guide gradient cap failed')
    loss.mean().backward()
    for prefix in ['frozen.coarse.','frozen.estimator.','prior.']:
        gs=[v.grad for k,v in student.named_parameters() if k.startswith(prefix) and v.grad is not None]
        if not gs or not all(torch.isfinite(v).all() for v in gs) or not any(v.abs().sum()>0 for v in gs):raise ValueError('Actual joint gradient missing: '+prefix)
    # Both the original prior output and new modulation output start at zero.
    # Their expected initial dormant upstream derivatives are not an error.
    if any(v.grad is None or not torch.isfinite(v.grad).all() for prior in student.prior for v in prior.conditioner.parameters()):
        raise ValueError('Context modulation disconnected or nonfinite')
    with torch.no_grad():expected=student(values['received']).clone()
    checkpoint=p['output']/'smoke-scratch_joint.pt';f_guided.save_model(checkpoint,student,'scratch_joint',0,0,args.commit)
    loaded=parent.load_model(checkpoint,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    from experiments.pamf_stage2_inference.engine import TensorPipeline
    with torch.no_grad():
        if not torch.equal(loaded(values['received']),expected):raise ValueError('Strict context FiLM E0 reload changed IQ')
        optimized,valid=TensorPipeline(loaded,None,reuse=True,native=False)(values['received'])
        if not valid:raise ValueError('Optimized context FiLM inference invalid')
        torch.testing.assert_close(optimized,expected,atol=3e-5,rtol=3e-5);scores=reader(optimized)
    if scores.shape!=(2,6) or not torch.isfinite(scores).all() or any(v.grad is not None for v in reader.parameters()) or any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()):
        raise ValueError('Frozen original F/defaulthead changed')
    base.write_json(p['output']/'smoke.json',dict(status='PASS',real_nonclean_Ls=2,G_inputs=12,
        F_full_inputs=2,F_prefix_inputs=4,F_input_VJP_inputs=2,all_original_G_initial_parameters_and_IQ_matched=True,
        joint_G_gradients=True,conditioner_gradients_connected_finite=True,zero_output_initial_dormancy_expected=True,
        strict_E0_reload=True,native_optimized_match=True,G_parameters=sum(v.numel() for v in student.parameters()),
        context_parameters=student.initialization_report['context_film_parameters'],SIG_IQ_gradient_cap=.5,
        F_unchanged=True,F_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,G_initialization_checkpoint_loads=0,
        source_cache_reuse=reuse,teacher_load=rng_info))
    print(json.dumps(dict(event='REAL_SOURCE_CONTEXT_FILM_E30_SMOKE_PASS')),flush=True)


def configure():
    parent.configure();previous_configuration=base.configuration;previous_command=base.command
    def configuration():
        value=previous_configuration();value.update(run=RUN,recipe='scratch_E30_SIG_received_context_FiLM',parameters=281415,
            context_film_parameters=14464,ConvLinear_MACs=25082088,
            context_film_input='per-recordprior9channelmean-energy-adjacent-difference-energy/27D',
            context_film_hidden=HIDDEN,context_film_bound=BOUND,context_film_levels=4,context_film_prior_count=2,
            context_film_output_zero_init=True,context_film_init_seed_offset=2026103110,
            context_film_scene_RX_state_labels=False,context_film_inference_active=True,
            matched_control=parent.RUN,matched_control_commit='47591cdfe2e6e6dd7534cdea846badd942321b8f',
            F_clean_cache_reused_from_run=parent.RUN,new_F_clean_cache_inputs=0,new_F_clean_cache_numeric_bytes=0,
            training_regime='source_F_guided_actual_SIG_prefix_single_joint_E30_plus_received_context_FiLM',
            source_qualification_claim=False,source_diagnostic_failures_preserved=True,
            context_claim='Learnedglobalreceived-derivedpriorconditioning;nottruechannel-stateorhardwareextraction')
        return value
    def command(args,mode,arm=None,role=None):return [MODULE if x==parent.MODULE else x for x in previous_command(args,mode,arm,role)]
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.scratch_model=factory;base.smoke=smoke
    parent.build_targets=reuse_targets


if __name__=='__main__':configure();base.main()
