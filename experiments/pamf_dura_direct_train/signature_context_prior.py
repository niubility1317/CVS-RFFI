"""Scratch E30 with two source-SIG prior predictors and optional decoder context."""
import json
import numpy as np
import torch
from . import signature_guided as parent,f_guided
from .antithetic_noise import reuse_targets
from experiments.pamf_dura_efficiency.model import scratch_model as original_factory,direct_objective as source_objective
from experiments.pamf_dura_efficiency.signature_guidance import signature_guide,CAP
from experiments.pamf_dura_efficiency.signature_context_prior import (
    attach,signature_errors,WEIGHT,BOUND,BINS,DIMENSION,SEED_OFFSET,ADDED_PARAMETERS,TOTAL_PARAMETERS,
    AUX_ONLY_ADDED_PARAMETERS,AUX_ONLY_TOTAL_PARAMETERS,CONV_LINEAR_MAC,METRICS)

base=parent.base
RUN='20261031-phase1-DURA-SIGcontextPrior-E30-s2026092701-r01'
AUX_RUN='20261031-phase1-DURA-SIGpriorAux-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.signature_context_prior'


def objective(student,reader,received,clean,action,variance,probes,target):
    _,terms,restored,predictions=source_objective(student,received,clean,action,variance,probes,return_prior_signatures=True)
    errors=signature_errors(predictions,target);error=errors.mean(0);aux=WEIGHT*error
    with torch.random.fork_rng(devices=[restored.device.index] if restored.is_cuda else []):
        guide,info=signature_guide(reader,restored,clean,target)
    loss=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']+guide+aux
    return loss,dict(terms,**{parent.METRIC:info['feature_error'],parent.GUIDE:guide,
        'F_balance_weight':info['weight'],'F_balance_limited':(info['weight']<1).float(),
        'F_raw_pressure':info['raw_pressure'],'F_balanced_pressure':info['bounded_pressure'],
        'F_pressure_defined_fraction':info['pressure_defined'].float(),'F_guide_value_exceeds_IQ':(guide.detach()>terms['iq_nmse'].detach()).float(),
        METRICS[0]:error,METRICS[1]:aux,METRICS[2]:errors[0],METRICS[3]:errors[1]}),restored


def smoke(args,p):
    clean,roles=base.source_training(p);targets,reuse=reuse_targets(p,None,clean,roles);folder=p['cache']/'epochs/01'
    variance=np.load(folder/'variance.npy',mmap_mode='r',allow_pickle=False);take=np.flatnonzero((roles=='L_s')&(variance>np.float32(1e-8)))[:2]
    if len(take)!=2:raise ValueError('Two actual nonclean source L_s required')
    values={k:base.tensor(np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu') for k in ['received','kernel','coefficients','variance']}
    x=base.tensor(clean[take],'cpu');target=base.tensor(targets[take],'cpu');student=base.scratch_model(2026092701);original=original_factory(2026092701)
    context=base.configuration()['prior_SIG_context_decoder_enabled']
    if any(not torch.equal(v,student.state_dict()[k]) for k,v in original.state_dict().items()):raise ValueError('Original G initial parameters changed')
    with torch.no_grad():torch.testing.assert_close(student(values['received']),original(values['received']),atol=0,rtol=0)
    reader,rng_info=parent.load_teacher_preserving_rng(p,'cpu');before={k:v.clone() for k,v in reader.state_dict().items()}
    with torch.no_grad():
        features,parts=parent.components(reader,x);torch.testing.assert_close(parts['signature'],target,atol=3e-5,rtol=2e-5);reader.classify_features(features)
    generator=torch.Generator().manual_seed(2026103004);probes=[torch.complex(torch.randn(2,256,generator=generator),torch.randn(2,256,generator=generator)) for _ in range(2)]
    loss,terms,restored=objective(student,reader,values['received'],x,base.supervised_action(values['kernel'],values['coefficients']),values['variance'],probes,target)
    torch.testing.assert_close(terms[METRICS[0]],.5*(terms[METRICS[2]]+terms[METRICS[3]]),atol=1e-8,rtol=1e-6)
    torch.testing.assert_close(terms[METRICS[1]],WEIGHT*terms[METRICS[0]],atol=1e-8,rtol=1e-6)
    selected={}
    for index,core in enumerate(student.prior):
        selected[f'prior{index}_shallow']=core.encoder.blocks[0][0].net[0].weight
        selected[f'prior{index}_middle']=core.encoder.blocks[1][0].net[0].weight
        selected[f'prior{index}_prediction']=core.signature_projection.weight
    grads=torch.autograd.grad(terms[METRICS[1]].sum(),list(selected.values()),retain_graph=True)
    if not all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grads):raise ValueError('Actual prior source-SIG auxiliary gradients missing')
    actual=torch.autograd.grad(terms[parent.GUIDE].sum(),restored,retain_graph=True)[0]
    iqnorm=2*(restored.detach()-x).flatten(1).norm(dim=1)/(512*x.square().mean((1,2)).clamp_min(1e-4))
    if not actual.abs().sum()>0 or not torch.isfinite(actual).all() or (actual.flatten(1).norm(dim=1)>CAP*iqnorm*(1+2e-5)).any():raise ValueError('Original SIG output-IQ gradient cap failed')
    loss.mean().backward()
    for prefix in ['frozen.coarse.','frozen.estimator.','prior.']:
        gs=[v.grad for k,v in student.named_parameters() if k.startswith(prefix) and v.grad is not None]
        if not gs or not all(torch.isfinite(g).all() for g in gs) or not any(g.abs().sum()>0 for g in gs):raise ValueError('Actual joint G gradient missing: '+prefix)
    if context and any(p.grad is None or not torch.isfinite(p.grad).all() for core in student.prior for p in core.signature_conditioner.parameters()):
        raise ValueError('Decoder context disconnected or nonfinite')
    # Initial prior IQ outputs and new conditioners are zero-initialized.
    # Zero conditioner derivatives at E0 are expected, while the source-SIG
    # auxiliary signal above already reaches both shallow prior encoders.
    calls=[];handles=[]
    for core in student.prior:
        handles.append(core.signature_projection.register_forward_hook(lambda *a:calls.append('prediction')))
        if core.signature_conditioner is not None:handles.append(core.signature_conditioner.register_forward_hook(lambda *a:calls.append('conditioner')))
    try:
        with torch.no_grad():expected=student(values['received']).clone()
    finally:
        for handle in handles:handle.remove()
    expected_calls=2 if context else 0
    if calls.count('prediction')!=expected_calls or calls.count('conditioner')!=expected_calls:raise ValueError('Registered inference feature path differs')
    checkpoint=p['output']/'smoke-scratch_joint.pt';f_guided.save_model(checkpoint,student,'scratch_joint',0,0,args.commit)
    loaded=parent.load_model(checkpoint,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    from experiments.pamf_stage2_inference.engine import TensorPipeline
    with torch.no_grad():
        if not torch.equal(loaded(values['received']),expected):raise ValueError('Strict source-SIG context E0 reload changed IQ')
        optimized,valid=TensorPipeline(loaded,None,reuse=True,native=False)(values['received'])
        if not valid:raise ValueError('Optimized context inference invalid')
        torch.testing.assert_close(optimized,expected,atol=3e-5,rtol=3e-5);scores=reader(optimized)
    if scores.shape!=(2,6) or not torch.isfinite(scores).all() or any(p.grad is not None for p in reader.parameters()) or any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()):raise ValueError('Original F/defaulthead changed')
    base.write_json(p['output']/'smoke.json',dict(status='PASS',real_nonclean_Ls=2,G_inputs=12,F_full_inputs=4,F_prefix_inputs=2,F_input_VJP_inputs=2,
        smoke_aux_parameter_VJP_inputs=2,source_SIG_aux_predictions=4,initial_base_parameters_and_IQ_matched=True,
        prior_SIG_aux_gradient_norms={k:float(g.norm()) for k,g in zip(selected,grads)},decoder_context_enabled=context,
        actual_inference_prediction_calls=calls.count('prediction'),actual_inference_conditioner_calls=calls.count('conditioner'),
        zero_output_initial_conditioner_dormancy_expected=True,strict_E0_reload=True,native_optimized_match=True,
        G_parameters=sum(p.numel() for p in student.parameters()),SIG_IQ_gradient_cap=CAP,aux_gradient_outside_output_cap=True,
        F_unchanged=True,F_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,G_initialization_checkpoint_loads=0,source_cache_reuse=reuse,teacher_load=rng_info))
    print(json.dumps(dict(event='REAL_SOURCE_SIG_CONTEXT_PRIOR_E30_SMOKE_PASS',decoder_context=context)),flush=True)


def configure(*,decoder_context=True,run=RUN,module=MODULE):
    parent.configure();previous_configuration=base.configuration;previous_command=base.command;previous_paths=base.paths
    def factory(seed):return attach(original_factory(seed),seed,decoder_context=decoder_context)
    def configuration():
        value=previous_configuration();value.update(run=run,recipe='scratch_E30_source_SIG_prior_context' if decoder_context else 'scratch_E30_source_SIG_prior_aux_only',
            parameters=TOTAL_PARAMETERS if decoder_context else AUX_ONLY_TOTAL_PARAMETERS,
            prior_SIG_context_parameters=ADDED_PARAMETERS if decoder_context else AUX_ONLY_ADDED_PARAMETERS,
            ConvLinear_MACs=CONV_LINEAR_MAC if decoder_context else 25068072,prior_SIG_context_heads=2,
            prior_SIG_context_dimension=DIMENSION,prior_SIG_context_bins=BINS,prior_SIG_context_decoder_levels=3,
            prior_SIG_context_bound=BOUND,prior_SIG_context_seed_offset=SEED_OFFSET,prior_SIG_context_decoder_enabled=bool(decoder_context),
            prior_SIG_context_inference_active=bool(decoder_context),prior_SIG_context_inference_heads=2 if decoder_context else 0,
            prior_SIG_context_aux_weight=WEIGHT,prior_SIG_context_aux_reduction='mean_of_both_prior_relativeMSE_per_record/thenbatchmean',
            prior_SIG_context_aux_predictions=3780000,prior_SIG_context_aux_gradient_outside_output_SIG_cap=True,
            prior_SIG_context_conditioner_zero_init=True,prior_SIG_context_received_only=True,prior_SIG_context_TX_RX_scene_inputs=False,
            F_clean_cache_reused_from_run=parent.RUN,new_F_clean_cache_inputs=0,new_F_clean_cache_numeric_bytes=0,
            matched_control=parent.RUN,matched_control_commit='47591cdfe2e6e6dd7534cdea846badd942321b8f',
            matched_prior_aux_run=AUX_RUN,matched_prior_context_run=RUN,
            training_regime='source_F_guided_SIG_output_plus_internal_prior_SIG_context' if decoder_context else 'source_F_guided_SIG_output_plus_trainonly_prior_SIG_aux',
            objective='originalfourphysical+actualSIGalpha1cap.5+.01mean2priorSIGrelativeMSE;varianceauxloggedonly',
            source_qualification_claim=False,old_diagnostic_FAILs_preserved=True,
            CPU_scorer_coordination_lock='runs/'+RUN+'/score.lock')
        return value
    def command(args,mode,arm=None,role=None):return [module if x==parent.MODULE else x for x in previous_command(args,mode,arm,role)]
    def paths(root):
        value=previous_paths(root);value['score_lock']=root/'runs'/RUN/'score.lock';return value
    base.RUN=run;base.configuration=configuration;base.command=command;base.scratch_model=factory;base.smoke=smoke;base.paths=paths
    parent.build_targets=reuse_targets;parent.objective=objective;parent.EXTRA=(*parent.EXTRA,*METRICS)


if __name__=='__main__':configure();base.main()
