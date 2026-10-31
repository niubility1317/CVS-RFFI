"""Scratch single E30 with deeper-feature SIG residual predictors."""
import json
import numpy as np
import torch
from . import signature_context_prior as parent
from experiments.pamf_dura_efficiency.signature_deep_prior import attach,ADDED,PARAMETERS,MACS,SEED_OFFSET

base=parent.base
RUN='20261031-phase1-DURA-SIGdeepLinear-E30-s2026092701-r01'
MLP_RUN='20261031-phase1-DURA-SIGdeepMLP-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.signature_deep_prior'
previous_factory=None


def smoke(args,p):
    clean,roles=base.source_training(p);targets,reuse=parent.reuse_targets(p,None,clean,roles);folder=p['cache']/'epochs/01'
    variance=np.load(folder/'variance.npy',mmap_mode='r',allow_pickle=False);take=np.flatnonzero((roles=='L_s')&(variance>np.float32(1e-8)))[:2]
    if len(take)!=2:raise ValueError('Two actual nonclean source L_s required')
    values={k:base.tensor(np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu') for k in ['received','kernel','coefficients','variance']}
    x=base.tensor(clean[take],'cpu');target=base.tensor(targets[take],'cpu');student=base.scratch_model(2026092701);original=previous_factory(2026092701)
    if any(not torch.equal(v,student.state_dict()[k]) for k,v in original.state_dict().items()):raise ValueError('Main G initial parameters changed')
    with torch.no_grad():torch.testing.assert_close(student(values['received']),original(values['received']),atol=0,rtol=0)
    reader,rng_info=parent.parent.load_teacher_preserving_rng(p,'cpu');before={k:v.clone() for k,v in reader.state_dict().items()}
    with torch.no_grad():
        features,parts=parent.parent.components(reader,x);torch.testing.assert_close(parts['signature'],target,atol=3e-5,rtol=2e-5);reader.classify_features(features)
    generator=torch.Generator().manual_seed(2026103004);probes=[torch.complex(torch.randn(2,256,generator=generator),torch.randn(2,256,generator=generator)) for _ in range(2)]
    loss,terms,restored=parent.objective(student,reader,values['received'],x,base.supervised_action(values['kernel'],values['coefficients']),values['variance'],probes,target)
    selected={}
    for index,core in enumerate(student.prior):
        last=core.deep_signature_projection if core.deep_mode=='linear' else core.deep_signature_projection[-1]
        selected[f'prior{index}_deep_output']=last.weight;selected[f'prior{index}_base_prediction']=core.signature_projection.weight
    grads=torch.autograd.grad(terms[parent.METRICS[1]].sum(),list(selected.values()),retain_graph=True)
    if not all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grads):raise ValueError('Actual source auxiliary gradients to deep output heads missing')
    actual=torch.autograd.grad(terms[parent.parent.GUIDE].sum(),restored,retain_graph=True)[0]
    iqnorm=2*(restored.detach()-x).flatten(1).norm(dim=1)/(512*x.square().mean((1,2)).clamp_min(1e-4))
    if not actual.abs().sum()>0 or not torch.isfinite(actual).all() or (actual.flatten(1).norm(dim=1)>parent.CAP*iqnorm*(1+2e-5)).any():raise ValueError('Original SIG output IQ gradient cap failed')
    loss.mean().backward()
    for prefix in ['frozen.coarse.','frozen.estimator.','prior.']:
        gs=[v.grad for k,v in student.named_parameters() if k.startswith(prefix) and v.grad is not None]
        if not gs or not all(torch.isfinite(g).all() for g in gs) or not any(g.abs().sum()>0 for g in gs):raise ValueError('Joint G gradient missing: '+prefix)
    for core in student.prior:
        if core.deep_mode=='mlp':
            g=core.deep_signature_projection[0].weight.grad
            if g is None or not torch.isfinite(g).all() or not torch.equal(g,torch.zeros_like(g)):raise ValueError('Zero-output-initialized MLP first-layer gradient differs')
    calls=[];handles=[q.deep_signature_projection.register_forward_hook(lambda *args:calls.append('deep')) for q in student.prior]
    try:
        with torch.no_grad():expected=student(values['received']).clone()
    finally:
        for handle in handles:handle.remove()
    if calls!=['deep','deep']:raise ValueError('Actual deep predictor inference path differs')
    checkpoint=p['output']/'smoke-scratch_joint.pt';parent.f_guided.save_model(checkpoint,student,'scratch_joint',0,0,args.commit)
    loaded=base.load_model(checkpoint,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    from experiments.pamf_stage2_inference.engine import TensorPipeline
    with torch.no_grad():
        if not torch.equal(loaded(values['received']),expected):raise ValueError('Strict deep predictor E0 reload changed IQ')
        optimized,valid=TensorPipeline(loaded,None,reuse=True,native=False)(values['received'])
        if not valid:raise ValueError('Optimized deep predictor inference invalid')
        torch.testing.assert_close(optimized,expected,atol=3e-5,rtol=3e-5);scores=reader(optimized)
    if scores.shape!=(2,6) or not torch.isfinite(scores).all() or any(p.grad is not None for p in reader.parameters()) or any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()):raise ValueError('Frozen original F/defaulthead changed')
    base.write_json(p['output']/'smoke.json',dict(status='PASS',real_nonclean_Ls=2,G_inputs=12,F_full_inputs=4,F_prefix_inputs=2,F_input_VJP_inputs=2,
        smoke_aux_parameter_VJP_inputs=2,deep_SIG_mode=student.prior[0].deep_mode,main_initial_parameters_and_IQ_matched=True,
        source_aux_gradient_norms={k:float(g.norm()) for k,g in zip(selected,grads)},MLP_first_layer_zero_gradient_expected=True,
        actual_deep_inference_calls=2,strict_E0_reload=True,native_optimized_match=True,G_parameters=sum(p.numel() for p in student.parameters()),
        SIG_IQ_gradient_cap=.5,aux_gradient_outside_output_cap=True,F_unchanged=True,F_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,
        G_initialization_checkpoint_loads=0,source_cache_reuse=reuse,teacher_load=rng_info))
    print(json.dumps(dict(event='REAL_SOURCE_SIG_DEEP_PRIOR_E30_SMOKE_PASS',mode=student.prior[0].deep_mode)),flush=True)


def configure(*,mode='linear'):
    global previous_factory
    if mode not in PARAMETERS:raise ValueError('Fixed linear or mlp deep predictor required')
    parent.configure();previous_factory=base.scratch_model;previous_configuration=base.configuration;previous_command=base.command;previous_paths=base.paths
    run=RUN if mode=='linear' else MLP_RUN;module=MODULE if mode=='linear' else 'experiments.pamf_dura_direct_train.signature_deep_mlp'
    def factory(seed):return attach(previous_factory(seed),seed,mode)
    def configuration():
        value=previous_configuration();value.update(run=run,recipe='scratch_E30_deep_SIG_'+mode,parameters=PARAMETERS[mode],ConvLinear_MACs=MACS[mode],
            deep_SIG_mode=mode,deep_SIG_parameters=ADDED[mode],deep_SIG_dimension=192,deep_SIG_encoder_levels=[2,3],deep_SIG_pool_bins=[4,2],
            deep_SIG_seed_offset=SEED_OFFSET,deep_SIG_zero_output_init=True,deep_SIG_base_prediction_preserved=True,deep_SIG_predictions=3780000,
            source_oracle_evidence_run='20261031-diagnostic-DURA-source-priorContext-oracle-s2026092701-r01',
            source_oracle_evidence_commit='9d266929d4e0eea27057eebeda8239fd24f0b6d9',source_oracle_evidence_decision='PASS',
            source_qualification_claim=False,source_oracle_is_privileged_not_deployed=True,
            matched_control=parent.RUN,matched_control_commit='4452e09285b650536ab5e9dcc2acbfd9268c00e8',matched_deep_linear_run=RUN,matched_deep_MLP_run=MLP_RUN,
            CPU_scorer_coordination_lock='runs/'+RUN+'/score.lock')
        return value
    def command(args,task,arm=None,role=None):return [module if x==parent.MODULE else x for x in previous_command(args,task,arm,role)]
    def paths(root):
        value=previous_paths(root);value['score_lock']=root/'runs'/RUN/'score.lock';return value
    base.RUN=run;base.scratch_model=factory;base.configuration=configuration;base.command=command;base.paths=paths;base.smoke=smoke


if __name__=='__main__':configure();base.main()
