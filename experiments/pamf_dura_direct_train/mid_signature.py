"""Scratch joint E30 with actual source SIG supervision at G temporal level 2."""
import json,time
import numpy as np
import torch
from . import signature_guided as parent,f_guided
from experiments.pamf_dura_efficiency.model import scratch_model as original_factory
from experiments.pamf_dura_efficiency.mid_signature import attach,capture,auxiliary_loss,auxiliary_gradient_parameters,WEIGHT,PARAMETERS,METRIC,GUIDE

base=parent.base
RUN='20261031-phase1-DURA-midSIG-E30-s2026092701-r02'
MODULE='experiments.pamf_dura_direct_train.mid_signature'
_SIG_OBJECTIVE=parent.objective


def factory(seed):return attach(original_factory(seed),seed)


def objective(student,reader,received,clean,action,variance,probes,target):
    (loss,terms,restored),h=capture(student,lambda:_SIG_OBJECTIVE(student,reader,received,clean,action,variance,probes,target))
    error=auxiliary_loss(student,h,target);weighted=WEIGHT*error
    return loss+weighted,dict(terms,**{METRIC:error,GUIDE:weighted}),restored


def reuse_targets(p,reader,clean,roles):
    started=time.perf_counter();ids=f_guided.source_ids(p,roles)
    folder=p['output'].parent/parent.RUN/'source_signature_cache';m=base.read_json(folder/'complete.json')
    expected=dict(status='SOURCE_ACTUAL_SIG_COMPLETE',count=63000,shape=[63000,160],numeric_bytes=40320000,
        source_contract='original7/63/30',F_teacher=f_guided.teacher_reference(),actual_weighted_TripleSIG=True,
        full_F_inputs=0,F_prefix_inputs=63000,F_input_VJP_inputs=0,F_updates=0,model_optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0)
    if any(m.get(k)!=v for k,v in expected.items()) or not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids):
        raise ValueError('Same-source original F50 actual clean SIG cache/order required')
    targets=np.load(folder/'targets.npy',mmap_mode='r',allow_pickle=False)
    if targets.shape!=(63000,160) or targets.dtype!=np.float32 or not np.isfinite(targets).all():raise ValueError('Complete finite source SIG cache required')
    reuse=dict(status='SOURCE_ACTUAL_SIG_REUSED',count=63000,shape=[63000,160],numeric_bytes=40320000,
        source_contract='original7/63/30',F_teacher=f_guided.teacher_reference(),reused_from_run=parent.RUN,
        original_metadata=m,new_F_prefix_inputs=0,new_F_input_VJP_inputs=0,new_numeric_bytes=0,
        TX_labels_read=0,V_reads=0,target_reads=0,seconds=time.perf_counter()-started)
    return targets,reuse


def smoke(args,p):
    clean,roles=base.source_training(p);targets,reuse=reuse_targets(p,None,clean,roles)
    take=np.flatnonzero(roles=='L_s')[:2];folder=p['cache']/'epochs/01'
    values={k:base.tensor(np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu') for k in ['received','kernel','coefficients','variance']}
    x=base.tensor(clean[take],'cpu');target=base.tensor(targets[take],'cpu');student=factory(2026092701)
    original=original_factory(2026092701)
    if any(not torch.equal(v,student.state_dict()[k]) for k,v in original.state_dict().items()):raise ValueError('Base G initial state differs from SIG control')
    if sum(v.numel() for v in student.parameters())!=287591:raise ValueError('Complete midSIG G parameter accounting failed')
    reader,rng_info=parent.load_teacher_preserving_rng(p,'cpu');before={k:v.clone() for k,v in reader.state_dict().items()}
    with torch.no_grad():torch.testing.assert_close(parent.signature_features(reader,x),target,atol=3e-5,rtol=2e-5)
    rng=torch.Generator().manual_seed(2026103004);probes=[torch.complex(torch.randn(2,256,generator=rng),torch.randn(2,256,generator=rng)) for _ in range(2)]
    loss,terms,restored=objective(student,reader,values['received'],x,base.supervised_action(values['kernel'],values['coefficients']),values['variance'],probes,target)
    actual=torch.autograd.grad(terms[parent.GUIDE].sum(),restored,retain_graph=True)[0]
    iqnorm=2*(restored.detach()-x).flatten(1).norm(dim=1)/(512*x.square().mean((1,2)).clamp_min(1e-4))
    if not actual.abs().sum()>0 or not torch.isfinite(actual).all() or (actual.flatten(1).norm(dim=1)>parent.CAP*iqnorm*(1+2e-5)).any():raise ValueError('Online SIG input-gradient cap failed')
    selected=auxiliary_gradient_parameters(student)
    gradients=torch.autograd.grad(terms[GUIDE].sum(),list(selected.values()),retain_graph=True)
    norms={k:float(v.norm()) for k,v in zip(selected,gradients)}
    if not all(torch.isfinite(v).all() and v.abs().sum()>0 for v in gradients):raise ValueError('Actual auxiliary shared G gradients missing: '+json.dumps(norms))
    loss.mean().backward()
    for prefix in ['frozen.coarse.','frozen.estimator.','prior.','mid_signature_aux.']:
        gs=[v.grad for k,v in student.named_parameters() if k.startswith(prefix) and v.grad is not None]
        if not gs or not all(torch.isfinite(v).all() for v in gs) or not any(v.abs().sum()>0 for v in gs):raise ValueError('Actual joint gradients missing: '+prefix)
    invoked=[];handle=student.mid_signature_aux.register_forward_hook(lambda *args:invoked.append(True))
    try:
        with torch.no_grad():expected=student(values['received']).clone()
    finally:handle.remove()
    if invoked or student.frozen.coarse.temporal.encoder.blocks[1]._forward_hooks:raise ValueError('Auxiliary feature path leaked into G inference')
    checkpoint=p['output']/'smoke-scratch_joint.pt';f_guided.save_model(checkpoint,student,'scratch_joint',0,0,args.commit)
    loaded=parent.load_model(checkpoint,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    from experiments.pamf_stage2_inference.engine import TensorPipeline
    with torch.no_grad():
        if not torch.equal(loaded(values['received']),expected):raise ValueError('Strict midSIG checkpoint reload changed IQ')
        optimized,valid=TensorPipeline(loaded,None,reuse=True,native=False)(values['received'])
        if not valid:raise ValueError('Optimized midSIG inference invalid')
        torch.testing.assert_close(optimized,expected,atol=3e-5,rtol=3e-5);scores=reader(optimized)
    if scores.shape!=(2,6) or not torch.isfinite(scores).all() or any(v.grad is not None for v in reader.parameters()) or any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()):raise ValueError('Frozen original F/defaulthead changed')
    base.write_json(p['output']/'smoke.json',dict(status='PASS',real_Ls=2,F_prefix_inputs=4,F_input_VJP_inputs=2,full_F_inputs=2,G_inputs=8,
        actual_mid_aux_gradient_norms=norms,aux_projection_parameters=PARAMETERS,complete_G_parameters=287591,
        initial_spectral_output_zero_init_blocks_encoder_gradient=True,
        base_initial_G_state_matched=True,aux_train_only_no_inference_call=True,hook_cleanup=True,strict_E0_reload=True,native_optimized_close=True,
        online_SIG_input_gradient_cap=.5,aux_gradient_not_in_online_cap=True,F_unchanged=True,F_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,
        initialization_checkpoint_loads=0,source_cache_reuse=reuse,teacher_load=rng_info))
    print(json.dumps(dict(event='REAL_MID_SIG_E30_SMOKE_PASS')),flush=True)


def configure():
    parent.configure();previous_configuration=base.configuration;previous_command=base.command
    def configuration():
        value=previous_configuration();value.update(run=RUN,recipe='scratch_E30_SIG_output_plus_middle_aux',parameters=287591,
            mid_signature_aux_parameters=PARAMETERS,mid_signature_weight=WEIGHT,mid_signature_layer='frozen.coarse.temporal.encoder.blocks.1',
            mid_signature_shape=[16,128],mid_signature_bins=8,mid_signature_train_only=True,mid_signature_inference_calls=0,
            mid_aux_feature_inputs=1890000,new_F_clean_cache_inputs=0,reused_F_clean_cache_run=parent.RUN,
            auxiliary_gradient_semantics='Ordinary0.01sourcefeatureMSEgradientintocoarseandprojection;NOTincludedinonlineSIGIQ-outputcap0.5;globalGclip1stillapplies',
            matched_control=parent.RUN,matched_control_commit='47591cdfe2e6e6dd7534cdea846badd942321b8f',
            training_regime='source_F_guided_SIG_output_and_middle_aux_single_joint_E30',
            objective='fournoVarphysicalterms+unchangedonlineSIGcap0.5+0.01middleSIGrelativeMSE;sourcefeatureprojectionnotclassifier/headreplacement')
        return value
    def command(args,mode,arm=None,role=None):return [MODULE if x==parent.MODULE else x for x in previous_command(args,mode,arm,role)]
    parent.EXTRA=(*parent.EXTRA,METRIC,GUIDE);parent.objective=objective;parent.build_targets=reuse_targets
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.scratch_model=factory;base.smoke=smoke;base.__doc__=__doc__


if __name__=='__main__':configure();base.main()
