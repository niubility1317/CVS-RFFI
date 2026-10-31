"""Scratch joint E30: original SIG guidance plus a weak source-clean coarse anchor."""
import json
import numpy as np
import torch
from . import signature_guided as parent,f_guided
from .antithetic_noise import reuse_targets
from experiments.pamf_dura_efficiency.model import direct_objective as source_objective
from experiments.pamf_dura_efficiency.coarse_anchor import coarse_anchor,ALPHA,CALIBRATION_RUN,CALIBRATION_RATIO,CALIBRATION_PRESSURE
from experiments.pamf_dura_efficiency.signature_guidance import signature_guide,CAP

base=parent.base
RUN='20261031-phase1-DURA-coarseAnchor-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.coarse_anchor'
METRICS=('coarse_clean_iq_nmse','coarse_clean_guide')


def build_targets(p,reader,clean,roles):return reuse_targets(p,reader,clean,roles)


def objective(student,reader,received,clean,action,variance,probes,target):
    _,terms,restored,coarse=source_objective(student,received,clean,action,variance,probes,return_coarse=True)
    anchor,error=coarse_anchor(coarse,clean)
    with torch.random.fork_rng(devices=[restored.device.index] if restored.is_cuda else []):
        guide,info=signature_guide(reader,restored,clean,target)
    loss=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']+guide+anchor
    return loss,dict(terms,**{parent.METRIC:info['feature_error'],parent.GUIDE:guide,
        'F_balance_weight':info['weight'],'F_balance_limited':(info['weight']<1).float(),
        'F_raw_pressure':info['raw_pressure'],'F_balanced_pressure':info['bounded_pressure'],
        'F_pressure_defined_fraction':info['pressure_defined'].float(),'F_guide_value_exceeds_IQ':(guide.detach()>terms['iq_nmse'].detach()).float(),
        METRICS[0]:error,METRICS[1]:anchor}),restored


def smoke(args,p):
    clean,roles=base.source_training(p);targets,reuse=build_targets(p,None,clean,roles);folder=p['cache']/'epochs/01'
    variance=np.load(folder/'variance.npy',mmap_mode='r',allow_pickle=False)
    take=np.flatnonzero((roles=='L_s')&(variance>np.float32(1e-8)))[:2]
    if len(take)!=2:raise ValueError('Two actual nonclean source L_s required')
    values={k:base.tensor(np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu') for k in ['received','kernel','coefficients','variance']}
    x=base.tensor(clean[take],'cpu');target=base.tensor(targets[take],'cpu');student=base.scratch_model(2026092701)
    reader,rng_info=parent.load_teacher_preserving_rng(p,'cpu');before={k:v.clone() for k,v in reader.state_dict().items()}
    with torch.no_grad():
        features,parts=parent.components(reader,x);torch.testing.assert_close(parts['signature'],target,atol=3e-5,rtol=2e-5)
        reader.classify_features(features)
    generator=torch.Generator().manual_seed(2026103004)
    probes=[torch.complex(torch.randn(2,256,generator=generator),torch.randn(2,256,generator=generator)) for _ in range(2)]
    loss,terms,restored=objective(student,reader,values['received'],x,base.supervised_action(values['kernel'],values['coefficients']),values['variance'],probes,target)
    torch.testing.assert_close(terms[METRICS[1]],ALPHA*terms[METRICS[0]],atol=1e-8,rtol=1e-6)
    named=list(student.named_parameters());aux=torch.autograd.grad(terms[METRICS[1]].mean(),[v for _,v in named],allow_unused=True,retain_graph=True)
    coarse_nonzero=False
    for (name,_),g in zip(named,aux):
        if name.startswith('frozen.coarse.'):
            if g is not None:
                if not torch.isfinite(g).all():raise ValueError('Finite coarse anchor gradient required')
                coarse_nonzero=coarse_nonzero or bool(g.abs().sum()>0)
        elif g is not None and bool(g.abs().sum()>0):raise ValueError('Coarse anchor unexpectedly changes action/prior parameters directly')
    if not coarse_nonzero:raise ValueError('Actual coarse anchor derivative missing')
    actual=torch.autograd.grad(terms[parent.GUIDE].sum(),restored,retain_graph=True)[0]
    iqnorm=2*(restored.detach()-x).flatten(1).norm(dim=1)/(512*x.square().mean((1,2)).clamp_min(1e-4))
    if not actual.abs().sum()>0 or not torch.isfinite(actual).all() or (actual.flatten(1).norm(dim=1)>CAP*iqnorm*(1+2e-5)).any():raise ValueError('Original actual SIG IQ-gradient cap failed')
    loss.mean().backward()
    for prefix in ['frozen.coarse.','frozen.estimator.','prior.']:
        grads=[v.grad for k,v in student.named_parameters() if k.startswith(prefix) and v.grad is not None]
        if not grads or not all(torch.isfinite(v).all() for v in grads) or not any(v.abs().sum()>0 for v in grads):raise ValueError('Actual joint G gradient missing: '+prefix)
    with torch.no_grad():expected=student(values['received']).clone()
    checkpoint=p['output']/'smoke-scratch_joint.pt';f_guided.save_model(checkpoint,student,'scratch_joint',0,0,args.commit)
    loaded=parent.load_model(checkpoint,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    from experiments.pamf_stage2_inference.engine import TensorPipeline
    with torch.no_grad():
        if not torch.equal(loaded(values['received']),expected):raise ValueError('Strict coarse-anchor E0 reload changed IQ')
        optimized,valid=TensorPipeline(loaded,None,reuse=True,native=False)(values['received'])
        if not valid:raise ValueError('Optimized G inference invalid')
        torch.testing.assert_close(optimized,expected,atol=3e-5,rtol=3e-5);scores=reader(optimized)
    if scores.shape!=(2,6) or not torch.isfinite(scores).all() or any(v.grad is not None for v in reader.parameters()) or any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()):raise ValueError('Original F/defaulthead changed')
    base.write_json(p['output']/'smoke.json',dict(status='PASS',real_nonclean_Ls=2,G_inputs=8,F_full_inputs=4,F_prefix_inputs=2,F_input_VJP_inputs=2,
        smoke_only_coarse_aux_parameter_VJP_inputs=2,actual_anchor_coarse_only_nonzero_gradient=True,coarse_clean_alpha=ALPHA,
        SIG_IQ_gradient_cap=CAP,joint_G_gradients=True,strict_E0_reload=True,native_optimized_match=True,G_parameters=266951,
        F_unchanged=True,F_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,G_initialization_checkpoint_loads=0,source_cache_reuse=reuse,teacher_load=rng_info))
    print(json.dumps(dict(event='REAL_SOURCE_COARSE_ANCHOR_E30_SMOKE_PASS')),flush=True)


def configure():
    parent.configure();previous_configuration=base.configuration;previous_command=base.command
    def configuration():
        value=previous_configuration();value.update(run=RUN,recipe='scratch_E30_original_SIG_plus_weak_coarse_clean_anchor',
            coarse_clean_alpha=ALPHA,coarse_clean_calibration_run=CALIBRATION_RUN,coarse_clean_calibration_ratio=CALIBRATION_RATIO,
            coarse_clean_calibration_pressure=CALIBRATION_PRESSURE,coarse_clean_calibration_scope='Historicalsource-noVarE20/raw32batchcoarsegradientratio-to-finalIQ;notactualboundorcurrentSIGGcalibration',
            coarse_clean_all_epochs=True,coarse_clean_target='OriginalpairedsourcecleanIQ/notR17hintornewFtarget',
            coarse_clean_direct_gradient_modules=['coarse'],coarse_clean_extra_G_or_F_forward=0,coarse_clean_extra_training_parameter_VJP=0,
            F_clean_cache_reused_from_run=parent.RUN,new_F_clean_cache_inputs=0,new_F_clean_cache_numeric_bytes=0,
            matched_control=parent.RUN,matched_control_commit='47591cdfe2e6e6dd7534cdea846badd942321b8f',
            training_regime='source_F_guided_original_SIG_weak_coarse_clean_single_joint_E30',
            objective='originalfourphysical+actualSIGalpha1cap.5+fixedalpha*coarsecleanNMSE;varianceauxloggedonly',
            source_qualification_claim=False,old_gradient_separation_primary_FAIL_preserved=True,coarse_clean_new_G_parameters=0)
        return value
    def command(args,mode,arm=None,role=None):return [MODULE if x==parent.MODULE else x for x in previous_command(args,mode,arm,role)]
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.smoke=smoke
    parent.build_targets=build_targets;parent.objective=objective;parent.EXTRA=(*parent.EXTRA,*METRICS)


if __name__=='__main__':configure();base.main()
