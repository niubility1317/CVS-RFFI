"""Original scratch SIG E30 with fixed-grid interpolation in training only."""
import json
import numpy as np
import torch
from . import signature_guided as parent,f_guided
from .antithetic_noise import reuse_targets
from experiments.pamf_dura_efficiency.linear2x import install_switch
from experiments.pamf_dura_efficiency.training_interpolation import train_with_slices

base=parent.base
RUN='20261031-phase1-DURA-SIGfastTrain-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.signature_fast_train'
SCORER_COORDINATION_RUN='20261031-phase1-DURA-coarseAnchor-E30-s2026092701-r01'


def build_targets(p,reader,clean,roles):return reuse_targets(p,reader,clean,roles)


def assert_native(model):
    from experiments.pamf_reception_restore.models import Core
    if any(type(core) is not Core for core in [model.frozen.coarse.spectral,model.frozen.coarse.temporal,*model.prior]):
        raise ValueError('Frozen G inference must use the original native four Core paths')


def smoke(args,p):
    clean,roles=base.source_training(p);targets,reuse=build_targets(p,None,clean,roles);folder=p['cache']/'epochs/01'
    variance=np.load(folder/'variance.npy',mmap_mode='r',allow_pickle=False);take=np.flatnonzero((roles=='L_s')&(variance>np.float32(1e-8)))[:2]
    if len(take)!=2:raise ValueError('Two actual nonclean source L_s required')
    values={k:base.tensor(np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu') for k in ['received','kernel','coefficients','variance']}
    x=base.tensor(clean[take],'cpu');target=base.tensor(targets[take],'cpu')
    native=base.scratch_model(2026092701);sliced=base.scratch_model(2026092701);assert_native(native);install_switch(sliced)('slice_linear2x')
    if any(not torch.equal(v,native.state_dict()[k]) for k,v in sliced.state_dict().items()):raise ValueError('Training execution changed initial G state')
    reader,rng_info=parent.load_teacher_preserving_rng(p,'cpu');before={k:v.clone() for k,v in reader.state_dict().items()}
    with torch.no_grad():
        features,parts=parent.components(reader,x);torch.testing.assert_close(parts['signature'],target,atol=3e-5,rtol=2e-5);reader.classify_features(features)
    generator=torch.Generator().manual_seed(2026103004);probes=[torch.complex(torch.randn(2,256,generator=generator),torch.randn(2,256,generator=generator)) for _ in range(2)]
    gradients=[];outputs=[];losses=[]
    for student in [native,sliced]:
        loss,terms,restored=parent.objective(student,reader,values['received'],x,base.supervised_action(values['kernel'],values['coefficients']),values['variance'],probes,target)
        actual=torch.autograd.grad(terms[parent.GUIDE].sum(),restored,retain_graph=True)[0]
        iqnorm=2*(restored.detach()-x).flatten(1).norm(dim=1)/(512*x.square().mean((1,2)).clamp_min(1e-4))
        if not actual.abs().sum()>0 or not torch.isfinite(actual).all() or (actual.flatten(1).norm(dim=1)>.5*iqnorm*(1+2e-5)).any():raise ValueError('Original actual SIG cap failed')
        loss.mean().backward()
        for prefix in ['frozen.coarse.','frozen.estimator.','prior.']:
            g=[v.grad for k,v in student.named_parameters() if k.startswith(prefix) and v.grad is not None]
            if not g or not all(torch.isfinite(v).all() for v in g) or not any(v.abs().sum()>0 for v in g):raise ValueError('Actual G gradient missing: '+prefix)
        torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True)
        gradients.append([None if v.grad is None else v.grad.clone() for v in student.parameters()]);outputs.append(restored.detach());losses.append(loss.detach())
    torch.testing.assert_close(outputs[0],outputs[1],rtol=2e-5,atol=2e-6);torch.testing.assert_close(losses[0],losses[1],rtol=2e-5,atol=2e-6)
    for a,b in zip(*gradients):
        if (a is None)!=(b is None):raise ValueError('Interpolation changed gradient participation')
        if a is not None:torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    with torch.no_grad():expected=sliced(values['received']);native_expected=native(values['received'])
    checkpoint=p['output']/'smoke-scratch_joint.pt';f_guided.save_model(checkpoint,sliced,'scratch_joint',0,0,args.commit)
    loaded=parent.load_model(checkpoint,'cpu',arm='scratch_joint',commit=args.commit,epoch=0);assert_native(loaded)
    from experiments.pamf_stage2_inference.engine import TensorPipeline
    with torch.no_grad():
        torch.testing.assert_close(loaded(values['received']),native_expected,rtol=0,atol=0)
        torch.testing.assert_close(loaded(values['received']),expected,rtol=2e-5,atol=2e-6)
        optimized,valid=TensorPipeline(loaded,None,reuse=True,native=False)(values['received'])
        if not valid:raise ValueError('Original optimized inference invalid')
        torch.testing.assert_close(optimized,native_expected,atol=3e-5,rtol=3e-5);scores=reader(optimized)
    if scores.shape!=(2,6) or not torch.isfinite(scores).all() or any(v.grad is not None for v in reader.parameters()) or any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()):raise ValueError('Original F/defaulthead changed')
    base.write_json(p['output']/'smoke.json',dict(status='PASS',real_nonclean_Ls=2,G_inputs=14,F_full_inputs=4,F_prefix_inputs=4,F_input_VJP_inputs=4,
        native_slice_same_initial_state=True,actual_training_loss_and_all_clipped_gradients_close=True,comparison_rtol=2e-5,comparison_atol=2e-6,
        strict_E0_reload=True,reloaded_G_original_native_Cores=True,native_optimized_match=True,G_parameters=266951,
        F_unchanged=True,F_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,G_initialization_checkpoint_loads=0,source_cache_reuse=reuse,teacher_load=rng_info))
    print(json.dumps(dict(event='REAL_SOURCE_SIG_FAST_TRAIN_E30_SMOKE_PASS')),flush=True)


def configure():
    parent.configure();previous_configuration=base.configuration;previous_command=base.command;previous_train=base.train;previous_load=base.load_model;previous_paths=base.paths
    def configuration():
        value=previous_configuration();value.update(run=RUN,recipe='scratch_E30_original_SIG_training_only_fixedgrid_interpolation',
            training_interpolation='12of15fixed_halfpixel_linear2x_slices/fourexistingCores',training_interpolation_calls_changed=12,training_interpolation_calls_native=3,
            frozen_inference_interpolation='OriginalnativefourCorepaths;nativefactoryrestoredaftertraining',
            training_interpolation_qualification_run='20261030-diagnostic-DURA-interpolation-s2026092701-r01',
            prior_source_step_speed_evidence='SamecardsourceB256alternating14.437percent;notE30speedguarantee',
            interpolation_trajectory_claim='Localforward-gradienttoleranceagreementnotbitwise7410AdamWtrajectoryequivalence',
            F_clean_cache_reused_from_run=parent.RUN,new_F_clean_cache_inputs=0,new_F_clean_cache_numeric_bytes=0,
            matched_control=parent.RUN,matched_control_commit='47591cdfe2e6e6dd7534cdea846badd942321b8f',
            training_regime='source_F_guided_original_SIG_single_joint_E30_training_only_interpolation',
            objective='Originalfourphysical+actualSIGalpha1cap.5;noaux/newloss/augmentation',source_qualification_claim=False)
        value['CPU_scorer_coordination_lock']='runs/'+SCORER_COORDINATION_RUN+'/score.lock'
        return value
    def command(args,mode,arm=None,role=None):return [MODULE if x==parent.MODULE else x for x in previous_command(args,mode,arm,role)]
    def train(args,p):return train_with_slices(base,previous_train,args,p)
    def load_model(*args,**kwargs):
        model=previous_load(*args,**kwargs);assert_native(model);return model
    def paths(root):
        value=previous_paths(root);value['score_lock']=root/'runs'/SCORER_COORDINATION_RUN/'score.lock';return value
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.smoke=smoke;base.train=train;base.load_model=load_model;base.paths=paths;parent.build_targets=build_targets


if __name__=='__main__':configure();base.main()
