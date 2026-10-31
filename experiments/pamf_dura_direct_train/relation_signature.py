"""Scratch single E30 with frozen-source cross-RX SIG cosine relations."""
import json
import numpy as np
import torch
from . import signature_guided as parent,f_guided
from .antithetic_noise import reuse_targets
from experiments.pamf_dura_efficiency.model import direct_objective as source_objective
from experiments.pamf_dura_efficiency.relation_guidance import relation_guide,CAP

base=parent.base
RUN='20261031-phase1-DURA-relationSIG-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.relation_signature'
METRICS=('F_relation_relative_error','F_relation_crossRX_peers','F_relation_active_fraction','F_relation_teacher_squared_distance')


class RelationTargets:
    def __init__(self,signatures,rx):self.signatures,self.rx=signatures,rx
    def __getitem__(self,take):
        features=np.asarray(self.signatures[take],dtype=np.float32)
        context=np.asarray(self.rx[take],dtype=np.float32)
        if features.ndim!=2 or context.shape!=(len(features),):raise ValueError('Batched source target indices required')
        return np.concatenate((features,context[:,None]),axis=1)


def build_targets(p,reader,clean,roles):
    targets,metadata=reuse_targets(p,reader,clean,roles)
    rx=np.load(p['cache']/'train/rx.npy',mmap_mode='r',allow_pickle=False)
    if rx.shape!=(63000,) or rx.dtype!=np.int64 or set(np.unique(rx).tolist())!={1,3,4,6,8}:raise ValueError('Original five source RX contexts required')
    return RelationTargets(targets,rx),dict(metadata,relation_RX_context_count=63000,relation_RX_training_loss_only=True,
        relation_target_batch_columns=161,relation_target_full_array_copy_bytes=0,relation_extra_F_inputs=0)


def objective(student,reader,received,clean,action,variance,probes,target):
    if target.shape!=(len(received),161):raise ValueError('Source SIG160 plus source-RX training context required')
    _,terms,restored=source_objective(student,received,clean,action,variance,probes,return_restored=True)
    with torch.random.fork_rng(devices=[restored.device.index] if restored.is_cuda else []):
        guide,info=relation_guide(reader,restored,clean,target[:,:160],target[:,160].long())
    loss=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']+guide
    return loss,dict(terms,**{parent.METRIC:info['instance_error'],parent.GUIDE:guide,
        'F_balance_weight':info['weight'],'F_balance_limited':(info['weight']<1).float(),
        'F_raw_pressure':info['raw_pressure'],'F_balanced_pressure':info['bounded_pressure'],
        'F_pressure_defined_fraction':info['pressure_defined'].float(),'F_guide_value_exceeds_IQ':(guide.detach()>terms['iq_nmse'].detach()).float(),
        METRICS[0]:info['relation_error'],METRICS[1]:info['peers'],METRICS[2]:info['active'].float(),METRICS[3]:info['denominator']}),restored


def smoke(args,p):
    clean,roles=base.source_training(p);targets,reuse=build_targets(p,None,clean,roles);folder=p['cache']/'epochs/01'
    variance=np.load(folder/'variance.npy',mmap_mode='r',allow_pickle=False)
    take=np.asarray([np.flatnonzero((roles=='L_s')&(targets.rx==rx)&(variance>np.float32(1e-8)))[0] for rx in [1,3]])
    values={k:base.tensor(np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu') for k in ['received','kernel','coefficients','variance']}
    x=base.tensor(clean[take],'cpu');target=base.tensor(targets[take],'cpu');student=base.scratch_model(2026092701)
    if sum(v.numel() for v in student.parameters())!=266951:raise ValueError('Original complete G required')
    reader,rng_info=parent.load_teacher_preserving_rng(p,'cpu');before={k:v.clone() for k,v in reader.state_dict().items()}
    with torch.no_grad():
        features,parts=parent.components(reader,x);torch.testing.assert_close(parts['signature'],target[:,:160],atol=3e-5,rtol=2e-5)
        reader.classify_features(features)
    generator=torch.Generator().manual_seed(2026103004);probes=[torch.complex(torch.randn(2,256,generator=generator),torch.randn(2,256,generator=generator)) for _ in range(2)]
    loss,terms,restored=objective(student,reader,values['received'],x,base.supervised_action(values['kernel'],values['coefficients']),values['variance'],probes,target)
    if not torch.equal(terms[METRICS[1]],torch.ones(2)) or not torch.equal(terms[METRICS[2]],torch.ones(2)):raise ValueError('Both real source endpoints must participate')
    actual=torch.autograd.grad(terms[parent.GUIDE].sum(),restored,retain_graph=True)[0]
    iqnorm=2*(restored.detach()-x).flatten(1).norm(dim=1)/(512*x.square().mean((1,2)).clamp_min(1e-4))
    if not actual.abs().sum()>0 or not torch.isfinite(actual).all() or (actual.flatten(1).norm(dim=1)>CAP*iqnorm*(1+2e-5)).any():raise ValueError('Actual relation IQ-gradient cap failed')
    loss.mean().backward()
    for prefix in ['frozen.coarse.','frozen.estimator.','prior.']:
        grads=[v.grad for k,v in student.named_parameters() if k.startswith(prefix) and v.grad is not None]
        if not grads or not all(torch.isfinite(v).all() for v in grads) or not any(v.abs().sum()>0 for v in grads):raise ValueError('Actual joint G gradient missing: '+prefix)
    with torch.no_grad():expected=student(values['received']).clone()
    checkpoint=p['output']/'smoke-scratch_joint.pt';f_guided.save_model(checkpoint,student,'scratch_joint',0,0,args.commit)
    loaded=parent.load_model(checkpoint,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    from experiments.pamf_stage2_inference.engine import TensorPipeline
    with torch.no_grad():
        if not torch.equal(loaded(values['received']),expected):raise ValueError('Strict relation G E0 reload changed IQ')
        optimized,valid=TensorPipeline(loaded,None,reuse=True,native=False)(values['received'])
        if not valid:raise ValueError('Optimized G inference invalid')
        torch.testing.assert_close(optimized,expected,atol=3e-5,rtol=3e-5);scores=reader(optimized)
    if scores.shape!=(2,6) or not torch.isfinite(scores).all() or any(v.grad is not None for v in reader.parameters()) or any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()):raise ValueError('Original F/defaulthead changed')
    base.write_json(p['output']/'smoke.json',dict(status='PASS',real_nonclean_Ls=2,source_RX=[1,3],G_inputs=8,F_full_inputs=4,F_prefix_inputs=2,F_input_VJP_inputs=2,
        actual_cross_RX_relation_endpoints=True,IQ_gradient_cap=CAP,joint_G_gradients=True,strict_E0_reload=True,native_optimized_match=True,
        G_parameters=266951,G_inference_RX_or_pair_inputs=False,F_unchanged=True,F_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,
        G_initialization_checkpoint_loads=0,source_cache_reuse=reuse,teacher_load=rng_info))
    print(json.dumps(dict(event='REAL_SOURCE_RELATION_SIG_E30_SMOKE_PASS')),flush=True)


def configure():
    parent.configure();previous_configuration=base.configuration;previous_command=base.command
    def configuration():
        value=previous_configuration();value.update(run=RUN,recipe='scratch_E30_cross_RX_cosine_relation_SIG',
            F_guidance_metric=METRICS[0],F_signature_instance_error_diagnostic_only=True,
            F_relation='mean_crossRX_cosinedistance_error/mean_squared_teacher_cosinedistance_clamp1e-4',
            F_relation_all_cross_RX_pairs=True,F_relation_empty_graph_error_zero=True,F_relation_RX_training_loss_only=True,
            F_relation_teacher_context_batch_columns=161,F_relation_no_extra_F_forward_or_VJP=True,
            F_relation_derivative='fullcoupledSUMrelationerrorinputVJP;detachedper-recordcapinjection;surrogatevaluenotweightedglobalpairloss',
            F_clean_cache_reused_from_run=parent.RUN,new_F_clean_cache_inputs=0,new_F_clean_cache_numeric_bytes=0,
            matched_control=parent.RUN,matched_control_commit='47591cdfe2e6e6dd7534cdea846badd942321b8f',
            training_regime='source_F_guided_cross_RX_relation_SIG_single_joint_E30',
            objective='fourphysicalterms+sourcecrossRXrelativecosinedistanceSIGinputderivative/cap.5;instanceSIGerrorloggedonly',
            relation_inference_calls=0,relation_extra_G_parameters=0,source_qualification_claim=False,
            related_work='ParketalCVPR2019relationaldistillationinspiration;customIQcosinerelation/notoriginalRKDreproduction')
        return value
    def command(args,mode,arm=None,role=None):return [MODULE if x==parent.MODULE else x for x in previous_command(args,mode,arm,role)]
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.smoke=smoke
    parent.build_targets=build_targets;parent.objective=objective;parent.EXTRA=(*parent.EXTRA,*METRICS)


if __name__=='__main__':configure();base.main()
