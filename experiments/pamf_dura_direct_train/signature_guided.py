"""Independent scratch E30: physical IQ losses plus frozen actual TripleSIG."""
import json,os,sys,time
import numpy as np
import torch
from . import no_variance,f_guided
from .coarse_hint import learning_rate
from .exact_reference import load_teacher_preserving_rng
from experiments.pamf_dura_efficiency.model import direct_objective as source_objective
from experiments.pamf_dura_efficiency.signature_iq import signature_features
from experiments.pamf_dura_efficiency.signature_guidance import signature_guide,CAP
from experiments.pamf_dura_efficiency.branch_features import components

base=no_variance.base
RUN='20261031-phase1-DURA-signatureGuide-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.signature_guided'
CONTROL='20261031-phase1-DURA-repeatViews-E30-s2026092701-r01'
METRIC='F_signature_relative_error';GUIDE='F_signature_guide'
EXTRA=('F_balance_weight','F_balance_limited','F_raw_pressure','F_balanced_pressure','F_pressure_defined_fraction','F_guide_value_exceeds_IQ')


def load_model(path,device,*,arm,commit,epoch=None):
    return f_guided.load_model(path,device,arm=arm,commit=commit,epoch=30 if epoch is None else epoch)


def objective(student,reader,received,clean,action,variance,probes,target):
    _,terms,restored=source_objective(student,received,clean,action,variance,probes,return_restored=True)
    with torch.random.fork_rng(devices=[restored.device.index] if restored.is_cuda else []):
        guide,info=signature_guide(reader,restored,clean,target)
    loss=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']+guide
    return loss,dict(terms,**{METRIC:info['feature_error'],GUIDE:guide,'F_balance_weight':info['weight'],
        'F_balance_limited':(info['weight']<1).float(),'F_raw_pressure':info['raw_pressure'],'F_balanced_pressure':info['bounded_pressure'],
        'F_pressure_defined_fraction':info['pressure_defined'].float(),'F_guide_value_exceeds_IQ':(guide.detach()>terms['iq_nmse'].detach()).float()}),restored


def smoke(args,p):
    clean,roles=base.source_training(p);f_guided.source_ids(p,roles)
    take=np.flatnonzero(roles=='L_s')[:2];folder=p['cache']/'epochs/01'
    values={key:base.tensor(np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu') for key in ['received','kernel','coefficients','variance']}
    x=base.tensor(clean[take],'cpu').requires_grad_(True);student=base.scratch_model(2026092701)
    reader,rng_info=load_teacher_preserving_rng(p,'cpu');before={k:v.clone() for k,v in reader.state_dict().items()}
    _,parts=components(reader,x);signature=parts['signature'];direction=torch.linspace(-1,1,160)[None]
    full_vjp=torch.autograd.grad((signature*direction).sum(),x)[0]
    target=signature_features(reader,x);prefix_vjp=torch.autograd.grad((target*direction).sum(),x)[0]
    torch.testing.assert_close(signature,target,atol=2e-6,rtol=2e-6);torch.testing.assert_close(full_vjp,prefix_vjp,atol=2e-6,rtol=2e-5)
    x=x.detach();target=target.detach();rng=torch.Generator().manual_seed(2026103004)
    probes=[torch.complex(torch.randn(2,256,generator=rng),torch.randn(2,256,generator=rng)) for _ in range(2)]
    loss,terms,restored=objective(student,reader,values['received'],x,base.supervised_action(values['kernel'],values['coefficients']),values['variance'],probes,target)
    actual=torch.autograd.grad(terms[GUIDE].sum(),restored,retain_graph=True)[0]
    iqnorm=2*(restored.detach()-x).flatten(1).norm(dim=1)/(512*x.square().mean((1,2)).clamp_min(1e-4))
    if not actual.abs().sum()>0 or not torch.isfinite(actual).all() or (actual.flatten(1).norm(dim=1)>CAP*iqnorm*(1+2e-5)).any():
        raise ValueError('Actual SIG guidance gradient ceiling failed')
    loss.mean().backward()
    for prefix in ['frozen.coarse.','frozen.estimator.','prior.']:
        grads=[v.grad for k,v in student.named_parameters() if k.startswith(prefix) and v.grad is not None]
        if not grads or not all(torch.isfinite(v).all() for v in grads) or not any(v.abs().sum()>0 for v in grads):
            raise ValueError('Actual joint G gradient missing: '+prefix)
    with torch.no_grad():expected=student(values['received']).clone()
    checkpoint=p['output']/'smoke-scratch_joint.pt';f_guided.save_model(checkpoint,student,'scratch_joint',0,0,args.commit)
    loaded=f_guided.load_model(checkpoint,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    with torch.no_grad():
        if not torch.equal(loaded(values['received']),expected):raise ValueError('Strict SIG-guided G reload changed IQ')
        scores=reader(expected)
    if (scores.shape!=(2,6) or not torch.isfinite(scores).all() or any(v.grad is not None for v in reader.parameters())
        or any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items())):raise ValueError('Original frozen F/defaulthead changed')
    base.write_json(p['output']/'smoke.json',dict(status='PASS',real_Ls=2,full_F_inputs=4,F_prefix_inputs=4,F_input_VJP_inputs=6,G_inputs=6,
        exact_original_SIG_prefix_output_and_VJP=True,F_unchanged=True,F_updates=0,SIG_guide_IQ_gradient_cap=CAP,
        joint_coarse_action_prior_gradients=True,strict_reload=True,initialization_checkpoint_loads=0,TX_labels_read=0,V_reads=0,target_reads=0,teacher_load=rng_info))
    print(json.dumps(dict(event='REAL_SIGNATURE_GUIDE_E30_SMOKE_PASS')),flush=True)


def build_targets(p,reader,clean,roles):
    started=time.perf_counter();ids=f_guided.source_ids(p,roles);folder=p['output']/'source_signature_cache';folder.mkdir()
    targets=[];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(),torch.random.fork_rng(devices=[0]):
        for start in range(0,63000,512):
            targets.append(base.numpy_values(signature_features(reader,base.tensor(clean[start:start+512],'cuda:0'))))
    targets=np.concatenate(targets);torch.cuda.synchronize()
    if targets.shape!=(63000,160) or targets.dtype!=np.float32 or not np.isfinite(targets).all():
        raise ValueError('Complete actual source clean SIG cache required')
    np.save(folder/'targets.npy',targets,allow_pickle=False);np.save(folder/'physical_ids.npy',ids,allow_pickle=False)
    metadata=dict(status='SOURCE_ACTUAL_SIG_COMPLETE',count=63000,shape=[63000,160],numeric_bytes=targets.nbytes,
        source_contract='original7/63/30',F_teacher=f_guided.teacher_reference(),actual_weighted_TripleSIG=True,
        full_F_inputs=0,F_prefix_inputs=63000,F_input_VJP_inputs=0,F_updates=0,model_optimizer_updates=0,
        TX_labels_read=0,V_reads=0,target_reads=0,seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    base.write_json(folder/'complete.json',metadata);print(json.dumps(dict(event='SOURCE_ACTUAL_SIG_CACHE_COMPLETE',**metadata)),flush=True)
    return targets,metadata


def train(args,p):
    started_total=time.perf_counter();row=p['output']/args.arm;row.mkdir()
    base.write_json(row/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],commit=args.commit))
    clean,roles=base.source_training(p)
    torch.manual_seed(2026092701);torch.cuda.manual_seed_all(2026092701);student=base.scratch_model(2026092701).to('cuda:0')
    beforeG={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
    reader,rng_info=load_teacher_preserving_rng(p,'cuda:0');beforeF={k:v.detach().cpu().clone() for k,v in reader.state_dict().items()}
    targets,cache=build_targets(p,reader,clean,roles)
    base.write_json(row/'source_reference.json',cache);base.write_json(row/'teacher_load.json',rng_info)
    optimizer=torch.optim.AdamW(student.parameters(),lr=.001,weight_decay=1e-4)
    base.write_json(row/'resolved_config.json',base.configuration());base.write_json(row/'initialization.json',student.initialization_report)
    preparation=time.perf_counter()-started_total;started=time.perf_counter();rng=np.random.default_rng(2026103004);updates=seen_l=seen_u=0
    torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x',encoding='utf-8') as stream:
        for epoch in range(1,31):
            view=(epoch-1)%20+1;folder=p['cache']/'epochs'/f'{view:02d}'
            arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ['received','kernel','coefficients','variance']}
            shapes=dict(received=(63000,2,256),kernel=(63000,2,64),coefficients=(63000,4,256),variance=(63000,))
            if any(v.shape!=shapes[k] or v.dtype!=np.float32 for k,v in arrays.items()):raise ValueError('Original full source caches required')
            totals=dict.fromkeys(['loss','iq_nmse','prior_nmse','spectral_error','action_probe_nmse','log_variance_mse',METRIC,GUIDE,*EXTRA],0.)
            clipped=0;maximum_pressure=0.;order=rng.permutation(63000)
            for start in range(0,63000,256):
                take=order[start:start+256];x=base.tensor(clean[take],'cuda:0');target=base.tensor(targets[take],'cuda:0')
                values={k:base.tensor(v[take],'cuda:0') for k,v in arrays.items()};action=base.supervised_action(values['kernel'],values['coefficients'])
                probes=[torch.complex(torch.randn(len(take),256,device='cuda:0'),torch.randn(len(take),256,device='cuda:0')) for _ in range(2)]
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=base.learning_rate(updates)
                loss,terms,_=objective(student,reader,values['received'],x,action,values['variance'],probes,target)
                if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite SIG-guided source objective')
                maximum_pressure=max(maximum_pressure,float(terms['F_balanced_pressure'].max()))
                if maximum_pressure>CAP*(1+2e-5):raise FloatingPointError('SIG guide IQ gradient ceiling violated')
                loss.mean().backward();norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True);clipped+=int(norm>1)
                optimizer.step();updates+=1;seen_l+=int((roles[take]=='L_s').sum());seen_u+=int((roles[take]=='U_s').sum())
                for key,value in dict(loss=loss,**terms).items():totals[key]+=float(value.detach().sum())
            record=dict(epoch=epoch,source_cache_view=view,updates=updates,losses={k:v/63000 for k,v in totals.items()},L_exposures=seen_l,U_exposures=seen_u,
                clip_fraction=clipped/247,maximum_balanced_IQ_pressure=maximum_pressure,seconds=time.perf_counter()-started,
                F_prefix_inputs=seen_l+seen_u,F_input_VJP_inputs=seen_l+seen_u,F_full_inputs=0,F_updates=0,TX_labels_read=0,initialization_checkpoint_loads=0,
                mid_aux_feature_inputs=seen_l+seen_u if hasattr(student,'mid_signature_aux') else 0)
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();print(json.dumps(dict(event='SOURCE_SIGNATURE_GUIDE_EPOCH_COMPLETE',**record)),flush=True)
    if (updates,seen_l,seen_u)!=(7410,189000,1701000):raise ValueError('Incomplete single E30 budget')
    if any(v.grad is not None for v in reader.parameters()) or any(not torch.equal(v.detach().cpu(),beforeF[k]) for k,v in reader.state_dict().items()):
        raise ValueError('Training changed frozen original F')
    changed=sum(not torch.equal(v.detach().cpu(),beforeG[k]) for k,v in student.state_dict().items())
    if not changed:raise ValueError('Scratch G did not train')
    f_guided.save_model(row/'final.pt',student,args.arm,30,7410,args.commit)
    base.write_json(row/'training_complete.json',dict(status='SOURCE_TRAINED',arm=args.arm,epoch=30,updates=7410,L_exposures=seen_l,U_exposures=seen_u,
        training_TX_labels_read=0,training_F_access=True,V_reads=0,target_reads=0,F_updates=0,F_unchanged=True,F_teacher=f_guided.teacher_reference(),
        F_loaded_in_training=True,F_clean_teacher_inputs=cache.get('new_F_prefix_inputs',63000),F_clean_J_VJP_inputs=0,F_restored_training_inputs=1890000,
        F_prefix_training_inputs=1890000,F_full_training_inputs=0,F_input_VJP_inputs=1890000,F_J_reference_inputs=0,source_reference=cache,teacher_load=rng_info,
        initialization_checkpoint_loads=0,G_ancestor_epochs=0,cumulative_G_epochs=30,F_reused_teacher_epochs=50,head_fit_updates=0,parameters=sum(v.numel() for v in student.parameters()),
        changed_state_tensors=changed,seconds=time.perf_counter()-started,source_preparation_seconds=preparation,
        total_G_training_pipeline_seconds=time.perf_counter()-started_total,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        training_mid_aux_feature_inputs=1890000 if hasattr(student,'mid_signature_aux') else 0,
        mid_aux_parameters=sum(v.numel() for v in student.mid_signature_aux.parameters()) if hasattr(student,'mid_signature_aux') else 0,
        training_regime=base.configuration().get('training_regime','source_F_guided_actual_SIG_prefix_single_joint_E30')))


def configure():
    no_variance.configure();previous_configuration=base.configuration;previous_command=base.command
    base.__doc__=__doc__
    def configuration():
        value=previous_configuration()
        value.update(run=RUN,recipe='scratch_E30_actual_SIG_guided',epochs=30,updates=7410,new_cumulative_epochs=30,L_exposures=189000,U_exposures=1701000,
            source_freeze='fixedfinalE30/7410;noearlystop/source-target-ranking',corruption_epochs=20,source_corruption_schedule='01..20then01..10',
            CUDA_probe_seed=2026092701,numpy_shuffle_seed=2026103004,teacher_model=f_guided.teacher_reference(),F_teacher=f_guided.teacher_reference(),
            training_F_access=True,F_guided_training=True,F_guidance_metric=METRIC,F_guidance_weighted_metric=GUIDE,F_guidance_dimension=160,
            F_signature_source='actual weighted TripleSignaturePool,exactnative t2 prefix',F_guidance_alpha=1.,F_guidance_IQ_gradient_cap=CAP,
            F_clean_cache_count=63000,F_clean_cache_shape=[63000,160],F_clean_cache_numeric_bytes=40320000,F_loaded_in_training=True,
            F_prefix_training_inputs=1890000,F_input_VJP_inputs=1890000,F_full_training_inputs=0,F_load_RNG_preserved=True,F_forward_RNG_isolated=True,
            source_supervision='OriginalpairedsourceIQ/action plusactualfrozenFsourceSIGtarget;noTX/V/target;inheritsF50knowledge/notF-free',
            source_research_diagnostics=['20261031-diagnostic-DURA-source-branchFeatures-s2026092701-r01','20261031-diagnostic-DURA-source-signatureIQ-s2026092701-r01','20261031-diagnostic-DURA-source-signatureLossStep-s2026092701-r01'],
            diagnostic_primary_FAILs_preserved=True,source_qualification_claim=False,not_deploying_paired_clean_IQ_optimizer=True,
            objective='fournoVarphysicalterms+actualSIG step-local derivative cappedto0.5IQgradientnorm;noFhead/classloss',
            matched_control=CONTROL,matched_control_commit='8cf8e22c9ffb54d271b22e621401de8e34c69068',
            allowed_GPUs=[4,6,7],max_total_visible_GPU_processes=3,host_numpy_conversion='DLPack_owned_FP32')
        return value
    def command(args,mode,arm=None,role=None):return [MODULE if x==no_variance.MODULE else x for x in previous_command(args,mode,arm,role)]
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.smoke=smoke;base.train=train
    base.save_model=f_guided.save_model;base.load_model=load_model;base.learning_rate=learning_rate


if __name__=='__main__':configure();base.main()
