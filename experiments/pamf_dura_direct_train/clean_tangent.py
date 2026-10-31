"""Single joint E30 with cached source-clean frozen-F local fidelity."""
import gc
import json
import os
import sys
import time
import numpy as np
import torch
from . import f_guided
from .coarse_hint import learning_rate
from experiments.pamf_dura_efficiency.model import direct_objective as source_objective
from experiments.pamf_dura_efficiency.clean_tangent import clean_jacobian,tangent_loss
from experiments.pamf_dura_efficiency.head_features import head_feature_basis

base=f_guided.base
RUN='20261031-phase1-DURA-cleanTangent-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.clean_tangent'
PROBE='20261031-diagnostic-DURA-source-cleanTangent-s2026092701-r01'
CONTROL='20261031-phase1-DURA-repeatViews-E30-s2026092701-r01'
CONTROL_COMMIT='8cf8e22c9ffb54d271b22e621401de8e34c69068'
WEIGHT=.0033130025035573727
METRIC='F_clean_tangent_error'


def verify_qualification(result):
    expected=dict(status='ANALYZED_SOURCE_ONLY',F_teacher=f_guided.teacher_reference(),source_checkpoint=CONTROL,
        source_commit=CONTROL_COMMIT,F_unchanged=True,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,new_G_inputs=0)
    if any(result.get(k)!=v for k,v in expected.items()) or result.get('decision')!={
        'eligible':True,'views':[{'view':'1','passed':True},{'view':'20','passed':True}],
        'fixed_source_weight':WEIGHT,'automatic_training':False}:
        raise ValueError('Complete fixed source qualification and weight required')


def objective(student,received,clean,action,variance,probes,jacobian):
    _,terms,restored=source_objective(student,received,clean,action,variance,probes,return_restored=True)
    guide=tangent_loss(restored,clean,jacobian)
    loss=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']+WEIGHT*guide
    return loss,dict(terms,**{METRIC:guide}),restored


def smoke(args,p):
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    verify_qualification(base.read_json(args.root/'runs'/PROBE/'results.json'))
    clean,roles=base.source_training(p);f_guided.source_ids(p,roles);take=np.flatnonzero(roles=='L_s')[:2]
    folder=p['cache']/'epochs/01'
    values={k:base.tensor(np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu') for k in ['received','kernel','coefficients','variance']}
    x=base.tensor(clean[take],'cpu');student=base.scratch_model(2026092701)
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    before={k:v.clone() for k,v in reader.state_dict().items()};basis=head_feature_basis(reader).float()
    _,jac=clean_jacobian(reader,x,basis)
    generator=torch.Generator().manual_seed(2026103004)
    probes=[torch.complex(torch.randn(2,256,generator=generator),torch.randn(2,256,generator=generator)) for _ in range(2)]
    loss,terms,restored=objective(student,values['received'],x,base.supervised_action(values['kernel'],values['coefficients']),values['variance'],probes,jac)
    derivative=torch.autograd.grad(terms[METRIC].mean(),restored,retain_graph=True)[0]
    if not torch.isfinite(derivative).all() or not derivative.abs().sum()>0:raise ValueError('Actual clean-J metric IQ gradient required')
    loss.mean().backward();grads=[p.grad for p in student.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum()>0 for g in grads):raise ValueError('Actual G gradients required')
    with torch.no_grad():expected=student(values['received']).clone()
    path=p['output']/'smoke-scratch_joint.pt';f_guided.save_model(path,student,'scratch_joint',0,0,args.commit)
    loaded=base.load_model(path,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    with torch.no_grad():
        if not torch.equal(loaded(values['received']),expected):raise ValueError('Strict scratch G reload changed IQ')
        score=reader(expected)
    if score.shape!=(2,6) or not torch.isfinite(score).all():raise ValueError('Original F/defaulthead noquery smoke failed')
    if any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()) or any(p.grad is not None for p in reader.parameters()):raise ValueError('Smoke changed frozen F')
    base.write_json(p['output']/'smoke.json',dict(status='PASS',real_Ls=2,training_regime='source_F_clean_tangent',
        F_inputs=4,F_J_VJP_inputs=10,G_training_forwards=1,F_updates=0,F_unchanged=True,TX_labels_read=0,target_reads=0,
        nonzero_clean_tangent_to_IQ_gradient=True,head_fit_updates=0,initialization_checkpoint_loads=0,strict_reload=True))
    print(json.dumps(dict(event='REAL_SOURCE_CLEAN_TANGENT_G_F_SMOKE_PASS')),flush=True)


def train(args,p):
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    started_total=time.perf_counter();config=base.configuration();row=p['output']/args.arm;row.mkdir()
    base.write_json(row/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],commit=args.commit))
    verify_qualification(base.read_json(args.root/'runs'/PROBE/'results.json'))
    clean,roles=base.source_training(p);ids=f_guided.source_ids(p,roles)
    reader=load_reader(p['reader'],p['source_cache'],'cuda:0').eval().requires_grad_(False)
    before_F={k:v.detach().cpu().clone() for k,v in reader.state_dict().items()};basis=head_feature_basis(reader).to('cuda:0',torch.float32)
    cache=row/'source_clean_J_cache';cache.mkdir();cache_started=time.perf_counter()
    targets=np.lib.format.open_memmap(cache/'jacobian.npy',mode='w+',dtype=np.float32,shape=(63000,5,2,256))
    for offset in range(0,63000,256):
        end=min(offset+256,63000);_,jac=clean_jacobian(reader,base.tensor(clean[offset:end],'cuda:0'),basis)
        if jac.shape!=(end-offset,5,2,256) or not torch.isfinite(jac).all():raise ValueError('Complete finite source clean-J cache required')
        targets[offset:end]=base.numpy_values(jac)
        if offset%8192==0:print(json.dumps(dict(event='SOURCE_CLEAN_J_CACHE',count=end,total=63000)),flush=True)
    targets.flush();np.save(cache/'physical_ids.npy',ids,allow_pickle=False);np.save(cache/'basis.npy',base.numpy_values(basis),allow_pickle=False)
    if any(not torch.equal(v.detach().cpu(),before_F[k]) for k,v in reader.state_dict().items()) or any(p.grad is not None for p in reader.parameters()):raise ValueError('Cache construction changed F')
    cache_seconds=time.perf_counter()-cache_started
    base.write_json(cache/'complete.json',dict(status='SOURCE_F_CLEAN_J_COMPLETE',count=63000,shape=[63000,5,2,256],numeric_bytes=63000*5*512*4,
        F_teacher=f_guided.teacher_reference(),F_inputs=63000,F_J_VJP_inputs=315000,F_updates=0,F_unchanged=True,
        TX_labels_read=0,V_reads=0,target_reads=0,seconds=cache_seconds))
    del reader,before_F,basis,jac;gc.collect();torch.cuda.empty_cache()
    # Source-F construction must not alter the matched G or CUDA probe stream.
    torch.manual_seed(2026092701);torch.cuda.manual_seed_all(2026092701)
    student=base.scratch_model(2026092701).to('cuda:0');before={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
    optimizer=torch.optim.AdamW(student.parameters(),lr=.001,weight_decay=1e-4)
    base.write_json(row/'resolved_config.json',config);base.write_json(row/'initialization.json',student.initialization_report)
    preparation_seconds=time.perf_counter()-started_total;rng=np.random.default_rng(2026103004);updates=seen_l=seen_u=0
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x',encoding='utf-8') as stream:
        for epoch in range(1,31):
            view=(epoch-1)%20+1;folder=p['cache']/'epochs'/f'{view:02d}'
            arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ['received','kernel','coefficients','variance']}
            shapes=dict(received=(63000,2,256),kernel=(63000,2,64),coefficients=(63000,4,256),variance=(63000,))
            if any(v.shape!=shapes[k] or v.dtype!=np.float32 for k,v in arrays.items()):raise ValueError('Original complete source cache required')
            totals=dict.fromkeys(['loss','iq_nmse','prior_nmse','spectral_error','action_probe_nmse','log_variance_mse',METRIC],0.)
            clipped=0;order=rng.permutation(63000)
            for offset in range(0,63000,256):
                take=order[offset:offset+256];x=base.tensor(clean[take],'cuda:0');jac=base.tensor(targets[take],'cuda:0')
                values={k:base.tensor(v[take],'cuda:0') for k,v in arrays.items()};action=base.supervised_action(values['kernel'],values['coefficients'])
                probes=[torch.complex(torch.randn(len(take),256,device='cuda:0'),torch.randn(len(take),256,device='cuda:0')) for _ in range(2)]
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=learning_rate(updates)
                loss,terms,_=objective(student,values['received'],x,action,values['variance'],probes,jac)
                if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite source tangent objective')
                loss.mean().backward();norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True)
                clipped+=int(norm>1);optimizer.step();updates+=1;seen_l+=int((roles[take]=='L_s').sum());seen_u+=int((roles[take]=='U_s').sum())
                for key,value in dict(loss=loss,**terms).items():totals[key]+=float(value.detach().sum())
            record=dict(epoch=epoch,source_cache_view=view,updates=updates,losses={k:v/63000 for k,v in totals.items()},L_exposures=seen_l,U_exposures=seen_u,
                clip_fraction=clipped/247,seconds=time.perf_counter()-started,F_inputs=0,F_clean_J_cache_inputs=63000,F_J_reference_inputs=seen_l+seen_u,
                F_updates=0,TX_labels_read=0,initialization_checkpoint_loads=0)
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();print(json.dumps(dict(event='SOURCE_CLEAN_TANGENT_EPOCH_COMPLETE',**record)),flush=True)
    if (updates,seen_l,seen_u)!=(7410,189000,1701000):raise ValueError('Incomplete single E30 budget')
    changed=sum(not torch.equal(v.detach().cpu(),before[k]) for k,v in student.state_dict().items())
    if not changed:raise ValueError('Scratch G did not train')
    f_guided.save_model(row/'final.pt',student,args.arm,30,7410,args.commit)
    base.write_json(row/'training_complete.json',dict(status='SOURCE_TRAINED',arm=args.arm,epoch=30,updates=7410,L_exposures=seen_l,U_exposures=seen_u,
        training_TX_labels_read=0,training_F_access=True,V_reads=0,target_reads=0,F_updates=0,F_unchanged=True,F_teacher=f_guided.teacher_reference(),
        F_clean_teacher_inputs=63000,F_clean_J_VJP_inputs=315000,F_restored_training_inputs=0,F_J_reference_inputs=1890000,
        initialization_checkpoint_loads=0,G_ancestor_epochs=0,cumulative_G_epochs=30,F_reused_teacher_epochs=50,head_fit_updates=0,
        parameters=266951,changed_state_tensors=changed,seconds=time.perf_counter()-started,source_preparation_seconds=preparation_seconds,
        source_J_cache_seconds=cache_seconds,source_J_cache_numeric_bytes=645120000,total_G_training_pipeline_seconds=time.perf_counter()-started_total,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),training_regime='authorized_source_F_clean_tangent_no_direct_TX'))


def configure():
    f_guided.configure();previous_configuration=base.configuration;previous_command=base.command
    def configuration():
        value=previous_configuration();value.update(run=RUN,recipe='scratch_E30_cached_clean_head_visible_tangent',epochs=30,updates=7410,new_cumulative_epochs=30,
            L_exposures=189000,U_exposures=1701000,corruption_epochs=20,source_corruption_schedule='01..20then01..10',
            source_freeze='fixedfinalE30/7410,noearlystop/source-target-ranking',F_guidance_metric=METRIC,F_feature_weight=WEIGHT,
            F_clean_J_cache_count=63000,F_clean_J_shape=[63000,5,2,256],F_clean_J_numeric_bytes=645120000,F_clean_J_VJP_inputs=315000,
            F_restored_training_inputs=0,F_training_forward_after_cache=0,F_J_reference_inputs=1890000,F_clean_cache_count=0,
            F_tangent_reference='sourceclean local normalizedhead-visible featureJacobian,rank5 original head basis,not fitted',
            objective='finalIQNMSE+.1internalpriorIQNMSE+.05spectral+.05actionprobe+fixedalpha sourceclean-J tangent quadratic',
            research_qualification=PROBE,fixed_source_weight=WEIGHT,matched_control=CONTROL,matched_control_commit=CONTROL_COMMIT,
            CUDA_probe_seed=2026092701,numpy_shuffle_seed=2026103004,allowed_GPUs=[4,6,7],max_total_visible_GPU_processes=3,
            host_numpy_conversion='DLPack_owned_FP32',source_F_informed_method_research=True,
            training_regime='source_F_guided_cached_clean_derivatives',F_reused_teacher_epochs=50,
            source_supervision='PairedsourceIQ/action plus fixedoriginalF cleanJacobian;no directTX/V/target,notF-free')
        return value
    def command(args,mode,arm=None,role=None):return [MODULE if item==f_guided.MODULE else item for item in previous_command(args,mode,arm,role)]
    def load_model(path,device,*,arm,commit,epoch=None):return f_guided.load_model(path,device,arm=arm,commit=commit,epoch=30 if epoch is None else epoch)
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.learning_rate=learning_rate
    base.smoke=smoke;base.train=train;base.load_model=load_model


if __name__=='__main__':configure();base.main()
