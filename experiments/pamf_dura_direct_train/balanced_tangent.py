"""Scratch joint E30: direction-preserving source metric balance, cached F only."""
import json
import os
import sys
import time
import numpy as np
import torch
from . import clean_tangent as parent
from . import f_guided
from experiments.pamf_dura_efficiency.model import direct_objective as source_objective
from experiments.pamf_dura_efficiency.tangent_balance import balanced_tangent_loss
from experiments.pamf_dura_diagnostics.runner import source_indices

base=parent.base
RUN='20261031-phase1-DURA-balancedTangent-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.balanced_tangent'
CACHE_RUN=parent.RUN
PRESSURE_RUN='20261031-diagnostic-DURA-source-boundedTangent-s2026092701-r01'
WEIGHT=parent.WEIGHT
METRIC=parent.METRIC
EXTRA=('F_tangent_raw_error','F_balance_weight','F_balance_limited','F_raw_pressure','F_balanced_pressure')


def verify_cache_metadata(value):
    expected=dict(status='SOURCE_F_CLEAN_J_COMPLETE',count=63000,shape=[63000,5,2,256],numeric_bytes=645120000,
        F_teacher=f_guided.teacher_reference(),F_inputs=63000,F_J_VJP_inputs=315000,F_updates=0,F_unchanged=True,TX_labels_read=0,V_reads=0,target_reads=0)
    if any(value.get(key)!=item for key,item in expected.items()):raise ValueError('Original source-only raw F derivative cache required')


def metric_cache(root,p,roles):
    started=time.perf_counter();ids=f_guided.source_ids(p,roles)
    folder=root/'runs'/CACHE_RUN/'scratch_joint/source_clean_J_cache';metadata=base.read_json(folder/'complete.json');verify_cache_metadata(metadata)
    if not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Original source physical derivative order required')
    values=np.load(folder/'jacobian.npy',mmap_mode='r',allow_pickle=False)
    if values.shape!=(63000,5,2,256) or values.dtype!=np.float32:raise ValueError('Complete original FP32 source derivatives required')
    return values,dict(source_cache_run=CACHE_RUN,cache_kind='raw_source_clean_F_J',source_metadata=metadata,
        cache_reference_count=63000,numeric_bytes=645120000,new_F_inputs=0,new_F_VJP_inputs=0,new_cache_bytes=0,
        F_teacher=f_guided.teacher_reference(),historical_cache_F_inputs=63000,historical_cache_F_VJP_inputs=315000,
        historical_cache_seconds=metadata['seconds'],seconds=time.perf_counter()-started)


def source_balance_check(root,p,roles):
    folder=root/'runs'/PRESSURE_RUN;meta=base.read_json(folder/'results.json');ids=f_guided.source_ids(p,roles);indices=source_indices(roles)
    if (meta['F_teacher']!=f_guided.teacher_reference() or not meta['F_unchanged'] or meta['source_derivative_cache_run']!=CACHE_RUN
        or meta['source_error_checkpoint']!=parent.CONTROL or meta['source_error_commit']!=parent.CONTROL_COMMIT or meta['weight']!=WEIGHT
        or any(meta[key]!=0 for key in ['new_G_inputs','optimizer_updates','TX_labels_read','V_reads','target_reads'])):raise ValueError('Original source pressure geometry required')
    rows={}
    for view in ['1','20']:
        with np.load(folder/f'source-{view}.npz',allow_pickle=False) as a:v={k:a[k] for k in a.files}
        for key,expected in dict(source_positions=indices,physical_ids=ids[indices],roles=roles[indices]).items():
            if not np.array_equal(v[key],expected):raise ValueError('Fixed source pressure positions required')
        if not v['raw_defined'].all():raise ValueError('All fixed512 source directions must be defined')
        pressure=v['raw_pressure'].astype(np.float64);scale=np.minimum(1,1/np.maximum(pressure,1e-12))
        raw=v['raw_dot'].astype(np.float64)/(v['raw_norm'].astype(np.float64)*v['exact_norm'].astype(np.float64))
        balanced=(v['raw_dot']*scale)/(v['raw_norm']*scale*v['exact_norm'])
        rx={str(r):float(balanced[v['rx']==r].mean()) for r in np.unique(v['rx'])}
        if (len(raw)!=512 or set(rx)!=set(map(str,[1,3,4,6,8])) or np.quantile(pressure,.95)<=1 or raw.mean()<.5
            or (balanced>=0).sum()<384 or any(value<0 for value in rx.values()) or not np.allclose(raw,balanced,atol=2e-6,rtol=2e-6)
            or (pressure*scale).max()>1+2e-5):raise ValueError('Fixed source balance/direction falsification failed')
        rows[view]=dict(count=512,raw_mean_cosine=float(raw.mean()),balanced_mean_cosine=float(balanced.mean()),
            limited_fraction=float((scale<1).mean()),maximum_balanced_pressure=float((pressure*scale).max()),rx_mean_cosine=rx)
    return dict(status='SOURCE_BALANCE_CHECK_PASS',rows=rows,source_geometry_run=PRESSURE_RUN,alpha=WEIGHT,
        new_G_F_inputs=0,TX_labels_read=0,V_reads=0,target_reads=0,
        scope='Reused source geometry/step-local scalar construction;not targetaccuracy or independentheldout qualification;static spectral variant remains failed')


def objective(student,received,clean,action,variance,probes,jacobian):
    _,terms,restored=source_objective(student,received,clean,action,variance,probes,return_restored=True)
    guide,info=balanced_tangent_loss(restored,clean,jacobian,WEIGHT)
    loss=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']+WEIGHT*guide
    with torch.no_grad():raw=(jacobian.detach()*(restored.detach()-clean.detach())[:,None]).sum((2,3)).square().sum(1)
    return loss,dict(terms,**{METRIC:guide,'F_tangent_raw_error':raw,'F_balance_weight':info['weight'],
        'F_balance_limited':info['limited'].float(),'F_raw_pressure':info['raw_pressure'],'F_balanced_pressure':info['bounded_pressure']}),restored


def smoke(args,p):
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    clean,roles=base.source_training(p);cache,_=metric_cache(args.root,p,roles)
    base.write_json(p['output']/'source_balance_check.json',source_balance_check(args.root,p,roles))
    take=np.flatnonzero(roles=='L_s')[:2];folder=p['cache']/'epochs/01'
    values={key:base.tensor(np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu') for key in ['received','kernel','coefficients','variance']}
    x=base.tensor(clean[take],'cpu');jac=base.tensor(cache[take],'cpu');student=base.scratch_model(2026092701)
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False);before={k:v.clone() for k,v in reader.state_dict().items()}
    g=torch.Generator().manual_seed(2026103004);probes=[torch.complex(torch.randn(2,256,generator=g),torch.randn(2,256,generator=g)) for _ in range(2)]
    loss,terms,restored=objective(student,values['received'],x,base.supervised_action(values['kernel'],values['coefficients']),values['variance'],probes,jac)
    gradient=torch.autograd.grad((WEIGHT*terms[METRIC]).sum(),restored,retain_graph=True)[0]
    IQnorm=2*(restored.detach()-x).flatten(1).norm(dim=1)/(512*x.square().mean((1,2)).clamp_min(1e-4))
    if not torch.isfinite(gradient).all() or not gradient.abs().sum()>0 or (gradient.flatten(1).norm(dim=1)>IQnorm*(1+2e-5)).any():raise ValueError('Actual G-IQ balanced gradient check failed')
    loss.mean().backward()
    for prefix in ['frozen.coarse.','frozen.estimator.','prior.']:
        grads=[p.grad for name,p in student.named_parameters() if name.startswith(prefix) and p.grad is not None]
        if not grads or not all(torch.isfinite(value).all() for value in grads) or not any(value.abs().sum()>0 for value in grads):raise ValueError('Actual joint module gradient missing: '+prefix)
    with torch.no_grad():expected=student(values['received']).clone()
    path=p['output']/'smoke-scratch_joint.pt';f_guided.save_model(path,student,'scratch_joint',0,0,args.commit)
    loaded=base.load_model(path,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    with torch.no_grad():
        if not torch.equal(loaded(values['received']),expected):raise ValueError('Strict scratch G reload changed IQ')
        scores=reader(expected)
    if scores.shape!=(2,6) or not torch.isfinite(scores).all():raise ValueError('Actual original F/defaulthead smoke failed')
    if any(not torch.equal(value,before[key]) for key,value in reader.state_dict().items()) or any(p.grad is not None for p in reader.parameters()):raise ValueError('Frozen F changed')
    base.write_json(p['output']/'smoke.json',dict(status='PASS',real_Ls=2,F_inputs=2,new_F_J_VJP_inputs=0,TX_labels_read=0,target_reads=0,
        F_unchanged=True,F_updates=0,balanced_IQ_gradient_bound=True,joint_coarse_action_prior_gradients=True,strict_reload=True,initialization_checkpoint_loads=0))
    print(json.dumps(dict(event='REAL_SOURCE_BALANCED_TANGENT_SMOKE_PASS')),flush=True)


def train(args,p):
    started_total=time.perf_counter();row=p['output']/args.arm;row.mkdir()
    base.write_json(row/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],commit=args.commit))
    clean,roles=base.source_training(p);targets,reuse=metric_cache(args.root,p,roles);base.write_json(row/'cache_reuse.json',reuse)
    torch.manual_seed(2026092701);torch.cuda.manual_seed_all(2026092701);student=base.scratch_model(2026092701).to('cuda:0')
    before={k:v.detach().cpu().clone() for k,v in student.state_dict().items()};optimizer=torch.optim.AdamW(student.parameters(),lr=.001,weight_decay=1e-4)
    base.write_json(row/'resolved_config.json',base.configuration());base.write_json(row/'initialization.json',student.initialization_report)
    preparation=time.perf_counter()-started_total;started=time.perf_counter();rng=np.random.default_rng(2026103004);updates=seen_l=seen_u=0
    torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x',encoding='utf-8') as stream:
        for epoch in range(1,31):
            view=(epoch-1)%20+1;folder=p['cache']/'epochs'/f'{view:02d}'
            arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ['received','kernel','coefficients','variance']}
            shapes=dict(received=(63000,2,256),kernel=(63000,2,64),coefficients=(63000,4,256),variance=(63000,))
            if any(v.shape!=shapes[k] or v.dtype!=np.float32 for k,v in arrays.items()):raise ValueError('Original complete source cache required')
            totals=dict.fromkeys(['loss','iq_nmse','prior_nmse','spectral_error','action_probe_nmse','log_variance_mse',METRIC,*EXTRA],0.)
            clipped=0;maximum_pressure=0.;order=rng.permutation(63000)
            for offset in range(0,63000,256):
                take=order[offset:offset+256];x=base.tensor(clean[take],'cuda:0');jac=base.tensor(targets[take],'cuda:0')
                values={k:base.tensor(v[take],'cuda:0') for k,v in arrays.items()};action=base.supervised_action(values['kernel'],values['coefficients'])
                probes=[torch.complex(torch.randn(len(take),256,device='cuda:0'),torch.randn(len(take),256,device='cuda:0')) for _ in range(2)]
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=base.learning_rate(updates)
                loss,terms,_=objective(student,values['received'],x,action,values['variance'],probes,jac)
                if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite balanced source objective')
                maximum_pressure=max(maximum_pressure,float(terms['F_balanced_pressure'].max()))
                if maximum_pressure>1+2e-5:raise FloatingPointError('Source IQ-output guidance bound violated')
                loss.mean().backward();norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True);clipped+=int(norm>1)
                optimizer.step();updates+=1;seen_l+=int((roles[take]=='L_s').sum());seen_u+=int((roles[take]=='U_s').sum())
                for key,value in dict(loss=loss,**terms).items():totals[key]+=float(value.detach().sum())
            record=dict(epoch=epoch,source_cache_view=view,updates=updates,losses={k:v/63000 for k,v in totals.items()},L_exposures=seen_l,U_exposures=seen_u,
                clip_fraction=clipped/247,maximum_balanced_IQ_pressure=maximum_pressure,seconds=time.perf_counter()-started,
                F_inputs=0,new_F_cache_inputs=0,F_J_reference_inputs=seen_l+seen_u,F_updates=0,TX_labels_read=0,initialization_checkpoint_loads=0)
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();print(json.dumps(dict(event='SOURCE_BALANCED_TANGENT_EPOCH_COMPLETE',**record)),flush=True)
    if (updates,seen_l,seen_u)!=(7410,189000,1701000):raise ValueError('Incomplete single E30 budget')
    changed=sum(not torch.equal(value.detach().cpu(),before[key]) for key,value in student.state_dict().items())
    if not changed:raise ValueError('Scratch G did not train')
    f_guided.save_model(row/'final.pt',student,args.arm,30,7410,args.commit)
    base.write_json(row/'training_complete.json',dict(status='SOURCE_TRAINED',arm=args.arm,epoch=30,updates=7410,L_exposures=seen_l,U_exposures=seen_u,
        training_TX_labels_read=0,training_F_access=True,V_reads=0,target_reads=0,F_updates=0,F_teacher=f_guided.teacher_reference(),
        F_loaded_in_training=False,F_clean_teacher_inputs=0,F_clean_J_VJP_inputs=0,F_restored_training_inputs=0,F_J_reference_inputs=1890000,
        cache_reuse=reuse,initialization_checkpoint_loads=0,G_ancestor_epochs=0,cumulative_G_epochs=30,F_reused_teacher_epochs=50,head_fit_updates=0,
        parameters=266951,changed_state_tensors=changed,seconds=time.perf_counter()-started,source_preparation_seconds=preparation,
        total_G_training_pipeline_seconds=time.perf_counter()-started_total,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        training_regime='source_F_guided_direction_preserving_step_balance'))


def configure():
    parent.configure();previous_configuration=base.configuration;previous_command=base.command
    def configuration():
        value=previous_configuration();value.update(run=RUN,recipe='scratch_E30_direction_preserving_step_balance',
            objective='noVarfourphysicalterms+alpha detached-step-weight timesclean-tangent quadratic',step_weight_detached=True,
            step_weight='min(1,IQ_input_gradient_norm/raw_feature_input_gradient_norm),referencesandweightdetached',
            relative_IQ_gradient_cap=1,F_guidance_value_semantics='Step-local weighted surrogate,not total derivative of globally recomputed weights',
            new_F_cache_inputs=0,new_F_J_cache_VJP_inputs=0,F_clean_J_VJP_inputs=0,reused_F_J_cache_run=CACHE_RUN,historical_F_J_cache_inputs=63000,historical_F_J_cache_VJP_inputs=315000,
            research_qualification=PRESSURE_RUN,source_qualification_semantics='Reuse only rawsourcepressure/cosine;failedstatic spectral variant notpromoted',
            uncapped_tangent_control=CACHE_RUN,uncapped_tangent_commit='7dd799ebb294ffe8711f530872e4922f100ed498')
        return value
    def command(args,mode,arm=None,role=None):return [MODULE if item==parent.MODULE else item for item in previous_command(args,mode,arm,role)]
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.smoke=smoke;base.train=train


if __name__=='__main__':configure();base.main()
