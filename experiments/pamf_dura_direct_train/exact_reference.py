"""Scratch joint E30 with exact online frozen-F source10 reference guidance."""
import json
import os
import sys
import time
import numpy as np
import torch
from . import soft_reference as parent,f_guided
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_dura_efficiency.model import direct_objective as source_objective
from experiments.pamf_dura_efficiency.exact_reference import exact_guide
from experiments.pamf_dura_efficiency.soft_reference_tangent import ALPHA
from experiments.pamf_dura_efficiency.soft_rx_reference import make_reference,BETA
from experiments.pamf_dura_efficiency.soft_rx_reference_probe import RUN as GEOMETRY_RUN
from experiments.pamf_dura_efficiency.nonlinear_reference_probe import source as source_reference,RUN as NONLINEAR_RUN,REFERENCE as BASIS_RUN
from experiments.pamf_dura_efficiency.nonlinear_profile import RUN as COST_RUN

base=parent.base
RUN='20261031-phase1-DURA-exactReference-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.exact_reference'
METRIC='F_exact_reference_error'
GUIDE='F_exact_reference_guide'
EXTRA=('F_balance_weight','F_balance_limited','F_raw_pressure','F_balanced_pressure','F_pressure_defined_fraction','F_guide_value_exceeds_IQ')


def validate_qualifications(nonlinear,cost):
    for value in [nonlinear,cost]:
        if (not value['decision']['eligible'] or not value['F_unchanged'] or value['F_teacher']!=f_guided.teacher_reference()
            or any(value[k]!=0 for k in ['optimizer_updates','TX_labels_read','V_reads','target_reads'])):
            raise ValueError('Fixed legal source nonlinear/cost qualifications required')
    if (not nonlinear['G_state_unchanged'] or nonlinear['iterations']!=8 or nonlinear['half_step']!=.5 or nonlinear['dimension']!=10
        or not cost['G_unchanged'] or cost['source_views']!=['1','20'] or cost['batch']!=256 or cost['timed_per_case_view']!=12):
        raise ValueError('Complete fixed source qualification matrix required')


def validate_target_metadata(value):
    expected=dict(status='SOURCE_EXACT_REFERENCE_TARGETS_COMPLETE',count=63000,shape=[63000,10],numeric_bytes=2520000,
        source_contract='original7/63/30',F_teacher=f_guided.teacher_reference(),reference_run=GEOMETRY_RUN,basis_run=BASIS_RUN,
        nonlinear_qualification_run=NONLINEAR_RUN,cost_qualification_run=COST_RUN,beta=BETA,alpha=ALPHA,
        new_F_inputs=0,new_F_VJP_inputs=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0)
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('This run fixed source10 target cache required')


def build_targets(args,p,roles):
    started=time.perf_counter();ids=f_guided.source_ids(p,roles)
    nonlinear=base.read_json(args.root/'runs'/NONLINEAR_RUN/'results.json');cost=base.read_json(args.root/'runs'/COST_RUN/'results.json')
    validate_qualifications(nonlinear,cost);a,_,_=source_reference(args,p)
    cache=args.root/'runs'/f_guided.RUN/'scratch_joint/source_feature_cache';m=base.read_json(cache/'complete.json')
    expected=dict(status='SOURCE_F_FEATURES_COMPLETE',count=63000,dimension=160,TX_labels_read=0,V_reads=0,target_reads=0,F_updates=0,F_teacher=f_guided.teacher_reference())
    if any(m.get(k)!=v for k,v in expected.items()) or not np.array_equal(np.load(cache/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Original source F160 cache/order required')
    features=np.load(cache/'features.npy',mmap_mode='r',allow_pickle=False)
    if features.shape!=(63000,160) or features.dtype!=np.float32 or not np.isfinite(features).all():raise ValueError('Complete finite source F160 features required')
    unit=np.asarray(features,dtype=np.float64);unit/=np.maximum(np.linalg.norm(unit,axis=1,keepdims=True),1e-4)
    with np.load(args.root/'runs'/GEOMETRY_RUN/'reference_geometry.npz',allow_pickle=False) as f:
        probability=f['full_probabilities'];centroids=f['leave_RX_centroids']
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False)
    reference=make_reference(unit,probability,rx,centroids);basis=a['basis'];targets=(4*reference@basis).astype(np.float32)
    if not np.allclose(targets[a['source_positions']],a['target_phi'],atol=3e-6,rtol=2e-6):raise ValueError('Production targets differ from qualified source512')
    if targets.shape!=(63000,10) or not np.isfinite(targets).all():raise ValueError('Finite complete source10 targets required')
    folder=p['output']/'source_exact_reference_cache';folder.mkdir()
    np.save(folder/'targets.npy',targets,allow_pickle=False);np.save(folder/'basis.npy',basis,allow_pickle=False);np.save(folder/'physical_ids.npy',ids,allow_pickle=False)
    metadata=dict(status='SOURCE_EXACT_REFERENCE_TARGETS_COMPLETE',count=63000,shape=[63000,10],numeric_bytes=targets.nbytes,
        source_contract='original7/63/30',F_teacher=f_guided.teacher_reference(),reference_run=GEOMETRY_RUN,basis_run=BASIS_RUN,
        nonlinear_qualification_run=NONLINEAR_RUN,cost_qualification_run=COST_RUN,beta=BETA,alpha=ALPHA,
        new_F_inputs=0,new_F_VJP_inputs=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,
        seconds=time.perf_counter()-started,source_feature_cache_metadata=m,source_nonlinear_qualification_seconds=nonlinear['seconds'],
        source_cost_qualification_F_inputs=cost['F_inputs'],source_cost_qualification_G_inputs=cost['G_inputs'])
    validate_target_metadata(metadata);base.write_json(folder/'complete.json',metadata)
    print(json.dumps(dict(event='SOURCE_EXACT_REFERENCE_TARGETS_COMPLETE',numeric_bytes=targets.nbytes,seconds=metadata['seconds'])),flush=True)


def load_targets(p,roles):
    started=time.perf_counter();folder=p['output']/'source_exact_reference_cache';metadata=base.read_json(folder/'complete.json');validate_target_metadata(metadata)
    if not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),f_guided.source_ids(p,roles)):raise ValueError('Source target physical order differs')
    targets=np.load(folder/'targets.npy',mmap_mode='r',allow_pickle=False);basis=np.load(folder/'basis.npy',allow_pickle=False)
    if (targets.shape!=(63000,10) or targets.dtype!=np.float32 or basis.shape!=(160,10) or basis.dtype!=np.float32
        or not np.allclose(basis.T@basis,np.eye(10),atol=3e-6,rtol=3e-6)):raise ValueError('Complete original source10 target/basis required')
    return targets,basis,dict(metadata=metadata,seconds=time.perf_counter()-started)


def load_teacher_preserving_rng(p,device):
    device=torch.device(device);devices=[device.index if device.index is not None else 0] if device.type=='cuda' else []
    beforeCPU=torch.get_rng_state().clone();beforeCUDA=[torch.cuda.get_rng_state(index).clone() for index in devices]
    with torch.random.fork_rng(devices=devices):reader=load_reader(p['reader'],p['source_cache'],str(device)).eval().requires_grad_(False)
    preserved=torch.equal(torch.get_rng_state(),beforeCPU) and all(torch.equal(torch.cuda.get_rng_state(index),state) for index,state in zip(devices,beforeCUDA))
    if not preserved:raise ValueError('F loading changed matched G/probe random state')
    return reader,dict(F_load_RNG_restored=True,CPU_rng_restored=True,CUDA_rng_restored=True if device.type=='cuda' else None,F_checkpoint_loads=1)


def objective(student,reader,received,clean,action,variance,probes,target,basis):
    _,terms,restored=source_objective(student,received,clean,action,variance,probes,return_restored=True)
    # The additional teacher path must not advance the matched G/probe stream.
    with torch.random.fork_rng(devices=[restored.device.index] if restored.is_cuda else []):
        guide,info=exact_guide(reader,restored,clean,target,basis)
    loss=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']+guide
    return loss,dict(terms,**{METRIC:info['feature_error'],GUIDE:guide,'F_balance_weight':info['weight'],
        'F_balance_limited':(info['weight']<1).float(),'F_raw_pressure':info['raw_pressure'],'F_balanced_pressure':info['bounded_pressure'],
        'F_pressure_defined_fraction':info['pressure_defined'].float(),'F_guide_value_exceeds_IQ':(guide.detach()>terms['iq_nmse'].detach()).float()}),restored


def smoke(args,p):
    clean,roles=base.source_training(p);build_targets(args,p,roles);targets,basis_np,_=load_targets(p,roles)
    take=np.flatnonzero(roles=='L_s')[:2];folder=p['cache']/'epochs/01'
    values={key:base.tensor(np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu') for key in ['received','kernel','coefficients','variance']}
    x=base.tensor(clean[take],'cpu');student=base.scratch_model(2026092701);reader,rng_info=load_teacher_preserving_rng(p,'cpu')
    before={key:value.clone() for key,value in reader.state_dict().items()};basis=base.tensor(basis_np,'cpu');target=base.tensor(targets[take],'cpu')
    rng=torch.Generator().manual_seed(2026103004);probes=[torch.complex(torch.randn(2,256,generator=rng),torch.randn(2,256,generator=rng)) for _ in range(2)]
    loss,terms,restored=objective(student,reader,values['received'],x,base.supervised_action(values['kernel'],values['coefficients']),values['variance'],probes,target,basis)
    actual=torch.autograd.grad(terms[GUIDE].sum(),restored,retain_graph=True)[0]
    IQnorm=2*(restored.detach()-x).flatten(1).norm(dim=1)/(512*x.square().mean((1,2)).clamp_min(1e-4))
    if not torch.isfinite(actual).all() or not actual.abs().sum()>0 or (actual.flatten(1).norm(dim=1)>IQnorm*(1+2e-5)).any():raise ValueError('Actual exact G guidance cap failed')
    loss.mean().backward()
    for prefix in ['frozen.coarse.','frozen.estimator.','prior.']:
        grads=[v.grad for k,v in student.named_parameters() if k.startswith(prefix) and v.grad is not None]
        if not grads or not all(torch.isfinite(v).all() for v in grads) or not any(v.abs().sum()>0 for v in grads):raise ValueError('Real joint G gradient missing: '+prefix)
    with torch.no_grad():expected=student(values['received']).clone()
    checkpoint=p['output']/'smoke-scratch_joint.pt';f_guided.save_model(checkpoint,student,'scratch_joint',0,0,args.commit)
    loaded=base.load_model(checkpoint,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    with torch.no_grad():
        if not torch.equal(loaded(values['received']),expected):raise ValueError('Strict exact-guided G reload changed IQ')
        scores=reader(expected)
    if scores.shape!=(2,6) or not torch.isfinite(scores).all() or any(v.grad is not None for v in reader.parameters()) or any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()):raise ValueError('Original F/defaulthead/frozen-state smoke failed')
    base.write_json(p['output']/'smoke.json',dict(status='PASS',real_Ls=2,F_inputs=4,F_input_VJP_inputs=2,G_inputs=6,F_unchanged=True,F_updates=0,
        exact_guide_IQ_gradient_bound=True,joint_coarse_action_prior_gradients=True,strict_reload=True,initialization_checkpoint_loads=0,
        TX_labels_read=0,V_reads=0,target_reads=0,teacher_load=rng_info))
    print(json.dumps(dict(event='REAL_EXACT_REFERENCE_E30_SMOKE_PASS')),flush=True)


def train(args,p):
    started_total=time.perf_counter();row=p['output']/args.arm;row.mkdir()
    base.write_json(row/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],commit=args.commit))
    clean,roles=base.source_training(p);targets,basis_np,reference=load_targets(p,roles);base.write_json(row/'source_reference.json',reference)
    torch.manual_seed(2026092701);torch.cuda.manual_seed_all(2026092701);student=base.scratch_model(2026092701).to('cuda:0')
    beforeG={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
    reader,rng_info=load_teacher_preserving_rng(p,'cuda:0');beforeF={k:v.detach().cpu().clone() for k,v in reader.state_dict().items()}
    base.write_json(row/'teacher_load.json',rng_info);basis=base.tensor(basis_np,'cuda:0');optimizer=torch.optim.AdamW(student.parameters(),lr=.001,weight_decay=1e-4)
    base.write_json(row/'resolved_config.json',base.configuration());base.write_json(row/'initialization.json',student.initialization_report)
    preparation=time.perf_counter()-started_total;started=time.perf_counter();rng=np.random.default_rng(2026103004);updates=seen_l=seen_u=0
    torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x',encoding='utf-8') as stream:
        for epoch in range(1,31):
            view=(epoch-1)%20+1;folder=p['cache']/'epochs'/f'{view:02d}'
            arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ['received','kernel','coefficients','variance']}
            shapes=dict(received=(63000,2,256),kernel=(63000,2,64),coefficients=(63000,4,256),variance=(63000,))
            if any(v.shape!=shapes[k] or v.dtype!=np.float32 for k,v in arrays.items()):raise ValueError('Original full source caches required')
            totals=dict.fromkeys(['loss','iq_nmse','prior_nmse','spectral_error','action_probe_nmse','log_variance_mse',METRIC,GUIDE,*EXTRA],0.)
            clipped=0;maximum_pressure=0.;order=rng.permutation(63000)
            for start in range(0,63000,256):
                take=order[start:start+256];x=base.tensor(clean[take],'cuda:0');target=base.tensor(targets[take],'cuda:0')
                values={key:base.tensor(v[take],'cuda:0') for key,v in arrays.items()};action=base.supervised_action(values['kernel'],values['coefficients'])
                probes=[torch.complex(torch.randn(len(take),256,device='cuda:0'),torch.randn(len(take),256,device='cuda:0')) for _ in range(2)]
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=base.learning_rate(updates)
                loss,terms,_=objective(student,reader,values['received'],x,action,values['variance'],probes,target,basis)
                if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite exact source objective')
                maximum_pressure=max(maximum_pressure,float(terms['F_balanced_pressure'].max()))
                if maximum_pressure>1+2e-5:raise FloatingPointError('Exact guide IQ gradient cap violated')
                loss.mean().backward();norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True);clipped+=int(norm>1)
                optimizer.step();updates+=1;seen_l+=int((roles[take]=='L_s').sum());seen_u+=int((roles[take]=='U_s').sum())
                for key,value in dict(loss=loss,**terms).items():totals[key]+=float(value.detach().sum())
            record=dict(epoch=epoch,source_cache_view=view,updates=updates,losses={key:v/63000 for key,v in totals.items()},L_exposures=seen_l,U_exposures=seen_u,
                clip_fraction=clipped/247,maximum_balanced_IQ_pressure=maximum_pressure,seconds=time.perf_counter()-started,
                F_inputs=seen_l+seen_u,F_input_VJP_inputs=seen_l+seen_u,F_target_reference_inputs=seen_l+seen_u,new_F_cache_inputs=0,F_J_reference_inputs=0,
                F_updates=0,TX_labels_read=0,initialization_checkpoint_loads=0)
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();print(json.dumps(dict(event='SOURCE_EXACT_REFERENCE_EPOCH_COMPLETE',**record)),flush=True)
    if (updates,seen_l,seen_u)!=(7410,189000,1701000):raise ValueError('Incomplete single E30 budget')
    if any(v.grad is not None for v in reader.parameters()) or any(not torch.equal(v.detach().cpu(),beforeF[k]) for k,v in reader.state_dict().items()):raise ValueError('Training changed frozen F')
    changed=sum(not torch.equal(v.detach().cpu(),beforeG[k]) for k,v in student.state_dict().items())
    if not changed:raise ValueError('Scratch G did not train')
    f_guided.save_model(row/'final.pt',student,args.arm,30,7410,args.commit)
    base.write_json(row/'training_complete.json',dict(status='SOURCE_TRAINED',arm=args.arm,epoch=30,updates=7410,L_exposures=seen_l,U_exposures=seen_u,
        training_TX_labels_read=0,training_F_access=True,V_reads=0,target_reads=0,F_updates=0,F_unchanged=True,F_teacher=f_guided.teacher_reference(),
        F_loaded_in_training=True,F_clean_teacher_inputs=0,F_clean_J_VJP_inputs=0,F_restored_training_inputs=1890000,F_input_VJP_inputs=1890000,F_J_reference_inputs=0,
        F_target_reference_inputs=1890000,new_F_cache_inputs=0,source_reference=reference,teacher_load=rng_info,initialization_checkpoint_loads=0,G_ancestor_epochs=0,
        cumulative_G_epochs=30,F_reused_teacher_epochs=50,head_fit_updates=0,parameters=266951,changed_state_tensors=changed,seconds=time.perf_counter()-started,
        source_preparation_seconds=preparation,total_G_training_pipeline_seconds=time.perf_counter()-started_total,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        training_regime='source_F_guided_exact_nonlinear_reference_one_input_reverse'))


def configure():
    parent.configure();previous_configuration=base.configuration;previous_command=base.command
    def configuration():
        value=previous_configuration()
        for key in ['F_clean_cache_count','F_clean_J_cache_count','F_clean_J_shape','F_clean_J_numeric_bytes','reused_F_J_cache_run','historical_F_J_cache_inputs',
            'historical_F_J_cache_VJP_inputs','F_tangent_reference','uncapped_tangent_control','uncapped_tangent_commit','source_reference_offset_shape',
            'source_reference_offset_numeric_bytes','source_reference_offset_inputs','source_IQ_calibration_run']:
            value.pop(key,None)
        value.update(run=RUN,recipe='scratch_E30_exact_nonlinear_source10_reference',objective='noVarfourphysicalterms+already-alpha-weighted exact frozen-F step-local surrogate',
            F_guidance_metric=METRIC,F_guidance_weighted_metric=GUIDE,F_guidance_dimension=10,F_loaded_in_training=True,F_training_forward_after_cache=1890000,
            F_restored_training_inputs=1890000,F_input_VJP_inputs=1890000,F_J_reference_inputs=0,F_target_reference_inputs=1890000,training_reads_J_cache=False,
            F_load_RNG_preserved=True,F_forward_RNG_isolated=True,source_teacher_target_shape=[63000,10],source_teacher_target_numeric_bytes=2520000,
            source_nonlinear_qualification=NONLINEAR_RUN,source_cost_qualification=COST_RUN,research_qualification=COST_RUN,
            source_qualification_semantics='Source nonlinear reachability andpairedGPUcost pass;notGlearningor targetefficacy guarantee',
            matched_control=parent.RUN,matched_control_commit='b6136c53106c9e8e6273a606262d5152a4e08d23',
            F_guidance_value_semantics='ExactFinputgradient,detachedstepweight/injectedgradient;notguidevaluebound/Gparameterbound/globalfixedenergyderivative',
            training_regime='source_F_guided_exact_nonlinear_reference_one_input_reverse',
            source_supervision='SourcepairedIQ/action plusfixedoriginalFsource10softRXtargets/exactrestoredfeatures;no directTX/V/target,notF-free')
        return value
    def command(args,mode,arm=None,role=None):return [MODULE if item==parent.MODULE else item for item in previous_command(args,mode,arm,role)]
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.smoke=smoke;base.train=train


if __name__=='__main__':configure();base.main()
