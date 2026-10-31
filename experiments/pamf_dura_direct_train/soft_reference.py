"""Scratch joint E30 with a qualified soft cross-RX reference and cached F derivatives."""
import json
import os
import sys
import time
import numpy as np
import torch
from . import balanced_tangent as parent
from . import f_guided
from experiments.pamf_dura_efficiency.model import direct_objective as source_objective
from experiments.pamf_dura_efficiency.soft_rx_reference import make_reference,BETA
from experiments.pamf_dura_efficiency.soft_rx_reference_probe import RUN as REFERENCE_RUN
from experiments.pamf_dura_efficiency.soft_reference_IQ_probe import RUN as CALIBRATION_RUN
from experiments.pamf_dura_efficiency.soft_reference_tangent import ALPHA,prototype_tangent_loss
from experiments.pamf_dura_diagnostics.runner import source_indices

base=parent.base
RUN='20261031-phase1-DURA-softReference-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.soft_reference'
METRIC='F_soft_reference_error'
EXTRA=('F_balance_weight','F_balance_limited','F_raw_pressure','F_balanced_pressure','F_pressure_defined_fraction',
       'F_reference_offset_norm','F_guide_value_exceeds_IQ')


def validate_reference_sources(reference,calibration):
    teacher=f_guided.teacher_reference()
    if (not reference.get('source_reference_geometry_eligible') or reference.get('beta')!=BETA or reference.get('F_teacher')!=teacher
        or reference.get('source_records')!=63000 or reference.get('probe_records')!=512 or not reference.get('F_unchanged')
        or any(reference.get(k)!=0 for k in ['G_checkpoint_reads','optimizer_updates','TX_labels_read','V_reads','target_reads'])):
        raise ValueError('Fixed qualified source soft reference required')
    if (not calibration.get('decision',{}).get('eligible') or calibration.get('alpha')!=ALPHA or calibration.get('reference_beta')!=BETA
        or calibration.get('reference_run')!=REFERENCE_RUN or calibration.get('raw_J_cache_run')!=parent.CACHE_RUN
        or calibration.get('F_teacher')!=teacher or not calibration.get('F_unchanged')
        or calibration.get('source_checkpoint')!='20261031-phase1-DURA-repeatViews-E30-s2026092701-r01'
        or calibration.get('source_commit')!='8cf8e22c9ffb54d271b22e621401de8e34c69068'
        or any(calibration.get(k)!=0 for k in ['new_G_inputs','new_F_clean_J_inputs','optimizer_updates','TX_labels_read','V_reads','target_reads'])):
        raise ValueError('Fixed complete source IQ qualification required')


def validate_offset_metadata(metadata):
    expected=dict(status='SOURCE_SOFT_REFERENCE_OFFSETS_COMPLETE',count=63000,shape=[63000,5],numeric_bytes=1260000,
        source_contract='original7/63/30',reference_run=REFERENCE_RUN,calibration_run=CALIBRATION_RUN,
        raw_J_cache_run=parent.CACHE_RUN,F_teacher=f_guided.teacher_reference(),beta=BETA,alpha=ALPHA,
        new_F_inputs=0,new_F_VJP_inputs=0,TX_labels_read=0,V_reads=0,target_reads=0,optimizer_updates=0)
    if any(metadata.get(key)!=value for key,value in expected.items()):raise ValueError('This run fixed source offset cache required')


def build_offsets(root,p,roles):
    started=time.perf_counter();ids=f_guided.source_ids(p,roles);indices=source_indices(roles)
    reference=root/'runs'/REFERENCE_RUN;calibration=root/'runs'/CALIBRATION_RUN
    refmeta=base.read_json(reference/'results.json');calmeta=base.read_json(calibration/'results.json');validate_reference_sources(refmeta,calmeta)
    with np.load(reference/'reference_geometry.npz',allow_pickle=False) as f:a={key:f[key] for key in f.files}
    for key,value in dict(source_positions=indices,physical_ids=ids[indices],roles=roles[indices]).items():
        if not np.array_equal(a[key],value):raise ValueError('Source reference physical role/order mismatch')
    cache=root/'runs'/f_guided.RUN/'scratch_joint/source_feature_cache';meta=base.read_json(cache/'complete.json')
    expected=dict(status='SOURCE_F_FEATURES_COMPLETE',count=63000,dimension=160,TX_labels_read=0,V_reads=0,target_reads=0,F_updates=0,F_teacher=f_guided.teacher_reference())
    if any(meta.get(key)!=value for key,value in expected.items()) or not np.array_equal(np.load(cache/'physical_ids.npy',allow_pickle=False),ids):
        raise ValueError('Original source F feature cache required')
    features=np.load(cache/'features.npy',mmap_mode='r',allow_pickle=False)
    if features.shape!=(63000,160) or features.dtype!=np.float32 or not np.isfinite(features).all():raise ValueError('Complete source F160 features required')
    u=np.asarray(features,dtype=np.float64);u/=np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-4)
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False)
    probabilities=a['full_probabilities'];centroids=a['leave_RX_centroids']
    if probabilities.shape!=(63000,6) or not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1),1):
        raise ValueError('Original full source soft coordinates required')
    target=make_reference(u,probabilities,rx,centroids)
    raw=root/'runs'/parent.CACHE_RUN/'scratch_joint/source_clean_J_cache';parent.verify_cache_metadata(base.read_json(raw/'complete.json'))
    if not np.array_equal(np.load(raw/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Original raw J source order required')
    basis=np.load(raw/'basis.npy',allow_pickle=False)
    if basis.shape!=(160,5) or np.max(abs(basis-a['head_basis']@(a['head_basis'].T@basis)))>2e-6:raise ValueError('Original same head span required')
    offsets=((4*target@basis).astype(np.float32)-(4*u@basis).astype(np.float32)).astype(np.float32)
    with np.load(calibration/'feasibility.npz',allow_pickle=False) as c:
        if not np.array_equal(c['physical_ids'],ids[indices]) or not np.allclose(offsets[indices],c['offset'],atol=3e-6,rtol=2e-6):
            raise ValueError('Production reference offsets differ from qualified source offsets')
    if offsets.shape!=(63000,5) or not np.isfinite(offsets).all():raise ValueError('Finite complete source reference offsets required')
    folder=p['output']/'source_soft_reference_cache';folder.mkdir()
    np.save(folder/'offsets.npy',offsets,allow_pickle=False);np.save(folder/'physical_ids.npy',ids,allow_pickle=False)
    metadata=dict(status='SOURCE_SOFT_REFERENCE_OFFSETS_COMPLETE',count=63000,shape=[63000,5],numeric_bytes=offsets.nbytes,
        source_contract='original7/63/30',reference_run=REFERENCE_RUN,calibration_run=CALIBRATION_RUN,raw_J_cache_run=parent.CACHE_RUN,
        F_teacher=f_guided.teacher_reference(),beta=BETA,alpha=ALPHA,new_F_inputs=0,new_F_VJP_inputs=0,
        TX_labels_read=0,V_reads=0,target_reads=0,optimizer_updates=0,seconds=time.perf_counter()-started,
        source_feature_cache_metadata=meta,reference_geometry_seconds=refmeta['seconds'],source_IQ_calibration_seconds=calmeta['seconds'])
    validate_offset_metadata(metadata);base.write_json(folder/'complete.json',metadata)
    print(json.dumps(dict(event='SOURCE_SOFT_REFERENCE_OFFSETS_COMPLETE',count=63000,numeric_bytes=offsets.nbytes,seconds=metadata['seconds'])),flush=True)


def load_offsets(p,roles):
    started=time.perf_counter();folder=p['output']/'source_soft_reference_cache';metadata=base.read_json(folder/'complete.json');validate_offset_metadata(metadata)
    ids=f_guided.source_ids(p,roles)
    if not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Source offset physical order mismatch')
    offsets=np.load(folder/'offsets.npy',mmap_mode='r',allow_pickle=False)
    if offsets.shape!=(63000,5) or offsets.dtype!=np.float32:raise ValueError('Complete FP32 source offsets required')
    return offsets,dict(metadata=metadata,seconds=time.perf_counter()-started)


def objective(student,received,clean,action,variance,probes,jacobian,offset):
    _,terms,restored=source_objective(student,received,clean,action,variance,probes,return_restored=True)
    guide,info=prototype_tangent_loss(restored,clean,jacobian,offset,ALPHA)
    loss=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']+ALPHA*guide
    return loss,dict(terms,**{METRIC:guide,'F_balance_weight':info['weight'],'F_balance_limited':(info['weight']<1).float(),
        'F_raw_pressure':info['raw_pressure'],'F_balanced_pressure':info['bounded_pressure'],
        'F_pressure_defined_fraction':info['pressure_defined'].float(),'F_reference_offset_norm':offset.detach().norm(dim=1),
        'F_guide_value_exceeds_IQ':(ALPHA*guide.detach()>terms['iq_nmse'].detach()).float()}),restored


def smoke(args,p):
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    clean,roles=base.source_training(p);build_offsets(args.root,p,roles)
    cache,_=parent.metric_cache(args.root,p,roles);offsets,_=load_offsets(p,roles)
    take=np.flatnonzero(roles=='L_s')[:2];folder=p['cache']/'epochs/01'
    values={key:base.tensor(np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu') for key in ['received','kernel','coefficients','variance']}
    x=base.tensor(clean[take],'cpu');jac=base.tensor(cache[take],'cpu');off=base.tensor(offsets[take],'cpu');student=base.scratch_model(2026092701)
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False);before={key:value.clone() for key,value in reader.state_dict().items()}
    rng=torch.Generator().manual_seed(2026103004);probes=[torch.complex(torch.randn(2,256,generator=rng),torch.randn(2,256,generator=rng)) for _ in range(2)]
    loss,terms,restored=objective(student,values['received'],x,base.supervised_action(values['kernel'],values['coefficients']),values['variance'],probes,jac,off)
    gradient=torch.autograd.grad((ALPHA*terms[METRIC]).sum(),restored,retain_graph=True)[0]
    IQnorm=2*(restored.detach()-x).flatten(1).norm(dim=1)/(512*x.square().mean((1,2)).clamp_min(1e-4))
    if not torch.isfinite(gradient).all() or not gradient.abs().sum()>0 or (gradient.flatten(1).norm(dim=1)>IQnorm*(1+2e-5)).any():
        raise ValueError('Actual G-IQ shifted guidance bound failed')
    loss.mean().backward()
    for prefix in ['frozen.coarse.','frozen.estimator.','prior.']:
        grads=[parameter.grad for name,parameter in student.named_parameters() if name.startswith(prefix) and parameter.grad is not None]
        if not grads or not all(torch.isfinite(value).all() for value in grads) or not any(value.abs().sum()>0 for value in grads):raise ValueError('Actual joint G gradient missing: '+prefix)
    with torch.no_grad():expected=student(values['received']).clone()
    path=p['output']/'smoke-scratch_joint.pt';f_guided.save_model(path,student,'scratch_joint',0,0,args.commit)
    loaded=base.load_model(path,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    with torch.no_grad():
        if not torch.equal(loaded(values['received']),expected):raise ValueError('Strict scratch G reload changed IQ')
        scores=reader(expected)
    if scores.shape!=(2,6) or not torch.isfinite(scores).all() or any(not torch.equal(value,before[key]) for key,value in reader.state_dict().items()) or any(parameter.grad is not None for parameter in reader.parameters()):
        raise ValueError('Original F/defaulthead/state smoke failed')
    base.write_json(p['output']/'smoke.json',dict(status='PASS',real_Ls=2,F_inputs=2,TX_labels_read=0,target_reads=0,F_updates=0,F_unchanged=True,
        shifted_IQ_gradient_bound=True,joint_coarse_action_prior_gradients=True,strict_reload=True,initialization_checkpoint_loads=0))
    print(json.dumps(dict(event='REAL_SOURCE_SOFT_REFERENCE_G_SMOKE_PASS')),flush=True)


def train(args,p):
    started_total=time.perf_counter();row=p['output']/args.arm;row.mkdir()
    base.write_json(row/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],commit=args.commit))
    clean,roles=base.source_training(p);targets,reuse=parent.metric_cache(args.root,p,roles);offsets,reference=load_offsets(p,roles)
    base.write_json(row/'cache_reuse.json',reuse);base.write_json(row/'source_reference.json',reference)
    torch.manual_seed(2026092701);torch.cuda.manual_seed_all(2026092701);student=base.scratch_model(2026092701).to('cuda:0')
    before={key:value.detach().cpu().clone() for key,value in student.state_dict().items()};optimizer=torch.optim.AdamW(student.parameters(),lr=.001,weight_decay=1e-4)
    base.write_json(row/'resolved_config.json',base.configuration());base.write_json(row/'initialization.json',student.initialization_report)
    preparation=time.perf_counter()-started_total;started=time.perf_counter();rng=np.random.default_rng(2026103004);updates=seen_l=seen_u=0
    torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x',encoding='utf-8') as stream:
        for epoch in range(1,31):
            view=(epoch-1)%20+1;folder=p['cache']/'epochs'/f'{view:02d}'
            arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ['received','kernel','coefficients','variance']}
            shapes=dict(received=(63000,2,256),kernel=(63000,2,64),coefficients=(63000,4,256),variance=(63000,))
            if any(value.shape!=shapes[key] or value.dtype!=np.float32 for key,value in arrays.items()):raise ValueError('Original full source caches required')
            totals=dict.fromkeys(['loss','iq_nmse','prior_nmse','spectral_error','action_probe_nmse','log_variance_mse',METRIC,*EXTRA],0.)
            clipped=0;maximum_pressure=0.;order=rng.permutation(63000)
            for start in range(0,63000,256):
                take=order[start:start+256];x=base.tensor(clean[take],'cuda:0');jac=base.tensor(targets[take],'cuda:0');off=base.tensor(offsets[take],'cuda:0')
                values={key:base.tensor(value[take],'cuda:0') for key,value in arrays.items()};action=base.supervised_action(values['kernel'],values['coefficients'])
                probes=[torch.complex(torch.randn(len(take),256,device='cuda:0'),torch.randn(len(take),256,device='cuda:0')) for _ in range(2)]
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=base.learning_rate(updates)
                loss,terms,_=objective(student,values['received'],x,action,values['variance'],probes,jac,off)
                if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite shifted reference objective')
                maximum_pressure=max(maximum_pressure,float(terms['F_balanced_pressure'].max()))
                if maximum_pressure>1+2e-5:raise FloatingPointError('Shifted guide IQ-output gradient bound violated')
                loss.mean().backward();norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True);clipped+=int(norm>1)
                optimizer.step();updates+=1;seen_l+=int((roles[take]=='L_s').sum());seen_u+=int((roles[take]=='U_s').sum())
                for key,value in dict(loss=loss,**terms).items():totals[key]+=float(value.detach().sum())
            record=dict(epoch=epoch,source_cache_view=view,updates=updates,losses={key:value/63000 for key,value in totals.items()},
                L_exposures=seen_l,U_exposures=seen_u,clip_fraction=clipped/247,maximum_balanced_IQ_pressure=maximum_pressure,
                seconds=time.perf_counter()-started,F_inputs=0,new_F_cache_inputs=0,F_J_reference_inputs=seen_l+seen_u,
                source_reference_offset_inputs=seen_l+seen_u,F_updates=0,TX_labels_read=0,initialization_checkpoint_loads=0)
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();print(json.dumps(dict(event='SOURCE_SOFT_REFERENCE_EPOCH_COMPLETE',**record)),flush=True)
    if (updates,seen_l,seen_u)!=(7410,189000,1701000):raise ValueError('Incomplete single E30 budget')
    changed=sum(not torch.equal(value.detach().cpu(),before[key]) for key,value in student.state_dict().items())
    if not changed:raise ValueError('Scratch G did not train')
    f_guided.save_model(row/'final.pt',student,args.arm,30,7410,args.commit)
    base.write_json(row/'training_complete.json',dict(status='SOURCE_TRAINED',arm=args.arm,epoch=30,updates=7410,L_exposures=seen_l,U_exposures=seen_u,
        training_TX_labels_read=0,training_F_access=True,V_reads=0,target_reads=0,F_updates=0,F_teacher=f_guided.teacher_reference(),
        F_loaded_in_training=False,F_clean_teacher_inputs=0,F_clean_J_VJP_inputs=0,F_restored_training_inputs=0,F_J_reference_inputs=1890000,
        source_reference_offset_inputs=1890000,cache_reuse=reuse,source_reference=reference,
        initialization_checkpoint_loads=0,G_ancestor_epochs=0,cumulative_G_epochs=30,F_reused_teacher_epochs=50,head_fit_updates=0,
        parameters=266951,changed_state_tensors=changed,seconds=time.perf_counter()-started,source_preparation_seconds=preparation,
        total_G_training_pipeline_seconds=time.perf_counter()-started_total,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        training_regime='source_F_guided_soft_crossRX_reference_cached_derivatives'))


def configure():
    parent.configure();previous_configuration=base.configuration;previous_command=base.command
    def configuration():
        value=previous_configuration();value.update(run=RUN,recipe='scratch_E30_soft_crossRX_reference_shifted_tangent',
            F_guidance_metric=METRIC,objective='noVarfourphysicalterms+alpha detached-step-weight shiftedclean-J quadratic',
            F_tangent_reference='Qualified soft source otherRX/day feature mean beta.25 mixedwithoriginalcleanfeature;notTXtruth',
            reference_run=REFERENCE_RUN,source_IQ_calibration_run=CALIBRATION_RUN,reference_beta=BETA,
            research_qualification=CALIBRATION_RUN,source_qualification_semantics='Source reference geometry andminIQ/localgradient pass;notearlytrajectory/targetefficacy guarantee',
            matched_control=parent.RUN,matched_control_commit='75da21407e7a6790f45e4a8ac4bcc2ea5b6d120e',
            matched_pureG_control='20261031-phase1-DURA-repeatViews-E30-s2026092701-r01',
            F_guidance_value_semantics='Step-local detached scalar surrogate;offsetloss value notboundedbyIQ;IQ-outputgradientonly,notGparameterbound',
            source_reference_offset_shape=[63000,5],source_reference_offset_numeric_bytes=1260000,source_reference_offset_inputs=1890000,
            source_reference_cache_new_F_inputs=0,source_reference_cache_new_F_VJP_inputs=0,
            source_supervision='SourcepairedIQ/action plusfixedoriginalFsourceJ andsoftRXreferences;no directTX/V/target,notF-free')
        return value
    def command(args,mode,arm=None,role=None):return [MODULE if item==parent.MODULE else item for item in previous_command(args,mode,arm,role)]
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.smoke=smoke;base.train=train


if __name__=='__main__':configure();base.main()
