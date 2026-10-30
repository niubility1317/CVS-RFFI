"""Scratch compact E30 with fixed original-R17 final IQ hints on corrupted source IQ."""
import json
import os
import sys
import time
import numpy as np
import torch
from . import no_variance
from .coarse_hint import learning_rate
from experiments.pamf_dura_efficiency.model import direct_objective as source_objective
from experiments.pamf_dura_efficiency.r17_teacher_paths import TEACHER_RUN,TEACHER_COMMIT

base=no_variance.base
RUN='20261031-phase1-DURA-nativeIQhint-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.native_iq_hint'
CACHE_RUN='20261031-source-DURA-R17-nativeIQ-cache-s2026092701-r01'
CACHE_COMMIT='6848d46e95741082e2783a525e6d0d376e8ff7c6'


def validate_cache_metadata(record,launch,plan,complete):
    expected=dict(status='SOURCE_NATIVE_IQ_CACHE_COMPLETE',source_contract='original7/63/30',
        code_commit=CACHE_COMMIT,source_action_run=base.ACTION,teacher_run=TEACHER_RUN,teacher_commit=TEACHER_COMMIT,
        teacher_component='native_map_prior.stage3',teacher_used_parameters=4665073,
        teacher_coarse_epochs=20,teacher_action_epochs=20,teacher_prior_epochs=20,teacher_bundle_component_epochs=60,
        teacher_source_selection_before_target=True,teacher_unchanged=True,views=list(range(1,21)),count_per_view=63000,
        teacher_inputs=1260000,teacher_smoke_inputs=2,L_visits=126000,U_visits=1134000,numeric_bytes=2580480000,
        teacher_updates=0,F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0,model_optimizer_updates=0)
    if any(record.get(k)!=v for k,v in expected.items()):raise ValueError('Complete original source-only R17 final IQ cache required')
    if launch.get('commit')!=CACHE_COMMIT or plan.get('run')!=CACHE_RUN or plan.get('source_contract')!='original7/63/30':
        raise ValueError('Matched source cache producer required')
    if complete.get('status')!='ANALYZED' or complete.get('failed')!=[]:
        raise ValueError('Finished successful source cache producer required')


def cached_teacher(args,p,row=None):
    started=time.perf_counter();producer=args.root/'runs'/CACHE_RUN
    folder=producer/'scratch_joint/native_teacher_cache';original=base.read_json(folder/'complete.json')
    validate_cache_metadata(original,base.read_json(producer/'launch.json'),base.read_json(producer/'plan.json'),
        base.read_json(producer/'complete.json'))
    for name in ('physical_ids','roles'):
        a=np.load(p['cache']/'train'/(name+'.npy'),allow_pickle=False)
        b=np.load(folder/(name+'.npy'),allow_pickle=False)
        if a.shape!=(63000,) or not np.array_equal(a,b):raise ValueError('Source cache physical order and roles must match')
    record=dict(status='SOURCE_NATIVE_IQ_CACHE_REUSED',source_cache_run=CACHE_RUN,source_cache_commit=CACHE_COMMIT,
        source_cache_path=str(folder),source_cache_metadata=original,source_physical_ids_and_roles_match=True,
        teacher_inputs=0,teacher_checkpoint_loads=0,teacher_updates=0,numeric_bytes=original['numeric_bytes'],
        historic_cache_teacher_inputs=1260000,historic_cache_seconds=original['seconds'],
        historic_teacher_bundle_component_epochs=60,F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0,
        seconds=time.perf_counter()-started)
    if row is not None:base.write_json(row/'teacher_cache_reuse.json',record)
    return folder,record


def nonclean_mask(folder):
    with np.load(folder/'groups.npz',allow_pickle=False) as groups:scene=groups['scene']
    if scene.shape!=(63000,) or not set(np.unique(scene)).issubset(set(base.VIEWS)):
        raise ValueError('Original full source scene metadata required')
    return scene!='clean'


def final_hint(restored,clean,reference,active):
    if reference.shape!=restored.shape or active.shape!=(restored.shape[0],) or active.dtype!=torch.bool:
        raise ValueError('Matched source final IQ and Boolean source corruption mask required')
    energy=clean.detach().square().mean((1,2)).clamp_min(1e-4)
    return (restored-reference.detach()).square().mean((1,2))/energy*active.detach()


def objective(student,received,clean,action,variance,probes,reference,active):
    _,terms,restored=source_objective(student,received,clean,action,variance,probes,return_restored=True)
    hint=final_hint(restored,clean,reference,active)
    loss=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']+hint
    return loss,dict(terms,native_hint_nmse=hint,native_hint_active=active.detach().to(restored.dtype))


def smoke(args,p):
    clean,roles=base.source_training(p);folder=p['cache']/'epochs/01';active=nonclean_mask(folder)
    take=np.flatnonzero((roles=='L_s')&active)[:2]
    if len(take)!=2:raise ValueError('Two real corrupted source L_s required')
    cache,_=cached_teacher(args,p)
    reference=base.tensor(np.load(cache/'epoch01.npy',mmap_mode='r',allow_pickle=False)[take],'cpu')
    values={k:base.tensor(np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu')
        for k in ('received','kernel','coefficients','variance')}
    student=base.scratch_model(2026092701);x=base.tensor(clean[take],'cpu')
    generator=torch.Generator().manual_seed(2026103004)
    probes=[torch.complex(torch.randn(2,256,generator=generator),torch.randn(2,256,generator=generator)) for _ in range(2)]
    loss,_=objective(student,values['received'],x,base.supervised_action(values['kernel'],values['coefficients']),
        values['variance'],probes,reference,torch.ones(2,dtype=torch.bool))
    loss.mean().backward();gradients=[v.grad for v in student.parameters() if v.grad is not None]
    if not gradients or not all(torch.isfinite(v).all() for v in gradients) or not any(v.abs().sum()>0 for v in gradients):
        raise ValueError('Finite nonzero source native IQ hint gradients required')
    with torch.no_grad():expected=student(values['received']).clone()
    checkpoint=p['output']/'smoke-scratch_joint.pt';base.save_model(checkpoint,student,'scratch_joint',0,0,args.commit)
    loaded=base.load_model(checkpoint,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    with torch.no_grad():torch.testing.assert_close(loaded(values['received']),expected,atol=0,rtol=0)
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    with torch.no_grad():logits=reader(expected)
    if logits.shape!=(2,6) or not torch.isfinite(logits).all():raise ValueError('Original F/defaulthead no-query smoke failed')
    base.write_json(p['output']/'smoke.json',dict(status='PASS',real_Ls=2,source_nonclean=True,student_scratch=True,
        parameters=266951,strict_reload=True,source_student_gradient_nonzero=True,source_teacher_checkpoint_loads=0,
        source_teacher_forwards=0,cached_source_final_IQ=True,TX_labels_read=0,target_reads=0,training_F_access=False,
        F_used_only_in_noquery_smoke=True,F_updates=0,head_fit_updates=0))
    print('{"event":"REAL_SOURCE_NATIVE_IQ_HINT_NOQUERY_SMOKE_PASS"}',flush=True)


def train(args,p):
    started_total=time.perf_counter();clean,roles=base.source_training(p);config=base.configuration()
    row=p['output']/args.arm;row.mkdir()
    base.write_json(row/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
        physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],commit=args.commit))
    base.write_json(row/'resolved_config.json',config);teacher_cache,cache_record=cached_teacher(args,p,row)
    torch.manual_seed(2026092701);torch.cuda.manual_seed_all(2026092701)
    student=base.scratch_model(2026092701).to('cuda:0')
    before={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
    optimizer=torch.optim.AdamW(student.parameters(),lr=.001,weight_decay=1e-4)
    base.write_json(row/'initialization.json',student.initialization_report)
    rng=np.random.default_rng(2026103004);updates=seen_l=seen_u=teacher_visits=active_visits=0;scene_counts={}
    preparation_seconds=time.perf_counter()-started_total;started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x',encoding='utf-8') as stream:
        for epoch in range(1,31):
            cache_epoch=(epoch-1)%20+1;folder=p['cache']/'epochs'/f'{cache_epoch:02d}'
            arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('received','kernel','coefficients','variance')}
            expected=dict(received=(63000,2,256),kernel=(63000,2,64),coefficients=(63000,4,256),variance=(63000,))
            if any(v.shape!=expected[k] or v.dtype!=np.float32 for k,v in arrays.items()):raise ValueError('Matched source IQ/action required')
            targets=np.load(teacher_cache/f'epoch{cache_epoch:02d}.npy',mmap_mode='r',allow_pickle=False)
            if targets.shape!=(63000,2,256) or targets.dtype!=np.float32:raise ValueError('Matched full source final IQ cache required')
            active=nonclean_mask(folder);scene_counts[str(epoch)]=dict(nonclean=int(active.sum()),clean=int((~active).sum()))
            totals=dict.fromkeys(('loss','iq_nmse','spectral_error','prior_nmse','action_probe_nmse','log_variance_mse','native_hint_nmse','native_hint_active'),0.)
            clipped=0;order=rng.permutation(63000)
            for offset in range(0,63000,256):
                take=order[offset:offset+256];x=base.tensor(clean[take],'cuda:0')
                values={k:base.tensor(v[take],'cuda:0') for k,v in arrays.items()}
                action=base.supervised_action(values['kernel'],values['coefficients'])
                probes=[torch.complex(torch.randn(len(take),256,device='cuda:0'),torch.randn(len(take),256,device='cuda:0')) for _ in range(2)]
                reference=base.tensor(targets[take],'cuda:0');mask=torch.tensor(active[take],device='cuda:0',dtype=torch.bool)
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=learning_rate(updates)
                loss,terms=objective(student,values['received'],x,action,values['variance'],probes,reference,mask)
                if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite source native IQ hint objective')
                loss.mean().backward();norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True)
                clipped+=int(norm>1);optimizer.step();updates+=1
                seen_l+=int((roles[take]=='L_s').sum());seen_u+=int((roles[take]=='U_s').sum())
                teacher_visits+=len(take);active_visits+=int(active[take].sum())
                for k,v in dict(loss=loss,**terms).items():totals[k]+=float(v.detach().sum())
            record=dict(epoch=epoch,source_cache_view=cache_epoch,updates=updates,losses={k:v/63000 for k,v in totals.items()},
                L_exposures=seen_l,U_exposures=seen_u,clip_fraction=clipped/247,seconds=time.perf_counter()-started,
                F_inputs=0,TX_labels_read=0,teacher_reference_visits=teacher_visits,teacher_active_nonclean_visits=active_visits,
                live_teacher_training_inputs=0,learning_rate_last=optimizer.param_groups[0]['lr'],initialization_checkpoint_loads=0)
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
            print(json.dumps(dict(event='SOURCE_EPOCH_COMPLETE',arm=args.arm,**record)),flush=True)
    if (updates,seen_l,seen_u,teacher_visits)!=(7410,189000,1701000,1890000):raise ValueError('Incomplete fixed source budget')
    if active_visits!=sum(r['nonclean'] for r in scene_counts.values()):raise ValueError('Incomplete source mask budget')
    changed=sum(not torch.equal(v.detach().cpu(),before[k]) for k,v in student.state_dict().items())
    if not changed:raise ValueError('Scratch source model did not train')
    base.save_model(row/'final.pt',student,args.arm,30,updates,args.commit)
    base.write_json(row/'training_complete.json',dict(status='SOURCE_TRAINED',arm=args.arm,epoch=30,updates=updates,
        L_exposures=seen_l,U_exposures=seen_u,training_TX_labels_read=0,training_F_access=False,V_reads=0,target_reads=0,
        teacher_updates=0,teacher_reference_visits=teacher_visits,teacher_active_nonclean_visits=active_visits,
        source_scene_counts=scene_counts,live_teacher_training_inputs=0,initialization_checkpoint_loads=0,ancestor_epochs=0,
        cumulative_epochs=30,head_fit_updates=0,parameters=sum(v.numel() for v in student.parameters()),changed_state_tensors=changed,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.perf_counter()-started,
        source_preparation_seconds=preparation_seconds,total_G_training_pipeline_seconds=time.perf_counter()-started_total,
        teacher_cache=cache_record,teacher_cache_forward_inputs=0,teacher_cache_numeric_bytes=cache_record['numeric_bytes'],
        teacher_bundle_component_epochs=60,student_copied_teacher_parameters=0))


def configure():
    no_variance.configure();previous_configuration=base.configuration;previous_command=base.command
    def configuration():
        value=previous_configuration()
        value.update(run=RUN,recipe='scratch_E30_nonclean_final_native_IQ_hint',epochs=30,updates=7410,new_cumulative_epochs=30,
            L_exposures=189000,U_exposures=1701000,corruption_epochs=20,
            source_freeze='fixedfinalE30/7410;noearlystop/source-target-ranking',
            objective='noVar4base + meanbatch(nonclean_source_mask*NMSE(studentfinal,originalR17final));weight1all30',
            teacher_model=dict(run=TEACHER_RUN,commit=TEACHER_COMMIT,component='native_map_prior.stage3',
                bundle_component_epochs=60,student_copied_weights=0),teacher_cache_reused_from=CACHE_RUN,
            teacher_cache_inputs=0,historic_teacher_cache_inputs=1260000,teacher_training_visits=1890000,
            teacher_cache_numeric_bytes=2580480000,source_clean_teacher_weight=0.,source_nonclean_teacher_weight=1.,
            source_mask_training_only=True,inference_needs_scene_or_teacher=False,
            source_supervision='Sourcepairedclean/action andfrozenoriginalR17finalIQ;0TX/F/V/targettraining',
            training_stages=1,parameters=266951,training_F_access=False,F_guided_training=False,
            source_F_informed_method_research=True,historical_R17_source_V_selection=True,
            extra_loss_metrics=['native_hint_nmse','native_hint_active'],allowed_GPUs=[6],max_total_visible_GPU_processes=3,
            comparison_control='20261031-phase1-DURA-IQenhance-E30-s2026092701-r01 physicalreadout;wholeprocessFguided',
            source_rationale='OriginalsourcefinalIQNMSE.017367/.016427 vsnoVar.022352/.021122;Fcleanchoice+10/+7 butlocaldamage;fixedbeforewideformalscores',
            host_numpy_conversion='DLPack_owned_FP32')
        return value
    def command(args,mode,arm=None,role=None):
        return [MODULE if v==no_variance.MODULE else v for v in previous_command(args,mode,arm,role)]
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.learning_rate=learning_rate
    base.smoke=smoke;base.train=train


if __name__=='__main__':configure();base.main()
