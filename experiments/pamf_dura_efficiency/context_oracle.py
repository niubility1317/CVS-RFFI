"""Fixed source context interventions; never deployable or recovery bounds."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_dura_direct_train import repeat_views, f_guided
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor,numpy_values
from .phase_sensitivity import metrics

base=repeat_views.base
RUN='20261031-diagnostic-DURA-source-context-oracles-s2026092701-r01'
PREVIOUS='20261030-diagnostic-DURA-source-phase-oracle-s2026092701-m2-r01'
SOURCE_COMMIT='8cf8e22c9ffb54d271b22e621401de8e34c69068'
CASES=('native','known_action','known_coarse','known_both','rank32_action')
RANK=32
RULE=('Only primaryrepeatE30 may qualify an independent scratchE30 study. Both sourcecached01/20 require '
 'IQNMSE reduction>=10percent and cleanFchoice gain>=6/512, no nonclean scene loses>1choice,clean scene loses0; '
 'all512clean-input choices must not fall. Rank32 additionally requires actual-source-action two-white-probe '
 'relativeNMSE<=.01 at EACH scene andoverall in BOTH views. Prefer qualifiedrank32,elsequalifiedknownaction '
 'supports a separate source-action-curriculumstudy;neverautomatictraining. DefaultDURA/knowncoarse/both aredescriptive. '
 'These sourceproxies are not trueTXaccuracy/targetbenefit or recoverylimits;failure neverloosensrule.')


def low_rank_trace(value,rank=RANK):
    if value.ndim!=2 or value.shape[1]!=256 or not 1<=rank<=256:raise ValueError('Fixed 256-point source action trace required')
    k=torch.arange(rank,device=value.device,dtype=value.real.dtype)[:,None]
    n=torch.arange(256,device=value.device,dtype=value.real.dtype)[None,:]
    basis=torch.cos(math.pi*k*(n+.5)/256)*math.sqrt(2/256);basis[0]/=math.sqrt(2)
    def project(x):return (x@basis.T)@basis
    return torch.complex(project(value.real),project(value.imag))


def project_action(action):
    return Action(action.kernel,low_rank_trace(action.main),low_rank_trace(action.image))


def contexts(clean,context,actual):
    coarse,estimated,variance=context
    return dict(native=context,known_action=(coarse,actual,variance),known_coarse=(clean,estimated,variance),
        known_both=(clean,actual,variance),rank32_action=(coarse,project_action(estimated),variance))


def aggregate(vectors,mask):
    return dict(count=int(mask.sum()),operator_rank32_nmse=float(vectors['operator_rank32_nmse'][mask].mean(dtype=np.float64)),
        cases={case:dict(iq_nmse=float(vectors[case+'/iq_nmse'][mask].mean(dtype=np.float64)),
            teacher_correct=int(vectors[case+'/teacher_agreement'][mask].sum()),
            feature_cosine_error=float(vectors[case+'/feature_cosine_error'][mask].mean(dtype=np.float64)),
            posterior_KL=float(vectors[case+'/posterior_KL'][mask].mean(dtype=np.float64))) for case in CASES})


def decide(primary):
    result={}
    for case in ('known_action','rank32_action'):
        clean=primary['clean']['all'];clean_ok=clean['cases'][case]['teacher_correct']>=clean['cases']['native']['teacher_correct']
        rows=[]
        for view in ('1','20'):
            group=primary[view];overall=group['all'];scenes={k:v for k,v in group.items() if k.startswith('scene:')}
            if overall['count']!=512 or len(scenes)!=7:raise ValueError('All fixed source records/scenes required')
            native=overall['cases']['native'];current=overall['cases'][case]
            gain=current['teacher_correct']-native['teacher_correct']
            choice_ok=all(v['cases'][case]['teacher_correct']-v['cases']['native']['teacher_correct']>=
                (0 if key=='scene:clean' else -1) for key,v in scenes.items())
            physical_ok=case!='rank32_action' or (overall['operator_rank32_nmse']<=.01 and all(v['operator_rank32_nmse']<=.01 for v in scenes.values()))
            passed=native['iq_nmse']>0 and current['iq_nmse']<=.9*native['iq_nmse'] and gain>=6 and choice_ok and physical_ok
            rows.append(dict(source_view=view,passed=bool(passed),choice_gain=gain,physical_rank_fit=bool(physical_ok)))
        result[case]=dict(clean_preserved=bool(clean_ok),views=rows,eligible=bool(clean_ok and all(r['passed'] for r in rows)))
    result['preferred_independent_study']='rank32_action' if result['rank32_action']['eligible'] else ('source_action_curriculum' if result['known_action']['eligible'] else None)
    result['automatic_training']=False
    return result


def validate_source(done):
    expected=dict(status='SOURCE_TRAINED',epoch=30,updates=7410,L_exposures=189000,U_exposures=1701000,
        training_TX_labels_read=0,training_F_access=False,V_reads=0,target_reads=0,teacher_updates=0,
        learned_teacher_inputs=0,initialization_checkpoint_loads=0,ancestor_epochs=0,head_fit_updates=0,parameters=266951)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Complete fixed pure source E30 control required')


@torch.no_grad()
def run(args):
    if args.output!=args.root/'runs'/RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU-only source diagnostic root required')
    repeat_views.configure();p=base.paths(args.root)
    validate_source(base.read_json(p['output']/'scratch_joint/training_complete.json'))
    primary=base.load_model(p['output']/'scratch_joint/final.pt','cpu',arm='scratch_joint',commit=SOURCE_COMMIT)
    default,_=base.parent_model(p,'cpu');models=dict(repeat_E30=primary,DURA_default=default)
    clean,roles=base.source_training(p);ids=f_guided.source_ids(p,roles);indices=source_indices(roles)
    previous=args.root/'runs'/PREVIOUS;old=base.read_json(previous/'results.json')
    if old['F_teacher']!=f_guided.teacher_reference() or not old['G_unchanged'] or not old['F_unchanged']:
        raise ValueError('Original frozen source clean F-reference cache required')
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    f_before={k:v.clone() for k,v in reader.state_dict().items()}
    snapshots={name:{k:v.clone() for k,v in model.state_dict().items()} for name,model in models.items()}
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,
        CPU_threads=2,GPU_workers=0,F_teacher=f_guided.teacher_reference(),source_checkpoint=repeat_views.RUN,source_commit=SOURCE_COMMIT,rule=RULE))
    for name,model in models.items():
        score=reader(model(tensor(clean[indices[:2]],'cpu')))
        if score.shape!=(2,6) or not torch.isfinite(score).all():raise ValueError('Actual source2Ls G-F no-query smoke failed')
        print(json.dumps(dict(event='REAL_SOURCE_CONTEXT_CHECKPOINT_SMOKE_PASS',model=name,Ls=2)),flush=True)
    base.write_json(args.output/'smoke.json',dict(status='PASS',real_Ls_per_model=2,models=list(models),
        G_inputs=4,F_inputs=4,TX_labels_read=0,V_reads=0,target_reads=0,optimizer_updates=0))
    started=time.perf_counter();summary={};native_delta=0.;g=torch.Generator().manual_seed(2026103005)
    probes=[torch.complex(torch.randn(512,256,generator=g),torch.randn(512,256,generator=g)) for _ in range(2)]
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False)[indices];day=np.load(p['cache']/'train/day.npy',allow_pickle=False)[indices]
    for name,model in models.items():
        summary[name]={}
        for view in ('clean','1','20'):
            old_name='noVar' if name=='repeat_E30' else 'DURA_default'
            with np.load(previous/f'{old_name}-{view}.npz',allow_pickle=False) as archived:
                for key,expected in dict(source_positions=indices,physical_ids=ids[indices],roles=roles[indices],clean=clean[indices],rx=rx,day=day).items():
                    if not np.array_equal(archived[key],expected):raise ValueError('Original paired source reference order required')
                reference_features=archived['F_features'][0].copy();reference_logits=archived['F_logits'][0].copy()
                previous_native=archived['native'].copy();native_features=archived['F_features'][1].copy();native_logits=archived['F_logits'][1].copy()
                labels={key:archived[key].copy() for key in ('scene','state','lock')}
            if reference_features.shape!=(512,160) or reference_logits.shape!=(512,6):raise ValueError('Complete original source clean F reference required')
            arrays=None
            if view=='clean':received=clean
            else:
                folder=p['cache']/'epochs'/f'{int(view):02d}'
                arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ('received','kernel','coefficients')};received=arrays['received']
                if received.shape!=(63000,2,256) or received.dtype!=np.float32:raise ValueError('Original full source FP32 observation required')
                with np.load(folder/'groups.npz',allow_pickle=False) as groups:
                    if any(not np.array_equal(groups[key][indices],labels[key]) for key in labels):raise ValueError('Matched source scene metadata required')
            vectors={};iq=[];features=[];logits=[]
            for offset in range(0,512,32):
                take=indices[offset:offset+32];x=tensor(clean[take],'cpu');y=tensor(received[take],'cpu')
                if arrays is None:
                    kernel=x.new_zeros((32,64),dtype=torch.complex64);kernel[:,0]=1
                    actual=Action(kernel,torch.ones(32,256,dtype=torch.complex64),torch.zeros(32,256,dtype=torch.complex64))
                else:actual=base.supervised_action(tensor(arrays['kernel'][take],'cpu'),tensor(arrays['coefficients'][take],'cpu'))
                projected=project_action(actual);errors=[]
                for probe in probes:
                    z=probe[offset:offset+32];target=actual(z)
                    errors.append((projected(z)-target).abs().square().mean(1)/target.abs().square().mean(1).clamp_min(1e-4))
                local=dict(operator_rank32_nmse=torch.stack(errors).mean(0));context=model.estimate_context(y)
                cells=contexts(x,context,actual);iq_batch=[];f_batch=[tensor(reference_features[offset:offset+32],'cpu')];z_batch=[tensor(reference_logits[offset:offset+32],'cpu')]
                for case in CASES:
                    restored=model.stage_outputs(y,cells[case])[0][-1]
                    if restored.shape!=x.shape or not torch.isfinite(restored).all():raise FloatingPointError('Finite source context IQ required')
                    if case=='native' and name=='DURA_default':
                        delta=float((restored-tensor(previous_native[offset:offset+32],'cpu')).abs().max());native_delta=max(native_delta,delta)
                        if delta>2e-5:raise ValueError('Native full-stage replay differs from archived source IQ')
                        f=tensor(native_features[offset:offset+32],'cpu');z=tensor(native_logits[offset:offset+32],'cpu')
                    else:f=reader.features(restored);z=reader.classify_features(f)
                    local[case+'/iq_nmse']=(restored-x).square().mean((1,2))/x.square().mean((1,2)).clamp_min(1e-4)
                    local.update({case+'/'+key:v for key,v in metrics(f,z,f_batch[0],z_batch[0]).items()})
                    iq_batch.append(numpy_values(restored));f_batch.append(f);z_batch.append(z)
                for key,value in local.items():
                    if value.shape!=(32,) or not torch.isfinite(value).all():raise FloatingPointError('Finite complete source context metrics required')
                    vectors.setdefault(key,[]).append(numpy_values(value))
                iq.append(np.stack(iq_batch));features.append(numpy_values(torch.stack(f_batch)));logits.append(numpy_values(torch.stack(z_batch)))
            values={key:np.concatenate(v) for key,v in vectors.items()}
            np.savez(args.output/f'{name}-{view}.npz',source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx,day=day,
                **labels,clean=clean[indices],IQ=np.concatenate(iq,axis=1),F_features=np.concatenate(features,axis=1),F_logits=np.concatenate(logits,axis=1),**values)
            masks={'all':np.ones(512,bool)}
            for key,labels_for_group in dict(role=roles[indices],rx=rx,day=day,scene=labels['scene']).items():
                for value in np.unique(labels_for_group):masks[f'{key}:{value}']=labels_for_group==value
            summary[name][view]={key:aggregate(values,mask) for key,mask in masks.items()}
            print(json.dumps(dict(event='SOURCE_CONTEXT_CELL_COMPLETE',model=name,view=view,count=512,seconds=time.perf_counter()-started)),flush=True)
    for name,model in models.items():
        if any(not torch.equal(v,snapshots[name][k]) for k,v in model.state_dict().items()) or any(p.grad is not None for p in model.parameters()):raise ValueError('Frozen source G changed')
    if any(not torch.equal(v,f_before[k]) for k,v in reader.state_dict().items()) or any(p.grad is not None for p in reader.parameters()):raise ValueError('Frozen source F changed')
    decision=decide(summary['repeat_E30'])
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',summary=summary,decision=decision,rule=RULE,cases=list(CASES),
        source_records=512,L_s=128,U_s=384,source_views=['clean',1,20],source_context_examples=3072,source_stage_restoration_examples=15360,
        G_smoke_inputs=4,F_smoke_inputs=4,new_F_feature_inputs=13824,reused_default_native_F_inputs=1536,source_rank=RANK,
        source_checkpoint=repeat_views.RUN,source_commit=SOURCE_COMMIT,F_teacher=f_guided.teacher_reference(),native_replay_max_abs=native_delta,
        G_unchanged=True,F_unchanged=True,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,seconds=time.perf_counter()-started,
        scope='Source training-pool counterfactuals with unavailable sideinformation;notdeployable,nottrueTXaccuracy,targeteffect,MMSE orrecoverybound;no automatictraining'))
    base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',optimizer_updates=0,target_reads=0,goal_complete=False,decision=decision))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args();args.root=args.root.resolve();args.output=args.output.resolve()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
