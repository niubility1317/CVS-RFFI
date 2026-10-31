"""Source-only IQ-domain C4 symmetrization of fixed DURA and a scratch control."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from experiments.pamf_dura_direct_train import repeat_views,f_guided
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor,numpy_values
from .context_oracle import validate_source
from .quarter_orbit import quarter_orbit,rotate_iq,decide

base=repeat_views.base
RUN='20261031-diagnostic-DURA-source-quarterOrbit-s2026092701-r01'
SOURCE_COMMIT='8cf8e22c9ffb54d271b22e621401de8e34c69068'
RULE=('PrimarydefaultDURA only:source512128L384U,cleanandcache01/20. Four exactquarter-turn received views;'
 'inverse-rotate each restoredIQ thenaverage;Fsees nativeoroneaveragedIQ,notrotatedIQ/logitensemble. '
 'Bothcachedviews require orbit/nativeIQNMSE<=.9 andcleanFchoicegain>=6/512;eachnoncleanscene loses<=1choice, '
 'clean scene loses0;full512cleaninputchoices maynotfall. Failure doesnotrelax. PureG E30 descriptivecontrol '
 'cannotbeselectedinstead. Passonlysupportsseparateregistrationoffullsevenformalruntime,notautomaticdeployment/training.')


def aggregate(vectors,mask):
    cases={}
    for name in ['native','orbit']:
        cases[name]=dict(iq_nmse=float(vectors[name+'_nmse'][mask].mean(dtype=np.float64)),
            teacher_correct=int(vectors[name+'_agreement'][mask].sum()),feature_cosine_error=float(vectors[name+'_cosine_error'][mask].mean(dtype=np.float64)))
    metrics={key:dict(mean=float(value[mask].mean(dtype=np.float64)),median=float(np.median(value[mask])),p95=float(np.quantile(value[mask],.95)))
        for key,value in vectors.items()}
    return dict(count=int(mask.sum()),cases=cases,metrics=metrics)


def run(args):
    if args.output!=args.root/'runs'/RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU-only source diagnostic required')
    repeat_views.configure();p=base.paths(args.root)
    validate_source(base.read_json(p['output']/'scratch_joint/training_complete.json'))
    primary=base.load_model(p['output']/'scratch_joint/final.pt','cpu',arm='scratch_joint',commit=SOURCE_COMMIT)
    default,_=base.parent_model(p,'cpu');models=dict(DURA_default=default,pureG_E30=primary)
    for model in models.values():model.eval().requires_grad_(False)
    clean,roles=base.source_training(p);ids=f_guided.source_ids(p,roles);indices=source_indices(roles)
    cache=args.root/'runs'/f_guided.RUN/'scratch_joint/source_feature_cache';meta=base.read_json(cache/'complete.json')
    if (meta['status'],meta['count'],meta['dimension'],meta['F_teacher'],meta['TX_labels_read'],meta['target_reads'])!=(
        'SOURCE_F_FEATURES_COMPLETE',63000,160,f_guided.teacher_reference(),0,0):raise ValueError('Original source F clean cache required')
    if not np.array_equal(np.load(cache/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Original source physical F order required')
    features=np.load(cache/'features.npy',mmap_mode='r',allow_pickle=False)
    if features.shape!=(63000,160) or features.dtype!=np.float32:raise ValueError('Complete original source FP32 F features required')
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    snapshots={name:{key:value.clone() for key,value in model.state_dict().items()} for name,model in models.items()}
    f_snapshot={key:value.clone() for key,value in reader.state_dict().items()}
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False)[indices];day=np.load(p['cache']/'train/day.npy',allow_pickle=False)[indices]
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,
        CPU_threads=2,GPU_workers=0,primary_model='DURA_default',rule=RULE))
    smoke=[]
    with torch.no_grad():
        x=tensor(clean[indices[:2]],'cpu')
        for name,model in models.items():
            orbit,routes=quarter_orbit(model,x);native=model(x)
            rotated,_=quarter_orbit(model,rotate_iq(x,1))
            native_error=float((routes[0]-native).abs().max());equivariance_error=float((rotated-rotate_iq(orbit,1)).abs().max())
            scores=reader(torch.cat((native,orbit),0))
            if native_error>2e-5 or equivariance_error>2e-5 or scores.shape!=(4,6) or not torch.isfinite(scores).all():
                raise ValueError('Actual source2Ls native/equivariance/originalF smoke failed')
            smoke.append(dict(model=name,native_replay_max_error=native_error,quarter_equivariance_max_error=equivariance_error))
            print(json.dumps(dict(event='SOURCE_QUARTER_ORBIT_SMOKE_PASS',real_Ls=2,**smoke[-1])),flush=True)
    base.write_json(args.output/'smoke.json',dict(status='PASS',real_Ls_per_model=2,checks=smoke,G_inputs=36,F_inputs=8,TX_labels_read=0,target_reads=0))
    summary={};started=time.perf_counter()
    with torch.no_grad():
        for name,model in models.items():
            summary[name]={}
            for view in ['clean','1','20']:
                if view=='clean':
                    received=clean;labels=dict(scene=np.full(512,'clean'),state=np.full(512,'clean'),lock=np.full(512,'not_applicable'))
                else:
                    folder=p['cache']/'epochs'/f'{int(view):02d}';received=np.load(folder/'received.npy',mmap_mode='r',allow_pickle=False)
                    with np.load(folder/'groups.npz',allow_pickle=False) as groups:labels={key:groups[key][indices] for key in ['scene','state','lock']}
                    if set(labels['scene'].tolist())!=set(base.VIEWS):raise ValueError('All seven original source scenes required')
                if received.shape!=(63000,2,256) or received.dtype!=np.float32:raise ValueError('Original complete FP32 source IQ required')
                vectors={};outputs=[];route_outputs=[];F_features=[];F_logits=[]
                for offset in range(0,512,32):
                    take=indices[offset:offset+32];x=tensor(clean[take],'cpu');y=tensor(received[take],'cpu')
                    reference=tensor(features[take],'cpu');teacher=reader.classify_features(reference)
                    orbit,routes=quarter_orbit(model,y);iq=torch.stack((routes[0],orbit))
                    embedding=torch.stack([reader.features(value) for value in iq]);logits=torch.stack([reader.classify_features(value) for value in embedding])
                    if not torch.isfinite(routes).all() or not torch.isfinite(embedding).all() or not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite source orbit response')
                    power=x.square().mean((1,2)).clamp_min(1e-4);errors=(iq-x[None]).square().mean((2,3))/power[None]
                    cos=1-F.cosine_similarity(embedding,reference[None],dim=2,eps=1e-8);agree=(logits.argmax(2)==teacher.argmax(1)[None]).float()
                    values=dict(native_nmse=errors[0],orbit_nmse=errors[1],native_cosine_error=cos[0],orbit_cosine_error=cos[1],
                        native_agreement=agree[0],orbit_agreement=agree[1],
                        route_disagreement_nmse=(routes-orbit[None]).square().mean((0,2,3))/power,
                        average_route_nmse=(routes-x[None]).square().mean((0,2,3))/power,
                        orbit_change_nmse=(orbit-routes[0]).square().mean((1,2))/power)
                    if any(value.shape!=(len(take),) or not torch.isfinite(value).all() for value in values.values()):raise ValueError('Finite complete source orbit metrics required')
                    for key,value in values.items():vectors.setdefault(key,[]).append(numpy_values(value))
                    outputs.append(numpy_values(iq));route_outputs.append(numpy_values(routes));F_features.append(numpy_values(embedding))
                    F_logits.append(numpy_values(torch.cat((teacher[None],logits),0)))
                values={key:np.concatenate(value) for key,value in vectors.items()}
                if any(value.shape!=(512,) for value in values.values()):raise ValueError('Full512 source metrics required')
                np.savez(args.output/f'{name}-{view}.npz',source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx,day=day,
                    **labels,clean=clean[indices],received=received[indices],IQ=np.concatenate(outputs,axis=1),routes=np.concatenate(route_outputs,axis=1),
                    F_features=np.concatenate(F_features,axis=1),F_logits=np.concatenate(F_logits,axis=1),clean_F_features=features[indices],**values)
                masks={'all':np.ones(512,bool)}
                for key,group in dict(role=roles[indices],rx=rx,day=day,**labels).items():
                    for value in np.unique(group):masks[f'{key}:{value}']=group==value
                summary[name][view]={key:aggregate(values,mask) for key,mask in masks.items()}
                print(json.dumps(dict(event='SOURCE_QUARTER_ORBIT_CELL_COMPLETE',model=name,view=view,count=512,seconds=time.perf_counter()-started)),flush=True)
    for name,model in models.items():
        if any(not torch.equal(value,snapshots[name][key]) for key,value in model.state_dict().items()) or any(v.grad is not None for v in model.parameters()):raise ValueError('Frozen source G changed')
    if any(not torch.equal(value,f_snapshot[key]) for key,value in reader.state_dict().items()) or any(v.grad is not None for v in reader.parameters()):raise ValueError('Frozen source F changed')
    decisions={name:decide(cells) for name,cells in summary.items()};eligible=decisions['DURA_default']['eligible']
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',summary=summary,decisions=decisions,runtime_study_eligible=eligible,
        primary_model='DURA_default',control_not_selectable=True,source_rule=RULE,source_records=512,source_L=128,source_U=384,views=['clean','1','20'],
        seconds=time.perf_counter()-started,G_inputs=12288,G_smoke_inputs=36,F_inputs=6144,F_smoke_inputs=8,
        source_G=repeat_views.RUN,source_G_commit=SOURCE_COMMIT,default_G=base.PARENT,F_teacher=f_guided.teacher_reference(),source_F_cache_metadata=meta,
        G_unchanged=True,F_unchanged=True,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,
        route_cost=dict(G_input_multiplier=4,shared_parameters=266951,prior_calls_total=8,CG_updates_total=16,FFT_IFFT_total=108,F_input_multiplier=1),
        scope='Source C4-projection/IQ/cleanF-choice proxy only;not F globalphaseinvariance/trueTXaccuracy/targetbenefit/recoverylimit;no trainednetwork/automaticdeployment'))
    base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',runtime_study_eligible=eligible,optimizer_updates=0,predictions=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
