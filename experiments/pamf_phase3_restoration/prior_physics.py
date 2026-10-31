"""Source-only known-action correction of frozen learned priors and F geometry."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from .cache import tensor,load_F0,check_F0_state
from .source_pair import read,validate_source
from .source_inverse import panel,action_values,SCENES
from .multiview_trial import source,load,RUN as G_RUN
from .main_split_eval import dura,valid_identity
from .cost_probe import gpu_choice,write

RUN='20261101-diagnostic-Restore-learned-prior-physics-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.prior_physics'
G_COMMIT='8720a8584803366d42d2879fb99502b8d220ac26'
PRIOR_VARIANCE=.01
METHODS=('DURA_v1','MSAT','DURA_oracle16','DURA_oracle64','MSAT_oracle16','MSAT_oracle64')
METRICS=('nmse','feature_cosine_distance','feature_standardized_mse','clean_decision_agreement')


def source_feature_variance(total,squared,count):
    if count!=63000 or total.shape!=(160,) or squared.shape!=(160,) or not torch.isfinite(total).all() or not torch.isfinite(squared).all():
        raise ValueError('All original63000 source fused160 moments required')
    variance=(squared/count-(total/count).square()).clamp_min(0)
    floor=(.01*variance.mean()).clamp_min(1e-10)
    return variance,variance.clamp_min(floor),floor


def feature_errors(actual,target,scale):
    if actual.ndim!=2 or actual.shape[1]!=160 or target.shape!=actual.shape or scale.shape!=(160,):
        raise ValueError('Matched per-record fused160 features required')
    return (actual-target).square()/scale


def worker(args):
    from experiments.pamf_action_restore.operators import soft_map
    started=time.perf_counter();device='cuda:0';torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    old,roles=source(args);selected=panel(roles);out=args.root/'runs'/RUN
    c=validate_source(read(args.inputs));F,identity=load_F0(c,device);valid_identity(identity)
    # Import/factories above precede the older DURA namespace bootstrap.
    G=load(args.root/'runs'/G_RUN/'rows/MSAT_M20/training/final.pt','MSAT_M20',G_COMMIT,device)
    D,engine=dura(c,device);models=[F,G,D]
    snapshots=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in models]
    clean=np.load(old/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    if clean.shape!=(63000,2,256) or clean.dtype!=np.float32:raise ValueError('Actual source clean FP32 required')
    targets=tensor(clean[selected],device)
    # First NN work is a real source2 smoke, then continue immediately.
    with torch.inference_mode():
        for model_output in (G(targets[:2]),engine(targets[:2])[0]):
            scores=F.classify_features(F.features(model_output))
            if model_output.shape!=(2,2,256) or scores.shape!=(2,6) or not torch.isfinite(scores).all() or not torch.isfinite(model_output).all():
                raise ValueError('Actual frozen source checkpoint smoke failed')
    write(out/'smoke.json',dict(status='PASS',actual_source_L_s=2,F0_identity=identity,
        MSAT_fixed30_7410=G_COMMIT,default_DURA=True,optimizer_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0))
    print('LEARNED_PRIOR_PHYSICS_SOURCE_SMOKE_PASS',flush=True)
    total=torch.zeros(160,device=device,dtype=torch.float64);squared=total.clone();count=0
    with torch.inference_mode():
        for start in range(0,63000,256):
            z=F.features(tensor(clean[start:start+256],device)).double()
            total+=z.sum(0);squared+=z.square().sum(0);count+=len(z)
        variance,scale,floor=source_feature_variance(total,squared,count)
        scale=scale.float();bands=torch.argsort(variance,stable=True).reshape(4,40)
        target_features=F.features(targets);target_decisions=F.classify_features(target_features).argmax(1)
    clean._mmap.close()
    stats={name:value.cpu().tolist() for name,value in [('variance',variance),('scale',scale),('bands',bands)]}
    write(out/'feature_geometry.json',dict(count=count,source_only=True,TX_labels_read=0,variance_floor=float(floor),**stats))
    records={key:[] for key in ['epoch','source_index','scene','state','lock',
        *[m+'__'+metric for m in METHODS for metric in (*METRICS,*['variance_band'+str(i) for i in range(4)])]]}
    per_dim={scene:{m:np.zeros(160,np.float64) for m in METHODS} for scene in SCENES};scene_counts={s:0 for s in SCENES}
    with torch.inference_mode():
        for epoch in range(1,21):
            folder=old/'epochs'/f'{epoch:02d}';done=read(folder/'complete.json')
            if (done.get('count'),done.get('TX_labels_read'),done.get('target_reads'))!=(63000,0,0):raise ValueError('Complete original source draw required')
            arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('received','kernel','coefficients','variance')}
            with np.load(folder/'groups.npz',allow_pickle=False) as group:groups={k:group[k][selected] for k in ('scene','state','lock')}
            try:
                for offset in range(0,512,64):
                    take=selected[offset:offset+64];y=targets[offset:offset+64];n=len(y)
                    v={k:tensor(a[take],device) for k,a in arrays.items()};action=action_values(v['kernel'],v['coefficients'])
                    observed=torch.complex(v['received'][:,0],v['received'][:,1])
                    values={'DURA_v1':engine(v['received'])[0],'MSAT':G(v['received'])}
                    for name,base in list(values.items()):
                        p=torch.complex(base[:,0],base[:,1])
                        for steps in (16,64):
                            z=soft_map(action,observed,p,v['variance'],PRIOR_VARIANCE,steps=steps)
                            values[('DURA' if name=='DURA_v1' else 'MSAT')+'_oracle'+str(steps)]=torch.stack((z.real,z.imag),1)
                    energy=y.square().mean((1,2))
                    scenes=groups['scene'][offset:offset+n]
                    for scene in np.unique(scenes):scene_counts[str(scene)]+=int((scenes==scene).sum())
                    for method,value in values.items():
                        features=F.features(value);component=feature_errors(features,target_features[offset:offset+n],scale)
                        metrics=dict(nmse=(value-y).square().mean((1,2))/energy,
                            feature_cosine_distance=1-torch.nn.functional.cosine_similarity(features,target_features[offset:offset+n],dim=1),
                            feature_standardized_mse=component.mean(1),
                            clean_decision_agreement=F.classify_features(features).argmax(1)==target_decisions[offset:offset+n])
                        for i in range(4):metrics['variance_band'+str(i)]=component[:,bands[i]].mean(1)
                        if not torch.isfinite(value).all() or any(not torch.isfinite(x).all() for x in metrics.values()):raise ValueError('Nonfinite source result; never omit cases')
                        for key,x in metrics.items():records[method+'__'+key].extend(x.cpu().tolist())
                        for scene in np.unique(scenes):
                            mask=torch.tensor((scenes==scene).tolist(),device=device)
                            per_dim[str(scene)][method]+=np.asarray(component[mask].double().sum(0).cpu().tolist())
                    records['epoch'].extend([epoch]*n);records['source_index'].extend(take.tolist())
                    for key in groups:records[key].extend(groups[key][offset:offset+n].tolist())
            finally:
                for a in arrays.values():a._mmap.close()
            print('LEARNED_PRIOR_PHYSICS_DRAW_COMPLETE',epoch,512,flush=True)
    for model,before in zip(models,snapshots):check_F0_state(model,before)
    records={k:np.asarray(v) for k,v in records.items()}
    with (out/'source_records.npz').open('xb') as f:np.savez(f,**records)
    write(out/'per_dimension_errors.json',{s:{m:(v/scene_counts[s]).tolist() for m,v in methods.items()} for s,methods in per_dim.items()})
    write(out/'results.json',dict(status='LEARNED_PRIOR_PHYSICS_COMPLETE',run_id=RUN,code_commit=args.commit,F0_identity=identity,
        count=10240,source_packets=512,source_draws=20,source_Ls=128,source_Us=384,source_feature_moment_rows=63000,
        prior_real_variance=PRIOR_VARIANCE,feature_variance_floor=float(floor),summary=summarize(records),
        G_parameters={'MSAT':4249922,'DURA':266951},G_training_updates=0,optimizer_updates=0,model_state_updates=0,
        TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,
        known_action_is_oracle_not_deployment=True,agreement_is_TX_accuracy=False,claim_theoretical_ceiling=False,
        seconds=time.perf_counter()-started,default_DURA_changed=False,goal_complete=False))


def summarize(records):
    scene=records['scene']
    if len(scene)!=10240 or set(scene)!=set(SCENES):raise ValueError('All fixed source cases/scenes required')
    result={}
    for name in ('ALL',*SCENES):
        mask=np.ones(len(scene),bool) if name=='ALL' else scene==name
        result[name]=dict(count=int(mask.sum()),methods={})
        for method in METHODS:
            result[name]['methods'][method]={k:float(records[method+'__'+k][mask].mean())
                for k in (*METRICS,*['variance_band'+str(i) for i in range(4)])}
    return result


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty launch owner required')
    out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir()
    choices=gpu_choice()
    while not choices:time.sleep(5);choices=gpu_choice()
    admission=choices[0]
    command=[sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),'--commit',args.commit,
        '--inputs',str(args.inputs),'--cache',str(args.cache)]
    with (out/'logs/worker.log').open('xb') as log:
        p=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(admission['GPU'])),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'submitted.json',dict(pid=p.pid,command=command,CWD=os.getcwd(),admission=admission))
    code=p.wait();write(out/'terminal.json',dict(pid=p.pid,exit_code=code))
    if code:raise RuntimeError('Source counterfactual failed; preserve artifacts/no auto retry')
    if read(out/'results.json')['status']!='LEARNED_PRIOR_PHYSICS_COMPLETE':raise ValueError('Missing complete source results')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',optimizer_updates=0,target_inputs=0,goal_complete=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    a=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Actual Git code identity required')
    if a.mode=='run':controller(a)
    else:worker(a)


if __name__=='__main__':main()
