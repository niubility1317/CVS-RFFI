"""Frozen source-only coordinate counterfactuals; no deployed route or ensemble."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .covariance_detail_source_validation import load_final
from .cache import tensor, load_F0, check_F0_state
from .source_pair import read, validate_source
from .source_inverse import R07, SCENES
from .main_split_eval import valid_identity
from .variance_trial import recovery_gpu_choice
from .cost_probe import write

RUN = '20261102-diagnostic-Restore-CovarianceDetail-source-coordinate-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.source_coordinate_probe'
BASE = '20261102-diagnostic-Restore-CovarianceDetail-sourceV-s2026092701-r01'
BASE_CODE = 'f29b31417ce98a280eac25ea2fa78e0c49a5f0ff'


def rotate_iq(x, quarter):
    if x.ndim != 3 or x.shape[1:] != (2, 256) or x.dtype != torch.float32 or quarter not in range(4):
        raise ValueError('FP32 IQ2x256 and a fixed quarter rotation required')
    if quarter == 0:return x
    if quarter == 1:return torch.stack((-x[:, 1], x[:, 0]), 1)
    if quarter == 2:return -x
    return torch.stack((x[:, 1], -x[:, 0]), 1)


def counterfactual_readout(model, received, quarter):
    return rotate_iq(model(rotate_iq(received, quarter)), (-quarter)%4)


def select_indices(rx, day):
    rx, day = np.asarray(rx), np.asarray(day)
    if rx.shape != (1800,) or day.shape != rx.shape or len(np.unique(rx)) != 5 or len(np.unique(day)) != 3:
        raise ValueError('Original five-RX/three-day source V required')
    indices = []
    for r in np.unique(rx):
        for d in np.unique(day):
            group = np.flatnonzero((rx == r)&(day == d))
            if len(group) != 120:raise ValueError('Original 120-record source V strata required')
            indices.extend(group[:32].tolist())
    return np.asarray(indices, np.int64)


def validate_baseline(done, identity):
    expected = dict(status='SOURCE_V_PREDICTIONS_COMPLETE', code_commit=BASE_CODE,
        methods=['received', 'physical', 'guided', 'detail'], views=list(SCENES), source_V_packets=1800,
        count=12600, TX_labels_read=0, target_inputs=0, optimizer_updates=0, model_state_updates=0)
    if any(done.get(k) != value for k, value in expected.items()) or done.get('F0_identity') != identity:
        raise ValueError('Exact frozen source V baseline/F0 required')


def metrics_numpy(iq, features, logits, reference, baseline_iq, baseline_prediction):
    target = reference['features'].astype(np.float64)
    clean = reference['clean'].astype(np.float64)
    prediction = logits.argmax(1)
    power = np.maximum((clean**2).mean((1, 2)), 1e-4)
    cosine = 1-(features*target).sum(1)/(np.maximum(np.linalg.norm(features, axis=1), 1e-8)*np.maximum(np.linalg.norm(target, axis=1), 1e-8))
    return dict(iq_nmse=((iq-clean)**2).mean((1, 2))/power,
        coordinate_equivariance_nmse=((iq-baseline_iq)**2).mean((1, 2))/power,
        feature_cosine_distance=cosine, feature_standardized_error=((features-target)**2/reference['scale']).mean(1),
        Fclean_decision_agreement=(prediction == reference['logits'].argmax(1)).astype(float),
        original_G_decision_agreement=(prediction == baseline_prediction).astype(float))


def worker(args):
    if not os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
        raise ValueError('One admitted UUID GPU required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter();out=args.root/'runs'/RUN;device='cuda:0'
    # Constructors precede historical F0 package activation.
    G,parent_identity=load_final(args.root,device)
    c=validate_source(read(args.inputs));F,identity=load_F0(c,device);valid_identity(identity)
    if identity != parent_identity:raise ValueError('Same original ground Tri-SIG-TIME SSL required')
    snapshots=[{k:v.detach().clone() for k,v in model.state_dict().items()} for model in [G,F]]
    old=args.root/'runs'/R07/'cache';roles=np.load(old/'train/roles.npy',allow_pickle=False)
    take=np.flatnonzero(roles=='L_s')[:2]
    a=np.load(old/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    try:source=tensor(a[take],device)
    finally:a._mmap.close()
    with torch.inference_mode():
        for quarter in range(4):
            y=counterfactual_readout(G,source,quarter);z=F.features(y);logits=F.classify_features(z)
            if y.shape!=(2,2,256) or z.shape!=(2,160) or logits.shape!=(2,6) or any(not torch.isfinite(v).all() for v in [y,z,logits]):
                raise ValueError('Actual source-only all-coordinate checkpoint smoke failed')
    write(out/'smoke.json',dict(status='PASS',actual_Ls=2,quarters=[0,1,2,3],actual_final30=True,
        actual_original_F0=True,G_rows=8,F0_rows=8,TX_labels_read=0,target_inputs=0,optimizer_updates=0))
    print('SOURCE_COORDINATE_REAL_SOURCE_SMOKE_PASS',flush=True)
    baseline=args.root/'runs'/BASE
    if read(baseline/'complete.json')['status']!='ARTIFACTS_COMPLETE':raise ValueError('Completed frozen source V baseline required')
    validate_baseline(read(baseline/'predictions_complete.json'),identity)
    with np.load(baseline/'references.npz',allow_pickle=False) as f:
        original={k:f[k].copy() for k in ['ids','rx','day','clean','features','logits','scale']}
    indices=select_indices(original['rx'],original['day'])
    ref={k:(v if k=='scale' else v[indices]) for k,v in original.items()}
    with (out/'references.npz').open('xb') as f:np.savez(f,**ref,source_indices=indices)
    masks=[('overall','ALL',np.ones(480,bool))]
    for dimension,labels in [('RX',ref['rx']),('day',ref['day']),
        ('RXday',np.asarray([str(r)+'/'+str(d) for r,d in zip(ref['rx'],ref['day'])]))]:
        masks.extend((dimension,str(value),labels==value) for value in np.unique(labels))
    records=[]
    with torch.inference_mode():
        for scene in SCENES:
            with np.load(baseline/(scene+'.npz'),allow_pickle=False) as f:
                if not np.array_equal(f['ids'],original['ids']):raise ValueError('Same original source V baseline order required')
                received=f['received__iq'][indices].copy()
                zero={k:f['detail__'+k][indices].astype(np.float64) for k in ['iq','features','logits']}
                original_prediction=f['detail__prediction'][indices].copy()
            if not np.array_equal(zero['logits'].argmax(1),original_prediction):raise ValueError('Exact cached original G decisions required')
            saved={'ids':ref['ids']}
            for quarter in range(4):
                if quarter==0:values=zero
                else:
                    parts={k:[] for k in ['iq','features','logits']}
                    for offset in range(0,480,32):
                        x=tensor(received[offset:offset+32],device)
                        y=counterfactual_readout(G,x,quarter);z=F.features(y);logits=F.classify_features(z)
                        if any(not torch.isfinite(v).all() for v in [y,z,logits]):raise ValueError('Nonfinite coordinate readout;preserve')
                        for k,v in dict(iq=y,features=z,logits=logits).items():parts[k].extend(v.cpu().tolist())
                    values={k:np.asarray(v,np.float64) for k,v in parts.items()}
                metrics=metrics_numpy(**values,reference=ref,baseline_iq=zero['iq'],baseline_prediction=original_prediction)
                saved.update({f'quarter{quarter}__'+k:v for k,v in dict(**values,**metrics).items()})
                before=original_prediction==ref['logits'].argmax(1);after=values['logits'].argmax(1)==ref['logits'].argmax(1)
                for dimension,stratum,mask in masks:
                    records.append(dict(scene=scene,quarter=quarter,degrees=90*quarter,dimension=dimension,stratum=stratum,count=int(mask.sum()),
                        metrics={k:float(v[mask].mean()) for k,v in metrics.items()},
                        Fclean_repairs_vs_original_G=int((~before&after&mask).sum()),Fclean_damage_vs_original_G=int((before&~after&mask).sum())))
                print('SOURCE_COORDINATE_READOUT_COMPLETE',scene,quarter,480,flush=True)
            with (out/(scene+'.npz')).open('xb') as f:np.savez(f,**saved)
    for model,before in zip([G,F],snapshots):check_F0_state(model,before)
    if len(records)!=672:raise ValueError('All fixed source coordinate cells/groups required')
    write(out/'predictions_complete.json',dict(status='SOURCE_COORDINATE_PREDICTIONS_COMPLETE',code_commit=args.commit,
        source_V_packets=480,source_RX_day_strata=15,per_stratum=32,views=list(SCENES),quarters=[0,1,2,3],
        new_G_rows=10080,new_F0_rows=10080,extra_Ls_smoke_G_rows=8,extra_Ls_smoke_F0_rows=8,cached_original_G_rows_reused=3360,
        cached_clean_F0_rows_reused=480,TX_labels_read=0,target_inputs=0,optimizer_updates=0,model_state_updates=0))
    write(out/'results.json',dict(status='SOURCE_COORDINATE_COMPLETE',run_id=RUN,code_commit=args.commit,
        G_code_commit='1fcbc3eb3cc16e7577ecc3cbedda99174b4c5cc6',F0_identity=identity,baseline_run=BASE,baseline_code_commit=BASE_CODE,
        records=records,source_V_packets=480,count_coordinate_readouts=13440,neural_new_G_rows=10088,neural_new_F0_rows=10088,
        cached_original_G_rows_reused=3360,cached_clean_F0_rows_reused=480,model_states_unchanged=True,
        TX_labels_read=0,OE_reads=0,novel_reads=0,target_inputs=0,optimizer_updates=0,new_training_epochs=0,
        output_ensemble=False,angle_selection=False,F0_phase_invariance_assumed=False,new_deployed_G_version=False,
        goal_complete=False,seconds=time.perf_counter()-started))


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty controller required')
    out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),command=sys.argv,CWD=os.getcwd(),code_commit=args.commit))
    choices=recovery_gpu_choice()
    while not choices:time.sleep(5);choices=recovery_gpu_choice()
    admission=choices[0];cmd=[sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),'--commit',args.commit,'--inputs',str(args.inputs)]
    with (out/'logs/worker.log').open('xb') as log:
        p=subprocess.Popen(cmd,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['CUDA_ID']),stdin=subprocess.DEVNULL,
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'submitted.json',dict(pid=p.pid,command=cmd,CWD=os.getcwd(),admission=admission));observed=False
    while p.poll() is None:
        if not observed and (out/'smoke.json').is_file():
            proc=Path('/proc')/str(p.pid)
            try:
                command=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode();cwd=os.readlink(proc/'cwd')
                cuda=[x.decode() for x in (proc/'environ').read_bytes().split(bytes([0])) if x.startswith(b'CUDA_VISIBLE_DEVICES=')]
                gpu=subprocess.check_output(['nvidia-smi','-i',admission['CUDA_ID'],'--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
                if MODULE+' worker ' in command and args.commit in command and cwd==os.getcwd() and cuda==['CUDA_VISIBLE_DEVICES='+admission['CUDA_ID']] and any(s.split(',')[0].strip()==admission['CUDA_ID'] and int(s.split(',')[1])==p.pid for s in gpu.splitlines()):
                    write(out/'runtime_worker_observation.json',dict(status='VERIFIED',pid=p.pid,CWD=cwd,command=command,CUDA=cuda,GPU=gpu,
                        source_smoke=read(out/'smoke.json'),log_bytes=(out/'logs/worker.log').stat().st_size,log_growth_from_zero=True,observer_PID=os.getpid()));observed=True
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
        time.sleep(.2)
    code=p.wait();write(out/'terminal.json',dict(pid=p.pid,exit_code=code))
    if code:raise RuntimeError('Coordinate source probe failed;preserve/no automatic retry')
    if read(out/'results.json')['status']!='SOURCE_COORDINATE_COMPLETE':raise ValueError('Complete fixed source matrix required')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',live_worker_observation=observed,optimizer_updates=0,target_inputs=0,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--inputs',type=Path,required=True);args=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Actual code identity required')
    if args.mode=='run':controller(args)
    else:worker(args)
