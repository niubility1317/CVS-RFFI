"""Independent Phase1 source-V diagnostic, predictions sealed before truth."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow import anchor_source_smoke as prior
from experiments.cric_dura_comparison.source_iq import aligned_source_iq

FIXED_RUN='20261102-phase1-complex-mask-source-s2026092701-r02'
FIXED_COMMIT='9160406b331bad9495e005afc898335d3c14214e'
FIXED_RELEASE='20261102-phase1-complex-mask-source-s2026092701-r02-release'
VIEWS=('clean','practical_low_suburban','practical_mid_urban','practical_low_urban')
SAMPLE_SEED=2026110561
CHANNEL_SEED=2026110562
HARDWARE_SEED=2026110563


def load_fixed(root):
    folder=root/'releases'/FIXED_RELEASE/'experiments'/'phase1_complex_multires'
    if (root/'releases'/FIXED_RELEASE/'release_commit.txt').read_text(encoding='ascii').strip()!=FIXED_COMMIT:
        raise ValueError('Exact frozen mask release required')
    sys.path.insert(0,str(folder))
    from phase1_complex_multires_predict import load_fixed, apply_mask
    model,alpha=load_fixed(root/'runs'/FIXED_RUN/'source',FIXED_COMMIT,'cpu')
    return model,alpha,apply_mask


def source_paths(root,contract):
    complete=core.read(root/'runs'/core.F_RUN/'F'/core.F_ROW/'source'/'complete.json')
    source=Path(complete['config']['source_cache'])
    expected=np.asarray(contract['role_ids']['V'])
    if len(expected)!=27000 or len(set(map(str,expected)))!=len(expected):raise ValueError('Exact source V IDs required')
    if set(map(str,expected)) & (set(map(str,contract['role_ids']['L_s']))|set(map(str,contract['role_ids']['U_s']))):raise ValueError('Source V overlaps training')
    return source,expected


def score_counts(truth,base,candidate):
    old=base==truth;new=candidate==truth
    return dict(count=int(len(truth)),baseline_correct=int(old.sum()),candidate_correct=int(new.sum()),repairs=int((~old&new).sum()),damages=int((old&~new).sum()))


@torch.no_grad()
def smoke(root,out):
    teacher,contract=core.teacher(root,'cpu');_,clean,roles=core.data(root,contract)
    dura=prior.frozen_dura(root,'cpu');model,alpha,apply_mask=load_fixed(root)
    idx=np.flatnonzero(roles=='L_s')[:4];x=core.tensor(clean[idx],'cpu');a=dura(x);y,g=apply_mask(model,alpha,a,x)
    for z in (x,a,y):
        scores=teacher.classify_features(teacher.features(z))
        if scores.shape!=(len(idx),6) or not torch.isfinite(scores).all():raise ValueError('Finite frozen source reader required')
    if any(p.requires_grad for p in teacher.parameters()) or any(p.requires_grad for p in dura.parameters()) or any(p.requires_grad for p in model.parameters()):raise ValueError('Models must be frozen')
    core.write(out/'source_smoke.json',dict(status='PASS',source_role='L_s',count=len(idx),TX_labels_read=0,V_reads=0,target_reads=0))


@torch.no_grad()
def predict(root,out,commit):
    if core.read(out/'source_smoke.json').get('status')!='PASS':raise ValueError('Source smoke required')
    teacher,contract=core.teacher(root,'cpu');dura=prior.frozen_dura(root,'cpu');model,alpha,apply_mask=load_fixed(root)
    source,expected=source_paths(root,contract)
    clean,binding=aligned_source_iq(source,'V',expected)
    with np.load(source/'V_metadata.npz',allow_pickle=False) as z:
        stored=z['ids'].copy();rx=z['rx_i'].copy();day=z['day_i'].copy();eq=z['eq_i'].copy()
    if len(stored)!=len(expected) or len(set(map(str,stored)))!=len(stored) or set(map(str,stored))!=set(map(str,expected)):raise ValueError('V metadata IDs differ')
    lookup={str(v):i for i,v in enumerate(stored)};order=np.asarray([lookup[str(v)] for v in expected],dtype=np.int64)
    rx,day,eq=rx[order],day[order],eq[order]
    rng=np.random.default_rng(SAMPLE_SEED);selected=np.sort(rng.choice(len(expected),size=4096,replace=False))
    ids=expected[selected];rx=rx[selected];day=day[selected];eq=eq[selected];clean=np.asarray(clean[selected],dtype=np.float32)
    if not np.all(eq==1):raise ValueError('Expected equalized Phase1 ground samples')
    dst=out/'prediction';dst.mkdir(exist_ok=False)
    np.save(dst/'physical_ids.npy',ids,allow_pickle=False);np.save(dst/'rx.npy',rx,allow_pickle=False);np.save(dst/'day.npy',day,allow_pickle=False)
    # These channel imports are available after core.teacher activates the pinned runtime.
    from leo_practical.channel import Config,receiver_for_session,stable_seed,normalize_input
    from leo_practical.residual import ResidualChannel
    gates={}
    for view in VIEWS:
        if view=='clean':received=clean
        else:
            cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=view,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect');rows=[]
            for i,pid in enumerate(ids):
                session='rx%d:day%d:eq%d'%(int(rx[i]),int(day[i]),int(eq[i]));hardware=receiver_for_session(cfg,HARDWARE_SEED,session)
                signal,_=normalize_input(clean[i,0].astype(np.float64)+1j*clean[i,1].astype(np.float64))
                value,meta=ResidualChannel(cfg,stable_seed(CHANNEL_SEED,'source-v-fixed-restoration-2026110562',view,str(pid)),hardware).process(signal)
                if not np.isfinite(value).all() or meta['channel_equalization_applied']:raise ValueError('Invalid source channel')
                rows.append(np.stack((value.real,value.imag)).astype(np.float32))
            received=np.stack(rows)
        arrays={name:np.empty(len(ids),dtype=np.int64) for name in ('raw','dura','candidate')};active=0
        for start in range(0,len(ids),32):
            x=core.tensor(received[start:start+32],'cpu');a=dura(x);y,gate=apply_mask(model,alpha,a,x);active+=int(gate.sum())
            for name,wave in (('raw',x),('dura',a),('candidate',y)):
                logits=teacher.classify_features(teacher.features(wave))
                if not torch.isfinite(logits).all():raise ValueError('Finite source V prediction required')
                arrays[name][start:start+len(x)]=np.from_dlpack(logits.argmax(1).cpu().contiguous())
        for name,values in arrays.items():np.save(dst/(view+'.'+name+'.npy'),values,allow_pickle=False)
        gates[view]=active;print('SOURCE_V_VIEW_FIXED',view,len(ids),flush=True)
    core.write(dst/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(VIEWS),arms=['raw','dura','candidate'],commit=commit,fixed_model_commit=FIXED_COMMIT,sample_seed=SAMPLE_SEED,channel_seed=CHANNEL_SEED,hardware_seed=HARDWARE_SEED,gate_counts=gates,source_V_labels_read=0,U_s_labels_read=0,target_reads=0,model_updates=0,binding=binding))
    print('ALL_SOURCE_V_PREDICTIONS_SEALED_BEFORE_TRUTH',flush=True)


def score(root,out,commit):
    dst=out/'prediction';manifest=core.read(dst/'complete.json')
    expected=dict(status='PREDICTIONS_COMPLETE',count=4096,views=list(VIEWS),arms=['raw','dura','candidate'],commit=commit,fixed_model_commit=FIXED_COMMIT,source_V_labels_read=0,U_s_labels_read=0,target_reads=0,model_updates=0)
    if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError('Sealed source V predictions required')
    ids=np.load(dst/'physical_ids.npy',allow_pickle=False);rx=np.load(dst/'rx.npy',allow_pickle=False);day=np.load(dst/'day.npy',allow_pickle=False)
    if ids.shape!=(4096,) or len(set(map(str,ids)))!=len(ids) or rx.shape!=ids.shape or day.shape!=ids.shape:raise ValueError('Complete unique source V IDs required')
    predictions={}
    for view in VIEWS:
        predictions[view]={}
        for name in ('raw','dura','candidate'):
            arr=np.load(dst/(view+'.'+name+'.npy'),allow_pickle=False)
            if arr.shape!=ids.shape or arr.dtype!=np.int64 or np.any((arr<0)|(arr>=6)):raise ValueError('Complete class decisions required')
            predictions[view][name]=arr
    # Source V labels are opened only after every decision array is validated.
    f=core.read(root/'runs'/core.F_RUN/'F'/core.F_ROW/'source'/'complete.json');source=Path(f['config']['source_cache'])
    with np.load(source/'V_metadata.npz',allow_pickle=False) as z:
        stored=z['ids'].copy();truth=z['y'].copy();stored_rx=z['rx_i'].copy();stored_day=z['day_i'].copy()
    if len(set(map(str,stored)))!=len(stored) or truth.shape!=stored.shape:raise ValueError('Valid source V truth required')
    lookup={str(v):i for i,v in enumerate(stored)};pos=np.asarray([lookup[str(v)] for v in ids],dtype=np.int64);truth=truth[pos]
    if not np.array_equal(rx,stored_rx[pos]) or not np.array_equal(day,stored_day[pos]) or np.any((truth<0)|(truth>=6)):raise ValueError('V truth metadata mismatch')
    result={};groups=[]
    for view in VIEWS:
        p=predictions[view];result[view]=dict(raw_correct=int((p['raw']==truth).sum()),**score_counts(truth,p['dura'],p['candidate']))
        for name,vector in (('TX',truth),('RX',rx),('day',day)):
            for value in np.unique(vector):
                mask=vector==value;groups.append(dict(view=view,dimension=name,value=str(value),**score_counts(truth[mask],p['dura'][mask],p['candidate'][mask])))
    core.write(out/'summary.json',dict(status='ANALYZED_SOURCE_V_ONLY',count=len(ids),results=result,groups=groups,source_V_labels_read_after_predictions=True,U_s_labels_read=0,target_reads=0,model_updates=0,limitations='Sampled Phase1 source validation; F checkpoint may have used V for model selection, but V was not used in candidate training or source IQ selection. Simulated satellite is not validated real satellite.'))
    print('SOURCE_V_INDEPENDENT_SCORE_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke','predict','score'));p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    {'smoke':smoke,'predict':predict,'score':score}[a.mode](a.root,a.out,a.commit)


if __name__=='__main__':main()
