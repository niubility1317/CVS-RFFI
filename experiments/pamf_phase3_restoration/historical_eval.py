"""Full historical F50 seven-view regression; separate from Phase3 G10 claims."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from .checkpoint import RUNS,load_model
from .cache import tensor
from .cost_probe import write
from .source_pair import read

VIEWS=('clean','practical_high','practical_mid','practical_low_suburban',
       'practical_high_urban','practical_mid_urban','practical_low_urban')
N=168000
F_RUN='20261016-phase1-F-supcon-e50-clean-manysig-m9-r02'


def paths(root,arm):
    return dict(output=root/'runs'/RUNS[arm],reader=root/'runs'/F_RUN/'CROSS_RX-s2026092701/source',
        source_cache=root/'data_caches/phase1_manysig_reference_20261019/source_cache',
        received=root/'runs/20261019-phase1-restoration-formal-frozen-residual-s2026092701-r05/reference',
        raw=root/'runs/20261019-phase1-restoration-formal-FcrossRX50-s2026092701-r09/prediction',
        dura=root/'runs/20261021-phase1-restoration-stage2-graph-s2026092701-r01/stage2_residual_graph/target_confirmation/native_map_prior/prediction',
        truth=root/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json')


def original(p):
    m=read(p['received']/'manifest.json')
    if any(m.get(k)!=v for k,v in dict(status='FORMAL_REFERENCE_READY',count=N,views=list(VIEWS),
            original_IQ_bytes_reused=True,TX_truth_reads=0).items()):
        raise ValueError('Original fixed historical received handles required')
    ids=np.load(p['received']/'physical_ids.npy',allow_pickle=False)
    if ids.shape!=(N,) or len(set(ids.tolist()))!=N:raise ValueError('Original historical population required')
    return ids


@torch.no_grad()
def predict(args):
    started=time.perf_counter();p=paths(args.root,args.arm);ids=original(p)
    trained=read(p['output']/'training/training_complete.json')
    if (trained.get('status'),trained.get('epoch'),trained.get('updates'),trained.get('training_F_access'),
            trained.get('target_reads'))!=('SOURCE_TRAINED',30,7410,args.arm=='F0',0):
        raise ValueError('Complete frozen source candidate before every test required')
    # Fresh process, original 47c runtime namespace; never bootstrap Phase3 F0 here.
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    F=load_reader(p['reader'],p['source_cache'],'cuda:0').eval().requires_grad_(False)
    G=load_model(p['output']/'training/final.pt','cuda:0',arm=args.arm,commit=args.commit)
    states=[(m,{k:v.detach().clone() for k,v in m.state_dict().items()}) for m in [G,F]]
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    gen=torch.Generator().manual_seed(2026092701);probe=torch.randn(2,2,256,generator=gen).to('cuda:0')
    smoke=F(G(probe))
    if smoke.shape!=(2,6) or not torch.isfinite(smoke).all():raise ValueError('Actual TF/F50 default-head smoke failed')
    out=p['output']/'historical_prediction';out.mkdir()
    write(out/'smoke.json',dict(status='PASS',actual_TF_F50_checkpoints=True,real_data_reads=0))
    timings={}
    for view in VIEWS:
        received=np.load(p['received']/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        if received.shape!=(N,2,256) or received.dtype!=np.float32:raise ValueError('Original full FP32 view required')
        iq=np.lib.format.open_memmap(out/(view+'.iq.npy'),mode='w+',dtype=np.float32,shape=(N,2,256))
        logits=np.lib.format.open_memmap(out/(view+'.logits.npy'),mode='w+',dtype=np.float32,shape=(N,6))
        begin=time.perf_counter()
        try:
            for offset in range(0,N,128):
                x=tensor(received[offset:offset+128],'cuda:0');restored=G(x);scores=F(restored)
                if not torch.isfinite(restored).all() or not torch.isfinite(scores).all():raise ValueError('Nonfinite frozen prediction')
                iq[offset:offset+len(x)]=np.from_dlpack(restored.cpu()).copy()
                logits[offset:offset+len(x)]=np.from_dlpack(scores.cpu()).copy()
        finally:
            received._mmap.close();iq.flush();logits.flush();iq._mmap.close();logits._mmap.close()
        timings[view]=time.perf_counter()-begin;print('TF_HISTORICAL_VIEW_COMPLETE',view,N,flush=True)
    for model,before in states:
        if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Frozen model changed')
    with (out/'physical_ids.npy').open('xb') as f:np.save(f,ids,allow_pickle=False)
    write(out/'complete.json',dict(status='TF_HISTORICAL_PREDICTIONS_COMPLETE',run_id=RUNS[args.arm],
        arm=args.arm,code_commit=args.commit,count=N,views=list(VIEWS),F_run=F_RUN,
        TX_truth_reads=0,state_updates=0,head_fit_updates=0,all_predictions_before_truth=True,
        seconds=time.perf_counter()-started,per_view_seconds=timings,
        scope='Historical Phase1 received views/F50 regression;not Phase3 G10 or Phase3 F0'))


def _baseline(folder,ids,readout):
    done=read(folder/'complete.json')
    required=dict(status='PREDICTIONS_COMPLETE',count=N,views=list(VIEWS),TX_truth_reads=0,
                  state_updates=0,F_run=F_RUN,F_seed=2026092701,original_default_head=True)
    if any(done.get(k)!=v for k,v in required.items()):raise ValueError('Complete same-F50 controls required before truth')
    if not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Control IDs differ')
    arrays={}
    for view in VIEWS:
        a=np.load(folder/(view+'.'+readout+'.logits.npy'),mmap_mode='r',allow_pickle=False)
        if a.shape!=(N,6) or not np.isfinite(a).all():raise ValueError('Incomplete control keeps truth closed')
        arrays[view]=a
    return arrays


def statistics(y,a,b):
    matrix=np.zeros((6,6),np.int64);np.add.at(matrix,(y,a),1)
    denominator=matrix.sum(0)+matrix.sum(1);f1=np.divide(2*np.diag(matrix),denominator,out=np.zeros(6),where=denominator>0)
    return dict(count=len(y),correct=int((a==y).sum()),accuracy=100*float((a==y).mean()),
        macro_F1=100*float(f1.mean()),wrong_to_correct=int(((b!=y)&(a==y)).sum()),
        correct_to_wrong=int(((b==y)&(a!=y)).sum()),net_corrections=int((a==y).sum()-(b==y).sum()))


def score(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU-only scorer required')
    p=paths(args.root,args.arm);ids=original(p);out=p['output']/'historical_prediction';done=read(out/'complete.json')
    required=dict(status='TF_HISTORICAL_PREDICTIONS_COMPLETE',run_id=RUNS[args.arm],arm=args.arm,
        code_commit=args.commit,count=N,views=list(VIEWS),TX_truth_reads=0,state_updates=0,
        head_fit_updates=0,all_predictions_before_truth=True)
    if any(done.get(k)!=v for k,v in required.items()):raise ValueError('All seven predictions must precede truth')
    if not np.array_equal(np.load(out/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Prediction IDs differ')
    arrays={}
    for view in VIEWS:
        a={key:np.load(out/(view+'.'+key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ['iq','logits']}
        if any(a[key].shape!=shape or a[key].dtype!=np.float32 or not np.isfinite(a[key]).all()
               for key,shape in [('iq',(N,2,256)),('logits',(N,6))]):
            raise ValueError('All finite complete arrays must precede truth')
        arrays[view]=a
    raw=_baseline(p['raw'],ids,'identity');dura=_baseline(p['dura'],ids,'restored')
    truth=read(p['truth'])  # First private truth read, after all views and controls.
    y=np.asarray([truth[str(pid)]['label'] for pid in ids],np.int64)
    rx=np.asarray([str(truth[str(pid)]['receiver']) for pid in ids])
    day=np.asarray([str(truth[str(pid)].get('day','unavailable')) for pid in ids])
    clean=np.load(p['received']/'clean.npy',mmap_mode='r',allow_pickle=False)
    raw_clean=100*float((raw['clean'].argmax(1)==y).mean());overall={};records=[]
    for view in VIEWS:
        a=arrays[view]['logits'].argmax(1);b=raw[view].argmax(1);d=dura[view].argmax(1)
        metrics=statistics(y,a,b);metrics['vs_DURA']=statistics(y,a,d)
        metrics['raw_clean_gap_pp']=raw_clean-metrics['accuracy']
        metrics['raw_clean_accuracy_retention_percent']=100*metrics['accuracy']/raw_clean
        error=0.
        for start in range(0,N,2048):
            z=clean[start:start+2048].astype(np.float64);v=arrays[view]['iq'][start:start+2048].astype(np.float64)
            error+=float(np.sum(np.mean((v-z)**2,axis=(1,2))/np.maximum(np.mean(z*z,axis=(1,2)),1e-4)))
        metrics['paired_iq_nmse']=error/N;overall[view]=metrics
        for key,groups in [('TX',y),('RX',rx),('day',day)]:
            for group in np.unique(groups):
                mask=groups==group;records.append(dict(view=view,dimension=key,value=str(group),**statistics(y[mask],a[mask],b[mask])))
    write(p['output']/'historical_summary.json',dict(status='ANALYZED',run_id=RUNS[args.arm],
        code_commit=args.commit,count=N,F_run=F_RUN,overall=overall,records=records,raw_clean_accuracy=raw_clean,
        six_scene_gap3pp=all(overall[v]['raw_clean_gap_pp']<=3 for v in VIEWS[1:]),
        clean_preserved=overall['clean']['accuracy']>=raw_clean,all_predictions_before_truth=True,
        target_feedback=False,scope='Historical Phase1 received/F50;not current Phase3 F0/G10'))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['predict','score']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--arm',choices=tuple(RUNS),required=True);p.add_argument('--commit',required=True);args=p.parse_args()
    if args.mode=='predict':predict(args)
    else:score(args)


if __name__=='__main__':main()
