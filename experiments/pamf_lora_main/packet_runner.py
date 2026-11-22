"""One owner for the corrected LoRa input pilot and paired frozen predictions."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from .data import TARGET, VIEWS
from .packet_data import CHIRPS, TARGET_LIMIT
from .packet_models import EPOCHS, G_EPOCHS, METHODS, SEED, GuardedPacketDura, packet_tri, train_f, train_g
from .packet_physics import aligned_view, chirp_windows, make_source, make_target
from .runner import activate, available


RUN = '20261122-phase1-lora-packet-aligned-pilot-m5-r02'
READOUTS = {m: ('direct','dura') if m in ('packet_pl','packet_tri') else ('direct',) for m in METHODS}


def smoke(rt):
    root = Path(rt['run_root'])
    source = Path(rt['source_root'])
    packets = np.load(source/'L_s_packets.npy',mmap_mode='r',allow_pickle=False)[:3]
    clean = tensor(np.stack([chirp_windows(z) for z in packets]))
    received, meta = aligned_view(packets[0],VIEWS[1],'source-smoke',0,'smoke',trace=True)
    if clean.shape != (3,len(CHIRPS),2,256) or not np.isfinite(received).all():
        raise ValueError('Actual-source LoRa packet forward width or physical output differs')
    for name in ('single_sup','packet_tri'):
        model = packet_tri(name,'cpu').eval()
        a = root/f'{name}_init_smoke.pt'
        torch.save(model.state_dict(),a)
        model.load_state_dict(torch.load(a,weights_only=True),strict=True)
        with torch.no_grad():
            logit,_ = model(clean)
        if logit.shape != (3,10) or not torch.isfinite(logit).all():
            raise ValueError('Actual-source Tri checkpoint failed')
    dura = GuardedPacketDura().eval()
    path = root/'dura_init_smoke.pt'
    torch.save(dura.state_dict(),path)
    dura.load_state_dict(torch.load(path,weights_only=True),strict=True)
    with torch.no_grad():
        clean_out, clean_gate = dura(clean,with_gate=True)
        synthetic = torch.randn_like(clean)
        noisy_out,_ = dura(synthetic,with_gate=True)
    if not torch.equal(clean_out[clean_gate==0],clean[clean_gate==0]) or not torch.isfinite(noisy_out).all():
        raise ValueError('DURA identity or nonfinite smoke failure')
    write(root/'smoke.json',dict(status='PASS',source_packets=3,target_reads=0,TX_U_reads=0,
          fs=meta['fs_hz'],numerical_delay_compensated=meta['numerical_filter_latency_samples'],
          clean_gate=clean_gate.tolist()))


def load_models(root, source, method, device):
    folder = root/'F'/method
    cfg, expected = read(folder/'complete.json'), read(source/'contract.json')
    if (cfg['status']!='SOURCE_FROZEN' or cfg['epochs']!=EPOCHS
            or cfg['method']!=method or cfg['seed']!=SEED
            or cfg['source_contract']!=expected or cfg['target_read']):
        raise ValueError('Frozen LoRa packet F provenance differs')
    model = packet_tri(method,device).eval()
    model.load_state_dict(torch.load(folder/'model.pt',map_location=device,weights_only=True),strict=True)
    restorer = None
    if 'dura' in READOUTS[method]:
        g = read(root/'G'/'complete.json')
        if (g['status']!='SOURCE_FROZEN' or g['epochs']!=G_EPOCHS or g['seed']!=SEED or g['source_contract']!=expected
                or g['TX_labels_read'] or g['F_reads'] or g['target_read']):
            raise ValueError('Frozen LoRa packet DURA provenance differs')
        restorer = GuardedPacketDura().to(device).eval()
        restorer.load_state_dict(torch.load(root/'G'/'model.pt',map_location=device,weights_only=True),strict=True)
    return model, restorer


@torch.no_grad()
def infer(direct, restorer, x):
    truth_blind = {k:[] for k in ('direct','dura') if k=='direct' or restorer is not None}
    gate_stats = []
    for batch in x.split(32):
        result = direct(batch)[0]
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite packet identity logits')
        truth_blind['direct'].extend(result.argmax(1).tolist())
        if restorer is not None:
            restored, gate = restorer(batch,with_gate=True)
            logits = direct(restored)[0]
            if not torch.isfinite(logits).all():
                raise ValueError('Invalid restored packet identity logits')
            truth_blind['dura'].extend(logits.argmax(1).tolist())
            gate_stats.extend(gate.tolist())
    return {k:np.asarray(v,dtype=np.int16) for k,v in truth_blind.items()},gate_stats


def source_audit(root,source):
    device='cuda:0'
    path=root/'source_views'/'V.npy'
    x = np.load(path,mmap_mode='r',allow_pickle=False)
    with np.load(source/'V_metadata.npz',allow_pickle=False) as m:
        y, rx = m['y'], m['rx_i']
    output = {}
    for name in METHODS:
        f,g=load_models(root,source,name,device)
        output[name]={}
        for i, view in enumerate(VIEWS):
            xp=tensor(x[i],device)
            predictions, gate=infer(f,g,xp)
            by = {k:dict(correct=int((p==y).sum()), total=len(y),
                     worst_receiver=min(float((p[rx==r]==y[rx==r]).mean()) for r in range(5)))
                  for k,p in predictions.items()}
            if gate:
                by['gate_pass_through']=sum(g==0 for g in gate)
                by['dura_minus_direct']=int((predictions['dura']==y).sum()-(predictions['direct']==y).sum())
            # Counterfactual preserves physical draws, inserting the old
            # numerical delay and window boundary after correct alignment.
            if i:
                delayed = np.pad(x[i],((0,0),(0,0),(0,0),(24,0)),mode='reflect')[...,:256]
                wrong,_=infer(f,None,tensor(delayed,device))
                by['delay24_only_perturbation']=dict(correct=int((wrong['direct']==y).sum()),total=len(y))
            output[name][view]=by
    write(root/'source_audit.json',dict(status='SOURCE_AUDIT_COMPLETE',target_reads=0,records=output))


def predict(root,source,name):
    target=read(root/'target_views'/'complete.json')
    if target['status'] != 'TARGET_READY' or target['target_truth_read'] or target['per_receiver']!=TARGET_LIMIT:
        raise ValueError('Blind prepared target packet handle required')
    model,restore=load_models(root,source,name,'cuda:0')
    out=root/'predictions'/name
    out.mkdir(parents=True,exist_ok=False)
    counts = {}
    for rx in TARGET:
        data=np.load(root/'target_views'/f'{rx}.npy',mmap_mode='r',allow_pickle=False)
        for i, view in enumerate(VIEWS):
            result,gate=infer(model,restore,tensor(data[i],'cuda:0'))
            for mode,p in result.items():
                if len(p)!=TARGET_LIMIT:
                    raise ValueError('Incomplete packet prediction')
                np.save(out/f'{rx}__{view}__{mode}.npy',p)
            if gate:
                counts[f'{rx}/{view}']=sum(g==0 for g in gate)
    write(out/'complete.json',dict(status='PREDICTIONS_FROZEN',target_truth_read=False,
          seed=SEED,method=name,receivers=list(TARGET),per_receiver=TARGET_LIMIT,views=list(VIEWS),
          modes=list(READOUTS[name]),pass_through=counts))


def worker(rt, kind, name):
    activate(rt)
    root=Path(rt['run_root'])
    s=Path(rt['source_root'])
    if kind=='smoke':
        return smoke(rt)
    if kind=='source_views':
        return make_source(s,root/'source_views')
    if kind=='target_views':
        return make_target(Path(rt['target_root']),root/'target_views')
    if kind=='train_f':
        return train_f(s,root/'source_views',root/'F'/name,name)
    if kind=='train_g':
        return train_g(root/'source_views',root/'G',s)
    if kind=='source_audit':
        return source_audit(root,s)
    if kind=='predict':
        return predict(root,s,name)
    raise ValueError('Unknown LoRa packet job')


def start(rt, kind, name, gpu=None):
    root=Path(rt['run_root'])
    cfg=root/'configs'/f'{kind}-{name}.json'
    fields=('run_root','source_root','frozen_sources')
    if kind=='target_views':
        fields+=('target_root',)
    write(cfg,{k:rt[k] for k in fields})
    cmd=[sys.executable,'-u','-m',__spec__.name,kind,str(cfg),name]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    p=root/'logs'/f'{kind}-{name}.log'
    p.parent.mkdir(exist_ok=True)
    with p.open('x') as f:
        c=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
    print('PACKET_LAUNCHED',kind,name,c.pid,gpu,flush=True)
    return c


def schedule(rt,jobs,launch=start):
    pending=list(jobs)
    active={}
    max_workers=rt.get('max_workers',len(rt['allowed_gpus']))
    while pending or active:
        for gpu,(p,k,n) in list(active.items()):
            code=p.poll()
            if code is None:
                continue
            if code:
                raise RuntimeError(f'{k} {n} exited {code}')
            print('PACKET_COMPLETED',k,n,flush=True)
            del active[gpu]
        if pending and len(active)<max_workers:
            free=set(available(rt['allowed_gpus'],rt.get('max_compute_processes',2)))
            for gpu in rt['allowed_gpus']:
                if len(active)>=max_workers:
                    break
                if gpu in free and gpu not in active and pending:
                    k,n=pending.pop(0)
                    active[gpu]=(launch(rt,k,n,gpu),k,n)
        if pending or active:
            time.sleep(15)


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state=root/'state.json'
    try:
        write(state,dict(status='SOURCE_SMOKE'))
        if start(rt,'smoke','packet').wait():
            raise RuntimeError('Source no-query smoke failed')
        write(state,dict(status='PREPARING_SOURCE_PRACTICAL'),exclusive=False)
        if start(rt,'source_views','aligned').wait():
            raise RuntimeError('Corrected practical source construction failed')
        write(state,dict(status='SOURCE_TRAINING'),exclusive=False)
        schedule(rt,[('train_f','packet_tri'),('train_g','bounded'),('train_f','packet_pl'),
                     ('train_f','packet_sup'),('train_f','single_sup')])
        write(state,dict(status='SOURCE_AUDIT'),exclusive=False)
        schedule(rt,[('source_audit','paired')])
        write(state,dict(status='PREPARING_BLIND_TARGET'),exclusive=False)
        if start(rt,'target_views','aligned').wait():
            raise RuntimeError('Corrected practical target construction failed')
        write(state,dict(status='PREDICTING'),exclusive=False)
        schedule(rt,[('predict',m) for m in METHODS])
        write(state,dict(status='PREDICTIONS_FROZEN',target_truth_read=False),exclusive=False)
    except Exception as exc:
        write(state,dict(status='FAILED',error=str(exc)),exclusive=False)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('kind')
    p.add_argument('runtime',type=Path)
    p.add_argument('name',nargs='?',default='')
    a=p.parse_args()
    rt=read(a.runtime)
    if a.kind=='dispatch':
        dispatch(rt)
    else:
        worker(rt,a.kind,a.name)
