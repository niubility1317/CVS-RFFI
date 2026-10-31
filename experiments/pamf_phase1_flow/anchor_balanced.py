"""Phase1 frozen-DURA anchored flow with source gradient balancing."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import experiment as core
from .anchor_flow import FixedAnchoredFlow,source_objective
from .anchor_source_smoke import DURA,frozen_dura
from .gradient_balance import balance_gradients

RUN='20261101-phase1-dura-anchored-balanced-E30-s2026092701-r01'
KL_WEIGHT=.01
TRUST_WEIGHT=.05
MAX_AUX_RATIO=1.
read,write=core.read,core.write


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def step(field,f,dura,received,clean,optimizer,noise):
    if f.training or dura.training or any(p.requires_grad for p in f.parameters()) or any(p.requires_grad for p in dura.parameters()):
        raise ValueError('Both pretrained networks must be frozen')
    with torch.no_grad():anchor=dura(received)
    if anchor.shape!=received.shape or not torch.isfinite(anchor).all():raise ValueError('Finite frozen DURA anchor required')
    _,terms=source_objective(field,f,received,anchor,clean,noise,kl_weight=KL_WEIGHT,trust_weight=TRUST_WEIGHT)
    waveform=(terms['iq']+TRUST_WEIGHT*terms['deviation']).mean()
    semantic=(terms['feature']+.25*terms['agreement']+KL_WEIGHT*terms['semantic']).mean()
    if not torch.isfinite(waveform) or not torch.isfinite(semantic):raise FloatingPointError('Nonfinite source objective')
    params=tuple(field.parameters());optimizer.zero_grad(set_to_none=True)
    reference=torch.autograd.grad(waveform,params,retain_graph=True)
    auxiliary=torch.autograd.grad(semantic,params)
    balanced,stats=balance_gradients(reference,auxiliary,max_aux_ratio=MAX_AUX_RATIO)
    for p,g in zip(params,balanced):p.grad=g
    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
    return terms,dict(stats,gradient_norm=float(norm),waveform=float(waveform.detach()),semantic=float(semantic.detach()))


def save(path,field,commit,epoch,dura_hash):
    payload=dict(schema='phase1_dura_anchor_balanced_v1',run=RUN,commit=commit,epoch=epoch,
        updates=epoch*((core.COUNT+core.BATCH-1)//core.BATCH),F_run=core.F_RUN,F_row=core.F_ROW,
        DURA_checkpoint=DURA,DURA_sha256=dura_hash,KL_weight=KL_WEIGHT,trust_weight=TRUST_WEIGHT,
        max_aux_ratio=MAX_AUX_RATIO,model=field.state_dict())
    with Path(path).open('xb') as f:torch.save(payload,f)


def load(path,device,commit,dura_hash,*,epoch=core.EPOCHS):
    p=torch.load(path,map_location='cpu',weights_only=False)
    expected=dict(schema='phase1_dura_anchor_balanced_v1',run=RUN,commit=commit,epoch=epoch,
        updates=epoch*((core.COUNT+core.BATCH-1)//core.BATCH),F_run=core.F_RUN,F_row=core.F_ROW,
        DURA_checkpoint=DURA,DURA_sha256=dura_hash,KL_weight=KL_WEIGHT,trust_weight=TRUST_WEIGHT,max_aux_ratio=MAX_AUX_RATIO)
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('Exact source-trained anchored checkpoint required')
    field=core.make_model();field.load_state_dict(p['model'],strict=True)
    return FixedAnchoredFlow(field).to(device).eval().requires_grad_(False)


def snapshot(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def unchanged(model,before):
    return not any(p.grad is not None for p in model.parameters()) and all(torch.equal(v.detach().cpu(),before[k]) for k,v in model.state_dict().items())


def smoke(root,out,commit):
    f,contract=core.teacher(root,'cpu');cache,clean,roles=core.data(root,contract);dura=frozen_dura(root,'cpu')
    with np.load(cache/'epochs/01/groups.npz',allow_pickle=False) as z:scene=z['scene']
    ix=np.flatnonzero((roles=='L_s')&(scene!='clean'))[:2]
    if len(ix)!=2:raise ValueError('Two source L_s received rows required')
    raw=np.load(cache/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    x=core.tensor(raw[ix],'cpu');target=core.tensor(clean[ix],'cpu')
    before_f=snapshot(f);before_d=snapshot(dura);field=core.make_model();optimizer=torch.optim.AdamW(field.parameters(),lr=.0003,weight_decay=.0001)
    with torch.no_grad():
        baseline=dura(x);initial=FixedAnchoredFlow(field)(x,baseline)
        torch.testing.assert_close(initial,baseline,rtol=1e-6,atol=3e-7)
    noise=torch.randn(x.shape,generator=torch.Generator().manual_seed(2026110112))
    terms,stats=step(field,f,dura,x,target,optimizer,noise)
    if not unchanged(f,before_f) or not unchanged(dura,before_d) or stats['ref_norm']<=0 or not torch.isfinite(terms['iq']).all():
        raise ValueError('Real frozen networks and nonzero source gradients required')
    digest=sha(root/DURA);save(out/'smoke.pt',field,commit,0,digest);restored=load(out/'smoke.pt','cpu',commit,digest,epoch=0)
    with torch.no_grad():
        a=restored(x,baseline);b=FixedAnchoredFlow(field)(x,baseline);torch.testing.assert_close(a,b,rtol=0,atol=0)
        logits=f.classify_features(f.features(a))
    if logits.shape!=(2,6) or not torch.isfinite(logits).all():raise ValueError('Finite actual frozen pipeline required')
    write(out/'smoke.json',dict(status='PASS',F_run=core.F_RUN,F_row=core.F_ROW,DURA_checkpoint=DURA,DURA_sha256=digest,
        source_rows=2,source_role='L_s',F_frozen=True,DURA_frozen=True,initial_anchor_max_abs=float((initial-baseline).abs().max()),
        grad_stats=stats,checkpoint_reload=True,TX_labels_read=0,V_reads=0,test_IQ_reads=0))


def gpu_smoke(root,out,commit):
    """One actual batch at the registered training size, source only."""
    f,contract=core.teacher(root,'cuda:0');cache,clean,roles=core.data(root,contract);dura=frozen_dura(root,'cuda:0')
    with np.load(cache/'epochs/01/groups.npz',allow_pickle=False) as z:scene=z['scene']
    ix=np.flatnonzero((roles=='L_s')&(scene!='clean'))[:core.BATCH]
    if len(ix)!=core.BATCH:raise ValueError('Full source L_s smoke batch required')
    raw=np.load(cache/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    x=core.tensor(raw[ix],'cuda:0');target=core.tensor(clean[ix],'cuda:0')
    before_f=snapshot(f);before_d=snapshot(dura);field=core.make_model().cuda();optimizer=torch.optim.AdamW(field.parameters(),lr=.0003,weight_decay=.0001)
    torch.manual_seed(2026110112);torch.cuda.manual_seed_all(2026110112);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
    _,stats=step(field,f,dura,x,target,optimizer,torch.randn_like(x));torch.cuda.synchronize();seconds=time.perf_counter()-started
    if not unchanged(f,before_f) or not unchanged(dura,before_d) or stats['ref_norm']<=0:raise ValueError('Full-batch frozen source smoke failed')
    write(out/'gpu_smoke.json',dict(status='PASS',source_rows=len(ix),source_role='L_s',F_frozen=True,DURA_frozen=True,
        elapsed_seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),grad_stats=stats,
        TX_labels_read=0,V_reads=0,test_IQ_reads=0))


def train(root,out,commit,*,device='cuda:0',epochs=core.EPOCHS):
    if epochs!=core.EPOCHS:raise ValueError('Full registered source budget required')
    f,contract=core.teacher(root,device);cache,clean,roles=core.data(root,contract);dura=frozen_dura(root,device)
    before_f=snapshot(f);before_d=snapshot(dura);digest=sha(root/DURA)
    field=core.make_model().to(device).train();optimizer=torch.optim.AdamW(field.parameters(),lr=.0003,weight_decay=.0001)
    rng=np.random.default_rng(2026110104);torch.manual_seed(2026110105)
    if device.startswith('cuda'):torch.cuda.manual_seed_all(2026110105)
    row=out/'flow';row.mkdir();updates=0;started=time.perf_counter()
    with (row/'training.jsonl').open('x',encoding='utf8') as log:
        for epoch in range(1,core.EPOCHS+1):
            view=(epoch-1)%20+1;received=np.load(cache/'epochs'/f'{view:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
            if received.shape!=clean.shape or received.dtype!=np.float32:raise ValueError('Phase1 source received view required')
            order=rng.permutation(core.COUNT);clean_count=core.COUNT//4;totals=dict(iq=0.,semantic=0.,feature=0.,deviation=0.)
            projections=0;scales=0.;batches=0
            for offset in range(0,core.COUNT,core.BATCH):
                ix=order[offset:offset+core.BATCH];target=core.tensor(clean[ix],device);x=core.tensor(received[ix],device)
                n=max(0,min(len(ix),clean_count-offset))
                if n:x[:n]=target[:n]
                for group in optimizer.param_groups:group['lr']=core.lr(updates)
                terms,stats=step(field,f,dura,x,target,optimizer,torch.randn_like(x));updates+=1;batches+=1
                projections+=int(stats['projected']);scales+=stats['scale']
                for key in totals:totals[key]+=float(terms[key].detach().sum())
            record=dict(epoch=epoch,view=view,updates=updates,losses={k:v/core.COUNT for k,v in totals.items()},
                projected_fraction=projections/batches,mean_aux_scale=scales/batches,seconds=time.perf_counter()-started)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush();print(json.dumps(record),flush=True)
    if updates!=core.STEPS or not unchanged(f,before_f) or not unchanged(dura,before_d) or sha(root/DURA)!=digest:
        raise ValueError('Fixed source budget or pretrained identity changed')
    save(row/'final.pt',field,commit,core.EPOCHS,digest)
    write(row/'training_complete.json',dict(status='SOURCE_TRAINED',epoch=core.EPOCHS,updates=updates,F_run=core.F_RUN,F_row=core.F_ROW,
        F_frozen=True,DURA_frozen=True,DURA_sha256=digest,TX_labels_read=0,V_reads=0,test_IQ_reads=0,source_roles=['L_s','U_s'],
        parameters=sum(p.numel() for p in field.parameters()),gradient_balance='project opposing auxiliary and cap norm at waveform norm before AdamW'))


@torch.no_grad()
def predict(root,out,commit):
    done=read(out/'flow/training_complete.json');digest=sha(root/DURA)
    if (done.get('status'),done.get('epoch'),done.get('updates'),done.get('F_frozen'),done.get('DURA_frozen'),done.get('DURA_sha256'),done.get('test_IQ_reads'))!=('SOURCE_TRAINED',core.EPOCHS,core.STEPS,True,True,digest,0):
        raise ValueError('Complete fixed anchored source training required')
    f,_=core.teacher(root,'cuda:0');dura=frozen_dura(root,'cuda:0');field=load(out/'flow/final.pt','cuda:0',commit,digest)
    folder,ids=core.reference(root);dst=out/'flow/test_prediction';dst.mkdir();np.save(dst/'physical_ids.npy',ids,allow_pickle=False)
    for view in core.VIEWS:
        values=np.load(folder/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        if values.shape!=(len(ids),2,256) or values.dtype!=np.float32:raise ValueError('Complete Phase1 received IQ required')
        raw=np.lib.format.open_memmap(dst/(view+'.raw.npy'),mode='w+',dtype=np.int64,shape=(len(ids),))
        restored=np.lib.format.open_memmap(dst/(view+'.restored.npy'),mode='w+',dtype=np.int64,shape=(len(ids),))
        for offset in range(0,len(ids),core.BATCH):
            x=core.tensor(values[offset:offset+core.BATCH],'cuda:0');anchor=dura(x);y=field(x,anchor)
            a=f.classify_features(f.features(x));b=f.classify_features(f.features(y))
            if not torch.isfinite(anchor).all() or not torch.isfinite(y).all() or not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Finite frozen predictions required')
            raw[offset:offset+len(x)]=np.from_dlpack(a.argmax(1).cpu());restored[offset:offset+len(x)]=np.from_dlpack(b.argmax(1).cpu())
        raw.flush();restored.flush();print(json.dumps(dict(view=view,count=len(ids))),flush=True)
    write(dst/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(core.VIEWS),F_run=core.F_RUN,F_row=core.F_ROW,
        commit=commit,TX_truth_reads=0,model_updates=0))


def score(root,out,commit):
    core.RUN=RUN;return core.score(root,out,commit)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('smoke','score'));parser.add_argument('--root',type=Path,required=True);parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    {'smoke':smoke,'score':score}[args.mode](args.root,args.root/'runs'/RUN,args.commit)


if __name__=='__main__':main()
