"""Phase1 learned FIR with strongly regularized residual targets."""
import argparse
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
from . import anchor_balanced as prior, experiment as core
from .filter_student import FilterStudent, project_delta
from .source_fir_oracle import fit_paired_fir
from .resume_state import load_state, save_state

RUN=os.environ.get('PHASE1_FILTER_RESIDUAL_RUN','20261101-phase1-filter-residual-E30-s2026092701-r01')
if not re.fullmatch(r'\d{8}-phase1-filter-residual-E30-s2026092701-r\d{2}',RUN):
    raise ValueError('Valid registered filter-residual run ID required')
EPSILON=0.2
TAPS=9
RIDGE=1.0
ORACLE_WEIGHT=0.7
CLEAN_WEIGHT=0.3
read,write=core.read,core.write


def make_model(device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026092701)
        model=FilterStudent(taps=TAPS,epsilon=EPSILON)
    return model.to(device)



def save(path,model,commit,epoch,dura_hash):
    payload=dict(schema='phase1_filter_residual_v1',run=RUN,commit=commit,epoch=epoch,
        updates=epoch*((core.COUNT+core.BATCH-1)//core.BATCH),F_run=core.F_RUN,F_row=core.F_ROW,
        DURA_checkpoint=prior.DURA,DURA_sha256=dura_hash,epsilon=EPSILON,taps=TAPS,ridge=RIDGE,oracle_weight=ORACLE_WEIGHT,clean_weight=CLEAN_WEIGHT,
        model=model.state_dict())
    with Path(path).open('xb') as f:torch.save(payload,f)


def load(path,device,commit,dura_hash,*,epoch=core.EPOCHS):
    p=torch.load(path,map_location='cpu',weights_only=False)
    expected=dict(schema='phase1_filter_residual_v1',run=RUN,commit=commit,epoch=epoch,
        updates=epoch*((core.COUNT+core.BATCH-1)//core.BATCH),F_run=core.F_RUN,F_row=core.F_ROW,
        DURA_checkpoint=prior.DURA,DURA_sha256=dura_hash,epsilon=EPSILON,taps=TAPS,ridge=RIDGE,oracle_weight=ORACLE_WEIGHT,clean_weight=CLEAN_WEIGHT)
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('Exact fixed source model required')
    model=make_model(device);model.load_state_dict(p['model'],strict=True)
    return model.eval().requires_grad_(False)


def make_source_target(anchor,clean):
    fitted=anchor+fit_paired_fir(anchor,clean-anchor,taps=TAPS,widely_linear=True,ridge=RIDGE)
    return project_delta(anchor,fitted,epsilon=EPSILON)


def step(model,teacher,dura,received,clean,optimizer):
    if teacher.training or dura.training or any(p.requires_grad for p in teacher.parameters()) or any(p.requires_grad for p in dura.parameters()):
        raise ValueError('Frozen teacher and DURA required')
    with torch.no_grad():
        anchor=dura(received);features=teacher.features(anchor);baseline=teacher.classify_features(features)
        oracle=make_source_target(anchor,clean)
    candidate=model(received,anchor,features,baseline)
    energy=clean.square().mean((1,2)).clamp_min(1e-4)
    oracle_loss=((candidate-oracle).square().mean((1,2))/energy).mean()
    clean_loss=((candidate-clean).square().mean((1,2))/energy).mean()
    loss=ORACLE_WEIGHT*oracle_loss+CLEAN_WEIGHT*clean_loss
    if not torch.isfinite(loss):raise FloatingPointError('Finite paired-source objective required')
    optimizer.zero_grad(set_to_none=True);loss.backward()
    head_grad=model.output.weight.grad.detach().norm()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    return dict(loss=float(loss.detach()),oracle=float(oracle_loss.detach()),clean=float(clean_loss.detach()),
        grad_norm=float(norm),head_grad=float(head_grad))


def smoke_impl(root,out,commit,*,device,n,filename):
    teacher,contract=core.teacher(root,device);cache,clean,roles=core.data(root,contract);dura=prior.frozen_dura(root,device)
    ix=np.flatnonzero(roles=='L_s')[:n]
    if len(ix)!=n:raise ValueError('Fixed Phase1 L_s source smoke rows required')
    raw=np.load(cache/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    x=core.tensor(raw[ix],device);target=core.tensor(clean[ix],device)
    before_f=prior.snapshot(teacher);before_d=prior.snapshot(dura)
    model=make_model(device).train();optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    with torch.no_grad():
        base=dura(x);feat=teacher.features(base);logits=teacher.classify_features(feat)
        if not torch.equal(model(x,base,feat,logits),base):raise ValueError('Zero model must preserve anchor exactly')
    stats=step(model,teacher,dura,x,target,optimizer)
    if stats['grad_norm']<=0 or stats['head_grad']<=0 or not prior.unchanged(teacher,before_f) or not prior.unchanged(dura,before_d):
        raise ValueError('Real source gradients and frozen networks required')
    digest=prior.sha(root/prior.DURA);checkpoint=out/(filename+'.pt');save(checkpoint,model,commit,0,digest)
    restored=load(checkpoint,device,commit,digest,epoch=0)
    with torch.no_grad():
        a=model.eval().requires_grad_(False)(x,base,feat,logits);b=restored(x,base,feat,logits)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        scores=teacher.classify_features(teacher.features(b))
    if scores.shape!=(n,6) or not torch.isfinite(scores).all():raise ValueError('Finite true pipeline required')
    write(out/(filename+'.json'),dict(status='PASS',source_rows=n,source_role='L_s',TX_labels_read=0,stats=stats,
        F_run=core.F_RUN,F_row=core.F_ROW,F_frozen=True,DURA_frozen=True,DURA_sha256=digest,
        checkpoint_reload=True,V_reads=0,test_IQ_reads=0))


def smoke(root,out,commit):return smoke_impl(root,out,commit,device='cpu',n=2,filename='smoke')


def gpu_smoke(root,out,commit):return smoke_impl(root,out,commit,device='cuda:0',n=128,filename='gpu_smoke')


def train(root,out,commit,*,device='cuda:0',resume_checkpoint=None):
    teacher,contract=core.teacher(root,device);cache,clean,roles=core.data(root,contract);dura=prior.frozen_dura(root,device)
    before_f=prior.snapshot(teacher);before_d=prior.snapshot(dura);digest=prior.sha(root/prior.DURA)
    model=make_model(device).train();optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    rng=np.random.default_rng(2026110104);torch.manual_seed(2026110105)
    if device.startswith('cuda'):torch.cuda.manual_seed_all(2026110105)
    row=out/'flow';row.mkdir();(row/'checkpoints').mkdir();updates=0;start_epoch=0;started=time.perf_counter()
    contract=dict(method='phase1_filter_residual',commit=commit,F_run=core.F_RUN,F_row=core.F_ROW,
        DURA_sha256=digest,count=core.COUNT,batch=core.BATCH,epochs=core.EPOCHS,epsilon=EPSILON,
        taps=TAPS,ridge=RIDGE,oracle_weight=ORACLE_WEIGHT,clean_weight=CLEAN_WEIGHT)
    if resume_checkpoint is not None:
        start_epoch,updates=load_state(resume_checkpoint,model,optimizer,rng,contract=contract,device=device)
        if start_epoch>core.EPOCHS or updates!=start_epoch*((core.COUNT+core.BATCH-1)//core.BATCH):
            raise ValueError('Fixed-budget epoch-boundary checkpoint required')
    with (row/'training.jsonl').open('x',encoding='utf8') as log:
        for epoch in range(start_epoch+1,core.EPOCHS+1):
            view=(epoch-1)%20+1;received=np.load(cache/'epochs'/f'{view:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
            if received.shape!=clean.shape or received.dtype!=np.float32:raise ValueError('Phase1 received source view required')
            order=rng.permutation(core.COUNT);clean_count=core.COUNT//4
            sums={k:0. for k in ('loss','oracle','clean')}
            for offset in range(0,core.COUNT,core.BATCH):
                ix=order[offset:offset+core.BATCH];target=core.tensor(clean[ix],device);x=core.tensor(received[ix],device)
                n=max(0,min(len(ix),clean_count-offset))
                if n:x[:n]=target[:n]
                for group in optimizer.param_groups:group['lr']=core.lr(updates)
                stats=step(model,teacher,dura,x,target,optimizer);updates+=1
                for key in sums:sums[key]+=stats[key]*len(ix)
            if not prior.unchanged(teacher,before_f) or not prior.unchanged(dura,before_d) or prior.sha(root/prior.DURA)!=digest:
                raise ValueError('Frozen networks must be unchanged before checkpointing')
            record=dict(epoch=epoch,view=view,updates=updates,losses={k:v/core.COUNT for k,v in sums.items()},
                        seconds=time.perf_counter()-started)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
            save_state(row/'checkpoints'/f'epoch-{epoch:02d}.pt',model,optimizer,rng,epoch=epoch,updates=updates,contract=contract,device=device)
            print(json.dumps(record),flush=True)
    if updates!=core.STEPS or not prior.unchanged(teacher,before_f) or not prior.unchanged(dura,before_d) or prior.sha(root/prior.DURA)!=digest:
        raise ValueError('Fixed budget or frozen model identity changed')
    save(row/'final.pt',model,commit,core.EPOCHS,digest)
    write(row/'training_complete.json',dict(status='SOURCE_TRAINED',epoch=core.EPOCHS,updates=updates,
        F_run=core.F_RUN,F_row=core.F_ROW,F_frozen=True,DURA_frozen=True,DURA_sha256=digest,
        TX_labels_read=0,V_reads=0,test_IQ_reads=0,source_roles=['L_s','U_s'],
        parameters=sum(p.numel() for p in model.parameters()),epsilon=EPSILON,taps=TAPS,ridge=RIDGE,
        resumed_from=str(resume_checkpoint) if resume_checkpoint else None,resumed_after_epoch=start_epoch))


@torch.no_grad()
def predict(root,out,commit):
    done=read(out/'flow/training_complete.json');digest=prior.sha(root/prior.DURA)
    if (done.get('status'),done.get('epoch'),done.get('updates'),done.get('F_frozen'),done.get('DURA_frozen'),done.get('DURA_sha256'),done.get('TX_labels_read'),done.get('test_IQ_reads'))!=('SOURCE_TRAINED',core.EPOCHS,core.STEPS,True,True,digest,0,0):
        raise ValueError('Complete frozen source model required')
    teacher,_=core.teacher(root,'cuda:0');dura=prior.frozen_dura(root,'cuda:0');model=load(out/'flow/final.pt','cuda:0',commit,digest)
    folder,ids=core.reference(root);dst=out/'flow/test_prediction';dst.mkdir();np.save(dst/'physical_ids.npy',ids,allow_pickle=False)
    for view in core.VIEWS:
        values=np.load(folder/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        if values.shape!=(len(ids),2,256) or values.dtype!=np.float32:raise ValueError('Complete Phase1 received IQ required')
        raw=np.lib.format.open_memmap(dst/(view+'.raw.npy'),mode='w+',dtype=np.int64,shape=(len(ids),))
        restored=np.lib.format.open_memmap(dst/(view+'.restored.npy'),mode='w+',dtype=np.int64,shape=(len(ids),))
        for offset in range(0,len(ids),core.BATCH):
            x=core.tensor(values[offset:offset+core.BATCH],'cuda:0');anchor=dura(x)
            feat=teacher.features(anchor);baseline=teacher.classify_features(feat);y=model(x,anchor,feat,baseline)
            a=teacher.classify_features(teacher.features(x));b=teacher.classify_features(teacher.features(y))
            if not torch.isfinite(y).all() or not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Finite frozen predictions required')
            raw[offset:offset+len(x)]=np.from_dlpack(a.argmax(1).cpu());restored[offset:offset+len(x)]=np.from_dlpack(b.argmax(1).cpu())
        raw.flush();restored.flush();print(json.dumps(dict(view=view,count=len(ids))),flush=True)
    write(dst/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(core.VIEWS),F_run=core.F_RUN,F_row=core.F_ROW,
        commit=commit,TX_truth_reads=0,model_updates=0))


def score(root,out,commit):core.RUN=RUN;return core.score(root,out,commit)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('smoke','score'));parser.add_argument('--root',type=Path,required=True);parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    {'smoke':smoke,'score':score}[args.mode](args.root,args.root/'runs'/RUN,args.commit)


if __name__=='__main__':main()
