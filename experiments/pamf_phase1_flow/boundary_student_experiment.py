"""Phase1 source-only amortized boundary student and truth-last evaluation."""
import argparse
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch

from . import anchor_balanced as prior, experiment as core
from .boundary_student import BoundedBoundaryStudent, distillation_loss, make_source_oracle
from .dura_source_trisig_diagnostic import PARENT
from .resume_state import load_state, save_state

RUN = os.environ.get('PHASE1_STUDENT_RUN','20261101-phase1-boundary-student-E30-s2026092701-r02')
if not re.fullmatch(r'\d{8}-phase1-boundary-student-E30-s2026092701-r\d{2}',RUN):
    raise ValueError('Valid registered boundary-student run ID required')
EPSILON, ORACLE_STEPS, HARD_WEIGHT = 0.05, 12, 8.0
read, write = core.read, core.write


def make_student(device):
    return BoundedBoundaryStudent(core.make_model(),epsilon=EPSILON).to(device)


def save(path, student, commit, epoch, dura_hash):
    payload=dict(schema='phase1_boundary_student_v1',run=RUN,commit=commit,epoch=epoch,
        updates=epoch*((core.COUNT+core.BATCH-1)//core.BATCH),F_run=core.F_RUN,F_row=core.F_ROW,
        DURA_checkpoint=prior.DURA,DURA_sha256=dura_hash,epsilon=EPSILON,
        oracle_steps=ORACLE_STEPS,hard_weight=HARD_WEIGHT,model=student.state_dict())
    with Path(path).open('xb') as f:torch.save(payload,f)


def load(path,device,commit,dura_hash,*,epoch=core.EPOCHS):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    expected=dict(schema='phase1_boundary_student_v1',run=RUN,commit=commit,epoch=epoch,
        updates=epoch*((core.COUNT+core.BATCH-1)//core.BATCH),F_run=core.F_RUN,F_row=core.F_ROW,
        DURA_checkpoint=prior.DURA,DURA_sha256=dura_hash,epsilon=EPSILON,
        oracle_steps=ORACLE_STEPS,hard_weight=HARD_WEIGHT)
    if any(payload.get(k)!=v for k,v in expected.items()):raise ValueError('Exact fixed source student required')
    student=make_student(device);student.load_state_dict(payload['model'],strict=True)
    return student.eval().requires_grad_(False)


def step(student,teacher,dura,received,clean,optimizer):
    if teacher.training or dura.training or any(p.requires_grad for p in teacher.parameters()) or any(p.requires_grad for p in dura.parameters()):
        raise ValueError('Frozen teacher and DURA required')
    with torch.no_grad():anchor=dura(received)
    if not torch.isfinite(anchor).all():raise FloatingPointError('Finite DURA anchor required')
    oracle,wrong=make_source_oracle(teacher,anchor,clean,epsilon=EPSILON,steps=ORACLE_STEPS)
    candidate=student(received,anchor)
    loss=distillation_loss(candidate,oracle,anchor,wrong,hard_weight=HARD_WEIGHT)
    if not torch.isfinite(loss):raise FloatingPointError('Finite source student loss required')
    optimizer.zero_grad(set_to_none=True);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    with torch.no_grad():
        target=teacher.classify_features(teacher.features(clean)).argmax(1)
        reached=teacher.classify_features(teacher.features(oracle)).argmax(1)==target
    return dict(loss=float(loss.detach()),wrong=int(wrong.sum()),oracle_reached=int((wrong&reached).sum()),
                grad_norm=float(norm))


def smoke_indices(root,teacher,cache,clean,roles,device,n):
    parent=root/'runs'/PARENT
    with np.load(parent/'DURA_default-1.npz',allow_pickle=False) as z:
        saved={k:z[k] for k in ('source_positions','roles','clean','native')}
    ix=saved['source_positions']
    if not np.array_equal(saved['roles'],roles[ix]) or not np.array_equal(saved['clean'],clean[ix]):
        raise ValueError('Fixed paired source smoke records required')
    with torch.no_grad():
        target=teacher.classify_features(teacher.features(core.tensor(saved['clean'][:128],device))).argmax(1)
        baseline=teacher.classify_features(teacher.features(core.tensor(saved['native'][:128],device))).argmax(1)
    wrong=(target!=baseline).nonzero(as_tuple=True)[0].tolist()
    if n==2:
        if len(wrong)<2:raise ValueError('Two real source teacher disagreements required')
        return ix[wrong[:2]]
    if n==128:
        if not wrong:raise ValueError('Real full-batch source teacher disagreement required')
        return ix[:128]
    raise ValueError('Unsupported fixed smoke size')


def smoke_impl(root,out,commit,*,device,n,filename):
    teacher,contract=core.teacher(root,device);cache,clean,roles=core.data(root,contract);dura=prior.frozen_dura(root,device)
    ix=smoke_indices(root,teacher,cache,clean,roles,device,n)
    raw=np.load(cache/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    received=core.tensor(raw[ix],device);target=core.tensor(clean[ix],device)
    before_f=prior.snapshot(teacher);before_d=prior.snapshot(dura)
    student=make_student(device).train();optimizer=torch.optim.AdamW(student.parameters(),lr=.0003,weight_decay=.0001)
    with torch.no_grad():
        base=dura(received)
        if not torch.equal(student(received,base),base):raise ValueError('Zero student must preserve anchor exactly')
    stats=step(student,teacher,dura,received,target,optimizer)
    if stats['wrong']<1 or stats['grad_norm']<=0 or not prior.unchanged(teacher,before_f) or not prior.unchanged(dura,before_d):
        raise ValueError('Real source errors and finite gradient with frozen networks required')
    digest=prior.sha(root/prior.DURA)
    checkpoint=out/(filename+'.pt');save(checkpoint,student,commit,0,digest)
    reloaded=load(checkpoint,device,commit,digest,epoch=0)
    with torch.no_grad():
        expected=student.eval().requires_grad_(False)(received,base);actual=reloaded(received,base)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        logits=teacher.classify_features(teacher.features(actual))
    if logits.shape!=(n,6) or not torch.isfinite(logits).all():raise ValueError('Finite true pipeline output required')
    write(out/(filename+'.json'),dict(status='PASS',source_rows=n,source_role='L_s',
        source_IQ_materialized=512,materialized_roles={'L_s':128,'U_s':384},stats=stats,
        F_run=core.F_RUN,F_row=core.F_ROW,F_frozen=True,DURA_frozen=True,DURA_sha256=digest,
        checkpoint_reload=True,TX_labels_read=0,V_reads=0,test_IQ_reads=0))


def smoke(root,out,commit):return smoke_impl(root,out,commit,device='cpu',n=2,filename='smoke')


def gpu_smoke(root,out,commit):return smoke_impl(root,out,commit,device='cuda:0',n=128,filename='gpu_smoke')


def train(root,out,commit,*,device='cuda:0',resume_checkpoint=None):
    teacher,contract=core.teacher(root,device);cache,clean,roles=core.data(root,contract);dura=prior.frozen_dura(root,device)
    before_f=prior.snapshot(teacher);before_d=prior.snapshot(dura);digest=prior.sha(root/prior.DURA)
    student=make_student(device).train();optimizer=torch.optim.AdamW(student.parameters(),lr=.0003,weight_decay=.0001)
    rng=np.random.default_rng(2026110104);torch.manual_seed(2026110105)
    if device.startswith('cuda'):torch.cuda.manual_seed_all(2026110105)
    row=out/'flow';row.mkdir();(row/'checkpoints').mkdir();updates=0;start_epoch=0;started=time.perf_counter()
    contract=dict(method='phase1_boundary_student',commit=commit,F_run=core.F_RUN,F_row=core.F_ROW,
        DURA_sha256=digest,count=core.COUNT,batch=core.BATCH,epochs=core.EPOCHS,epsilon=EPSILON,
        oracle_steps=ORACLE_STEPS,hard_weight=HARD_WEIGHT)
    if resume_checkpoint is not None:
        start_epoch,updates=load_state(resume_checkpoint,student,optimizer,rng,contract=contract,device=device)
        if start_epoch>core.EPOCHS or updates!=start_epoch*((core.COUNT+core.BATCH-1)//core.BATCH):
            raise ValueError('Fixed-budget epoch-boundary checkpoint required')
    with (row/'training.jsonl').open('x',encoding='utf8') as log:
        for epoch in range(start_epoch+1,core.EPOCHS+1):
            view=(epoch-1)%20+1;received=np.load(cache/'epochs'/f'{view:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
            if received.shape!=clean.shape or received.dtype!=np.float32:raise ValueError('Phase1 received source view required')
            order=rng.permutation(core.COUNT);clean_count=core.COUNT//4;loss_sum=0.;wrong_count=0;reached_count=0
            for offset in range(0,core.COUNT,core.BATCH):
                ix=order[offset:offset+core.BATCH];target=core.tensor(clean[ix],device);x=core.tensor(received[ix],device)
                n=max(0,min(len(ix),clean_count-offset))
                if n:x[:n]=target[:n]
                for group in optimizer.param_groups:group['lr']=core.lr(updates)
                stats=step(student,teacher,dura,x,target,optimizer);updates+=1
                loss_sum+=stats['loss']*len(ix);wrong_count+=stats['wrong'];reached_count+=stats['oracle_reached']
            if not prior.unchanged(teacher,before_f) or not prior.unchanged(dura,before_d) or prior.sha(root/prior.DURA)!=digest:
                raise ValueError('Frozen networks must be unchanged before checkpointing')
            record=dict(epoch=epoch,view=view,updates=updates,mean_batch_weighted_loss=loss_sum/core.COUNT,
                        teacher_disagreements=wrong_count,oracle_reached=reached_count,seconds=time.perf_counter()-started)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
            save_state(row/'checkpoints'/f'epoch-{epoch:02d}.pt',student,optimizer,rng,epoch=epoch,updates=updates,contract=contract,device=device)
            print(json.dumps(record),flush=True)
    if updates!=core.STEPS or not prior.unchanged(teacher,before_f) or not prior.unchanged(dura,before_d) or prior.sha(root/prior.DURA)!=digest:
        raise ValueError('Fixed budget or frozen model identity changed')
    save(row/'final.pt',student,commit,core.EPOCHS,digest)
    write(row/'training_complete.json',dict(status='SOURCE_TRAINED',epoch=core.EPOCHS,updates=updates,
        F_run=core.F_RUN,F_row=core.F_ROW,F_frozen=True,DURA_frozen=True,DURA_sha256=digest,
        TX_labels_read=0,V_reads=0,test_IQ_reads=0,source_roles=['L_s','U_s'],parameters=sum(p.numel() for p in student.parameters()),
        epsilon=EPSILON,oracle_steps=ORACLE_STEPS,hard_weight=HARD_WEIGHT,
        resumed_from=str(resume_checkpoint) if resume_checkpoint else None,resumed_after_epoch=start_epoch))


@torch.no_grad()
def predict(root,out,commit):
    done=read(out/'flow/training_complete.json');digest=prior.sha(root/prior.DURA)
    if (done.get('status'),done.get('epoch'),done.get('updates'),done.get('F_frozen'),done.get('DURA_frozen'),done.get('DURA_sha256'),done.get('test_IQ_reads'))!=('SOURCE_TRAINED',core.EPOCHS,core.STEPS,True,True,digest,0):
        raise ValueError('Complete frozen source student required')
    teacher,_=core.teacher(root,'cuda:0');dura=prior.frozen_dura(root,'cuda:0');student=load(out/'flow/final.pt','cuda:0',commit,digest)
    folder,ids=core.reference(root);dst=out/'flow/test_prediction';dst.mkdir();np.save(dst/'physical_ids.npy',ids,allow_pickle=False)
    for view in core.VIEWS:
        values=np.load(folder/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        if values.shape!=(len(ids),2,256) or values.dtype!=np.float32:raise ValueError('Complete Phase1 received IQ required')
        raw=np.lib.format.open_memmap(dst/(view+'.raw.npy'),mode='w+',dtype=np.int64,shape=(len(ids),))
        restored=np.lib.format.open_memmap(dst/(view+'.restored.npy'),mode='w+',dtype=np.int64,shape=(len(ids),))
        for offset in range(0,len(ids),core.BATCH):
            x=core.tensor(values[offset:offset+core.BATCH],'cuda:0');anchor=dura(x);y=student(x,anchor)
            a=teacher.classify_features(teacher.features(x));b=teacher.classify_features(teacher.features(y))
            if not torch.isfinite(y).all() or not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Finite frozen predictions required')
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
