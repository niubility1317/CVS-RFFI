"""Phase1 class-conditional source training and truth-last scoring."""
import argparse
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import anchor_balanced as prior, experiment as core
from .dura_source_trisig_diagnostic import PARENT
from .feature_conditioned import source_targets
from .class_hypothesis import ClassHypothesis, select_verified, source_oracle
from .labeled_source_probe import align_labeled_positions
from .resume_state import load_state, save_state

RUN=os.environ.get('PHASE1_HYPOTHESIS_RUN','20261101-phase1-class-hypothesis-E30-s2026092701-r01')
if not re.fullmatch(r'\d{8}-phase1-class-hypothesis-E30-s2026092701-r\d{2}',RUN):
    raise ValueError('Valid registered feature-conditioned run ID required')
EPSILON=0.05
THRESHOLD=0.95
HARD_WEIGHT=8.0
POSTERIOR_WEIGHT=1.0
ORACLE_WEIGHT=1.0
read,write=core.read,core.write


def make_model(device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026092701)
        model=ClassHypothesis(epsilon=EPSILON)
    return model.to(device)


def source_labels(root,cache,clean,roles,contract):
    complete=read(root/'runs'/core.F_RUN/'F'/core.F_ROW/'source/complete.json')
    source=Path(complete['config']['source_cache'])
    with np.load(source/'L_s_metadata.npz',allow_pickle=False) as data:
        ids=data['ids'];truth=data['y']
    physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
    positions=align_labeled_positions(roles,physical,ids)
    if list(ids)!=contract['role_ids']['L_s'] or truth.shape!=(6300,) or np.any(truth<0) or np.any(truth>=6):
        raise ValueError('Exact legitimate L_s class labels required')
    from experiments.cric_dura_comparison.source_iq import aligned_source_iq
    source_iq,_=aligned_source_iq(source,'L_s',ids)
    if not np.allclose(source_iq,clean[positions],rtol=1e-5,atol=1e-6):
        raise ValueError('L_s source IQ binding differs')
    labels=np.full(len(roles),-1,dtype=np.int64);labels[positions]=truth
    if (labels[roles=='L_s']<0).any() or (labels[roles=='U_s']!=-1).any():
        raise ValueError('No U_s labels may be read or materialized')
    return labels


def save(path,model,commit,epoch,dura_hash):
    payload=dict(schema='phase1_class_hypothesis_v1',run=RUN,commit=commit,epoch=epoch,
        updates=epoch*((core.COUNT+core.BATCH-1)//core.BATCH),F_run=core.F_RUN,F_row=core.F_ROW,
        DURA_checkpoint=prior.DURA,DURA_sha256=dura_hash,epsilon=EPSILON,threshold=THRESHOLD,
        hard_weight=HARD_WEIGHT,posterior_weight=POSTERIOR_WEIGHT,oracle_weight=ORACLE_WEIGHT,oracle_steps=12,
        model=model.state_dict())
    with Path(path).open('xb') as f:torch.save(payload,f)


def load(path,device,commit,dura_hash,*,epoch=core.EPOCHS):
    p=torch.load(path,map_location='cpu',weights_only=False)
    expected=dict(schema='phase1_class_hypothesis_v1',run=RUN,commit=commit,epoch=epoch,
        updates=epoch*((core.COUNT+core.BATCH-1)//core.BATCH),F_run=core.F_RUN,F_row=core.F_ROW,
        DURA_checkpoint=prior.DURA,DURA_sha256=dura_hash,epsilon=EPSILON,threshold=THRESHOLD,
        hard_weight=HARD_WEIGHT,posterior_weight=POSTERIOR_WEIGHT,oracle_weight=ORACLE_WEIGHT,oracle_steps=12)
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('Exact fixed source model required')
    model=make_model(device);model.load_state_dict(p['model'],strict=True)
    return model.eval().requires_grad_(False)


def step(model,teacher,dura,received,clean,labeled,optimizer):
    if teacher.training or dura.training or any(p.requires_grad for p in teacher.parameters()) or any(p.requires_grad for p in dura.parameters()):
        raise ValueError('Frozen teacher and DURA required')
    with torch.no_grad():
        anchor=dura(received);features=teacher.features(anchor);baseline=teacher.classify_features(features)
        clean_logits=teacher.classify_features(teacher.features(clean))
        target,valid=source_targets(labeled,clean_logits,threshold=THRESHOLD)
        before=baseline.argmax(1)
    oracle,reached=source_oracle(teacher,anchor,target,valid,epsilon=EPSILON,steps=12)
    q,candidate=model(received,anchor,features,baseline,target)
    weights=torch.where(before!=target,HARD_WEIGHT,1.).to(q)*valid.to(q)
    posterior=(F.cross_entropy(q,target,reduction='none')*weights).sum()/weights.sum().clamp_min(1)
    # Unreachable or invalid examples retain the anchor rather than distilling a failed target.
    desired=torch.where(reached[:,None,None],oracle,anchor)
    distill_weights=torch.where(reached,HARD_WEIGHT,1.).to(q)
    energy=anchor.square().mean((1,2)).clamp_min(1e-8)
    per_row=(candidate-desired).square().mean((1,2))/energy
    distill=(per_row*distill_weights).sum()/distill_weights.sum()
    loss=POSTERIOR_WEIGHT*posterior+ORACLE_WEIGHT*distill
    if not torch.isfinite(loss):raise FloatingPointError('Finite source objective required')
    optimizer.zero_grad(set_to_none=True);loss.backward()
    posterior_grad=model.posterior.weight.grad.detach().norm()
    proposal_grad=model.output.weight.grad.detach().norm()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    return dict(loss=float(loss.detach()),posterior=float(posterior.detach()),distill=float(distill.detach()),
        labeled=int((labeled>=0).sum()),pseudo_valid=int(((labeled<0)&valid).sum()),
        valid_wrong=int((valid&(before!=target)).sum()),oracle_reached=int(reached.sum()),grad_norm=float(norm),
        posterior_grad=float(posterior_grad),proposal_grad=float(proposal_grad))


def smoke_indices(root,teacher,labels,device,n):
    with np.load(root/'runs'/PARENT/'DURA_default-1.npz',allow_pickle=False) as z:
        saved={k:z[k] for k in ('source_positions','roles','native')}
    ix=saved['source_positions'][:128]
    if not np.all(saved['roles'][:128]=='L_s') or np.any(labels[ix]<0):raise ValueError('Fixed L_s smoke rows required')
    with torch.no_grad():
        prediction=teacher.classify_features(teacher.features(core.tensor(saved['native'][:128],device))).argmax(1)
        truth=torch.tensor(labels[ix].tolist(),dtype=torch.long,device=device)
    wrong=(prediction!=truth).nonzero(as_tuple=True)[0].tolist()
    if n==2:
        if len(wrong)<2:raise ValueError('Two true source-label errors required')
        return ix[wrong[:2]]
    if n==128:
        if not wrong:raise ValueError('Real full-batch source-label errors required')
        return ix
    raise ValueError('Unsupported smoke size')


def smoke_impl(root,out,commit,*,device,n,filename):
    teacher,contract=core.teacher(root,device);cache,clean,roles=core.data(root,contract);dura=prior.frozen_dura(root,device)
    labels=source_labels(root,cache,clean,roles,contract);ix=smoke_indices(root,teacher,labels,device,n)
    raw=np.load(cache/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    x=core.tensor(raw[ix],device);target=core.tensor(clean[ix],device);y=torch.tensor(labels[ix].tolist(),dtype=torch.long,device=device)
    before_f=prior.snapshot(teacher);before_d=prior.snapshot(dura)
    model=make_model(device).train();optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    with torch.no_grad():
        base=dura(x);feat=teacher.features(base);logits=teacher.classify_features(feat)
        q,initial=model(x,base,feat,logits)
        if not torch.equal(initial,base) or not torch.equal(q,logits):raise ValueError('Zero model must preserve anchor exactly')
    stats=step(model,teacher,dura,x,target,y,optimizer)
    if stats['labeled']!=n or stats['valid_wrong']<1 or stats['grad_norm']<=0 or stats['oracle_reached']<1 or stats['posterior_grad']<=0 or stats['proposal_grad']<=0 or not prior.unchanged(teacher,before_f) or not prior.unchanged(dura,before_d):
        raise ValueError('Real L_s errors, nonzero gradients and frozen networks required')
    digest=prior.sha(root/prior.DURA);checkpoint=out/(filename+'.pt');save(checkpoint,model,commit,0,digest)
    restored=load(checkpoint,device,commit,digest,epoch=0)
    with torch.no_grad():
        qa,a=model.eval().requires_grad_(False)(x,base,feat,logits);qb,b=restored(x,base,feat,logits)
        torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(qa,qb,rtol=0,atol=0)
        scores=teacher.classify_features(teacher.features(b))
    if scores.shape!=(n,6) or not torch.isfinite(scores).all():raise ValueError('Finite true pipeline required')
    write(out/(filename+'.json'),dict(status='PASS',source_rows=n,source_role='L_s',L_s_labels_materialized=6300,
        U_s_labels_read=0,source_archive_IQ_materialized=512,archive_roles={'L_s':128,'U_s':384},stats=stats,
        F_run=core.F_RUN,F_row=core.F_ROW,F_frozen=True,DURA_frozen=True,DURA_sha256=digest,
        checkpoint_reload=True,V_reads=0,test_IQ_reads=0))


def smoke(root,out,commit):return smoke_impl(root,out,commit,device='cpu',n=2,filename='smoke')


def gpu_smoke(root,out,commit):return smoke_impl(root,out,commit,device='cuda:0',n=128,filename='gpu_smoke')


def train(root,out,commit,*,device='cuda:0',resume_checkpoint=None):
    teacher,contract=core.teacher(root,device);cache,clean,roles=core.data(root,contract);dura=prior.frozen_dura(root,device)
    labels=source_labels(root,cache,clean,roles,contract)
    before_f=prior.snapshot(teacher);before_d=prior.snapshot(dura);digest=prior.sha(root/prior.DURA)
    model=make_model(device).train();optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    rng=np.random.default_rng(2026110104);torch.manual_seed(2026110105)
    if device.startswith('cuda'):torch.cuda.manual_seed_all(2026110105)
    row=out/'flow';row.mkdir();(row/'checkpoints').mkdir();updates=0;start_epoch=0;started=time.perf_counter()
    contract=dict(method='phase1_class_hypothesis',commit=commit,F_run=core.F_RUN,F_row=core.F_ROW,
        DURA_sha256=digest,count=core.COUNT,batch=core.BATCH,epochs=core.EPOCHS,epsilon=EPSILON,
        threshold=THRESHOLD,hard_weight=HARD_WEIGHT,posterior_weight=POSTERIOR_WEIGHT,oracle_weight=ORACLE_WEIGHT,oracle_steps=12)
    if resume_checkpoint is not None:
        start_epoch,updates=load_state(resume_checkpoint,model,optimizer,rng,contract=contract,device=device)
        if start_epoch>core.EPOCHS or updates!=start_epoch*((core.COUNT+core.BATCH-1)//core.BATCH):
            raise ValueError('Fixed-budget epoch-boundary checkpoint required')
    with (row/'training.jsonl').open('x',encoding='utf8') as log:
        for epoch in range(start_epoch+1,core.EPOCHS+1):
            view=(epoch-1)%20+1;received=np.load(cache/'epochs'/f'{view:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
            if received.shape!=clean.shape or received.dtype!=np.float32:raise ValueError('Phase1 received source view required')
            order=rng.permutation(core.COUNT);clean_count=core.COUNT//4
            sums={k:0. for k in ('loss','posterior','distill')};counts={k:0 for k in ('labeled','pseudo_valid','valid_wrong','oracle_reached')}
            for offset in range(0,core.COUNT,core.BATCH):
                ix=order[offset:offset+core.BATCH];target=core.tensor(clean[ix],device);x=core.tensor(received[ix],device)
                y=torch.tensor(labels[ix].tolist(),dtype=torch.long,device=device)
                n=max(0,min(len(ix),clean_count-offset))
                if n:x[:n]=target[:n]
                for group in optimizer.param_groups:group['lr']=core.lr(updates)
                stats=step(model,teacher,dura,x,target,y,optimizer);updates+=1
                for key in sums:sums[key]+=stats[key]*len(ix)
                for key in counts:counts[key]+=stats[key]
            if counts['labeled']!=6300:raise ValueError('Every legal L_s label must appear exactly once per epoch')
            if not prior.unchanged(teacher,before_f) or not prior.unchanged(dura,before_d) or prior.sha(root/prior.DURA)!=digest:
                raise ValueError('Frozen networks must be unchanged before checkpointing')
            record=dict(epoch=epoch,view=view,updates=updates,losses={k:v/core.COUNT for k,v in sums.items()},
                        **counts,seconds=time.perf_counter()-started)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
            save_state(row/'checkpoints'/f'epoch-{epoch:02d}.pt',model,optimizer,rng,epoch=epoch,updates=updates,contract=contract,device=device)
            print(json.dumps(record),flush=True)
    if updates!=core.STEPS or not prior.unchanged(teacher,before_f) or not prior.unchanged(dura,before_d) or prior.sha(root/prior.DURA)!=digest:
        raise ValueError('Fixed budget or frozen model identity changed')
    save(row/'final.pt',model,commit,core.EPOCHS,digest)
    write(row/'training_complete.json',dict(status='SOURCE_TRAINED',epoch=core.EPOCHS,updates=updates,
        F_run=core.F_RUN,F_row=core.F_ROW,F_frozen=True,DURA_frozen=True,DURA_sha256=digest,
        L_s_labels_read=6300,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,source_roles=['L_s','U_s'],
        parameters=sum(p.numel() for p in model.parameters()),epsilon=EPSILON,threshold=THRESHOLD,
        resumed_from=str(resume_checkpoint) if resume_checkpoint else None,resumed_after_epoch=start_epoch))


@torch.no_grad()
def predict(root,out,commit):
    done=read(out/'flow/training_complete.json');digest=prior.sha(root/prior.DURA)
    if (done.get('status'),done.get('epoch'),done.get('updates'),done.get('F_frozen'),done.get('DURA_frozen'),done.get('DURA_sha256'),done.get('U_s_labels_read'),done.get('test_IQ_reads'))!=('SOURCE_TRAINED',core.EPOCHS,core.STEPS,True,True,digest,0,0):
        raise ValueError('Complete frozen source model required')
    teacher,_=core.teacher(root,'cuda:0');dura=prior.frozen_dura(root,'cuda:0');model=load(out/'flow/final.pt','cuda:0',commit,digest)
    folder,ids=core.reference(root);dst=out/'flow/test_prediction';dst.mkdir();np.save(dst/'physical_ids.npy',ids,allow_pickle=False)
    gate_counts={}
    for view in core.VIEWS:
        values=np.load(folder/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        if values.shape!=(len(ids),2,256) or values.dtype!=np.float32:raise ValueError('Complete Phase1 received IQ required')
        raw=np.lib.format.open_memmap(dst/(view+'.raw.npy'),mode='w+',dtype=np.int64,shape=(len(ids),))
        restored=np.lib.format.open_memmap(dst/(view+'.restored.npy'),mode='w+',dtype=np.int64,shape=(len(ids),))
        accepted_count=0
        for offset in range(0,len(ids),core.BATCH):
            x=core.tensor(values[offset:offset+core.BATCH],'cuda:0');anchor=dura(x)
            feat=teacher.features(anchor);baseline=teacher.classify_features(feat);q,candidate=model(x,anchor,feat,baseline)
            checked=teacher.classify_features(teacher.features(candidate))
            y,accepted=select_verified(anchor,candidate,F.softmax(q,dim=1),baseline.argmax(1),checked.argmax(1),threshold=THRESHOLD)
            a=teacher.classify_features(teacher.features(x));b=teacher.classify_features(teacher.features(y))
            if not all(torch.isfinite(v).all() for v in (q,candidate,checked,y,a,b)):raise ValueError('Finite frozen predictions required')
            if not torch.equal(b.argmax(1),torch.where(accepted,checked.argmax(1),baseline.argmax(1))):
                raise ValueError('Verified gate must preserve the selected classifier decision')
            accepted_count+=int(accepted.sum())
            raw[offset:offset+len(x)]=np.from_dlpack(a.argmax(1).cpu());restored[offset:offset+len(x)]=np.from_dlpack(b.argmax(1).cpu())
        raw.flush();restored.flush();gate_counts[view]=accepted_count
        print(json.dumps(dict(view=view,count=len(ids),accepted=accepted_count)),flush=True)
    write(dst/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(core.VIEWS),F_run=core.F_RUN,F_row=core.F_ROW,
        commit=commit,TX_truth_reads=0,model_updates=0,accepted_by_view=gate_counts))


def score(root,out,commit):core.RUN=RUN;return core.score(root,out,commit)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('smoke','score'));parser.add_argument('--root',type=Path,required=True);parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    {'smoke':smoke,'score':score}[args.mode](args.root,args.root/'runs'/RUN,args.commit)


if __name__=='__main__':main()
