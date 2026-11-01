"""Frozen source-trained guarded router, evaluated only on Phase1 views."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import experiment as core, anchor_balanced as prior
from . import boundary_student_experiment as boundary_model
from . import feature_conditioned_experiment as feature_model

RUN='20261101-phase1-noise-guard-router-s2026092701-r01'
ASSET=Path(__file__).with_name('guarded_router_weights.json')
ASSET_SHA='0c54f2d155617422e6dfc72766402dcf84d2addcd822f227ed813538b6b8620a'
BOUNDARY_RUN='20261101-phase1-boundary-student-E30-s2026092701-r02'
FEATURE_RUN='20261101-phase1-feature-conditioned-E30-s2026092701-r02'
read=core.read
write=core.write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_models(root,device):
    if sha(ASSET)!=ASSET_SHA:raise ValueError('Frozen router asset mismatch')
    weights=read(ASSET)
    if weights['selected']!={'kind':'source_router','epoch':20,'threshold':0.0,'energy_limit':-1.0,'validation_net':131}:
        raise ValueError('Unexpected source-selected routing rule')
    dura_sha=prior.sha(root/prior.DURA)
    if dura_sha!=weights['dura_checkpoint_sha256']:raise ValueError('Frozen DURA mismatch')
    bp=root/'runs'/BOUNDARY_RUN/'flow/final.pt';fp=root/'runs'/FEATURE_RUN/'flow/final.pt'
    if sha(bp)!=weights['boundary_checkpoint_sha256'] or sha(fp)!=weights['feature_checkpoint_sha256']:
        raise ValueError('Frozen expert checkpoint mismatch')
    teacher,contract=core.teacher(root,device);dura=prior.frozen_dura(root,device)
    commit=weights['trained_expert_commit']
    boundary=boundary_model.load(bp,device,commit,dura_sha)
    feature=feature_model.load(fp,device,commit,dura_sha)
    router=nn.Sequential(nn.Linear(26,64),nn.GELU(),nn.Linear(64,4)).to(device)
    router.load_state_dict({k:torch.tensor(v,dtype=torch.float32,device=device) for k,v in weights['router_state'].items()},strict=True)
    router.eval().requires_grad_(False)
    mean=torch.tensor(weights['feature_mean'],dtype=torch.float32,device=device)
    std=torch.tensor(weights['feature_std'],dtype=torch.float32,device=device)
    if mean.shape!=(26,) or std.shape!=(26,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std<=0).any():
        raise ValueError('Invalid saved feature normalization')
    return teacher,contract,dura,boundary,feature,router,mean,std


@torch.no_grad()
def route(models,x):
    teacher,_,dura,boundary,feature,router,mean,std=models
    anchor=dura(x);feat=teacher.features(anchor);baseline=teacher.classify_features(feat)
    outputs=(x,anchor,boundary(x,anchor),feature(x,anchor,feat,baseline))
    logits=[teacher.classify_features(teacher.features(outputs[0])),baseline,
            teacher.classify_features(teacher.features(outputs[2])),teacher.classify_features(teacher.features(outputs[3]))]
    probs=torch.stack([F.softmax(z,dim=1) for z in logits],dim=1)
    energy=torch.stack((x.square().mean((1,2)).clamp_min(1e-8).log(),anchor.square().mean((1,2)).clamp_min(1e-8).log()),1)
    all_features=torch.cat((probs.reshape(len(x),24),energy),1)
    gate=F.softmax(router(((all_features-mean)/std).clamp(-10,10)),dim=1)
    alternatives=gate.clone();alternatives[:,1]=-1
    choice=alternatives.argmax(1);gap=alternatives.gather(1,choice[:,None]).squeeze(1)-gate[:,1]
    choice=torch.where((gap>=0)&(energy[:,1]<=-1.0),choice,torch.ones_like(choice))
    predictions=probs.argmax(2);selected=predictions.gather(1,choice[:,None]).squeeze(1)
    if not all(torch.isfinite(t).all() for t in (*outputs,*logits,probs,energy,gate)):
        raise ValueError('Finite source and expert outputs required')
    return predictions[:,0],selected,choice


def smoke(root,out,device,mode):
    models=load_models(root,device);cache,clean,roles=core.data(root,models[1])
    ix=np.flatnonzero((roles=='L_s')|(roles=='U_s'))[:(128 if device!='cpu' else 2)]
    source=np.load(cache/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    values=core.tensor(source[ix],device);first=route(models,values);second=route(models,values)
    if not all(torch.equal(a,b) for a,b in zip(first,second)):raise ValueError('Deterministic frozen inference required')
    if not torch.all((first[2]>=0)&(first[2]<4)):raise ValueError('Invalid route')
    write(out/(mode+'.json'),dict(status='PASS',source_rows=len(ix),TX_labels_read=0,V_reads=0,test_IQ_reads=0,
          F_run=core.F_RUN,F_row=core.F_ROW,F_frozen=True,DURA_frozen=True,asset_sha256=ASSET_SHA,
          boundary_sha256=sha(root/'runs'/BOUNDARY_RUN/'flow/final.pt'),feature_sha256=sha(root/'runs'/FEATURE_RUN/'flow/final.pt'),
          deterministic=True,device=device,choice_counts=[int((first[2]==i).sum().item()) for i in range(4)]))


def predict(root,out,commit):
    if read(out/'gpu_smoke.json').get('status')!='PASS':raise ValueError('GPU source smoke required')
    models=load_models(root,'cuda:0');folder,ids=core.reference(root)
    dst=out/'flow/test_prediction';dst.mkdir(parents=True);np.save(dst/'physical_ids.npy',ids,allow_pickle=False)
    choice_counts={}
    for view in core.VIEWS:
        values=np.load(folder/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        if values.shape!=(len(ids),2,256) or values.dtype!=np.float32:raise ValueError('Complete Phase1 received IQ required')
        raw=np.lib.format.open_memmap(dst/(view+'.raw.npy'),mode='w+',dtype=np.int64,shape=(len(ids),))
        restored=np.lib.format.open_memmap(dst/(view+'.restored.npy'),mode='w+',dtype=np.int64,shape=(len(ids),))
        choice=np.lib.format.open_memmap(dst/(view+'.choice.npy'),mode='w+',dtype=np.uint8,shape=(len(ids),))
        for offset in range(0,len(ids),core.BATCH):
            x=core.tensor(values[offset:offset+core.BATCH],'cuda:0');a,b,c=route(models,x)
            raw[offset:offset+len(x)]=np.from_dlpack(a.cpu());restored[offset:offset+len(x)]=np.from_dlpack(b.cpu())
            choice[offset:offset+len(x)]=np.from_dlpack(c.to(torch.uint8).cpu())
        raw.flush();restored.flush();choice.flush();choice_counts[view]=[int(np.sum(choice==i)) for i in range(4)]
        print(json.dumps(dict(view=view,count=len(ids),choices=choice_counts[view])),flush=True)
    write(out/'flow/choice_counts.json',dict(experts=['raw','dura','boundary','feature'],views=choice_counts,truth_reads=0))
    write(dst/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(core.VIEWS),F_run=core.F_RUN,F_row=core.F_ROW,
          commit=commit,TX_truth_reads=0,model_updates=0))


def score(root,out,commit):
    core.RUN=RUN
    return core.score(root,out,commit)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke','score'));p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=a.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.root/'runs'/RUN
    if a.mode=='smoke':smoke(a.root,out,'cpu','smoke')
    else:score(a.root,out,a.commit)


if __name__=='__main__':main()
