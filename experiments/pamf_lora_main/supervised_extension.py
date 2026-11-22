"""Long, matched labeled-only LoRa continuations and source-side selection."""
import argparse
from functools import partial
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, tensor, write
from .comparison import seed_all
from .data import VIEWS
from .packet_models import SEED, load_source_labels
from .packet_runner import schedule
from .refine_f import build, labeled_weights, measure
from .refine_runner import source_handles, start
from .repair_features import CoherentPacket, complete_labeled_batches
from .runner import activate


METHODS = ('spectral_clean', 'spectral_practical', 'coherent_practical')
EPOCHS = 120
WARM_EPOCHS = 36
CHECKPOINT_EPOCHS = (36, 60, 84, EPOCHS)


def construct(parent, contract, method, device):
    if method == 'coherent_practical':
        return CoherentPacket(build(parent, contract, 'raw_pl', device)).to(device)
    if method in METHODS:
        return build(parent, contract, 'spectral_focus', device)
    raise ValueError('Unregistered LoRa supervised extension arm')


def source_arrays(source, cache, device):
    arrays = []
    for role in ('L_s', 'V'):
        p = cache / f'{role}.npy'
        values = np.load(p, mmap_mode='r', allow_pickle=False)
        if values.shape != (len(VIEWS), 1600 if role=='L_s' else 800, 6, 2, 256):
            raise ValueError(f'Unexpected cached source view dimensions for {role}')
        y, rx = load_source_labels(source, role)
        arrays.append((tensor(values, device), y, rx))
    return arrays


@torch.no_grad()
def metrics(model, x, y, rx):
    model.eval()
    logits = torch.cat([model(z)[0] for z in x.split(48)])
    pred = logits.argmax(1)
    good = pred == y
    return dict(accuracy=float(good.float().mean()), ce=float(F.cross_entropy(logits,y)),
                by_class=torch.bincount(y[good],minlength=10).tolist(),
                by_rx=[float(good[rx==r].float().mean()) for r in range(5)])


def lr_scale(step):
    if step < 1800:
        return .30 + .70*(1.+math.cos(math.pi*step/1800))*.5
    return .08 + .52*(1.+math.cos(math.pi*min(step-1800,4200)/4200))*.5


def train(rt, method):
    source,parent,cache,contract = source_handles(rt)
    folder = Path(rt['run_root']) / 'F' / method
    folder.mkdir(parents=True, exist_ok=False)
    dev = 'cuda:0'
    seed_all(SEED)
    (l,y,rx),(v,vy,vr) = source_arrays(source,cache,dev)
    model = construct(parent,contract,method,dev)
    _,_,prediction = measure(model,l[0],y)
    _,class_weight = labeled_weights(prediction,y,True)
    y_np, rx_np = np.asarray(y.tolist()), np.asarray(rx.tolist())
    labels = np.random.default_rng(721)
    scenes = np.random.default_rng(267)
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    step = 0
    t0 = time.monotonic()
    best = None

    def report(epoch, trace, clean_loss=None, practical_loss=None):
        item = dict(epoch=epoch, step=step, seconds=round(time.monotonic()-t0,3),
                    L=metrics(model,l[0],y,rx), V=metrics(model,v[0],vy,vr),
                    lr=optimizer.param_groups[0]['lr'], train_clean_ce=clean_loss, train_practical_ce=practical_loss)
        if epoch == 0 or epoch in CHECKPOINT_EPOCHS:
            item['practical_V'] = {name:metrics(model,v[i],vy,vr) for i,name in enumerate(VIEWS) if i}
            p = sum(q['accuracy'] for q in item['practical_V'].values())/6
            worst = min(item['V']['by_rx'])
            item['source_selection'] = (item['V']['accuracy'] + .5*p + .3*worst)/1.8
        row=json.dumps(item,allow_nan=False)
        trace.write(row+'\n')
        trace.flush()
        print('LORA_SUPERVISED',method,row,flush=True)
        return item

    with (folder/'epochs.jsonl').open('x',encoding='utf8') as trace:
        report(0,trace)
        for epoch in range(1,EPOCHS+1):
            alpha = min(.35, max(0.,(epoch-WARM_EPOCHS)*.035)) if method!='spectral_clean' else 0.
            c_tot = torch.zeros((),device=dev)
            p_tot = torch.zeros((),device=dev)
            for batch in complete_labeled_batches(y_np,rx_np,labels):
                idx=tensor(batch,dev)
                scene=tensor(scenes.integers(1,len(VIEWS),size=len(batch)),dev)
                model.train()
                x=l[0,idx]
                if alpha:
                    clean, practical = model(torch.cat((x,l[scene,idx])))[0].chunk(2)
                    p_ce = F.cross_entropy(practical,y[idx],weight=class_weight)
                else:
                    clean = model(x)[0]
                    p_ce = c_tot.new_zeros(())
                c_ce=F.cross_entropy(clean,y[idx],weight=class_weight)
                optimizer.zero_grad(set_to_none=True)
                (c_ce+alpha*p_ce).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                c_tot+=c_ce.detach()
                p_tot+=p_ce.detach()
                step+=1
            if step!=epoch*50:
                raise ValueError('Source extension must use every labeled physical package each epoch')
            if epoch==1 or epoch%6==0:
                m=report(epoch,trace,float(c_tot/50),float(p_tot/50))
            if epoch in CHECKPOINT_EPOCHS:
                torch.save(model.state_dict(),folder/f'e{epoch}.pt')
                if best is None or m['source_selection']>best['source_selection']:
                    best=m
    write(folder/'complete.json',dict(status='SOURCE_FROZEN',method=method,steps=step,epochs=EPOCHS,
          source_contract=contract,seed=SEED,U_reads=0,target_reads=0,best_source=best))


def smoke(rt):
    source,parent,cache,contract=source_handles(rt)
    with np.load(source/'L_s_metadata.npz',allow_pickle=False) as meta:
        if meta['ids'].tolist()!=contract['role_ids']['L_s']:
            raise ValueError('L source order differs')
        y,rx=meta['y'],meta['rx_i']
    batch=np.asarray(list(complete_labeled_batches(y,rx,np.random.default_rng(721))))
    if batch.shape!=(50,32) or not np.array_equal(np.sort(batch.ravel()),np.arange(len(y))):
        raise ValueError('Source balanced epoch does not cover all physical rows')
    view=np.load(cache/'L_s.npy',mmap_mode='r',allow_pickle=False)
    x=tensor(np.concatenate((view[0,:3],view[1,:3])))
    scores={}
    for method in METHODS:
        f=construct(parent,contract,method,'cpu')
        output,_=f(x)
        loss=F.cross_entropy(output,tensor(np.tile(y[:3],2)))
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(f.parameters(),5.,error_if_nonfinite=True)
        if output.shape!=(6,10) or float(norm)<=0.:
            raise ValueError('Real-checkpoint source supervised backward failed')
        scores[method]=float(norm)
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',L_unique=1600,source_only=True,
          initial_gradient_norms=scores,U_reads=0,target_reads=0))


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    launch=partial(start,module=__spec__.name)
    state=root/'state.json'
    try:
        write(state,dict(status='SOURCE_SMOKE'))
        if launch(rt,'smoke','supervised').wait():
            raise RuntimeError('Real-checkpoint source supervised smoke failed')
        write(state,dict(status='SOURCE_TRAINING'),exclusive=False)
        schedule(rt,[('train',m) for m in METHODS],launch)
        rows=[read(root/'F'/m/'complete.json') for m in METHODS]
        winner=max(rows,key=lambda z:z['best_source']['source_selection'])
        write(root/'source_decision.json',dict(status='SOURCE_SELECTION_FROZEN',selected=winner['method'],
              epoch=winner['best_source']['epoch'],U_reads=0,target_reads=0,
              candidates={row['method']:row['best_source'] for row in rows}))
        write(state,dict(status='SOURCE_ONLY_COMPLETE',U_reads=0,target_reads=0),exclusive=False)
    except Exception as error:
        write(state,dict(status='FAILED',error=str(error)),exclusive=False)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('kind',choices=('dispatch','smoke','train'))
    parser.add_argument('runtime',type=Path)
    parser.add_argument('name',nargs='?')
    a=parser.parse_args()
    rt=read(a.runtime)
    if a.kind=='dispatch':
        dispatch(rt)
    else:
        activate(rt)
        if a.kind=='train':
            train(rt,a.name)
        else:
            smoke(rt)
