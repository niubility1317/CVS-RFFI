"""Author-style triplet source training on Phase2's 256-IQ extension."""
from __future__ import annotations

import json
from pathlib import Path
import time
import torch

from experiments.orthogonal_paper_repro.phase2_source import split_labeled
from experiments.pamf_phase2_practical import external_additional as extra
from experiments.pamf_phase2_practical.group1_resources import Measure
from experiments.pamf_phase2_practical.io import read, require, write
from experiments.lora_paper_repro.train import triplet_loss


def train_triplets(model, spectra, labels, train_idx, val_idx, *, seed, callback=None,
                   max_epochs=1000, patience=20, lr_patience=10):
    if min(max_epochs,patience,lr_patience)<=0:
        raise ValueError('invalid LoRa training budget')
    dev=next(model.parameters()).device
    split_indices=(train_idx.cpu(),val_idx.cpu())
    generators=(torch.Generator().manual_seed(seed),torch.Generator().manual_seed(seed+1))
    optimizer=torch.optim.RMSprop(model.parameters(),lr=.001,alpha=.9,eps=1e-7)
    best=float('inf'); stale=lr_stale=0; history=[]
    for epoch in range(1,max_epochs+1):
        started=time.perf_counter()
        losses=[]
        for training,indices,gen in zip((True,False),split_indices,generators):
            model.train(training)
            y=labels[indices]
            steps=len(indices)//32
            if steps<1 or torch.unique(y).numel()!=6:
                raise ValueError('each source split requires six classes and at least one batch')
            total=0.
            for _ in range(steps):
                a,p,n=extra._triplets(y,32,gen)
                batch=torch.cat((indices[a],indices[p],indices[n]))
                with torch.set_grad_enabled(training):
                    anchor,positive,negative=model.network(spectra[batch].to(dev)).split(32)
                    loss=triplet_loss(anchor,positive,negative,margin=.1).mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('nonfinite LoRa triplet loss')
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    checks=[torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
                    if not checks or not bool(torch.stack(checks).all()):
                        raise FloatingPointError('nonfinite LoRa gradient')
                    optimizer.step()
                total+=float(loss.detach())
            losses.append(total/steps)
        val=losses[1]
        if val<best:
            best,stale,lr_stale=val,0,0
        else:
            stale+=1
            lr_stale+=1
            if lr_stale>=lr_patience:
                for group in optimizer.param_groups:group['lr']*=.2
                lr_stale=0
        item=dict(epoch=epoch,training_loss=losses[0],validation_loss=val,
                  learning_rate=optimizer.param_groups[0]['lr'],elapsed_seconds=time.perf_counter()-started)
        history.append(item)
        if callback is not None:callback(item)
        if stale>=patience:break
    return history


def train_source(c, seed, output, device):
    from experiments.pamf_phase2_practical.group1_source import extract,model_for,recipe,source_data

    output=Path(output)
    L=source_data(c,'L_s')
    labels=torch.tensor(L.metadata['y'].tolist(),dtype=torch.long)
    train_idx,val_idx=split_labeled(labels,seed)
    torch.manual_seed(seed)
    model=model_for('lora_paper_v2',device)
    output.mkdir(parents=True,exist_ok=False)
    config=recipe('lora_paper_v2')
    contract=read(c['source_contract'])
    write(output/'recipe.json',config)
    write(output/'source_contract.json',contract)
    initial=output/'initial_source_only.pt'
    torch.save(dict(model=model.state_dict(),recipe=config),initial)
    model.load_state_dict(torch.load(initial,map_location=device,weights_only=False)['model'],strict=True)
    model.eval()
    with torch.no_grad():
        z=extra.native_features(model,L.x[:2].to(device))
    require(z.shape==(2,512) and bool(torch.isfinite(z).all()),'Real source checkpoint smoke failed')
    write(output/'source_smoke.json',dict(status='PASS',real_checkpoint=True,source_rows=2,
                                          target_rows=0,query_rows=0))
    gen=torch.Generator().manual_seed(seed)
    noisy=extra._awgn(L.x,gen)
    with Measure(output/'A_resources.json',dict(profile='lora_paper_v2',seed=seed,stage='A'),
                 model,optimizers=True) as probe:
        with torch.no_grad():
            spectra=torch.cat([extra.channel_spectrogram(chunk.to(device)).cpu()
                               for chunk in noisy.split(128)])
        with (output/'epochs.jsonl').open('x',encoding='utf8') as log:
            def observe(item):
                log.write(json.dumps(item,allow_nan=False)+'\n')
                log.flush()
                print('LORA_SOURCE_EPOCH',item['epoch'],item['training_loss'],item['validation_loss'],flush=True)
            history=train_triplets(model,spectra,labels,train_idx,val_idx,seed=seed,callback=observe)
        model.eval()
        native=dict(source_enrollment_features=extract(model,'lora_paper_v2',L.x,device),
                    source_enrollment_y=labels)
        torch.save(dict(schema='group1_A_v1',model=model.state_dict(),method='lora_paper_v2',
            profile='lora_paper_v2',seed=seed,recipe=config,source_contract=contract,native_state=native,
            ancestors=[],initialization='scratch',target_contact=False),output/'source.pt')
        probe.result.update(actual_epochs=len(history),source_selection='final_early_stop_epoch',
                            source_train_count=len(train_idx),source_validation_count=len(val_idx),
                            source_U_used=False,target_contact=False)
