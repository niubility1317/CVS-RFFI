"""Six-class Phase2 source adaptation of the selected paper-reproduction recipe."""
from __future__ import annotations

import json
from pathlib import Path
import torch

from experiments.orthogonal_paper_repro.base import train_base
from experiments.pamf_phase2_practical.group1_resources import Measure
from experiments.pamf_phase2_practical.io import read, require, write


def split_labeled(labels, seed):
    """Stratified 90/10 source-L split; no target or U labels enter this function."""
    if labels.ndim != 1 or torch.unique(labels.cpu(),sorted=True).tolist()!=list(range(6)):
        raise ValueError('source L must contain exactly the six old classes')
    gen=torch.Generator().manual_seed(seed)
    train,validation=[],[]
    for class_id in range(6):
        members=torch.where(labels.cpu()==class_id)[0]
        nval=max(2,int(len(members)*.1))
        if len(members)<=nval:
            raise ValueError('too few source-L examples for validation')
        order=members[torch.randperm(len(members),generator=gen)]
        validation.append(order[:nval])
        train.append(order[nval:])
    return torch.cat(train),torch.cat(validation)


def train_source(c, seed, output, device):
    from experiments.pamf_phase2_practical.group1_source import model_for, recipe, source_data

    output=Path(output)
    L=source_data(c,'L_s')
    labels=torch.tensor(L.metadata['y'].tolist(),dtype=torch.long)
    train_idx,val_idx=split_labeled(labels,seed)
    torch.manual_seed(seed)
    model=model_for('orthogonal_paper_v2',device)
    output.mkdir(parents=True,exist_ok=False)
    config=recipe('orthogonal_paper_v2')
    write(output/'recipe.json',config)
    contract=read(c['source_contract'])
    write(output/'source_contract.json',contract)
    initial=output/'initial_source_only.pt'
    torch.save(dict(model=model.state_dict(),recipe=config),initial)
    model.load_state_dict(torch.load(initial,map_location=device,weights_only=False)['model'],strict=True)
    model.eval()
    with torch.no_grad():
        z=model.encoder(L.x[:2].to(device))
    require(z.shape==(2,256) and bool(torch.isfinite(z).all()),'Real source checkpoint smoke failed')
    write(output/'source_smoke.json',dict(status='PASS',real_checkpoint=True,source_rows=2,
                                          target_rows=0,query_rows=0))
    with Measure(output/'A_resources.json',dict(profile='orthogonal_paper_v2',seed=seed,stage='A'),
                 model,optimizers=True) as probe:
        with (output/'epochs.jsonl').open('x',encoding='utf8') as log:
            def observe(item):
                log.write(json.dumps(item,allow_nan=False)+'\n')
                log.flush()
                print('ORTHOGONAL_SOURCE_EPOCH',item['epoch'],item['training_loss'],item['validation_loss'],flush=True)
            trained, history=train_base(model.encoder,L.x[train_idx],labels[train_idx],L.x[val_idx],labels[val_idx],
                num_classes=6,num_targets=200,feature_dim=256,epochs=100,batch_size=128,
                learning_rate=.01,patience=4,seed=seed,callback=observe)
        with torch.no_grad():
            model.pseudo_targets.copy_(trained['targets'])
            model.classifier_weight.copy_(trained['base_weights'])
        model.eval()
        torch.save(dict(schema='group1_A_v1',model=model.state_dict(),method='orthogonal_paper_v2',
            profile='orthogonal_paper_v2',seed=seed,recipe=config,source_contract=contract,native_state={},
            ancestors=[],initialization='scratch',target_contact=False),output/'source.pt')
        probe.result.update(actual_epochs=len(history),best_source_epoch=trained['best_epoch'],
                            source_train_count=len(train_idx),source_validation_count=len(val_idx),
                            source_U_used=False,target_contact=False)
