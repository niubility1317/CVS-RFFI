"""No target construction, no augmentation, no teacher, one CE update/batch."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'code')]
import numpy as np
import torch
import torch.nn.functional as F
from experiments.cvs_equivariant_identity.model import build, VARIANTS, equivariant_contract, frozen_synthetic_diagnostics
from experiments.cvs_identity_ce.source import source_args, checkpoint_smoke, write
from baselines.common.practical_source import build_contract_split
from baselines.common.cvs_data import make_cvs_loader


def validate_config(c):
    for key in ('checkpoint','resume','teacher','initial_checkpoint','target_inputs','target_truth','p1_truth','p1_capsule'):
        if c.get(key): raise ValueError('Source-only scratch training rejects ' + key)
    required = dict(method='cvs_equivariant_identity', epochs=200, batch_size=128, lr=.0002,
        lr_min=1e-6, weight_decay=.0001, drop_last=False, augmentation=False,
        domain_backbone=False, extra_losses=[], selection='fixed_last_epoch', split_seed=392005)
    if any(c.get(k)!=v for k,v in required.items()): raise ValueError('Clean matched training contract mismatch')
    if c['variant'] not in VARIANTS: raise ValueError('Unknown variant')
    if c.get('equivariant')!=equivariant_contract(c['variant']):raise ValueError('RF operator contract mismatch')
    return c


@torch.no_grad()
def validate(model, loader, device):
    model.eval(); correct=count=0; total=0.; groups={}
    for batch in loader:
        y=batch['label'].to(device); logits=model(batch['iq'].to(device))
        ok=(logits.argmax(1)==y).cpu()
        total+=float(F.cross_entropy(logits,y,reduction='sum')); correct+=int(ok.sum()); count+=len(y)
        for rx in batch['receiver'].unique().tolist():
            mask=batch['receiver']==rx
            g=groups.setdefault(str(rx),[0,0]);g[0]+=int(ok[mask].sum());g[1]+=int(mask.sum())
    rates={rx:hit/n for rx,(hit,n) in groups.items()}
    return dict(source_val_count=count, source_val_accuracy=correct/count, source_val_ce=total/count,
        source_val_rx_accuracy=rates, source_val_worst_rx=min(rates.values()))


@torch.no_grad()
def final_source_diagnostics(model,loader,device):
    model.eval();groups={};count=0
    for batch in loader:
        emb=model.features(batch['iq'].to(device))
        logits=model.classify_features(emb)
        ok=(logits.argmax(1).cpu()==batch['label'])
        normalized=F.normalize(emb,dim=1,eps=1e-4).cpu().double()
        for j in range(len(emb)):
            key=(int(batch['label'][j]),int(batch['receiver'][j]),int(batch['day'][j]))
            g=groups.setdefault(key,dict(count=0,correct=0,embedding_sum=torch.zeros(160,dtype=torch.float64),square_sum=0.))
            g['count']+=1;g['correct']+=int(ok[j]);g['embedding_sum']+=normalized[j];g['square_sum']+=float(normalized[j].square().sum())
        count+=len(emb)
    if count!=27000:raise ValueError('IncompletefinalsourceVdiagnostics')
    records=[];tx_groups={};rx_groups={}
    for (tx,rx,day),g in sorted(groups.items()):
        centroid=g['embedding_sum']/g['count']
        records.append(dict(tx=tx,receiver=rx,day=day,count=g['count'],accuracy=g['correct']/g['count'],
                            unit_embedding_mean=centroid.tolist(),unit_embedding_trace_variance=max(0.,g['square_sum']/g['count']-float(centroid.square().sum()))))
        for key,collection in ((tx,tx_groups),((tx,rx),rx_groups)):
            h=collection.setdefault(key,dict(count=0,total=torch.zeros(160,dtype=torch.float64)))
            h['count']+=g['count'];h['total']+=g['embedding_sum']
    tx_means={k:g['total']/g['count'] for k,g in tx_groups.items()}
    between=[float((tx_means[a]-tx_means[b]).square().sum()) for a in tx_means for b in tx_means if a<b]
    within=[float((g['total']/g['count']-tx_means[tx]).square().sum()) for (tx,rx),g in rx_groups.items()]
    return dict(role='V',count=count,groups=records,mean_between_tx_centroid_squared_distance=sum(between)/len(between),
                mean_within_tx_rx_centroid_squared_distance=sum(within)/len(within),
                target_access=False,used_for_training=False,used_for_selection=False,
                claim='Aggregateidentityembeddinggeometryonly;not TX hardwareparameterrecovery or calibratedRXseparation.')


def train(c):
    validate_config(c);out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    seed=c['model_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2);device=torch.device(c.get('device','cuda:0'))
    model=build(c['variant']).to(device)
    physical=model
    if physical.contract()!=c['equivariant']:raise ValueError('Actual RF operator differs from registered configuration')
    checkpoint_smoke(model,out/'initial_smoke.pt',device)
    split=build_contract_split(source_args(c))
    if split.test or split.named_tests or split.split_info['counts']!={'L_s':6300,'U_s':56700,'V':27000}:
        raise ValueError('Source roles changed or target constructed')
    train_loader=make_cvs_loader(split.train,batch_size=128,shuffle=True,num_workers=0,device=device,drop_last=False)
    train_loader.sampler.generator=torch.Generator().manual_seed(seed)
    val_loader=make_cvs_loader(split.val,batch_size=256,shuffle=False,num_workers=0,device=device,drop_last=False)
    initial=dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],
        target_access=False,target_contact=False,model_seed=seed,physical_roles='EXACT_MATCH',selection='fixed_last_epoch')
    write(out/'initialization.json',initial)
    resolved=dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,
        commit=(ROOT/'release_commit.txt').read_text().strip() if (ROOT/'release_commit.txt').exists() else 'LOCAL',
        torch_version=torch.__version__,hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        source_counts=split.split_info['counts'],steps_per_epoch=len(train_loader),
        total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        U_s_use='unused',augmentation=False,target_access=False,precision='float32',gradient_clipping=None,
        optimizer='AdamW+CosineAnnealingLR',loader_seed=seed,equivariant_actual=physical.contract(),equivariant_active=True,classifier_scale=30.0)
    write(out/'resolved_config.json',resolved);print('RESOLVED_CONFIG '+json.dumps(resolved),flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=200,eta_min=1e-6)
    started=time.perf_counter();step=0
    with (out/'step_metrics.jsonl').open('x',encoding='utf-8') as sf, (out/'epoch_metrics.jsonl').open('x',encoding='utf-8') as ef, (out/'epoch_metrics.csv').open('x',encoding='utf-8',newline='') as cf:
        writer=None
        for epoch in range(1,201):
            tic=time.perf_counter();model.train();ce_sum=norm_sum=0.;exposure=0;n=0;lr=optimizer.param_groups[0]['lr'];grad_params=0
            for batch in train_loader:
                x,y=batch['iq'].to(device),batch['label'].to(device);optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(x),y)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite CE')
                loss.backward();grads=[p.grad.detach() for p in model.parameters() if p.grad is not None]
                norm=torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                grad_params=sum(p.numel() for p in model.parameters() if p.grad is not None)
                optimizer.step();step+=1;n+=1;exposure+=len(y);ce_sum+=float(loss.detach());norm_sum+=float(norm)
                rec=dict(epoch=epoch,step=step,clean_ce=float(loss.detach()),total_loss=float(loss.detach()),ce_weight=1.0,
                    learning_rate=lr,gradient_norm=float(norm),gradient_used_parameters=grad_params,
                    source_samples=len(y),satellite_samples=0,augmentation_active=False,domain_backbone_active=False,
                    pseudo_labels_active=False,extra_losses_active=False,equivariant_active=True)
                sf.write(json.dumps(rec,allow_nan=False)+'\n')
            if n!=50 or exposure!=6300:raise ValueError('Matched 50-step/6300-sample budget changed')
            scheduler.step();validation=validate(model,val_loader,device)
            stats=dict(epoch=epoch,optimizer_steps=n,optimizer_steps_total=step,source_sample_exposure=exposure,
                clean_ce=ce_sum/n,total_loss=ce_sum/n,ce_weight=1.,learning_rate=lr,gradient_norm=norm_sum/n,
                gradient_used_parameters=grad_params,augmentation_active=False,satellite_sample_exposure=0,
                domain_backbone_active=False,extra_losses_active=False,pseudo_labels_active=False,
                equivariant_active=True,equivariant_diagnostics=model.diagnostics(x),equivariant_diagnostic_scope='last source batch of epoch',
                **validation,elapsed_seconds=time.perf_counter()-tic,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
            line=json.dumps(stats,allow_nan=False);print('EPOCH '+line,flush=True);ef.write(line+'\n');ef.flush();sf.flush()
            compact={k:v for k,v in stats.items() if not isinstance(v,dict)}
            if writer is None:writer=csv.DictWriter(cf,fieldnames=list(compact));writer.writeheader()
            writer.writerow(compact);cf.flush()
    contract=json.loads((out/'source_contract.json').read_text(encoding='utf-8'))
    torch.save(dict(model=model.state_dict(),method='cvs_equivariant_identity',variant=c['variant'],epoch=200,
        classes=contract['classes'],num_classes=6,source_contract=contract,initialization=initial,
        selection='fixed_last_epoch',config=resolved),out/'last.pt')
    write(out/'source_final_diagnostics.json',final_source_diagnostics(model,val_loader,device))
    physical_check=frozen_synthetic_diagnostics(model)
    write(out/'source_physical_diagnostics.json',physical_check)
    print('FROZEN_PHYSICS '+json.dumps(physical_check,allow_nan=False),flush=True)
    from experiments.cvs_clean_design.profile import resource_profile
    profile=resource_profile(model,device)
    profile.update(equivariant_contract=physical.contract(),all_identity_paths_phase_constrained=True,
        normalization_state='Per-packet complex RMS only; no mutable source/query moments',
        base_architecture='native_frequency_fusion_cosine_with_complex_time_and_behavior_scratch')
    write(out/'resource_profile.json',profile)
    write(out/'completion.json',dict(status='SOURCE_TRAINED',epoch=200,steps=step,checkpoint=str(out/'last.pt'),
        elapsed_seconds=time.perf_counter()-started,target_access=False,target_evaluated=False,final_source_metrics=validation))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    train(json.loads(Path(a.config).read_text(encoding='utf-8')))
