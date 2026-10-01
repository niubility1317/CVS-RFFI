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
from experiments.cvs_stability_identity.model import build, VARIANTS
from experiments.cvs_identity_ce.source import source_args, checkpoint_smoke, write
from baselines.common.practical_source import build_contract_split
from baselines.common.cvs_data import make_cvs_loader


def validate_config(c):
    for key in ('checkpoint','resume','teacher','initial_checkpoint','target_inputs','target_truth','p1_truth','p1_capsule'):
        if c.get(key): raise ValueError('Source-only scratch training rejects ' + key)
    required = dict(method='cvs_stability_identity', epochs=200, batch_size=128, lr=.0002,
        lr_min=1e-6, weight_decay=.0001, drop_last=False, augmentation=False,
        domain_backbone=False, extra_losses=[], selection='fixed_last_epoch', split_seed=392005)
    if any(c.get(k)!=v for k,v in required.items()): raise ValueError('Clean matched training contract mismatch')
    if c['variant'] not in VARIANTS: raise ValueError('Unknown variant')
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


def train(c):
    validate_config(c);out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    seed=c['model_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2);device=torch.device(c.get('device','cuda:0'))
    model=build(c['variant']).to(device)
    phase_active=model.id_backbone.time_stability is not None
    dsq_active=model.id_backbone.freq_stability is not None
    if not phase_active or dsq_active!=(c['variant']=='phase_dsq'):raise ValueError('Physical cue activation differs from registered variant')
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
        optimizer='AdamW+CosineAnnealingLR',loader_seed=seed,phase_delta_active=phase_active,dsq_active=dsq_active,
        time_stability_channels=8,freq_stability_channels=4 if dsq_active else 0)
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
                    pseudo_labels_active=False,extra_losses_active=False,phase_delta_active=phase_active,dsq_active=dsq_active)
                sf.write(json.dumps(rec,allow_nan=False)+'\n')
            if n!=50 or exposure!=6300:raise ValueError('Matched 50-step/6300-sample budget changed')
            scheduler.step();validation=validate(model,val_loader,device)
            stats=dict(epoch=epoch,optimizer_steps=n,optimizer_steps_total=step,source_sample_exposure=exposure,
                clean_ce=ce_sum/n,total_loss=ce_sum/n,ce_weight=1.,learning_rate=lr,gradient_norm=norm_sum/n,
                gradient_used_parameters=grad_params,augmentation_active=False,satellite_sample_exposure=0,
                domain_backbone_active=False,extra_losses_active=False,pseudo_labels_active=False,phase_delta_active=phase_active,dsq_active=dsq_active,
                **validation,elapsed_seconds=time.perf_counter()-tic,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
            line=json.dumps(stats,allow_nan=False);print('EPOCH '+line,flush=True);ef.write(line+'\n');ef.flush();sf.flush()
            compact={k:v for k,v in stats.items() if not isinstance(v,dict)}
            if writer is None:writer=csv.DictWriter(cf,fieldnames=list(compact));writer.writeheader()
            writer.writerow(compact);cf.flush()
    contract=json.loads((out/'source_contract.json').read_text(encoding='utf-8'))
    torch.save(dict(model=model.state_dict(),method='cvs_stability_identity',variant=c['variant'],epoch=200,
        classes=contract['classes'],num_classes=6,source_contract=contract,initialization=initial,
        selection='fixed_last_epoch',config=resolved),out/'last.pt')
    training_peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
    from experiments.cvs_clean_design.profile import resource_profile
    write(out/'resource_profile.json',resource_profile(model,device))
    write(out/'completion.json',dict(status='SOURCE_TRAINED',epoch=200,steps=step,checkpoint=str(out/'last.pt'),
        elapsed_seconds=time.perf_counter()-started,training_peak_cuda_allocated_bytes=training_peak,target_access=False,target_evaluated=False,final_source_metrics=validation))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    train(json.loads(Path(a.config).read_text(encoding='utf-8')))
