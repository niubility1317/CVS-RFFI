"""Same-contract F0 G mechanism students; paired IQ only, no F or TX labels."""
from pathlib import Path
import time
import json
import numpy as np
import torch
from .io import read,write,tensor
from .g_train import batches,learning_rate
from .g_cache import CacheView
from .g_ablation import build,loss
from .teacher_targets import EpochTargets,SharedEpochTargets,binding,exposure_count


def train(c,device='cuda:0'):
    if any(c.get(k) for k in ('F_checkpoint','reader','truth','tests','target_inputs')):raise ValueError('G training rejects F/target')
    from experiments.pamf_r46_delayed_prune.runtime import load_checkpoint
    contract=read(c['cache']['source_contract'])
    old=read(c['teacher_source_contract'])
    if any(old[k]!=contract[k] for k in ('role_ids','counts','source_rxs','source_days')) or c['cache']['roles']!='LU':
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    teacher,payload=load_checkpoint(c['teacher_checkpoint'],'cpu')
    if payload['arm']!='prune_after10' or payload['epoch']!=20 or payload['updates']!=4940:raise ValueError('Original R65 final teacher required')
    teacher.eval().requires_grad_(False)
    torch.manual_seed(c['model_seed']);torch.cuda.manual_seed_all(c['model_seed'])
    model=build(teacher,c['arm']).to(device)
    if c['arm'] not in ('copy_no_KD','random_no_KD'):teacher=teacher.to(device)
    view=CacheView(c['cache'])
    if not view.ready():raise ValueError('G source IQ dependency incomplete')
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False)
    write(root/'config.json',c)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    rng=np.random.default_rng(2026102001+c['model_seed']-2026092701)
    acceleration=c.get('training_acceleration',{})
    mode=acceleration.get('teacher_cache','online')
    if mode not in ('online','epoch','shared'):raise ValueError('Unknown teacher acceleration mode')
    needs_teacher=c['arm'] not in ('copy_no_KD','random_no_KD')
    reduce_syncs=acceleration.get('reduce_syncs',False)
    updates=seen_l=seen_u=0;started=time.perf_counter()
    with (root/'training.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,21):
            model.train();observed=0
            total=torch.zeros((),dtype=torch.float64,device=device) if reduce_syncs else 0.
            target_cache=None
            if needs_teacher and mode=='shared':
                target_cache=SharedEpochTargets(acceleration['shared_root'],binding(c['cache'],c['teacher_checkpoint'],'r65'),view.ids,epoch)
            elif needs_teacher and mode=='epoch' and exposure_count(len(view))>len(view):
                target_cache=EpochTargets(teacher,len(view),epoch,binding(c['cache'],c['teacher_checkpoint'],'r65'),
                    predict=lambda x:teacher.stage_outputs(x)[0][-2:],storage=acceleration.get('storage','auto'),
                    device_limit_mib=acceleration.get('device_limit_mib',64),max_mib=acceleration.get('max_mib',512))
            for selected in batches(len(view),rng):
                values=view.batch(selected,epoch)
                received=tensor(values['received'],device);clean=tensor(values['clean'],device)
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=learning_rate(updates)
                desired=target_cache.get(selected,received,epoch) if target_cache is not None else None
                objective=loss(model,teacher,c['arm'],received,clean,epoch,desired=desired)
                if not bool(torch.isfinite(objective).all()):raise FloatingPointError('Nonfinite G ablation objective')
                objective.mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                total+=objective.detach().sum().double() if reduce_syncs else float(objective.detach().sum())
                observed+=len(selected);updates+=1
                seen_l+=int(view.is_L[selected].sum());seen_u+=int((~view.is_L[selected]).sum())
            row=dict(epoch=epoch,updates=updates,loss=float(total)/observed,L_exposures=seen_l,U_exposures=seen_u,seconds=time.perf_counter()-started)
            if acceleration:row['teacher_cache']=target_cache.stats() if target_cache else None
            del target_cache
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();write(root/'progress.json',row,exclusive=False)
            print('G_ABLATION_EPOCH',c['arm'],row,flush=True)
    if updates!=4940 or any(p.grad is not None for p in teacher.parameters()):raise ValueError('G complete budget/frozen teacher violation')
    ck=dict(schema='cric_dura_G_ablation_v1',model=model.state_dict(),source_contract=contract,config=c,
        epoch=20,updates=4940,TX_labels_read=0,F_reads=0,target_reads=0,code_commit=c['code_commit'])
    torch.save(ck,root/'final.pt')
    fresh=build(load_checkpoint(c['teacher_checkpoint'],'cpu')[0],c['arm']).to(device)
    fresh.load_state_dict(torch.load(root/'final.pt',map_location='cpu',weights_only=False)['model'],strict=True)
    with torch.no_grad():
        model.eval();fresh.eval()
        if not torch.allclose(model(received[:2]),fresh(received[:2]),atol=1e-6,rtol=1e-5):raise ValueError('G ablation reload mismatch')
    write(root/'complete.json',dict(status='SOURCE_FROZEN',arm=c['arm'],epoch=20,updates=4940,
        parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-started,
        L_exposures=seen_l,U_exposures=seen_u,TX_labels_read=0,F_reads=0,target_reads=0))
