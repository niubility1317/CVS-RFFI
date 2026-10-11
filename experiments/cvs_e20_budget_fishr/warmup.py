"""Source-L cached-feature readout warmup; identity parameters never optimized."""
import json
import time
from pathlib import Path

import torch
from . import design as d


def _rss():
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return None


def head_warmup(c,model,L,device,output):
    output=Path(output)
    if (output/'head_warmup.json').exists() or (output/'head_warmup.jsonl').exists():
        raise FileExistsError('Warm-head evidence already exists; refusing overwrite')
    plan=c['multilevel'];cuda=str(device).startswith('cuda')
    if not plan['enabled']:
        result=dict(status='NOT_APPLICABLE',enabled=False,source_L_cache_packets=0,cache_forward_calls=0,
                    head_packet_occurrences=0,optimizer_steps=0,seconds=0.,cache_tensor_bytes=0,
                    optimizer_state_tensor_bytes=0,base_state_unchanged=True,backbone_updates=0,
                    peak_cuda_memory_bytes=0,rss_bytes=None,auxiliary_inheritance=False)
        (output/'head_warmup.jsonl').open('x',encoding='utf-8').close()
        d.write(output/'head_warmup.json',result)
        return result
    if len(L['ids'])!=6300 or len(set(L['ids']))!=6300:
        raise ValueError('Warm heads require exactly the fixed 6300 unique source L packets')
    if plan['head_warmup_steps']!=200 or plan['head_warmup_batch']!=60:
        raise ValueError('Readout warmup must be fixed 200 updates of balanced B60')
    if plan['head_warmup_lr']!=.003 or plan['head_warmup_weight_decay']!=.0001:
        raise ValueError('Readout warmup optimizer differs from fixed plan')
    started=time.perf_counter();before={k:v.detach().cpu().clone() for k,v in model.base.state_dict().items()}
    modes=[(m,m.training) for m in model.modules()]
    rng=torch.random.get_rng_state().clone()
    devices=sorted({p.device.index for p in model.parameters() if p.device.type=='cuda'})
    cached={level:[] for level in plan['levels']};forward_calls=0
    if cuda:torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval();cache_started=time.perf_counter()
            for start in range(0,6300,128):
                with torch.no_grad():
                    batch=model.multilevel.extract_clean_features(model,L['x'][start:start+128].to(device))
                for level,value in batch.items():cached[level].append(value.detach().cpu())
                forward_calls+=1
            cached={level:torch.cat(values) for level,values in cached.items()}
            if any(len(value)!=6300 or value.requires_grad for value in cached.values()):
                raise ValueError('Invalid detached source feature cache')
            if cuda:torch.cuda.synchronize(device)
            cache_seconds=time.perf_counter()-cache_started
            optimizer=torch.optim.AdamW(model.multilevel.heads.parameters(),lr=.003,weight_decay=.0001)
            head_started=time.perf_counter();last_loss=None;first_loss=None
            with (output/'head_warmup.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
                for step in range(1,201):
                    tick=time.perf_counter()
                    index=model.multilevel.balanced_indices(L,step*plan['interval'],plan['interval'])
                    features={level:value[index].to(device) for level,value in cached.items()}
                    labels=L['y'][index].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss=model.multilevel.probe_ce(features,labels)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite warm-head CE')
                    loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.multilevel.heads.parameters(),float('inf'),error_if_nonfinite=True)
                    optimizer.step()
                    if cuda:torch.cuda.synchronize(device)
                    last_loss=float(loss.detach());first_loss=last_loss if first_loss is None else first_loss
                    row=dict(step=step,source_L_packets=60,CE=last_loss,head_gradient_norm=float(norm),
                             lr=.003,weight_decay=.0001,backbone_updates=0,seconds=time.perf_counter()-tick)
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                    if step in (1,50,100,150,200):
                        stream.flush();print('HEAD_WARMUP '+json.dumps(row),flush=True)
            optimizer.zero_grad(set_to_none=True)
            head_seconds=time.perf_counter()-head_started
            for name,value in model.base.state_dict().items():
                if not torch.equal(value.detach().cpu(),before[name]):
                    raise ValueError('Readout warmup changed identity base state: '+name)
            from experiments.cvs_stage2_rx_risk.source import state_bytes
            result=dict(status='HEAD_WARMUP_COMPLETE',enabled=True,source_role='L_s only',
                source_L_cache_packets=6300,cache_forward_calls=forward_calls,head_packet_occurrences=12000,
                optimizer_steps=200,backbone_updates=0,base_state_unchanged=True,
                optimizer='fresh AdamW for heads only; discarded before main AdamW',lr=.003,weight_decay=.0001,
                initial_head_loss=first_loss,final_head_loss=last_loss,cache_seconds=cache_seconds,
                head_optimization_seconds=head_seconds,seconds=time.perf_counter()-started,
                cache_tensor_bytes=sum(v.numel()*v.element_size() for v in cached.values()),
                optimizer_state_tensor_bytes=state_bytes(optimizer.state_dict()),
                peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,
                rss_bytes=_rss(),rss_unavailable_reason=None if _rss() is not None else 'psutil unavailable',
                auxiliary_inheritance=False,old_features_consumed=False,target_access=False)
    finally:
        for module,mode in modes:module.training=mode
    if not torch.equal(rng,torch.random.get_rng_state()):raise ValueError('Warm heads changed native RNG')
    d.write(output/'head_warmup.json',result)
    return result
