"""Opt-in practical v3 integration; explicit physical IDs, no TX-label consumption."""
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import torch
from leo_practical import Config, apply_leo_practical_channel_batch

PRACTICAL=('clean','practical_high','practical_mid','practical_low_urban')
_context=ContextVar('practical_context',default=None)
_last_meta=None
_recorded=set()

def practical_scenario(name):
    return dict(zip(('leo_clear_weak','leo_low_elev_weak','leo_rain_weak'),PRACTICAL[1:])).get(name,name)

def _plain(value):
    return value.detach().cpu().tolist() if torch.is_tensor(value) else list(value)

def set_training_context(meta,epoch,role):
    # base_index refers to a fixed physical record in the verified source index.
    ids=['source:'+str(i) for i in _plain(meta['base_index'])]
    sessions=['virtual_rx'+str(r)+'_day'+str(d) for r,d in zip(_plain(meta['rx_i']),_plain(meta['day_i']))]
    _context.set((ids,sessions,f'source_dynamic_E{epoch}_{role}'))

def set_evaluation_context(ids):
    # Opaque predictor package intentionally carries no receiver/TX truth.
    _context.set((list(ids),['virtual_target_session0']*len(ids),'target_fixed_v3'))

def set_source_evaluation_context(meta):
    set_training_context(meta,0,'V')
    ids,sessions,_=_context.get()
    _context.set((ids,sessions,'source_validation_fixed_v3'))

def set_smoke_context(n):
    _context.set(([f'synthetic{i}' for i in range(n)],['virtual_smoke']*n,'source_smoke'))

def practical_config(scene,args):
    return Config(fs_hz=float(args.practical_fs_hz),fc_hz=float(args.practical_fc_hz),scenario=scene,
        processing_route=args.practical_route,equalization_enabled=bool(args.practical_equalization),
        equalizer_method=args.practical_equalizer_method,zf_regularization=1e-6,equalizer_max_gain_db=20,
        input_processing_state='WiSig_ManySig_equalized1_center256_shared_normalization')

def apply_practical(x,scene,args,*,gen=None,return_meta=False):
    global _last_meta
    ctx=_context.get()
    if ctx is None or len(ctx[0])!=len(x):raise ValueError('Practical channel requires aligned physical record IDs')
    ids,sessions,namespace=ctx
    if gen is None:raise ValueError('Explicit channel generator required')
    # Evaluation does not depend on batch partition; training advances caller's view RNG.
    seed=392005 if namespace=='target_fixed_v3' else int(torch.randint(0,2**31-1,(),device=gen.device,generator=gen).item())
    cfg=practical_config(scene,args)
    # N607 Torch 2.1 / NumPy 2.2 has an unsafe ndarray ABI bridge.
    # Explicit value copies preserve float64 reference math without .numpy/from_numpy.
    array=np.array(x.detach().to(device='cpu',dtype=torch.float64).tolist(),dtype=np.float64)
    from .practical_view_cache import cached_evaluation_batch
    (y,records,states),cache_event=cached_evaluation_batch(array,cfg,
        cache_dir=getattr(args,'practical_eval_cache_dir',''),compute=apply_leo_practical_channel_batch,
        seed=seed,sample_ids=ids,session_ids=sessions,realization_namespace=namespace,
        receiver_seed=int(args.practical_receiver_seed))
    y=torch.tensor(y.tolist(),device=x.device,dtype=x.dtype)
    _last_meta={'config':asdict(cfg),'records':records,'namespace':namespace,'cache_event':cache_event}
    evidence_key=(getattr(args,'output_dir',''),namespace,scene)
    if namespace.startswith('source_dynamic_') and evidence_key not in _recorded:
        output=Path(args.output_dir)
        output.mkdir(parents=True,exist_ok=True)
        summary={'namespace':namespace,'scenario':scene,'config':asdict(cfg),'batch_size':len(records),
            'equalization_applied_count':sum(m['channel_equalization_applied'] for m in records),
            'lock_counts':{s:sum(m['receiver_lock_status']==s for m in records) for s in ('locked','degraded','unlocked')},
            'example':records[0]}
        with (output/'practical_channel_execution.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(summary,ensure_ascii=False)+'\n')
        _recorded.add(evidence_key)
    if not return_meta:
        return y,None
    states=torch.tensor(states.tolist(),device=x.device,dtype=torch.long)
    meta={'state':states,'snr_db':torch.tensor([m['quality_snr_db'] for m in records],device=x.device),
        'residual_cfo_hz':torch.tensor([m['output_frequency_hz'] for m in records],device=x.device),
        'theta_deg':torch.tensor([m['geometry']['elevation_deg'] for m in records],device=x.device),
        'h_km':torch.tensor([m['geometry']['altitude_m']/1000 for m in records],device=x.device),
        'practical_equalization_applied':torch.tensor([m['channel_equalization_applied'] for m in records],device=x.device)}
    return y,meta if return_meta else None
