"""Value-preserving telemetry readback for future workers only."""
import math
import json
import time
from pathlib import Path
from contextlib import contextmanager
import torch


def batch_clean(value):
    groups={}
    def collect(v):
        if torch.is_tensor(v):
            if v.numel()==1 and v.device.type!='cpu':groups.setdefault((v.device,v.dtype),{})[id(v)]=v.detach().reshape(())
        elif isinstance(v,dict):
            for x in v.values():collect(x)
        elif isinstance(v,(tuple,list)):
            for x in v:collect(x)
    collect(value);scalars={}
    for tensors in groups.values():
        values=torch.stack(list(tensors.values())).cpu().tolist()
        scalars.update(zip(tensors,values))
    def convert(v):
        if torch.is_tensor(v):return convert(scalars[id(v)] if id(v) in scalars else v.detach().cpu().item() if v.numel()==1 else v.detach().cpu().tolist())
        if isinstance(v,float) and not math.isfinite(v):return None
        if isinstance(v,dict):return {str(k):convert(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return [convert(x) for x in v]
        return v
    return convert(value)


POLICY = dict(a1_gradient_snapshot=True, source_validation_reuse=True,
              native_epoch_telemetry='incremental', scalar_readback='batched_same_dtype',
              compact_epoch_telemetry='original_full_history', a1_runtime_fast=False)


def configure(args):
    args.a1_gradient_snapshot = True
    args.source_validation_reuse = True
    # Deliberately leave the broad fast switch, EMA, numerical policy and all
    # scientific options as in the original immutable worker release.
    if args.a1_runtime_fast or args.incremental_epoch_telemetry:
        raise ValueError('Unexpected broad runtime switch in original recipe')
    return args


@contextmanager
def incremental_native_writer(native):
    from cvsrffi.incremental_telemetry import IncrementalTelemetry
    original=native._write_ssdg_epoch_telemetry
    original_eval=native.evaluate_loader
    original_geometry=native._evaluate_source_val_tail_geometry
    writers={};timing={}
    def evaluate(*args,**kwargs):
        start=time.perf_counter()
        result=original_eval(*args,**kwargs)
        if kwargs.get('feature_cache') is not None:
            timing['base_validation_s']=time.perf_counter()-start
        return result
    def geometry(*args,**kwargs):
        start=time.perf_counter();result=original_geometry(*args,**kwargs)
        timing['geometry_validation_s']=time.perf_counter()-start
        cache=kwargs.get('feature_cache')
        if cache is not None:
            timing.update(cache_hits=cache.hits,cache_misses=cache.misses,cache_allowed=cache.allowed,cache_bytes=cache.used)
        return result
    def write(cp,jp,rows):
        key=(str(cp),str(jp))
        if key not in writers:writers[key]=IncrementalTelemetry(cp,jp)
        start=time.perf_counter()
        writers[key].write(rows)
        measured=dict(epoch=rows[-1]['epoch'],native_telemetry_io_s=time.perf_counter()-start,**timing)
        with (Path(jp).parent/'execution_epoch.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(measured,allow_nan=False)+'\n')
        print('[EXECUTION] '+json.dumps(measured),flush=True)
        timing.clear()
    native._write_ssdg_epoch_telemetry=write
    native.evaluate_loader=evaluate;native._evaluate_source_val_tail_geometry=geometry
    try:yield native
    finally:
        native._write_ssdg_epoch_telemetry=original
        native.evaluate_loader=original_eval;native._evaluate_source_val_tail_geometry=original_geometry
