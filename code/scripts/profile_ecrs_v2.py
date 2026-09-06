"""Bounded synthetic ECRS V2 component profile; no dataset or accuracy claims."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvsrffi.ecrs_v2 import ECRSV2Branch


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values)-1)*fraction
    left = int(position)
    right = min(left+1, len(values)-1)
    return values[left] + (values[right]-values[left])*(position-left)


def measure(operation, device, warmup, steps):
    for _ in range(warmup):
        operation()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline = None
    durations = []
    for _ in range(steps):
        if device.type == 'cuda':
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            operation()
            stop.record()
            stop.synchronize()
            durations.append(start.elapsed_time(stop))
        else:
            start = time.perf_counter()
            operation()
            durations.append((time.perf_counter()-start)*1000)
    peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    return {'p50_ms': statistics.median(durations), 'p95_ms': percentile(durations, .95),
            'samples_ms': durations, 'peak_vram_allocated_bytes': peak,
            'baseline_vram_allocated_bytes': baseline,
            'incremental_peak_vram_bytes': None if peak is None else max(0, peak-baseline),
            'timer': 'cuda_events' if device.type == 'cuda' else 'perf_counter'}


def profile(*, batch=2, input_len=64, device='cpu', warmup=1, steps=3, amp=False):
    if batch < 1 or input_len < 16 or warmup < 0 or steps < 1:
        raise ValueError('batch>=1, input_len>=16, warmup>=0 and steps>=1 are required')
    dev = torch.device(device)
    if dev.type not in ('cpu', 'cuda') or (amp and dev.type != 'cuda'):
        raise ValueError('use CPU FP32 or CUDA with optional AMP')
    torch.manual_seed(20260906)
    branch = ECRSV2Branch(fusion_mode='fixed').to(dev).eval()
    x, raw = torch.randn(batch,2,input_len,device=dev), torch.randn(batch,160,device=dev)
    with torch.no_grad():
        output = branch(x,raw,return_diagnostics=True)
    encoder_input = output['encoder_input']

    def physical():
        branch.physical(x)

    def crossfit():
        branch.physical.cross_fit(x)

    def encoder():
        with torch.no_grad(), torch.autocast(dev.type, enabled=amp):
            branch.encoder(encoder_input)

    def inference():
        with torch.no_grad(), torch.autocast(dev.type, enabled=amp):
            branch(x,raw)

    def backward():
        branch.zero_grad(set_to_none=True)
        with torch.autocast(dev.type, enabled=amp):
            out = branch(x,raw)
            loss = out['z_resp'][:,0].float().sum() + out['z_id_fused'][:,0].float().sum()
        loss.backward()

    timings = {}
    for name, function in [('physical',physical), ('crossfit_two_directions',crossfit),
                           ('response_encoder',encoder), ('full_inference',inference)]:
        timings[name] = measure(function,dev,warmup,steps)
    branch.train()
    timings['train_forward_backward'] = measure(backward,dev,warmup,steps)
    diagnostics = branch.physical.cross_fit(x)
    crossfit_quality = []
    for d in diagnostics:
        crossfit_quality.append({name: d[name].detach().cpu().tolist() for name in
            ('delta_pred','nmse_full','nmse_nuisance_only','denominator_energy','evaluation_valid')})
        crossfit_quality[-1].update(fit_quality_valid=d['fit']['quality_valid'].cpu().tolist(),
            effective_points=d['effective_points'], discarded_boundary_points=d['discarded_boundary_points'], guard=d['guard'])
    return {'schema':'ecrs_v2_synthetic_profile_v1', 'synthetic':True,
        'claim_scope':'component timing and diagnostic arithmetic only; no real-IQ accuracy, identifiability or speedup claim',
        'full_inference_scope':'ECRSV2Branch with synthetic raw embedding; ADV3B02 backbones excluded',
        'train_scope':'forward+backward, without optimizer step or data loading',
        'hardware':{'device':str(dev), 'actual_gpu':torch.cuda.get_device_name(dev) if dev.type=='cuda' else None,
            'torch':torch.__version__, 'cuda_runtime':torch.version.cuda, 'platform':platform.platform()},
        'config':{'batch':batch,'input_shape':[batch,2,input_len],'raw_shape':[batch,160],
            'warmup':warmup,'steps':steps,'physical_precision':'float32/complex64',
            'neural_precision':'autocast_default' if amp else 'float32', 'amp':amp},
        'timings':timings,'crossfit':crossfit_quality,'metadata':branch.bundle_metadata()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--input-len',type=int,default=64)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--amp',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output already exists; choose a new path')
    result = profile(batch=args.batch,input_len=args.input_len,device=args.device,
                     warmup=args.warmup,steps=args.steps,amp=args.amp)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output':str(args.output.resolve()),'synthetic':True,
                      'timings':result['timings']},allow_nan=False))


if __name__=='__main__':
    main()
