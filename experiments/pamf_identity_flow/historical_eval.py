"""Frozen seven-view historical F50 evaluation; CPU truth scoring is separate."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from . import source_train as candidate
from experiments.pamf_dura_direct_train import runner as base
from experiments.pamf_dura_direct_train import evaluation
from experiments.pamf_fcrossrx_readout.frozen import load_reader


def paths(root):
    value=base.paths(root);value['output']=root/'runs'/candidate.RUN
    return value


@torch.no_grad()
def predict(args,p,role):
    done=base.read_json(p['output']/candidate.ARM/'training_complete.json')
    if (done['status'],done['epoch'],done['updates'],done['F0_frozen'],done['target_IQ_reads']) != ('SOURCE_TRAINED',candidate.EPOCHS,candidate.STEPS,True,0):
        raise ValueError('Fixed source-trained flow required')
    folder,ids,_,_=evaluation.reference(p,role)
    model=candidate.load(p['output']/candidate.ARM/'final.pt','cuda:0',args.commit)
    reader=load_reader(p['reader'],p['source_cache'],'cuda:0').eval().requires_grad_(False)
    out=p['output']/candidate.ARM/(role+'_prediction');out.mkdir()
    timings={};started=time.perf_counter()
    for view in base.VIEWS:
        values=np.load(folder/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        if values.shape!=(len(ids),2,256) or values.dtype!=np.float32:raise ValueError('Fixed full received view required')
        iq=np.lib.format.open_memmap(out/(view+'.iq.npy'),mode='w+',dtype=np.float32,shape=values.shape)
        logits=np.lib.format.open_memmap(out/(view+'.logits.npy'),mode='w+',dtype=np.float32,shape=(len(ids),6))
        raw=np.lib.format.open_memmap(out/(view+'.raw_logits.npy'),mode='w+',dtype=np.float32,shape=(len(ids),6)) if role=='source' else None
        begin=time.perf_counter()
        for offset in range(0,len(ids),128):
            x=base.tensor(values[offset:offset+128],'cuda:0');y=model(x);scores=reader(y)
            if y.shape!=x.shape or scores.shape!=(len(x),6) or not torch.isfinite(y).all() or not torch.isfinite(scores).all():
                raise ValueError('Finite frozen restorer/F50 prediction required')
            iq[offset:offset+len(x)]=base.numpy_values(y);logits[offset:offset+len(x)]=base.numpy_values(scores)
            if raw is not None:raw[offset:offset+len(x)]=base.numpy_values(reader(x))
        iq.flush();logits.flush()
        if raw is not None:raw.flush()
        timings[view]=time.perf_counter()-begin
        print(json.dumps(dict(event='FROZEN_VIEW_COMPLETE',role=role,view=view,count=len(ids))),flush=True)
    np.save(out/'physical_ids.npy',ids,allow_pickle=False)
    base.write_json(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(base.VIEWS),role=role,
        arm=candidate.ARM,F_run=base.F_RUN,F_seed=2026092701,original_default_head=True,TX_truth_reads=0,
        state_updates=0,head_fit_updates=0,code_commit=args.commit,seconds=time.perf_counter()-started,
        per_view_seconds=timings,inference=candidate.config()['inference']))


@torch.no_grad()
def profile(args,p):
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    model=candidate.load(p['output']/candidate.ARM/'final.pt','cuda:0',args.commit)
    baseline,_=base.parent_model(p,'cuda:0');baseline.eval().requires_grad_(False)
    reader=load_reader(p['reader'],p['source_cache'],'cuda:0').eval().requires_grad_(False)
    x=base.tensor(np.load(p['cache']/'validation/practical_low_urban.npy',mmap_mode='r',allow_pickle=False)[:128],'cuda:0')
    mac=evaluation.conv_linear_mac(model,x[:1]);results={}
    torch.cuda.reset_peak_memory_stats()
    for scope in ('G','G_plus_F'):
        old=InferenceEngine(baseline,reader if scope=='G_plus_F' else None,reuse=True,graph=True,native=False)
        def new(z):
            restored=model(z)
            return reader(restored) if scope=='G_plus_F' else restored
        for batch in (1,128):
            times={'DURA_graph':[],'flow_eager':[]}
            for repeat in range(3):
                pairs=[('DURA_graph',old),('flow_eager',new)]
                if repeat%2:pairs.reverse()
                for name,fn in pairs:
                    for _ in range(3):fn(x[:batch])
                    torch.cuda.synchronize()
                    for _ in range(16):
                        t=time.perf_counter();fn(x[:batch]);torch.cuda.synchronize()
                        times[name].append((time.perf_counter()-t)*1000)
            results[scope+'_B'+str(batch)]={name:dict(median_ms=float(np.median(v)),p95_ms=float(np.quantile(v,.95)),
                throughput_samples_s=float(batch*1000/np.mean(v)),iterations=len(v)) for name,v in times.items()}
    base.write_json(p['output']/candidate.ARM/'matched_profile.json',dict(status='PROFILE_COMPLETE',cases=results,
        parameters=sum(v.numel() for v in model.parameters()),ConvLinear_MAC=mac,
        checkpoint_bytes=(p['output']/candidate.ARM/'final.pt').stat().st_size,velocity_field_calls=8,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],
        timing_scope='Same GPU FP32; DURA graph vs fixed eight-field-call eager candidate; preloaded input, setup excluded',
        memory_scope='Shared resident F and both G models; not standalone VRAM'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('predict','score','profile'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--commit',required=True)
    parser.add_argument('--role',choices=('source','target'));args=parser.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=paths(args.root);args.arm=candidate.ARM
    if args.mode=='profile':profile(args,p)
    elif args.mode=='predict':
        if args.role is None:parser.error('Role required')
        predict(args,p,args.role)
    else:
        if args.role is None:parser.error('Role required')
        original_write=evaluation.write_json
        def write_with_provenance(path,value):
            value.update(candidate_training_F_access=True,
                candidate_source_guidance='Frozen original Phase3 F0 source feature anchor and path agreement',
                candidate_source_training_epochs=candidate.EPOCHS,candidate_source_training_updates=candidate.STEPS,
                evaluation_neural_updates=0)
            return original_write(path,value)
        evaluation.write_json=write_with_provenance
        evaluation.score(args,p,args.role)


if __name__=='__main__':main()
