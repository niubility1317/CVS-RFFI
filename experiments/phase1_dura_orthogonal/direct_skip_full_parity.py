"""After fixed predictions, replay direct FIR decisions without DURA or truth."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_innovation_probe.source_probe import SOURCE_PILOT, source_inputs
from experiments.phase1_dura_orthogonal.ground import load_fixed
from experiments.phase1_dura_orthogonal.source_direct_skip_probe import run_batch


def wait_for_fixed(root, source, ground):
    reference=root/'runs'/ground/'flow/test_prediction'
    controller=root/'runs'/(source+'-controller')/'status.json'
    while not (reference/'complete.json').is_file():
        if controller.exists() and core.read(controller).get('state') == 'FAILED':
            raise RuntimeError('Original prediction controller failed; preserve its outputs')
        time.sleep(20)
    return reference


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--ground',required=True)
    p.add_argument('--candidate-commit',required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique parity output required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model,dura,reader=load_fixed(args.root,args.source,args.candidate_commit)
    ids,rx,scene,_,received,_=source_inputs(args.root)
    positions=[]
    for receivers in ((4,6),(8,)):
        for view in core.VIEWS:
            found=np.flatnonzero(np.isin(rx,receivers)&(scene==view))
            if len(found)<8:
                raise ValueError('Eight source records per group required')
            positions.extend(found[:8].tolist())
    positions=np.asarray(positions,dtype=np.int64)
    with np.load(args.root/'runs'/SOURCE_PILOT/'source_predictions.npz',allow_pickle=False) as z:
        expected=z['values'][positions][:,[3,5]].astype(np.int64)
    for start in range(0,len(positions),8):
        x=core.tensor(received[ids[positions[start:start+8]]],'cpu')
        original=run_batch(x,model,dura,reader,True)
        fast=run_batch(x,model,dura,reader,False)
        if any(not torch.equal(original[i],fast[i]) for i in (0,1,3)):
            raise ValueError('Source waveform or logits differ')
        actual=np.from_dlpack(torch.stack((original[1].argmax(1),original[2].argmax(1)),1).cpu().contiguous())
        if not np.array_equal(actual,expected[start:start+len(x)]):
            raise ValueError('Fixed source raw/anchor pilot mismatch')
    args.out.mkdir(parents=True)
    core.write(args.out/'source_smoke.json',dict(status='PASS',count=len(positions),
        waveform_and_logits_elementwise_equal=True,fixed_pilot_mismatches=0,target_reads=0,TX_truth_reads=0))
    print('DIRECT_SKIP_SOURCE_SMOKE_PASS',len(positions),flush=True)
    print('WAITING_FOR_FIXED_SEVEN_VIEW_PREDICTIONS',flush=True)
    reference=wait_for_fixed(args.root,args.source,args.ground)
    folder,ids=core.reference(args.root)
    manifest=core.read(reference/'complete.json')
    core.validate_prediction_manifest(manifest,count=len(ids),commit=args.candidate_commit)
    if not np.array_equal(ids,np.load(reference/'physical_ids.npy',allow_pickle=False)):
        raise ValueError('Fixed prediction physical IDs must match')
    started=time.monotonic()
    results={}
    for view in core.VIEWS:
        begin=time.monotonic()
        values=np.load(folder/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        raw_ref=np.load(reference/(view+'.raw.npy'),mmap_mode='r',allow_pickle=False)
        restored_ref=np.load(reference/(view+'.restored.npy'),mmap_mode='r',allow_pickle=False)
        if (values.shape!=(len(ids),2,256) or values.dtype!=np.float32 or
                raw_ref.shape!=(len(ids),) or restored_ref.shape!=(len(ids),)):
            raise ValueError('Complete fixed inputs and reference arrays required')
        for start in range(0,len(ids),8):
            x=core.tensor(values[start:start+8],'cpu')
            _,raw,_,restored=run_batch(x,model,dura,reader,False)
            end=start+len(x)
            if not np.array_equal(np.from_dlpack(raw.argmax(1).cpu().contiguous()),raw_ref[start:end]):
                raise ValueError('Raw decision mismatch at '+view+' '+str(start))
            if not np.array_equal(np.from_dlpack(restored.argmax(1).cpu().contiguous()),restored_ref[start:end]):
                raise ValueError('Restored decision mismatch at '+view+' '+str(start))
            if end%16384==0:
                print('DIRECT_SKIP_PROGRESS',view,end,round(time.monotonic()-started,1),flush=True)
        results[view]=dict(count=len(ids),raw_mismatches=0,restored_mismatches=0,elapsed_seconds=time.monotonic()-begin)
        print('DIRECT_SKIP_VIEW_COMPLETE',view,len(ids),flush=True)
    core.write(args.out/'result.json',dict(status='FULL_DECISION_PARITY_PASS',commit=args.commit,
        candidate=args.source,candidate_commit=args.candidate_commit,ground_reference=args.ground,
        F_run=core.F_RUN,F_row=core.F_ROW,batch_size=8,threads=2,views=results,
        elapsed_seconds=time.monotonic()-started,TX_truth_reads=0,model_updates=0,
        scope='Raw and restored class decisions only; ground logits and waveforms not stored in reference'))
    print('DIRECT_SKIP_FULL_PARITY_PASS',flush=True)


if __name__=='__main__':
    main()
