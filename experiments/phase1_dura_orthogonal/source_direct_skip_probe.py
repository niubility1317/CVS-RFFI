"""Source-only direct-student prediction parity and CPU timing probe."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_innovation_probe.source_probe import SOURCE_PILOT, source_inputs
from experiments.phase1_dura_orthogonal.direct_student_fir import DirectStudentFIR
from experiments.phase1_dura_orthogonal.ground import load_fixed
from experiments.phase1_dura_orthogonal.train_source import extract_context


@torch.no_grad()
def run_batch(x, model, dura, reader, original):
    if not isinstance(model, DirectStudentFIR):
        raise ValueError('This probe is limited to the fixed direct FIR student')
    if original:
        anchor, direction, variance = extract_context(dura, x)
        restored = model(x, anchor, direction, variance)
    else:
        anchor = None
        restored = model(x)
    raw = reader.classify_features(reader.features(x))
    anchor_logits = reader.classify_features(reader.features(anchor)) if original else None
    candidate = reader.classify_features(reader.features(restored))
    if not all(torch.isfinite(v).all() for v in (restored,raw,candidate)):
        raise FloatingPointError('Finite source outputs required')
    return restored,raw,anchor_logits,candidate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--candidate-commit',required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source diagnostic output required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ids,rx,scene,_,received,_=source_inputs(args.root)
    positions=[]
    for receivers in ((4,6),(8,)):
        for view in core.VIEWS:
            found=np.flatnonzero(np.isin(rx,receivers)&(scene==view))
            if len(found)<8:
                raise ValueError('Eight source observations per group required')
            positions.extend(found[:8].tolist())
    positions=np.asarray(positions,dtype=np.int64)
    with np.load(args.root/'runs'/SOURCE_PILOT/'source_predictions.npz',allow_pickle=False) as z:
        expected=z['values'][positions][:,[3,5]].astype(np.int64)
    model,dura,reader=load_fixed(args.root,args.source,args.candidate_commit)
    warm=core.tensor(received[ids[positions[:8]]],'cpu')
    for original in (True,False):
        run_batch(warm,model,dura,reader,original)
    durations={'original':0.,'direct':0.}
    max_abs={'waveform':0.,'raw_logits':0.,'restored_logits':0.}
    mismatches={'raw':0,'restored':0,'pilot_raw_anchor':0}
    for start in range(0,len(positions),8):
        x=core.tensor(received[ids[positions[start:start+8]]],'cpu')
        result={}
        for original in ((True,False) if start%16==0 else (False,True)):
            tick=time.perf_counter()
            result[original]=run_batch(x,model,dura,reader,original)
            durations['original' if original else 'direct']+=time.perf_counter()-tick
        standard,fast=result[True],result[False]
        for key,index in (('waveform',0),('raw_logits',1),('restored_logits',3)):
            max_abs[key]=max(max_abs[key],float((standard[index]-fast[index]).abs().max()))
        for key,index in (('raw',1),('restored',3)):
            mismatches[key]+=int((standard[index].argmax(1)!=fast[index].argmax(1)).sum())
        actual=np.from_dlpack(torch.stack((standard[1].argmax(1),standard[2].argmax(1)),1).cpu().contiguous())
        mismatches['pilot_raw_anchor']+=int(np.count_nonzero(actual!=expected[start:start+len(x)]))
    if any(mismatches.values()) or any(max_abs.values()):
        raise ValueError('Source outputs differ: '+str((mismatches,max_abs)))
    args.out.mkdir(parents=True)
    core.write(args.out/'result.json',dict(status='SOURCE_PARITY_PASS',commit=args.commit,candidate=args.source,
        candidate_commit=args.candidate_commit,count=len(positions),batch_size=8,threads=2,
        durations_seconds=durations,max_abs_diff=max_abs,decision_mismatches=mismatches,
        TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,target_reads=0,
        parity_scope='112 source records; target prediction equality not established'))
    print('DIRECT_SOURCE_PARITY_PASS',len(positions),durations,flush=True)


if __name__=='__main__':
    main()
