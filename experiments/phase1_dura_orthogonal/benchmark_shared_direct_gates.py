"""Source-only exact and timed sharing for two fixed DURA/direct gates."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.signal_mix_gate_ground import load_gate as load_adaptive
from experiments.phase1_dura_orthogonal.signal_balanced_mix_ground import load_gate as load_balanced
from experiments.phase1_dura_orthogonal.source_direct_dura_mix import load_direct
from experiments.phase1_dura_orthogonal.shared_source_benchmark import classify, same_bits
from experiments.phase1_dura_orthogonal.train_source import extract_context

ADAPTIVE='20261108-phase1-dura-direct-adaptive-mix-source-s2026092701-r01'
ADAPTIVE_COMMIT='708a3d8a9b1ce57907de5a26f9b6bbf6ba9553ab'
BALANCED='20261108-phase1-dura-direct-balanced-mix-source-s2026092701-r02'
BALANCED_COMMIT='597b621aa76558c4ba6fdea3487c7f89eb904639'
FIELDS=('anchor','direction','variance','direct','restored','raw_logits','anchor_logits','restored_logits')


def predict(x,context,direct,reader,gates,shared):
    def common():
        a,d,v=context(x)
        p=direct(x)
        return a,d,v,p,classify(reader,x),classify(reader,a)
    common_result=common() if shared else None
    rows=[]
    for gate in gates:
        a,d,v,p,raw,anchor_logits=common_result if shared else common()
        restored=gate(x,a,d,p,v)
        rows.append((a,d,v,p,restored,raw,anchor_logits,classify(reader,restored)))
    return rows


def compare(a,b):
    if len(a)!=2 or len(b)!=2:raise ValueError('Both frozen gate outputs required')
    for first,second in zip(a,b):
        if len(first)!=len(FIELDS) or len(second)!=len(FIELDS):raise ValueError('All output fields required')
        for name,x,y in zip(FIELDS,first,second):
            if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not same_bits(x,y):
                raise ValueError('Nonfinite or different source output: '+name)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Unique source benchmark output required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ids,rx,scene,_,received,_=source_inputs(args.root)
    for source in (ADAPTIVE,BALANCED):
        state=torch.load(args.root/'runs'/source/'final.pt',map_location='cpu',weights_only=False)
        if np.intersect1d(ids,np.asarray(state['fit_ids'])).size:raise ValueError('Source pilot must be gate-disjoint')
    direct=load_direct(args.root,ids)
    gates=(load_adaptive(args.root,ADAPTIVE,ADAPTIVE_COMMIT),load_balanced(args.root,BALANCED,BALANCED_COMMIT))
    dura=frozen_dura(args.root,'cpu')
    reader,_=core.teacher(args.root,'cpu');reader=reader.eval().requires_grad_(False)
    for obj in (direct,*gates,dura,reader):
        if any(p.requires_grad for p in obj.parameters()):raise ValueError('Fixed frozen models required')
    positions=[]
    for receivers in ((4,6),(8,)):
        for view in core.VIEWS:
            cell=np.flatnonzero(np.isin(rx,receivers)&(scene==view))
            if len(cell)!=48:raise ValueError('Complete fixed source groups required')
            positions.extend(cell.tolist())
    selected=ids[np.asarray(positions)]
    batches=[core.tensor(received[selected[i:i+8]],'cpu') for i in range(0,len(selected),8)]
    context=lambda x:extract_context(dura,x)
    proposal=lambda x:direct(x,validate=False)
    with torch.no_grad():
        a=predict(batches[0],context,proposal,reader,gates,False)
        b=predict(batches[0],context,proposal,reader,gates,True)
        compare(a,b)
        print('DIRECT_GATES_SHARED_SMOKE_PASS',len(batches[0]),flush=True)
        for _ in range(2):
            predict(batches[0],context,proposal,reader,gates,False)
            predict(batches[0],context,proposal,reader,gates,True)
        pairs=[]
        for shared_first in (False,True):
            timing={False:0.,True:0.};comparisons=0
            for i,x in enumerate(batches):
                order=(shared_first,not shared_first) if i%2==0 else (not shared_first,shared_first)
                out={}
                for mode in order:
                    start=time.perf_counter();out[mode]=predict(x,context,proposal,reader,gates,mode)
                    timing[mode]+=time.perf_counter()-start
                compare(out[False],out[True]);comparisons+=len(x)
            pairs.append(dict(shared_first=shared_first,independent_seconds=timing[False],
                shared_seconds=timing[True],records=comparisons,all_eight_fields_bitwise_equal=True))
            print('DIRECT_GATES_SHARED_PAIR',shared_first,'records',comparisons,flush=True)
    args.out.mkdir(parents=True)
    result=dict(status='DIRECT_GATES_SHARED_SOURCE_COMPLETE',commit=args.commit,records=len(selected),
        fields=list(FIELDS),batch=8,cpu_threads=2,pairs=pairs,
        adaptive_source=ADAPTIVE,balanced_source=BALANCED,TX_labels_read=0,U_s_truth_reads=0,
        V_reads=0,target_reads=0,model_updates=0,
        caveat='Source-only two-model sequential CPU comparison, not full ground-test parity or parallel throughput. Timings exclude loading, comparison, and I/O.')
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(result['status'],flush=True)


if __name__=='__main__':main()
