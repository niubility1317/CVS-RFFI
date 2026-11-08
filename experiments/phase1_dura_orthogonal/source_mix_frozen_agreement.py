"""Source-only frozen clean-class agreement for fixed DURA/direct mixtures."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs, SOURCE_PILOT
from experiments.phase1_dura_orthogonal.source_direct_dura_mix import load_direct
from experiments.phase1_dura_orthogonal.signal_mix_gate_ground import load_gate as load_adaptive
from experiments.phase1_dura_orthogonal.signal_balanced_mix_ground import load_gate as load_balanced
from experiments.phase1_dura_orthogonal.source_frozen_agreement import classes, summarize
from experiments.phase1_dura_orthogonal.train_source import extract_context

ADAPTIVE='20261108-phase1-dura-direct-adaptive-mix-source-s2026092701-r01'
ADAPTIVE_COMMIT='708a3d8a9b1ce57907de5a26f9b6bbf6ba9553ab'
BALANCED='20261108-phase1-dura-direct-balanced-mix-source-s2026092701-r02'
BALANCED_COMMIT='597b621aa76558c4ba6fdea3487c7f89eb904639'


def paired(proxy, dura, candidate):
    a=np.asarray(dura)==np.asarray(proxy)
    b=np.asarray(candidate)==np.asarray(proxy)
    return dict(gains=int((~a & b).sum()), losses=int((a & ~b).sum()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Unique source diagnostic output required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ids,rx,scene,_,received,clean=source_inputs(args.root)
    with np.load(args.root/'runs'/SOURCE_PILOT/'source_predictions.npz',allow_pickle=False) as z:
        if not np.array_equal(z['indices'],ids):raise ValueError('Fixed source pilot IDs required')
        saved=z['values'][:,[1,3,5]].astype(np.int64)
    for source in (ADAPTIVE,BALANCED):
        state=torch.load(args.root/'runs'/source/'final.pt',map_location='cpu',weights_only=False)
        if np.intersect1d(ids,np.asarray(state['fit_ids'])).size:raise ValueError('Source pilot must be disjoint from gate fit')
    direct=load_direct(args.root,ids)
    adaptive=load_adaptive(args.root,ADAPTIVE,ADAPTIVE_COMMIT)
    balanced=load_balanced(args.root,BALANCED,BALANCED_COMMIT)
    dura=frozen_dura(args.root,'cpu')
    reader,_=core.teacher(args.root,'cpu')
    reader=reader.eval().requires_grad_(False)
    batches=[]
    with torch.no_grad():
        for start in range(0,len(ids),8):
            sel=ids[start:start+8]
            x,y=core.tensor(received[sel],'cpu'),core.tensor(clean[sel],'cpu')
            anchor,direction,variance=extract_context(dura,x)
            proposal=direct(x,validate=False)
            candidates=(x,anchor,proposal,(anchor+proposal)*.5,adaptive(x,anchor,direction,proposal,variance),balanced(x,anchor,direction,proposal,variance))
            columns=[classes(reader,y),*(classes(reader,z) for z in candidates)]
            block=np.from_dlpack(torch.stack(columns,1).cpu().contiguous()).copy()
            if not np.array_equal(block[:,:3],saved[start:start+len(x)]):raise ValueError('Fixed clean, raw and DURA source predictions required')
            batches.append(block)
            if start==0:print('MIX_FROZEN_PROXY_SMOKE_PASS',len(x),flush=True)
            if (start+len(x))%168==0:print('MIX_FROZEN_PROXY_PROGRESS',start+len(x),flush=True)
    values=np.concatenate(batches,0)
    names=['raw','dura','direct','half','adaptive','balanced']
    predictions={name:values[:,i+1] for i,name in enumerate(names)}
    impaired=scene!='clean'
    pairs={name:paired(values[impaired,0],predictions['dura'][impaired],predictions[name][impaired]) for name in names}
    args.out.mkdir(parents=True)
    with (args.out/'source_predictions.npz').open('xb') as f:
        np.savez(f,ids=ids,rx=rx,scene=scene,clean_proxy=values[:,0],**predictions)
    result=dict(status='MIX_FROZEN_PROXY_COMPLETE',commit=args.commit,records=len(ids),
        metric='agreement_with_frozen_clean_pseudoclass_not_accuracy',
        groups=summarize(values[:,0],predictions,rx,scene,views=core.VIEWS),paired_vs_dura_impaired=pairs,
        adaptive_source=ADAPTIVE,balanced_source=BALANCED,TX_labels_read=0,U_s_truth_reads=0,
        V_reads=0,target_reads=0,model_updates=0,
        note='Frozen F used only for post-training source analysis; previously explored clean pseudo-label is not TX truth.')
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('MIX_FROZEN_PROXY_COMPLETE',len(ids),flush=True)


if __name__=='__main__':main()
