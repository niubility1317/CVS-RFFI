"""Independent Phase1 paired DURA comparison, truth after sealed predictions."""
import argparse
from pathlib import Path
import numpy as np
from experiments.pamf_phase1_flow import experiment as core

BASELINE='20261101-phase1-dura-trisigtime-baseline-s2026092701-r01'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);args=ap.parse_args()
    _,ids=core.reference(args.root);new=args.out/'flow'/'test_prediction';old=args.root/'runs'/BASELINE/'flow'/'test_prediction'
    core.validate_prediction_manifest(core.read(new/'complete.json'),count=len(ids),commit=args.commit)
    old_manifest=core.read(old/'complete.json')
    expected=dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(core.VIEWS),F_run=core.F_RUN,F_row=core.F_ROW,TX_truth_reads=0,model_updates=0)
    if any(old_manifest.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen paired DURA decisions required')
    if not np.array_equal(ids,np.load(new/'physical_ids.npy',allow_pickle=False)) or not np.array_equal(ids,np.load(old/'physical_ids.npy',allow_pickle=False)):raise ValueError('Exact physical ID ordering required')
    pred={}
    for view in core.VIEWS:
        a=np.load(old/(view+'.restored.npy'),allow_pickle=False);b=np.load(new/(view+'.restored.npy'),allow_pickle=False)
        if a.shape!=(len(ids),) or b.shape!=a.shape or a.dtype!=np.int64 or b.dtype!=np.int64 or np.any((a<0)|(a>=6)|(b<0)|(b>=6)):raise ValueError('Complete predictions required')
        pred[view]=(a,b)
    truth=core.read(args.root/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json')
    labels=np.asarray([truth[str(i)]['label'] for i in ids]);rx=np.asarray([str(truth[str(i)]['receiver']) for i in ids]);day=np.asarray([str(truth[str(i)].get('day','unavailable')) for i in ids]);results={};groups=[]
    for view,(a,b) in pred.items():
        oldok=a==labels;newok=b==labels
        def metrics(mask):return dict(count=int(mask.sum()),baseline_correct=int((mask&oldok).sum()),candidate_correct=int((mask&newok).sum()),repairs=int((mask&~oldok&newok).sum()),damages=int((mask&oldok&~newok).sum()))
        results[view]=metrics(np.ones(len(ids),dtype=bool))
        for name,vec in (('TX',labels),('RX',rx),('day',day)):
            for value in np.unique(vec):groups.append(dict(view=view,dimension=name,value=str(value),**metrics(vec==value)))
    core.write(args.out/'flow'/'paired_dura.json',dict(status='PAIRED_COMPLETE',count=len(ids),views=results,groups=groups,all_predictions_before_truth=True,target_feedback=False))


if __name__=='__main__':main()
