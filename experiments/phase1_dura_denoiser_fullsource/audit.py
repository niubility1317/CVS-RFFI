"""Independent set-based recount of fixed source pair scores."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.phase1_dura_denoiser_fullsource.score import VIEWS,load_fixed,source_truth

def count_sets(target,baseline,candidate,mask):
    eligible=set(np.flatnonzero(mask).tolist())
    first={i for i in eligible if baseline[i]==target[i]}
    second={i for i in eligible if candidate[i]==target[i]}
    return dict(count=len(eligible),baseline_correct=len(first),candidate_correct=len(second),
                repairs=len(second-first),damages=len(first-second))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--score',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Unique audit path required')
    ids,split,rx,scene,pred,old=load_fixed(args.root,args.source,args.commit)
    truth,known=source_truth(args.root,ids);pseudo=old[:,0].astype(np.int64)
    trusted=old[:,1]>=.95;dura=old[:,2].astype(np.int64)
    score=json.loads((args.score/'source_score.json').read_text())
    if score.get('status')!='SOURCE_SCORE_COMPLETE' or not score.get('fixed_predictions_before_truth'):
        raise ValueError('Complete truth-last score required')
    checked=0
    for group,part in (('RX4_6',1),('RX8',2)):
        for view in ('all_impaired',)+VIEWS:
            cell=(split==part)&((scene!='clean') if view=='all_impaired' else (scene==view))
            for kind,target,eligible in (('L_s',truth,known),('pseudo',pseudo,trusted)):
                mask=cell&eligible
                for name,baseline,candidate in (
                    ('context_vs_DURA',dura,pred[:,0]),
                    ('denoiser_vs_DURA',dura,pred[:,1]),
                    ('denoiser_vs_context',pred[:,0],pred[:,1])):
                    expected=score['results'][group][view][kind][name]
                    if count_sets(target,baseline,candidate,mask)!=expected:
                        raise ValueError('Independent recount differs: '+group+'/'+view+'/'+kind+'/'+name)
                    checked+=1
    if checked!=96:raise ValueError('All source cells must be checked')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(status='PASS',groups=16,comparisons=checked,fields=checked*5,
                        count=len(ids),L_s_scored=int(known.sum()),U_s_truth_reads=0,V_reads=0,test_reads=0),indent=2)+'\n',encoding='utf8')
    print('FULLSOURCE_PAIR_AUDIT_PASS groups=16 comparisons=96 fields=480',flush=True)

if __name__=='__main__':main()
