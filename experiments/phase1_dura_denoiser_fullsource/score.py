"""Truth-last source L_s and clean-F pseudo scoring of frozen pair predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.phase1_dura_denoiser_fullsource.logic import pair_counts

VIEWS=('clean','practical_high','practical_mid','practical_low_suburban',
       'practical_high_urban','practical_mid_urban','practical_low_urban')
PRIOR='20261102-phase1-unroll-source-replay-s2026092701-r01'

def load_fixed(root,source,commit):
    m=json.loads((source/'prediction_complete.json').read_text())
    expected=dict(status='SOURCE_PREDICTIONS_FIXED',commit=commit,count=11761,pilot_replayed=672,
                  TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0,model_updates=0)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed source prediction manifest required')
    path=source/'predictions.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=m['prediction_sha256']:raise ValueError('Prediction hash mismatch')
    with np.load(path,allow_pickle=False) as z:ids,split,rx,scene,pred=[z[k].copy() for k in ('indices','split','rx','scene','predictions')]
    if ids.shape!=(11761,) or len(np.unique(ids))!=len(ids) or pred.shape!=(len(ids),2) or not np.isin(pred,np.arange(6)).all():
        raise ValueError('Complete unique fixed source predictions required')
    prior=root/'runs'/PRIOR
    pm=json.loads((prior/'prediction_complete.json').read_text())
    pp=prior/'source_predictions.npz'
    if hashlib.sha256(pp.read_bytes()).hexdigest()!=pm['sha256'] or pm['sha256']!=m['comparator_sha256']:
        raise ValueError('Fixed source comparator hash mismatch')
    with np.load(pp,allow_pickle=False) as z:pi,ps,pr,pv,old=[z[k].copy() for k in ('indices','split','rx','scene','values')]
    if not all(np.array_equal(a,b) for a,b in ((ids,pi),(split,ps),(rx,pr),(scene,pv))) or old.shape!=(len(ids),15):
        raise ValueError('Source comparator IDs and views must match')
    return ids,split,rx,scene,pred,old

def source_truth(root,ids):
    cache=root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache'
    physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
    roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if np.any((ids<0)|(ids>=len(roles))):raise ValueError('Valid source indices required')
    known=roles[ids]=='L_s';unknown=roles[ids]=='U_s'
    if not np.all(known|unknown) or known.sum()!=1047:raise ValueError('Only permitted source roles required')
    complete=json.loads((root/'runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F/main_split-repl-shuffled_time-3view-s2026092701/source/complete.json').read_text())
    config=complete['config'];contract=json.loads(Path(config['source_contract']).read_text())
    with np.load(Path(config['source_cache'])/'L_s_metadata.npz',allow_pickle=False) as z:keys,labels=z['ids'].copy(),z['y'].copy()
    if (len(keys)!=6300 or labels.shape!=(6300,) or keys.tolist()!=contract['role_ids']['L_s']
            or len(set(map(str,keys)))!=len(keys) or not np.isin(labels,np.arange(6)).all()):
        raise ValueError('Exact legal L_s identity metadata required')
    lookup={str(k):int(y) for k,y in zip(keys,labels)}
    if not all(str(k) in lookup for k in physical[ids[known]]) or any(str(k) in lookup for k in physical[ids[unknown]]):
        raise ValueError('Source role-to-physical-ID binding failed')
    truth=np.full(len(ids),-1,dtype=np.int64)
    truth[known]=[lookup[str(k)] for k in physical[ids[known]]]
    return truth,known

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--commit',required=True);args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Unique source score required')
    ids,split,rx,scene,pred,old=load_fixed(args.root,args.source,args.commit)
    truth,known=source_truth(args.root,ids)
    pseudo=old[:,0].astype(np.int64);trusted=old[:,1]>=.95;dura=old[:,2].astype(np.int64)
    results={}
    for group,part in (('RX4_6',1),('RX8',2)):
        results[group]={}
        for view in ('all_impaired',)+VIEWS:
            chosen=(split==part)&((scene!='clean') if view=='all_impaired' else (scene==view))
            if not chosen.any():raise ValueError('Each source receiver/view cell is required')
            cell={}
            for kind,target,eligible in (('L_s',truth,known),('pseudo',pseudo,trusted)):
                m=chosen&eligible
                cell[kind]=dict(context_vs_DURA=pair_counts(target,dura,pred[:,0],m),
                                denoiser_vs_DURA=pair_counts(target,dura,pred[:,1],m),
                                denoiser_vs_context=pair_counts(target,pred[:,0],pred[:,1],m))
            results[group][view]=cell
    args.out.mkdir(parents=True)
    output=dict(status='SOURCE_SCORE_COMPLETE',commit=args.commit,count=len(ids),L_s_scored=int(known.sum()),
                fixed_predictions_before_truth=True,U_s_truth_reads=0,V_reads=0,test_reads=0,
                caveat='Previously explored source records and F source exposure; pseudo labels are not identity truth.',
                results=results)
    (args.out/'source_score.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('FULLSOURCE_PAIR_SCORE_COMPLETE',len(ids),'L_s',int(known.sum()),flush=True)

if __name__=='__main__':main()
