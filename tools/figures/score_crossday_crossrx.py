"""Post-hoc fixed-prediction D0/unseen-RX stratification; no inference or fitting."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from comparison_suite.score import metrics

PARENTS=('20261010-phase1-standard-ce-baselines-manysig-m56-r01',
         '20261012-phase1-standard-ce-seed-extension-manysig-m52-r01',
         '20261014-phase1-sigtime-purece-e200-clean-manysig-m8-r01')
VARIANTS=('resnet18','resnet34','resnet50','vgg16_bn','densenet121','convnext_tiny','transformer6',
          'bilstm3','cvresnet18_1d','cvresnet18_2d','resnet18_2d','resnet50_2d','sig_time')


def read(path):return json.loads(Path(path).read_text())


def fixed_prediction(folder,ids):
    done=read(folder/'complete.json')
    if (done.get('status')!='PREDICTIONS_COMPLETE' or done.get('truth_read') is not False
        or done.get('query_fit') is not False or done.get('views')!=['clean'] or done.get('count')!=len(ids)):
        raise ValueError('Truth remains closed: incomplete/invalid prediction')
    with np.load(folder/'predictions.npz',allow_pickle=False) as f:
        if set(f.files)!={'ids','clean'} or not np.array_equal(f['ids'],ids):raise ValueError('Prediction ID mismatch')
        pred=f['clean'].copy()
    if pred.shape!=ids.shape or pred.dtype.kind not in 'iu' or pred.min()<0 or pred.max()>5:raise ValueError('Class coverage mismatch')
    return pred


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--include-ablation',action='store_true');a=p.parse_args()
    spec=read(a.project/'releases/sigtime_purece_e200_20261014_r01/experiments/standard_ce_baselines/configs/sigtime_20261014/launch_spec.json')
    capsule=Path(spec['capsule']);manifest=read(capsule/'manifest.json')
    if manifest['status']!='VALIDATED_ONCE' or manifest['count']!=168000 or not manifest['source_target_disjoint']:
        raise ValueError('Existing clean capsule mismatch')
    with np.load(capsule/'index.npz',allow_pickle=False) as f:ids=f['ids'].copy()
    planned=[(v,s,PARENTS[2] if v=='sig_time' else PARENTS[0 if s<=2026092704 else 1]) for v in VARIANTS for s in range(2026092701,2026092709)]
    if a.include_ablation:planned += [('sig_time_removed',s,'20261026-phase1-pamf-nosigtime-purece-e200-manysig-m8-r01') for s in range(2026092701,2026092709)]
    # Every original prediction is checked before truth opens; no new prediction is computed.
    predictions=[]
    for v,s,parent in planned:
        folder=a.project/'runs'/parent/f'{v}-s{s}'
        predictions.append((v,s,parent,folder,fixed_prediction(folder/'prediction',ids)))
    truth=read(spec['truth']);y=np.asarray([truth[str(s)]['label'] for s in ids]);rx=np.asarray([str(truth[str(s)]['receiver']) for s in ids]);day=np.asarray([str(truth[str(s)]['day']) for s in ids])
    expected_rx={'1-1','14-7','2-1','20-1','7-14','7-7','8-8'}
    if set(rx)!=expected_rx or set(day)!={'2021_03_01','2021_03_08','2021_03_15','2021_03_23'}:raise ValueError('Historical target metadata mismatch')
    mask=day=='2021_03_01'
    if mask.sum()!=42000 or Counter(rx[mask])!=Counter({r:6000 for r in expected_rx}) or Counter(y[mask])!=Counter({i:7000 for i in range(6)}):raise ValueError('D0/unseenRX coverage mismatch')
    records=[];audit=[]
    for v,s,parent,folder,pred in predictions:
        full=metrics(y,pred,6);old=read(folder/'scores/scores.json')['results'][0]
        if full['confusion']!=old['confusion']:raise ValueError('Original full confusion does not reproduce')
        audit.append(dict(variant=v,model_seed=s,full_query_count=168000,full_accuracy=full['accuracy'],original_confusion_reproduced=True))
        for dimension,strata in [('overall',['ALL']),('receiver',sorted(expected_rx)),('transmitter',manifest['classes'])]:
            for stratum in strata:
                select=mask if dimension=='overall' else mask&(rx==stratum) if dimension=='receiver' else mask&(y==manifest['classes'].index(stratum))
                records.append(dict(variant=v,model_seed=s,parent_run_id=parent,view='clean',subset='unseen_day_unseen_rx',day='2021_03_01',
                    dimension=dimension,stratum=stratum,**metrics(y[select],pred[select],6)))
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(status='VERIFIED',classes=manifest['classes'],subset=dict(source_days=['2021_03_08','2021_03_15','2021_03_23'],target_day='2021_03_01',
        target_rx=sorted(expected_rx),query_count=42000,per_rx=6000,per_tx=7000,day_index=0,
        temporal_note='D0 predates source D1-D3; unseen-day generalization, not future-date extrapolation'),
        actual_models=len(planned),records=records,audit=audit,truth_last=True,no_training_or_new_inference=True)
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='VERIFIED',actual_models=len(planned),metric_rows=len(records),query_count=42000)),flush=True)


if __name__=='__main__':main()
