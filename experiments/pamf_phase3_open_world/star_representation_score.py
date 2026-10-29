"""Independent source truth-last scores and descriptive cross-RX retrieval."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import star_representation_eval as evaluation
from .metrics import open_set_metrics
from .star_nuisance_score import _gate

SCHEMA='pamf_phase3_star_representation_scores_v1'
STATUS='STAR_REPRESENTATION_SCORES_COMPLETE'
UNKNOWN_TX=('19-15','19-16','7-1')  # Required only on the independent truth side.


def _prediction(root):
    root=Path(root);marker=evaluation._read(root/'complete.json')
    if (marker.get('status')!=evaluation.STATUS or marker.get('schema')!=evaluation.SCHEMA or marker.get('run_id')!=evaluation.RUN_ID
            or marker.get('arm') not in evaluation.ARMS or marker.get('path_names')!=list(evaluation.PATHS)
            or marker.get('method_names')!=list(evaluation.METHODS) or marker.get('primary_method')!='energy' or marker.get('primary_path')!='P1'
            or marker.get('source_only') is not True or any(marker.get(k)!=0 for k in ('target_reads','source_novel_truth_reads','source_OE_records_consumed','eval_neural_updates'))
            or any(marker.get(k) is not True for k in ('model_parameters_buffers_unchanged','L_aggregates_unchanged_during_V','all_thresholds_frozen_before_novel'))
            or marker.get('baseline')!='cache original F0 same-input same-path complete head'):
        raise ValueError('All source-only two-path seven-score predictions must complete before truth access')
    calibration={};thresholds={}
    for path in evaluation.PATHS:
        row=root/'by_path'/path;cal=evaluation._read(row/'calibration/complete.json');threshold=np.load(row/'thresholds.npy',allow_pickle=False)
        if (cal.get('status')!=evaluation.CALIBRATION_STATUS or cal.get('schema')!=evaluation.SCHEMA
                or any(cal.get(k)!=marker[k] for k in ('run_id','code_commit','arm')) or cal.get('path')!=path
                or cal.get('method_names')!=list(evaluation.METHODS) or cal.get('group_count')!=42
                or cal.get('source_only') is not True or cal.get('target_reads')!=0 or cal.get('source_V_metric_fit') is not False
                or cal.get('unknown_inputs_used') is not False or set(cal.get('methods',{}))!=set(evaluation.METHODS)
                or threshold.shape!=(7,) or threshold.dtype!=np.float64 or not np.isfinite(threshold).all()
                or any(cal['methods'][name].get('threshold')!=float(threshold[j]) for j,name in enumerate(evaluation.METHODS))):
            raise ValueError('Matching known-only original-F0 risk thresholds required for both frozen paths')
        calibration[path]=cal;thresholds[path]=threshold
    rows=[]
    for role,fields,key in (('known',evaluation.KNOWN_FIELDS,'known_count'),('novel',evaluation.NOVEL_FIELDS,'novel_count')):
        n=marker.get(key)
        if type(n) is not int or n<=0:raise ValueError('Positive completed prediction counts required')
        with np.load(root/(role+'_predictions.npz'),allow_pickle=False) as saved:
            if set(saved.files)!=fields:raise ValueError('Only exact public prediction fields allowed; novel TX/scene/labels forbidden')
            data={name:saved[name].copy() for name in saved.files}
        if (not evaluation._ids(data['ids'],n) or data['scores'].shape!=(n,2,7) or data['scores'].dtype!=np.float64
                or not np.isfinite(data['scores']).all() or data['closed'].shape!=(n,2) or data['closed'].dtype!=np.int64
                or np.any((data['closed']<0)|(data['closed']>=6))):raise ValueError('Finite seven-score arrays and complete six-class predictions required')
        for k,roster in (('rx_i',evaluation.RECEIVERS),('day_i',(1,2,3))):
            if data[k].shape!=(n,) or data[k].dtype!=np.int64 or not np.isin(data[k],roster).all():
                raise ValueError('Original source RX/day prediction metadata required')
        if role=='known':
            if (data['y'].shape!=(n,) or data['y'].dtype!=np.int64 or np.any((data['y']<0)|(data['y']>=6))
                    or data['scenario'].shape!=(n,) or data['scenario'].dtype.kind!='U' or not np.isin(data['scenario'],evaluation.SCENES).all()
                    or data['baseline_closed'].shape!=(n,2) or data['baseline_closed'].dtype!=np.int64 or np.any((data['baseline_closed']<0)|(data['baseline_closed']>=6))):
                raise ValueError('Legal source V known labels/scenes and ORIGINAL F0 baselines required')
        else:
            fused=data['fused_unit']
            if (fused.shape!=(n,2,160) or fused.dtype!=np.float32 or not np.isfinite(fused).all()
                    or not np.allclose(np.linalg.norm(fused.astype(np.float64),axis=2),1.,rtol=0,atol=2e-6)):
                raise ValueError('Finite unit fused embeddings required only for source retrieval diagnostic')
        rows.append(data)
    if set(rows[0]['ids'].tolist())&set(rows[1]['ids'].tolist()):raise ValueError('Source known V and novel physical IDs must be separate')
    return marker,calibration,thresholds,rows[0],rows[1]


def _truth(path,novel):
    with np.load(path,allow_pickle=False) as saved:
        if set(saved.files)!={'ids','tx','rx_i','day_i','scenario'}:raise ValueError('Private source novel truth fields differ')
        data={k:saved[k].copy() for k in saved.files}
    n=len(novel['ids'])
    if not evaluation._ids(data['ids'],n) or set(data['ids'].tolist())!=set(novel['ids'].tolist()) or any(a.shape!=(n,) for a in data.values()):
        raise ValueError('Private truth must uniquely cover exactly the completed opaque novel IDs')
    lookup={sid:i for i,sid in enumerate(data['ids'].tolist())};order=np.array([lookup[sid] for sid in novel['ids'].tolist()],np.int64)
    data={k:value[order] for k,value in data.items()}
    if (data['tx'].dtype.kind!='U' or not np.isin(data['tx'],UNKNOWN_TX).all() or data['scenario'].dtype.kind!='U'
            or not np.isin(data['scenario'],evaluation.SCENES).all() or any(data[k].dtype!=np.int64 or not np.array_equal(data[k],novel[k]) for k in ('rx_i','day_i'))):
        raise ValueError('Source-heldout roster, actual scene and RX/day alignment differ')
    return data


def cross_rx_retrieval(features,ids,rx,tx,required_classes=None):
    """1-NN from other RX only; exact ties choose lexicographically first ID.

    Private TX is used after neighbor selection solely to count hits. No
    threshold, prototype, identity mapping or fitted state is produced.
    """
    x=np.asarray(features,np.float64);ids,rx,tx=map(np.asarray,(ids,rx,tx));n=len(ids)
    if (x.ndim!=2 or x.shape[0]!=n or not x.shape[1] or not np.isfinite(x).all() or not evaluation._ids(ids,n)
            or rx.shape!=(n,) or rx.dtype.kind not in 'iu' or tx.shape!=(n,) or tx.dtype.kind not in 'USiu'
            or not np.allclose(np.linalg.norm(x,axis=1),1.,rtol=0,atol=2e-6)):
        raise ValueError('Aligned finite unit features and opaque IDs required for cross-RX retrieval')
    order=np.argsort(ids,kind='stable');reference=x[order];reference_rx=rx[order]
    neighbor=np.full(n,-1,np.int64)
    for start in range(0,n,128):
        end=min(start+128,n);similarity=x[start:end]@reference.T
        valid=rx[start:end,None]!=reference_rx[None];similarity[~valid]=-np.inf
        selected=similarity.argmax(axis=1);has=valid.any(axis=1)
        neighbor[start:end][has]=order[selected[has]]
    valid=neighbor>=0;hit=np.zeros(n,bool);hit[valid]=tx[valid]==tx[neighbor[valid]]
    roster=tuple(sorted(set(tx.tolist()))) if required_classes is None else tuple(required_classes)
    if not roster or len(set(roster))!=len(roster) or not set(tx.tolist()).issubset(set(roster)):
        raise ValueError('Retrieval truth labels must belong to the required diagnostic roster')
    per_tx={str(label):dict(count=int((tx==label).sum()),eligible_count=int(((tx==label)&valid).sum()),
        hits=int(((tx==label)&hit).sum()),hit_rate=float(hit[tx==label].mean()) if np.any(tx==label) and np.all(valid[tx==label]) else None) for label in roster}
    values=[row['hit_rate'] for row in per_tx.values()]
    return dict(count=n,eligible_count=int(valid.sum()),hits=int(hit.sum()),macro_hit_rate=float(np.mean(values)) if all(v is not None for v in values) else None,
        pooled_hit_rate=float(hit.mean()) if valid.all() else None,by_TX=per_tx,
        by_RX={str(r):dict(count=int((rx==r).sum()),hits=int(((rx==r)&hit).sum()),hit_rate=float(hit[rx==r].mean()) if np.all(valid[rx==r]) else None) for r in sorted(set(rx.tolist()))},
        by_record=[dict(id=str(ids[i]),neighbor_id=str(ids[neighbor[i]]) if valid[i] else None,hit=bool(hit[i]) if valid[i] else None) for i in range(n)],
        exclude_same_RX=True,tie_break='lexicographically smallest opaque ID among exactly equal unit-dot maxima',
        required_TX=list(roster),coverage_complete=all(np.any(tx==label) for label in roster),
        fitted_parameters=0,diagnostic_only=True,not_future_registration_accuracy=True)


def score(prediction_root,novel_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('A new immutable independent source score root is required')
    marker,calibration,thresholds,known,novel=_prediction(prediction_root)
    # Sole truth access, after both paths and all seven score arrays are complete.
    truth=_truth(novel_truth,novel)
    known_masks=dict(evaluation.serialization._masks(known['rx_i'],known['scenario']))
    unknown_masks=dict(evaluation.serialization._masks(novel['rx_i'],truth['scenario']))
    paths={};retrieval={}
    for pi,path in enumerate(evaluation.PATHS):
        methods={}
        for j,method in enumerate(evaluation.METHODS):
            threshold=thresholds[path][j]
            def calculate(kmask,umask,roster=UNKNOWN_TX,known_roster=tuple(range(6))):
                return open_set_metrics(known['y'][kmask],known['closed'][kmask,pi],known['scores'][kmask,pi,j],truth['tx'][umask],
                    novel['scores'][umask,pi,j],threshold,known_roster,roster,baseline_prediction=known['baseline_closed'][kmask,pi])
            groups={name:calculate(mask,unknown_masks[name]) for name,mask in known_masks.items()}
            methods[method]=dict(overall=groups['overall'],groups=groups,threshold=float(threshold),
                by_unknown_TX={tx:calculate(known_masks['overall'],truth['tx']==tx,(tx,)) for tx in UNKNOWN_TX},
                by_known_TX={str(label):calculate(known['y']==label,unknown_masks['overall'],UNKNOWN_TX,(label,)) for label in range(6)},
                goals=_gate(groups,calibration[path]['methods'][method].get('constraints_passed') is True),
                diagnostic_only=path!='P1' or method!='energy' or marker['arm']!='oe_vic')
        paths[path]=dict(methods=methods,energy_source_gate_passed=methods['energy']['goals']['passed'])
        retrieval[path]=cross_rx_retrieval(novel['fused_unit'][:,pi],novel['ids'],novel['rx_i'],truth['tx'],UNKNOWN_TX)
    arm_passed=paths['P1']['energy_source_gate_passed'];eligible=arm_passed and marker['arm']==evaluation.PRIMARY_ARM
    result=dict(status=STATUS,schema=SCHEMA,run_id=evaluation.RUN_ID,code_commit=marker['code_commit'],arm=marker['arm'],
        source_only=True,target_reads=0,truth_last=True,private_source_novel_truth_read=True,paths=paths,
        novel_cross_RX_retrieval=retrieval,primary_arm='oe_vic',primary_path='P1',primary_method='energy',
        arm_source_gate_passed=arm_passed,eligible_primary=eligible,
        next_step='SOURCE_ELIGIBLE_FOR_FROZEN_TARGET_VALIDATION' if eligible else 'NO_PROMOTION',
        known_count=len(known['ids']),novel_count=len(novel['ids']),known_protection_groups=42,
        required_unknown_groups='overall + 5 source RX + 6 scenes; RX-scene unknown diagnostic only',
        source_unknown_roster_repeated_validation=True,formal_target_accessed=False,final_phase3_complete=False,A_N=None,H_OW=None,
        baseline='cache original F0 same-input same-path complete head',
        primary_rule='only oe_vic/P1/energy may advance; anchor/oe or another method cannot replace a failed primary')
    output.mkdir(parents=True,exist_ok=False);evaluation._write(output/'results.json',result)
    evaluation._write(output/'complete.json',{k:v for k,v in result.items() if k not in ('paths','novel_cross_RX_retrieval')})
    print(STATUS,marker['arm'],result['next_step'],flush=True);return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_REPRESENTATION_SCORE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),
        argv=list(sys.argv) if argv is None else list(argv))),flush=True)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prediction-root',required=True)
    parser.add_argument('--novel-truth',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv);score(args.prediction_root,args.novel_truth,args.output);return 0


if __name__=='__main__':raise SystemExit(main())
