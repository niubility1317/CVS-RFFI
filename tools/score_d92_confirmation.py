"""Validate every frozen baseline/candidate prediction before independent truth access."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from run_d92_confirmation import candidate_definition, baseline_root, reuse_frozen_rows


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_prediction_capsule(confirmation):
    """Read physical IDs and split metadata, never IQ values or truth."""
    capsule=Path(confirmation['capsule']);manifest=read(capsule/'manifest.json')
    expected_splits=confirmation.get('expected_split_count')
    if expected_splits is not None and (type(expected_splits) is not int or expected_splits<=0 or manifest['split_count']!=expected_splits):
        raise ValueError('Capsule split count differs from preregistered expected_split_count')
    with np.load(capsule/'received.npz',allow_pickle=False) as d:ids=d['ids'].astype(str)
    if len(ids)!=len(set(ids)):raise ValueError('Repeated physical IDs')
    splits={p.stem:read(p) for p in (capsule/'splits').glob('*.json')}
    if len(splits)!=manifest['split_count']:raise ValueError('Incomplete capsule split set')
    return manifest,ids,splits


def validate_method_records(folder,method,manifest,ids,splits,candidate):
    """Validate complete immutable predictions against their capsule, without scoring."""
    folder=Path(folder)
    marker=read(folder/'predictions_complete.json')
    if marker.get('status')!='PREDICTIONS_COMPLETE' or marker.get('split_count')!=len(splits) or marker.get('capsule_id')!=manifest['capsule_id']:
        raise ValueError('Incomplete method predictions')
    entries=[json.loads(line) for line in (folder/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    main_seen=set();dg_seen=set()
    for record in entries:
        sid=record['split_id']
        if sid not in splits:raise ValueError('Unknown split')
        split=splits[sid]
        for key in ['capsule_id','receiver','scenario','k','support_seed']:
            if record[key]!=split[key]:raise ValueError('Prediction split binding mismatch')
        if record['classes']!=split['registered_classes'] or record['query_ids']!=ids[split['query_indices']].tolist():
            raise ValueError('Prediction class/query-ID binding mismatch')
        score=np.asarray(record['scores'],dtype=float);pred=np.asarray(record['predicted_indices'])
        if score.shape!=(len(record['query_ids']),len(record['classes'])) or not np.isfinite(score).all() or pred.dtype.kind not in 'iu':
            raise ValueError('Invalid prediction scores/argmax')
        if method in ('D92-SGJoint-v1','D92-MVKME-v1','D92-BNNA-v1','D92-OSC-v1','D92-MVRidge-v1','D92-BranchRidge-v1'):
            order=np.asarray(sorted(range(len(record['classes'])),key=lambda i:record['classes'][i]))
            expected_pred=order[score[:,order].argmax(1)]
        else:
            expected_pred=score.argmax(1)
        if not np.array_equal(expected_pred,pred):
            raise ValueError('Invalid prediction scores/argmax')
        if record['mode']=='frozen_dg' and method=='D92':
            key=(record['receiver'],record['scenario'])
            if len(record['classes'])!=6 or key in dg_seen:raise ValueError('Invalid DG record')
            dg_seen.add(key)
        else:
            expected='d92_registration' if method=='D92' else candidate['candidate_mode']
            if record['mode']!=expected or sid in main_seen:raise ValueError('Invalid/duplicate method record')
            main_seen.add(sid)
    if main_seen!=set(splits) or len(entries)!=marker['predictions']:raise ValueError('Missing or duplicated prediction coverage')
    expected_dg={(s['receiver'],s['scenario']) for s in splits.values()} if method=='D92' else set()
    if dg_seen!=expected_dg:raise ValueError('Incomplete DG coverage')
    return entries


def load_frozen_records(spec):
    """This validation phase has no truth-path access."""
    candidate=candidate_definition(spec['confirmation'])
    reused=reuse_frozen_rows(spec)
    root=Path(spec['execution']['remote_run_root']);state=read(root/'state.json')
    if set(state)!=set(r['row_id'] for r in spec['rows']) or any(v['status']!='PREDICTIONS_COMPLETE' for v in state.values()):
        raise ValueError('All preregistered model rows must finish before scoring')
    manifest,ids,splits=load_prediction_capsule(spec['confirmation'])
    records=[]
    for row in spec['rows']:
        if reused:
            from run_d92_confirmation import validate_reused_row
            validate_reused_row(row,spec['confirmation'],(manifest,ids,splits),validate_predictions=False)
        for method,folder in [('D92',baseline_root(row)),(candidate['candidate_method'],Path(row['output_root'])/candidate['candidate_folder'])]:
            entries=validate_method_records(folder,method,manifest,ids,splits,candidate)
            records.extend((row['row_id'],row['seeds']['model'],method,record) for record in entries)
    return records


def score(spec, output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    records=load_frozen_records(spec)
    # Independent scorer is the first phase that opens query truth.
    truth=read(spec['confirmation']['truth'])
    results=[]
    for row_id,seed,method,record in records:
        classes=record['classes'];lookup={c:i for i,c in enumerate(classes)}
        cm=np.zeros((len(classes),len(classes)),dtype=np.int64)
        for sid,pred in zip(record['query_ids'],record['predicted_indices']):
            target=truth[sid]
            if (target['pool_role']!='query' or target['receiver']!=record['receiver'] or target['scene']!=record['scenario']
                    or target['transmitter'] not in lookup or bool(target['old'])!=(lookup[target['transmitter']]<6)):
                raise ValueError('Truth binding mismatch')
            cm[lookup[target['transmitter']],pred]+=1
        den=cm.sum(1)
        if np.any(den==0):raise ValueError('Missing registered-class query coverage')
        acc=cm.diagonal()/den;old=float(cm.diagonal()[:6].sum()/den[:6].sum())
        new=float(cm.diagonal()[6:].sum()/den[6:].sum()) if len(classes)>6 else None
        fden=cm.sum(0)+den;f1=np.divide(2*cm.diagonal(),fden,out=np.zeros(len(classes)),where=fden!=0)
        results.append(dict(row_id=row_id,model_seed=seed,method='frozen_dg' if record['mode']=='frozen_dg' else method,
            **{k:record[k] for k in ['split_id','receiver','scenario','k','support_seed']},class_count=len(classes),
            new_count=len(classes)-6,accuracy=float(cm.trace()/cm.sum()),old_accuracy=old,new_accuracy=new,
            harmonic_mean=(2*old*new/(old+new) if old+new else 0.0) if new is not None else None,
            macro_f1=float(f1.mean()),old_macro_f1=float(f1[:6].mean()),new_macro_f1=float(f1[6:].mean()) if new is not None else None,
            old_floor=float(acc[:6].min()),new_floor=float(acc[6:].min()) if new is not None else None,
            class_accuracy=acc.tolist(),classes=classes,confusion=cm.tolist(),query_count=int(cm.sum())))
    before={(r['model_seed'],r['method'],r['receiver'],r['scenario'],r['k'],r['support_seed']):r['old_accuracy'] for r in results if r['new_count']==0 and r['method']!='frozen_dg'}
    for r in results:
        key=tuple(r[k] for k in ['model_seed','method','receiver','scenario','k','support_seed'])
        r['forgetting']=before[key]-r['old_accuracy'] if r['new_count'] and r['method']!='frozen_dg' else None
    with output.open('x',encoding='utf-8') as f:
        json.dump(dict(status='SCORED',results=results,selection_feedback_forbidden=True,
            claim_scope=spec.get('permissions',{}).get('claim_scope',
                'Repeated benchmark on existing frozen data; not fresh independent confirmation' if reuse_frozen_rows(spec)
                else 'Fresh relative to current matched source/target RX and this candidate development; all-time historical non-exposure not certified')),f)
    print(json.dumps(dict(status='SCORED',records=len(results))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();score(read(a.spec),a.output)


if __name__=='__main__':main()
