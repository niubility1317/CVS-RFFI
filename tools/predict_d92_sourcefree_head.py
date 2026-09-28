"""Support-only SFHead training and immutable query inference. No source/truth input."""
import argparse
import csv
import json
from collections.abc import Mapping
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
from predict_d92_support_cv import validate_split
from cvsrffi.stage2_d92_sourcefree_head import fit_sourcefree_head


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def plain(value):
    if isinstance(value, Mapping):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value


def predict(*,row_root,capsule,output,expected_capsule_id,expected_checkpoint_sha256,config):
    row_root,capsule,out=map(Path,(row_root,capsule,output))
    manifest=read(capsule/'manifest.json')
    if (manifest.get('protocol_schema')!='p2_min_v1' or manifest.get('phase2_data_status')!='VALIDATED_ONCE'
            or manifest.get('capsule_id')!=expected_capsule_id):raise ValueError('Capsule mismatch')
    previous=read(row_root/'d92_startup.json')
    feature_root=row_root/'received_features'
    marker=read(feature_root/'features_complete.json')
    provenance=read(feature_root/'checkpoint_provenance.json')
    if (previous.get('checkpoint_sha256')!=expected_checkpoint_sha256
            or previous.get('capsule')!=str(capsule)
            or previous.get('features')!=str(feature_root/'received_features.npz')
            or previous.get('query_fit_access') is not False or previous.get('truth_read') is not False
            or marker.get('status')!='FROZEN_FEATURES_COMPLETE' or marker.get('capsule_id')!=expected_capsule_id
            or marker.get('query_used_for_fitting') is not False
            or provenance.get('target_access_before_freeze') is not False
            or provenance.get('checkpoint_inheritance')!=[]):raise ValueError('Frozen feature provenance mismatch')
    old_classes=provenance['classes']
    with np.load(feature_root/'received_features.npz',allow_pickle=False) as d:
        if set(d.files)!={'identity160','logits','ids'}:raise ValueError('Unexpected feature members')
        identity,logits,ids=d['identity160'],d['logits'],d['ids'].astype(str)
    with np.load(capsule/'received.npz',allow_pickle=False) as d:capsule_ids=d['ids'].astype(str)
    if (not np.array_equal(ids,capsule_ids) or len(set(ids))!=len(ids)
            or identity.shape!=(len(ids),160) or logits.shape!=(len(ids),len(old_classes))
            or not np.isfinite(identity).all() or not np.isfinite(logits).all()):raise ValueError('Feature alignment mismatch')
    paths=sorted((capsule/'splits').glob('*.json'))
    if len(paths)!=manifest['split_count']:raise ValueError('Incomplete split matrix')
    splits=[read(p) for p in paths]
    for split in splits:validate_split(split,manifest,ids,old_classes)
    out.mkdir(parents=True,exist_ok=False)
    (out/'startup.json').write_text(json.dumps(dict(argv=sys.argv,config=config,
        checkpoint_sha256=expected_checkpoint_sha256,capsule_id=expected_capsule_id,
        query_fit_access=False,truth_read=False,source_data_access=False,
        new_ground_statistics_bytes=0,model_seed=previous['seed']),indent=2)+'\n',encoding='utf-8')
    fields=['split_id','completed','total','k','classes','fit_seconds','optimizer_steps',
            'optimizer_status','persistent_state_bytes','total_seconds','query_rows_used_for_fit','source_rows_used_for_fit']
    with (out/'predictions.jsonl').open('x',encoding='utf-8') as pred, \
            (out/'fit_trace.jsonl').open('x',encoding='utf-8') as trace, \
            (out/'compact.jsonl').open('x',encoding='utf-8') as compact, \
            (out/'compact.csv').open('x',encoding='utf-8',newline='') as csvfile:
        writer=csv.DictWriter(csvfile,fieldnames=fields);writer.writeheader()
        for index,split in enumerate(splits):
            s,q,y=validate_split(split,manifest,ids,old_classes)
            started=time.perf_counter()
            state=fit_sourcefree_head(support_features=identity[s],support_labels=y,support_logits=logits[s],
                classes=split['registered_classes'],old_count=len(old_classes),**config)
            audit=plain(state.audit)
            scores=state.score(identity[q])
            if scores.shape!=(len(q),len(state.classes)) or not np.isfinite(scores).all():raise ValueError('Invalid query scores')
            record={k:split[k] for k in ('split_id','capsule_id','receiver','scenario','k','support_seed')}
            record.update(mode='d92_sfhead_registration',classes=split['registered_classes'],query_ids=ids[q].tolist(),
                predicted_indices=scores.argmax(1).tolist(),scores=scores.tolist())
            pred.write(json.dumps(record)+'\n');pred.flush()
            trace.write(json.dumps(dict(split_id=split['split_id'],**audit),allow_nan=False)+'\n');trace.flush()
            small=dict(split_id=split['split_id'],completed=index+1,total=len(splits),k=split['k'],classes=len(state.classes),
                fit_seconds=audit['fit_seconds'],optimizer_steps=audit['optimizer_steps'],
                optimizer_status=audit['optimizer_status'],persistent_state_bytes=audit['persistent_state_bytes'],
                total_seconds=time.perf_counter()-started,query_rows_used_for_fit=0,source_rows_used_for_fit=0)
            compact.write(json.dumps(small,allow_nan=False)+'\n');compact.flush()
            writer.writerow(small);csvfile.flush()
            print(json.dumps(dict(split_id=split['split_id'],**audit),allow_nan=False),flush=True)
    (out/'predictions_complete.json').write_text(json.dumps(dict(status='PREDICTIONS_COMPLETE',split_count=len(splits),
        predictions=len(splits),capsule_id=expected_capsule_id,truth_read=False,source_data_access=False),indent=2)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('row-root','capsule','output','config','expected-capsule-id','expected-checkpoint-sha256'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    predict(row_root=a.row_root,capsule=a.capsule,output=a.output,config=read(a.config),
        expected_capsule_id=a.expected_capsule_id,expected_checkpoint_sha256=a.expected_checkpoint_sha256)

if __name__=='__main__':main()
