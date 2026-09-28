"""Frozen D92-SCV capsule prediction, with no truth/scoring interface."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
from cvs_d92_matched import registered_features, read
from cvsrffi.stage2_d92_support_cv import fit_support_cv


def validate_split(split, manifest, ids, old_classes):
    classes=split['registered_classes'];s=split['support_indices'];q=split['query_indices']
    labels=np.asarray(split['support_labels'])
    if (split.get('protocol_schema')!='p2_min_v1' or split.get('phase2_data_status')!='VALIDATED_ONCE'
            or split.get('capsule_id')!=manifest['capsule_id']
            or classes[:len(old_classes)]!=old_classes or len(set(classes))!=len(classes)
            or not s or not q or any(type(i) is not int for i in s+q)
            or len(set(s))!=len(s) or len(set(q))!=len(q) or set(s)&set(q)
            or min(s+q)<0 or max(s+q)>=len(ids)
            or labels.shape!=(len(s),) or labels.dtype.kind not in 'iu'
            or set(labels.tolist())!=set(range(len(classes)))
            or not np.all(np.bincount(labels,minlength=len(classes))==split['k'])):
        raise ValueError('Split identity, physical support/query or registration mismatch')
    forbidden={'query_labels','query_truth','query_roles','query_class_counts'}
    if forbidden & set(split):raise ValueError('Forbidden query information in split')
    return np.asarray(s),np.asarray(q),labels


def predict(*,row_root,capsule,output,expected_capsule_id,expected_checkpoint_sha256,config):
    row_root,capsule,out=map(Path,(row_root,capsule,output))
    manifest=read(capsule/'manifest.json')
    if (manifest.get('protocol_schema')!='p2_min_v1' or manifest.get('phase2_data_status')!='VALIDATED_ONCE'
            or manifest['capsule_id']!=expected_capsule_id):raise ValueError('Capsule mismatch')
    previous=read(row_root/'d92_startup.json')
    ground=read(row_root/'ground/manifest.json')
    if (previous['checkpoint_sha256']!=expected_checkpoint_sha256
            or ground['checkpoint_sha256']!=expected_checkpoint_sha256
            or previous['capsule']!=str(capsule)
            or previous['features']!=str(row_root/'received_features/received_features.npz')
            or previous['query_fit_access'] is not False or previous['truth_read'] is not False):
        raise ValueError('Frozen encoder/feature provenance mismatch')
    marker=read(row_root/'received_features/features_complete.json')
    if (marker['status']!='FROZEN_FEATURES_COMPLETE' or marker['capsule_id']!=expected_capsule_id
            or marker['query_used_for_fitting'] is not False):raise ValueError('Incomplete frozen features')
    from cvsrffi.phase1_center_lowrank_prototype_bundle import NPZ_NAME
    with np.load(row_root/'ground'/NPZ_NAME,allow_pickle=False) as d:old_classes=d['class_registry'].astype(str).tolist()
    with np.load(row_root/'received_features/received_features.npz',allow_pickle=False) as d:
        identity,logits,ids=d['identity160'],d['logits'],d['ids'].astype(str)
    with np.load(capsule/'received.npz',allow_pickle=False) as d:iq,capsule_ids=d['iq'],d['ids'].astype(str)
    if (not np.array_equal(ids,capsule_ids) or len(set(ids))!=len(ids)
            or identity.shape!=(len(ids),160) or logits.shape!=(len(ids),len(old_classes))
            or not np.isfinite(logits).all()):raise ValueError('Feature/physical-ID alignment mismatch')
    paths=sorted((capsule/'splits').glob('*.json'))
    if len(paths)!=manifest['split_count']:raise ValueError('Incomplete split matrix')
    # Validate all split boundaries before creating output or making predictions.
    splits=[read(p) for p in paths]
    for split in splits:validate_split(split,manifest,ids,old_classes)
    features=registered_features(iq,identity)
    out.mkdir(parents=True,exist_ok=False)
    (out/'startup.json').write_text(json.dumps(dict(argv=sys.argv,config=config,checkpoint_sha256=expected_checkpoint_sha256,
        capsule_id=expected_capsule_id,query_fit_access=False,truth_read=False,model_seed=previous['seed']),indent=2)+'\n',encoding='utf-8')
    with (out/'predictions.jsonl').open('x',encoding='utf-8') as pred, (out/'fit_trace.jsonl').open('x',encoding='utf-8') as trace, (out/'compact.jsonl').open('x',encoding='utf-8') as compact:
        for index,split in enumerate(splits):
            s,q,y=validate_split(split,manifest,ids,old_classes)
            started=time.perf_counter()
            state=fit_support_cv(support_features=features[s],support_labels=y,support_logits=logits[s],
                classes=split['registered_classes'],old_count=len(old_classes),**config)
            scores=state.score(features[q],logits[q])
            if not np.isfinite(scores).all():raise ValueError('Nonfinite prediction')
            record={k:split[k] for k in ('split_id','capsule_id','receiver','scenario','k','support_seed')}
            record.update(mode='d92_scv_registration',classes=split['registered_classes'],query_ids=ids[q].tolist(),
                predicted_indices=scores.argmax(1).tolist(),scores=scores.tolist())
            pred.write(json.dumps(record)+'\n');pred.flush()
            trace.write(json.dumps(dict(split_id=split['split_id'],**state.audit))+'\n');trace.flush()
            small=dict(split_id=split['split_id'],completed=index+1,total=len(splits),k=split['k'],classes=len(state.classes),
                selected=state.audit['selected'],fit_seconds=state.audit['fit_seconds'],
                total_seconds=time.perf_counter()-started,state_bytes=state.audit['persistent_state_bytes'],
                query_rows_used_for_fit=0,loss=None,lr=None,gradient=None,
                unavailable_reason='closed_form_no_gradient_training; support validation scores in fit_trace.jsonl')
            compact.write(json.dumps(small)+'\n');compact.flush();print(json.dumps(small),flush=True)
    (out/'predictions_complete.json').write_text(json.dumps(dict(status='PREDICTIONS_COMPLETE',split_count=len(splits),
        predictions=len(splits),capsule_id=expected_capsule_id,truth_read=False),indent=2)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('row-root','capsule','output','config','expected-capsule-id','expected-checkpoint-sha256'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    predict(row_root=a.row_root,capsule=a.capsule,output=a.output,config=read(a.config),
        expected_capsule_id=a.expected_capsule_id,expected_checkpoint_sha256=a.expected_checkpoint_sha256)


if __name__=='__main__':main()
