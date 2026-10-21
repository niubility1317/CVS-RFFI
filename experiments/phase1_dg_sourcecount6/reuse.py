"""Reuse only identical five-source frozen runs; restrict existing predictions."""
from pathlib import Path
import numpy as np
from experiments.phase1_external_dg_aug import source, evaluation
from experiments.phase1_external_dg_aug.plan import make_plan as parent_plan
from experiments.phase1_external_dg_aug.analysis import check_source
from experiments.phase1_external_dg.data import read,write
from .runtime import bind
from .plan import VIEWS


def parent_source(rt,row,handle):
    if row['source_count']!=5 or row['label_budget']!='natural' or not row['reuse_parent_row_id']:
        raise ValueError('Unregistered parent reuse')
    root=Path(rt['parent_run_root'])/row['reuse_parent_row_id']
    old=parent_plan();source._PLAN=old;source._ROWS={r['row_id']:r for r in old['rows']}
    try:
        original=source._ROWS[row['reuse_parent_row_id']]
        ignore={'row_id','group','protocol_id','supervised_control_row_id'}
        if any(row.get(k)!=v for k,v in original.items() if k not in ignore):
            raise ValueError('Parent scientific recipe differs')
        cost=check_source(root/'source',original)
        frozen=read(root/'source/freeze.json')
        if frozen['source_contract']!=read(handle['source_contract']):
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    finally:bind()
    return root,cost


def predict(c):
    # This configuration contains no truth. Parent source legality is checked
    # by the worker before this separate prediction process starts.
    evaluation.reject_truth(c)
    row=evaluation.row_spec(c)
    parent=Path(c['parent_prediction']);done=read(parent/'complete.json')
    if (done.get('status')!='ALL_PREDICTIONS_COMPLETE' or done.get('row_id')!=row['reuse_parent_row_id'] or
        done.get('truth_read') is not False or done.get('query_fit') is not False or done.get('views')!=list(VIEWS)):
        raise ValueError('Parent seven-view predictions are not closed')
    old_done=read(parent/'main/complete.json')
    if old_done['status']!='PREDICTIONS_COMPLETE' or old_done['views']!=list(VIEWS):
        raise ValueError('Incomplete parent view coverage')
    meta,ids,_=evaluation.handle(c['tests']['main']['views_root'],c['classes'])
    if meta['count']!=144000:raise ValueError('Fixed target size differs')
    with np.load(parent/'main/predictions.npz',allow_pickle=False) as old:
        if set(old.files)!={'ids',*VIEWS} or len(set(old['ids'].tolist()))!=len(old['ids']):
            raise ValueError('Malformed parent predictions')
        lookup={sid:i for i,sid in enumerate(old['ids'].tolist())}
        indices=np.asarray([lookup[sid] for sid in ids.tolist()],dtype=np.int64)
        values={v:old[v][indices].copy() for v in VIEWS}
    for a in values.values():
        if a.shape!=ids.shape or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=len(c['classes']):
            raise ValueError('Invalid reused classes')
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);(out/'main').mkdir()
    np.savez(out/'main/predictions.npz',ids=ids,**values)
    seconds={v:0. for v in VIEWS}
    write(out/'main/complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(VIEWS),
        num_classes=len(c['classes']),truth_read=False,query_fit=False,seconds=seconds))
    write(out/'complete.json',dict(status='ALL_PREDICTIONS_COMPLETE',row_id=row['row_id'],
        tests=dict(main=dict(count=len(ids),views=list(VIEWS),seconds=seconds)),views=list(VIEWS),
        truth_read=False,query_fit=False,reused_parent_row=row['reuse_parent_row_id']))
