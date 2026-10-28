"""Source-only rejection development; no target inputs or target feedback."""
import argparse
import json
from pathlib import Path

import numpy as np
from .scores import METHODS, fit_prototypes, knownness_scores
from .scoring import group_metrics

SHRINKAGE=(.05,.2,.5)
NAMES=METHODS+('mahalanobis_s005','mahalanobis_s020','mahalanobis_s050')
FEATURE_KEYS={'train_features','train_y','train_ids','validation_features','validation_logits',
              'validation_y','validation_ids','validation_receivers'}


def _finite_matrix(value,name):
    x=np.asarray(value,dtype=np.float64)
    if x.ndim!=2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('Nonempty finite matrix required: '+name)
    return x


def fit_geometry(train_features,train_labels):
    z=_finite_matrix(train_features,'source train features')
    y=np.asarray(train_labels)
    prototypes=fit_prototypes(z,y,4)
    unit=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
    residual=unit-prototypes[y]
    covariance=residual.T@residual/max(len(z)-4,1)
    scale=max(float(np.trace(covariance)/z.shape[1]),1e-8)
    eye=np.eye(z.shape[1]);whiteners=[]
    for shrinkage in SHRINKAGE:
        regularized=(1.-shrinkage)*covariance+shrinkage*scale*eye
        lower=np.linalg.cholesky(regularized)
        # If C=L L^T, ||(z-mu) L^{-T}||^2 is its Mahalanobis distance.
        whiteners.append(np.linalg.solve(lower,eye))
    return dict(prototypes=prototypes,whiteners=np.asarray(whiteners),shrinkage=np.asarray(SHRINKAGE))


def geometry_scores(features,logits,bundle):
    z=_finite_matrix(features,'source evaluation features')
    base=knownness_scores(z,logits,bundle['prototypes'])
    unit=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
    values=[base]
    for whitener in bundle['whiteners']:
        whitened=unit@whitener.T
        centers=bundle['prototypes']@whitener.T
        distances=(whitened*whitened).sum(axis=1)[:,None]+(centers*centers).sum(axis=1)[None,:]-2.*whitened@centers.T
        values.append(-np.maximum(distances.min(axis=1),0.)[:,None])
    result=np.concatenate(values,axis=1)
    if result.shape!=(len(z),len(NAMES)) or not np.isfinite(result).all():
        raise ValueError('Invalid source geometry scores')
    return result


def calibrate_known(scores,labels):
    values=_finite_matrix(scores,'source V scores');y=np.asarray(labels)
    if values.shape[1]!=len(NAMES) or y.shape!=(len(values),) or y.dtype.kind not in 'iu' or np.any(y<0):
        raise ValueError('Invalid source validation labels/scores')
    known=y<4
    if not known.any():raise ValueError('Source V known calibration is empty')
    return np.quantile(values[known],.01,axis=0,method='lower')


def eligible(overall,heldout):
    for row in (overall,heldout):
        if not row['known_count'] or not row['unknown_count']:return False
        if (row['unknown_macro_rejection'] is None or row['unknown_macro_rejection']<.95
                or row['known_macro_frr'] is None or row['known_macro_frr']>.05
                or row['known_accuracy_drop'] is None or row['known_accuracy_drop']>.02+1e-12):return False
    return True


def _write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def evaluate(features,output,heldout_receiver):
    features=Path(features);out=Path(output)
    if out.exists():raise FileExistsError('Source rule output must not exist')
    complete=json.loads((features.parent/'complete.json').read_text(encoding='utf-8-sig'))
    if (complete.get('status')!='SOURCE_DEVELOPMENT_FROZEN' or complete.get('source_only') is not True
            or complete.get('target_reads')!=0 or complete.get('U_reads')!=0
            or complete.get('checkpoint_initialization')!='scratch' or complete.get('epochs')!=20
            or complete.get('updates')!=540 or complete.get('heldout_source_rx')!=heldout_receiver):
        raise ValueError('Source-only fixed development model is not complete')
    with np.load(features,allow_pickle=False) as stored:
        if set(stored.files)!=FEATURE_KEYS:raise ValueError('Source feature schema differs')
        arrays={key:stored[key].copy() for key in FEATURE_KEYS}
    train=arrays['train_features'];validation=arrays['validation_features'];raw=arrays['validation_logits']
    y=arrays['validation_y'];rx=arrays['validation_receivers']
    if (train.shape!=(complete['train_count'],160) or validation.shape!=(complete['validation_count'],160)
            or raw.shape!=(len(validation),4) or y.shape!=(len(validation),) or y.dtype.kind not in 'iu'
            or set(np.unique(y).tolist())!=set(range(6)) or rx.shape!=y.shape or rx.dtype.kind not in 'iu'):
        raise ValueError('Source class, feature, receiver or count schema differs')
    for key,n in (('train_ids',len(train)),('validation_ids',len(validation))):
        ids=arrays[key]
        if ids.shape!=(n,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=n or np.any(ids==''):
            raise ValueError('Unique physical source IDs required')
    if set(arrays['train_ids'].tolist())&set(arrays['validation_ids'].tolist()):
        raise ValueError('Source L and V physical records overlap')
    bundle=fit_geometry(train,arrays['train_y'])
    values=geometry_scores(validation,raw,bundle)
    thresholds=calibrate_known(values,y)
    closed=raw.argmax(axis=1);held=rx==heldout_receiver
    if not held.any():raise ValueError('No held-out source receiver validation')
    methods={}
    for j,name in enumerate(NAMES):
        accepted=values[:,j]>=thresholds[j]
        overall=group_metrics(y,closed,values[:,j],accepted,4)
        heldout=group_metrics(y[held],closed[held],values[held,j],accepted[held],4)
        per_receiver={str(r):group_metrics(y[rx==r],closed[rx==r],values[rx==r,j],accepted[rx==r],4) for r in np.unique(rx)}
        methods[name]=dict(overall=overall,heldout_receiver=heldout,by_receiver=per_receiver,
                           threshold=float(thresholds[j]),eligible=eligible(overall,heldout))
    allowed=[name for name in NAMES if methods[name]['eligible']]
    result=dict(status='SOURCE_RULES_EVALUATED',evidence_scope='source_proxy_only_not_target',
        target_reads=0,target_results_used=False,source_unknown_used_for_network_training=False,
        prototype_covariance_role='selected_source_L_only',threshold_role='source_V_known_only_99pct_retention',
        heldout_receiver=int(heldout_receiver),methods=methods,eligible_methods=allowed,
        next_step='FIRST_FOLD_PASS_REQUIRES_OTHER_FOLDS' if allowed else 'NO_PROMOTION',
        final_six_class_method_selected=False)
    out.mkdir(parents=True,exist_ok=False)
    with (out/'source_geometry.npz').open('xb') as f:
        np.savez(f,**bundle,thresholds=thresholds,method_names=np.asarray(NAMES))
    _write(out/'results.json',result)
    _write(out/'complete.json',dict(status='SOURCE_RULES_EVALUATED',source_only=True,target_reads=0,
        eligible_methods=allowed,next_step=result['next_step'],evidence_scope=result['evidence_scope']))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--heldout-receiver',type=int,required=True)
    args=parser.parse_args();result=evaluate(args.features,args.output,args.heldout_receiver)
    print(json.dumps(dict(status=result['status'],eligible_methods=result['eligible_methods'],next_step=result['next_step'])))


if __name__=='__main__':main()
