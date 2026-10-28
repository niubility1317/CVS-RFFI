"""Source calibration, then source-free unlabeled state fitting and frozen inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import stream_geometry as geometry


STATES=('F','D0R0','D1R0','D0R1','D1R1')
SOURCE_FIELDS={'train_features','train_y','calibration_features'}
FEATURE_FIELDS={'ids','features','logits'}
BUNDLE_FIELDS={'schema','feature_dim','class_count','source_count','source_counts','calibration_count',
    'prototypes','covariance','chol','whitener','anchors','lambda_det','lambda_cluster'}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(v) for v in value]
    return value


def _write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(_plain(value),stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def _save(path,values):
    with Path(path).open('xb') as stream:np.savez(stream,**values)


def _npz(path,keys):
    with np.load(path,allow_pickle=False) as stored:
        if set(stored.files)!=set(keys):raise ValueError('NPZ whitelist differs: '+str(path))
        return {key:stored[key].copy() for key in keys}


def _marker(path,status):
    try:value=_read(path)
    except (OSError,ValueError) as exc:raise ValueError('Completed source-only input required') from exc
    if (not isinstance(value,dict) or value.get('status')!=status or value.get('source_only') is not True
            or type(value.get('target_reads')) is not int or value['target_reads']!=0):
        raise ValueError('Source-only input completion differs')
    return value


def _matrix(value,name,count,width):
    if (value.dtype!=np.dtype('float32') or value.shape!=(count,width) or not np.isfinite(value).all()):
        raise ValueError('Finite raw FP32 source features/logits required: '+name)


def calibrate(source_inputs,output):
    """Use legal source L and seen-RX known V, never any stream or future input."""
    source_inputs,output=Path(source_inputs),Path(output)
    if output.exists():raise FileExistsError('Source calibration output cannot be overwritten')
    done=_marker(source_inputs.parent/'complete.json','SOURCE_STREAM_INPUTS_EXPORTED')
    if (done.get('train_count')!=3360 or done.get('calibration_count')!=14400
            or done.get('head_changed') is not False or done.get('optimizer_updates')!=0
            or done.get('parent_epochs')!=20 or done.get('parent_updates')!=540):
        raise ValueError('Own frozen source calibration roles differ')
    data=_npz(source_inputs,SOURCE_FIELDS)
    _matrix(data['train_features'],'L',3360,160)
    _matrix(data['calibration_features'],'seen-RX known V',14400,160)
    y=data['train_y']
    if (y.shape!=(3360,) or y.dtype.kind not in 'iu' or np.any(y<0) or np.any(y>3)
            or not np.array_equal(np.bincount(y,minlength=4),np.full(4,840))):
        raise ValueError('Visible source L four-class contract differs')
    bundle=geometry.fit_source(data['train_features'],y,data['calibration_features'])
    _check_bundle(bundle)
    output.mkdir(parents=True,exist_ok=False)
    _save(output/'bundle.npz',bundle)
    result=dict(status='SOURCE_STREAM_CALIBRATED',source_only=True,target_reads=0,
        train_count=3360,calibration_count=14400,class_count=4,U_reads=0,future_reads=0,
        source_summary='aggregate_only_no_individual_source_features_or_labels',
        calibration_role='seen_source_RX1_3_4_6_known_V_only',heldout_RX8_calibration=False,
        quantile=.99,quantile_method='higher',lambda_det=bundle['lambda_det'],lambda_cluster=bundle['lambda_cluster'])
    _write(output/'complete.json',result)
    print('SOURCE_STREAM_CALIBRATED',bundle['lambda_det'],flush=True)
    return result


def _check_bundle(value):
    if not isinstance(value,dict) or set(value)!=BUNDLE_FIELDS:raise ValueError('Aggregate bundle whitelist differs')
    geometry._check_bundle(value)
    if (value['class_count']!=4 or value['source_count']!=3360 or value['calibration_count']!=14400
            or not np.array_equal(value['source_counts'],np.full(4,840))):
        raise ValueError('Own source aggregate identity differs')


def load_bundle(path):
    path=Path(path)
    done=_marker(path.parent/'complete.json','SOURCE_STREAM_CALIBRATED')
    if (done.get('train_count')!=3360 or done.get('calibration_count')!=14400 or done.get('class_count')!=4
            or done.get('U_reads')!=0 or done.get('future_reads')!=0):
        raise ValueError('Source calibration completion differs')
    values=_npz(path,BUNDLE_FIELDS)
    for key in ('schema','feature_dim','class_count','source_count','calibration_count','lambda_det','lambda_cluster'):
        if values[key].shape!=():raise ValueError('Scalar aggregate metadata required: '+key)
        values[key]=values[key].item()
    _check_bundle(values)
    return values


def _features(path,count):
    path=Path(path)
    if set(p.name for p in path.parent.iterdir())!={'features.npz'}:
        raise ValueError('Feature stream public directory must contain features.npz only')
    data=_npz(path,FEATURE_FIELDS)
    ids=data['ids']
    if (ids.shape!=(count,) or ids.dtype.kind!='U' or np.any(ids=='') or len(set(ids.tolist()))!=count):
        raise ValueError('Unique aligned opaque feature IDs required')
    _matrix(data['features'],'public features',count,160)
    _matrix(data['logits'],'complete original head',count,4)
    return data


def _load_frozen_head(path):
    from .stream_export import validate_config
    from .branch_export import _parent,_load_model
    config=_read(path);validate_config(config)
    config={key:value for key,value in config.items() if key!='branch_export_root'}
    return _load_model(config,_parent(config),'cpu')


def _head_logits(model,features):
    import torch
    parts=[]
    with torch.no_grad():
        for start in range(0,len(features),128):
            x=torch.tensor(np.asarray(features[start:start+128],dtype=np.float32).tolist(),dtype=torch.float32)
            logits=model.classify_features(x)
            parts.append(np.asarray(logits.detach().cpu().tolist(),dtype=np.float32))
    values=np.concatenate(parts)
    _matrix(values,'domain-corrected complete frozen head',len(features),4)
    return values


def predict(bundle,discovery,future,model_config,output):
    """Freeze D0/D1 using U; only then open future features for per-row inference."""
    output=Path(output)
    if output.exists():raise FileExistsError('Stream prediction output cannot be overwritten')
    frozen=load_bundle(bundle)
    U=_features(discovery,1050)
    output.mkdir(parents=True,exist_ok=False)
    states={name:geometry.fit_stream(frozen,U['features'],U['logits'],U['ids'],adapt_domain=adapt)
        for name,adapt in (('D0',False),('D1',True))}
    registry={}
    for name,state in states.items():
        registry[name+'_anonymous_ids']=np.sort(state['anonymous_ids'][state['qualified']]).astype(np.int64)
        for key in ('converged','cap_reached','promotion_eligible'):
            registry[name+'_'+key]=np.asarray(state[key],dtype=bool)
        print('SOURCE_STREAM_STATE',name,'rounds',state['rounds'],'free',len(state['free_centers']),
            'registered',int(state['qualified'].sum()),'converged',state['converged'],'cap',state['cap_reached'],flush=True)
    # Persist all adaptation/registration state before the first future feature IO.
    _write(output/'states.json',dict(source_only=True,target_reads=0,query_fit=False,
        states=states,threshold_role='frozen_source_seen_RX_known_V',head_weights_changed=False))
    _save(output/'registry.npz',registry)
    u_new=[]
    for state in states.values():
        row=geometry.predict_geometry(U['features'],U['logits'].argmax(axis=1),frozen,state,registered=True)
        u_new.append(np.where(row['prediction']>=4,row['prediction'],-1))
    _save(output/'discovery_predictions.npz',dict(ids=U['ids'],state_names=np.asarray(('D0','D1')),
        anonymous_prediction=np.column_stack(u_new).astype(np.int64)))
    # No source sample, U truth, future truth or Full target input is read here.
    Q=_features(future,5400)
    if set(U['ids'].tolist()) & set(Q['ids'].tolist()):raise ValueError('Discovery and future IDs overlap')
    baseline=Q['logits'].argmax(axis=1).astype(np.int64)
    closed_D1=baseline.copy()
    if np.any(states['D1']['b']):
        model=_load_frozen_head(model_config)
        adjusted=geometry.corrected_head_features(Q['features'],frozen,states['D1'])
        closed_D1=_head_logits(model,adjusted).argmax(axis=1).astype(np.int64)
    columns=[];closed=[]
    for name in STATES:
        state=None if name=='F' else states[name[:2]]
        head=closed_D1 if name.startswith('D1') else baseline
        closed.append(head)
        columns.append(geometry.predict_geometry(Q['features'],head,frozen,state,registered=name.endswith('R1')))
    values=dict(ids=Q['ids'],state_names=np.asarray(STATES),baseline_closed=baseline,
        closed_prediction=np.column_stack(closed),knownness=np.column_stack([r['knownness'] for r in columns]),
        accepted=np.column_stack([r['accepted'] for r in columns]),prediction=np.column_stack([r['prediction'] for r in columns]))
    _save(output/'future_predictions.npz',values)
    result=dict(status='ALL_STREAM_PREDICTIONS_COMPLETE',truth_read=False,query_fit=False,
        source_only=True,target_reads=0,discovery_count=1050,future_count=5400,state_names=list(STATES),
        source_IQ_reads=0,individual_source_feature_reads=0,source_loader_reads=0,
        head_weights_changed=False,head_input_correction='D1 raw_f_minus_norm_f_cholC_b',
        states_frozen_before_future_read=True,F_decision_U_dependency=False,
        prediction_ref='future_predictions.npz',discovery_prediction_ref='discovery_predictions.npz',registry_ref='registry.npz')
    _write(output/'complete.json',result)
    print('ALL_STREAM_PREDICTIONS_COMPLETE',5400,5,flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='stage',required=True)
    cal=sub.add_parser('calibrate');cal.add_argument('--source-inputs',required=True);cal.add_argument('--output',required=True)
    pred=sub.add_parser('predict')
    for key in ('bundle','discovery','future','model-config','output'):pred.add_argument('--'+key,required=True)
    args=vars(parser.parse_args(argv));stage=args.pop('stage')
    result=calibrate(**args) if stage=='calibrate' else predict(**args)
    print(json.dumps(dict(status=result['status'])))
    return 0


if __name__=='__main__':raise SystemExit(main())
