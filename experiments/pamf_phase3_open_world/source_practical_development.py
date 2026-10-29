"""Frozen source practical readouts on isolated development Q, with no U fit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import os

import numpy as np

from . import development_transfer as transfer, development_alignment as alignment
from .source_risk_calibration import METHODS


RUN = '20261029-phase3-source-practical-readout-s2026092701-r01'
STATES = ('original', 'calibration_only', 'augmented_readout')
ARMS = STATES[1:]
PRIMARY = 'mlp16_oe'
FIELDS = {'run_id','code_commit','condition','view_family','Q_public_root','Q_capsule_id','Q_split_id',
    'Q_expected_count','source_row_root','risk_prediction_root','d0_prediction_root','output_root','batch_size',
    'dura_checkpoint','dura_release','fit_root','calibration_root'}
PREDICTION_FIELDS = {'ids','state_names','method_names','closed_prediction','baseline_closed_prediction',
                     'knownness','accepted','prediction','thresholds'}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def validate_config(c):
    if not isinstance(c,dict) or set(c)!=FIELDS:
        raise ValueError('Only frozen source models and public Q inputs; no U, truth or unknown composition')
    for key in FIELDS-{'Q_expected_count','batch_size','dura_checkpoint','dura_release'}:
        if not isinstance(c[key],str) or not c[key].strip():raise ValueError('Nonempty identity/path required: '+key)
    if (c['run_id']!=RUN or len(c['code_commit'])!=40 or any(x not in '0123456789abcdef' for x in c['code_commit'])
            or type(c['Q_expected_count']) is not int or c['Q_expected_count']!=864
            or type(c['batch_size']) is not int or c['batch_size']!=128):
        raise ValueError('Fixed source-practical run, actual commit and public counts required')
    family={'C0':'clean_cross_receiver','P1':'practical_cross_receiver'}
    if c['condition'] not in family or c['view_family']!=family[c['condition']]:raise ValueError('Condition and input family differ')
    public=Path(c['Q_public_root'])
    if public.name!='public' or public.parent.name!='evaluation':raise ValueError('Only development evaluation public is allowed')
    if Path(c['output_root']).resolve().is_relative_to(public.resolve()):raise ValueError('Cannot write prediction into public input')
    if c['condition']=='C0':
        if c['dura_checkpoint'] is not None or c['dura_release'] is not None:raise ValueError('Clean has no G or paired reference')
    elif any(not isinstance(c[k],str) or not c[k].strip() for k in ('dura_checkpoint','dura_release')):
        raise ValueError('Practical P1 requires frozen G, no fallback')
    return c


def _base(c):
    return dict(run_id=c['run_id'],code_commit=c['code_commit'],public_root=c['Q_public_root'],
        capsule_id=c['Q_capsule_id'],split_id=c['Q_split_id'],expected_count=c['Q_expected_count'],
        source_row_root=c['source_row_root'],risk_prediction_root=c['risk_prediction_root'],output_root=c['output_root'],
        input_paths=[c['condition']],batch_size=c['batch_size'],view_family=c['view_family'],
        dura_checkpoint=c['dura_checkpoint'],dura_release=c['dura_release'])


def _sources(c):
    from .source_practical_readout import load_bundle
    bundles={arm:load_bundle(c['fit_root'],c['calibration_root'],arm) for arm in ARMS}
    old=transfer._source_bundle(_base(c))
    for arm,bundle in bundles.items():
        ident=bundle['identities'];g=ident['dura_identity']
        if (ident['run_id']!=RUN or ident['arm']!=arm
                or ident['encoder_state_identity']!=old['marker']['encoder_state_identity']
                or ident['view_order']!=['clean','practical_low_suburban','practical_mid_urban','practical_low_urban']
                or g['deployment_id']!='dura_v1' or g['arm']!='stage2' or g['epoch']!=20 or g['updates']!=4940
                or g['training_commit']!='b2a2ce937a64654e17e84c2aa817b9da4257a48f' or g['parameters']!=266951
                or g['reuse'] is not True or g['graph'] is not True or g['native'] is not False
                or g['precision']!='FP32' or g['TF32'] is not False):
            raise ValueError('Source E51, four-view training/calibration and original frozen G identity differ')
        if c['condition']=='P1' and (g['checkpoint']!=c['dura_checkpoint'] or g['release_root']!=c['dura_release']):
            raise ValueError('Training and inference must use the same declared frozen DURA')
    if bundles[ARMS[0]]['identities']['export_identity']!=bundles[ARMS[1]]['identities']['export_identity']:
        raise ValueError('Both source arms must share one source export and calibration population')
    return old,bundles


def _backend(source,c,device):
    reader,original=transfer._load_readers(source,device)
    if c['condition']=='P1':
        g,engine=transfer._load_dura(source,_base(c),device)
        return reader,original,(g,engine)
    return reader,original,None


def _public(c):
    return transfer._public(_base(c))


def _reference(c,source,ids):
    return alignment._reference(c,source,ids)


def _compact(reader,x,source):
    from .source_consensus import descriptor
    from .source_open_export import DIMENSIONS
    from .scores import knownness_scores
    data=transfer._encode_new(reader,x)
    branches={name:data['features__'+name] for name in DIMENSIONS}
    desc=descriptor(branches,data['logits'],source['geometry'])
    static=knownness_scores(branches['fused160'],data['logits'],source['geometry']['branches']['fused160']['prototypes'])
    return desc,static,data['logits'].argmax(1).astype(np.int64)


def _score_compact(desc,static,state):
    from .source_practical_readout import score_compact
    return score_compact(desc,static,state)


def _smoke(source,bundles,reader,original,g,c,device):
    import torch
    engine=None if g is None else g[1]
    transfer._synthetic_smoke(source,reader,original,engine,[c['condition']],device)
    t=torch.arange(256,dtype=torch.float32,device=device)
    x=torch.stack((torch.stack((torch.cos(.11*t),torch.sin(.11*t))),torch.stack((torch.cos(.19*t),torch.sin(.19*t)))))
    if engine is not None:
        with torch.no_grad():x=engine(x)[0].detach().clone()
    desc,static,_=_compact(reader,x,source)
    for arm in ARMS:
        scores=_score_compact(desc,static,bundles[arm]['state'])
        if scores.shape!=(2,7) or not np.isfinite(scores).all():raise ValueError('Actual frozen source practical readout smoke failed')


def predict(c,device='cuda:0'):
    validate_config(c);output=Path(c['output_root'])
    if output.exists():raise FileExistsError('Immutable development output already exists')
    source,bundles=_sources(c)
    import torch
    from .source_open_export import _tensor
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    reader,original,g=_backend(source,c,device);engine=None if g is None else g[1]
    models=[reader,original]+([] if g is None else [g[0]])
    before=[transfer._snapshot_model(m) for m in models]
    arrays=transfer._array_snapshot(dict(source=source,bundles=bundles))
    _smoke(source,bundles,reader,original,g,c,device)
    manifest,ids,iq=_public(c)
    try:
        old_marker,old=_reference(c,source,ids)
        n=len(ids);values=np.empty((3,n,7),np.float64);closed=np.empty((3,n),np.int64)
        values[0]=old['knownness'];closed[0]=old['closed_prediction']
        for start in range(0,n,c['batch_size']):
            stop=min(start+c['batch_size'],n);x=_tensor(iq[start:stop],device)
            if engine is not None:
                with torch.no_grad():x=engine(x)[0].detach().clone()
            desc,static,current_closed=_compact(reader,x,source)
            for index,arm in enumerate(ARMS,1):
                values[index,start:stop]=_score_compact(desc,static,bundles[arm]['state'])
                closed[index,start:stop]=current_closed
            if start==0 or stop==n:print('SOURCE_PRACTICAL_DEV_PROGRESS',c['condition'],stop,n,flush=True)
        if not np.isfinite(values).all():raise ValueError('Nonfinite source-frozen development score')
        for model,snapshot in zip(models,before):transfer._check_model(model,snapshot)
        after=transfer._array_snapshot(dict(source=source,bundles=bundles))
        if set(after)!=set(arrays) or any(not np.array_equal(arrays[k],after[k]) for k in arrays):raise ValueError('Source aggregate/readout/threshold state changed on Q')
        thresholds=np.stack([source['thresholds']]+[bundles[arm]['thresholds'] for arm in ARMS])
        accepted=values>=thresholds[:,None,:]
        prediction=np.where(accepted,closed[:,:,None],-1).astype(np.int64)
        baseline=np.repeat(old['baseline_closed_prediction'][None,:],3,axis=0)
        output.mkdir(parents=True,exist_ok=False)
        with (output/'predictions.npz').open('xb') as stream:
            np.savez(stream,ids=ids,state_names=np.asarray(STATES),method_names=np.asarray(METHODS),
                closed_prediction=closed,baseline_closed_prediction=baseline,knownness=values,
                accepted=accepted,prediction=prediction,thresholds=thresholds)
        result=dict(status='SOURCE_PRACTICAL_DEVELOPMENT_PREDICTIONS_COMPLETE',schema='pamf_phase3_source_practical_dev_v1',
            run_id=c['run_id'],code_commit=c['code_commit'],condition=c['condition'],view_family=c['view_family'],
            count=n,state_names=list(STATES),method_names=list(METHODS),primary_state='augmented_readout',primary_method=PRIMARY,
            public_capsule_id=c['Q_capsule_id'],public_split_id=c['Q_split_id'],
            encoder_state_identity=source['marker']['encoder_state_identity'],source_row_root=c['source_row_root'],
            fit_root=c['fit_root'],calibration_root=c['calibration_root'],source_arms_frozen_before_Q=True,
            D0_reused=True,D0_run_id=old_marker['run_id'],same_input_original_F6_baseline_reused=True,
            current_two_arms_share_one_forward_and_head=True,current_closed_diff_from_historical_D0=int(np.sum(closed[1]!=closed[0])),
            truth_read=False,query_fit=False,target_U_reads=0,formal_target_reads=0,source_IQ_reads=0,
            new_encoder_updates=0,new_identity_head_updates=0,new_readout_updates=0,new_threshold_calibrations=0,
            frozen_parameters_buffers_unchanged=True,frozen_source_aggregates_unchanged=True,
            real_checkpoint_synthetic_smoke_passed=True,anonymous_registration_performed=False,final_phase3_complete=False)
        _write(output/'complete.json',result);return result
    finally:
        if hasattr(iq,'_mmap'):iq._mmap.close()


def _predictions(root):
    root=Path(root)
    try:done=_read(root/'complete.json')
    except (OSError,ValueError) as exc:raise ValueError('Truth closed: complete source-frozen development predictions required') from exc
    if (done.get('status')!='SOURCE_PRACTICAL_DEVELOPMENT_PREDICTIONS_COMPLETE'
            or done.get('state_names')!=list(STATES) or done.get('method_names')!=list(METHODS)
            or done.get('primary_state')!='augmented_readout' or done.get('primary_method')!=PRIMARY
            or done.get('query_fit') is not False or done.get('truth_read') is not False
            or done.get('source_arms_frozen_before_Q') is not True
            or done.get('frozen_parameters_buffers_unchanged') is not True
            or done.get('frozen_source_aggregates_unchanged') is not True
            or any(done.get(k)!=0 for k in ('target_U_reads','formal_target_reads','source_IQ_reads','new_encoder_updates',
                                          'new_identity_head_updates','new_readout_updates','new_threshold_calibrations'))):
        raise ValueError('Truth closed: frozen state or no-target-fit boundary differs')
    with np.load(root/'predictions.npz',allow_pickle=False) as stored:
        if set(stored.files)!=PREDICTION_FIELDS:raise ValueError('Truth closed: prediction schema differs')
        p={k:stored[k].copy() for k in stored.files}
    n=done['count']
    if (p['ids'].shape!=(n,) or p['ids'].dtype.kind!='U' or len(set(p['ids'].tolist()))!=n or np.any(p['ids']=='')
            or p['state_names'].tolist()!=list(STATES) or p['method_names'].tolist()!=list(METHODS)
            or p['knownness'].shape!=(3,n,7) or not np.isfinite(p['knownness']).all()
            or p['thresholds'].shape!=(3,7) or p['thresholds'].dtype!=np.float64 or not np.isfinite(p['thresholds']).all()
            or p['accepted'].shape!=(3,n,7) or p['accepted'].dtype.kind!='b'
            or p['prediction'].shape!=(3,n,7) or p['prediction'].dtype.kind not in 'iu'):
        raise ValueError('Truth closed: aligned finite prediction arrays required')
    for k in ('closed_prediction','baseline_closed_prediction'):
        if p[k].shape!=(3,n) or p[k].dtype.kind not in 'iu' or np.any(p[k]<0) or np.any(p[k]>=6):raise ValueError('Truth closed: complete old-six head/baseline required')
    if (not np.array_equal(p['closed_prediction'][1],p['closed_prediction'][2])
            or not np.array_equal(p['accepted'],p['knownness']>=p['thresholds'][:,None,:])
            or not np.array_equal(p['prediction'],np.where(p['accepted'],p['closed_prediction'][:,:,None],-1))):
        raise ValueError('Truth closed: shared head, thresholds or decisions differ')
    return done,p


def score(predictions_root,truth,output,targets):
    done,p=_predictions(predictions_root)
    spec,minimum=transfer._targets(targets,done['view_family'])
    actual=transfer._truth(truth,p['ids']);output=Path(output)
    y,rx,scenarios=actual['y'],actual['receiver'],actual['scenario']
    if not bool((rx==26).all()):raise ValueError('Only the RX24-1 development receiver may be scored here')
    if np.any(y>=9):raise ValueError('Independent development truth TX mapping differs')
    if output.exists():raise FileExistsError('Immutable development score output exists')
    tx=['14-10','14-7','20-15','20-19','6-15','8-20','19-15','19-16','7-1']
    rows={}
    for i,name in enumerate(STATES):
        methods={}
        for j,method in enumerate(METHODS):
            def metrics(mask):return transfer._metrics(y[mask],p['closed_prediction'][i,mask],p['baseline_closed_prediction'][i,mask],p['knownness'][i,mask,j],p['accepted'][i,mask,j])
            methods[method]=dict(overall=metrics(np.ones(len(y),bool)),
                by_scenario={s:metrics(scenarios==s) for s in sorted(set(scenarios.tolist()))},
                by_transmitter={tx[int(label)]:metrics(y==label) for label in np.unique(y)},
                planned_discovery_unknown=metrics(np.isin(y,[6,7])),planned_later_unknown=metrics(y==8),
                diagnostic_only=method!=PRIMARY)
        main=methods[PRIMARY]
        later=main['planned_later_unknown']['unknown_macro_rejection']
        later_checks=[later is not None and later>=minimum]
        for s in set(scenarios.tolist()):
            mask=(scenarios==s)&(y==8)
            later_checks.append(bool(np.any(mask)) and float(np.mean(~p['accepted'][i,mask,6]))>=minimum)
        rows[name]=dict(methods=methods,primary_goals=transfer._goals(main['overall'],main['by_scenario'],minimum,done['view_family']),
                        primary_later_unknown_protection=all(later_checks))
    keys=('unknown_macro_rejection','known_macro_acceptance','known_macro_open_accuracy','known_macro_frr','original_closed_minus_new_open','auroc_known','oscr')
    deltas={pair:{method:{key:None if rows[a]['methods'][method]['overall'][key] is None or rows[b]['methods'][method]['overall'][key] is None
        else rows[a]['methods'][method]['overall'][key]-rows[b]['methods'][method]['overall'][key] for key in keys} for method in METHODS}
        for pair,a,b in [('calibration_only_minus_original','calibration_only','original'),('augmented_minus_calibration_only','augmented_readout','calibration_only'),('augmented_minus_original','augmented_readout','original')]}
    primary=rows['augmented_readout'];passed=primary['primary_goals']['pass'] and primary['primary_later_unknown_protection']
    result=dict(status='SCORED_SOURCE_PRACTICAL_DEVELOPMENT_COMPLETE',truth_last=True,development_only=True,
        condition=done['condition'],view_family=done['view_family'],count=len(y),states=rows,deltas=deltas,
        primary_state='augmented_readout',primary_method=PRIMARY,primary_pass=passed,
        state_selection='Augmented MLP predeclared; calibration-only and original diagnostic, no target selection',
        target_U_reads=0,query_fit=False,formal_target_reads=0,anonymous_registration_performed=False,final_phase3_complete=False,
        next_step='SOURCE_PRACTICAL_PASSED_DEVELOPMENT_ONLY' if passed else 'NO_PROMOTION',
        unknown_role_note='Source-only candidate has read no target U; planned cohorts are scorer-only dataset roles, not adaptation exposure',
        condition_unknown_rejection_min=minimum,targets_schema=spec['schema'])
    output.mkdir(parents=True,exist_ok=False);_write(output/'scores.json',result)
    _write(output/'complete.json',{k:result[k] for k in ('status','truth_last','condition','primary_state','primary_method','primary_pass','next_step','final_phase3_complete')})
    return result


def main(argv=None):
    print(json.dumps(dict(event='SOURCE_PRACTICAL_DEVELOPMENT_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv if argv is None else argv)),flush=True)
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    pred=sub.add_parser('predict');pred.add_argument('--config',required=True);pred.add_argument('--device',default='cuda:0')
    s=sub.add_parser('score');s.add_argument('--predictions-root',required=True);s.add_argument('--truth',required=True);s.add_argument('--output',required=True);s.add_argument('--targets',required=True)
    a=p.parse_args(argv);r=predict(_read(a.config),a.device) if a.action=='predict' else score(a.predictions_root,a.truth,a.output,a.targets)
    print(json.dumps(dict(status=r['status'])),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
