"""Own-L geometry/readouts and truth-last source evaluation for new encoders."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

from . import development_transfer as frozen, source_open_readout as core
from . import source_practical_export as augmentation, source_practical_readout as risk
from . import source_embedding_readout_run as serialization
from . import source_consensus as consensus, scores
from .source_open_export import DIMENSIONS, _tensor
from .source_open_run import _read, _write, _npz, _new
from .source_risk_calibration import METHODS, RECEIVERS


RUN = '20261029-phase3-source-practical-encoder-s2026092701-r01'
ARMS = ('clean_increment', 'practical_increment')
VIEWS = augmentation.VIEWS
COUNTS = dict(augmentation.COUNTS)
FIELDS = {'run_id','code_commit','arm','train_root','source_row_root','risk_prediction_root',
    'oe_data_root','dura_checkpoint','dura_release','output_root','batch_size'}
SCHEMA = 'pamf_phase3_source_practical_encoder_evaluation_v1'
FIT = 'SOURCE_PRACTICAL_ENCODER_READOUTS_FROZEN'
CALIBRATED = 'SOURCE_PRACTICAL_ENCODER_THRESHOLDS_FROZEN'
PREDICTED = 'ALL_SOURCE_PRACTICAL_ENCODER_PREDICTIONS_COMPLETE'
COMPLETE = 'SOURCE_PRACTICAL_ENCODER_SOURCE_COMPLETE'
SCORED = 'SCORED_SOURCE_PRACTICAL_ENCODER_COMPLETE'


def validate_config(config):
    if not isinstance(config,dict) or set(config)!=FIELDS:
        raise ValueError('Exactly the fixed source-only encoder evaluation inputs are allowed')
    if any(not isinstance(config[k],str) or not config[k].strip() for k in FIELDS-{'batch_size'}):
        raise ValueError('Nonempty immutable source/model paths and identities required')
    if (config['run_id']!=RUN or config['arm'] not in ARMS or not re.fullmatch('[0-9a-f]{40}',config['code_commit'])
            or type(config['batch_size']) is not int or config['batch_size']!=128):
        raise ValueError('Fixed encoder run/arm/actual commit and batch128 required')
    train,output=Path(config['train_root']),Path(config['output_root'])
    if output.name!='source' or output.parent!=train.parent or train.parent.name!=config['arm']:
        raise ValueError('Evaluation source output must belong to its own encoder row')
    source,risk_root=Path(config['source_row_root']),Path(config['risk_prediction_root'])
    if (source.name!='uniform_oe' or source.parent.name!='rows' or source.parent.parent.name!=frozen.READOUT_RUN
            or risk_root.name!='predict' or risk_root.parent.name!='uniform_oe'
            or risk_root.parent.parent.name!='rows' or risk_root.parent.parent.parent.name!=frozen.RISK_RUN
            or Path(config['oe_data_root']).name!='source_oe'):
        raise ValueError('Existing E51 source identity, F0 baseline and original source OE roles required')
    if any(output.resolve().is_relative_to(Path(config[k]).resolve()) for k in
            ('train_root','source_row_root','risk_prediction_root','oe_data_root','dura_release')):
        raise ValueError('New source output cannot overwrite an immutable input')
    return config


def _encoder_identity(identity,config,parent):
    if (not isinstance(identity,dict) or identity.get('run_id')!=RUN
            or identity.get('arm')!=config['arm'] or identity.get('code_commit')!=config['code_commit']
            or identity.get('updates')!=222 or identity.get('cumulative_updates')!=11544
            or identity.get('parent_encoder_identity')!=parent
            or not isinstance(identity.get('checkpoint'),str)
            or Path(identity['checkpoint']).parent!=Path(config['train_root'])):
        raise ValueError('New encoder must be this own-row 222-update child of the fixed E51')
    return identity


def _setup(config,device):
    # Resolve the current release's channel before historical F/G loaders alter paths.
    api=augmentation._canonical_channel_api()
    source=frozen._source_bundle(config)
    previous,baseline=frozen._load_readers(source,device)
    del previous  # Keep its verified train_config/data/F0 provenance, not its E51 forward.
    from .source_practical_encoder import load_trained
    model=load_trained(config['train_root'],device)
    identity=_encoder_identity(getattr(model,'_phase3_encoder_identity',None),config,
        source['marker']['encoder_state_identity'])
    reader=model.id_backbone.encoder
    dura,engine=frozen._load_dura(source,config,device)
    handles=augmentation.original._oe_handles(config['oe_data_root'])
    if handles!=source['marker']['source_OE_handles']:
        raise ValueError('Original four-OE/novel source handles differ')
    return dict(model=model,reader=reader,baseline=baseline,dura=dura,engine=engine,source=source,
        contract=augmentation._source_contract(source),identity=identity,api=api,sessions={},
        channels={v:augmentation._channel_config(v) for v in VIEWS[1:]})


def _load_role(config,current,role):
    return augmentation._load_role(config,current['source'],current['contract'],role)


def _meta(raw,meta,role,seen):
    keys={'ids','rx_i','day_i','domain'}|({'y'} if role in ('source_L','source_V') else set())
    if not isinstance(meta,dict) or set(meta)!=keys:
        raise ValueError('Only legal source metadata; OE/novel TX/y are forbidden')
    n=COUNTS[role];ids=np.asarray(meta['ids'])
    if (len(raw)!=n or ids.shape!=(n,) or ids.dtype.kind!='U' or np.any(np.char.strip(ids)=='')
            or len(set(ids.tolist()))!=n or seen & set(ids.tolist())):
        raise ValueError('Complete disjoint source role physical IDs/count required')
    for key in ('rx_i','day_i','domain'):
        if np.asarray(meta[key]).shape!=(n,) or np.asarray(meta[key]).dtype.kind not in 'iu':
            raise ValueError('Aligned integer source RX/day/domain required')
    if set(np.asarray(meta['rx_i']).tolist())!=set(RECEIVERS) or not np.isin(meta['day_i'],[1,2,3]).all():
        raise ValueError('Only original source RX/day roles are allowed')
    if 'y' in keys:
        y=np.asarray(meta['y'])
        if y.shape!=(n,) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(6)):
            raise ValueError('Only all six known source labels are allowed')
    seen.update(ids.tolist())
    return {k:np.asarray(v).copy() for k,v in meta.items()}


def _encode_batch(current,raw,meta,view,device,baseline=False):
    import torch
    values=augmentation._raw(raw)
    if view not in VIEWS:raise ValueError('Only the fixed four source views are allowed')
    if view!='clean':
        values=augmentation._synthesize(values,meta,view,current['channels'],current['api'],current['sessions'])
    x=_tensor(values,device)  # Byte-buffer bridge supports Torch2.1 + NumPy2.
    with torch.no_grad():
        if view!='clean':x=current['engine'](x)[0].detach().clone()
        if x.dtype!=torch.float32 or tuple(x.shape)!=(len(values),2,256) or not bool(torch.isfinite(x).all()):
            raise ValueError('Finite full256 FP32 source observation required after frozen G')
        data=frozen._encode_new(current['reader'],x)
        old=frozen._original_closed(current['baseline'],x) if baseline else np.full(len(values),-2,np.int64)
    branches={name:data['features__'+name] for name in DIMENSIONS}
    core._branches(branches)
    logits=data['logits']
    if logits.shape!=(len(values),6) or logits.dtype!=np.float32 or not np.isfinite(logits).all():
        raise ValueError('Complete new native default-head FP32 logits required')
    return dict(branches=branches,logits=logits,baseline_closed=old)


def _smoke(current,device):
    t=np.arange(256)
    raw=np.asarray([[np.cos(.11*t),np.sin(.11*t)],[np.cos(.19*t+.3),np.sin(.19*t+.3)]],np.float32)
    meta=dict(ids=np.asarray(['encoder-source-synthetic-0','encoder-source-synthetic-1']),
        rx_i=np.asarray([1,3],np.int64),day_i=np.asarray([1,2],np.int64))
    for view in VIEWS:
        result=_encode_batch(current,raw,meta,view,device,True)
        core._branches(result['branches'])
        if result['logits'].shape!=(2,6) or result['baseline_closed'].shape!=(2,):
            raise ValueError('New encoder actual-checkpoint smoke failed')
    result=dict(status='PASS',real_checkpoint=True,synthetic_physical_rows=2,synthetic_view_rows=8,
        source_IQ_reads=0,target_reads=0,optimizer_updates=0,view_order=list(VIEWS),
        encoder_forward_rows=8,original_F_forward_rows=8,DURA_forward_rows=6,channel_rows=6)
    print('SOURCE_PRACTICAL_ENCODER_SMOKE_PASS',json.dumps(result),flush=True)
    return result


def _compact(encoded,geometry):
    branches,logits=encoded['branches'],encoded['logits']
    return dict(descriptor=consensus.descriptor(branches,logits,geometry),
        static_scores=scores.knownness_scores(branches['fused160'],logits,geometry['branches']['fused160']['prototypes']),
        logits=logits,baseline_closed=encoded['baseline_closed'])


def _role_views(config,current,raw,meta,role,geometry,device,output,*,clean_full=None):
    n=COUNTS[role];result={}
    for view in VIEWS:
        values=dict(ids=meta['ids'].copy(),rx_i=np.asarray(meta['rx_i'],np.int64),day_i=np.asarray(meta['day_i'],np.int64),
            descriptor=np.empty((n,100),np.float32),static_scores=np.empty((n,5),np.float64),
            logits=np.empty((n,6),np.float32),baseline_closed=np.full(n,-2,np.int64))
        if role in ('source_L','source_V'):values['y']=np.asarray(meta['y'],np.int64).copy()
        if view=='clean' and clean_full is not None:
            values.update(_compact(clean_full,geometry))
        else:
            for start in range(0,n,config['batch_size']):
                stop=min(n,start+config['batch_size']);part={k:meta[k][start:stop] for k in ('ids','rx_i','day_i')}
                encoded=_encode_batch(current,raw[start:stop],part,view,device,role in ('source_V','novel_validation'))
                for key,value in _compact(encoded,geometry).items():values[key][start:stop]=value
                if start==0 or start%(config['batch_size']*32)==0 or stop==n:
                    print('SOURCE_PRACTICAL_ENCODER_SOURCE_PROGRESS',role,view,stop,n,flush=True)
        augmentation._arrays(values,role,n)
        path=output/'compact'/role/view/'data.npz';path.parent.mkdir(parents=True)
        _npz(path,values);result[view]=values
    return result


def _close(raw):
    if isinstance(raw,np.memmap):raw._mmap.close()


def _check_models(current,snapshots,old_arrays):
    for key,before in snapshots.items():frozen._check_model(current[key],before)
    after=frozen._array_snapshot({k:current['source'][k] for k in ('geometry','state','thresholds')})
    if set(after)!=set(old_arrays) or any(not np.array_equal(x,after[k]) for k,x in old_arrays.items()):
        raise ValueError('Original source aggregates changed')


def _common(config,current):
    return dict(schema=SCHEMA,run_id=RUN,code_commit=config['code_commit'],arm=config['arm'],
        encoder_identity=current['identity'],parent_export_identity=current['source']['marker']['parent_export_identity'],
        descriptor_schema=consensus.DESCRIPTOR_SCHEMA,method_names=list(METHODS),view_order=list(VIEWS),
        dura_identity=dict(deployment_id='dura_v1',checkpoint=config['dura_checkpoint'],release_root=config['dura_release'],
            arm='stage2',epoch=20,updates=4940,training_commit='b2a2ce937a64654e17e84c2aa817b9da4257a48f',
            parameters=266951,reuse=True,graph=True,native=False,precision='FP32',TF32=False),
        source_only=True,target_reads=0,original_U_s_reads=0,OE_truth_reads=0,
        new_encoder_updates=0,new_identity_head_updates=0,new_G_updates=0,
        encoder_updates_scope='Evaluation0; encoder identity records historical own-row222/cumulative11544',
        default_head_replaced=False)


def run(config,device='cuda:0'):
    validate_config(config);output=_new(config['output_root']);started=time.perf_counter()
    import torch
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    current=_setup(config,device)
    _encoder_identity(current['identity'],config,current['source']['marker']['encoder_state_identity'])
    snapshots={k:frozen._snapshot_model(current[k]) for k in ('model','baseline','dura')}
    old_arrays=frozen._array_snapshot({k:current['source'][k] for k in ('geometry','state','thresholds')})
    smoke=_smoke(current,device)
    _check_models(current,snapshots,old_arrays)
    output.mkdir(parents=True,exist_ok=False);_write(output/'smoke.json',smoke)
    common=_common(config,current);seen=set()
    # Full clean L branches live in memory only and define this encoder's own geometry.
    raw,meta=_load_role(config,current,'source_L')
    try:
        meta=_meta(raw,meta,'source_L',seen);parts=[]
        for start in range(0,COUNTS['source_L'],config['batch_size']):
            stop=min(COUNTS['source_L'],start+config['batch_size'])
            part={k:meta[k][start:stop] for k in ('ids','rx_i','day_i')}
            parts.append(_encode_batch(current,raw[start:stop],part,'clean',device))
        clean=dict(branches={name:np.concatenate([x['branches'][name] for x in parts]) for name in DIMENSIONS},
            logits=np.concatenate([x['logits'] for x in parts]),baseline_closed=np.full(COUNTS['source_L'],-2,np.int64))
        geometry=core.fit_geometry(clean['branches'],meta['y'])
        known=_role_views(config,current,raw,meta,'source_L',geometry,device,output,clean_full=clean)
        del parts,clean
    finally:_close(raw)
    _check_models(current,snapshots,old_arrays)
    raw,meta=_load_role(config,current,'exposure_train')
    try:
        meta=_meta(raw,meta,'exposure_train',seen)
        oe=_role_views(config,current,raw,meta,'exposure_train',geometry,device,output)
    finally:_close(raw)
    geometry_before=risk._arrays(geometry)
    fit_started=time.perf_counter()
    state=core.fit_readouts(np.concatenate([known[v]['descriptor'] for v in VIEWS]),
        np.concatenate([oe[v]['descriptor'] for v in VIEWS]),2026092701)
    training=state.get('training',{})
    if (training.get('steps')!=200 or training.get('known_count')!=4*COUNTS['source_L']
            or training.get('oe_count')!=4*COUNTS['exposure_train']
            or training.get('parameter_counts')!={'linear_oe':101,'mlp16_oe':1633}
            or set(training.get('losses',{}))!=set(core.METHODS)
            or any(len(x)!=200 or not np.isfinite(x).all() for x in training['losses'].values())):
        raise ValueError('Own L/OE four-view readouts must complete exactly200 CPU steps each')
    if not risk._unchanged(geometry,geometry_before):raise ValueError('Clean-L geometry changed during readout fitting')
    _check_models(current,snapshots,old_arrays)
    fit=output/'fit';fit.mkdir();serialization._save(fit,geometry,state)
    loaded_g,loaded_s=serialization.serialization._load_new(fit)
    if not risk._unchanged(geometry,loaded_g) or not np.array_equal(
            risk.score_compact(known[VIEWS[0]]['descriptor'][:2],known[VIEWS[0]]['static_scores'][:2],state),
            risk.score_compact(known[VIEWS[0]]['descriptor'][:2],known[VIEWS[0]]['static_scores'][:2],loaded_s)):
        raise ValueError('Own source geometry/readout serialization roundtrip differs')
    _write(fit/'training.json',dict(training=training,seconds=time.perf_counter()-fit_started,
        geometry_source='this new encoder clean L only',source_L_physical=COUNTS['source_L'],
        source_OE_physical=COUNTS['exposure_train'],source_V_fit_reads=0,novel_validation_reads=0,
        geometry_refitted_from_new_clean_L=True,source_membership_supervision=True,encoder_head_G_frozen=True))
    _write(fit/'complete.json',dict(common,status=FIT,geometry_ref='geometry.npz',readouts_ref='readouts.npz',
        own_clean_L_geometry_fitted=True,readout_steps_per_model=200,source_L_count=COUNTS['source_L'],
        source_L_view_exposures=4*COUNTS['source_L'],source_OE_count=COUNTS['exposure_train'],
        source_OE_view_exposures=4*COUNTS['exposure_train'],source_V_fit_reads=0,novel_validation_reads=0))
    del known,oe
    own_before=risk._arrays(dict(geometry=geometry,state=state))
    raw,meta=_load_role(config,current,'source_V')
    try:
        meta=_meta(raw,meta,'source_V',seen)
        validation=_role_views(config,current,raw,meta,'source_V',geometry,device,output)
    finally:_close(raw)
    _check_models(current,snapshots,old_arrays)
    rows=[validation[v] for v in VIEWS]
    calibrated=risk.calibrate_known(np.concatenate([r['ids'] for r in rows]),np.concatenate([r['y'] for r in rows]),
        np.concatenate([r['logits'].argmax(1) for r in rows]),np.concatenate([r['baseline_closed'] for r in rows]),
        np.concatenate([risk.score_compact(r['descriptor'],r['static_scores'],state) for r in rows]),
        np.concatenate([r['rx_i'] for r in rows]),np.repeat(np.arange(4,dtype=np.int64),COUNTS['source_V']))
    if not risk._unchanged(dict(geometry=geometry,state=state),own_before):
        raise ValueError('Known-V calibration changed the fitted source state')
    calibration=output/'calibration';calibration.mkdir();thresholds=calibrated.pop('thresholds')
    with (calibration/'thresholds.npy').open('xb') as stream:np.save(stream,thresholds,allow_pickle=False)
    calibrated.update(common,status=CALIBRATED,calibration_schema=risk.CALIBRATION_SCHEMA,
        thresholds_ref='thresholds.npy',source_V_labels_read=True,source_known_label_calibration=True,
        unknown_calibration_scores_read=0,unknown_calibration_labels_read=0,novel_validation_reads=0,new_readout_updates=0)
    _write(calibration/'complete.json',calibrated)
    # No novel IQ, descriptors or prediction references were opened before this marker.
    raw,meta=_load_role(config,current,'novel_validation')
    try:
        meta=_meta(raw,meta,'novel_validation',seen)
        novel=_role_views(config,current,raw,meta,'novel_validation',geometry,device,output)
    finally:_close(raw)
    references={};predict=output/'predict';predict.mkdir()
    for view in VIEWS:
        k,n=validation[view],novel[view];closed=np.r_[k['logits'].argmax(1),n['logits'].argmax(1)].astype(np.int64)
        values=risk.score_compact(np.concatenate((k['descriptor'],n['descriptor'])),
            np.concatenate((k['static_scores'],n['static_scores'])),state)
        accepted=values>=thresholds;path=predict/view;path.mkdir()
        _npz(path/'predictions.npz',dict(ids=np.r_[k['ids'],n['ids']],method_names=np.asarray(METHODS),thresholds=thresholds,
            closed_prediction=closed,baseline_closed_prediction=np.r_[k['baseline_closed'],n['baseline_closed']],
            knownness=values,accepted=accepted,prediction=np.where(accepted,closed[:,None],-1).astype(np.int64),
            rx_i=np.r_[k['rx_i'],n['rx_i']],day_i=np.r_[k['day_i'],n['day_i']],
            known_y=np.r_[k['y'],np.full(COUNTS['novel_validation'],-1,np.int64)]))
        references[view]=view+'/predictions.npz'
    _check_models(current,snapshots,old_arrays)
    if not risk._unchanged(dict(geometry=geometry,state=state),own_before):
        raise ValueError('Source prediction changed frozen own geometry/readouts')
    disk_thresholds=np.load(calibration/'thresholds.npy',allow_pickle=False)
    if not np.array_equal(thresholds,disk_thresholds):raise ValueError('Frozen threshold state changed')
    prediction=dict(common,status=PREDICTED,known_count=COUNTS['source_V'],novel_count=COUNTS['novel_validation'],
        count_per_view=COUNTS['source_V']+COUNTS['novel_validation'],prediction_refs=references,
        truth_read=False,unknown_truth_read=False,query_fit=False,new_readout_updates=0,new_threshold_calibrations=0)
    _write(predict/'complete.json',prediction)
    result=dict(common,status=COMPLETE,config=config,primary_method='mlp16_oe',smoke=smoke,
        source_role_counts=dict(COUNTS),source_physical_count=sum(COUNTS.values()),source_view_exposures=4*sum(COUNTS.values()),
        source_encoder_forward_rows=4*sum(COUNTS.values()),original_F_forward_rows=4*(COUNTS['source_V']+COUNTS['novel_validation']),
        DURA_forward_rows=3*sum(COUNTS.values()),source_V_fit_reads=0,novel_fit_reads=0,
        fit_ref='fit',calibration_ref='calibration',predictions_ref='predict',seconds=time.perf_counter()-started,
        frozen_parameters_buffers_unchanged=True,frozen_own_source_arrays_unchanged=True,
        own_clean_L_geometry_fitted=True,readout_steps_per_model=200,source_membership_supervision=True,
        source_fail_does_not_cancel_development=True,anonymous_registration_performed=False,final_phase3_complete=False)
    _write(output/'complete.json',result)
    return result


def load_bundle(source_root):
    """Load only own aggregate arrays/scalars and JSON markers, never source rows."""
    root=Path(source_root);source=_read(root/'complete.json')
    fit=_read(root/'fit/complete.json');calibration=_read(root/'calibration/complete.json')
    if (source.get('status')!=COMPLETE or source.get('schema')!=SCHEMA or source.get('run_id')!=RUN
            or source.get('arm') not in ARMS or source.get('source_only') is not True
            or source.get('own_clean_L_geometry_fitted') is not True
            or source.get('readout_steps_per_model')!=200):
        raise ValueError('A complete own-row encoder source evaluation is required')
    for marker,status in ((fit,FIT),(calibration,CALIBRATED)):
        if (marker.get('status')!=status or marker.get('schema')!=SCHEMA
                or any(marker.get(k)!=source.get(k) for k in ('run_id','arm','code_commit','encoder_identity',
                    'parent_export_identity','descriptor_schema','method_names','view_order','dura_identity'))
                or marker.get('method_names')!=list(METHODS) or marker.get('view_order')!=list(VIEWS)
                or marker.get('source_only') is not True
                or any(marker.get(k)!=0 for k in ('target_reads','original_U_s_reads','OE_truth_reads',
                    'new_encoder_updates','new_identity_head_updates','new_G_updates'))):
            raise ValueError('Own frozen source fit/calibration identity differs')
    if (fit.get('source_V_fit_reads')!=0 or fit.get('novel_validation_reads')!=0
            or fit.get('own_clean_L_geometry_fitted') is not True or fit.get('readout_steps_per_model')!=200
            or calibration.get('known_count')!=4*COUNTS['source_V']
            or calibration.get('calibration_schema')!=risk.CALIBRATION_SCHEMA
            or calibration.get('source_V_labels_read') is not True
            or any(calibration.get(k)!=0 for k in ('unknown_calibration_scores_read','unknown_calibration_labels_read',
                'novel_validation_reads','new_readout_updates'))):
        raise ValueError('Known-only calibration and L/OE-only fitting required')
    geometry,state=serialization.serialization._load_new(root/'fit')
    thresholds=np.load(root/'calibration/thresholds.npy',allow_pickle=False)
    if (thresholds.shape!=(7,) or thresholds.dtype!=np.float64 or not np.isfinite(thresholds).all()
            or any(thresholds[j]!=calibration['methods'][n]['threshold'] for j,n in enumerate(METHODS))):
        raise ValueError('Seven frozen known-only threshold scalars required')
    return dict(geometry=geometry,state=state,thresholds=thresholds,encoder_identity=source['encoder_identity'],
        dura_identity=source['dura_identity'],fit_marker=fit,calibration_marker=calibration,
        source_marker=source,method_names=list(METHODS))


def _predictions(source_root):
    bundle=load_bundle(source_root);root=Path(source_root)/'predict';marker=_read(root/'complete.json')
    if (marker.get('status')!=PREDICTED or marker.get('schema')!=SCHEMA
            or marker.get('encoder_identity')!=bundle['encoder_identity']
            or marker.get('known_count')!=COUNTS['source_V'] or marker.get('novel_count')!=COUNTS['novel_validation']
            or marker.get('method_names')!=list(METHODS) or marker.get('view_order')!=list(VIEWS)
            or marker.get('truth_read') is not False or marker.get('unknown_truth_read') is not False
            or marker.get('query_fit') is not False):
        raise ValueError('Private truth closed: all four own-row predictions must be fixed')
    expected={'ids','method_names','thresholds','closed_prediction','baseline_closed_prediction',
        'knownness','accepted','prediction','rx_i','day_i','known_y'}
    rows={};reference=None;n=COUNTS['source_V']+COUNTS['novel_validation'];nk=COUNTS['source_V']
    for view in VIEWS:
        ref=view+'/predictions.npz'
        if marker.get('prediction_refs',{}).get(view)!=ref:raise ValueError('Missing predicted source view')
        with np.load(root/ref,allow_pickle=False) as f:
            if set(f.files)!=expected:raise ValueError('Own source prediction schema differs')
            data={k:f[k].copy() for k in f.files}
        if (data['ids'].shape!=(n,) or data['ids'].dtype.kind!='U' or len(set(data['ids'].tolist()))!=n
                or np.any(np.char.strip(data['ids'])=='') or data['method_names'].tolist()!=list(METHODS)):
            raise ValueError('Prediction IDs/count/method order differ')
        for key in ('closed_prediction','baseline_closed_prediction','rx_i','day_i','known_y'):
            if data[key].shape!=(n,) or data[key].dtype!=np.int64:raise ValueError('Aligned source prediction metadata required')
        if (any(np.any((data[k]<0)|(data[k]>=6)) for k in ('closed_prediction','baseline_closed_prediction'))
                or set(data['rx_i'].tolist())!=set(RECEIVERS) or not np.isin(data['day_i'],[1,2,3]).all()
                or set(data['known_y'][:nk].tolist())!=set(range(6)) or not np.all(data['known_y'][nk:]==-1)):
            raise ValueError('Source only known labels, default classes and RX/day required')
        score,accept=data['knownness'],data['accepted']
        if (not np.array_equal(data['thresholds'],bundle['thresholds']) or score.shape!=(n,7)
                or score.dtype.kind!='f' or not np.isfinite(score).all() or accept.shape!=(n,7)
                or accept.dtype.kind!='b' or not np.array_equal(accept,score>=bundle['thresholds'])
                or not np.array_equal(data['prediction'],np.where(accept,data['closed_prediction'][:,None],-1))):
            raise ValueError('Frozen source predictions and known-only thresholds disagree')
        meta={k:data[k] for k in ('ids','rx_i','day_i','known_y')}
        if reference is not None and not risk._unchanged(meta,reference):raise ValueError('Source views do not share original ID order/metadata')
        rows[view]=data;reference=meta
    return bundle,rows


def _novel_truth(path,ids):
    return risk._novel_truth(path,ids)


def score_source(source_root,novel_truth,output):
    output=_new(output);bundle,pred=_predictions(source_root)
    first=pred[VIEWS[0]];nk=COUNTS['source_V'];tx=_novel_truth(novel_truth,first['ids'][nk:])
    mapping={t:6+i for i,t in enumerate(sorted(set(tx.tolist())))}
    y=np.r_[first['known_y'][:nk],np.asarray([mapping[t] for t in tx.tolist()],np.int64)]
    txmap={i:t for i,t in enumerate(('14-10','14-7','20-15','20-19','6-15','8-20'))}|{v:k for k,v in mapping.items()}
    from .development_transfer import _metrics
    methods={}
    for j,name in enumerate(METHODS):
        def metrics(view,mask):
            d=pred[view]
            return _metrics(y[mask],d['closed_prediction'][mask],d['baseline_closed_prediction'][mask],d['knownness'][mask,j],d['accepted'][mask,j])
        by_view={v:metrics(v,np.ones(len(y),bool)) for v in VIEWS}
        by_rx_view={str(r):{v:metrics(v,pred[v]['rx_i']==r) for v in VIEWS} for r in RECEIVERS}
        by_tx_view={t:{v:metrics(v,y==k) for v in VIEWS} for k,t in txmap.items()}
        joined={k:np.concatenate([pred[v][k] for v in VIEWS]) for k in
            ('closed_prediction','baseline_closed_prediction','knownness','accepted','rx_i')};yy=np.tile(y,4)
        def pooled(mask):
            z=_metrics(yy[mask],joined['closed_prediction'][mask],joined['baseline_closed_prediction'][mask],joined['knownness'][mask,j],joined['accepted'][mask,j])
            z['confidence_intervals95']=None;z['pooled_views_CI_not_reported']=True
            for key in ('known_per_class','unknown_per_class'):
                for row in z[key]:row['rejection_interval95']=None
            return z
        overall=pooled(np.ones(len(yy),bool));by_rx={str(r):pooled(joined['rx_i']==r) for r in RECEIVERS}
        by_tx={t:pooled(yy==k) for k,t in txmap.items()}
        overall_known=risk._known_goals(overall);rx_known={r:risk._known_goals(z) for r,z in by_rx.items()}
        view_goals={v:risk._goals(z,.95 if v=='clean' else .90) for v,z in by_view.items()}
        cross_goals={r:{v:risk._goals(z,.95 if v=='clean' else .90) for v,z in vv.items()} for r,vv in by_rx_view.items()}
        passed=(overall_known['passed'] and all(x['passed'] for x in rx_known.values())
            and all(x['passed'] for x in view_goals.values()) and all(x['passed'] for vv in cross_goals.values() for x in vv.values()))
        methods[name]=dict(overall=overall,by_view=by_view,by_receiver=by_rx,by_transmitter=by_tx,
            by_receiver_view=by_rx_view,by_transmitter_view=by_tx_view,
            goals=dict(overall_known=overall_known,by_receiver_known=rx_known,by_view=view_goals,by_receiver_view=cross_goals,passed=passed),
            diagnostic_only=name!='mlp16_oe')
    result=dict(status=SCORED,schema=SCHEMA,run_id=RUN,arm=bundle['source_marker']['arm'],source_only=True,
        target_reads=0,truth_last=True,unknown_truth_read=True,encoder_identity=bundle['encoder_identity'],
        dura_identity=bundle['dura_identity'],method_names=list(METHODS),view_order=list(VIEWS),methods=methods,
        primary_method='mlp16_oe',primary_source_goals=methods['mlp16_oe']['goals'],
        known_count_per_view=nk,novel_count_per_view=len(tx),unique_physical_validation_count=len(y),
        derived_view_count=4*len(y),novel_label_to_transmitter={str(v):k for k,v in mapping.items()},
        source_candidate_selection=False,source_failure_cancels_development=False,
        source_novel_evaluation_reused=True,independent_confirmation=False,
        evidence_scope='Repeated source novel at original source RX; same-known-V empirical calibration; not cross-RX evidence',
        domain_adaptation_performed=False,anonymous_registration_performed=False,final_phase3_complete=False,
        next_step='SOURCE_SCORED_FIXED_DEVELOPMENT_CONTINUES')
    output.mkdir(parents=True,exist_ok=False);_write(output/'scores.json',result)
    _write(output/'complete.json',dict(status=SCORED,schema=SCHEMA,truth_last=True,source_only=True,target_reads=0,
        arm=result['arm'],encoder_identity=bundle['encoder_identity'],primary_source_goals=result['primary_source_goals'],
        source_failure_cancels_development=False,final_phase3_complete=False,next_step=result['next_step']))
    return result


def main(argv=None):
    actual=list(sys.argv[1:] if argv is None else argv)
    print(json.dumps(dict(event='SOURCE_PRACTICAL_ENCODER_EVAL_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=[sys.argv[0]]+actual)),flush=True)
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='action',required=True)
    command=commands.add_parser('run');command.add_argument('--config',required=True);command.add_argument('--device',default='cuda:0')
    command=commands.add_parser('score');command.add_argument('--source-root',required=True)
    command.add_argument('--novel-truth',required=True);command.add_argument('--output',required=True)
    args=parser.parse_args(actual)
    result=run(_read(args.config),args.device) if args.action=='run' else score_source(args.source_root,args.novel_truth,args.output)
    print(json.dumps(dict(status=result['status'])),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
