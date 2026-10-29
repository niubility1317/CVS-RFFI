"""Frozen F0 identity / E51 auxiliary source aggregates for label-free target work."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

from . import development_transfer as frozen, source_practical_readout as readout
from . import source_practical_encoder_cache as cache_api, source_practical_export as exporter
from . import source_consensus as consensus, scores
from .source_open_export import DIMENSIONS, _tensor
from .source_open_run import _read, _write, _npz, _new
from .source_embedding_readout_run import _save
from .source_consensus_run import _load_new
from .source_risk_calibration import METHODS, RECEIVERS
from . import source_practical_readout as risk


RUN='20261029-phase3-unlabeled-registration-s2026092701-r01'
SCHEMA='pamf_phase3_unlabeled_registration_source_v1'
STATUS='UNLABELED_REGISTRATION_SOURCE_FROZEN'
CALIBRATED='UNLABELED_REGISTRATION_SOURCE_THRESHOLDS_FROZEN'
FIELDS={'run_id','code_commit','source_row_root','risk_prediction_root','aug_fit_root','aug_calibration_root',
    'training_cache_root','source_validation_export_root','dura_checkpoint','dura_release','output_root','batch_size'}
CONDITIONS=('C0','P1')
VIEWS=exporter.VIEWS
L_COUNT=6300
V_COUNT=27000
SELECTION_SEED=2026102935
STAT_FIELDS={'condition_names','eps','radius','sigma','anchor_units','anchor_scores','group_keys'}


def validate_config(c):
    if not isinstance(c,dict) or set(c)!=FIELDS:
        raise ValueError('Only fixed frozen source handles, no target/truth/unknown composition')
    if any(not isinstance(c[k],str) or not c[k].strip() for k in FIELDS-{'batch_size'}):
        raise ValueError('Nonempty frozen source paths and identities required')
    if c['run_id']!=RUN or not re.fullmatch('[0-9a-f]{40}',c['code_commit']) or type(c['batch_size']) is not int or c['batch_size']!=128:
        raise ValueError('Fixed registration run, actual commit and batch128 required')
    fit,cal=Path(c['aug_fit_root']),Path(c['aug_calibration_root'])
    if fit.name!='fit' or cal.name!='calibration' or fit.parent!=cal.parent:
        raise ValueError('Frozen augmented fit/calibration must share the existing source root')
    out=Path(c['output_root']).resolve()
    if any(out.is_relative_to(Path(c[k]).resolve()) for k in FIELDS-{'run_id','code_commit','output_root','batch_size','dura_checkpoint'}):
        raise ValueError('New source output cannot overwrite frozen inputs')
    return c


def _prepare(c):
    """All existing role/fit/identity handles are checked before model backend IO."""
    source=frozen._source_bundle(c)
    aug=readout.load_bundle(c['aug_fit_root'],c['aug_calibration_root'],'augmented_readout')
    cache=cache_api.cache_marker(c['training_cache_root'])
    exported=exporter.export_marker(c['source_validation_export_root'])
    identity=source['marker']['encoder_state_identity'];parent=source['marker']['parent_export_identity']
    g=aug.get('identities',{}).get('dura_identity')
    if (aug.get('arm')!='augmented_readout' or aug.get('identities',{}).get('encoder_state_identity')!=identity
            or aug['identities'].get('parent_export_identity')!=parent
            or not risk._unchanged(aug['geometry'],source['geometry'])
            or cache.get('encoder_state_identity')!=identity or exported.get('encoder_state_identity')!=identity
            or cache.get('parent_export_identity')!=parent or exported.get('parent_export_identity')!=parent
            or cache.get('dura_identity')!=g or exported.get('dura_identity')!=g
            or not isinstance(g,dict) or g.get('checkpoint')!=c['dura_checkpoint'] or g.get('release_root')!=c['dura_release']
            or cache.get('counts',{}).get('source_L')!=L_COUNT or exported.get('counts',{}).get('source_V')!=V_COUNT
            or cache.get('source_contract_handle',{}).get('counts',{}).get('L_s')!=L_COUNT
            or exported.get('descriptor_schema')!=consensus.DESCRIPTOR_SCHEMA
            or Path(exported.get('geometry_source_fit_root','')).resolve()!=(Path(c['source_row_root'])/'fit').resolve()):
        raise ValueError('Frozen E51/oldF0/aug-readout/clean geometry/G/source roles must match')
    source=dict(source,geometry=aug['geometry'],state=aug['state'])
    # Its old threshold array may remain as immutable provenance, but is never
    # consulted for selection, calibration, statistics or deployment output.
    return dict(source=source,aug=aug,cache=cache,export=exported)


def _backend(prepared,c,device):
    reader,original=frozen._load_readers(prepared['source'],device)
    dura,engine=frozen._load_dura(prepared['source'],c,device)
    return dict(reader=reader,original=original,dura=dura,engine=engine)


def _smoke(prepared,backend,device):
    frozen._synthetic_smoke(prepared['source'],backend['reader'],backend['original'],
        backend['engine'],list(CONDITIONS),device)
    return dict(status='PASS',real_checkpoint=True,synthetic_physical_rows=2,condition_names=list(CONDITIONS),
        synthetic_condition_rows=4,E51_forward_rows=4,F0_forward_rows=4,DURA_forward_rows=2,
        source_IQ_reads=0,source_V_reads=0,target_reads=0,optimizer_updates=0)


def _embedding(branches,geometry):
    from .unlabeled_registration_core import embedding
    return embedding(branches,geometry)


def _density(z,f0,y,rx,day):
    from .unlabeled_registration_core import source_density
    return source_density(z,f0,y,rx,day)


def _features(prepared,backend,raw,device):
    import torch
    x=_tensor(raw,device)
    with torch.no_grad():
        data=frozen._encode_new(backend['reader'],x)
        original=backend['original'].features(x)
    if original.dtype!=torch.float32 or tuple(original.shape)!=(len(x),160) or not bool(torch.isfinite(original).all()):
        raise ValueError('Frozen original F0 FP32 fused160 required')
    f0=np.asarray(original.detach().cpu().tolist(),np.float32)
    norms=np.linalg.norm(f0.astype(np.float64),axis=1,keepdims=True)
    if np.any(norms<=1e-12):raise ValueError('Nonzero F0 identity features required')
    f0=(f0/norms).astype(np.float32)
    branches={k:data['features__'+k] for k in DIMENSIONS}
    z=_embedding(branches,prepared['source']['geometry'])
    descriptor=consensus.descriptor(branches,data['logits'],prepared['source']['geometry'])
    static=scores.knownness_scores(branches['fused160'],data['logits'],prepared['source']['geometry']['branches']['fused160']['prototypes'])
    knownness=risk.score_compact(descriptor,static,prepared['source']['state'])
    if z.shape!=(len(x),688) or z.dtype!=np.float32 or not np.isfinite(z).all():
        raise ValueError('Finite five-branch source embedding688 required')
    return z,f0,knownness


def source_view_indices(count,condition):
    """No labels, group quotas or TX-dependent choices enter source view selection."""
    if type(count) is not int or count<=0 or condition not in CONDITIONS:
        raise ValueError('Positive physical count and C0/P1 required')
    return np.zeros(count,np.int64) if condition=='C0' else np.random.default_rng(SELECTION_SEED).integers(1,4,size=count,dtype=np.int64)


def _source_meta(meta):
    keys={'ids','y','domain','rx_i','day_i','eq_i','sig_i','base_index'}
    if set(meta)!=keys:raise ValueError('Only original source L metadata is allowed')
    if (meta['ids'].shape!=(L_COUNT,) or meta['ids'].dtype.kind!='U' or len(set(meta['ids'].tolist()))!=L_COUNT
            or np.any(np.char.strip(meta['ids'])=='') or set(meta['y'].tolist())!=set(range(6))
            or set(meta['rx_i'].tolist())!=set(RECEIVERS) or not np.isin(meta['day_i'],[1,2,3]).all()):
        raise ValueError('Complete original source-L six classes/RX/day/physical IDs required')
    for key in keys-{'ids'}:
        if meta[key].shape!=(L_COUNT,) or meta[key].dtype!=np.int64:raise ValueError('Aligned integer source L metadata required')


def _selected_batch(arrays,choice,start,stop):
    result=np.empty((stop-start,2,256),np.float32)
    for j,view in enumerate(VIEWS):
        offsets=np.flatnonzero(choice[start:stop]==j)
        if len(offsets):result[offsets]=arrays[view][start+offsets]
    if not np.isfinite(result).all():raise ValueError('Finite cached source input required')
    return result


def _check(backend,snapshots,source,before):
    for k,v in snapshots.items():frozen._check_model(backend[k],v)
    after=frozen._array_snapshot({k:source[k] for k in ('geometry','state','thresholds')})
    if set(after)!=set(before) or any(not np.array_equal(v,after[k]) for k,v in before.items()):
        raise ValueError('Frozen E51 geometry/augmented readout state changed')


def _statistics(values):
    if not isinstance(values,dict) or set(values)!=STAT_FIELDS:raise ValueError('Only source aggregate statistics are deployable')
    if values['condition_names'].tolist()!=list(CONDITIONS):raise ValueError('Statistics condition order differs')
    for key,shape,dtype in (('eps',(2,),np.float64),('radius',(2,),np.float64),('sigma',(2,7),np.float64),
            ('anchor_units',(2,90,160),np.float32),('anchor_scores',(2,90,7),np.float64),('group_keys',(90,3),np.int64)):
        x=values[key]
        if x.shape!=shape or x.dtype!=dtype or not np.isfinite(x).all():raise ValueError('Finite aggregate shape/dtype required: '+key)
    keys=np.asarray([(y,r,d) for y in range(6) for r in RECEIVERS for d in (1,2,3)],np.int64)
    if (not np.array_equal(values['group_keys'],keys) or np.any(values['eps']<0) or np.any(values['radius']<0)
            or np.any(values['sigma']<1e-6) or not np.allclose(np.linalg.norm(values['anchor_units'],axis=2),1.,atol=1e-5)):
        raise ValueError('Canonical old-source groups, unit anchors and sigma floor required')


def run(config,device='cuda:0'):
    validate_config(config);output=_new(config['output_root']);started=time.perf_counter()
    prepared=_prepare(config)  # Invalid fit/cache/V handles stop before loading model backends.
    import torch
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    backend=_backend(prepared,config,device);source=prepared['source']
    snapshots={k:frozen._snapshot_model(backend[k]) for k in ('reader','original','dura')}
    before=frozen._array_snapshot({k:source[k] for k in ('geometry','state','thresholds')})
    smoke=_smoke(prepared,backend,device);_check(backend,snapshots,source,before)
    arrays,meta=cache_api.load_role(config['training_cache_root'],'source_L',views=VIEWS)
    stats=dict(condition_names=np.asarray(CONDITIONS));selection={};anchors=[];means=[];eps=[];radii=[];group_keys=None
    try:
        _source_meta(meta)
        for condition in CONDITIONS:
            choice=source_view_indices(L_COUNT,condition)
            selection[condition]={v:int(np.sum(choice==j)) for j,v in enumerate(VIEWS)}
            parts=[]
            for start in range(0,L_COUNT,config['batch_size']):
                stop=min(start+config['batch_size'],L_COUNT)
                raw=_selected_batch(arrays,choice,start,stop)
                parts.append(_features(prepared,backend,raw,device))
                if start==0 or start%(config['batch_size']*16)==0 or stop==L_COUNT:
                    print('UNLABELED_SOURCE_L',condition,stop,L_COUNT,flush=True)
            z,f0,knownness=(np.concatenate([part[j] for part in parts]) for j in range(3))
            density=_density(z,f0,meta['y'],meta['rx_i'],meta['day_i'])
            keys=np.asarray(density['group_keys'],np.int64)
            indices=density['group_indices']
            if len(indices)!=90 or any(not len(idx) for idx in indices):raise ValueError('All90 known-source groups required')
            if group_keys is not None and not np.array_equal(keys,group_keys):raise ValueError('Source condition group order differs')
            group_keys=keys;eps.append(float(density['eps']));radii.append(float(density['radius']))
            anchors.append(np.asarray(density['anchor_units'],np.float32))
            means.append(np.asarray([knownness[np.asarray(idx,np.int64)].mean(axis=0) for idx in indices],np.float64))
            del parts,z,f0,knownness,density
    finally:
        for value in arrays.values():
            if isinstance(value,np.memmap):value._mmap.close()
    _check(backend,snapshots,source,before)
    # Four known-V compact views only; never reopen V IQ, OE21 or novel rows.
    validation={v:exporter.load_role_view(config['source_validation_export_root'],'source_V',v,include_y=True) for v in VIEWS}
    first=validation[VIEWS[0]];all_scores=[]
    for v in VIEWS:
        row=validation[v]
        if len(row['ids'])!=V_COUNT or any(not np.array_equal(row[k],first[k]) for k in ('ids','y','rx_i','day_i')):
            raise ValueError('Four source-V views must preserve original physical order/known labels')
        all_scores.append(risk.score_compact(row['descriptor'],row['static_scores'],source['state']))
    rows=list(validation.values())
    baseline=np.concatenate([r['baseline_closed'] for r in rows])
    calibration=risk.calibrate_known(np.concatenate([r['ids'] for r in rows]),np.concatenate([r['y'] for r in rows]),
        baseline,baseline,np.concatenate(all_scores),np.concatenate([r['rx_i'] for r in rows]),
        np.repeat(np.arange(4,dtype=np.int64),V_COUNT))
    thresholds=calibration.pop('thresholds')
    if any(z['overall']['encoder_closed_loss']!=0 for z in calibration['methods'].values()):
        raise ValueError('Calibration identity must be unchanged original F0, not auxiliary E51 head')
    stats.update(eps=np.asarray(eps,np.float64),radius=np.asarray(radii,np.float64),
        sigma=np.asarray([np.maximum(all_scores[0].std(axis=0),1e-6),
            np.maximum(np.concatenate(all_scores[1:]).std(axis=0),1e-6)],np.float64),
        anchor_units=np.asarray(anchors,np.float32),anchor_scores=np.asarray(means,np.float64),group_keys=group_keys)
    _statistics(stats);_check(backend,snapshots,source,before)
    output.mkdir(parents=True,exist_ok=False);folder=output/'frozen';folder.mkdir()
    _save(folder,source['geometry'],source['state'])
    with (output/'thresholds.npy').open('xb') as stream:np.save(stream,thresholds,allow_pickle=False)
    _npz(output/'statistics.npz',stats)
    identities=dict(auxiliary_encoder_identity=source['marker']['encoder_state_identity'],
        identity_parent_export_identity=source['marker']['parent_export_identity'],
        parent_augmented_readout_identity=prepared['aug']['identities'],dura_identity=prepared['cache']['dura_identity'])
    cal=output/'calibration';cal.mkdir()
    calibration.update(status=CALIBRATED,schema=SCHEMA,calibration_schema=risk.CALIBRATION_SCHEMA,
        identities=identities,source_only=True,target_reads=0,novel_scores_reads=0,OE_rows_read=0,
        original_F0_closed_equals_baseline=True,thresholds_ref='../thresholds.npy')
    _write(cal/'complete.json',calibration)
    marker=dict(status=STATUS,schema=SCHEMA,run_id=RUN,code_commit=config['code_commit'],config=config,
        identities=identities,method_names=list(METHODS),condition_names=list(CONDITIONS),descriptor_schema=consensus.DESCRIPTOR_SCHEMA,
        source_only=True,L_physical_per_condition=L_COUNT,L_encoder_forward_rows=2*2*L_COUNT,
        source_V_physical=V_COUNT,source_V_calibration_exposures=4*V_COUNT,
        sigma_exposures=dict(C0=V_COUNT,P1=3*V_COUNT),source_L_selection_seed=SELECTION_SEED,source_L_view_counts=selection,
        source_known_group_count=90,source_group_key_order='oldclass0..5, RX1/3/4/6/8, day1..3',
        eps_definition='per record group-excluding-self seventh other-neighbor distance, then N-record .95 quantile',
        radius_definition='F0 unit distance to same known-group leave-one-out unit center .95 quantile',
        statistics_ref='statistics.npz',thresholds_ref='thresholds.npy',frozen_ref='frozen',calibration_ref='calibration/complete.json',
        source_contract_handle=prepared['cache']['source_contract_handle'],source_V_export_identity=dict(
            encoder_state_identity=prepared['export']['encoder_state_identity'],parent_export_identity=prepared['export']['parent_export_identity']),
        source_V_labels_used_for_known_only_calibration=True,source_L_labels_used_for_aggregate_statistics=True,
        identity_model='original_F0_E50_frozen_defaulthead',auxiliary_model='E51_frozen_augmented_source_readout',
        old_augmented_thresholds_used=False,source_bank_deployed=False,source_row_features_saved=False,
        source_IQ_saved=False,source_V_IQ_forward_rows=0,real_source_G_forward_rows=0,
        OE21_rows_read=0,source_novel_rows_read=0,original_U_reads=0,target_reads=0,target_labels_read=0,
        optimizer_updates=0,new_encoder_updates=0,new_identity_head_updates=0,new_G_updates=0,new_readout_updates=0,
        parameters_buffers_unchanged=True,geometry_readout_arrays_unchanged=True,smoke=smoke,
        seconds=time.perf_counter()-started,anonymous_registration_performed=False,final_phase3_complete=False)
    loaded_g,loaded_s=_load_new(folder)
    if not risk._unchanged(loaded_g,source['geometry']) or not risk._unchanged(
            {k:loaded_s[k] for k in ('schema','descriptor_dim','method_names','mean','scale','models')},
            {k:source['state'][k] for k in ('schema','descriptor_dim','method_names','mean','scale','models')}):
        raise ValueError('Frozen augmented state/geometry copy differs after serialization')
    _write(output/'complete.json',marker)
    return marker


def load_source(root):
    """Aggregate/marker-only deployable source; no cached IQ or compact row access."""
    root=Path(root);marker=_read(root/'complete.json');cal=_read(root/'calibration/complete.json')
    if (marker.get('status')!=STATUS or marker.get('schema')!=SCHEMA or marker.get('run_id')!=RUN
            or marker.get('method_names')!=list(METHODS) or marker.get('condition_names')!=list(CONDITIONS)
            or marker.get('parameters_buffers_unchanged') is not True or marker.get('geometry_readout_arrays_unchanged') is not True
            or marker.get('old_augmented_thresholds_used') is not False or marker.get('source_bank_deployed') is not False
            or cal.get('status')!=CALIBRATED or cal.get('schema')!=SCHEMA
            or cal.get('identities')!=marker.get('identities') or cal.get('calibration_schema')!=risk.CALIBRATION_SCHEMA
            or cal.get('known_count')!=4*V_COUNT or cal.get('original_F0_closed_equals_baseline') is not True
            or any(marker.get(k)!=0 for k in ('OE21_rows_read','source_novel_rows_read','original_U_reads','target_reads',
                'target_labels_read','optimizer_updates','new_encoder_updates','new_identity_head_updates','new_G_updates','new_readout_updates'))):
        raise ValueError('Frozen source aggregates and known-only F0 calibration required')
    geometry,state=_load_new(root/'frozen');thresholds=np.load(root/'thresholds.npy',allow_pickle=False)
    if (thresholds.shape!=(7,) or thresholds.dtype!=np.float64 or not np.isfinite(thresholds).all()
            or any(thresholds[j]!=cal['methods'][name]['threshold'] for j,name in enumerate(METHODS))):
        raise ValueError('Frozen F0-risk thresholds artifact differs')
    with np.load(root/'statistics.npz',allow_pickle=False) as f:statistics={k:f[k].copy() for k in f.files}
    _statistics(statistics)
    return dict(geometry=geometry,state=state,thresholds=thresholds,statistics=statistics,marker=marker)


def main(argv=None):
    actual=list(sys.argv[1:] if argv is None else argv)
    print(json.dumps(dict(event='UNLABELED_REGISTRATION_SOURCE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=[sys.argv[0]]+actual)),flush=True)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args(actual);result=run(_read(args.config),args.device)
    print(json.dumps(dict(status=result['status'])),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
