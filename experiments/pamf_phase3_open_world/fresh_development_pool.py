"""Offline RX3-2 early/future selection and one fixed G10 observation per row."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
import zipfile
import numpy as np
from . import star_channel as channel
from .data import PUBLIC_FIELDS,_identity,load_public as _star_public
from .development_clean import load_clean_public as _clean_public
from .full_pool import _read_json,_write_json,_read_numpy_pickle

RUN_ID='20261031-phase3-fresh-development-rx3-2-s2026092701-r01'
SCHEMA='pamf_phase3_fresh_development_builder_v1'
STATUS='FRESH_DEVELOPMENT_POOL_VALIDATED'
RX='3-2'
RX_INDEX=34
OLD=['14-10','14-7','20-15','20-19','6-15','8-20']
NEW=['5-5','10-7','2-19']
LATER=['7-11']
SOURCE_RX=['1-19','18-2','19-2','2-19','3-19']
EARLY='2021_03_01'
FUTURE_DAYS=('2021_03_15','2021_03_23')
U_PER_TX=64
U_COUNT=576
Q_COUNT=10238
Q_BY_DAY=dict(zip(FUTURE_DAYS,(5525,4713)))
SEEDS=dict(selection=2026103131,receiver=2027,scene=2026103133,channel=2026103134,shuffle=2026103135)
EPISODE=dict(id='MIX01',scene_seed=SEEDS['scene'],channel_seed=SEEDS['channel'])
FIELDS={'run_id','code_commit','validation_root','full_root','metadata_audit','mixture_config_path'}
TRUTH_FIELDS={'ids','y','tx','receiver','day','scenario'}


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in FIELDS)):
        raise ValueError('Exact six-field fixed private builder configuration required')
    return c


def _metadata(c):
    root=Path(c['validation_root']);summary=_read_json(root/'summary.json');m=_read_json(root/'counts.json');audit=_read_json(c['metadata_audit'])
    if (summary.get('status')!='VERIFIED' or Path(summary.get('dataset_directory','')).resolve()!=Path(c['full_root']).resolve()
            or any(summary.get(k)!=v for k,v in dict(members_checked=158,expected_nonempty_RX_date_members=158,count_mismatch_cells=0,errors=[]).items())):
        raise ValueError('Reuse the matching completed Full validation handle')
    if set(m)!={'dates','receivers','transmitters','records'} or any(len(m[k])!=n or len(set(m[k]))!=n for k,n in (('dates',4),('receivers',41),('transmitters',174))):raise ValueError('Full metadata namespace differs')
    counts=np.asarray(m['records'])
    if counts.shape!=(4,41,174) or counts.dtype.kind not in 'iu' or np.any(counts<0) or m['receivers'][RX_INDEX]!=RX:raise ValueError('Full integer count table/RX index differs')
    scope=dict(receiver=RX,Full_receiver_index=RX_INDEX,early_day=EARLY,future_days=list(FUTURE_DAYS),registered_classes=OLD,fresh_discovery_candidates=NEW,future_only_unknown_candidates=LATER)
    if audit.get('schema')!='pamf_phase3_fresh_development_metadata_audit_v1' or any(audit.get('scope',{}).get(k)!=v for k,v in scope.items()):raise ValueError('Bounded receiver/TX metadata audit differs')
    exclusion=audit['fresh_TX_exclusions'];groups=exclusion['groups'];excluded_tx=set().union(*(set(v) for v in groups.values()))
    if groups.get('original_known_6')!=OLD or len(excluded_tx)!=34 or excluded_tx&set(NEW+LATER):raise ValueError('Fresh TX intersects declared known/OE/novel/formal roles')
    roles=audit['receiver_role_exclusions'];lists=roles['compact_RX_lists'];compact=set().union(*(set(v) for v in lists.values()))
    if (set(lists)!={'ManyRx.pkl','ManySig.pkl','ManyTx.pkl','SingleDay.pkl'} or len(compact)!=35 or compact!=set(roles['complete_compact_RX_union'])
            or roles['complete_compact_RX_union_count']!=35 or roles['current_source_receivers']!=SOURCE_RX
            or not set(SOURCE_RX+roles['current_Phase2_target_receivers']).issubset(compact)
            or roles['old_Phase3_development_receiver']!='24-1' or roles['formal_Phase3_reserved_receiver']!='24-18'):
        raise ValueError('Concrete compact/source/reserved receiver exclusions differ')
    excluded_rx=compact|set(SOURCE_RX+roles['current_Phase2_target_receivers']+['24-1','24-18'])
    examined=set().union(*(set(a['receiver_names']) for a in roles['examined_actual_canonical_role_indices']))
    if RX in excluded_rx|examined:raise ValueError('New receiver overlaps an examined existing role')
    cells={day:{tx:int(counts[m['dates'].index(day),RX_INDEX,m['transmitters'].index(tx)]) for tx in OLD+NEW+LATER} for day in (EARLY,*FUTURE_DAYS)}
    if min(cells[EARLY][tx] for tx in OLD+NEW)<U_PER_TX or any(sum(cells[d].values())!=Q_BY_DAY[d] for d in FUTURE_DAYS):raise ValueError('Fixed U/future metadata count differs')
    members={}
    with (root/'members.jsonl').open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line);day=row.get('date')
            if row.get('receiver')==RX and day in (EARLY,*FUTURE_DAYS):
                if day in members or row.get('status')!='PASS' or not all(isinstance(row.get(k),str) and row[k] for k in ('archive','member')):raise ValueError('Unique indexed PASS Full member required')
                members[day]=row
    if set(members)!={EARLY,*FUTURE_DAYS}:raise ValueError('Three existing RX3-2 members required')
    return summary,cells,members,audit,sorted(excluded_rx)


def _observe(raw,**kwargs):return channel.observe(raw,**kwargs)


def _channel_smoke(mixture):
    for i in range(2):
        raw=np.random.default_rng(2026103130+i).normal(size=(2,256)).astype(np.float64)
        value,_=_observe(raw,physical_id='synthetic-fresh-'+str(i),receiver=RX,day=EARLY,episode=EPISODE,mixture=mixture)
        if value.dtype!=np.float32 or value.shape!=(2,256) or not np.isfinite(value).all():raise ValueError('Synthetic actual G10 channel smoke failed')
    return dict(status='PASS',synthetic_channel_records=2,real_Full_reads=0)


def _read_member(c,locator):
    with zipfile.ZipFile(Path(c['full_root'])/locator['archive']) as archive:
        with archive.open(locator['member']) as stream:return _read_numpy_pickle(stream)


def _clean(raw):
    z=np.asarray(raw,np.float32)
    if z.shape!=(2,256) or not np.isfinite(z).all():raise ValueError('Finite FP32-convertible full256 raw input required')
    api,_=channel._channel_api();unit,_=api.normalize_input(z[0].astype(np.float64)+1j*z[1].astype(np.float64))
    return np.stack((unit.real,unit.imag)).astype(np.float32)


def _collect(c,cells,members,mixture):
    result={role:dict(ids=[],received=[],truth=[],mapping=[]) for role in ('discovery','future_star','future_clean')};canonical=set();api,_=channel._channel_api();labels={t:i for i,t in enumerate(OLD+NEW+LATER)}
    for day in (EARLY,*FUTURE_DAYS):
        payload=_read_member(c,members[day])
        if not isinstance(payload,dict) or set(payload)!={'node_list','data'}:raise ValueError('Full NumPy node/data member required')
        names,arrays=payload['node_list'],payload['data']
        if not isinstance(names,list) or not isinstance(arrays,list) or len(names)!=len(arrays) or len(set(names))!=len(names):raise ValueError('Unique aligned node_list/data required')
        lookup=dict(zip(names,arrays));wanted=OLD+NEW if day==EARLY else OLD+NEW+LATER
        for tx in wanted:
            count=cells[day][tx];values=lookup.get(tx)
            if values is None and count==0:continue
            if not isinstance(values,np.ndarray) or values.dtype!=np.float64 or values.ndim!=3 or values.shape[1:]!=(256,2) or len(values)!=count:raise ValueError('Selected Full TX array count/dtype/shape differs from existing metadata')
            indices=np.random.default_rng(api.stable_seed(SEEDS['selection'],tx,RX,day)).permutation(count)[:U_PER_TX] if day==EARLY else range(count)
            for index in indices:
                key=('FullWiSigEq',tx,RX,day,1,int(index))
                if key in canonical:raise ValueError('Duplicate canonical physical record')
                canonical.add(key);raw=values[int(index)].T.copy();physical=channel.identity(['WiSig/Full/eq1',tx,RX,day,int(index)])
                received,private=_observe(raw,physical_id=physical,receiver=RX,day=day,episode=EPISODE,mixture=mixture)
                if received.dtype!=np.float32 or received.shape!=(2,256) or not np.isfinite(received).all() or private.get('scenario') not in {x[0] for x in channel.SCENARIOS}:raise ValueError('One finite full256 G10 received observation required')
                identifier='p3fresh-'+_identity([RUN_ID,list(key)]);role='discovery' if day==EARLY else 'future_star'
                for target,value,scenario in [(role,received,private['scenario'])]+([] if day==EARLY else [('future_clean',_clean(raw),'clean')]):
                    a=result[target];a['ids'].append(identifier);a['received'].append(value);a['truth'].append((labels[tx],tx,RX_INDEX,day,scenario));a['mapping'].append(dict(opaque_id=identifier,canonical_record=list(key),physical_id=physical,source_archive=members[day]['archive'],source_member=members[day]['member'],**(private if target!='future_clean' else dict(scenario='clean'))))
        del lookup,arrays,payload
        print('FRESH_POOL_MEMBER_COMPLETE',day,flush=True)
    rng=np.random.default_rng(SEEDS['shuffle']);orders={role:rng.permutation(len(result[role]['ids'])) for role in ('discovery','future_star')};orders['future_clean']=orders['future_star']
    for role,a in result.items():
        order=orders[role];a['ids']=np.asarray(a['ids'])[order];a['received']=np.stack(a['received'])[order];a['truth']=[a['truth'][i] for i in order];a['mapping']=[a['mapping'][i] for i in order]
    if len(result['discovery']['ids'])!=U_COUNT or len(result['future_star']['ids'])!=Q_COUNT or len(canonical)!=U_COUNT+Q_COUNT:raise ValueError('Actual complete canonical counts differ')
    if set(result['discovery']['ids'].tolist())&set(result['future_star']['ids'].tolist()) or not np.array_equal(result['future_star']['ids'],result['future_clean']['ids']):raise ValueError('U/future disjointness or paired Q order differs')
    return result


def load_public(public_root,cap,split,expected_count):
    manifest=_read_json(Path(public_root)/'manifest.json');loader=_clean_public if manifest.get('schema')=='phase3_clean_min_v1' else _star_public
    m,ids,received=loader(public_root)
    if m['capsule_id']!=cap or m['split_id']!=split or m['count']!=expected_count or m['registered_classes']!=OLD:
        received._mmap.close();raise ValueError('Fixed public capsule/split/count/old-six identity differs')
    return m,ids,received


def build(c,output_root):
    validate_config(c);out=Path(output_root)
    if out.exists():raise FileExistsError('Immutable fresh pool output already exists')
    absolute=out.resolve()
    if any(absolute.is_relative_to(Path(c[k]).resolve()) or Path(c[k]).resolve().is_relative_to(absolute) for k in ('full_root','validation_root','metadata_audit','mixture_config_path')):raise ValueError('Pool output overlaps an input')
    summary,cells,members,audit,excluded=_metadata(c);mixture=channel.validate_mixture(_read_json(c['mixture_config_path']));smoke=_channel_smoke(mixture)
    observed=_collect(c,cells,members,mixture);out.mkdir(parents=True);handles={}
    for role,a in observed.items():
        clean=role=='future_clean';count=len(a['ids']);cap='p3fresh-'+_identity([RUN_ID,role,SEEDS,a['ids'].tolist()]);split=_identity([cap,'public-split'])
        manifest=dict(schema='phase3_clean_min_v1' if clean else 'phase3_min_v1',data_status='VALIDATED_ONCE',capsule_id=cap,split_id=split,count=count,iq_shape=[count,2,256],registered_classes=OLD.copy(),processing_route='clean' if clean else 'residual',mode='clean' if clean else 'post_sync',equalization_enabled=False,fs_hz=25000000,single_observation=True)
        if set(manifest)!=PUBLIC_FIELDS:raise ValueError('Public whitelist differs')
        public=out/role/'public';truth=out/role/'truth';public.mkdir(parents=True);truth.mkdir()
        for name in ('ids','received'):
            with (public/(name+'.npy')).open('xb') as stream:np.save(stream,a[name],allow_pickle=False)
        _write_json(public/'manifest.json',manifest);y,tx,rx,day,scene=zip(*a['truth'])
        with (truth/'truth.npz').open('xb') as stream:np.savez(stream,ids=a['ids'],y=np.asarray(y,np.int64),tx=np.asarray(tx),receiver=np.asarray(rx,np.int64),day=np.asarray(day),scenario=np.asarray(scene))
        _,_,checked=load_public(public,cap,split,count);checked._mmap.close();handles[role]=dict(public_root=str(public),truth_path=str(truth/'truth.npz'),count=count,capsule_id=cap,split_id=split)
    report=dict(schema=SCHEMA,run_id=RUN_ID,config=c,seeds=SEEDS,episode=EPISODE,channel_smoke=smoke,validation_reused=summary,metadata_audit=audit,metadata_counts=cells,indexed_members=members,
        receiver_role_exclusions=excluded,source_formal_history_receivers_disjoint=True,global_history_unused_verified=False,U_future_canonical_disjoint=True,canonical_physical_unique=True,unique_canonical_count=U_COUNT+Q_COUNT,
        mapping={k:a['mapping'] for k,a in observed.items()},actual_counts_by_TX_day={role:dict(Counter(t[1]+'/'+t[3] for t in a['truth'])) for role,a in observed.items()},
        archive_reads=3,pickle_reads=3,full_validation_repeated=False,channel_records=U_COUNT+Q_COUNT,clean_test_records=Q_COUNT,synthetic_channel_records=2,input_RMS_normalizations_per_view=1,output_RMS_normalization=False,clean_U_created=False,
        model_reads=0,model_predictions=0,optimizer_updates=0,formal_target_reads=0)
    _write_json(out/'builder_report.json',report)
    done=dict(status=STATUS,data_status='VALIDATED_ONCE',run_id=RUN_ID,code_commit=c['code_commit'],**handles,discovery_count=U_COUNT,evaluation_count=Q_COUNT,unique_canonical_count=U_COUNT+Q_COUNT,
        canonical_physical_unique=True,U_future_canonical_disjoint=True,paired_future_ids_and_order=True,source_formal_history_receivers_disjoint=True,excluded_receiver_count=len(excluded),global_history_unused_verified=False,
        archive_reads=3,pickle_reads=3,channel_records=U_COUNT+Q_COUNT,clean_test_records=Q_COUNT,synthetic_channel_records=2,clean_U_created=False,model_reads=0,model_predictions=0,optimizer_updates=0,formal_target_reads=0,development_only=True)
    _write_json(out/'complete.json',done);print(STATUS,U_COUNT,Q_COUNT,flush=True);return done


def main():
    print(json.dumps(dict(event='FRESH_POOL_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();build(_read_json(a.config),a.output)


if __name__=='__main__':main()
