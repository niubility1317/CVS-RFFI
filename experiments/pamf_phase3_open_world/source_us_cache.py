"""Original source U_s IDs -> one raw G10 observation -> frozen DURA P1.

Only the builder can locate TX/RX/day/eq/packet coordinates. The public role
contains no TX labels, coordinates, lookup table, teacher logits or raw paths.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys
import numpy as np
from . import star_relative_source as source
from . import star_representation_cache as cache

RUN_ID='20261031-phase3-source-us-replay-s2026092701-r01'
SCHEMA='pamf_phase3_source_us_cache_v1'
ROLE_SCHEMA='pamf_phase3_source_us_cache_role_v1'
STATUS='SOURCE_US_G10_CACHE_COMPLETE'
ROLE_STATUS='SOURCE_US_G10_ROLE_COMPLETE'
CAPSULE_ID='pamf-source-Us392005-G10-MIX01-20261031-r01'
SPLIT_ID='392005-Us-source5rx-3day-eq1'
COUNT=56700
FIELDS={'run_id','code_commit','source_config','output_root','batch_size'}
PUBLIC_FILES={'manifest.json','ids.npy','received.npy','restored.npy','metadata.npz'}
METADATA_FIELDS={'rx_i','day_i','domain'}
BUILDER_FIELDS=METADATA_FIELDS|{'ids','eq_i','sig_i','base_index'}
_read,_write=source._read,source._write
_backend,_snapshot,_check_models=cache._backend,cache._snapshot,cache._check_models
_smoke,_restore,_load_raw,_observe=cache._smoke,cache._restore,source._load_raw,source._observe


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128 or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):raise ValueError('Exact five-field source-Us cache config required; no labels/target inputs')
    s=c['source_config']
    if not isinstance(s,dict) or s.get('run_id')!=cache.RUN_ID:raise ValueError('The existing original star-representation source configuration is required')
    source.validate_config(dict(s,run_id=source.RUN_ID))
    out=Path(c['output_root']).resolve();inputs=[s[k] for k in ('source_cache','source_contract','raw_mainsig','output_root','dura_checkpoint','dura_release')]+[s['reader_config']['source_output']]
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in inputs):raise ValueError('New immutable Us cache cannot overlap any original input')
    return c


def _ids(ids,n):
    if ids.shape!=(n,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=n or np.any(np.char.strip(ids)==''):raise ValueError('Unique aligned original source Us opaque IDs required')


def _public_metadata(ids,meta):
    _ids(ids,COUNT)
    if set(meta)!=METADATA_FIELDS or any(a.shape!=(COUNT,) or a.dtype!=np.int64 for a in meta.values()):raise ValueError('Public Us metadata is exactly integer RX/day/domain; labels and coordinates forbidden')
    rx,day=meta['rx_i'],meta['day_i'];domain=np.searchsorted(source.RECEIVERS,rx)*3+day-1
    if not np.isin(rx,source.RECEIVERS).all() or not np.isin(day,(1,2,3)).all() or not np.array_equal(meta['domain'],domain) or set(domain.tolist())!=set(range(15)):raise ValueError('Original fifteen source RX/day domains required')


def _metadata(s):
    contract=source._contract(s)
    with np.load(Path(s['source_cache'])/'U_s_metadata.npz',allow_pickle=False) as stored:
        if set(stored.files)!=BUILDER_FIELDS:raise ValueError('Original Us metadata must be TX/y blind with only builder locator fields')
        meta={k:stored[k].copy() for k in stored.files}
    ids=meta['ids'];_ids(ids,COUNT)
    if contract['counts']['U_s']!=COUNT or ids.tolist()!=contract['role_ids']['U_s']:raise ValueError('Original source Us role identity/order differs')
    if any(meta[k].shape!=(COUNT,) or meta[k].dtype.kind not in 'iu' for k in BUILDER_FIELDS-{'ids'}):raise ValueError('Aligned original Us builder integers required')
    if (not np.isin(meta['rx_i'],source.RECEIVERS).all() or not np.isin(meta['day_i'],(1,2,3)).all() or not np.all(meta['eq_i']==1)
            or any(np.any(meta[k]<0) for k in ('sig_i','base_index','domain'))):raise ValueError('Only original source5RX/3day/eq1 Us locators are allowed')
    selected=set(ids.tolist())
    if any(selected&set(contract['role_ids'][role]) for role in ('L_s','V')):raise ValueError('Original source L/Us/V physical roles must be disjoint')
    return contract,meta


def _locate(payload,contract,meta):
    """Enumerate public raw coordinates; read IQ values only for selected Us IDs."""
    from experiments.cric_dura_comparison.data import sample_id
    if not isinstance(payload,dict) or not {'data','tx_list','rx_list','capture_date_list','equalized_list'}.issubset(payload):raise ValueError('Original raw ManySig namespace required')
    txs=list(map(str,payload['tx_list']));rxs=list(map(str,payload['rx_list']));days=list(map(str,payload['capture_date_list']));eqs=list(payload['equalized_list'])
    if (len(txs)!=6 or set(txs)!=set(contract['classes']) or len(set(txs))!=6 or rxs!=list(source.MAIN_RX_NAMES)
            or days!=['2021_03_01','2021_03_08','2021_03_15','2021_03_23'] or len(eqs)<=1 or eqs[1]!=1):raise ValueError('Original source TX/RX/day/eq index namespace differs')
    wanted=set(meta['ids'].tolist());lookup={};enumerated=0
    for tx in range(len(txs)):
        for rx in source.RECEIVERS:
            for day in (1,2,3):
                cell=payload['data'][tx][rx][day][1]
                for sig in range(0 if cell is None else len(cell)):
                    sid=sample_id('ManySig',tx,rx,day,1,sig);enumerated+=1
                    if sid in wanted:lookup[sid]=(tx,rx,day,1,sig)
    if set(lookup)!=wanted:raise ValueError('Every original Us ID must locate exactly its original raw record')
    records=[]
    for i,sid in enumerate(meta['ids']):
        row=lookup[str(sid)]
        if row[1:]!=tuple(int(meta[k][i]) for k in ('rx_i','day_i','eq_i','sig_i')):raise ValueError('Us ID and original raw locator metadata disagree')
        records.append(row)
    return records,enumerated,rxs,days


def _handle(contract):return {k:contract[k] for k in ('classes','source_rxs','source_days','split_seed','counts')}


def build(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root']);s=c['source_config']
    if out.exists():raise FileExistsError('New immutable source Us cache required')
    # Canonical G10 must resolve before legacy model bootstrap changes namespace.
    from .star_channel import _channel_api,validate_mixture
    _channel_api();mixture=validate_mixture(_read(s['mixture_config_path']));b=_backend(s,device,with_dura=True);snapshots=_snapshot(b)
    smoke=_smoke(b,s,device);_check_models(snapshots)
    if smoke.get('status')!='PASS':raise ValueError('Actual F0/G no-query checkpoint smoke must pass before source inputs')
    print('SOURCE_US_CHECKPOINT_SMOKE_PASS',flush=True);contract,meta=_metadata(s);payload=_load_raw(s['raw_mainsig']);records,enumerated,rxs,days=_locate(payload,contract,meta)
    public=dict(rx_i=meta['rx_i'].astype(np.int64),day_i=meta['day_i'].astype(np.int64));public['domain']=(np.searchsorted(source.RECEIVERS,public['rx_i'])*3+public['day_i']-1).astype(np.int64)
    ids=meta['ids'];_public_metadata(ids,public);out.mkdir(parents=True);folder=out/'U_s';folder.mkdir();_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    maps={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype=np.float32,shape=(COUNT,2,256)) for k in ('received','restored')};episode={k:v for k,v in s['episode'].items() if k!='condition'}
    try:
        for start in range(0,COUNT,c['batch_size']):
            stop=min(start+c['batch_size'],COUNT);received=np.empty((stop-start,2,256),np.float32)
            for j,index in enumerate(range(start,stop)):
                tx,rx,day,eq,sig=records[index];raw=source.raw_packet(payload['data'][tx][rx][day][eq][sig])
                if raw.dtype!=np.float64:raise ValueError('Original FP64 raw source packet required; normalized FP32 cache is not raw')
                value,_=_observe(raw,physical_id=str(ids[index]),receiver=rxs[rx],day=days[day],episode=episode,mixture=mixture)
                received[j]=cache._array(value,(2,256),'Us received')
            restored=cache._array(_restore(b,received,device),received.shape,'Us restored');maps['received'][start:stop]=received;maps['restored'][start:stop]=restored
            if stop%2048==0 or stop==COUNT:print('SOURCE_US_CACHE_PROGRESS',stop,COUNT,flush=True)
    finally:
        for value in maps.values():value.flush();value._mmap.close()
    _check_models(snapshots);del payload,records
    cache._save(folder/'ids.npy',ids)
    with (folder/'metadata.npz').open('xb') as stream:np.savez(stream,**public)
    manifest=dict(schema=ROLE_SCHEMA,status=ROLE_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],role='U_s',count=COUNT,iq_shape=[COUNT,2,256],precision='FP32',data_status='VALIDATED_ONCE',
        capsule_id=CAPSULE_ID,split_id=SPLIT_ID,files=sorted(PUBLIC_FILES),metadata_fields=sorted(METADATA_FIELDS),input_path='cached_P1_only',received_input_path='P0',restored_input_path='P1',source_only=True,
        parent_F0_identity=b['reader_identity'],dura_identity=b['dura_identity'],source_contract_handle=_handle(contract),episode=episode,receiver_seed=2027,second_RMS_applied=False,Us_labels_read=0)
    _write(folder/'manifest.json',manifest)
    _write(out/'builder_report.json',dict(schema=SCHEMA,run_id=RUN_ID,code_commit=c['code_commit'],builder_coordinate_enumeration=True,enumerated_source_coordinates=enumerated,selected_U_s_records=COUNT,
        raw_ManySig_whole_file_read=True,raw_payload_includes_other_roles=True,only_U_s_record_values_indexed=True,raw_packet_layout_only=True,observe_calls=COUNT,input_RMS_calls=COUNT,output_RMS_calls=0,
        physical_id_rule='Original contract sample_id ManySig: sha256(tx_i/rx_i/day_i/eq_i/sig_i)[:24]',source_roles_disjoint=True,Us_labels_read=0,private_truth_reads=0,source_locator_table_exported=False))
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,public_root=str(folder),manifest=manifest,data_status='VALIDATED_ONCE',original_U_s_reads=COUNT,
        Us_labels_read=0,private_truth_reads=0,target_reads=0,G_forward_rows=COUNT,real_F_forward_rows=0,original_models_unchanged=True,source_only=True,source_L_reads=0,source_V_reads=0,source_OE_reads=0,
        synthetic_F_forward_rows=smoke['synthetic_F0_rows'],G_synthetic_forward_rows=smoke['synthetic_G_rows'],optimizer_updates=0,new_independent_physical_records=0,
        fixed_G10_reconstruction=True,second_RMS_applied=False,final_phase3_complete=False)
    _write(out/'complete.json',done);print(STATUS,flush=True);return done


def load_public(root,cap,split,expected_count=None):
    """Only this five-file role is opened; private builder/parent files are unused."""
    root=Path(root);n=COUNT if expected_count is None else expected_count
    if type(n) is not int or n!=COUNT or cap!=CAPSULE_ID or split!=SPLIT_ID or {p.name for p in root.iterdir()}!=PUBLIC_FILES:raise ValueError('Exact source Us public files/count/capsule/split required')
    m=_read(root/'manifest.json');required=dict(schema=ROLE_SCHEMA,status=ROLE_STATUS,run_id=RUN_ID,role='U_s',count=n,iq_shape=[n,2,256],precision='FP32',data_status='VALIDATED_ONCE',
        capsule_id=cap,split_id=split,files=sorted(PUBLIC_FILES),metadata_fields=sorted(METADATA_FIELDS),input_path='cached_P1_only',received_input_path='P0',restored_input_path='P1',source_only=True,
        episode=source.EPISODE,receiver_seed=2027,second_RMS_applied=False,Us_labels_read=0)
    if (set(m)!=set(required)|{'code_commit','parent_F0_identity','dura_identity','source_contract_handle'} or any(m.get(k)!=v for k,v in required.items())
            or re.fullmatch('[0-9a-f]{40}',str(m.get('code_commit'))) is None or any(not isinstance(m.get(k),dict) or not m[k] for k in ('parent_F0_identity','dura_identity','source_contract_handle'))):raise ValueError('Validated source-only Us role manifest and frozen ancestors required')
    if m['source_contract_handle'].get('split_seed')!=392005 or m['source_contract_handle'].get('counts',{}).get('U_s')!=n:raise ValueError('Original source Us contract handle required')
    ids=np.load(root/'ids.npy',allow_pickle=False)
    with np.load(root/'metadata.npz',allow_pickle=False) as z:metadata={k:z[k].copy() for k in z.files}
    _public_metadata(ids,metadata);result=dict(manifest=m,ids=ids,metadata=metadata)
    try:
        for name in ('received','restored'):
            value=np.load(root/(name+'.npy'),mmap_mode='r',allow_pickle=False);result[name]=value
            if value.dtype!=np.float32 or value.shape!=(n,2,256):raise ValueError('Source Us public FP32 IQ shape differs')
    except Exception:close_public(result);raise
    return result


def close_public(bundle):
    for value in bundle.values():
        if isinstance(value,np.memmap) and not value._mmap.closed:value._mmap.close()


_close=close_public


def main():
    print(json.dumps(dict(event='SOURCE_US_CACHE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();build(_read(a.config),a.device)


if __name__=='__main__':main()
