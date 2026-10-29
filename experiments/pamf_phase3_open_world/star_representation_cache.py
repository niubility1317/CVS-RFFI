"""Frozen F0/DURA computation cache for four legal source-only G10 roles.

L/V use their exact original ManySig metadata to reconstruct the fixed MIX01
received observation. OE/novel copy already-built G10 public inputs only.
No source clean cache, U_s role, private truth or target input is consumed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import star_relative_source as source
from .source_open_export import _tensor


RUN_ID = '20261029-phase3-star-representation-s2026092701-r01'
SCHEMA = 'pamf_phase3_star_representation_cache_v1'
ROLE_SCHEMA = 'pamf_phase3_star_representation_cache_role_v1'
COUNTS = dict(L=6300, OE=6134, V=27000, novel=3150)
FIELDS = {'source_config', 'exposure_public_root', 'exposure_cap', 'exposure_split',
          'novel_public_root', 'novel_cap', 'novel_split', 'output_root', 'batch_size'}
METADATA_FIELDS = dict(L={'y', 'rx_i', 'day_i', 'domain'}, OE={'rx_i', 'day_i', 'domain'},
    V={'y', 'rx_i', 'day_i', 'domain', 'scenario'}, novel={'rx_i', 'day_i', 'domain'})
ZERO_COUNTS = ('original_U_s_reads', 'target_reads', 'formal_target_reads', 'private_truth_reads',
    'old_normalized_source_IQ_reads', 'optimizer_updates', 'encoder_updates', 'identity_head_updates',
    'DURA_updates', 'threshold_updates')
_read, _write = source._read, source._write
_backend, _snapshot, _check_models = source._backend, source._snapshot, source._check_models
_source_metadata, _load_raw, _observations = source.source_metadata, source._load_raw, source._observations


def validate_config(config):
    if (not isinstance(config, dict) or set(config) != FIELDS
            or any(not isinstance(config[k], str) or not config[k].strip() for k in FIELDS-{'source_config', 'batch_size'})
            or type(config['batch_size']) is not int or config['batch_size'] != 128
            or not isinstance(config['source_config'], dict)):
        raise ValueError('Only the nine declared source-G10 public cache inputs and batch128 are allowed')
    nested = config['source_config']
    if nested.get('run_id') != RUN_ID:
        raise ValueError('The fixed star-representation source run is required')
    source.validate_config(dict(nested, run_id=source.RUN_ID))
    if (nested['batch_size'] != config['batch_size']
            or Path(nested['output_root']).resolve() != Path(config['output_root']).resolve()):
        raise ValueError('Nested source and cache output/batch configuration must agree')
    return config


def role_files(role):
    if role not in COUNTS:
        raise ValueError('Only L/OE/V/novel source roles are allowed')
    files = {'manifest.json', 'ids.npy', 'received.npy', 'restored.npy', 'metadata.npz'}
    return files | ({'teacher_logits.npy'} if role == 'L' else {'baseline_closed.npy'} if role == 'V' else set())


def _metadata(role, ids, metadata):
    n = COUNTS[role]
    if (ids.shape != (n,) or ids.dtype.kind != 'U' or len(set(ids.tolist())) != n
            or np.any(np.char.strip(ids) == '') or set(metadata) != METADATA_FIELDS[role]):
        raise ValueError('Aligned original IDs and exact role-specific source metadata are required')
    for name, values in metadata.items():
        if values.shape != (n,) or (values.dtype.kind != 'U' if name == 'scenario' else values.dtype != np.int64):
            raise ValueError('Source metadata shape/dtype differs: '+name)
    rx, day, domain = (metadata[k] for k in ('rx_i', 'day_i', 'domain'))
    if (not np.isin(rx, source.RECEIVERS).all() or not np.isin(day, [1, 2, 3]).all()
            or not np.array_equal(domain, np.searchsorted(source.RECEIVERS, rx)*3+day-1)
            or set(np.unique(domain).tolist()) != set(range(15))):
        raise ValueError('The original fifteen source RX/day domains are required')
    if role in ('L', 'V') and set(metadata['y'].tolist()) != set(range(6)):
        raise ValueError('L/V must retain the original six known labels')
    if role == 'V' and not np.isin(metadata['scenario'], source.SCENES).all():
        raise ValueError('V scenarios must describe the fixed received G10 observations')


def _public(role, config):
    if role == 'OE':
        from .star_source_exposure import load_public
        return load_public(config['exposure_public_root'], config['exposure_cap'], config['exposure_split'])
    if role == 'novel':
        from .star_source_novel import load_public
        return load_public(config['novel_public_root'], config['novel_cap'], config['novel_split'])
    raise ValueError('Only prebuilt G10 OE/novel public inputs may be opened here')


def _array(value, shape, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError('Finite FP32 source '+name+' with the exact batch shape is required')
    return value


def _restore(backend, values, device):
    import torch
    with torch.no_grad():
        x = backend['dura'][1](_tensor(values, device))[0].detach().clone()
        if x.dtype != torch.float32 or tuple(x.shape) != values.shape or not bool(torch.isfinite(x).all()):
            raise ValueError('Frozen DURA must preserve finite FP32 source IQ')
        return np.asarray(x.cpu().tolist(), np.float32)


def _logits(backend, values, device):
    import torch
    with torch.no_grad():
        reader = backend['reader']; x = _tensor(values, device)
        logits = reader.classify_features(reader.features(x))
        if logits.dtype != torch.float32 or tuple(logits.shape) != (len(values), 6) or not bool(torch.isfinite(logits).all()):
            raise ValueError('The original complete six-class F0 head is required')
        return np.asarray(logits.detach().cpu().tolist(), np.float32)


def _smoke(backend, config, device):
    from .star_channel import observe
    mixture = _read(config['mixture_config_path']); n = np.arange(256)
    episode = {k:v for k,v in config['episode'].items() if k != 'condition'}
    values = np.stack([observe(np.stack((np.cos(w*n), np.sin(w*n))),
        physical_id='star-representation-synthetic-'+str(i), receiver='1-19', day='2021_03_08',
        episode=episode, mixture=mixture)[0] for i,w in enumerate((.07,.11))])
    restored = _array(_restore(backend, values, device), (2,2,256), 'smoke restoration')
    for current in (values, restored):
        _array(_logits(backend, current, device), (2,6), 'smoke logits')
    print('STAR_REPRESENTATION_CHECKPOINT_SMOKE_PASS', 2, flush=True)
    return dict(status='PASS', real_checkpoint=True, synthetic_physical_rows=2, synthetic_G_rows=2,
        synthetic_F0_rows=4, source_IQ_reads=0, target_reads=0, private_truth_reads=0)


def _save(path, value):
    with path.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)


def _save_role(output, role, values, ids, metadata, backend, config, device, snapshots):
    _metadata(role, ids, metadata)
    n = COUNTS[role]
    if values.shape != (n,2,256) or values.dtype != np.float32:
        raise ValueError('Received source rows must retain their original role count/order')
    folder = output/role; folder.mkdir(exist_ok=False)
    maps = {name: np.lib.format.open_memmap(folder/(name+'.npy'), mode='w+', dtype=np.float32, shape=(n,2,256))
            for name in ('received','restored')}
    if role == 'L':
        maps['teacher_logits'] = np.lib.format.open_memmap(folder/'teacher_logits.npy', mode='w+', dtype=np.float32, shape=(n,6))
    if role == 'V':
        maps['baseline_closed'] = np.lib.format.open_memmap(folder/'baseline_closed.npy', mode='w+', dtype=np.int64, shape=(n,2))
    f_rows = 0
    try:
        for start in range(0,n,config['batch_size']):
            stop = min(start+config['batch_size'],n); shape=(stop-start,2,256)
            current = _array(values[start:stop], shape, 'received')
            restored = _array(_restore(backend,current,device), shape, 'restored')
            maps['received'][start:stop] = current; maps['restored'][start:stop] = restored
            if role == 'L':
                maps['teacher_logits'][start:stop] = _array(_logits(backend,restored,device),(stop-start,6),'teacher logits')
                f_rows += stop-start
            elif role == 'V':
                for j,path in enumerate((current,restored)):
                    maps['baseline_closed'][start:stop,j] = _array(_logits(backend,path,device),(stop-start,6),'baseline logits').argmax(1)
                    f_rows += stop-start
            if start == 0 or stop == n or start % (128*32) == 0:
                print('STAR_REPRESENTATION_CACHE_PROGRESS',role,stop,n,flush=True)
    finally:
        for mapped in maps.values():
            mapped.flush(); mapped._mmap.close()
    _check_models(snapshots)
    _save(folder/'ids.npy',ids)
    with (folder/'metadata.npz').open('xb') as stream:
        np.savez(stream,**metadata)
    manifest = dict(status='STAR_REPRESENTATION_ROLE_COMPLETE',schema=ROLE_SCHEMA,run_id=RUN_ID,
        code_commit=config['source_config']['code_commit'],role=role,count=n,iq_shape=[n,2,256],precision='FP32',
        paths=['P0','P1'],received_input_path='P0',restored_input_path='P1',
        teacher_input_path='P1' if role=='L' else None,baseline_input_paths=['P0','P1'] if role=='V' else None,
        files=sorted(role_files(role)),metadata_fields=sorted(METADATA_FIELDS[role]),source_only=True,
        all_new_IQ_satellite=True,parent_F0_identity=backend['reader_identity'],dura_identity=backend['dura_identity'])
    _write(folder/'manifest.json',manifest)
    return manifest, f_rows


def build(config, device='cuda:0'):
    validate_config(config); output=Path(config['output_root']); nested=config['source_config']
    if output.exists():
        raise FileExistsError('A new immutable source representation cache root is required')
    # Activate the current release's canonical channel before legacy F/G sys.path changes.
    from .star_channel import _channel_api, validate_mixture
    _channel_api(); mixture=validate_mixture(_read(nested['mixture_config_path']))
    backend=_backend(nested,device,with_dura=True); snapshots=_snapshot(backend)
    smoke=_smoke(backend,nested,device); _check_models(snapshots)
    # No source IQ opens until actual-checkpoint synthetic smoke passes.
    contract, native_metadata=_source_metadata(nested)
    payload=_load_raw(nested['raw_mainsig'])
    output.mkdir(parents=True,exist_ok=False)
    manifests, seen, forward = {}, set(), dict(DURA=0,F0=0)
    for role,native in (('L','L_s'),('V','V')):
        original=native_metadata[native]
        values,scenes=_observations(payload,original,contract,nested,mixture)
        ids=original['ids']; metadata={k:original[k] for k in METADATA_FIELDS[role]-{'scenario'}}
        if role=='V':metadata['scenario']=scenes
        if seen & set(ids.tolist()):raise ValueError('Original source physical role IDs must remain disjoint')
        seen.update(ids.tolist())
        manifests[role],n_f=_save_role(output,role,values,ids,metadata,backend,config,device,snapshots)
        forward['DURA']+=len(ids);forward['F0']+=n_f
        del values
    del payload, native_metadata
    for role in ('OE','novel'):
        parent,ids,values,metadata=_public(role,config)
        try:
            if seen & set(ids.tolist()):raise ValueError('Original source physical role IDs must remain disjoint')
            seen.update(ids.tolist())
            manifests[role],n_f=_save_role(output,role,values,ids,metadata,backend,config,device,snapshots)
            forward['DURA']+=len(ids);forward['F0']+=n_f
        finally:
            if isinstance(values,np.memmap):values._mmap.close()
    _check_models(snapshots)
    if forward != dict(DURA=sum(COUNTS.values()),F0=COUNTS['L']+2*COUNTS['V']):
        raise ValueError('Actual frozen model forward counts differ from the cache plan')
    complete=dict(status='STAR_REPRESENTATION_CACHE_COMPLETE',schema=SCHEMA,run_id=RUN_ID,
        code_commit=nested['code_commit'],config=config,counts=dict(COUNTS),role_order=list(COUNTS),
        roles={role:dict(root=str(output/role),count=COUNTS[role],schema=ROLE_SCHEMA) for role in COUNTS},
        cache_model_forward_rows=forward,smoke=smoke,models_unchanged=True,source_only=True,
        parent_F0_identity=backend['reader_identity'],dura_identity=backend['dura_identity'],
        original_source_contract_handle={k:contract[k] for k in ('classes','source_rxs','source_days','split_seed','counts') if k in contract},
        public_handles=dict(OE=dict(capsule_id=config['exposure_cap'],split_id=config['exposure_split']),
                            novel=dict(capsule_id=config['novel_cap'],split_id=config['novel_split'])),
        physical_record_count=sum(COUNTS.values()),derived_view_record_count=2*sum(COUNTS.values()),
        new_independent_physical_records=0,channel_reconstructions=COUNTS['L']+COUNTS['V'],
        fixed_G10_reconstruction=True,all_new_IQ_satellite=True,episode=nested['episode'],receiver_seed=2027,
        raw_ManySig_whole_file_read=True,original_U_s_role_records_observed=0,precision='FP32',TF32=False,
        same_physical_ID_order_across_paths=True,second_RMS_applied=False,
        interpretation='Fixed computational received/restored views of original source records; not new physical observations',
        **{key:0 for key in ZERO_COUNTS})
    _write(output/'complete.json',complete)
    return complete


def load_role(root, expected_role):
    """Open exactly one role directory; never inspect its parent or siblings."""
    folder=Path(root); files=role_files(expected_role)
    if {p.name for p in folder.iterdir()} != files:
        raise ValueError('Source role file whitelist differs or the role is incomplete')
    manifest=_read(folder/'manifest.json');n=COUNTS[expected_role]
    required=dict(status='STAR_REPRESENTATION_ROLE_COMPLETE',schema=ROLE_SCHEMA,run_id=RUN_ID,role=expected_role,
        count=n,iq_shape=[n,2,256],precision='FP32',paths=['P0','P1'],received_input_path='P0',restored_input_path='P1',
        teacher_input_path='P1' if expected_role=='L' else None,
        baseline_input_paths=['P0','P1'] if expected_role=='V' else None,
        files=sorted(files),metadata_fields=sorted(METADATA_FIELDS[expected_role]),source_only=True,all_new_IQ_satellite=True)
    if (not isinstance(manifest,dict) or set(manifest) != set(required)|{'code_commit','parent_F0_identity','dura_identity'}
            or any(manifest.get(k)!=v for k,v in required.items())
            or any(not isinstance(manifest.get(k),dict) or not manifest[k] for k in ('parent_F0_identity','dura_identity'))):
        raise ValueError('Complete source role manifest/identity differs')
    import re
    if not isinstance(manifest['code_commit'],str) or not re.fullmatch('[0-9a-f]{40}',manifest['code_commit']):
        raise ValueError('Source role actual code identity required')
    ids=np.load(folder/'ids.npy',allow_pickle=False)
    with np.load(folder/'metadata.npz',allow_pickle=False) as stored:metadata={k:stored[k] for k in stored.files}
    _metadata(expected_role,ids,metadata)
    result=dict(manifest=manifest,ids=ids,metadata=metadata)
    specifications={'received':(np.float32,(n,2,256)),'restored':(np.float32,(n,2,256))}
    if expected_role=='L':specifications['teacher_logits']=(np.float32,(n,6))
    if expected_role=='V':specifications['baseline_closed']=(np.int64,(n,2))
    try:
        for name,(dtype,shape) in specifications.items():
            value=np.load(folder/(name+'.npy'),mmap_mode='r',allow_pickle=False);result[name]=value
            if not isinstance(value,np.memmap) or value.dtype!=dtype or value.shape!=shape:
                raise ValueError('Source role NPY dtype/shape differs: '+name)
        if expected_role=='L' and not np.isfinite(result['teacher_logits']).all():raise ValueError('Finite teacher logits required')
        if expected_role=='V' and not np.isin(result['baseline_closed'],range(6)).all():raise ValueError('Six-class baseline codes required')
    except Exception:
        close_role(result);raise
    return result


def close_role(bundle):
    for value in bundle.values():
        if isinstance(value,np.memmap) and not value._mmap.closed:value._mmap.close()


def load_L(root):return load_role(root,'L')


def load_OE(root):return load_role(root,'OE')


def main(argv=None):
    actual=list(sys.argv[1:] if argv is None else argv)
    print(json.dumps(dict(pid=os.getpid(),cwd=os.getcwd(),argv=actual),ensure_ascii=False),flush=True)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args(actual);done=build(_read(args.config),args.device)
    print(json.dumps(dict(status=done['status'],counts=done['counts'])));return 0


if __name__=='__main__':raise SystemExit(main())
