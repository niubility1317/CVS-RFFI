"""Label-blind original L_s/U_s pairs matched to existing G10 received IQ."""
import json
from pathlib import Path
import numpy as np

SOURCE_FIELDS = {'reader_config', 'source_cache', 'source_contract', 'raw_mainsig', 'mixture_config_path',
                 'episode', 'dura_checkpoint', 'dura_release', 'L_received_root', 'Us_received_root'}
EPISODE = dict(id='MIX01', scene_seed=2026103001, channel_seed=2026104001)
COUNTS = {'L_s': 6300, 'U_s': 56700, 'V': 27000}
METADATA = ('ids', 'rx_i', 'day_i', 'eq_i', 'sig_i', 'base_index', 'domain')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_source(c):
    if not isinstance(c, dict) or set(c) != SOURCE_FIELDS or c.get('episode') != EPISODE:
        raise ValueError('Only original L_s/U_s and fixed source G10 MIX01 handles required')
    for key in SOURCE_FIELDS - {'reader_config', 'episode'}:
        if not isinstance(c[key], str) or not c[key].strip():
            raise ValueError('Nonempty original source handle required: '+key)
    from experiments.pamf_phase3_open_world.star_relative_source import _reader_config
    _reader_config(c)
    return c


def selected_indices(role):
    if role not in ('L_s', 'U_s'):
        raise ValueError('Restorer training diagnostics only allow L_s/U_s')
    return np.linspace(0, COUNTS[role]-1, 128 if role=='L_s' else 384, dtype=int)


def metadata(c, *, full=False):
    validate_source(c)
    from experiments.pamf_phase3_open_world.star_relative_source import _contract
    contract = _contract(c)
    if contract['counts'] != COUNTS:
        raise ValueError('Original source physical roles differ')
    parts = []
    for role in ('L_s', 'U_s'):
        with np.load(Path(c['source_cache'])/(role+'_metadata.npz'), allow_pickle=False) as f:
            # L_s may contain y, but it is never fetched or exposed here.
            if set(f.files) != set(METADATA) | ({'y'} if role=='L_s' else set()):
                raise ValueError('Original label-hidden source metadata differs')
            if f['ids'].tolist() != contract['role_ids'][role]:
                raise ValueError('Source role physical ID order differs')
            take = np.arange(COUNTS[role]) if full else selected_indices(role)
            part = {name: f[name][take].copy() for name in METADATA}
        parts.append(part)
    result = {key: np.concatenate([part[key] for part in parts]) for key in METADATA}
    if len(set(result['ids'].tolist())) != (63000 if full else 512):
        raise ValueError('Selected L_s/U_s physical records overlap')
    return contract, result


def open_received(c, role, contract):
    if role not in ('L_s', 'U_s'):
        raise ValueError('Only original L_s/U_s received handles allowed')
    folder = Path(c['L_received_root' if role=='L_s' else 'Us_received_root'])
    m = read(folder/'manifest.json')
    expected = dict(role='L' if role=='L_s' else 'U_s', count=COUNTS[role], precision='FP32', source_only=True, received_input_path='P0')
    if any(m.get(k)!=v for k,v in expected.items()):
        raise ValueError('Completed original source G10 received handle required')
    if role=='L_s':
        if m.get('status')!='STAR_REPRESENTATION_ROLE_COMPLETE' or m.get('code_commit')!='d319e3196f90113045a4f4f0958eeffe497f8785':
            raise ValueError('Original source-L MIX01 producer differs')
    else:
        if (m.get('status')!='SOURCE_US_G10_ROLE_COMPLETE' or m.get('data_status')!='VALIDATED_ONCE'
                or m.get('episode')!=EPISODE or m.get('second_RMS_applied') is not False
                or m.get('capsule_id')!='pamf-source-Us392005-G10-MIX01-20261031-r01'
                or m.get('split_id')!='392005-Us-source5rx-3day-eq1' or m.get('Us_labels_read')!=0):
            raise ValueError('Validated original source-Us MIX01 capsule/split required')
    if any(m.get('parent_F0_identity', {}).get(k)!=v for k,v in c['reader_config'].items() if k in ('source_output','base_release','model_seed')):
        raise ValueError('Source G10 producer used a different original F0 identity')
    ids = np.load(folder/'ids.npy', allow_pickle=False)
    if ids.tolist()!=contract['role_ids'][role]:
        raise ValueError('Cached G10 source IDs differ from the original role')
    x = np.load(folder/'received.npy', mmap_mode='r', allow_pickle=False)
    if x.dtype!=np.float32 or x.shape!=(COUNTS[role], 2, 256):
        x._mmap.close()
        raise ValueError('Original source received FP32 shape differs')
    return x, m


def _received(c, role, contract):
    x,m = open_received(c,role,contract)
    try:
        result=np.array(x[selected_indices(role)], copy=True)
    finally:
        x._mmap.close()
    if not np.isfinite(result).all():
        raise ValueError('Source received is nonfinite; do not reselect')
    return result, m


def prepare(c, smoke=None):
    """Prepare only 512 fixed source pairs; no G/F/label/target calls."""
    validate_source(c)
    # Resolve canonical G10 before the frozen F0 bootstrap changes namespaces.
    from experiments.pamf_phase3_open_world import star_channel as channel
    api, _ = channel._channel_api()
    mixture = channel.validate_mixture(read(c['mixture_config_path']))
    from experiments.pamf_phase3_open_world import source_us_cache as locator
    from experiments.pamf_phase3_open_world import star_relative_source as source
    contract, meta = metadata(c)
    left, lm = _received(c, 'L_s', contract); right, um = _received(c, 'U_s', contract)
    received = np.concatenate((left, right))
    payload = source._load_raw(c['raw_mainsig'])
    records, enumerated, rxnames, days = locator._locate(payload, contract, meta)
    clean = np.empty((512,2,256), np.float32); maximum_audit_difference = 0.
    for i, coordinates in enumerate(records):
        tx, rx, day, eq, sig = coordinates
        raw = source.raw_packet(payload['data'][tx][rx][day][eq][sig])
        if raw.dtype!=np.float64:
            raise ValueError('Original raw source packets required; normalized cache is not raw')
        raw32 = raw.astype(np.float32)
        z = raw32[0].astype(np.float64)+1j*raw32[1].astype(np.float64)
        if i<2:
            audited, info = channel.observe(raw, physical_id=str(meta['ids'][i]), receiver=rxnames[rx], day=days[day], episode=EPISODE, mixture=mixture)
            difference=float(np.max(np.abs(audited-received[i]))); maximum_audit_difference=max(maximum_audit_difference,difference)
            if not np.array_equal(audited,received[i]):
                raise ValueError('Existing source received does not reproduce canonical G10')
            normalized=z/info['input_rms']
        else:
            normalized,_=api.normalize_input(z)
        clean[i]=np.stack((normalized.real,normalized.imag)).astype(np.float32)
        if i==1 and smoke is not None:
            smoke(clean[:2].copy(), received[:2].copy())
    del payload, records
    return dict(clean=clean, received=received, ids=meta['ids'], rx=meta['rx_i'], day=meta['day_i'],
        roles=np.asarray(['L_s']*128+['U_s']*384), contract=contract,
        evidence=dict(status='ORIGINAL_LU_G10_SOURCE_PAIRS_READY', count=512, L_s=128, U_s=384,
            original_role_counts=COUNTS, source_ID_rule='rolewise deterministic linspace', received_views=1, episode=EPISODE,
            raw_ManySig_whole_file_read=True, raw_payload_includes_other_roles=True, only_selected_Ls_Us_values_indexed=True,
            builder_coordinate_enumeration=enumerated, new_canonical_observe_audit_calls=2,
            new_raw_RMS_calls=512, post_channel_RMS_calls=0, maximum_G10_audit_difference=maximum_audit_difference,
            no_legacy_normalized_source_IQ_opened=True, TX_labels_read=0, OE_reads=0, V_reads=0, target_inputs=0,
            source_L_producer=lm['run_id'], source_Us_producer=um['run_id'],
            F0_source_output=c['reader_config']['source_output']))
