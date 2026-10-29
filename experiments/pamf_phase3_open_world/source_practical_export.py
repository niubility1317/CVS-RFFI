"""Compact frozen E51 descriptors for four legal source roles and four views.

Clean input is already normalized by its validated source builder. The other
three computational views use this release's canonical residual channel and
the same frozen DURA v1. No source U, target, OE TX truth, fitting, second RMS,
or full IQ/branch-bank output belongs to this exporter.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from . import development_transfer as frozen, runner, source_open_export as original
from . import source_consensus as consensus, scores
from .full_pool import _channel_config, _apply_channel


RUN_ID = '20261029-phase3-source-practical-readout-s2026092701-r01'
COUNTS = dict(source_L=6300, source_V=27000, exposure_train=4200, novel_validation=3150)
ROLES = tuple(COUNTS)
VIEWS = ('clean', 'practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
BASELINE = dict(source_L=False, source_V=True, exposure_train=False, novel_validation=True)
SCHEMA = 'pamf_phase3_source_practical_export_v1'
FIELDS = {'run_id', 'code_commit', 'source_row_root', 'risk_prediction_root', 'oe_data_root',
          'dura_checkpoint', 'dura_release', 'output_root', 'batch_size'}
DATA_FIELDS = {'ids', 'descriptor', 'static_scores', 'logits', 'baseline_closed', 'rx_i', 'day_i'}
RX_NAMES = {1: '1-19', 3: '18-2', 4: '19-2', 6: '2-19', 8: '3-19'}
DAYS = {1: '2021_03_08', 2: '2021_03_15', 3: '2021_03_23'}
RECEIVER_SEED, CHANNEL_SEED = 2026102912, 2026102931


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_config(config):
    if not isinstance(config, dict) or set(config) != FIELDS:
        raise ValueError('Only the nine fixed source-practical export inputs are allowed')
    if any(not isinstance(config[key], str) or not config[key].strip() for key in FIELDS - {'batch_size'}):
        raise ValueError('Existing immutable source and frozen model paths are required')
    if (config['run_id'] != RUN_ID or not re.fullmatch('[0-9a-f]{40}', config['code_commit'])
            or type(config['batch_size']) is not int or config['batch_size'] != 128):
        raise ValueError('Fixed source-practical run, actual commit and batch128 are required')
    row, risk = Path(config['source_row_root']), Path(config['risk_prediction_root'])
    if (row.name != 'uniform_oe' or row.parent.name != 'rows' or row.parent.parent.name != frozen.READOUT_RUN
            or risk.name != 'predict' or risk.parent.name != 'uniform_oe' or risk.parent.parent.name != 'rows'
            or risk.parent.parent.parent.name != frozen.RISK_RUN or Path(config['oe_data_root']).name != 'source_oe'):
        raise ValueError('Only the fixed E51 uniform-OE, original risk and original source4 OE roots are allowed')
    output = Path(config['output_root']).resolve()
    if any(output.is_relative_to(Path(config[key]).resolve()) for key in
           ('source_row_root', 'risk_prediction_root', 'oe_data_root', 'dura_release')):
        raise ValueError('Export output cannot modify a frozen source/code input root')
    return config


def _canonical_channel_api():
    """Select this release's channel before historical F/G path activation."""
    code = Path(__file__).resolve().parents[2] / 'code'
    package = code / 'leo_practical'
    if not (package / 'channel.py').is_file() or not (package / 'residual.py').is_file():
        raise ValueError('This release must include its canonical code/leo_practical implementation')
    cached = {name: module for name, module in sys.modules.items()
              if name == 'leo_practical' or name.startswith('leo_practical.')}
    stale = False
    for module in cached.values():
        filename = getattr(module, '__file__', None)
        if not filename or not Path(filename).resolve().is_relative_to(package.resolve()):
            stale = True
            break
    if stale:
        # Ordinary bounded import reconciliation: no hash/receipt or separate
        # raw-data validation. Future relative imports stay on this package.
        for name in cached:
            sys.modules.pop(name, None)
    sys.path[:] = [str(code)] + [path for path in sys.path if path != str(code)]
    from leo_practical.channel import Config, receiver_for_session, stable_seed
    from leo_practical.residual import ResidualChannel
    return Config, receiver_for_session, stable_seed, ResidualChannel


def _source_contract(source):
    from .source_open_embedding_train import _contract
    return _contract(source['train_config'])


def _raw(values):
    import torch
    if torch.is_tensor(values):
        if values.dtype != torch.float32:
            raise ValueError('Source input must remain native FP32')
        raw = np.asarray(values.detach().cpu().tolist(), dtype=np.float32)
    else:
        raw = np.asarray(values)
        if raw.dtype != np.float32:
            raise ValueError('Source input must remain native FP32')
        raw = np.array(raw, copy=True)
    if raw.ndim != 3 or raw.shape[1:] != (2, 256) or not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Finite original FP32[B,2,256] source input required')
    return raw


def _synthesize(normalized, meta, scene, channels, api, sessions):
    """Same scene/opaque-ID seed formula for every source role and TX."""
    if scene not in VIEWS[1:]:
        raise ValueError('Only the three fixed residual source scenes are allowed')
    raw = _raw(normalized)
    n = len(raw)
    ids, rx, day = (np.asarray(meta[key]) for key in ('ids', 'rx_i', 'day_i'))
    if (ids.shape != (n,) or ids.dtype.kind != 'U' or np.any(ids == '')
            or rx.shape != (n,) or day.shape != (n,) or rx.dtype.kind not in 'iu' or day.dtype.kind not in 'iu'
            or not np.isin(rx, list(RX_NAMES)).all() or not np.isin(day, list(DAYS)).all()):
        raise ValueError('Aligned original source Main RX/day and opaque IDs are required')
    _, receiver_for_session, stable_seed, _ = api
    received = np.empty_like(raw)
    for index, (opaque, receiver_index, day_index) in enumerate(zip(ids, rx, day)):
        session = RX_NAMES[int(receiver_index)] + '/' + DAYS[int(day_index)] + '/1'
        if session not in sessions:
            sessions[session] = receiver_for_session(channels[VIEWS[1]], RECEIVER_SEED, session)
        seed = stable_seed(CHANNEL_SEED, scene, str(opaque))
        value, _ = _apply_channel(raw[index], channels[scene], seed, sessions[session])
        if value.dtype != np.float32 or value.shape != (2, 256) or not np.isfinite(value).all():
            raise ValueError('Canonical residual returned invalid FP32[2,256]')
        received[index] = value
    return received


def _encode_view(reader, baseline, engine, source, raw, meta, view, channels, api, sessions, device, need_baseline):
    import torch
    from .source_open_export import _tensor
    current = _raw(raw)
    if view != 'clean':
        current = _synthesize(current, meta, view, channels, api, sessions)
    x = _tensor(current, device)
    if view != 'clean':
        with torch.no_grad():
            x = engine(x)[0].detach().clone()
    if x.dtype != torch.float32 or tuple(x.shape) != (len(current), 2, 256) or not bool(torch.isfinite(x).all()):
        raise ValueError('Frozen source view must retain finite FP32 full256 IQ')
    data = frozen._encode_new(reader, x)
    branches = {name: data['features__' + name] for name in original.DIMENSIONS}
    descriptor = consensus.descriptor(branches, data['logits'], source['geometry'])
    static = scores.knownness_scores(branches['fused160'], data['logits'], source['geometry']['branches']['fused160']['prototypes'])
    closed = frozen._original_closed(baseline, x) if need_baseline else np.full(len(x), -2, np.int64)
    return dict(descriptor=descriptor, static_scores=static, logits=data['logits'], baseline_closed=closed)


def _check_frozen(models, snapshots, source, array_snapshot):
    for model, before in zip(models, snapshots):
        frozen._check_model(model, before)
    current = frozen._array_snapshot({key: source[key] for key in ('geometry', 'state', 'thresholds')})
    if set(current) != set(array_snapshot) or any(not np.array_equal(value, current[key]) for key, value in array_snapshot.items()):
        raise ValueError('Original clean-L geometry or frozen source readout/scalars changed')


def _smoke(reader, baseline, engine, source, channels, api, sessions, device):
    n = np.arange(256)
    raw = np.asarray([[np.cos(.11*n), np.sin(.11*n)],
                      [np.cos(.19*n+.3), np.sin(.19*n+.3)]], np.float32)
    meta = dict(ids=np.asarray(['source-practical-synthetic-0', 'source-practical-synthetic-1']),
        rx_i=np.asarray([1, 3], np.int64), day_i=np.asarray([1, 2], np.int64))
    for view in VIEWS:
        values = _encode_view(reader, baseline, engine, source, raw, meta, view, channels, api, sessions, device, True)
        if (values['descriptor'].shape != (2, 100) or values['descriptor'].dtype != np.float32
                or values['static_scores'].shape != (2, 5) or values['static_scores'].dtype != np.float64
                or values['logits'].shape != (2, 6) or values['logits'].dtype != np.float32
                or any(not np.isfinite(value).all() for value in values.values())):
            raise ValueError('Actual-checkpoint synthetic source-view smoke failed')
    result = dict(status='PASS', real_checkpoint=True, synthetic_physical_rows=2, view_order=list(VIEWS),
        synthetic_view_rows=8, source_IQ_reads=0, target_reads=0, optimizer_updates=0,
        encoder_forward_rows=8, original_F_forward_rows=8, DURA_forward_rows=6, channel_rows=6,
        descriptor_dimension=100, default_head_classes=6)
    print('SOURCE_PRACTICAL_SYNTHETIC_SMOKE_PASS', json.dumps(result), flush=True)
    return result


def _load_role(config, source, contract, role):
    if role in ('source_L', 'source_V'):
        native_role = 'L_s' if role == 'source_L' else 'V'
        dataset = runner._source_dataset(source['train_config'], native_role, contract)
        return dataset.x, original._source_meta(dataset, native_role, contract)
    if role in ('exposure_train', 'novel_validation'):
        return original._oe_public(config['oe_data_root'], role)
    raise ValueError('Only four fixed legal source roles may be opened')


def _refs():
    return {role: {view: role + '/' + view + '/data.npz' for view in VIEWS} for role in ROLES}


def _arrays(values, role, count, *, include_y=True):
    expected = DATA_FIELDS | ({'y'} if include_y and role in ('source_L', 'source_V') else set())
    if set(values) != expected:
        raise ValueError('Compact descriptor fields or source label permissions differ')
    ids = values['ids']
    if ids.shape != (count,) or ids.dtype.kind != 'U' or np.any(ids == '') or len(set(ids.tolist())) != count:
        raise ValueError('Unique aligned original physical IDs are required within each view')
    for key, shape, dtype in [('descriptor', (count, 100), np.float32), ('static_scores', (count, 5), np.float64),
            ('logits', (count, 6), np.float32), ('baseline_closed', (count,), np.int64),
            ('rx_i', (count,), np.int64), ('day_i', (count,), np.int64)]:
        value = values[key]
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError('Compact descriptor dtype, shape or finite values differ: ' + key)
    if not np.isin(values['rx_i'], list(RX_NAMES)).all() or not np.isin(values['day_i'], list(DAYS)).all():
        raise ValueError('Only the original five source RX/three days are allowed')
    baseline = values['baseline_closed']
    if (BASELINE[role] and bool(((baseline < 0) | (baseline >= 6)).any())) or (not BASELINE[role] and not bool((baseline == -2).all())):
        raise ValueError('Same-input original-F baseline availability/sentinel differs')
    if 'y' in values and (values['y'].dtype != np.int64 or values['y'].shape != (count,)
                          or set(values['y'].tolist()) != set(range(6))):
        raise ValueError('Original known-source labels must retain all six old classes')


def export(config, device='cuda:0'):
    validate_config(config)
    output = Path(config['output_root'])
    if output.exists():
        raise FileExistsError('Source practical export output cannot be overwritten')
    # Bind the current channel before any immutable F/G bootstrap alters paths.
    api = _canonical_channel_api()
    channels = {scene: _channel_config(scene) for scene in VIEWS[1:]}
    source = frozen._source_bundle(config)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reader, baseline = frozen._load_readers(source, device)
    dura, engine = frozen._load_dura(source, config, device)
    contract = _source_contract(source)
    handles = original._oe_handles(config['oe_data_root'])
    if handles != source['marker']['source_OE_handles']:
        raise ValueError('The original source4 OE/novel3 handles must match the E51 readout source')
    models = [reader, baseline, dura]
    snapshots = [frozen._snapshot_model(model) for model in models]
    arrays_before = frozen._array_snapshot({key: source[key] for key in ('geometry', 'state', 'thresholds')})
    sessions = {}
    smoke = _smoke(reader, baseline, engine, source, channels, api, sessions, device)
    _check_frozen(models, snapshots, source, arrays_before)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'smoke.json', smoke)
    seen = set()
    refs = _refs()
    for role in ROLES:
        raw, meta = _load_role(config, source, contract, role)
        try:
            count = COUNTS[role]
            ids = np.asarray(meta['ids'])
            if ids.shape != (count,) or len(raw) != count or len(set(ids.tolist())) != count or seen & set(ids.tolist()):
                raise ValueError('Original source physical role IDs/counts must be disjoint')
            seen.update(ids.tolist())
            for view in VIEWS:
                values = dict(ids=ids.copy(), rx_i=np.asarray(meta['rx_i'], dtype=np.int64).copy(),
                    day_i=np.asarray(meta['day_i'], dtype=np.int64).copy(),
                    descriptor=np.empty((count, 100), np.float32), static_scores=np.empty((count, 5), np.float64),
                    logits=np.empty((count, 6), np.float32), baseline_closed=np.full(count, -2, np.int64))
                if role in ('source_L', 'source_V'):
                    values['y'] = np.asarray(meta['y'], dtype=np.int64).copy()
                for start in range(0, count, config['batch_size']):
                    stop = min(start + config['batch_size'], count)
                    current_meta = {key: np.asarray(meta[key])[start:stop] for key in ('ids', 'rx_i', 'day_i')}
                    encoded = _encode_view(reader, baseline, engine, source, raw[start:stop], current_meta,
                        view, channels, api, sessions, device, BASELINE[role])
                    for key, value in encoded.items():
                        values[key][start:stop] = value
                    if start == 0 or start % (config['batch_size'] * 32) == 0 or stop == count:
                        print('SOURCE_PRACTICAL_EXPORT', role, view, stop, count, flush=True)
                _arrays(values, role, count)
                path = output / refs[role][view]
                path.parent.mkdir(parents=True, exist_ok=False)
                with path.open('xb') as stream:
                    np.savez(stream, **values)
        finally:
            if isinstance(raw, np.memmap):
                raw._mmap.close()
    _check_frozen(models, snapshots, source, arrays_before)
    physical = sum(COUNTS.values())
    result = dict(status='SOURCE_PRACTICAL_DESCRIPTORS_EXPORTED', schema=SCHEMA,
        run_id=config['run_id'], code_commit=config['code_commit'], source_only=True,
        counts=dict(COUNTS), view_order=list(VIEWS), data_refs=refs, descriptor_schema=consensus.DESCRIPTOR_SCHEMA,
        encoder_state_identity=source['marker']['encoder_state_identity'],
        parent_export_identity=source['marker']['parent_export_identity'], source_OE_handles=handles,
        geometry_source_fit_root=str(Path(config['source_row_root']) / 'fit'),
        geometry_policy='Unchanged original E51 clean-L-only geometry; no refit on computational views',
        dura_identity=dict(deployment_id='dura_v1', checkpoint=config['dura_checkpoint'], release_root=config['dura_release'],
            arm='stage2', epoch=20, updates=4940, training_commit='b2a2ce937a64654e17e84c2aa817b9da4257a48f',
            parameters=266951, reuse=True, graph=True, native=False, precision='FP32', TF32=False),
        physical_record_count=physical, derived_view_record_count=physical * len(VIEWS),
        new_independent_physical_records=0, same_physical_ID_order_across_views=True,
        baseline_available=BASELINE.copy(), baseline_missing_sentinel=-2,
        source_roles_label_permissions=dict(source_L='known y', source_V='known y; calibration/evaluation only',
            exposure_train='membership only; no TX truth', novel_validation='no TX truth; independent scorer only'),
        source_encoder_forward_rows=physical * len(VIEWS),
        original_F_forward_rows=sum(COUNTS[role] for role in ROLES if BASELINE[role]) * len(VIEWS),
        DURA_forward_rows=physical * 3, channel_rows=physical * 3, smoke=smoke,
        original_U_s_reads=0, target_reads=0, OE_truth_reads=0, optimizer_updates=0, identity_head_updates=0,
        new_encoder_updates=0, new_readout_updates=0, new_threshold_calibrations=0, default_head_replaced=False,
        source_state_unchanged=True, raw_branches_written=False, IQ_written=False, second_RMS_applied=False,
        original_EQ_reapplied=False, canonical_channel_code=str(Path(__file__).resolve().parents[2] / 'code/leo_practical'),
        channel_configs={scene: asdict(value) for scene, value in channels.items()},
        seeds=dict(receiver=RECEIVER_SEED, source_channel=CHANNEL_SEED),
        channel_seed_rule='stable_seed(source_channel_seed, scene, opaqueID); identical across all roles',
        receiver_session_rule='physical source RX/date/1; first-scene receiver reused across three scenes',
        validation_reused='Existing VALIDATED_ONCE source role handles; no Full/ZIP/data revalidation',
        interpretation='Source computational augmentation views; not new independent physical data or target evidence')
    _write(output / 'complete.json', result)
    return result


def export_marker(root):
    try:
        marker = _read(Path(root) / 'complete.json')
    except (OSError, ValueError) as exc:
        raise ValueError('Complete immutable four-role/four-view source export required') from exc
    if (marker.get('status') != 'SOURCE_PRACTICAL_DESCRIPTORS_EXPORTED' or marker.get('schema') != SCHEMA
            or marker.get('counts') != COUNTS or marker.get('view_order') != list(VIEWS)
            or marker.get('data_refs') != _refs() or marker.get('source_only') is not True
            or marker.get('descriptor_schema') != consensus.DESCRIPTOR_SCHEMA
            or marker.get('physical_record_count') != sum(COUNTS.values())
            or marker.get('derived_view_record_count') != sum(COUNTS.values()) * 4
            or marker.get('same_physical_ID_order_across_views') is not True
            or marker.get('source_state_unchanged') is not True
            or marker.get('baseline_available') != BASELINE or marker.get('baseline_missing_sentinel') != -2
            or any(marker.get(key) != 0 for key in ('target_reads', 'original_U_s_reads', 'OE_truth_reads',
                'optimizer_updates', 'identity_head_updates', 'new_encoder_updates', 'new_readout_updates', 'new_threshold_calibrations'))):
        raise ValueError('The source practical export identity, permissions or full view matrix differs')
    return marker


def load_role_view(root, role, view, include_y=False):
    """Read one compact source view; never permit unknown-role TX labels."""
    if (role not in ROLES or view not in VIEWS or type(include_y) is not bool
            or (include_y and role not in ('source_L', 'source_V'))):
        raise ValueError('Only declared source views and legal known-source y can be read')
    marker = export_marker(root)
    with np.load(Path(root) / marker['data_refs'][role][view], allow_pickle=False) as stored:
        expected = DATA_FIELDS | ({'y'} if role in ('source_L', 'source_V') else set())
        if set(stored.files) != expected:
            raise ValueError('Compact source view includes missing or forbidden fields')
        # V y is only materialized for its explicitly requested calibration.
        wanted = expected if include_y else DATA_FIELDS
        values = {key: stored[key].copy() for key in wanted}
        _arrays(values, role, COUNTS[role], include_y=include_y)
    return values


def main(argv=None):
    print(json.dumps(dict(event='SOURCE_PRACTICAL_EXPORT_PROCESS', pid=os.getpid(), cwd=os.getcwd(),
        argv=list(sys.argv) if argv is None else list(argv)), ensure_ascii=False), flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    result = export(_read(args.config), args.device)
    print(json.dumps(dict(status=result['status'], physical_record_count=result['physical_record_count'],
                          derived_view_record_count=result['derived_view_record_count'])), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
