"""Export frozen native branches from registered support rows only; never fit.

The existing NPZ must store iq.npy without compression. Its NPY payload is
mapped read-only, and only registered support indices are gathered. No IQ hash,
full IQ load, query inference, source loader, or adaptation state is accepted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
from types import SimpleNamespace
import zipfile

import numpy as np
from cvs_native_artifacts import verify_source

CACHE_SCHEMA = 'd92_branch_support_features_v1'
CACHE_NAME = 'support_branch_features.npz'
BRANCHES = ('z_id', 't_emb', 'f_emb', 'pa_local')
FEATURE_CONTRACT = dict(input_shape=[2, 256], view='original_received_observation', view_count=1,
    identity_feature_key='feat_joint', branch_keys=list(BRANCHES), branch_dim=160, fft_dim=96,
    fft='historical_spectral_logmag_sketch', fft_norm_floor=1e-8, cache_dtype='float32',
    normalization='raw_native_aux_no_additional_branch_normalization')


def check(condition, message):
    if not condition: raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): result.update(block)
    return result.hexdigest()


def peak_process_rss():
    if sys.platform.startswith('linux'):
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    return None


def strings(values):
    return isinstance(values, list) and bool(values) and all(isinstance(v, str) and v for v in values) and len(set(values)) == len(values)


class SupportIQ:
    """Lazy ZIP_STORED NPY reader; returned arrays contain support rows only."""
    def __init__(self, path):
        self.path = Path(path)
        with zipfile.ZipFile(self.path) as archive:
            members = archive.infolist()
            check(len(members) == 2 and {m.filename for m in members} == {'iq.npy', 'ids.npy'}, 'Unexpected received NPZ members')
            item = archive.getinfo('iq.npy')
            check(item.compress_type == zipfile.ZIP_STORED and not item.flag_bits & 1
                  and item.compress_size == item.file_size, 'IQ must be unencrypted ZIP_STORED for support-only mmap')
            # This member contains identifiers only, never waveform data.
            with archive.open('ids.npy') as stream:
                ids = np.lib.format.read_array(stream, allow_pickle=False)
            check(ids.ndim == 1 and ids.dtype.kind in 'SU', 'Physical ID schema mismatch')
            self.ids = ids.astype(str)
            check(strings(self.ids.tolist()), 'Physical IDs must be unique and nonempty')
            with self.path.open('rb', buffering=0) as stream:
                stream.seek(item.header_offset)
                header = stream.read(30)
                check(len(header) == 30 and header[:4] == b'PK\x03\x04', 'Invalid local ZIP header')
                fields = struct.unpack('<4s5H3I2H', header)
                check(fields[3] == zipfile.ZIP_STORED and not fields[2] & 1, 'Local ZIP compression mismatch')
                filename_length, extra_length = fields[-2:]
                check(stream.read(filename_length) == b'iq.npy', 'Local IQ member mismatch')
                stream.seek(extra_length, 1)
                payload_start = stream.tell()
                version = np.lib.format.read_magic(stream)
                check(version in ((1, 0), (2, 0)), 'Unsupported NPY header version')
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = reader(stream)
                offset = stream.tell()
                check(shape == (len(self.ids), 2, 256) and not fortran and dtype == np.dtype('<f4'), 'Received IQ must be C-order float32 (N,2,256)')
                check(offset - payload_start + int(np.prod(shape))*dtype.itemsize == item.file_size,
                      'NPY payload size mismatch')
        self._array = np.memmap(self.path, dtype=dtype, mode='r', offset=offset, shape=shape, order='C')
        self.allowed = frozenset()
        self.rows_read = 0

    def allow(self, indices):
        check(not self.allowed and len(indices) == len(set(indices)), 'Support access plan already set or duplicated')
        self.allowed = frozenset(int(i) for i in indices)

    def take(self, indices):
        check(all(type(i) in (int, np.int64) and int(i) in self.allowed for i in indices), 'Attempted non-support IQ access')
        value = np.array(self._array[np.asarray(indices, dtype=np.int64)], dtype=np.float32, copy=True)
        check(np.isfinite(value).all(), 'Nonfinite support IQ')
        self.rows_read += len(indices)
        return value

    def close(self):
        self._array._mmap.close()


def support_plan(capsule, ids, expected_capsule_id, source_receivers):
    capsule = Path(capsule); manifest = read(capsule/'manifest.json')
    check(strings(source_receivers), 'Preregistered physical source receivers required')
    check(manifest.get('protocol_schema') == 'p2_min_v1' and manifest.get('phase2_data_status') == 'VALIDATED_ONCE'
          and manifest.get('capsule_id') == expected_capsule_id and manifest.get('signal_shape') == [2, 256]
          and manifest.get('received_count') == len(ids), 'Capsule binding/schema mismatch')
    paths = sorted((capsule/'splits').glob('*.json'))
    check(paths and len(paths) == manifest.get('split_count'), 'Incomplete registered split matrix')
    allowed = {'protocol_schema', 'phase2_data_status', 'capsule_id', 'split_id', 'receiver', 'scenario',
               'k', 'registered_classes', 'support_indices', 'support_labels', 'query_indices', 'support_seed'}
    labels_by_index, query_union, seen_splits, splits = {}, set(), set(), []
    for path in paths:
        split = read(path)
        check(set(split) == allowed, 'Forbidden or missing split member')
        check(split['protocol_schema'] == 'p2_min_v1' and split['phase2_data_status'] == 'VALIDATED_ONCE'
              and split['capsule_id'] == expected_capsule_id and split['split_id'] == path.stem
              and split['split_id'] not in seen_splits, 'Split identity mismatch')
        seen_splits.add(split['split_id'])
        classes, support, query, labels = (split[k] for k in ('registered_classes', 'support_indices', 'query_indices', 'support_labels'))
        check(strings(classes) and isinstance(support, list) and isinstance(query, list) and support and query
              and all(type(i) is int and 0 <= i < len(ids) for i in support+query)
              and len(set(support)) == len(support) and len(set(query)) == len(query)
              and not set(support) & set(query), 'Invalid physical support/query index metadata')
        check(type(split['k']) is int and split['k'] > 0 and isinstance(labels, list) and len(labels) == len(support)
              and all(type(y) is int and 0 <= y < len(classes) for y in labels)
              and all(labels.count(y) == split['k'] for y in range(len(classes))), 'Support labels/K mismatch')
        check(isinstance(split['receiver'], str) and split['receiver'] not in source_receivers
              and split['scenario'] in manifest['scenarios'], 'Source receiver overlap or unregistered scenario')
        for index, label in zip(support, labels):
            physical_class = classes[label]
            check(index not in labels_by_index or labels_by_index[index] == physical_class, 'Contradictory physical support label')
            labels_by_index[index] = physical_class
        query_union.update(query)
        entry = {k: split[k] for k in ('split_id', 'receiver', 'scenario', 'k', 'support_seed', 'registered_classes', 'support_labels', 'support_indices')}
        entry['support_ids'] = ids[support].tolist(); splits.append(entry)
    # These frozen capsules use globally disjoint pools. Do not silently read a
    # physical query from any registered split as support for a different split.
    check(not set(labels_by_index) & query_union, 'Cross-split support/query overlap')
    indices = sorted(labels_by_index, key=lambda i: ids[i])
    return manifest, splits, indices, [labels_by_index[i] for i in indices]


def historical_fft96(iq):
    """Same received-only FFT96 formula as the existing frozen feature cache."""
    result = np.empty((len(iq), 96), dtype=np.float32)
    source = np.linspace(0., 1., 256, dtype=np.float64); target = np.linspace(0., 1., 96, dtype=np.float64)
    for index, row in enumerate(iq):
        z = row[0].astype(np.float64) + 1j*row[1].astype(np.float64); z -= np.mean(z)
        rms = float(np.sqrt(np.mean(np.abs(z)**2)))
        if rms > 1e-8: z /= rms
        spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft(z*np.hanning(256)))))
        sketch = np.interp(target, source, spectrum).astype(np.float32)
        sketch -= np.mean(sketch, dtype=np.float64).astype(np.float32)
        sketch /= max(float(np.linalg.norm(sketch)), 1e-8)
        result[index] = sketch
    check(np.isfinite(result).all(), 'Nonfinite support FFT')
    return result


class FrozenBranches:
    def __init__(self, model, native_forward, device):
        import torch
        self.torch, self.model, self.forward, self.device = torch, model, native_forward, device
        model.eval(); model.requires_grad_(False)
        for value in model.buffers():
            if value.is_floating_point() or value.is_complex(): value.requires_grad_(False)
        b = model.id_backbone
        check(model.id_feature_key == 'feat_joint' and b.emb_dim == 160 and b.input_len == 256
              and b.use_time_path and b.use_freq_path and b.use_pa_path and not b.use_dac_path,
              'Unexpected frozen Phase1 active branches')
        self.attributes = dict(emb_dim=160, t_dim=b.t_proj.out_features, f_dim=b.f_proj.out_features,
            use_time_path=bool(b.use_time_path), use_freq_path=bool(b.use_freq_path),
            enable_dac=bool(b.use_dac_path), enable_pa=bool(b.use_pa_path), active_defects=['pa'],
            id_feature_key=model.id_feature_key)
        check(self.attributes['t_dim'] == self.attributes['f_dim'] == 160, 'Branch dimensions changed')
        self.parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
        self.buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
        self.batch_calls = self.physical_forward_count = self.identity_reference_checks = 0
        self.verify_frozen()

    def verify_frozen(self):
        check(all(not module.training for module in self.model.modules())
              and all(not value.requires_grad for value in self.model.parameters()), 'Encoder must remain frozen/eval')
        for current, before, kind in ((dict(self.model.named_parameters()), self.parameters, 'parameter'),
                                      (dict(self.model.named_buffers()), self.buffers, 'buffer')):
            check(set(current) == set(before) and all(self.torch.equal(value, before[name]) for name, value in current.items()),
                  'Native '+kind+' changed during support export')

    def __call__(self, iq):
        torch = self.torch
        with torch.inference_mode():
            # Scalar-list bridge avoids the deployed Torch/NumPy C-ABI split.
            tensor = torch.tensor(np.asarray(iq, dtype=np.float32).tolist(), dtype=torch.float32, device=self.device)
            aux = self.forward.backbone_forward_compat(self.model.id_backbone, tensor,
                y=None, return_aux=True, domain_labels=None)
            self.batch_calls += 1; self.physical_forward_count += len(iq)
            identity = self.model._pick_z_id(aux).float()
            # Exercise the original identity selector against the SAME aux.
            # The proxy serves cached output and never executes the encoder.
            proxy = SimpleNamespace(id_backbone=lambda *a, **kw: aux, _pick_z_id=self.model._pick_z_id)
            reference = self.forward.identity_only_feature_forward(proxy, tensor, 'z_id')
            check(reference is not None and torch.equal(identity, reference[0]), 'Original identity exporter mismatch')
            self.identity_reference_checks += len(iq)
            values = dict(z_id=identity, **{key: aux[key] for key in BRANCHES[1:]})
            for key, value in values.items():
                check(torch.is_tensor(value) and value.shape == (len(iq), 160) and torch.isfinite(value).all(), 'Invalid native branch: '+key)
            return {key: np.asarray(value.detach().float().cpu().tolist(), dtype=np.float32) for key, value in values.items()}


def load_native(*, checkpoint_root, native_code, source_contract, seed, expected_checkpoint_sha256, device):
    source, native_code = Path(checkpoint_root), Path(native_code).resolve()
    actual, cfg = verify_source(source, read(source_contract), seed)
    checkpoint = source/'final_ssdg.pth'
    check(sha256(checkpoint) == expected_checkpoint_sha256, 'Checkpoint SHA mismatch')
    sys.path.insert(0, str(native_code))
    import torch
    from cvsrffi import checkpoint_loading, identity_only_forward
    for module in (checkpoint_loading, identity_only_forward):
        check(Path(module.__file__).resolve().is_relative_to(native_code), 'Native module import was shadowed')
    torch.set_num_threads(2)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False); args = payload['args']
    check(payload['epoch'] == 200 and args['seed'] == seed and args['from_scratch']
          and not args['baseline_ckpt'] and not args['teacher_ckpt']
          and strings(actual['classes']) and len(actual['classes']) == actual['num_classes'], 'Final checkpoint/inheritance mismatch')
    model, audit = checkpoint_loading.build_exact_ssdg_model_from_checkpoint(payload, input_len=256, device=torch.device(device))
    infer = FrozenBranches(model, identity_only_forward, torch.device(device))
    return infer, dict(verdict='MATCHED_SOURCE_ONLY_SCRATCH', source_role_comparison='EXACT_MATCH',
        checkpoint=str(checkpoint), checkpoint_sha256=expected_checkpoint_sha256, checkpoint_epoch=200,
        checkpoint_inheritance=[], target_access_before_freeze=False, classes=actual['classes'], model_seed=seed,
        model_file_bytes=checkpoint.stat().st_size, native_code=str(native_code), loader=audit,
        source_arguments={k: cfg.get(k) for k in ('seed', 'model_variant', 'id_feature_key', 'branch_ablation', 'checkpoint_selection')},
        active_branches=infer.attributes)


def export(*, checkpoint_root, native_code, source_contract, source_receivers, seed, capsule, output,
           expected_checkpoint_sha256, expected_capsule_id, device='cpu', batch_size=32):
    out, capsule = Path(output), Path(capsule)
    if out.exists(): raise FileExistsError(out)
    check(type(batch_size) is int and batch_size > 0, 'Positive batch size required')
    check(isinstance(expected_checkpoint_sha256, str) and len(expected_checkpoint_sha256) == 64
          and all(c in '0123456789abcdef' for c in expected_checkpoint_sha256), 'Invalid expected checkpoint SHA')
    started = time.perf_counter(); reader = SupportIQ(capsule/'received.npz'); owns_output = False
    try:
        manifest, splits, indices, labels = support_plan(capsule, reader.ids, expected_capsule_id, source_receivers)
        reader.allow(indices)
        timing = dict(metadata_seconds=time.perf_counter()-started)
        stage = time.perf_counter()
        infer, provenance = load_native(checkpoint_root=checkpoint_root, native_code=native_code, source_contract=source_contract,
            seed=seed, expected_checkpoint_sha256=expected_checkpoint_sha256, device=device)
        timing['checkpoint_load_seconds'] = time.perf_counter()-stage
        old = provenance['classes']
        check(all(split['registered_classes'][:len(old)] == old for split in splits), 'Frozen old class registry mismatch')
        out.mkdir(parents=True, exist_ok=False)
        owns_output = True
        startup = dict(schema=CACHE_SCHEMA, argv=sys.argv, python=sys.executable, pid=os.getpid(),
            capsule_id=expected_capsule_id, checkpoint_sha256=expected_checkpoint_sha256, model_seed=seed,
            feature_contract=FEATURE_CONTRACT, device=device, batch_size=batch_size, source_receivers=source_receivers,
            provenance=provenance, query_iq_access=False, query_rows_read=0, query_fit_access=False, truth_read=False,
            source_data_access=False, adapted_state_inherited=False, native_eval=True,
            new_source_payload_bytes=0, new_ground_statistics_bytes=0, model_already_deployed=None,
            model_incremental_transfer_bytes=None, model_deployment_unknown_reason='Deployment status unknown',
            model_file_bytes=provenance['model_file_bytes'], model_file_bytes_scope='Full training checkpoint package, not a minimal inference package')
        write(out/'startup.json', startup); print(json.dumps(dict(event='STARTUP', **startup)), flush=True)
        arrays = {key: np.empty((len(indices), 160), dtype=np.float32) for key in BRANCHES}
        arrays['fft'] = np.empty((len(indices), 96), dtype=np.float32)
        timing.update(support_read_seconds=0., native_forward_seconds=0., fft_seconds=0., frozen_verify_seconds=0.)
        for offset in range(0, len(indices), batch_size):
            stop = min(offset+batch_size, len(indices)); stage = time.perf_counter()
            iq = reader.take(indices[offset:stop]); timing['support_read_seconds'] += time.perf_counter()-stage
            stage = time.perf_counter(); branches = infer(iq); timing['native_forward_seconds'] += time.perf_counter()-stage
            for key in BRANCHES: arrays[key][offset:stop] = branches[key]
            stage = time.perf_counter(); arrays['fft'][offset:stop] = historical_fft96(iq); timing['fft_seconds'] += time.perf_counter()-stage
            stage = time.perf_counter(); infer.verify_frozen(); timing['frozen_verify_seconds'] += time.perf_counter()-stage
            print(json.dumps(dict(event='SUPPORT_FEATURE_PROGRESS', completed=stop, total=len(indices),
                native_batch_calls=infer.batch_calls, elapsed_seconds=time.perf_counter()-started)), flush=True)
        stage = time.perf_counter()
        indices_array = np.asarray(indices, dtype=np.int64); ids = reader.ids[indices]; class_ids = np.asarray(labels)
        np.savez(out/CACHE_NAME, **arrays, ids=ids, labels=class_ids, indices=indices_array,
            checkpoint_sha256=np.asarray(expected_checkpoint_sha256), capsule_id=np.asarray(expected_capsule_id),
            feature_contract_json=np.asarray(json.dumps(FEATURE_CONTRACT, sort_keys=True)))
        write(out/'support_splits.json', dict(schema=CACHE_SCHEMA, capsule_id=expected_capsule_id,
            checkpoint_sha256=expected_checkpoint_sha256, splits=splits))
        write(out/'checkpoint_provenance.json', provenance)
        timing['write_seconds'] = time.perf_counter()-stage; timing['total_seconds'] = time.perf_counter()-started
        metadata_bytes = {name: (out/name).stat().st_size for name in ('startup.json', 'support_splits.json', 'checkpoint_provenance.json')}
        marker = dict(status='BRANCH_SUPPORT_FEATURES_COMPLETE', schema=CACHE_SCHEMA, feature_contract=FEATURE_CONTRACT,
            capsule_id=expected_capsule_id, checkpoint_sha256=expected_checkpoint_sha256, model_seed=seed, classes=old,
            count=len(indices), split_count=len(splits), dtype='float32', shapes={k: list(v.shape) for k, v in arrays.items()},
            feature_array_bytes=sum(v.nbytes for v in arrays.values()), index_bytes=indices_array.nbytes,
            registry_array_bytes=ids.nbytes+class_ids.nbytes, feature_file_bytes=(out/CACHE_NAME).stat().st_size,
            metadata_file_bytes=metadata_bytes,
            artifact_file_bytes_excluding_completion_marker=(out/CACHE_NAME).stat().st_size+sum(metadata_bytes.values()),
            model_file_bytes=provenance['model_file_bytes'], view_count_per_observation=1,
            native_physical_forward_count=infer.physical_forward_count, native_batch_calls=infer.batch_calls,
            identity_reference_checks=infer.identity_reference_checks, identity_check_additional_encoder_forwards=0,
            support_iq_rows_read=reader.rows_read, iq_access='ZIP_STORED_NPY_readonly_mmap_support_indices_only',
            capsule_revalidated=False, received_file_rehashed=False, native_eval=True, native_parameters_unchanged=True,
            native_buffers_unchanged=True, encoder_updated=False, active_branches=infer.attributes,
            query_iq_access=False, query_rows_read=0, query_used_for_fitting=False, source_data_access=False, truth_read=False,
            adapted_state_inherited=False, new_source_payload_bytes=0, new_ground_statistics_bytes=0,
            timing=timing, peak_process_rss_bytes=peak_process_rss(),
            peak_process_rss_reason='Linux process high-water RSS; null when unavailable on this platform',
            byte_scope='Support feature/index/registry arrays and cache file; received-side storage, not source transmission')
        write(out/'features_complete.json', marker); print(json.dumps(marker, allow_nan=False), flush=True)
        return marker
    except Exception as exc:
        if owns_output and not (out/'features_complete.json').exists() and not (out/'technical_failure.json').exists():
            write(out/'technical_failure.json', dict(status='TECHNICAL_FAILURE', error_type=type(exc).__name__, error=str(exc)))
        raise
    finally:
        reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint-root', 'native-code', 'source-contract', 'source-receivers', 'capsule', 'output',
                'expected-checkpoint-sha256', 'expected-capsule-id'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--seed', type=int, required=True); parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    args = vars(parser.parse_args()); args['source_receivers'] = json.loads(args['source_receivers']); export(**args)


if __name__ == '__main__': main()
