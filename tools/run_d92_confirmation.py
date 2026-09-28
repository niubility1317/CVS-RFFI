"""Single-owner paired D92 confirmation; scoring starts after every prediction."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

RELEASE = Path(__file__).resolve().parents[1]
MODEL_SEEDS = (2026092701, 2026092702, 2026092703, 2026092704)
CANDIDATE_FIELDS = ('candidate_method', 'candidate_folder', 'candidate_predictor', 'candidate_mode')
CANDIDATES = (
    ('D92-SCV-v1', 'scv', 'predict_d92_support_cv.py', 'd92_scv_registration'),
    ('D92-SFHead-v1', 'sfhead', 'predict_d92_sourcefree_head.py', 'd92_sfhead_registration'),
    ('D92-SGJoint-v1', 'sgjoint', 'predict_d92_summary_joint.py', 'd92_sgjoint_registration'),
    ('D92-MVKME-v1', 'mvkme', 'predict_d92_mv_kme.py', 'd92_mvkme_registration'),
    ('D92-BNNA-v1', 'bnna', 'predict_d92_bnna.py', 'd92_bnna_registration'),
    ('D92-OSC-v1', 'osc', 'predict_d92_orbit_shared.py', 'd92_osc_registration'),
    ('D92-MVRidge-v1', 'mvridge', 'predict_d92_multiview_ridge.py', 'd92_mvridge_registration'),
)


def candidate_definition(confirmation):
    """An exact allowlisted combination prevents script/path/method substitution."""
    values = tuple(confirmation.get(name, default) for name, default in zip(CANDIDATE_FIELDS, CANDIDATES[0]))
    if values not in CANDIDATES:
        raise ValueError('Candidate method/folder/predictor/mode combination is not allowed')
    return dict(zip(CANDIDATE_FIELDS, values))


def reuse_frozen_rows(spec):
    """Require an intentional, complete, same-parent-run reuse matrix."""
    rows=spec['rows']
    present=[('reuse_row_root' in row or 'expected_checkpoint_sha256' in row) for row in rows]
    if not any(present):
        return False
    if not all(present) or len(rows)!=4:
        raise ValueError('Frozen-row reuse requires all four model rows')
    if not spec['confirmation'].get('reuse_validated_capsule_id') or not spec['confirmation'].get('expected_split_count'):
        raise ValueError('Frozen-row reuse requires explicit capsule ID and split count')
    origins=[]
    root=Path(spec['execution']['remote_run_root']).resolve()
    for row in rows:
        if not row.get('reuse_row_root'):
            raise ValueError('Missing frozen row path')
        digest=row.get('expected_checkpoint_sha256')
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Missing or invalid expected checkpoint SHA256 for reused row')
        origin=Path(row['reuse_row_root']).resolve()
        if origin==root or origin in root.parents or root in origin.parents:
            raise ValueError('New run and reused model row must not overlap')
        origins.append(origin)
    if len(set(origins))!=4 or len({path.parent for path in origins})!=1:
        raise ValueError('Reused model rows must be distinct and belong to one original run')
    return True


def baseline_root(row):
    return Path(row.get('reuse_row_root',row['output_root']))


def needs_multiview(confirmation):
    return feature_definition(confirmation) is not None


def feature_definition(confirmation):
    method = candidate_definition(confirmation)['candidate_method']
    return {
        'D92-MVKME-v1': ('export_d92_mv_kme_features.py', 'mv_features', '--mv-features', 'MULTIVIEW_FEATURES_COMPLETE'),
        'D92-BNNA-v1': ('export_d92_bnna_features.py', 'bnna_features', '--bnna-features', 'BNNA_FEATURES_COMPLETE'),
    }.get(method)


def reuse_multiview_cache(spec):
    """Registered support methods can reuse the exact frozen four-view cache."""
    method = candidate_definition(spec['confirmation'])['candidate_method']
    paths = [row.get('reuse_multiview_features_root') for row in spec['rows']]
    if method not in ('D92-OSC-v1', 'D92-MVRidge-v1'):
        if any(paths): raise ValueError('Frozen multiview cache reuse is not registered for this method')
        return False
    if not reuse_frozen_rows(spec) or not all(isinstance(p,str) and p for p in paths):
        raise ValueError('Frozen orbit cache reuse requires baseline rows and all four feature paths')
    source = spec['confirmation'].get('frozen_feature_source_root')
    if not isinstance(source,str) or not source:
        raise ValueError('Frozen orbit cache reuse requires an explicit feature source root')
    source, output = Path(source).resolve(), Path(spec['execution']['remote_run_root']).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Frozen feature source and new output must not overlap')
    expected = [source/row['row_id']/'bnna_features' for row in spec['rows']]
    if [Path(p).resolve() for p in paths] != expected or len(set(paths)) != 4:
        raise ValueError('Frozen orbit cache paths must bind each model row to one registered source run')
    return True


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def worker_environment(gpu=False):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0' if gpu else '', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
               PYTHONUNBUFFERED='1')
    return env


def invoke(tool, arguments, log, *, gpu=False):
    command = [sys.executable, '-u', str(RELEASE / 'tools' / tool), *map(str, arguments)]
    with Path(log).open('x', encoding='utf-8') as stream:
        child = subprocess.Popen(command, cwd=RELEASE, env=worker_environment(gpu),
                                 stdout=stream, stderr=subprocess.STDOUT)
        write(str(log) + '.process.json', dict(pid=child.pid, argv=command, cwd=str(RELEASE),
              gpu='0' if gpu else None, blas_threads=2, started=time.time()))
        code = child.wait()
    if code:
        raise RuntimeError(f'{tool} failed with exit code {code}; preserved {log}; no automatic retry')


def prepare_ground(row, confirmation):
    # Native verifier reads only source records and does not import its model runtime.
    from cvs_native_artifacts import verify_source
    sys.path.insert(0, str(RELEASE / 'code'))
    from cvsrffi.phase1_center_lowrank_prototype_bundle import NPZ_NAME
    import numpy as np
    source = Path(row['source_root'])
    contract = read(confirmation['source_contract'])
    actual, _ = verify_source(source, contract, row['seeds']['model'])
    expected_classes = confirmation['old_classes']
    if (not isinstance(expected_classes, list) or len(expected_classes) != 6
            or len(set(expected_classes)) != 6 or actual.get('classes') != expected_classes):
        raise ValueError('Checkpoint source class registry mismatch')
    checkpoint_sha = sha(source / 'final_ssdg.pth')
    origin = Path(row['ground_source'])
    manifest = read(origin / 'manifest.json')
    if (manifest.get('checkpoint_sha256') != checkpoint_sha or manifest.get('target_access') is not False
            or manifest.get('source_role') != 'L_s' or manifest.get('formal_phase2_eligible') is not True
            or manifest.get('provenance_status') != 'CURRENT_FINAL_SCRATCH_EXACT_SOURCE_L'
            or manifest.get('member_allowlist') != [NPZ_NAME]):
        raise ValueError('Ground source/checkpoint provenance mismatch')
    with np.load(origin / NPZ_NAME, allow_pickle=False) as aggregate:
        if aggregate['class_registry'].astype(str).tolist() != actual['classes']:
            raise ValueError('Ground/source class registry mismatch')
    destination = Path(row['output_root']) / 'ground'
    destination.mkdir(exist_ok=False)
    members = {}
    for name in ('manifest.json', NPZ_NAME):
        before = sha(origin / name)
        shutil.copyfile(origin / name, destination / name)
        if sha(destination / name) != before or sha(origin / name) != before:
            raise ValueError('Ground changed during immutable copy')
        members[name] = before
    write(Path(row['output_root']) / 'ground_provenance.json', dict(
        source_root=str(source), ground_source=str(origin), checkpoint_sha256=checkpoint_sha,
        members=members, verification='MATCHED_SOURCE_ONLY_SCRATCH_FINAL200', target_access=False))
    return checkpoint_sha


def validate_reused_row(row, confirmation, capsule_data, *, validate_predictions=True):
    """Read frozen artifacts/metadata only; never source samples or old scores.json."""
    import numpy as np
    from score_d92_confirmation import validate_method_records
    sys.path.insert(0,str(RELEASE/'code'))
    from cvsrffi.d92_ground_summary import load_ground_summary
    origin=baseline_root(row)
    manifest,ids,splits=capsule_data
    expected=row['expected_checkpoint_sha256'];seed=row['seeds']['model']
    previous=read(origin/'d92_startup.json')
    features=origin/'received_features'
    if (previous.get('checkpoint_sha256')!=expected or previous.get('seed')!=seed
            or previous.get('capsule')!=str(Path(confirmation['capsule']))
            or previous.get('features')!=str(features/'received_features.npz')
            or previous.get('ground')!=str(origin/'ground')
            or previous.get('query_fit_access') is not False or previous.get('truth_read') is not False):
        raise ValueError('Reused D92 startup checkpoint/seed/capsule/artifact binding mismatch')
    marker=read(features/'features_complete.json')
    provenance=read(features/'checkpoint_provenance.json')
    feature_start=read(features/'startup.json')
    ground_origin=read(origin/'ground_provenance.json')
    complete=read(origin/'predictions_complete.json')
    classes=confirmation['old_classes']
    if (marker.get('status')!='FROZEN_FEATURES_COMPLETE' or marker.get('capsule_id')!=manifest['capsule_id']
            or marker.get('count')!=len(ids) or marker.get('query_used_for_fitting') is not False
            or provenance.get('verdict')!='MATCHED_SOURCE_ONLY_SCRATCH'
            or provenance.get('initialization')!='scratch' or provenance.get('checkpoint_inheritance')!=[]
            or provenance.get('target_access_before_freeze') is not False
            or provenance.get('source_role_comparison')!='EXACT_MATCH' or provenance.get('checkpoint_epoch')!=200
            or provenance.get('selection')!='fixed_final_epoch200' or provenance.get('model_seed')!=seed
            or provenance.get('classes')!=classes or feature_start.get('source_arguments',{}).get('seed')!=seed
            or ground_origin.get('checkpoint_sha256')!=expected
            or provenance.get('checkpoint')!=str(Path(ground_origin['source_root'])/'final_ssdg.pth')):
        raise ValueError('Reused frozen feature/ground provenance mismatch')
    if (complete.get('status')!='PREDICTIONS_COMPLETE' or complete.get('capsule_id')!=manifest['capsule_id']
            or complete.get('split_count')!=manifest['split_count'] or complete.get('truth_read') is not False):
        raise ValueError('Reused baseline completion binding mismatch')
    summary=load_ground_summary(origin/'ground',expected_checkpoint_sha256=expected,
                               expected_classes=classes,already_deployed=True)
    with np.load(features/'received_features.npz',allow_pickle=False) as arrays:
        if set(arrays.files)!={'identity160','logits','ids'}:
            raise ValueError('Reused feature member mismatch')
        if (not np.array_equal(arrays['ids'].astype(str),ids)
                or arrays['identity160'].shape!=(len(ids),160) or arrays['logits'].shape!=(len(ids),len(classes))
                or not np.isfinite(arrays['identity160']).all() or not np.isfinite(arrays['logits']).all()):
            raise ValueError('Reused features/physical IDs mismatch')
    for split in splits.values():
        if split['registered_classes'][:len(classes)]!=classes:
            raise ValueError('Reused capsule/ground class order mismatch')
    if validate_predictions:
        validate_method_records(origin,'D92',manifest,ids,splits,candidate_definition(confirmation))
    return dict(status='FROZEN_ARTIFACTS_REUSED',reuse_row_root=str(origin),baseline_root=str(origin),
        model_seed=seed,checkpoint_sha256=expected,capsule_id=manifest['capsule_id'],
        split_count=manifest['split_count'],baseline_predictions=complete['predictions'],
        ground_payload_bytes=summary.payload_audit['total_file_bytes'],encoder_executed=False,
        baseline_executed=False,source_samples_read=False,old_scores_read=False,truth_read=False)


def run(spec_path, commit):
    spec_path = Path(spec_path).resolve()
    spec = read(spec_path)
    root = Path(spec['execution']['remote_run_root'])
    confirmation, rows = spec['confirmation'], spec['rows']
    candidate = candidate_definition(confirmation)
    reuse_rows=reuse_frozen_rows(spec)
    reuse_views=reuse_multiview_cache(spec)
    multiview=needs_multiview(confirmation)
    if multiview and not reuse_rows:
        raise ValueError('Multiview candidate requires explicitly reused baseline rows')
    expected_splits = confirmation.get('expected_split_count')
    if expected_splits is not None and (type(expected_splits) is not int or expected_splits <= 0):
        raise ValueError('expected_split_count must be a positive integer')
    if sorted(r['seeds']['model'] for r in rows) != list(MODEL_SEEDS):
        raise ValueError('Expected exactly the four preregistered fresh model seeds')
    if len({r['row_id'] for r in rows}) != len(rows):
        raise ValueError('Repeated model row ID')
    paths = [Path(r['output_root']).resolve() for r in rows]
    if len(set(paths)) != len(rows) or any(p == root.resolve() or root.resolve() not in p.parents for p in paths):
        raise ValueError('Every row must own a distinct output below the new run root')
    if any(a in b.parents for a in paths for b in paths if a != b):
        raise ValueError('Model output directories must not nest')
    data = read(confirmation['data_config'])
    data_root = Path(data['output_root']).resolve()
    if (root.resolve() == data_root or root.resolve() in data_root.parents or data_root in root.resolve().parents
            or Path(confirmation['capsule']).resolve() != data_root / 'capsule'
            or Path(confirmation['truth']).resolve() != data_root / 'score_only' / 'truth.json'):
        raise ValueError('Data must use its separate preregistered capsule/truth paths')
    root.mkdir(parents=True, exist_ok=False)
    state = {r['row_id']: dict(status='PLANNED') for r in rows}
    lock = threading.Lock()

    def update(row_id, **values):
        with lock:
            state[row_id].update(values, updated=time.time())
            write(root / 'state.json', state)

    def stage(name, **values):
        write(root / 'workflow_state.json', dict(status=name, **values, updated=time.time()))

    write(root / 'startup.json', dict(pid=os.getpid(), argv=sys.argv, cwd=os.getcwd(),
          python=sys.executable, commit=commit, spec=spec, launch_owner=spec['execution']['launch_owner'],
          environment={k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
              'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
          worker_policy=dict(feature_gpu=None if reuse_rows and not multiview else 0, max_feature_workers=0 if reuse_rows and not multiview else 1,
                             cpu_row_workers=4, blas_threads=2, reuse_frozen_rows=reuse_rows),
          timestamp=time.time()))
    write(root / 'state.json', state)
    try:
        reuse_id = confirmation.get('reuse_validated_capsule_id')
        if reuse_id is None:
            stage('BUILDING_DATA')
            invoke('build_d92_confirmation_data.py', ['--config', confirmation['data_config']], root / 'build_data.log')
        else:
            stage('REUSING_VALIDATED_DATA')
        manifest = read(Path(confirmation['capsule']) / 'manifest.json')
        if (manifest.get('protocol_schema') != 'p2_min_v1' or manifest.get('phase2_data_status') != 'VALIDATED_ONCE'
                or not isinstance(manifest.get('split_count'), int) or manifest['split_count'] <= 0):
            raise ValueError('Data builder did not produce a validated complete capsule')
        if expected_splits is not None and manifest['split_count'] != expected_splits:
            raise ValueError('Capsule split count differs from preregistered expected_split_count')
        if reuse_id is not None:
            if (not isinstance(reuse_id, str) or not reuse_id or manifest.get('capsule_id') != reuse_id
                    or manifest['split_count'] != (900 if expected_splits is None else expected_splits)):
                raise ValueError('Existing validated capsule identity/split count mismatch')
            write(root / 'data_reuse.json', dict(status='VALIDATED_ONCE_REUSED', capsule_id=reuse_id,
                  capsule=confirmation['capsule'], split_count=manifest['split_count'],
                  protocol_schema=manifest['protocol_schema'], phase2_data_status=manifest['phase2_data_status'],
                  data_rebuilt=False, data_revalidated=False, truth_read=False, timestamp=time.time()))
        hashes = {}
        capsule_data=None
        if reuse_rows:
            from score_d92_confirmation import load_prediction_capsule
            capsule_data=load_prediction_capsule(confirmation)
        stage('VERIFYING_FROZEN_ARTIFACTS' if reuse_rows else 'EXPORTING_FROZEN_FEATURES')
        for row in rows:
            row_id, output = row['row_id'], Path(row['output_root'])
            output.mkdir(parents=True, exist_ok=False)
            if reuse_rows:
                update(row_id,status='VERIFYING_FROZEN_ARTIFACTS')
                evidence=validate_reused_row(row,confirmation,capsule_data)
                hashes[row_id]=row['expected_checkpoint_sha256']
                write(output/'artifact_reuse.json',evidence)
                if reuse_views:
                    cache=Path(row['reuse_multiview_features_root'])
                    marker=read(cache/'features_complete.json')
                    if (marker.get('status')!='BNNA_FEATURES_COMPLETE'
                            or marker.get('capsule_id')!=manifest['capsule_id']
                            or marker.get('checkpoint_sha256')!=hashes[row_id]
                            or marker.get('model_seed')!=row['seeds']['model']
                            or marker.get('query_used_for_fitting') is not False
                            or marker.get('encoder_updated') is not False):
                        raise ValueError('Reused four-view cache metadata binding mismatch')
                    write(output/'frozen_feature_reuse.json',dict(feature_root=str(cache),
                        cache_recomputed=False,checkpoint_reloaded=False,adapted_state_reused=False,
                        source_samples_read=False,query_used_for_fitting=False,
                        checkpoint_sha256=hashes[row_id],capsule_id=manifest['capsule_id']))
                if multiview:
                    exporter, feature_folder, feature_flag, feature_status = feature_definition(confirmation)
                    update(row_id,status='EXPORTING_MULTIVIEW_FEATURES')
                    invoke(exporter, ['--row-root', baseline_root(row),
                        '--source', row['source_root'], '--contract', confirmation['source_contract'],
                        '--native-code', confirmation['native_code'], '--seed', row['seeds']['model'],
                        '--capsule', confirmation['capsule'], '--output', output/feature_folder,
                        '--config', confirmation['candidate_config'], '--expected-capsule-id', manifest['capsule_id'],
                        '--expected-checkpoint-sha256', hashes[row_id], '--device', 'cuda:0'],
                        output/(feature_folder+'.log'), gpu=True)
                    marker=read(output/feature_folder/'features_complete.json')
                    if (marker.get('status')!=feature_status
                            or marker.get('capsule_id')!=manifest['capsule_id']
                            or marker.get('checkpoint_sha256')!=hashes[row_id]
                            or marker.get('query_used_for_fitting') is not False):
                        raise ValueError('Multiview feature completion or binding mismatch')
                update(row_id,status='FEATURES_COMPLETE',checkpoint_sha256=hashes[row_id],reuse_row_root=row['reuse_row_root'])
                continue
            update(row_id, status='VERIFYING_SOURCE')
            hashes[row_id] = prepare_ground(row, confirmation)
            update(row_id, status='EXPORTING_FROZEN_FEATURES', checkpoint_sha256=hashes[row_id])
            invoke('cvs_native_artifacts.py', ['received-features', '--source', row['source_root'],
                '--contract', confirmation['source_contract'], '--native-code', confirmation['native_code'],
                '--output', output / 'received_features', '--seed', row['seeds']['model'],
                '--capsule', confirmation['capsule'], '--device', 'cuda:0'], output / 'features.log', gpu=True)
            marker = read(output / 'received_features' / 'features_complete.json')
            if (marker.get('status') != 'FROZEN_FEATURES_COMPLETE' or marker.get('capsule_id') != manifest['capsule_id']
                    or marker.get('query_used_for_fitting') is not False):
                raise ValueError('Native feature completion or capsule mismatch')
            update(row_id, status='FEATURES_COMPLETE')

        def predict_row(row):
            row_id, output = row['row_id'], Path(row['output_root'])
            try:
                if not reuse_rows:
                    update(row_id, status='D92_PREDICTING')
                    invoke('cvs_d92_matched.py', ['predict', '--features', output / 'received_features' / 'received_features.npz',
                        '--capsule', confirmation['capsule'], '--ground', output / 'ground', '--output', output,
                        '--seed', row['seeds']['model']], output / 'd92.log')
                folder = candidate['candidate_folder']
                update(row_id, status=folder.upper() + '_PREDICTING')
                arguments=['--row-root', baseline_root(row), '--capsule', confirmation['capsule'],
                    '--output', output / folder, '--config', confirmation['candidate_config'],
                    '--expected-capsule-id', manifest['capsule_id'], '--expected-checkpoint-sha256', hashes[row_id]]
                if multiview:
                    _, feature_folder, feature_flag, _ = feature_definition(confirmation)
                    arguments.extend([feature_flag,output/feature_folder])
                if reuse_views: arguments.extend(['--orbit-features',row['reuse_multiview_features_root']])
                invoke(candidate['candidate_predictor'], arguments, output / (folder + '.log'))
                for prediction_root in (baseline_root(row), output / folder):
                    marker = read(prediction_root / 'predictions_complete.json')
                    if (marker.get('status') != 'PREDICTIONS_COMPLETE' or marker.get('split_count') != manifest['split_count']
                            or marker.get('capsule_id') != manifest['capsule_id'] or marker.get('truth_read') is not False):
                        raise ValueError('Paired method predictions incomplete')
                update(row_id, status='PREDICTIONS_COMPLETE')
            except Exception as error:
                update(row_id, status='TECHNICAL_FAILURE', error=str(error))
                raise

        stage('PREDICTING')
        errors = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(predict_row, row) for row in rows]
            # Let healthy paired rows finish; never terminate another process on failure.
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    errors.append(str(error))
        if errors:
            raise RuntimeError('; '.join(errors))
        if any(value['status'] != 'PREDICTIONS_COMPLETE' for value in state.values()):
            raise ValueError('Refusing scoring before every preregistered row completes')
        stage('SCORING')
        invoke('score_d92_confirmation.py', ['--spec', spec_path, '--output', root / 'scores.json'], root / 'score.log')
        scored = read(root / 'scores.json')
        if scored.get('status') != 'SCORED' or scored.get('selection_feedback_forbidden') is not True:
            raise ValueError('Independent scorer did not complete')
        stage('SCORED', records=len(scored['results']), commit=commit)
        write(root / 'complete.json', dict(status='SCORED', commit=commit, model_rows=len(rows),
              records=len(scored['results']), selection_feedback_forbidden=True))
    except Exception as error:
        for row_id, value in list(state.items()):
            if value['status'] in ('VERIFYING_SOURCE', 'EXPORTING_FROZEN_FEATURES','VERIFYING_FROZEN_ARTIFACTS','EXPORTING_MULTIVIEW_FEATURES'):
                update(row_id, status='TECHNICAL_FAILURE', error=str(error))
        stage('TECHNICAL_FAILURE', error=str(error), retry_policy='none; preserve all artifacts')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    run(args.spec, args.commit)


if __name__ == '__main__':
    main()
