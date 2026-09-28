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


def run(spec_path, commit):
    spec_path = Path(spec_path).resolve()
    spec = read(spec_path)
    root = Path(spec['execution']['remote_run_root'])
    confirmation, rows = spec['confirmation'], spec['rows']
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
          worker_policy=dict(feature_gpu=0, max_feature_workers=1, cpu_row_workers=4, blas_threads=2),
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
        if reuse_id is not None:
            if not isinstance(reuse_id, str) or not reuse_id or manifest.get('capsule_id') != reuse_id or manifest['split_count'] != 900:
                raise ValueError('Existing validated capsule identity/split count mismatch')
            write(root / 'data_reuse.json', dict(status='VALIDATED_ONCE_REUSED', capsule_id=reuse_id,
                  capsule=confirmation['capsule'], split_count=manifest['split_count'],
                  protocol_schema=manifest['protocol_schema'], phase2_data_status=manifest['phase2_data_status'],
                  data_rebuilt=False, data_revalidated=False, truth_read=False, timestamp=time.time()))
        hashes = {}
        stage('EXPORTING_FROZEN_FEATURES')
        for row in rows:
            row_id, output = row['row_id'], Path(row['output_root'])
            output.mkdir(parents=True, exist_ok=False)
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
                update(row_id, status='D92_PREDICTING')
                invoke('cvs_d92_matched.py', ['predict', '--features', output / 'received_features' / 'received_features.npz',
                    '--capsule', confirmation['capsule'], '--ground', output / 'ground', '--output', output,
                    '--seed', row['seeds']['model']], output / 'd92.log')
                update(row_id, status='SCV_PREDICTING')
                invoke('predict_d92_support_cv.py', ['--row-root', output, '--capsule', confirmation['capsule'],
                    '--output', output / 'scv', '--config', confirmation['candidate_config'],
                    '--expected-capsule-id', manifest['capsule_id'], '--expected-checkpoint-sha256', hashes[row_id]],
                    output / 'scv.log')
                for folder in (output, output / 'scv'):
                    marker = read(folder / 'predictions_complete.json')
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
            if value['status'] in ('VERIFYING_SOURCE', 'EXPORTING_FROZEN_FEATURES'):
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
