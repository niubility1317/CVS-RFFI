"""Collect saved public aggregates only; never reads IQ, weights, truth or models.

This is an explicit one-shot transfer, not a polling owner or a training command.
An existing local root or remote export is never overwritten. On transport timeout,
reconcile the recorded export location before considering another invocation.
"""
import argparse
import hashlib
import io
import json
import re
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath


RUN = '20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01'
SOURCE_FILES = ('source_complete.json', 'source_summary.json',
                'source_confusions.json', 'source_metrics.csv', 'training_logs.json')
FINAL_FILES = ('formal_summary.json', 'formal_confusions.json',
               'formal_metrics.csv', 'final_report.md', 'costs.json', 'complete.json')
SOURCE_FIGURES = ('satellite_dropout_source_overview',
                  'satellite_dropout_source_lowurban_strata',
                  'satellite_dropout_source_training')
FINAL_FIGURES = SOURCE_FIGURES + ('satellite_dropout_formal_views',
                                'satellite_dropout_formal_paired_deltas',
                                'satellite_dropout_formal_RX')
MANIFEST = 'collection_manifest.json'


def member_names(stage):
    if stage not in ('source', 'final'):
        raise ValueError('Expected explicit source or final collection stage')
    names = list(SOURCE_FILES)
    names += [f'source_figures/{base}.{ext}'
              for base in SOURCE_FIGURES for ext in ('png', 'pdf')]
    if stage == 'final':
        names += list(FINAL_FILES)
        names += [f'final_figures/{base}.{ext}'
                  for base in FINAL_FIGURES for ext in ('png', 'pdf')]
    return names


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def inspect_readiness(source, stage):
    """Only saved aggregate metadata and the selected public files are opened."""
    def read(name):
        return json.loads((source / name).read_text(encoding='utf-8-sig'))
    src = read('source_complete.json')
    if src.get('status') != 'SOURCE_ANALYZED' or src.get('target_scores_read') is not False:
        raise ValueError('Expected source-only analysis completion metadata')
    if stage == 'final':
        final = read('complete.json')
        if (final.get('status') != 'ANALYZED'
                or final.get('source_complete') is not True
                or final.get('formal_complete') is not True
                or final.get('training_run') != RUN):
            raise ValueError('Expected complete saved formal analysis metadata')
    names = member_names(stage)
    for name in names:
        item = source / name
        if item.is_symlink() or not item.is_file() or item.stat().st_size == 0:
            raise ValueError('Required public aggregate is absent, empty or a link: ' + name)
    return dict(status='FINAL_AGGREGATES_COMPLETE' if stage == 'final'
                else 'SOURCE_AGGREGATES_COMPLETE', stage=stage, training_run=RUN,
                source_complete=True, formal_complete=(stage == 'final'),
                filenames=names, file_count=len(names), model_NN=0, IQ_reads=0,
                checkpoint_reads=0, target_truth_reads=0, training_processes_untouched=True)


def pack_public(source, archive, stage):
    """Writes one new collection archive; does not recursively inspect the run."""
    manifest = inspect_readiness(source, stage)
    with archive.open('xb') as output:
        with tarfile.open(fileobj=output, mode='w:gz') as tar:
            for name in manifest['filenames']:
                item = source / name
                info = tar.gettarinfo(str(item), arcname=name)
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                with item.open('rb') as stream:
                    tar.addfile(info, stream)
            payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf8')
            info = tarfile.TarInfo(MANIFEST)
            info.size = len(payload)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(payload))
    # Exactly one archive digest; no member hashes, model seals or training rehash.
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return dict(manifest, archive_SHA256=digest)


def extract_public(archive, output, stage, expected_sha):
    """Checks one transferred archive digest and validates paths before extraction."""
    if output.exists():
        raise FileExistsError('Existing aggregate output; reconcile without overwriting')
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('Single collection archive transfer SHA256 mismatch')
    expected = set(member_names(stage)) | {MANIFEST}
    with tarfile.open(archive, 'r:gz') as tar:
        items = tar.getmembers()
        names = [m.name for m in items]
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError('Archive differs from explicit public aggregate whitelist')
        for member in items:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts
                    or '\\' in member.name or ':' in member.name):
                raise ValueError('Invalid collection archive member')
        manifest = json.load(tar.extractfile(MANIFEST))
        if (manifest.get('stage') != stage or manifest.get('training_run') != RUN
                or set(manifest.get('filenames', [])) != expected - {MANIFEST}
                or manifest.get('formal_complete') != (stage == 'final')):
            raise ValueError('Saved collection stage or manifest differs')
        output.mkdir()
        for member in items:
            target = output / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as dest:
                dest.write(tar.extractfile(member).read())
    # Readback of declared aggregate metadata, without repeating the archive hash.
    verified = inspect_readiness(output, stage)
    return dict(verified, status='VERIFIED', archive_SHA256=expected_sha,
                source_only=(stage == 'source'), formal_results_collected=(stage == 'final'))


def remote_script(config):
    """Return a self-contained collector; nothing from the training release imports."""
    # Reuse the exact local whitelist/metadata/archive routines. This transfers only
    # source text over stdin; no installed training module or model is executed.
    import inspect
    definitions = '\n\n'.join(inspect.getsource(fn) for fn in
                             (member_names, inspect_readiness, pack_public))
    constants = '\n'.join(f'{key} = {globals()[key]!r}' for key in
                          ('RUN', 'SOURCE_FILES', 'FINAL_FILES', 'SOURCE_FIGURES',
                           'FINAL_FIGURES', 'MANIFEST'))
    return ('import hashlib,io,json,os,subprocess,tarfile\nfrom pathlib import Path\n'
            + constants + '\n\n' + definitions + '\n\n' + f'c={config!r}\n' + r'''
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!=c['user'] or subprocess.check_output(['hostname'],text=True).strip()!=c['host']:
    raise ValueError('Expected configured ordinary account')
source=Path(c['source']);exports=Path(c['exports']);archive=exports/(c['export_id']+'.tar.gz')
if archive.exists():
    raise FileExistsError('Existing collection archive; reconcile before resubmission')
# Readiness precedes even creation of the export directory.
inspect_readiness(source,c['stage'])
exports.mkdir(exist_ok=True)
result=pack_public(source,archive,c['stage'])
print(json.dumps(result))
''')


def collect(runtime_path, local_root, stage, export_id, command=subprocess.run):
    """Explicit collection invocation; tests inject a mocked transport command."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', export_id):
        raise ValueError('Use a short simple immutable export identifier')
    member_names(stage)
    if local_root.exists():
        raise FileExistsError('Local collection root already exists; no overwrite')
    runtime = json.loads(runtime_path.read_text(encoding='utf-8-sig'))
    config = dict(user=runtime['user'], host=runtime['host'],
                  source=runtime['project'] + '/runs/' + RUN + '/analysis_auto_r01',
                  exports=runtime['project'] + '/runs/' + RUN + '/analysis_auto_r01-exports',
                  stage=stage, export_id=export_id)
    # Store no private account, absolute remote path, command or runtime payload.
    local_root.mkdir(parents=True)
    state = dict(status='UNKNOWN', stage=stage, training_run=RUN,
                 export_id=export_id, model_NN=0, IQ_reads=0, checkpoint_reads=0,
                 target_truth_reads=0, training_processes_untouched=True,
                 automatic_retry=False, started_at=time.time())
    write_json(local_root / 'collection_state.json', state)
    layer = 'remote_packaging'
    try:
        result = command(['ssh', '-F', runtime['ssh_config'], '-o', 'BatchMode=yes',
                          '-o', 'ConnectTimeout=10', '-T', 'N607', runtime['python'] + ' -'],
                         input=remote_script(config).encode('utf8'), capture_output=True,
                         timeout=55, check=True)
        remote = json.loads(result.stdout)
        if (remote.get('stage') != stage or remote.get('training_run') != RUN
                or not re.fullmatch(r'[0-9a-f]{64}', remote.get('archive_SHA256', ''))):
            raise ValueError('Remote collection metadata differs')
        write_json(local_root / 'remote_collection.json', remote)
        layer = 'archive_transfer'
        archive = local_root / (export_id + '.tar.gz')
        command(['scp', '-F', runtime['ssh_config'], '-o', 'BatchMode=yes',
                 '-o', 'ConnectTimeout=10',
                 'N607:' + config['exports'] + '/' + export_id + '.tar.gz', str(archive)],
                capture_output=True, timeout=55, check=True)
        layer = 'local_validation'
        verified = extract_public(archive, local_root / 'aggregates', stage,
                                  remote['archive_SHA256'])
        state.update(verified, export_id=export_id, finished_at=time.time())
        write_json(local_root / 'collection_state.json', state)
        return state
    except BaseException as exc:
        # A failed/expired SSH observation can follow a completed remote archive.
        # Preserve the export ID, never delete or automatically retry it.
        state.update(status='FAILED' if layer == 'local_validation' else 'UNKNOWN',
                     failed_layer=layer, error_type=type(exc).__name__,
                     finished_at=time.time(),
                     next_action='Reconcile the same export and local artifacts; no automatic retry')
        write_json(local_root / 'collection_state.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--local-root', required=True, type=Path)
    parser.add_argument('--stage', required=True, choices=('source', 'final'))
    parser.add_argument('--export-id', required=True)
    args = parser.parse_args()
    result = collect(args.runtime, args.local_root, args.stage, args.export_id)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
