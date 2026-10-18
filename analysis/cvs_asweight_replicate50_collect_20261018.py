"""Collect only saved public AsWeight E50 aggregates, using one archive transfer.

Never reads source IQ, checkpoints, prediction NPZ/IDs, truth or original logs.
No output overwrite or automatic retry is permitted after an ambiguous transport.
"""
import argparse
import hashlib
import inspect
import io
import json
import re
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath

RUN = '20261018-phase1-FcrossRX-Asweight-e50-s0203-m4-r01'
OUTPUT = 'analysis_auto_r01'
SOURCE_FILES = ('source_complete.json', 'source_summary.json',
                'source_confusions.json', 'source_metrics.csv', 'training_logs.json')
FINAL_FILES = ('formal_summary.json', 'formal_confusions.json',
               'formal_metrics.csv', 'final_report.md', 'costs.json', 'complete.json')
MANIFEST = 'collection_manifest.json'


def validate_output_name(output):
    if re.fullmatch(r'analysis_auto_r[0-9]{2}', output) is None:
        raise ValueError('Expected own immutable analysis output basename')
    return output


def required_names(stage):
    if stage not in ('source', 'final'):
        raise ValueError('Expected explicit source or final collection stage')
    return list(SOURCE_FILES) + (list(FINAL_FILES) if stage == 'final' else [])


def valid_public_name(name, stage):
    if name in required_names(stage):
        return True
    path = PurePosixPath(name)
    allowed = ('source_figures', 'final_figures') if stage == 'final' else ('source_figures',)
    return (len(path.parts) == 2 and path.parts[0] in allowed
            and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,119}\.(png|pdf)', path.parts[1]) is not None)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def member_names(source, stage):
    names = required_names(stage)
    folders = ('source_figures', 'final_figures') if stage == 'final' else ('source_figures',)
    for folder in folders:
        directory = source / folder
        if directory.is_symlink():
            raise ValueError('Optional figure directory is a link')
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise ValueError('Optional figure path is not a directory')
        for item in sorted(directory.iterdir()):
            name = folder + '/' + item.name
            if item.suffix.lower() not in ('.png', '.pdf'):
                continue
            if not valid_public_name(name, stage):
                raise ValueError('Unexpected public figure name')
            names.append(name)
    return names


def inspect_readiness(source, stage):
    def read(name):
        item = source / name
        if item.is_symlink() or not item.is_file():
            raise ValueError('Required aggregate metadata absent or a link: ' + name)
        return json.loads(item.read_text(encoding='utf-8-sig'))
    src = read('source_complete.json')
    if src.get('status') != 'SOURCE_ANALYZED' or src.get('target_scores_read') is not False:
        raise ValueError('Expected source-only completion metadata')
    if src.get('training_run', RUN) != RUN:
        raise ValueError('Source completion run differs')
    if stage == 'final':
        final = read('complete.json')
        if (final.get('status') != 'ANALYZED'
                or final.get('source_complete') is not True
                or final.get('formal_complete') is not True
                or final.get('log_records_complete') is not True
                or final.get('training_run') != RUN or final.get('NN') != 0):
            raise ValueError('Expected complete saved formal analysis metadata')
    names = member_names(source, stage)
    for name in names:
        item = source / name
        if item.is_symlink() or not item.is_file() or item.stat().st_size == 0:
            raise ValueError('Public aggregate is absent, empty or a link: ' + name)
    return dict(status='FINAL_AGGREGATES_COMPLETE' if stage == 'final' else 'SOURCE_AGGREGATES_COMPLETE',
                stage=stage, training_run=RUN, source_complete=True,
                formal_complete=(stage == 'final'), filenames=names, file_count=len(names),
                model_NN=0, IQ_reads=0, checkpoint_reads=0, target_truth_reads=0,
                training_processes_untouched=True)


def pack_public(source, archive, stage):
    manifest = inspect_readiness(source, stage)
    with archive.open('xb') as output, tarfile.open(fileobj=output, mode='w:gz') as tar:
        for name in manifest['filenames']:
            item = source / name
            info = tar.gettarinfo(str(item), arcname=name)
            info.mode = 0o644; info.uid = info.gid = 0; info.uname = info.gname = ''
            with item.open('rb') as stream:
                tar.addfile(info, stream)
        payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf8')
        info = tarfile.TarInfo(MANIFEST); info.size = len(payload); info.mode = 0o644
        tar.addfile(info, io.BytesIO(payload))
    return dict(manifest, archive_SHA256=hashlib.sha256(archive.read_bytes()).hexdigest())


def extract_public(archive, output, stage, expected_sha):
    if output.exists():
        raise FileExistsError('Existing aggregate output; reconcile without overwriting')
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('Single collection archive transfer SHA256 mismatch')
    with tarfile.open(archive, 'r:gz') as tar:
        items = tar.getmembers(); names = [item.name for item in items]
        if len(names) != len(set(names)) or MANIFEST not in names:
            raise ValueError('Archive duplicate member or missing manifest')
        for member in items:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts
                    or '\\' in member.name or ':' in member.name
                    or (member.name != MANIFEST and not valid_public_name(member.name, stage))):
                raise ValueError('Invalid or private collection archive member')
        manifest = json.load(tar.extractfile(MANIFEST))
        listed = manifest.get('filenames', [])
        if (manifest.get('stage') != stage or manifest.get('training_run') != RUN
                or not isinstance(listed, list) or len(listed) != len(set(listed))
                or set(listed) != set(names) - {MANIFEST}
                or not set(required_names(stage)).issubset(listed)
                or manifest.get('formal_complete') != (stage == 'final')):
            raise ValueError('Saved collection stage or manifest differs')
        output.mkdir()
        for member in items:
            target = output / member.name; target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as dest:
                dest.write(tar.extractfile(member).read())
    verified = inspect_readiness(output, stage)
    return dict(verified, status='VERIFIED', archive_SHA256=expected_sha,
                source_only=(stage == 'source'), formal_results_collected=(stage == 'final'))


def remote_script(config):
    definitions = '\n\n'.join(inspect.getsource(fn) for fn in
        (required_names, valid_public_name, member_names, inspect_readiness, pack_public))
    constants = '\n'.join(f'{key} = {globals()[key]!r}' for key in
                          ('RUN', 'SOURCE_FILES', 'FINAL_FILES', 'MANIFEST'))
    return ('import hashlib,io,json,os,re,subprocess,tarfile\nfrom pathlib import Path,PurePosixPath\n'
            + constants + '\n\n' + definitions + '\n\n' + f'c={config!r}\n' + r'''
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!=c['user'] or subprocess.check_output(['hostname'],text=True).strip()!=c['host']:
    raise ValueError('Expected configured ordinary account')
source=Path(c['source']); exports=Path(c['exports']); archive=exports/(c['export_id']+'.tar.gz')
if archive.exists(): raise FileExistsError('Existing collection archive; reconcile before resubmission')
inspect_readiness(source,c['stage'])
exports.mkdir(exist_ok=True)
print(json.dumps(pack_public(source,archive,c['stage'])))
''')


def collect(runtime_path, local_root, stage, export_id, output_name=OUTPUT, command=subprocess.run):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', export_id):
        raise ValueError('Use a short immutable export identifier')
    validate_output_name(output_name); required_names(stage)
    if local_root.exists():
        raise FileExistsError('Local collection root already exists; no overwrite')
    runtime = json.loads(runtime_path.read_text(encoding='utf-8-sig'))
    source = runtime['project'] + '/runs/' + RUN + '/' + output_name
    config = dict(user=runtime['user'], host=runtime['host'], source=source,
                  exports=source + '-exports', stage=stage, export_id=export_id)
    local_root.mkdir(parents=True)
    state = dict(status='UNKNOWN', stage=stage, training_run=RUN, output_name=output_name,
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
        layer = 'archive_transfer'; archive = local_root / (export_id + '.tar.gz')
        command(['scp', '-F', runtime['ssh_config'], '-o', 'BatchMode=yes', '-o',
                 'ConnectTimeout=10', 'N607:' + config['exports'] + '/' + export_id + '.tar.gz', str(archive)],
                capture_output=True, timeout=55, check=True)
        layer = 'local_validation'
        verified = extract_public(archive, local_root / 'aggregates', stage, remote['archive_SHA256'])
        state.update(verified, export_id=export_id, output_name=output_name, finished_at=time.time())
        write_json(local_root / 'collection_state.json', state)
        return state
    except BaseException as exc:
        state.update(status='FAILED' if layer == 'local_validation' else 'UNKNOWN',
                     failed_layer=layer, error_type=type(exc).__name__, finished_at=time.time(),
                     next_action='Reconcile the same export and local artifacts; no automatic retry')
        write_json(local_root / 'collection_state.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--local-root', required=True, type=Path)
    parser.add_argument('--stage', required=True, choices=('source', 'final'))
    parser.add_argument('--export-id', required=True)
    parser.add_argument('--output-name', default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(collect(args.runtime, args.local_root, args.stage, args.export_id, args.output_name), ensure_ascii=False))


if __name__ == '__main__':
    main()
