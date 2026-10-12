"""Task-scoped historical log compaction; preserves weights, data and predictions.

Run plan first, inspect its exact paths, then apply the same saved plan. Each log
is gzip-compressed and completely read back before its redundant raw copy is
unlinked. This is storage maintenance, not experiment evaluation.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys

CUTOFF = datetime.fromisoformat('2026-10-10T00:00:00+08:00').timestamp()
MINIMUM = 1024 * 1024
EVENT_NAMES = {
    'fit_trace.jsonl', 'solver_sweeps.jsonl', 'training_events.jsonl',
    'training_events_compact.jsonl', 'training_events_compact.csv',
    'training_objectives.jsonl', 'training_objectives.csv',
    'fit_stages.jsonl', 'fit_stages.csv', 'source_steps.jsonl',
    'training_steps.jsonl', 'train_steps.jsonl',
}
LOCAL_ZONES = [
    'logs', 'server_log_backups', 'runs', 'outputs', 'remote_artifacts',
    'local_artifacts', 'automation_reports', 'tmp_cli_logs', 'tmp_logs',
    'tmp_dry_logs', 'tmp_runs', 'tmp_cli_runs',
]
REMOTE_ZONES = ['runs', 'logs', 'server_log_backups']


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8')


def boundary(root, path):
    """Check every component, never follow reparse points or Git internals."""
    root = Path(root).resolve(strict=True)
    path = Path(path).absolute()
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        if part == '.git':
            raise ValueError('Git internal boundary')
        current = current / part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('link/reparse boundary')
    if path.resolve(strict=True) != path:
        raise ValueError('resolved path differs')
    return path


def active_remote_paths(root):
    if sys.platform == 'win32':
        return []
    active = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            cwd = os.readlink(proc / 'cwd')
            if str(root) in cmd or cwd.startswith(str(root)):
                if not any(x in cmd for x in ['sshd:', 'sftp-server']):
                    active.append({'pid': int(proc.name), 'cwd': cwd, 'command': cmd})
        except (OSError, PermissionError):
            continue
    return active


def signature(info):
    return [info.st_size, info.st_mtime_ns, info.st_ino, info.st_dev]


def plan(root, site):
    root = Path(root).resolve(strict=True)
    items, skipped, errors = [], [], []
    zones = LOCAL_ZONES if site == 'local' else REMOTE_ZONES
    for zone in zones:
        start = root / zone
        if not start.exists():
            continue
        try:
            boundary(root, start)
        except (OSError, ValueError) as exc:
            errors.append({'path': str(start), 'reason': str(exc)})
            continue
        def scan_error(exc):
            errors.append({'path': exc.filename, 'reason': str(exc)})
        for folder, dirs, files in os.walk(start, followlinks=False, onerror=scan_error):
            keep_dirs = []
            for name in dirs:
                candidate = Path(folder) / name
                if name == '.git' or (candidate / '.git').exists():
                    continue
                try:
                    boundary(root, candidate)
                    keep_dirs.append(name)
                except (OSError, ValueError):
                    continue
            dirs[:] = keep_dirs
            for name in files:
                candidate = Path(folder) / name
                if candidate.suffix.lower() not in {'.log', '.out', '.err', '.stdout', '.stderr'} and name not in EVENT_NAMES:
                    continue
                try:
                    candidate = boundary(root, candidate)
                    info = candidate.stat()
                    if not stat.S_ISREG(info.st_mode) or info.st_size < MINIMUM or info.st_mtime >= CUTOFF:
                        continue
                    if info.st_nlink != 1 or Path(str(candidate) + '.gz').exists():
                        skipped.append({'path': str(candidate), 'reason': 'existing archive or hard links'})
                        continue
                    items.append({'path': str(candidate), 'relative': candidate.relative_to(root).as_posix(),
                                  'signature': signature(info), 'bytes': info.st_size,
                                  'allocated': getattr(info, 'st_blocks', 0) * 512,
                                  'archive': str(candidate) + '.gz'})
                except (OSError, ValueError) as exc:
                    errors.append({'path': str(candidate), 'reason': str(exc)})
    return {'schema': 1, 'site': site, 'root': str(root), 'cutoff': '2026-10-10T00:00:00+08:00',
            'minimum_bytes': MINIMUM, 'policy': 'full gzip readback before removing redundant raw log; no datasets, weights, predictions, metrics or source changes',
            'files': sorted(items, key=lambda x: x['bytes'], reverse=True),
            'skipped': skipped, 'errors': errors, 'active_paths': active_remote_paths(root),
            'total_bytes': sum(x['bytes'] for x in items)}


def digest_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def apply(saved_plan, manifest):
    root = Path(saved_plan['root']).resolve(strict=True)
    active = active_remote_paths(root)
    if active:
        raise RuntimeError('Active project process detected; no mutation: ' + json.dumps(active))
    journal = Path(manifest)
    journal.parent.mkdir(parents=True, exist_ok=True)
    if journal.exists():
        raise FileExistsError('Use a new manifest, reconcile existing outcomes first')
    reclaimed, done, failed = 0, 0, 0
    with journal.open('x', encoding='utf-8') as records:
        for item in saved_plan['files']:
            result = dict(item)
            archive = Path(item['archive'])
            temporary = Path(item['archive'] + '.cleanup-partial')
            try:
                source = boundary(root, item['path'])
                before = source.stat()
                if signature(before) != item['signature'] or before.st_nlink != 1:
                    raise ValueError('source changed since plan')
                if archive.exists() or temporary.exists():
                    raise FileExistsError('archive/partial already exists; original preserved')
                original_digest = hashlib.sha256()
                with source.open('rb') as inp, temporary.open('xb') as raw:
                    with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=1, mtime=int(before.st_mtime)) as output:
                        for chunk in iter(lambda: inp.read(4 * 1024 * 1024), b''):
                            output.write(chunk)
                            original_digest.update(chunk)
                    raw.flush()
                    os.fsync(raw.fileno())
                with gzip.open(temporary, 'rb') as verified:
                    if digest_stream(verified) != original_digest.hexdigest():
                        raise ValueError('complete readback differs; original preserved')
                if signature(source.stat()) != signature(before):
                    raise ValueError('source changed while archiving; original preserved')
                # Resolve the actual source again immediately before unlink.
                boundary(root, source)
                os.chmod(temporary, stat.S_IMODE(before.st_mode))
                os.utime(temporary, ns=(before.st_atime_ns, before.st_mtime_ns))
                temporary.rename(archive)
                archive_info = archive.stat()
                result.update({'sha256_uncompressed': original_digest.hexdigest(),
                               'archive_bytes': archive_info.st_size,
                               'archive_allocated': getattr(archive_info, 'st_blocks', 0) * 512,
                               'status': 'ARCHIVE_VERIFIED_RAW_PRESENT'})
                records.write(json.dumps(result, ensure_ascii=False) + '\n')
                records.flush()
                os.fsync(records.fileno())
                source.unlink()
                if source.exists() or not archive.is_file():
                    raise ValueError('post-state does not match')
                result['status'] = 'VERIFIED'
                reclaimed += before.st_size - archive_info.st_size
                done += 1
                print(json.dumps({'done': done, 'total': len(saved_plan['files']),
                                  'reclaimed_logical_bytes': reclaimed,
                                  'path': item['relative']}, ensure_ascii=False), flush=True)
            except Exception as exc:
                result.update({'status': 'FAILED', 'error': str(exc),
                               'original_exists': Path(item['path']).exists(),
                               'archive_exists': archive.exists(), 'partial_exists': temporary.exists()})
                failed += 1
            records.write(json.dumps(result, ensure_ascii=False) + '\n')
            records.flush()
            os.fsync(records.fileno())
    return {'status': 'VERIFIED' if failed == 0 else 'PARTIAL', 'completed': done,
            'failed': failed, 'reclaimed_logical_bytes': reclaimed, 'manifest': str(journal)}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--root', required=True)
    p.add_argument('--site', choices=['local', 'remote'], required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('apply')
    p.add_argument('--plan', required=True)
    p.add_argument('--manifest', required=True)
    args = parser.parse_args()
    if args.mode == 'plan':
        value = plan(args.root, args.site)
        write_json(args.output, value)
        print(json.dumps({k: v for k, v in value.items() if k not in {'files', 'skipped'}}, ensure_ascii=False))
        print(json.dumps({'files': len(value['files']), 'top': value['files'][:12]}, ensure_ascii=False))
    else:
        value = json.loads(Path(args.plan).read_text(encoding='utf-8-sig'))
        print(json.dumps(apply(value, args.manifest), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
