"""Project file inventory and navigation. Never deletes or moves original assets."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

SOURCE = {'.py', '.sh', '.ps1', '.bat', '.cmd', '.cpp', '.h', '.cu', '.ipynb'}
DOCUMENT = {'.md', '.pdf', '.docx', '.pptx', '.tex', '.bib', '.html'}
DATA = {'.pkl', '.pickle', '.h5', '.hdf5', '.mat', '.npy', '.npz', '.parquet', '.wav', '.bin'}
WEIGHT = {'.pth', '.pt', '.ckpt', '.safetensors', '.onnx'}


def classify(rel):
    p = Path(rel)
    parts = p.parts
    low = rel.lower().replace('\\', '/')
    if '__pycache__' in parts or p.suffix.lower() in {'.pyc', '.pyo'}:
        return 'regenerable_python_cache'
    if '.pytest_cache' in parts:
        return 'regenerable_pytest_cache'
    if any(x.startswith('.pytest_tmp') or x.startswith('.pytest_verify') for x in parts):
        return 'historical_test_fixture'
    if p.suffix.lower() in WEIGHT:
        return 'protected_checkpoint'
    if p.suffix.lower() in DATA:
        return 'protected_data_or_prediction'
    if 'experiment_registry/' in low or '/project_governance/' in low:
        return 'governance_metadata'
    if low.endswith(('.tar', '.tar.gz', '.tgz', '.zip', '.7z', '.gz')):
        return 'archive_or_release'
    if p.suffix.lower() in SOURCE:
        if 'test' in p.name.lower() or 'smoke' in p.name.lower():
            return 'test_or_smoke_source'
        if 'tmp' in p.name.lower() or parts[0] in {'.codex_tmp', 'tmp', 'analysis_tmp'}:
            return 'oneoff_source_keep'
        return 'source_keep'
    if p.suffix.lower() in DOCUMENT:
        return 'report_or_reference_keep'
    if p.suffix.lower() in {'.json', '.jsonl', '.yaml', '.yml', '.toml', '.ini', '.cfg'}:
        return 'config_or_structured_evidence'
    if p.suffix.lower() in {'.log', '.csv', '.out', '.err', '.stdout', '.stderr', '.xml', '.txt'}:
        return 'log_or_evidence_keep'
    return 'other_review'


def iso():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def inventory(root, database, report_dir):
    root = root.resolve()
    database = database.resolve()
    report_dir = report_dir.resolve()
    if database.exists():
        raise ValueError('Inventory database already exists; use a new scan path')
    database.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(database)
    db.execute('CREATE TABLE files(path TEXT PRIMARY KEY, bytes INTEGER, mtime REAL, category TEXT, zone TEXT)')
    db.execute('CREATE TABLE boundaries(path TEXT, kind TEXT, reason TEXT)')
    summary = defaultdict(lambda: {'files': 0, 'bytes': 0, 'categories': Counter()})
    errors, boundaries, rows = [], [], []
    count = 0
    def walk(folder):
        nonlocal count
        try:
            entries = sorted(os.scandir(folder), key=lambda e: e.name.casefold())
        except OSError as exc:
            errors.append({'path': str(folder), 'error': str(exc)})
            return
        for entry in entries:
            p = Path(entry.path)
            rel = p.relative_to(root).as_posix()
            try:
                stat = entry.stat(follow_symlinks=False)
                # Reparse points can escape the workspace even when is_symlink() is false.
                if getattr(stat, 'st_file_attributes', 0) & 0x400 or entry.is_symlink():
                    boundaries.append((rel, 'reparse_point', 'record_only_no_follow'))
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if entry.name == '.git':
                        boundaries.append((rel, 'git_internal', 'git_ownership_recorded_separately'))
                        continue
                    if p == database.parent or p == report_dir:
                        boundaries.append((rel, 'current_scan_output', 'exclude_self'))
                        continue
                    yield from walk(p)
                else:
                    category = classify(rel)
                    zone = rel.split('/')[0] if '/' in rel else '[root files]'
                    count += 1
                    summary[zone]['files'] += 1
                    summary[zone]['bytes'] += stat.st_size
                    summary[zone]['categories'][category] += 1
                    yield (rel, stat.st_size, stat.st_mtime, category, zone)
            except OSError as exc:
                errors.append({'path': rel, 'error': str(exc)})
    for row in walk(root):
        rows.append(row)
        if len(rows) == 10000:
            db.executemany('INSERT INTO files VALUES(?,?,?,?,?)', rows)
            db.commit()
            rows.clear()
            if count % 100000 == 0:
                print(f'inventoried {count} files', file=sys.stderr, flush=True)
    db.executemany('INSERT INTO files VALUES(?,?,?,?,?)', rows)
    db.executemany('INSERT INTO boundaries VALUES(?,?,?)', boundaries)
    db.execute('CREATE INDEX files_category ON files(category)')
    db.execute('CREATE INDEX files_zone ON files(zone)')
    db.commit()
    result = {'schema': 'project_file_inventory_v1', 'observed_at': iso(), 'root': str(root),
              'database': str(database.resolve()), 'files': count, 'logical_bytes': sum(x['bytes'] for x in summary.values()),
              'zones': dict(sorted(summary.items(), key=lambda x: -x[1]['bytes'])),
              'errors': errors, 'boundaries': [{'path': p, 'kind': k, 'reason': r} for p,k,r in boundaries],
              'scope': 'workspace files including snapshot/worktree contents; .git internals and reparse targets are boundaries; no remote mutation'}
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / 'inventory_summary.json', result)
    # Compact versionable inventory; runtime SQLite stays outside Git.
    with gzip.open(report_dir / 'files.csv.gz', 'wt', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['path','bytes','mtime','category','zone'])
        writer.writerows(db.execute('SELECT * FROM files ORDER BY path'))
    with (report_dir / 'directory_map.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['zone', 'files', 'logical_bytes'])
        writer.writerows((z, v['files'], v['bytes']) for z,v in result['zones'].items())
    db.close()
    return result


def find(database, query, category=None, limit=30):
    db = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    where = 'path LIKE ?'
    args = ['%' + query + '%']
    if category:
        where += ' AND category=?'
        args.append(category)
    total = db.execute('SELECT count(*) FROM files WHERE ' + where, args).fetchone()[0]
    rows = [dict(r) for r in db.execute('SELECT * FROM files WHERE ' + where + ' ORDER BY path LIMIT ?', args + [limit])]
    db.close()
    return {'matched': total, 'shown': rows, 'boundary': 'inventory snapshot; check current existence before acting'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest='command', required=True)
    scan = commands.add_parser('scan')
    scan.add_argument('--database', type=Path, required=True)
    scan.add_argument('--report-dir', type=Path, required=True)
    query = commands.add_parser('find')
    query.add_argument('query')
    query.add_argument('--database', type=Path)
    query.add_argument('--category')
    query.add_argument('--limit', type=int, default=30)
    old = commands.add_parser('legacy')
    old.add_argument('legacy_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == 'scan':
        result = inventory(args.root, args.database, args.report_dir)
        print(json.dumps({k: result[k] for k in ['files', 'logical_bytes', 'database']}, ensure_ascii=False))
    elif args.command == 'find':
        database = args.database
        if database is None:
            database = Path(json.loads((args.root / 'docs/project_governance/current.json').read_text(encoding='utf-8'))['database'])
        print(json.dumps(find(database, args.query, args.category, args.limit), ensure_ascii=False, indent=2))
    else:
        if not args.legacy_args or args.legacy_args[0] not in {'status','find','experiment','repo','review'}:
            raise SystemExit('Legacy bridge only permits read-only status/find/experiment/repo/review')
        base = args.root / 'code/snapshots/project_governance_20260813_wt'
        result = subprocess.run([sys.executable, '-X', 'utf8', str(base / 'tools/project_governance_inventory.py'),
                                 *args.legacy_args, '--latest', str(base / 'docs/project_governance/latest.json'), '--json'])
        return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
