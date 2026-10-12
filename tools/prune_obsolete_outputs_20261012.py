"""Delete authorized, unreferenced pre-September disposable output files.

Keep source, datasets, all final/best weights, numerical NPZ states, experiment
reports, registration/configuration, metrics, and unique failure evidence.
"""
import argparse
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import re
import stat
import sys

from cleanup_historical_logs_20261012 import boundary, signature, assert_selected_inactive, write_json

CUTOFF = datetime.fromisoformat('2026-09-01T00:00:00+08:00').timestamp()
DATE = re.compile(r'2026[01][0-9][0-3][0-9]')
PRESERVE_NAMES = re.compile(r'(?i)(score|metric|summary|report|manifest|contract|config|provenance|completion|evaluation|initialization|experiment|events|status|truth|label|physical|split|capsule|protocol)')
FAILURE = re.compile(r'(?i)(?:status|状态)[^\n]{0,80}(?:FAILED|TECHNICAL_FAILURE|UNKNOWN|未完成|失败|未闭合)')


def old_name(name):
    dates = DATE.findall(name)
    return bool(dates) and max(dates) < '20260901'


def collect_references(root, output):
    root = Path(root)
    tokens = set()
    sources = []
    for zone in ['docs', 'configs', 'tools', 'experiments', 'automation_reports/CV-SincNet']:
        start = root / zone
        for folder, dirs, files in os.walk(start, followlinks=False):
            dirs[:] = [name for name in dirs if name not in {'.git', '__pycache__', 'results', 'artifacts', 'remote_artifacts', 'details', 'evidence', 'data', 'target_truth', 'target_inputs'}
                       and not (Path(folder) / name).is_symlink()]
            if zone.startswith('automation_reports') and Path(folder) != start:
                top = Path(folder).relative_to(start).parts[0]
                if old_name(top):
                    dirs[:] = []
                    continue
            for name in files:
                path = Path(folder) / name
                if path.suffix.lower() not in {'.md', '.json', '.yaml', '.yml', '.py', '.sh'}:
                    continue
                if 'project_governance' in path.parts or path.stat().st_size > 2 * 1024 * 1024:
                    continue
                if zone in {'docs', 'configs'} and any(old_name(part) for part in path.relative_to(start).parts):
                    continue
                try:
                    boundary(root, path)
                    text = path.read_text(encoding='utf-8-sig')
                    found = re.findall(r'[A-Za-z0-9_.-]*20260[5-8][0-3][0-9][A-Za-z0-9_.-]*', text)
                    tokens.update(found)
                    sources.append(path.relative_to(root).as_posix())
                except (OSError, UnicodeError, ValueError):
                    continue
    value = {'cutoff': '2026-09-01T00:00:00+08:00', 'tokens': sorted(tokens), 'reference_sources': sources}
    write_json(output, value)
    print(json.dumps({'sources': len(sources), 'reference_tokens': len(tokens)}))


def disposable(path):
    name = path.name
    lower = name.lower()
    suffix = path.suffix.lower()
    parts = {part.lower() for part in path.parts}
    if any(part in parts for part in {'data', 'datasets', 'dataset_wigsig', 'target_inputs', 'target_truth', 'source_inputs', 'received', 'capsules'}):
        return None
    if suffix == '.gz':
        lower = Path(name[:-3]).name.lower()
        suffix = Path(lower).suffix
    if PRESERVE_NAMES.search(lower):
        return None
    if suffix in {'.log', '.out', '.err', '.stdout', '.stderr'}:
        return 'expired_unreferenced_log'
    if suffix in {'.jsonl', '.csv'} and re.search(r'(?:fit_trace|solver_sweeps|training_events|training_objectives|fit_stages|train_steps|training_steps|source_steps)', lower):
        return 'expired_unreferenced_training_trace'
    if suffix in {'.pth', '.pt', '.ckpt'} and re.search(r'(?:^|_)(?:epoch|step|iter)[_-]?\d+', lower):
        retained = [p for p in path.parent.iterdir() if p.is_file() and p != path and
                    p.suffix.lower() in {'.pth', '.pt', '.ckpt'} and re.search(r'(?:final|best|selected)', p.name, re.I)]
        if retained:
            return 'intermediate_weight_with_final_best_retained'
    if suffix in {'.json', '.csv', '.jsonl'} and parts.intersection({'detail', 'raw', 'per_query', 'per_sample'}):
        return 'expired_unreferenced_detailed_output'
    if lower in {'predictions.json', 'prediction.json'} and any(p.name.lower().startswith(('score', 'metrics')) and p.is_file() for p in path.parent.iterdir()):
        return 'expired_prediction_copy_with_scores_retained'
    return None


def has_failure_evidence(path):
    markers = [b'Traceback (most recent call last):', b'RuntimeError:',
               b'CUDA out of memory', b'PermissionError:', b'OSError:']
    opener = gzip.open if path.suffix.lower() == '.gz' else open
    previous = b''
    with opener(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(512 * 1024), b''):
            chunk = previous + chunk
            if b'\x00' in chunk or any(marker in chunk for marker in markers):
                return True
            previous = chunk[-128:]
    return False


def make_plan(root, site, references):
    root = Path(root).resolve(strict=True)
    tokens = references['tokens']
    zones = ['runs', 'logs', 'logs/cvs/current', 'releases'] if site == 'remote' else ['automation_reports/CV-SincNet', 'local_artifacts', 'runs', 'logs', 'logs/cvs/current', 'server_log_backups']
    candidates, protected, errors = [], [], []
    for zone in zones:
        start = root / zone
        if not start.is_dir() or start.is_symlink():
            continue
        for run in start.iterdir():
            if not run.is_dir() or not old_name(run.name):
                continue
            try:
                boundary(root, run)
            except (OSError, ValueError):
                continue
            if any(run.name in token for token in tokens):
                protected.append({'path': run.relative_to(root).as_posix(), 'reason': 'current source/config/report reference'})
                continue
            failure = False
            for report in run.glob('*.md'):
                if report.stat().st_size < 2 * 1024 * 1024 and FAILURE.search(report.read_text(encoding='utf-8-sig', errors='replace')):
                    failure = True
                    break
            if failure:
                protected.append({'path': run.relative_to(root).as_posix(), 'reason': 'failure/unresolved evidence retained'})
                continue
            for folder, dirs, files in os.walk(run, followlinks=False):
                dirs[:] = [name for name in dirs if name != '.git' and not (Path(folder) / name / '.git').exists()
                           and not (Path(folder) / name).is_symlink()]
                for name in files:
                    path = Path(folder) / name
                    # Old releases retain every code/config file; only embedded output copies qualify.
                    if zone == 'releases' and not {'automation_reports', 'local_artifacts', 'logs', 'runs', 'outputs'}.intersection(path.relative_to(run).parts[:-1]):
                        continue
                    reason = disposable(path)
                    if reason is None:
                        continue
                    try:
                        boundary(root, path)
                        info = path.stat()
                        if not stat.S_ISREG(info.st_mode) or info.st_mtime >= CUTOFF:
                            continue
                        if reason == 'expired_unreferenced_log' and has_failure_evidence(path):
                            protected.append({'path': path.relative_to(root).as_posix(), 'reason': 'technical failure/unknown-encoding evidence retained'})
                            continue
                        candidates.append({'path': str(path), 'relative': path.relative_to(root).as_posix(), 'bytes': info.st_size,
                                           'allocated': getattr(info, 'st_blocks', 0) * 512, 'signature': signature(info), 'reason': reason})
                    except (OSError, ValueError) as exc:
                        errors.append({'path': str(path), 'reason': str(exc)})
    scopes = assert_selected_inactive(root, candidates)
    return {'schema': 1, 'site': site, 'root': str(root), 'cutoff': '2026-09-01T00:00:00+08:00',
            'files': candidates, 'total_bytes': sum(row['bytes'] for row in candidates),
            'protected': protected, 'errors': errors, 'protected_active_scopes': scopes}


def apply_plan(plan, manifest):
    root = Path(plan['root']).resolve(strict=True)
    assert_selected_inactive(root, plan['files'])
    rows, freed = [], 0
    with Path(manifest).open('x', encoding='utf-8') as output:
        for item in plan['files']:
            result = dict(item)
            try:
                path = boundary(root, item['path'])
                if signature(path.stat()) != item['signature']:
                    raise ValueError('file changed since deletion plan')
                result['status'] = 'ABOUT_TO_DELETE'
                output.write(json.dumps(result, ensure_ascii=False) + '\n')
                output.flush()
                path.unlink()
                if path.exists():
                    raise OSError('post-state still exists')
                result['status'] = 'VERIFIED'
                freed += item['bytes']
            except Exception as exc:
                result['status'] = 'FAILED'
                result['error'] = str(exc)
            rows.append(result)
            output.write(json.dumps(result, ensure_ascii=False) + '\n')
            output.flush()
            if len(rows) % 1000 == 0:
                os.fsync(output.fileno())
                print(json.dumps({'processed': len(rows), 'total': len(plan['files']), 'removed_bytes': freed}), flush=True)
        os.fsync(output.fileno())
    return {'status': 'VERIFIED' if all(row['status'] == 'VERIFIED' for row in rows) else 'PARTIAL',
            'deleted': sum(row['status'] == 'VERIFIED' for row in rows),
            'failed': sum(row['status'] != 'VERIFIED' for row in rows), 'removed_logical_bytes': freed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['references', 'plan', 'apply'])
    parser.add_argument('--root')
    parser.add_argument('--site', choices=['local', 'remote'])
    parser.add_argument('--references')
    parser.add_argument('--output', required=True)
    parser.add_argument('--plan')
    args = parser.parse_args()
    if args.mode == 'references':
        collect_references(args.root, args.output)
    elif args.mode == 'plan':
        value = make_plan(args.root, args.site, json.loads(Path(args.references).read_text(encoding='utf-8-sig')))
        write_json(args.output, value)
        print(json.dumps({'files': len(value['files']), 'bytes': value['total_bytes'], 'protected_runs': len(value['protected']),
                          'errors': value['errors'], 'top': sorted(value['files'], key=lambda row: row['bytes'], reverse=True)[:15]}, ensure_ascii=False))
    else:
        print(json.dumps(apply_plan(json.loads(Path(args.plan).read_text(encoding='utf-8-sig')), args.output)))


if __name__ == '__main__':
    main()
