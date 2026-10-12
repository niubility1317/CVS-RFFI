"""Retire obsolete pre-September locators, preserving compact summaries."""
from collections import Counter
import csv
import gzip
import io
import json
from pathlib import Path
import sys

import experiment_registry as registry
from prune_obsolete_outputs_20261012 import old_name
from cleanup_historical_logs_20261012 import boundary


def selection(records, tokens):
    retired, keep = [], []
    for record in records:
        path = str(record.get('path') or record['id']).replace('\\', '/')
        anchors = [part for part in path.split('/') if old_name(part)]
        protected = any(anchor in token for anchor in anchors for token in tokens)
        statuses = ' '.join(record.get('reported_statuses', []))
        failed = any(word in statuses.upper() for word in ['FAILED', 'TECHNICAL_FAILURE', 'UNKNOWN'])
        if record['record_type'] != 'managed_run' and old_name(path) and not protected and not failed:
            retired.append(record)
        else:
            keep.append(record)
    return retired, keep


def relative_locator(value):
    value = str(value or '').replace('\\', '/')
    if '/CV-SincNet/' in value:
        return 'N607:CV-SincNet/' + value.split('/CV-SincNet/', 1)[1]
    if value.startswith('E:/type10-7/'):
        return value[len('E:/type10-7/'):]
    return value


def run(root, references, mode, evidence, render=True):
    root = Path(root).resolve(strict=True)
    destination = root / 'experiment_registry'
    evidence = Path(evidence)
    tokens = json.loads(Path(references).read_text(encoding='utf-8-sig'))['tokens']
    with registry.lock(destination / '.build.lock'):
        records = [json.loads(line) for line in (destination / 'catalog.jsonl').read_text(encoding='utf-8-sig').splitlines()]
        retired, kept = selection(records, tokens)
        detail_files = {record.get(section + '_file') for record in retired for section in ['facts', 'artifacts'] if record.get(section + '_file')}
        kept_details = {record.get(section + '_file') for record in kept for section in ['facts', 'artifacts'] if record.get(section + '_file')}
        detail_files -= kept_details
        detail_files = sorted(path for path in detail_files if (destination / path).is_file())
        plan = {'status': 'PLANNED', 'retired_records': len(retired), 'retained_records': len(kept),
                'detail_files': detail_files, 'detail_bytes': sum((destination / path).stat().st_size for path in detail_files),
                'retired_by_type': dict(Counter(record['record_type'] for record in retired)),
                'ids': [registry.retirement_key(record['id']) for record in retired]}
        if mode == 'preview':
            evidence.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(json.dumps({k: value for k, value in plan.items() if k not in {'ids', 'detail_files'}}, ensure_ascii=False))
            return
        summaries = []
        for record in retired:
            item = {key: record.get(key) for key in ['record_type', 'name', 'title', 'kind', 'stage', 'status', 'seeds', 'reported_statuses', 'artifact_counts']}
            item['id'] = registry.retirement_key(record['id'])
            item['original_locator'] = relative_locator(record.get('report') or record.get('path'))
            item['retired_on'] = '2026-10-12'
            item['reason'] = 'pre-20260901 obsolete locator; no current source/config/report reference; compact historical summary retained'
            summaries.append(item)
        summary_path = destination / 'retired_pre20260901_summaries.jsonl.gz'
        if summary_path.exists() or (destination / 'retired_pre20260901.json').exists():
            raise FileExistsError('Retirement already exists; reconcile before retry')
        with gzip.open(summary_path, 'wt', encoding='utf-8') as stream:
            for item in summaries:
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
        with gzip.open(summary_path, 'rt', encoding='utf-8') as stream:
            assert [json.loads(line) for line in stream] == summaries
        registry.write_text(destination / 'retired_pre20260901.json', json.dumps({'schema': 1, 'date': '2026-10-12',
                            'cutoff': '2026-09-01', 'ids': plan['ids'], 'summaries': summary_path.name}, ensure_ascii=False, indent=2) + '\n')
        registry.write_text(destination / 'catalog.jsonl', ''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in kept))
        columns = ['id', 'record_type', 'name', 'title', 'kind', 'stage', 'status', 'methods', 'tags', 'seeds', 'report', 'path', 'evidence_state', 'fact_count']
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        for record in kept:
            writer.writerow({key: json.dumps(record[key], ensure_ascii=False) if isinstance(record.get(key), (list, dict)) else record.get(key, '') for key in columns})
        registry.write_text(destination / 'catalog.csv', buffer.getvalue())
        assert list(registry.catalog_rows(root)) == kept
        deleted = []
        for name in detail_files:
            path = boundary(root, destination / name)
            size = path.stat().st_size
            path.unlink()
            assert not path.exists()
            deleted.append({'relative': path.relative_to(root).as_posix(), 'bytes': size, 'status': 'VERIFIED'})
        coverage = registry.read_json(destination / 'coverage.json')
        coverage['retirement_20261012'] = {'retired_records': len(retired), 'retained_records': len(kept),
                                         'summary': summary_path.name, 'note': 'historical snapshot with explicit pre-September retirement increment'}
        coverage['active_record_counts'] = dict(Counter(record['record_type'] for record in kept))
        coverage['pre_retirement_record_counts'] = coverage['record_counts']
        coverage['record_counts'] = coverage['active_record_counts']
        registry.write_text(destination / 'coverage.json', registry.json_text(coverage))
        if render:
            registry.render_index(destination, kept, coverage)
        readme = destination / 'README.md'
        text = readme.read_text(encoding='utf-8')
        if not render:
            for kind, count in coverage['pre_retirement_record_counts'].items():
                text = text.replace(f'|{kind}|{count}|', f'|{kind}|{coverage["record_counts"].get(kind, 0)}|')
        registry.write_text(readme, text + '\n2026-10-12 已精简 2026-09-01 前的无当前引用旧记录，见[历史摘要](retired_pre20260901_summaries.jsonl.gz)。原始报告与精简结果保留；已删除旧元数据明细不会在重建时自动回填。\n')
        outcome = dict(plan, status='VERIFIED', deleted_details=deleted, original_catalog_rows=len(records),
                       summary_bytes=summary_path.stat().st_size, current_records_preserved=True)
        evidence.write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({key: value for key, value in outcome.items() if key not in {'ids', 'detail_files', 'deleted_details'}}, ensure_ascii=False))


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], render='--no-render' not in sys.argv)
