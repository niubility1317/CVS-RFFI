"""Native Windows controller for the authorized local/N607 cleanup."""
import json
from pathlib import Path
import subprocess
import sys

BASE = Path('E:/type10-7/local_artifacts/project_file_management/20261012_cleanup')
SOURCE = Path('E:/type10-7/tools/cleanup_historical_logs_20261012.py')
SSH = ['ssh', '-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', 'N607']

REMOTE_VERIFY = r'''
import tempfile
check_root = Path(tempfile.mkdtemp(prefix='cvs_log_cleanup_check_'))
(check_root / 'logs').mkdir()
check_file = check_root / 'logs' / 'train.log'
check_data = b'loss gradient lr epoch time\n' * 100000
check_file.write_bytes(check_data)
os.utime(check_file, (CUTOFF - 86400, CUTOFF - 86400))
check_plan = plan(check_root, 'remote')
check_result = apply(check_plan, check_root / 'check_manifest.jsonl')
assert check_result['status'] == 'VERIFIED'
assert gzip.open(str(check_file) + '.gz', 'rb').read() == check_data
assert not check_file.exists()
print('REMOTE_COMPRESSOR_ROUNDTRIP_VERIFIED', flush=True)
'''

REMOTE_APPLY = r'''
from concurrent.futures import ThreadPoolExecutor
saved = json.loads(Path('/tmp/cvs_cleanup_20261012_plan.json').read_text(encoding='utf-8'))
protected_scopes = assert_selected_inactive(saved['root'], saved['files'])
print(json.dumps({'protected_active_scopes': protected_scopes}), flush=True)
groups = []
for index in range(4):
    part = dict(saved)
    part['files'] = saved['files'][index::4]
    groups.append(part)
with ThreadPoolExecutor(max_workers=4) as pool:
    jobs = [pool.submit(apply, part, '/tmp/cvs_cleanup_20261012_manifest_' + str(index) + '.jsonl') for index, part in enumerate(groups)]
    results = [job.result() for job in jobs]
summary = {'status': 'VERIFIED' if all(result['status'] == 'VERIFIED' for result in results) else 'PARTIAL',
           'completed': sum(result['completed'] for result in results),
           'failed': sum(result['failed'] for result in results),
           'reclaimed_logical_bytes': sum(result['reclaimed_logical_bytes'] for result in results),
           'parts': results}
write_json('/tmp/cvs_cleanup_20261012_summary.json', summary)
print(json.dumps(summary), flush=True)
'''


def main():
    mode = sys.argv[1]
    if mode == 'local':
        command = [sys.executable, '-X', 'utf8', str(SOURCE), 'apply',
                   '--plan', str(BASE / 'local_plan.json'),
                   '--manifest', str(BASE / 'local_manifest.jsonl')]
        payload = None
    elif mode in {'remote', 'remote_verify'}:
        command = SSH + ['python3 -']
        runner = REMOTE_APPLY if mode == 'remote' else REMOTE_VERIFY
        payload = ("__name__ = 'cleanup_module'\n" + SOURCE.read_text(encoding='utf-8') + runner).encode('utf-8')
    else:
        raise ValueError('Unexpected mode')
    transcript = BASE / (mode + '_execution.txt')
    if transcript.exists():
        raise FileExistsError('Existing execution transcript; reconcile before retry')
    with transcript.open('x', encoding='utf-8') as log:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE if payload is not None else None,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if payload is not None:
            proc.stdin.write(payload)
            proc.stdin.close()
        count = 0
        for line in proc.stdout:
            value = line.decode('utf-8', errors='replace')
            log.write(value)
            log.flush()
            count += 1
            if count % 50 == 0 or '"completed"' in value or 'VERIFIED' in value or 'Traceback' in value:
                print(value.rstrip(), flush=True)
        code = proc.wait()
    print(json.dumps({'mode': mode, 'exit': code, 'transcript': str(transcript)}), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
