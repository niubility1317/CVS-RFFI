"""One authorized scratch source screen; never resumes or launches dependent rows."""
import argparse
import collections
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    spec = json.loads(args.spec.read_text(encoding='utf-8'))
    rows = spec['rows']
    assert [r['row_id'] for r in rows] == [m + '_s392005' for m in
        ('SIM', 'EG', 'IR', 'IR_G0', 'OR_EG', 'IR_ENCODER_OFF')]
    run = Path(spec['execution']['remote_run_root'])
    logs = Path(spec['execution']['remote_log_root'])
    assert Path(spec['code']['cwd']).resolve() == ROOT
    assert Path(sys.executable).resolve() == Path(spec['execution']['python']).resolve()
    for key in ('dataset_path', 'source_contract'):
        assert Path(spec['execution'][key]).is_file(), key
    if run.exists() or logs.exists():
        raise FileExistsError('Run/log root exists; reconcile previous launch before any retry')
    uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
    index_uuid = dict(line.strip().split(', ') for line in uuids.strip().splitlines())
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    counts = collections.Counter(line.split(',')[0].strip() for line in apps.splitlines())
    assigned = collections.Counter(str(r['gpu']) for r in rows)
    # Use empty devices for this paired screen. Other jobs are never modified.
    for gpu, count in assigned.items():
        if counts[index_uuid[gpu]] != 0 or count != 1:
            raise RuntimeError('Assigned device no longer idle: ' + gpu)
    run.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    write(run / 'effective_launch.json', dict(commit=args.commit, spec=spec,
        launcher_pid=os.getpid(), created_at=datetime.now(timezone.utc).isoformat()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rows[0]['gpu']), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    smoke_command = [sys.executable, str(ROOT / 'tools/check_release_scratch.py'),
        '--config', str(ROOT / 'configs/ir_rows/IR_s392005.json'),
        '--dataset', spec['execution']['dataset_path'],
        '--source-contract', spec['execution']['source_contract'],
        '--output', str(run / '_release_smoke'), '--device', 'cuda:0']
    with (logs / 'release_smoke.log').open('x', encoding='utf-8') as log:
        subprocess.run(smoke_command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    # A successful scratch roundtrip is diagnostic only, never inherited by rows.
    launched = []
    for row in rows:
        env['CUDA_VISIBLE_DEVICES'] = str(row['gpu'])
        with Path(row['log_path']).open('x', encoding='utf-8') as log:
            child = subprocess.Popen(row['command'], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        launched.append(dict(row_id=row['row_id'], pid=child.pid, gpu=row['gpu'],
            command=row['command'], cwd=str(ROOT), log=row['log_path'],
            output=row['output_root'], launched_at=datetime.now(timezone.utc).isoformat()))
        write(run / 'launch_state.json', dict(status='SUBMITTED', commit=args.commit, rows=launched))
    print(json.dumps(dict(status='SUBMITTED', run_id=spec['run_id'], rows=launched)), flush=True)


if __name__ == '__main__':
    main()
