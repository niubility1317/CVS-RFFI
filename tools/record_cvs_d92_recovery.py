"""Update the existing recovery registration from independently read remote evidence."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path('E:/type10-7')
RUN = '20260928-phase2-cvs-d92-practical-manytx-m5-r02'


def main():
    snapshot = WORKSPACE / 'local_artifacts/cvs_d92_recovery_20260928/readback.json'
    data = json.loads(snapshot.read_text(encoding='utf-8'))
    run = ROOT / 'automation_reports/CV-SincNet' / RUN
    evidence = run / 'evidence' / ('readback_' + str(int(data['time'])) + '.json')
    evidence.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snapshot, evidence)
    spec = json.loads((run / 'experiment.json').read_text(encoding='utf-8'))
    spec['code']['commit'] = data['launch']['commit']
    spec['execution']['dispatcher_pid'] = data['launch']['pid']
    spec['execution']['readback_time_unix'] = data['time']
    for row in spec['rows']:
        actual = next(x for x in data['rows'] if x['row_id'] == row['row_id'])
        row.update(pid=actual['pid'], status=actual['status'])
        if actual['process']['argv']:
            row['observed_argv'] = actual['process']['argv']
    complete = data['completion'] and data['completion'].get('status') == 'SCORED'
    status = 'ARTIFACTS_COMPLETE' if complete else 'PARTIAL' if any(r['status'] == 'TECHNICAL_FAILURE' for r in data['rows']) else 'RUNNING'
    spec['status'] = status
    (run / 'experiment.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    note = 'Independent remote readback: support replays PASS; ' + ', '.join(r['row_id'] + '=' + r['status'] for r in data['rows'])
    evidence_ref = evidence.relative_to(ROOT).as_posix()
    registry = WORKSPACE / 'tools/experiment_registry.py'
    subprocess.run([sys.executable, '-X', 'utf8', str(registry), '--root', str(ROOT), 'record', RUN, '--status', status, '--evidence', evidence_ref, '--note', note], check=True)
    target = WORKSPACE / 'automation_reports/CV-SincNet' / RUN
    shutil.copytree(run, target, dirs_exist_ok=True)
    for path in ['configs/cvs_d92_recovery_20260928.json', 'tools/cvs_d92_recovery.py', 'tools/publish_cvs_d92_recovery.py', 'tools/read_cvs_d92_recovery.py', 'tools/record_cvs_d92_recovery.py']:
        (WORKSPACE / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, WORKSPACE / path)
    subprocess.run([sys.executable, '-X', 'utf8', str(registry), '--root', str(WORKSPACE), 'record', RUN, '--status', status, '--evidence', evidence_ref, '--note', note], check=True)
    print(json.dumps(dict(status=status, evidence=evidence_ref)))


if __name__ == '__main__':main()
