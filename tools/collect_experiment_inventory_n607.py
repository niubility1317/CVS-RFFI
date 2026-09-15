"""Read-only, bounded-depth directory inventory; no training, file reads or remote writes.

Run the repository N607 preflight first. Result is an index snapshot, not job health.
"""
import argparse
import json
from pathlib import Path
import subprocess

REMOTE_PROGRAM = r'''
import datetime, getpass, json, os, socket
from pathlib import Path
root = Path('/home/szu2070436088/2510044040/CV-SincNet')
surfaces = ['runs', 'logs', 'automation_reports/CV-SincNet', 'paper_reproduction', 'baselines', 'Fedbase', 'releases']
records, errors, coverage = [], [], []
def entries(p):
    try:
        return sorted(os.scandir(p), key=lambda e: e.name)
    except OSError as e:
        errors.append({'path': str(p), 'error': str(e)})
        return []
for surface in surfaces:
    base = root / surface
    if not base.is_dir():
        coverage.append({'path': str(base), 'exists': False})
        continue
    groups = entries(base)
    for group in groups:
        if group.is_symlink() or group.name.startswith('.'):
            continue
        p = Path(group.path)
        if not group.is_dir(follow_symlinks=False):
            records.append({'path': str(p), 'name': group.name, 'surface': surface, 'entry_type': 'file', 'children': []})
            continue
        children = []
        for child in entries(p):
            if child.is_symlink():
                continue
            children.append({'name': child.name, 'type': 'directory' if child.is_dir(follow_symlinks=False) else 'file'})
        records.append({'path': str(p), 'name': group.name, 'surface': surface, 'entry_type': 'directory', 'children': children})
    coverage.append({'path': str(base), 'exists': True, 'entries': len(groups)})
print(json.dumps({'schema': 'n607_experiment_directory_inventory_v1', 'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'host': socket.gethostname(), 'user': getpass.getuser(), 'project_root': str(root), 'read_only': True,
                  'depth': 'surface/group/immediate_children', 'records': records, 'coverage': coverage, 'errors': errors}, ensure_ascii=False))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh-config', default=str(Path(__file__).with_name('n607_ssh_config')))
    parser.add_argument('--target', default='N607')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output exists; choose a new snapshot path')
    compile(REMOTE_PROGRAM, '<read-only-remote-inventory>', 'exec')
    result = subprocess.run(['ssh', '-F', args.ssh_config, '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                             args.target, 'python3 -'], input=REMOTE_PROGRAM.encode('utf-8'),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
    if result.returncode:
        raise SystemExit(result.stderr.decode('utf-8', errors='replace'))
    data = json.loads(result.stdout.decode('utf-8'))
    if data['user'] != 'szu2070436088' or data['project_root'] != '/home/szu2070436088/2510044040/CV-SincNet' or not data['read_only']:
        raise SystemExit('Unexpected remote identity/root/snapshot')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')
    saved = json.loads(args.output.read_text(encoding='utf-8'))
    assert len(saved['records']) == len(data['records'])
    print(json.dumps({'snapshot': str(args.output), 'host': saved['host'], 'records': len(saved['records']),
                      'errors': len(saved['errors']), 'observed_at': saved['observed_at']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
