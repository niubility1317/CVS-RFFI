"""Read-only process, configuration and artifact inspection of one owned run."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def inspect(project, run_id):
    project = Path(project).resolve()
    if Path(run_id).name != run_id or not run_id:
        raise ValueError('run_id must be a single path component')
    root = project / 'runs' / run_id
    result = {'time': time.time(), 'run_id': run_id, 'root': str(root),
              'exists': root.exists(), 'processes': [], 'metrics': [], 'logs': []}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            argv = (path / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            if argv and argv[-1] == '':
                argv.pop()
            if not any(run_id in arg for arg in argv):
                continue
            environment = (path / 'environ').read_bytes().decode(errors='replace').split('\0')
            cuda = next((v.split('=', 1)[1] for v in environment if v.startswith('CUDA_VISIBLE_DEVICES=')), None)
            status = (path / 'status').read_text()
            ppid = int(next(v.split()[1] for v in status.splitlines() if v.startswith('PPid:')))
            result['processes'].append({'pid': int(path.name), 'ppid': ppid,
                'cwd': str((path / 'cwd').resolve()), 'argv': argv, 'cuda_visible_devices': cuda})
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    state = root / 'pipeline.json'
    if state.is_file():
        result['pipeline'] = json.loads(state.read_text())
    for path in root.rglob('metrics_epoch.jsonl') if root.exists() else ():
        lines = path.read_text().splitlines()
        if lines:
            try:
                result['metrics'].append({'path': str(path), 'epochs_logged': len(lines), 'bytes': path.stat().st_size,
                                          'latest': json.loads(lines[-1])})
            except json.JSONDecodeError:
                result['metrics'].append({'path': str(path), 'status': 'PARTIAL_LAST_LINE'})
    log_root = project / 'logs' / run_id
    for path in log_root.rglob('*.log') if log_root.exists() else ():
        lines = path.read_text(errors='replace').splitlines()
        relevant = [line for line in lines if any(key in line for key in
            ('scratch', 'baseline_ckpt', 'CROSS-RESPONSE', 'Epoch ', 'Traceback', 'Error:'))]
        result['logs'].append({'path': str(path), 'bytes': path.stat().st_size,
                               'initialization_and_progress': relevant[-12:], 'last_lines': lines[-3:]})
    result['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                                            '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    result['compute_processes'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                                                          '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.project_root, args.run_id), ensure_ascii=False))
