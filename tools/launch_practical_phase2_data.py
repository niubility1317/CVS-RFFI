"""Launch the preregistered CPU data builder once, preserving logs and PID."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = ROOT / 'configs/phase2_practical_data_20260927.json'
    cfg = json.loads(config.read_text(encoding='utf-8'))
    if Path(cfg['output_root']).exists():
        raise FileExistsError(cfg['output_root'])
    with (ROOT / 'builder_launch_guard.json').open('x', encoding='utf-8') as f:
        json.dump({'config': str(config), 'owner': 'codex/root/practical-phase2-builder-20260927'}, f)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (ROOT / 'builder.log').open('xb') as log:
        proc = subprocess.Popen([sys.executable, '-u', 'tools/build_practical_phase2_data.py', '--config', str(config)],
                                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    result = {'pid': proc.pid, 'cwd': str(ROOT), 'log': str(ROOT / 'builder.log'), 'config': str(config)}
    (ROOT / 'builder_launch.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
