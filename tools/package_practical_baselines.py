"""Package committed source/configs only; excludes datasets and checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN = '20260927-phase1-baselines-practical-manysig-m5-r01'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase2-data', action='store_true')
    p.add_argument('--phase2-baselines', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    files = subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD'], cwd=ROOT, text=True).splitlines()
    selected = [s for s in files if
        s.startswith(('baselines/', 'code/cvsrffi/', 'code/leo_practical/', 'configs/baselines_practical_20260927/',
                      f'automation_reports/CV-SincNet/{RUN}/')) or
        (s.startswith('code/') and s.count('/') == 1 and s.endswith('.py')) or
        s in {'tools/run_practical_baseline.py','tools/launch_practical_baselines.py'}]
    if a.phase2_data:
        selected = [s for s in files if s.startswith(('code/cvsrffi/', 'code/leo_practical/')) or
                    (s.startswith('code/') and s.count('/') == 1 and s.endswith('.py')) or
                    s in {'tools/build_practical_phase2_data.py', 'tools/launch_practical_phase2_data.py',
                          'configs/phase2_practical_data_20260927.json'} or
                    s.startswith('automation_reports/CV-SincNet/20260927-phase2-practical-data-manytx-s2026092705-r01/')]
    if a.phase2_baselines:
        selected = [s for s in files if s.startswith(('baselines/', 'code/cvsrffi/', 'code/leo_practical/',
                    'configs/phase2_practical_baselines_20260927/',
                    'automation_reports/CV-SincNet/20260927-phase2-baselines-practical-manytx-m5-r01/')) or
                    (s.startswith('code/') and s.count('/') == 1 and s.endswith('.py')) or
                    s in {'tools/run_practical_phase2_baseline.py', 'tools/launch_practical_phase2_baselines.py',
                          'tools/score_practical_phase2_baselines.py'}]
    subprocess.run(['git','diff','--exit-code','HEAD','--',*selected], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git','archive','--format=tar.gz',f'--output={a.output.resolve()}',commit,*selected], cwd=ROOT, check=True)
    info = dict(commit=commit, members=len(selected), bytes=a.output.stat().st_size,
                sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())
    a.output.with_suffix('.manifest.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info))


if __name__ == '__main__':
    main()
