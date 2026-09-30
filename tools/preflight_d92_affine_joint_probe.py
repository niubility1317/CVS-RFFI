"""Read-only binding/availability check of existing legal support caches."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from preflight_d92_registration_diagnostic import REMOTE
from publish_d92_branch_support_probe import FLAGS
from run_d92_affine_joint_probe import validate_spec


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, required=True)
    args = p.parse_args()
    spec = json.loads(args.spec.read_text(encoding='utf-8'))
    validate_spec(spec)
    script = REMOTE.replace('SPEC', repr(spec))
    compile(script, 'affine-joint-preflight', 'exec')
    result = subprocess.run(['ssh', *FLAGS, '-T', 'N607', 'python3 -'],
        input=script.encode('utf-8'), capture_output=True, check=True)
    data = json.loads(result.stdout)
    folder = Path('E:/type10-7/automation_reports/CV-SincNet') / spec['run_id'] / 'evidence'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ('affine_joint_preflight_' + str(time.time_ns()) + '.json')
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(data))
    print(path)
    if data['status'] != 'VERIFIED':
        raise RuntimeError('Preflight unavailable; no launch')
