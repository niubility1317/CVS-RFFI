"""Apply reviewed local v2 repairs to a fresh v1 source copy."""
import shutil
import subprocess
from pathlib import Path


def apply_v2(destination, method):
    delivery = Path(__file__).resolve().parent
    patch = delivery / 'patches' / (method + '.patch')
    # No shell interpolation. Check the exact source state before mutating it.
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=destination, check=True)
    subprocess.run(['git', 'apply', str(patch)], cwd=destination, check=True)
    support_path = destination / ('WiSig/runtime_support.py' if method == 'asknet' else 'runtime_support.py')
    if support_path.exists():
        raise FileExistsError(f'Unexpected runtime support file: {support_path}')
    shutil.copyfile(delivery / 'runtime_support.py', support_path)
