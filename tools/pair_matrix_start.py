"""Detach one new Linux experiment dispatcher, refusing existing outputs/logs."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    manifest_path = Path(sys.argv[1]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    root = Path(manifest['output_root'])
    if root.exists():
        raise FileExistsError(root)
    log = root.parent / (root.name + '.dispatcher.log')
    with log.open('xb') as stream:
        process = subprocess.Popen([manifest['python'], str(Path(__file__).with_name('pair_matrix_dispatch.py')),
                                    'run', str(manifest_path)],
            cwd=manifest['code_root'], stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', 'PYTHONUNBUFFERED':'1'})
    print(json.dumps({'dispatcher_pid':process.pid,'log':str(log),'output_root':str(root)}))


if __name__ == '__main__':
    main()
